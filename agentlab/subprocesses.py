"""Bounded subprocess execution with process-tree timeout cleanup."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

_TERMINATE_GRACE_SECONDS = 1.0
_FORCE_KILL_WAIT_SECONDS = 5.0
_WINDOWS_LAUNCH_ERROR = 254

_WINDOWS_LAUNCHER = r"""
import json
import os
import subprocess
import sys
import time

gate_path, status_path, *command = sys.argv[1:]
deadline = time.monotonic() + 60.0
while not os.path.exists(gate_path):
    if time.monotonic() >= deadline:
        raise SystemExit(253)
    time.sleep(0.01)
try:
    child = subprocess.Popen(command)
except OSError as error:
    payload = {
        "type": type(error).__name__,
        "errno": error.errno,
        "winerror": getattr(error, "winerror", None),
        "strerror": error.strerror or str(error),
        "filename": error.filename,
    }
    with open(status_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream)
    raise SystemExit(254)
raise SystemExit(child.wait())
"""

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _KERNEL32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    _KERNEL32.CreateJobObjectW.restype = wintypes.HANDLE
    _KERNEL32.AssignProcessToJobObject.argtypes = (
        wintypes.HANDLE,
        wintypes.HANDLE,
    )
    _KERNEL32.AssignProcessToJobObject.restype = wintypes.BOOL
    _KERNEL32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _KERNEL32.TerminateJobObject.restype = wintypes.BOOL
    _KERNEL32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    _KERNEL32.WaitForSingleObject.restype = wintypes.DWORD
    _KERNEL32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _KERNEL32.CloseHandle.restype = wintypes.BOOL


def run_process(
    args: Sequence[str | os.PathLike[str]],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    capture_output: bool = False,
    text: bool = False,
    check: bool = False,
    timeout: float | None = None,
    encoding: str | None = None,
    errors: str | None = None,
) -> subprocess.CompletedProcess:
    """Run one command and terminate its normal descendant tree on timeout.

    The supported surface intentionally matches the options used by AgentLab's
    bounded invocations instead of acting as a general ``subprocess.run`` clone.
    """
    if isinstance(args, (str, bytes, os.PathLike)):
        raise TypeError("args must be a non-empty command sequence.")
    command = tuple(os.fspath(item) for item in args)
    if not command:
        raise ValueError("args must be a non-empty command sequence.")
    if timeout is not None and timeout < 0:
        raise ValueError("timeout must be non-negative or None.")

    if os.name == "nt":
        completed = _run_windows(
            command,
            cwd=cwd,
            env=env,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            encoding=encoding,
            errors=errors,
        )
    else:
        completed = _run_posix(
            command,
            cwd=cwd,
            env=env,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            encoding=encoding,
            errors=errors,
        )
    if check:
        completed.check_returncode()
    return completed


def _popen_options(
    *,
    cwd: str | os.PathLike[str] | None,
    env: Mapping[str, str] | None,
    capture_output: bool,
    text: bool,
    encoding: str | None,
    errors: str | None,
) -> dict:
    return {
        "cwd": cwd,
        "env": env,
        "stdout": subprocess.PIPE if capture_output else None,
        "stderr": subprocess.PIPE if capture_output else None,
        "text": text,
        "encoding": encoding,
        "errors": errors,
    }


def _run_posix(
    command: tuple[str, ...],
    *,
    cwd: str | os.PathLike[str] | None,
    env: Mapping[str, str] | None,
    capture_output: bool,
    text: bool,
    timeout: float | None,
    encoding: str | None,
    errors: str | None,
) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        command,
        start_new_session=True,
        **_popen_options(
            cwd=cwd,
            env=env,
            capture_output=capture_output,
            text=text,
            encoding=encoding,
            errors=errors,
        ),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as initial:
        stdout, stderr = _terminate_posix_tree(process, initial)
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=stdout,
            stderr=stderr,
        ) from None
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _terminate_posix_tree(
    process: subprocess.Popen,
    initial: subprocess.TimeoutExpired,
) -> tuple[object, object]:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        output = process.communicate(timeout=_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired as graceful:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output = _finish_communication(process, graceful, initial)
    else:
        if _posix_process_group_exists(process.pid):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    _wait_for_posix_process_group(process.pid)
    return output


def _run_windows(
    command: tuple[str, ...],
    *,
    cwd: str | os.PathLike[str] | None,
    env: Mapping[str, str] | None,
    capture_output: bool,
    text: bool,
    timeout: float | None,
    encoding: str | None,
    errors: str | None,
) -> subprocess.CompletedProcess:
    job = _create_windows_job()
    temporary = tempfile.TemporaryDirectory(prefix="agentlab-process-")
    gate_path = Path(temporary.name) / "start"
    status_path = Path(temporary.name) / "launch-error.json"
    launcher = (
        sys.executable,
        "-B",
        "-c",
        _WINDOWS_LAUNCHER,
        str(gate_path),
        str(status_path),
        *command,
    )
    process: subprocess.Popen | None = None
    try:
        process = subprocess.Popen(
            launcher,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            **_popen_options(
                cwd=cwd,
                env=env,
                capture_output=capture_output,
                text=text,
                encoding=encoding,
                errors=errors,
            ),
        )
        try:
            _assign_windows_job(job, process)
        except OSError:
            process.kill()
            process.communicate()
            raise
        gate_path.touch()
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as initial:
            stdout, stderr = _terminate_windows_tree(process, job, initial)
            raise subprocess.TimeoutExpired(
                command,
                timeout,
                output=stdout,
                stderr=stderr,
            ) from None
        if status_path.is_file():
            _raise_windows_launch_error(status_path)
        if process.returncode == _WINDOWS_LAUNCH_ERROR:
            raise RuntimeError("Windows subprocess launcher failed without details.")
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        _close_windows_handle(job)
        temporary.cleanup()


def _terminate_windows_tree(
    process: subprocess.Popen,
    job: int,
    initial: subprocess.TimeoutExpired,
) -> tuple[object, object]:
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT)
    except (AttributeError, OSError, ValueError):
        pass
    try:
        output = process.communicate(timeout=_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired as graceful:
        if (
            not _KERNEL32.TerminateJobObject(job, 1)
            and process.poll() is None
        ):
            process.kill()
        output = _finish_communication(process, graceful, initial)
    else:
        _KERNEL32.TerminateJobObject(job, 1)
    _KERNEL32.WaitForSingleObject(job, int(_FORCE_KILL_WAIT_SECONDS * 1_000))
    return output


def _finish_communication(
    process: subprocess.Popen,
    latest: subprocess.TimeoutExpired,
    initial: subprocess.TimeoutExpired,
) -> tuple[object, object]:
    try:
        return process.communicate(timeout=_FORCE_KILL_WAIT_SECONDS)
    except subprocess.TimeoutExpired as final:
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()
        process.wait(timeout=_FORCE_KILL_WAIT_SECONDS)
        return (
            final.output or latest.output or initial.output,
            final.stderr or latest.stderr or initial.stderr,
        )


def _create_windows_job() -> int:
    handle = _KERNEL32.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


def _assign_windows_job(job: int, process: subprocess.Popen) -> None:
    process_handle = wintypes.HANDLE(int(process._handle))
    if not _KERNEL32.AssignProcessToJobObject(job, process_handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _close_windows_handle(handle: int) -> None:
    if os.name == "nt" and handle:
        _KERNEL32.CloseHandle(handle)


def _posix_process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_posix_process_group(process_group: int) -> None:
    deadline = time.monotonic() + _FORCE_KILL_WAIT_SECONDS
    while _posix_process_group_exists(process_group):
        if time.monotonic() >= deadline:
            return
        time.sleep(0.02)


def _raise_windows_launch_error(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Could not decode Windows subprocess launch failure.") from error
    error_type = {
        "FileNotFoundError": FileNotFoundError,
        "PermissionError": PermissionError,
    }.get(payload.get("type"), OSError)
    errno = payload.get("errno")
    strerror = payload.get("strerror") or "Subprocess could not start."
    filename = payload.get("filename")
    winerror = payload.get("winerror")
    if winerror is not None:
        raise error_type(errno, strerror, filename, winerror)
    if errno is not None:
        raise error_type(errno, strerror, filename)
    raise error_type(strerror)
