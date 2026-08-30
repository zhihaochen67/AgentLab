from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agentlab.subprocesses import run_process

_TIMEOUT_SECONDS = 1.5
_RETURN_LIMIT_SECONDS = 12.0


def test_normal_process_preserves_cwd_env_and_text_capture(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["AGENTLAB_SUBPROCESS_TEST"] = "visible"
    command = [
        sys.executable,
        "-B",
        "-c",
        (
            "import os,pathlib,sys; print(pathlib.Path.cwd()); "
            "print(os.environ['AGENTLAB_SUBPROCESS_TEST']); "
            "print('normal-err', file=sys.stderr)"
        ),
    ]

    result = run_process(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert tuple(result.args) == tuple(command)
    assert result.returncode == 0
    assert result.stdout.splitlines() == [str(tmp_path), "visible"]
    assert result.stderr.strip() == "normal-err"


def test_timeout_none_waits_for_normal_completion() -> None:
    result = run_process(
        [sys.executable, "-B", "-c", "print('unbounded')"],
        capture_output=True,
        text=True,
        check=False,
        timeout=None,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "unbounded"


def test_nonzero_exit_is_returned_when_check_is_false() -> None:
    result = run_process(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; print('failed', file=sys.stderr); raise SystemExit(7)",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 7
    assert result.stderr.strip() == "failed"


def test_launch_error_is_not_converted_to_a_completed_process(tmp_path: Path) -> None:
    missing = tmp_path / "definitely-missing-executable"

    with pytest.raises(OSError):
        run_process([str(missing)], timeout=10)


def test_direct_child_timeout_terminates_the_child(tmp_path: Path) -> None:
    pid_path = tmp_path / "direct.pid"
    command = [
        sys.executable,
        "-u",
        "-B",
        "-c",
        (
            "import os,pathlib,sys,time; "
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8'); "
            "print('ready', flush=True); time.sleep(60)"
        ),
        str(pid_path),
    ]
    started = time.monotonic()
    pid: int | None = None
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            run_process(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=_TIMEOUT_SECONDS,
            )
        pid = _read_pid(pid_path)
        assert time.monotonic() - started < _RETURN_LIMIT_SECONDS
        assert _wait_until_stopped(pid)
    finally:
        _force_kill(pid)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group branch")
def test_posix_timeout_terminates_child_and_grandchild(tmp_path: Path) -> None:
    _assert_tree_timeout(tmp_path)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object branch")
def test_windows_timeout_terminates_child_and_grandchild(tmp_path: Path) -> None:
    _assert_tree_timeout(tmp_path)


def test_pipe_holding_descendant_timeout_returns_and_cleans_tree(
    tmp_path: Path,
) -> None:
    command, child_path, grandchild_path = _tree_command(
        tmp_path,
        child_exits=True,
    )
    started = time.monotonic()
    child_pid: int | None = None
    grandchild_pid: int | None = None
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            run_process(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=_TIMEOUT_SECONDS,
            )
        child_pid = _read_pid(child_path)
        grandchild_pid = _read_pid(grandchild_path)
        assert time.monotonic() - started < _RETURN_LIMIT_SECONDS
        assert _wait_until_stopped(child_pid)
        assert _wait_until_stopped(grandchild_pid)
    finally:
        _force_kill(grandchild_pid)
        _force_kill(child_pid)


def _assert_tree_timeout(tmp_path: Path) -> None:
    command, child_path, grandchild_path = _tree_command(
        tmp_path,
        child_exits=False,
    )
    child_pid: int | None = None
    grandchild_pid: int | None = None
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            run_process(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=_TIMEOUT_SECONDS,
            )
        child_pid = _read_pid(child_path)
        grandchild_pid = _read_pid(grandchild_path)
        assert _wait_until_stopped(child_pid)
        assert _wait_until_stopped(grandchild_pid)
    finally:
        _force_kill(grandchild_pid)
        _force_kill(child_pid)


def _tree_command(
    root: Path,
    *,
    child_exits: bool,
) -> tuple[list[str], Path, Path]:
    child_path = root / "child.pid"
    grandchild_path = root / "grandchild.pid"
    grandchild_script = root / "grandchild.py"
    child_script = root / "child.py"
    grandchild_script.write_text(
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "import time\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8')\n"
        "print('grandchild-stdout', flush=True)\n"
        "print('grandchild-stderr', file=sys.stderr, flush=True)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    child_script.write_text(
        "import os\n"
        "import pathlib\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "child_pid = pathlib.Path(sys.argv[1])\n"
        "grandchild_pid = pathlib.Path(sys.argv[2])\n"
        "grandchild_script = pathlib.Path(sys.argv[3])\n"
        "child_pid.write_text(str(os.getpid()), encoding='utf-8')\n"
        "subprocess.Popen(\n"
        "    [sys.executable, '-u', '-B', str(grandchild_script), str(grandchild_pid)],\n"
        "    stdout=sys.stdout,\n"
        "    stderr=sys.stderr,\n"
        ")\n"
        "deadline = time.monotonic() + 10\n"
        "while not grandchild_pid.is_file():\n"
        "    if time.monotonic() >= deadline:\n"
        "        raise RuntimeError('grandchild did not start')\n"
        "    time.sleep(0.01)\n"
        "print('child-ready', flush=True)\n"
        "if sys.argv[4] == 'exit':\n"
        "    raise SystemExit(0)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-u",
        "-B",
        str(child_script),
        str(child_path),
        str(grandchild_path),
        str(grandchild_script),
        "exit" if child_exits else "sleep",
    ]
    return command, child_path, grandchild_path


def _read_pid(path: Path) -> int:
    assert path.is_file(), f"subprocess did not write {path.name}"
    return int(path.read_text(encoding="utf-8"))


def _wait_until_stopped(pid: int) -> bool:
    deadline = time.monotonic() + 5.0
    while _pid_is_running(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _pid_is_running(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(0x00100000, False, pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x00000102
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            return proc_stat.read_text(encoding="utf-8").split()[2] != "Z"
        except (OSError, IndexError):
            pass
    return True


def _force_kill(pid: int | None) -> None:
    if pid is None or not _pid_is_running(pid):
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
