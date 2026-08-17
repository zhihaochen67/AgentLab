from datetime import datetime
from uuid import UUID

from agentlab.tracer import Tracer


def test_tracer_generates_unique_uuid_run_ids() -> None:
    first = Tracer()
    second = Tracer()

    assert first.run_id != second.run_id
    assert str(UUID(first.run_id)) == first.run_id
    assert str(UUID(second.run_id)) == second.run_id


def test_tracer_sequence_is_strictly_increasing_and_stable() -> None:
    tracer = Tracer()

    tracer.emit("first")
    tracer.emit("second")
    tracer.emit("third")

    assert [event.sequence for event in tracer.events] == [1, 2, 3]
    assert [event.event_type for event in tracer.events] == ["first", "second", "third"]
    assert all(event.run_id == tracer.run_id for event in tracer.events)
    assert all(datetime.fromisoformat(event.timestamp) for event in tracer.events)
