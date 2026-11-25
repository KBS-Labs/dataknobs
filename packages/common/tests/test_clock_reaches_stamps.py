"""The common package's own stamps are read from the clock.

Under :func:`~dataknobs_common.clock.use_clock` with a
:class:`~dataknobs_common.testing.FixedClock`, a provenance record's
``created_at``, an event's ``timestamp`` and a subscription's ``created_at``
**equal** the fixed instant, including an event read from a dict that carries
no timestamp.
"""

from __future__ import annotations

from datetime import UTC, datetime

from dataknobs_common.clock import use_clock
from dataknobs_common.events import Event, EventType, Subscription
from dataknobs_common.provenance import LLMInvocation, ProvenanceRecord, create_provenance
from dataknobs_common.testing import FixedClock

AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def test_provenance_records_are_stamped_by_the_clock() -> None:
    with use_clock(FixedClock(AT)):
        assert create_provenance("user:a", "manual").created_at == AT.isoformat()
        assert ProvenanceRecord().created_at == AT.isoformat()
        assert LLMInvocation(purpose="write").timestamp == AT.isoformat()


def test_an_event_is_stamped_by_the_clock() -> None:
    with use_clock(FixedClock(AT)):
        event = Event(type=EventType.CREATED, topic="t")
        read = Event.from_dict({"type": "created", "topic": "t"})
        subscription = Subscription(subscription_id="s", topic="t", handler=print)
    assert event.timestamp == AT
    assert read.timestamp == AT
    assert subscription.created_at == AT


def _refusing_clock() -> datetime:
    """A clock no read may reach: it returns a naive value, which is refused."""
    return datetime(2026, 1, 1)


def test_event_from_dict_does_not_read_the_clock_for_a_stored_timestamp() -> None:
    """The default is read only when it is needed, not on every call."""
    with use_clock(FixedClock(AT)):
        stored = Event(type=EventType.CREATED, topic="t", payload={}).to_dict()
    # A payload built in process carries the datetime itself, not its text.
    stored["timestamp"] = AT
    with use_clock(_refusing_clock):
        restored = Event.from_dict(stored)
    assert restored.timestamp == AT
