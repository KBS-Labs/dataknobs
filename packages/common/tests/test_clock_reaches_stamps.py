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
