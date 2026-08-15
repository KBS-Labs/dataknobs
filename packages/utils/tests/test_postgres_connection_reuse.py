"""When a pooled connection may be handed back — the reuse decision itself.

Reproduce-first for three defects in ``DotenvPostgresConnector._is_usable``
and its caller, all of which shipped behind a ``requires_postgres`` gate that
no CI job in this repository satisfies. The lifecycle suite that would have
caught them runs only where a server is reachable; these run everywhere.

**An aborted transaction was assumed alive rather than asked.** ``_is_usable``
short-circuited ``TRANSACTION_STATUS_INERROR`` to "reusable" on the grounds
that a caller's failed transaction is theirs to unwind — correct as a policy,
but implemented by skipping the check entirely. ``INERROR`` is a local flag and
says nothing about the backend, and ``idle_in_transaction_session_timeout``
reaps an aborted transaction exactly as it reaps an open one. So the one state
a server is most likely to have killed was the one state never verified.

**A refused statement was read as a dead connection.** The probe treated any
``psycopg2.Error`` as fatal. A statement cancelled by ``statement_timeout`` or
``pg_cancel_backend`` raises ``QueryCanceled`` from a connection that is alive,
idle and immediately reusable — the server answered, which is the very thing
the probe exists to establish.

**And an unusable-but-open connection was dropped rather than closed.** Its
only handle lived in the connector's ``WeakSet``; discarding it there left it
to be reclaimed by refcounting, which is the accident this class replaced with
a lifecycle in the first place.

The three share one cause: the probe asked "did my statement succeed?" when the
question is "did the server answer?". ``conn.closed`` answers the second, which
is why it is now the verdict. The ``stand_in`` fixture in ``conftest`` records
what real psycopg2 reports in each case; the integration suite proves it does.
"""

from __future__ import annotations

from typing import Any, Callable

import psycopg2
import psycopg2.errors
from psycopg2 import extensions

from dataknobs_utils.sql_utils import CALLER_LANE, INTERNAL_LANE, DotenvPostgresConnector

IDLE = extensions.TRANSACTION_STATUS_IDLE
INTRANS = extensions.TRANSACTION_STATUS_INTRANS
INERROR = extensions.TRANSACTION_STATUS_INERROR
UNKNOWN = extensions.TRANSACTION_STATUS_UNKNOWN

CANCELLED = psycopg2.errors.QueryCanceled("canceling statement due to statement timeout")
ABORTED = psycopg2.errors.InFailedSqlTransaction("current transaction is aborted")
DROPPED = psycopg2.OperationalError("server closed the connection unexpectedly")

StandIn = Callable[..., Any]


def make_connector(**kwargs: Any) -> DotenvPostgresConnector:
    """A connector that has not connected to anything."""
    return DotenvPostgresConnector(host="h", db="d", user="u", pwd="p", port=5432, **kwargs)


class TestTheServerIsAskedNotAssumed:
    """The probe establishes that the server answered — nothing weaker."""

    def test_an_aborted_transaction_on_a_dead_backend_is_not_reusable(
        self, stand_in: StandIn
    ) -> None:
        """The state most likely to have been reaped was the one never checked.

        ``idle_in_transaction_session_timeout`` applies to an aborted
        transaction as much as to an open one, and several managed offerings
        set it by default. Handing this connection back produces exactly the
        ``server closed the connection unexpectedly`` that validation exists
        to prevent — one call later, in the caller's code rather than here.
        """
        conn = stand_in(status=INERROR, probe_error=DROPPED, error_is_transport=True)
        assert make_connector()._is_usable(conn) is False

    def test_an_aborted_transaction_on_a_live_backend_is_left_alone(
        self, stand_in: StandIn
    ) -> None:
        """Asking must not become taking.

        The reason ``INERROR`` was short-circuited is real: replacing a live
        aborted transaction discards work the caller has not been told about.
        A probe inside an aborted transaction *always* raises — the server
        refuses every statement until the transaction ends — so a check that
        replaced the connection whenever the probe failed would destroy every
        one of these. The connection stays open, and that is what decides it.
        """
        conn = stand_in(status=INERROR, probe_error=ABORTED, error_is_transport=False)
        assert make_connector()._is_usable(conn) is True
        assert conn.closed == 0

    def test_a_cancelled_statement_does_not_condemn_a_live_connection(
        self, stand_in: StandIn
    ) -> None:
        """``QueryCanceled`` is the server replying, which is proof of life.

        A role-level ``statement_timeout`` or a stray ``pg_cancel_backend``
        cancels the probe on a connection that is idle and immediately
        reusable. Reading that as death throws away a healthy backend and pays
        a fresh handshake for it.
        """
        conn = stand_in(status=IDLE, probe_error=CANCELLED, error_is_transport=False)
        assert make_connector()._is_usable(conn) is True

    def test_a_dropped_backend_is_not_reusable(self, stand_in: StandIn) -> None:
        conn = stand_in(status=IDLE, probe_error=DROPPED, error_is_transport=True)
        assert make_connector()._is_usable(conn) is False


class TestTheFreeChecksAreNotOptional:
    """``validate_on_reuse`` buys off the round trip, not the local reads."""

    def test_an_unknown_status_is_rejected_without_validation(self, stand_in: StandIn) -> None:
        """The status read costs nothing, so opting out must not skip it.

        ``validate_on_reuse=False`` is documented as trading 0.29 ms for
        staleness detection. ``get_transaction_status()`` is local and answers
        for free, and ``UNKNOWN`` means libpq has already given up on the
        connection — returning it is a guaranteed failure at the caller.
        """
        conn = stand_in(status=UNKNOWN)
        assert make_connector(validate_on_reuse=False)._is_usable(conn) is False

    def test_opting_out_still_skips_the_round_trip(self, stand_in: StandIn) -> None:
        conn = stand_in(status=IDLE)
        assert make_connector(validate_on_reuse=False)._is_usable(conn) is True
        assert conn.executed == []

    def test_a_closed_connection_is_rejected_without_a_round_trip(self, stand_in: StandIn) -> None:
        conn = stand_in(closed=1)
        assert make_connector()._is_usable(conn) is False
        assert conn.executed == []

    def test_no_connection_is_not_a_usable_one(self) -> None:
        assert make_connector()._is_usable(None) is False


class TestTheProbeLeavesNoTrace:
    """Validating is a question, not a change of state."""

    def test_an_idle_connection_is_probed_under_autocommit(self, stand_in: StandIn) -> None:
        """Otherwise the probe opens a transaction nothing closes.

        A connection handed back idle-in-transaction holds a snapshot and
        blocks VACUUM for as long as the caller keeps it.
        """
        conn = stand_in(status=IDLE, autocommit=False)
        assert make_connector()._is_usable(conn) is True
        assert conn.executed == ["SELECT 1"]
        assert conn.autocommit_at_execute is True
        assert conn.autocommit is False, "the caller's setting must be restored"

    def test_an_open_transaction_is_joined_rather_than_disturbed(self, stand_in: StandIn) -> None:
        """Switching autocommit mid-transaction is an error, and not our call."""
        conn = stand_in(status=INTRANS, autocommit=False)
        assert make_connector()._is_usable(conn) is True
        assert conn.autocommit_at_execute is False

    def test_a_caller_running_in_autocommit_keeps_it(self, stand_in: StandIn) -> None:
        conn = stand_in(status=IDLE, autocommit=True)
        assert make_connector()._is_usable(conn) is True
        assert conn.autocommit is True


class TestNothingOpenIsLeftToTheGarbageCollector:
    """The registry is what ``close()`` reaches; being unusable is not exit."""

    def test_an_unusable_but_open_connection_is_still_closed_by_close(
        self, stand_in: StandIn, hand_out: Callable[[list[Any]], None]
    ) -> None:
        """Replacing a connection must not also forget it.

        ``get_conn`` dropped the outgoing connection from the ``WeakSet``
        whenever it was replaced, whether or not it was closed. A connection
        the caller still holds is not collectable, so nothing else would ever
        close it: ``close()`` would return having missed a live backend, which
        is the one thing it exists to prevent.
        """
        stale = stand_in(status=UNKNOWN)
        fresh = stand_in(status=IDLE)
        hand_out([stale, fresh])

        connector = make_connector()
        assert connector.get_conn() is stale
        assert connector.get_conn() is fresh, "the unusable one should be replaced"

        connector.close()
        assert stale.closed, "close() lost track of a connection it opened"
        assert fresh.closed

    def test_a_replaced_closed_connection_is_not_retained(
        self, stand_in: StandIn, hand_out: Callable[[list[Any]], None]
    ) -> None:
        """The registry holds what is open; a closed connection has nothing owed."""
        gone = stand_in(closed=2)
        fresh = stand_in(status=IDLE)
        hand_out([fresh])

        connector = make_connector()
        connector._lane_conns()[CALLER_LANE] = gone
        assert connector.get_conn() is fresh
        assert gone not in connector._open_conns

    def test_one_connection_that_will_not_close_does_not_strand_the_rest(
        self, stand_in: StandIn, hand_out: Callable[[list[Any]], None]
    ) -> None:
        """``close()`` is a sweep, and a sweep that stops halfway is a leak.

        psycopg2 rarely raises from ``close()``, which is precisely the danger:
        the one time it does, every connection after it in the iteration would
        be left open with the registry already emptied — nothing able to reach
        them and no second chance to try.
        """
        stubborn = stand_in(status=IDLE, close_error=psycopg2.OperationalError("nope"))
        other = stand_in(status=IDLE)
        hand_out([stubborn, other])

        connector = make_connector()
        connector.get_conn(lane=CALLER_LANE)
        connector.get_conn(lane=INTERNAL_LANE)

        connector.close()

        assert other.closed, "the sweep stopped at the connection that raised"
