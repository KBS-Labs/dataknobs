"""Shared resilient supervised-loop helper for event-bus backends.

An event-bus listener is an *infinite supervised loop*: it must survive
transient backend failures and back off between them, but it must
**never give up** (unlike :class:`dataknobs_common.retry.RetryExecutor`,
which retries N times then raises).

:func:`run_supervised_loop` is that loop, defined once and shared by
every backend listener. It owns the ``while should_run()`` lifecycle,
the cancel semantics, and the exponential-with-jitter back-off (via the
shared :func:`dataknobs_common.retry.compute_backoff_delay`). The jitter
is what keeps listeners across replicas from waking on the same
1-second boundary and re-hammering a degraded backend in lockstep; the
escalation (capped, reset on a clean iteration) keeps a sustained
outage from being polled tightly. It is an internal
backend-implementation helper, **not** part of the public ``EventBus``
protocol surface, so it is intentionally not re-exported from
``dataknobs_common.events``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from dataknobs_common.retry import BackoffStrategy, compute_backoff_delay

logger = logging.getLogger(__name__)


async def run_supervised_loop(
    one_iteration: Callable[[], Awaitable[None]],
    *,
    should_run: Callable[[], bool],
    name: str,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Run ``one_iteration`` in a never-give-up supervised loop.

    Args:
        one_iteration: Awaitable performing exactly one unit of work
            (e.g. one ``receive_message`` + dispatch). It is called
            repeatedly while ``should_run()`` is true.
        should_run: Predicate checked before every iteration. When it
            returns false the loop returns cleanly (no exception).
        name: Human-readable loop name used only in the failure log line
            (never the payload or credentials — security rule 5).
        base_delay: Back-off delay (seconds) after the first failure in a
            run of consecutive failures.
        max_delay: Upper bound on the back-off delay.
        sleep: Injectable awaitable sleep (defaults to
            :func:`asyncio.sleep`); overridden in tests for determinism.

    Behavior:
        - Loops while ``should_run()`` is true; returns cleanly when it
          flips false.
        - On :class:`asyncio.CancelledError` from ``one_iteration``:
          **breaks/returns — does not re-raise.** This preserves the
          observable semantics of the loops it replaces (both
          ``SqsEventBus._poll_loop`` and
          ``RedisEventBus._message_listener`` ``break`` on cancel; their
          callers ``await`` the task under
          ``contextlib.suppress(CancelledError)``, which stays correct
          whether the task ends by break or by re-raise). ``CancelledError``
          is a ``BaseException`` (py3.8+), so the generic ``except
          Exception`` arm cannot swallow it — it is still handled
          explicitly here.
        - On any other ``Exception``: logs (no payload/credentials) and
          backs off ``compute_backoff_delay(JITTER, ...)`` — exponential
          with jitter, capped at ``max_delay``.
        - Resets the failure counter to 0 after any iteration that
          returns without raising, so a recovered loop does not stay
          permanently escalated.
    """
    consecutive_failures = 0

    while should_run():
        try:
            await one_iteration()
        except asyncio.CancelledError:
            break
        except Exception:
            consecutive_failures += 1
            logger.exception("%s iteration failed", name)
            delay = compute_backoff_delay(
                BackoffStrategy.JITTER,
                attempt=consecutive_failures,
                initial_delay=base_delay,
                max_delay=max_delay,
            )
            try:
                await sleep(delay)
            except asyncio.CancelledError:
                break
        else:
            consecutive_failures = 0
