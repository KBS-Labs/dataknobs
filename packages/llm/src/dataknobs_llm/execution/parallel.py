"""Parallel and sequential LLM task execution.

Provides a focused utility for running multiple LLM calls (and deterministic
functions) concurrently with concurrency control, per-task error isolation,
and LLM-typed inputs/outputs.

Example:
    >>> from dataknobs_llm import ParallelLLMExecutor, LLMTask, LLMMessage
    >>> executor = ParallelLLMExecutor(provider, max_concurrency=3)
    >>> results = await executor.execute({
    ...     "stem": LLMTask(messages=[LLMMessage(role="user", content="Generate a question")]),
    ...     "answer": LLMTask(messages=[LLMMessage(role="user", content="Generate an answer")]),
    ... })
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from dataknobs_common.retry import RetryConfig, RetryExecutor

from dataknobs_llm.llm.base import AsyncLLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class LLMTask:
    """A single LLM call to execute.

    Attributes:
        messages: Messages to send to the LLM provider.
        config_overrides: Per-task provider config overrides (temperature, model, etc.).
        retry: Per-task retry policy. Overrides the executor's default_retry.
        tag: Identifier for result lookup. Populated automatically from the dict key
            when using ``execute()`` or ``execute_mixed()``.
        timeout: Per-task timeout in seconds. Overrides the executor's
            ``default_per_task_timeout``. ``None`` defers to that default.
            When combined with ``retry``, the timeout bounds **each retry
            attempt individually** — total elapsed across retries remains
            the consumer's responsibility (e.g. via ``RetryConfig.max_delay``
            or an outer ``asyncio.wait_for``).
    """

    messages: list[LLMMessage]
    config_overrides: dict[str, Any] | None = None
    retry: RetryConfig | None = None
    tag: str = ""
    timeout: float | None = None


@dataclass
class DeterministicTask:
    """A sync or async callable to execute alongside LLM tasks.

    Attributes:
        fn: The callable to execute. May be sync or async.
        args: Positional arguments forwarded to fn.
        kwargs: Keyword arguments forwarded to fn.
        tag: Identifier for result lookup.
        timeout: Per-task timeout in seconds. Overrides the executor's
            ``default_per_task_timeout``. ``None`` defers to that default.
            Sync callables run on the thread executor and cannot be
            pre-empted mid-call: the timeout still fires (the awaiter
            stops waiting) but the underlying thread continues until the
            sync function returns.
    """

    fn: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    tag: str = ""
    timeout: float | None = None


@dataclass
class TaskResult:
    """Result of a single task execution.

    Attributes:
        tag: The task identifier.
        success: Whether the task completed without error.
        value: The return value. ``LLMResponse`` for LLM tasks, arbitrary for
            deterministic tasks. ``None`` on failure.
        error: The exception if the task failed, ``None`` otherwise. Typed
            as ``BaseException | None`` because cancelled tasks (under
            ``fail_fast=True``) carry an ``asyncio.CancelledError`` which
            in Python 3.12+ inherits from ``BaseException`` rather than
            ``Exception``. Completion-failure tasks still carry an
            ``Exception``-typed error.
        duration_ms: Wall-clock execution time in milliseconds.
    """

    tag: str
    success: bool
    value: LLMResponse | Any
    error: BaseException | None = None
    duration_ms: float = 0.0


class ParallelLLMExecutor:
    """Runs multiple LLM calls and deterministic functions concurrently.

    Features:
    - Concurrency control via ``asyncio.Semaphore``
    - Per-task error isolation (one failure does not cancel others)
    - Optional per-task retry via ``RetryExecutor``
    - Mixed execution of LLM tasks and deterministic callables

    Example:
        >>> executor = ParallelLLMExecutor(provider, max_concurrency=3)
        >>> results = await executor.execute({
        ...     "q1": LLMTask(messages=[LLMMessage(role="user", content="Hello")]),
        ...     "q2": LLMTask(messages=[LLMMessage(role="user", content="World")]),
        ... })
        >>> assert results["q1"].success
    """

    def __init__(
        self,
        provider: AsyncLLMProvider,
        max_concurrency: int = 5,
        default_retry: RetryConfig | None = None,
        default_config_overrides: dict[str, Any] | None = None,
        fail_fast: bool = False,
        default_per_task_timeout: float | None = None,
    ) -> None:
        """Initialize the executor.

        Args:
            provider: The LLM provider to use for LLM tasks.
            max_concurrency: Maximum number of concurrent tasks.
            default_retry: Default retry policy applied to tasks that do not
                specify their own.
            default_config_overrides: Config overrides applied to every LLM
                task.  Per-task ``config_overrides`` take precedence over
                these defaults when both are set.
            fail_fast: When True, the first task failure cancels all pending
                tasks. Cancelled tasks return ``TaskResult(success=False,
                error=asyncio.CancelledError(...))`` so callers can
                distinguish them from completion-failures. Default ``False``
                preserves the historical "isolate-and-continue" contract.
                Each ``execute()`` / ``execute_mixed()`` /
                ``execute_sequential()`` call may override this via a
                ``fail_fast=`` keyword argument.
            default_per_task_timeout: Default timeout in seconds applied to
                tasks that do not specify their own ``timeout``. ``None``
                disables the default (no timeout). Per-task
                ``LLMTask.timeout`` / ``DeterministicTask.timeout`` overrides
                this. With ``RetryConfig`` the timeout bounds each retry
                attempt individually.
        """
        self._provider = provider
        self._max_concurrency = max_concurrency
        self._default_retry = default_retry
        self._default_config_overrides = default_config_overrides
        self._fail_fast = fail_fast
        self._default_per_task_timeout = default_per_task_timeout

    async def execute(
        self,
        tasks: dict[str, LLMTask],
        *,
        fail_fast: bool | None = None,
    ) -> dict[str, TaskResult]:
        """Run LLM tasks concurrently with error isolation.

        Args:
            tasks: Mapping of tag to LLMTask. Tags identify results.
            fail_fast: Override the executor-level ``fail_fast`` flag for
                this call. ``None`` (default) defers to the executor's
                ``__init__`` value.

        Returns:
            Mapping of tag to TaskResult. Under ``fail_fast=True``,
            cancelled tasks return ``TaskResult(success=False,
            error=asyncio.CancelledError(...))``.
        """
        if not tasks:
            return {}

        effective_fail_fast = (
            self._fail_fast if fail_fast is None else fail_fast
        )
        semaphore = asyncio.Semaphore(self._max_concurrency)
        start_all = time.monotonic()

        async def _run(tag: str, task: LLMTask) -> TaskResult:
            async with semaphore:
                return await self._execute_single_llm(tag, task)

        if effective_fail_fast:
            results = await self._gather_fail_fast(
                [(tag, _run(tag, task)) for tag, task in tasks.items()]
            )
        else:
            coros = [_run(tag, task) for tag, task in tasks.items()]
            results = await asyncio.gather(*coros)
        result_map = {r.tag: r for r in results}

        total_ms = (time.monotonic() - start_all) * 1000
        successes = sum(1 for r in results if r.success)
        failures = len(results) - successes
        logger.info(
            "Parallel execution complete: %d tasks (%d ok, %d failed) in %.1fms",
            len(results),
            successes,
            failures,
            total_ms,
        )

        return result_map

    async def execute_mixed(
        self,
        tasks: dict[str, LLMTask | DeterministicTask],
        *,
        fail_fast: bool | None = None,
    ) -> dict[str, TaskResult]:
        """Run a mix of LLM and deterministic tasks concurrently.

        Args:
            tasks: Mapping of tag to task (LLMTask or DeterministicTask).
            fail_fast: Override the executor-level ``fail_fast`` flag for
                this call. ``None`` (default) defers to the executor's
                ``__init__`` value. Note that synchronous
                ``DeterministicTask`` callables run on the default thread
                executor and cannot be pre-empted mid-execution; only tasks
                that have not yet started, or async tasks suspended at an
                ``await`` point, are reliably cancellable.

        Returns:
            Mapping of tag to TaskResult. Under ``fail_fast=True``,
            cancelled tasks return ``TaskResult(success=False,
            error=asyncio.CancelledError(...))``.
        """
        if not tasks:
            return {}

        effective_fail_fast = (
            self._fail_fast if fail_fast is None else fail_fast
        )
        semaphore = asyncio.Semaphore(self._max_concurrency)
        start_all = time.monotonic()

        async def _run(tag: str, task: LLMTask | DeterministicTask) -> TaskResult:
            async with semaphore:
                if isinstance(task, LLMTask):
                    return await self._execute_single_llm(tag, task)
                return await self._execute_single_deterministic(tag, task)

        if effective_fail_fast:
            results = await self._gather_fail_fast(
                [(tag, _run(tag, task)) for tag, task in tasks.items()]
            )
        else:
            coros = [_run(tag, task) for tag, task in tasks.items()]
            results = await asyncio.gather(*coros)
        result_map = {r.tag: r for r in results}

        total_ms = (time.monotonic() - start_all) * 1000
        successes = sum(1 for r in results if r.success)
        failures = len(results) - successes
        logger.info(
            "Mixed execution complete: %d tasks (%d ok, %d failed) in %.1fms",
            len(results),
            successes,
            failures,
            total_ms,
        )

        return result_map

    async def execute_sequential(
        self,
        tasks: list[LLMTask],
        pass_result: bool = False,
        *,
        fail_fast: bool | None = None,
    ) -> list[TaskResult]:
        """Run LLM tasks sequentially, optionally passing results forward.

        When ``pass_result`` is True, each task's messages are augmented with
        the previous task's response as an assistant message.

        Args:
            tasks: Ordered list of LLM tasks to run.
            pass_result: If True, append previous result as assistant message.
            fail_fast: Override the executor-level ``fail_fast`` flag for
                this call. ``None`` (default) defers to the executor's
                ``__init__`` value. Under ``fail_fast=True`` the loop stops
                at the first failed task; the returned list is **shorter
                than** ``tasks`` (callers can detect short-circuit via
                ``len(results) < len(tasks)``). Tasks that never ran are
                not represented in the result.

        Returns:
            List of TaskResult in execution order.
        """
        effective_fail_fast = (
            self._fail_fast if fail_fast is None else fail_fast
        )
        results: list[TaskResult] = []
        for i, task in enumerate(tasks):
            tag = task.tag or f"step_{i}"
            if pass_result and results and results[-1].success:
                prev_response = results[-1].value
                task = LLMTask(
                    messages=list(task.messages)
                    + [LLMMessage(role="assistant", content=prev_response.content)],
                    config_overrides=task.config_overrides,
                    retry=task.retry,
                    tag=tag,
                )
            result = await self._execute_single_llm(tag, task)
            results.append(result)
            if effective_fail_fast and not result.success:
                logger.info(
                    "Sequential execution aborted by fail_fast at step %d/%d (tag=%r)",
                    i + 1,
                    len(tasks),
                    tag,
                )
                break
        return results

    async def _execute_single_llm(self, tag: str, task: LLMTask) -> TaskResult:
        """Execute a single LLM task with optional retry.

        Args:
            tag: Task identifier.
            task: The LLM task to execute.

        Returns:
            TaskResult with success/failure status.
        """
        start = time.monotonic()
        logger.debug("Starting LLM task '%s'", tag)
        try:
            # Merge config overrides: executor defaults < task-level
            effective_overrides: dict[str, Any] | None = None
            if self._default_config_overrides or task.config_overrides:
                effective_overrides = {
                    **(self._default_config_overrides or {}),
                    **(task.config_overrides or {}),
                }

            effective_timeout = (
                task.timeout
                if task.timeout is not None
                else self._default_per_task_timeout
            )

            async def _bounded_complete(
                messages: list[LLMMessage],
                config_overrides: dict[str, Any] | None = None,
            ) -> LLMResponse:
                """Per-attempt timeout wrapper around provider.complete."""
                if effective_timeout is None:
                    return await self._provider.complete(
                        messages, config_overrides=config_overrides
                    )
                return await asyncio.wait_for(
                    self._provider.complete(
                        messages, config_overrides=config_overrides
                    ),
                    timeout=effective_timeout,
                )

            retry_config = task.retry or self._default_retry
            if retry_config:
                executor = RetryExecutor(retry_config)
                response = await executor.execute(
                    _bounded_complete,
                    task.messages,
                    config_overrides=effective_overrides,
                )
            else:
                response = await _bounded_complete(
                    task.messages,
                    config_overrides=effective_overrides,
                )
            duration = (time.monotonic() - start) * 1000
            logger.debug("LLM task '%s' completed in %.1fms", tag, duration)
            return TaskResult(
                tag=tag, success=True, value=response, duration_ms=duration
            )
        except asyncio.CancelledError:
            # Defensive: in Python 3.12+ CancelledError extends BaseException
            # so it already escapes the `except Exception` block below. The
            # explicit re-raise documents intent and guards against future
            # refactors broadening the catch (e.g., to BaseException).
            raise
        except asyncio.TimeoutError as e:
            duration = (time.monotonic() - start) * 1000
            logger.warning(
                "LLM task '%s' timed out after %.1fms", tag, duration
            )
            return TaskResult(
                tag=tag, success=False, value=None, error=e, duration_ms=duration
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            logger.warning(
                "LLM task '%s' failed after %.1fms: %s", tag, duration, e
            )
            return TaskResult(
                tag=tag, success=False, value=None, error=e, duration_ms=duration
            )

    async def _execute_single_deterministic(
        self, tag: str, task: DeterministicTask
    ) -> TaskResult:
        """Execute a single deterministic task.

        Sync callables are run in a thread executor to avoid blocking.

        Args:
            tag: Task identifier.
            task: The deterministic task to execute.

        Returns:
            TaskResult with success/failure status.
        """
        start = time.monotonic()
        logger.debug("Starting deterministic task '%s'", tag)
        effective_timeout = (
            task.timeout
            if task.timeout is not None
            else self._default_per_task_timeout
        )
        try:
            if asyncio.iscoroutinefunction(task.fn):
                inner = task.fn(*task.args, **task.kwargs)
            else:
                loop = asyncio.get_running_loop()
                inner = loop.run_in_executor(
                    None, lambda: task.fn(*task.args, **task.kwargs)
                )
            if effective_timeout is None:
                value = await inner
            else:
                value = await asyncio.wait_for(inner, timeout=effective_timeout)
            duration = (time.monotonic() - start) * 1000
            logger.debug("Deterministic task '%s' completed in %.1fms", tag, duration)
            return TaskResult(
                tag=tag, success=True, value=value, duration_ms=duration
            )
        except asyncio.CancelledError:
            # Defensive: see note in _execute_single_llm.
            raise
        except asyncio.TimeoutError as e:
            duration = (time.monotonic() - start) * 1000
            logger.warning(
                "Deterministic task '%s' timed out after %.1fms", tag, duration
            )
            return TaskResult(
                tag=tag, success=False, value=None, error=e, duration_ms=duration
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            logger.warning(
                "Deterministic task '%s' failed after %.1fms: %s", tag, duration, e
            )
            return TaskResult(
                tag=tag, success=False, value=None, error=e, duration_ms=duration
            )

    async def _gather_fail_fast(
        self,
        tagged_coros: list[tuple[str, Coroutine[Any, Any, TaskResult]]],
    ) -> list[TaskResult]:
        """Run tagged coroutines concurrently with fail-fast cancellation.

        On the first ``TaskResult`` with ``success=False``, all remaining
        pending tasks are cancelled. Tasks that completed before the
        cancellation trigger keep their original ``TaskResult``. Tasks that
        were cancelled mid-flight (or never started, e.g. queued behind a
        semaphore) are reported as
        ``TaskResult(success=False, error=asyncio.CancelledError(...))``.

        Result order matches the order of ``tagged_coros``.

        Args:
            tagged_coros: Pairs of (tag, coroutine). Each coroutine is
                expected to return a ``TaskResult`` for its tag — the inner
                ``_execute_single_*`` methods already guarantee this for
                non-cancelled completions.

        Returns:
            List of TaskResult, one per submitted coroutine, in submission
            order.
        """
        order: list[asyncio.Task[TaskResult]] = []
        start_times: dict[asyncio.Task[TaskResult], float] = {}
        tag_for: dict[asyncio.Task[TaskResult], str] = {}
        completed: dict[asyncio.Task[TaskResult], TaskResult] = {}

        for tag, coro in tagged_coros:
            t: asyncio.Task[TaskResult] = asyncio.create_task(coro, name=tag)
            order.append(t)
            start_times[t] = time.monotonic()
            tag_for[t] = tag

        failure_seen = False
        pending: set[asyncio.Task[TaskResult]] = set(order)
        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for d in done:
                    if d.cancelled():
                        # Cancelled by us below; resolved as CancelledError
                        # in the post-loop result assembly.
                        continue
                    # Reaching here implies `not d.cancelled()`, so calling
                    # `d.exception()` is safe — it returns the raised
                    # exception (or None on success) rather than re-raising
                    # CancelledError. CancelledError therefore cannot
                    # surface through this branch.
                    exc = d.exception()
                    if exc is None:
                        result = d.result()
                    else:
                        # Defensive: inner methods catch Exception and
                        # return TaskResult, so this branch should be
                        # unreachable in practice. If something does leak
                        # through, surface it as a failed TaskResult to
                        # preserve the result-map contract. `error` is
                        # typed `BaseException | None`, so the original
                        # exception is preserved without lossy wrapping.
                        logger.warning(
                            "Unexpected exception escaped task %r: %s",
                            tag_for[d],
                            exc,
                        )
                        result = TaskResult(
                            tag=tag_for[d],
                            success=False,
                            value=None,
                            error=exc,
                            duration_ms=(time.monotonic() - start_times[d])
                            * 1000,
                        )
                    completed[d] = result
                    if not result.success and not failure_seen:
                        failure_seen = True
                        for p in pending:
                            p.cancel()
        finally:
            # Make sure no orphan task remains; cancellation is idempotent.
            for t in order:
                if not t.done():
                    t.cancel()
            # Drain so cancellation completes before we return. Use
            # return_exceptions to swallow CancelledError surfaced via
            # gather without raising into our caller.
            await asyncio.gather(*order, return_exceptions=True)

        results: list[TaskResult] = []
        for t in order:
            if t in completed:
                results.append(completed[t])
            else:
                results.append(
                    TaskResult(
                        tag=tag_for[t],
                        success=False,
                        value=None,
                        error=asyncio.CancelledError(
                            "Cancelled by ParallelLLMExecutor fail_fast"
                        ),
                        duration_ms=(time.monotonic() - start_times[t]) * 1000,
                    )
                )
        return results
