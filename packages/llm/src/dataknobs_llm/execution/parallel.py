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
from collections.abc import Callable
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
    """

    messages: list[LLMMessage]
    config_overrides: dict[str, Any] | None = None
    retry: RetryConfig | None = None
    tag: str = ""


@dataclass
class DeterministicTask:
    """A sync or async callable to execute alongside LLM tasks.

    Attributes:
        fn: The callable to execute. May be sync or async.
        args: Positional arguments forwarded to fn.
        kwargs: Keyword arguments forwarded to fn.
        tag: Identifier for result lookup.
    """

    fn: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    tag: str = ""


@dataclass
class TaskResult:
    """Result of a single task execution.

    Attributes:
        tag: The task identifier.
        success: Whether the task completed without error.
        value: The return value. ``LLMResponse`` for LLM tasks, arbitrary for
            deterministic tasks. ``None`` on failure.
        error: The exception if the task failed, ``None`` otherwise.
        duration_ms: Wall-clock execution time in milliseconds.
    """

    tag: str
    success: bool
    value: LLMResponse | Any
    error: Exception | None = None
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
    ) -> None:
        """Initialize the executor.

        Args:
            provider: The LLM provider to use for LLM tasks.
            max_concurrency: Maximum number of concurrent tasks.
            default_retry: Default retry policy applied to tasks that do not
                specify their own.
        """
        self._provider = provider
        self._max_concurrency = max_concurrency
        self._default_retry = default_retry

    async def execute(
        self,
        tasks: dict[str, LLMTask],
    ) -> dict[str, TaskResult]:
        """Run LLM tasks concurrently with error isolation.

        Args:
            tasks: Mapping of tag to LLMTask. Tags identify results.

        Returns:
            Mapping of tag to TaskResult.
        """
        if not tasks:
            return {}

        semaphore = asyncio.Semaphore(self._max_concurrency)
        start_all = time.monotonic()

        async def _run(tag: str, task: LLMTask) -> TaskResult:
            async with semaphore:
                return await self._execute_single_llm(tag, task)

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
    ) -> dict[str, TaskResult]:
        """Run a mix of LLM and deterministic tasks concurrently.

        Args:
            tasks: Mapping of tag to task (LLMTask or DeterministicTask).

        Returns:
            Mapping of tag to TaskResult.
        """
        if not tasks:
            return {}

        semaphore = asyncio.Semaphore(self._max_concurrency)
        start_all = time.monotonic()

        async def _run(tag: str, task: LLMTask | DeterministicTask) -> TaskResult:
            async with semaphore:
                if isinstance(task, LLMTask):
                    return await self._execute_single_llm(tag, task)
                return await self._execute_single_deterministic(tag, task)

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
    ) -> list[TaskResult]:
        """Run LLM tasks sequentially, optionally passing results forward.

        When ``pass_result`` is True, each task's messages are augmented with
        the previous task's response as an assistant message.

        Args:
            tasks: Ordered list of LLM tasks to run.
            pass_result: If True, append previous result as assistant message.

        Returns:
            List of TaskResult in execution order.
        """
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
            retry_config = task.retry or self._default_retry
            if retry_config:
                executor = RetryExecutor(retry_config)
                response = await executor.execute(
                    self._provider.complete,
                    task.messages,
                    config_overrides=task.config_overrides,
                )
            else:
                response = await self._provider.complete(
                    task.messages,
                    config_overrides=task.config_overrides,
                )
            duration = (time.monotonic() - start) * 1000
            logger.debug("LLM task '%s' completed in %.1fms", tag, duration)
            return TaskResult(
                tag=tag, success=True, value=response, duration_ms=duration
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
        try:
            if asyncio.iscoroutinefunction(task.fn):
                value = await task.fn(*task.args, **task.kwargs)
            else:
                loop = asyncio.get_running_loop()
                value = await loop.run_in_executor(
                    None, lambda: task.fn(*task.args, **task.kwargs)
                )
            duration = (time.monotonic() - start) * 1000
            logger.debug("Deterministic task '%s' completed in %.1fms", tag, duration)
            return TaskResult(
                tag=tag, success=True, value=value, duration_ms=duration
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            logger.warning(
                "Deterministic task '%s' failed after %.1fms: %s", tag, duration, e
            )
            return TaskResult(
                tag=tag, success=False, value=None, error=e, duration_ms=duration
            )
