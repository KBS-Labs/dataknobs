"""Behavioural tests for the FSM LLM transform functions.

Until this file, ``test_functions.py`` checked that the classes import and
that one error message keeps its shape. Nothing called ``transform()``. That
is how three defects reached a consumer, all of them silent:

1. ``FunctionCaller`` dispatched on ``asyncio.iscoroutinefunction``, which
   answers ``False`` for a callable *object* with an ``async def __call__``.
   Such a tool took the sync branch, where calling it merely *constructs* a
   coroutine, and the record came back carrying that coroutine with
   ``function_called`` set — a claim that a call happened when it had not.
2. The same sync branch called the consumer's function inline on the event
   loop, so a tool that does I/O stalled every other task on that loop.
3. ``LLMCaller`` and ``EmbeddingGenerator`` guard on ``LLMResource``, the
   *base*, then use the async API that only ``AsyncLLMResource`` supplies.
   The base has no ``generate`` at all and a **sync** ``embed``.

Each test below names which of those it pins.
"""

import inspect
import threading

import pytest
from dataknobs_fsm.functions.base import TransformError
from dataknobs_llm.fsm_integration import AsyncLLMResource, LLMResource
from dataknobs_llm.fsm_integration.functions import (
    EmbeddingGenerator,
    FunctionCaller,
    LLMCaller,
    PromptBuilder,
)
from dataknobs_llm.llm import EchoProvider, LLMConfig


def _echo_provider() -> EchoProvider:
    """An EchoProvider that needs no network."""
    return EchoProvider(LLMConfig(provider="echo", model="echo-test", options={}))


def _async_resource(name: str = "llm") -> AsyncLLMResource:
    """An AsyncLLMResource with an injected EchoProvider."""
    return AsyncLLMResource(
        name,
        provider="echo",
        model="echo-test",
        async_provider=_echo_provider(),
    )


def _sync_resource(name: str = "llm") -> LLMResource:
    """A plain (sync) LLMResource — the base class, built without network."""
    return LLMResource(name, provider="huggingface", model="local-test")


def _not_a_coroutine(value: object, field: str) -> object:
    """Fail with a diagnosis if ``value`` is an un-awaited coroutine."""
    if inspect.iscoroutine(value):
        value.close()
        pytest.fail(
            f"the callable was dispatched as sync: {field!r} holds an "
            "un-awaited coroutine instead of the call's result"
        )
    return value


class AsyncCallableTool:
    """A stateful tool — the shape ``iscoroutinefunction`` cannot see."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, x: int) -> int:
        self.calls += 1
        return x * 2


class SyncCallableTool:
    """A stateful *synchronous* tool; records the thread it ran on."""

    def __init__(self) -> None:
        self.calls = 0
        self.thread_name: str | None = None

    def __call__(self, x: int) -> int:
        self.calls += 1
        self.thread_name = threading.current_thread().name
        return x * 2


async def _double(x: int) -> int:
    return x * 2


def returns_a_coroutine(x: int) -> object:
    """A plain ``def`` whose *return value* is awaitable.

    No inspection of this callable can reveal that; only judging the result
    can.
    """
    return _double(x)


def _call_request(name: str, **arguments: object) -> dict:
    """The record shape ``FunctionCaller`` reads its call out of."""
    return {"llm_response": {"function": name, "arguments": arguments}}


class TestFunctionCallerDispatch:
    """Defects 1 and 2: how the registered tool is dispatched."""

    async def test_async_callable_object_is_actually_called(self):
        tool = AsyncCallableTool()
        caller = FunctionCaller(function_registry={"double": tool})

        result = await caller.transform(_call_request("double", x=21))

        assert _not_a_coroutine(result["function_result"], "function_result") == 42
        assert tool.calls == 1, "the tool's body never ran"
        assert result["function_called"] == "double"

    async def test_the_record_never_claims_a_call_that_did_not_happen(self):
        """``function_called`` and a real result are one fact, not two."""
        tool = AsyncCallableTool()
        caller = FunctionCaller(function_registry={"double": tool})

        result = await caller.transform(_call_request("double", x=1))

        if "function_called" in result:
            assert tool.calls == 1, (
                "the record reports the function was called, but the tool's body never ran"
            )

    async def test_sync_tool_runs_off_the_event_loop(self):
        """Defect 2: a consumer's tool may block; it must not block the loop."""
        tool = SyncCallableTool()
        caller = FunctionCaller(function_registry={"double": tool})
        loop_thread = threading.current_thread().name

        result = await caller.transform(_call_request("double", x=21))

        assert result["function_result"] == 42
        assert tool.thread_name is not None
        assert tool.thread_name != loop_thread, (
            "the tool ran on the event loop thread; a tool that blocks would "
            "stall every other task on that loop"
        )

    async def test_plain_function_returning_a_coroutine_is_awaited(self):
        """The shape no inspection of the callable can detect."""
        caller = FunctionCaller(function_registry={"double": returns_a_coroutine})

        result = await caller.transform(_call_request("double", x=21))

        assert _not_a_coroutine(result["function_result"], "function_result") == 42

    async def test_plain_async_function_still_awaited(self):
        """Regression guard: the one shape that already worked."""
        caller = FunctionCaller(function_registry={"double": _double})

        result = await caller.transform(_call_request("double", x=21))

        assert result["function_result"] == 42

    async def test_a_tool_taking_its_own_callback_argument(self):
        """LLM-authored arguments must not collide with the helper's own.

        ``arguments`` is parsed from model output, so a model emitting a
        ``callback`` key must not shadow a dispatch helper's parameter.
        """

        def register(callback: str, value: int) -> str:
            return f"{callback}:{value}"

        caller = FunctionCaller(function_registry={"register": register})

        result = await caller.transform(_call_request("register", callback="on_done", value=7))

        assert result["function_result"] == "on_done:7"


class TestFunctionCallerBaseline:
    """The contract this class has never had a test for."""

    async def test_no_response_field_returns_data_unchanged(self):
        caller = FunctionCaller(function_registry={"double": _double})
        data = {"other": "value"}

        assert await caller.transform(data) == data

    async def test_a_non_json_string_response_is_not_a_function_call(self):
        caller = FunctionCaller(function_registry={"double": _double})
        data = {"llm_response": "just prose, no call"}

        assert await caller.transform(data) == data

    async def test_a_response_without_a_function_name_is_passed_through(self):
        caller = FunctionCaller(function_registry={"double": _double})
        data = {"llm_response": {"arguments": {"x": 1}}}

        assert await caller.transform(data) == data

    async def test_a_json_string_response_is_parsed(self):
        caller = FunctionCaller(function_registry={"double": _double})

        result = await caller.transform(
            {"llm_response": '{"function": "double", "arguments": {"x": 21}}'}
        )

        assert result["function_result"] == 42

    async def test_an_unregistered_function_raises(self):
        caller = FunctionCaller(function_registry={})

        with pytest.raises(TransformError, match="Function not found: missing"):
            await caller.transform(_call_request("missing"))

    async def test_a_raising_tool_is_wrapped_without_leaking_its_message(self):
        def explode(**_: object) -> None:
            raise RuntimeError("secret internal detail")

        caller = FunctionCaller(function_registry={"explode": explode})

        with pytest.raises(TransformError) as excinfo:
            await caller.transform(_call_request("explode"))

        assert "secret internal detail" not in str(excinfo.value)
        assert "RuntimeError" in str(excinfo.value)


class TestResourceGuards:
    """Defect 3: the guard admits a base class that lacks the async API."""

    async def test_llm_caller_rejects_a_sync_resource_by_name(self):
        caller = LLMCaller(resource_name="llm")
        data = {"_resources": {"llm": _sync_resource()}, "prompt": "hi"}

        with pytest.raises(TransformError) as excinfo:
            await caller.transform(data)

        message = str(excinfo.value)
        assert "AsyncLLMResource" in message, (
            "the error must name what the resource has to be; today a base "
            "LLMResource reaches resource.generate() and fails with an "
            "AttributeError wrapped as 'LLM call failed'"
        )

    async def test_embedding_generator_rejects_a_sync_resource_by_name(self):
        generator = EmbeddingGenerator(resource_name="llm")
        data = {"_resources": {"llm": _sync_resource()}, "text": "hi"}

        with pytest.raises(TransformError) as excinfo:
            await generator.transform(data)

        assert "AsyncLLMResource" in str(excinfo.value), (
            "today a base LLMResource's sync embed() returns a list, which is "
            "then awaited and fails with a TypeError"
        )

    async def test_llm_caller_still_reports_a_genuinely_missing_resource(self):
        caller = LLMCaller(resource_name="llm")

        with pytest.raises(TransformError, match="not found"):
            await caller.transform({"_resources": {}, "prompt": "hi"})

    async def test_llm_caller_generates_with_an_async_resource(self):
        resource = _async_resource()
        caller = LLMCaller(resource_name="llm")
        data = {"_resources": {"llm": resource}, "prompt": "Hello world"}

        try:
            result = await caller.transform(data)
        finally:
            await resource.aclose()

        response = _not_a_coroutine(result["llm_response"], "llm_response")
        assert "Hello world" in response["choices"][0]["text"]

    async def test_embedding_generator_embeds_with_an_async_resource(self):
        resource = _async_resource()
        generator = EmbeddingGenerator(resource_name="llm")
        data = {"_resources": {"llm": resource}, "text": "embed me"}

        try:
            result = await generator.transform(data)
        finally:
            await resource.aclose()

        vectors = _not_a_coroutine(result["embedding"], "embedding")
        assert isinstance(vectors, list)
        assert vectors and isinstance(vectors[0], list)

    async def test_embedding_generator_batches_a_list(self):
        resource = _async_resource()
        generator = EmbeddingGenerator(resource_name="llm", batch_size=2)
        data = {"_resources": {"llm": resource}, "text": ["a", "b", "c"]}

        try:
            result = await generator.transform(data)
        finally:
            await resource.aclose()

        assert len(result["embedding"]) == 3

    async def test_embedding_generator_passes_through_empty_text(self):
        generator = EmbeddingGenerator(resource_name="llm")
        data = {"_resources": {"llm": _async_resource()}, "text": ""}

        assert await generator.transform(data) == data


class TestLLMCallerResponseContract:
    """A non-streaming `generate()` must hand back a dict, and say so if not."""

    async def test_a_non_dict_response_is_reported_not_silently_dropped(self):
        """The narrowing must not turn a broken contract into a quiet None.

        `tokens_used` is read off the response, so a non-dict response used to
        raise `AttributeError` and surface -- uninformatively -- as "LLM call
        failed (AttributeError)". Narrowing the type for the checker must not
        replace that with `tokens_used: None` and no error at all: this
        module's whole subject is calls that report success without having
        worked.
        """

        class BrokenResource(AsyncLLMResource):
            """Violates `generate()`'s own contract for a non-streaming call."""

            async def generate(self, *args: object, **kwargs: object) -> object:
                return ["not", "a", "dict"]

        resource = BrokenResource(
            "llm",
            provider="echo",
            model="echo-test",
            async_provider=_echo_provider(),
        )
        caller = LLMCaller(resource_name="llm", stream=False)
        data = {"_resources": {"llm": resource}, "prompt": "hi"}

        with pytest.raises(TransformError) as excinfo:
            await caller.transform(data)

        message = str(excinfo.value)
        assert "list" in message, "the error must name what came back"
        assert "dict" in message, "the error must name what was required"

    async def test_the_contract_error_is_not_swallowed_by_the_call_wrapper(self):
        """The message must survive, not become 'failed (TransformError)'.

        The provider call is wrapped by a blanket handler that reports only
        the exception's type name -- deliberately, so vendor text cannot
        reach a caller. A contract error raised inside that handler's reach
        would be re-wrapped and its message lost.
        """

        class BrokenResource(AsyncLLMResource):
            async def generate(self, *args: object, **kwargs: object) -> object:
                return 42

        resource = BrokenResource(
            "llm",
            provider="echo",
            model="echo-test",
            async_provider=_echo_provider(),
        )
        caller = LLMCaller(resource_name="llm")

        with pytest.raises(TransformError) as excinfo:
            await caller.transform({"_resources": {"llm": resource}, "prompt": "hi"})

        assert "TransformError" not in str(excinfo.value), (
            "the contract error was re-wrapped by the blanket handler and its "
            "message replaced with the wrapper's type name"
        )

    async def test_a_provider_failure_still_reports_only_its_type(self):
        """The disclosure bound on the provider call is unchanged."""

        class ExplodingResource(AsyncLLMResource):
            async def generate(self, *args: object, **kwargs: object) -> object:
                raise RuntimeError("endpoint https://internal.example/v1 refused")

        resource = ExplodingResource(
            "llm",
            provider="echo",
            model="echo-test",
            async_provider=_echo_provider(),
        )
        caller = LLMCaller(resource_name="llm")

        with pytest.raises(TransformError) as excinfo:
            await caller.transform({"_resources": {"llm": resource}, "prompt": "hi"})

        message = str(excinfo.value)
        assert "internal.example" not in message, "vendor text reached the caller"
        assert "RuntimeError" in message


class TestPromptBuilderNestedVariables:
    """The nested-access path, whose local is re-bound to None on a miss."""

    def test_a_nested_variable_is_extracted(self):
        builder = PromptBuilder(template="Hi {user.name}", variables=["user.name"])

        result = builder.transform({"user": {"name": "Ada"}})

        assert result["prompt"] == "Hi Ada"

    def test_a_missing_nested_variable_reports_the_name(self):
        builder = PromptBuilder(template="Hi {user.name}", variables=["user.name"])

        with pytest.raises(TransformError, match="Missing variable"):
            builder.transform({"user": {}})

    def test_a_variable_that_bottoms_out_on_a_non_dict(self):
        builder = PromptBuilder(template="Hi {user.name}", variables=["user.name"])

        with pytest.raises(TransformError, match="Missing variable"):
            builder.transform({"user": "not-a-dict"})
