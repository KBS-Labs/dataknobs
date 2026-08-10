"""Anthropic SDK stand-ins, shared by the Anthropic provider test modules.

No dataknobs testing construct produces a real Anthropic request or response,
so these are the sanctioned narrow case: thin stand-ins at the SDK boundary
that let the real provider wiring (``adapt_messages`` → ``_build_api_kwargs``
→ ``messages.create`` → ``adapt_response``) run end to end without a live API
or the ``anthropic`` package installed.

Underscore-prefixed so pytest does not collect it. It lives in its own module
because four test modules need these stand-ins, and they reached them by
importing ``test_anthropic_param_handling`` and
``test_anthropic_model_constraints`` — which resolves only while pytest is
inserting each collected file's directory onto ``sys.path``, and which imports
an already-collected module a second time under a second name, running its
module-level code twice. ``test_anthropic_model_constraints`` clears a
process-global cache at module scope, so running it twice was not free.
"""

from __future__ import annotations

import asyncio
from typing import Any

from dataknobs_llm.llm.base import LLMConfig
from dataknobs_llm.llm.providers.anthropic import AnthropicProvider

__all__ = [
    "_AsyncModelPage",
    "_CaptureAnthropicClient",
    "_ModelsStub",
    "_ScriptedModel",
    "_SlowModelPage",
    "_SlowModelsStub",
    "_provider_with_capture",
    "make_anthropic_response",
]


def make_anthropic_response(
    content_blocks: list[dict],
    model: str = "claude-3",
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> object:
    """Build a fake Anthropic Message-like response object.

    Avoids a hard test dependency on the ``anthropic`` package while
    faithfully reproducing the attribute-access interface of
    ``anthropic.types.Message``.
    """
    class Block:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    class Usage:
        def __init__(self) -> None:
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    class Response:
        def __init__(self) -> None:
            self.content = [Block(**b) for b in content_blocks]
            self.model = model
            self.stop_reason = stop_reason
            self.usage = Usage()

    return Response()


class _ScriptedModel:
    """A minimal ``anthropic`` ``ModelInfo`` stand-in.

    Carries ``id`` + ``max_tokens`` (the output ceiling) + ``max_input_tokens``
    (the input/context ceiling). ``max_input_tokens`` defaults to ``None`` so the
    many existing two-arg constructions keep working; the input-ceiling tests
    pass it explicitly.
    """

    def __init__(
        self,
        model_id: str,
        max_tokens: int | None,
        max_input_tokens: int | None = None,
    ) -> None:
        self.id = model_id
        self.max_tokens = max_tokens
        self.max_input_tokens = max_input_tokens


class _AsyncModelPage:
    """Async-iterable page mimicking the SDK's ``AsyncPaginator``."""

    def __init__(self, models: list[Any]) -> None:
        self._it = iter(models)

    def __aiter__(self) -> _AsyncModelPage:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


class _ModelsStub:
    """Stand-in for ``client.models`` — scripts ``list()`` + tracks calls."""

    def __init__(self) -> None:
        self.models: list[Any] = []
        self.list_calls = 0
        self.raise_on_list = False

    def list(self, **_kwargs: Any) -> _AsyncModelPage:
        self.list_calls += 1
        if self.raise_on_list:
            raise RuntimeError("simulated Models API failure")
        return _AsyncModelPage(list(self.models))


class _SlowModelPage:
    """Async-iterable page whose first step sleeps — a *hung* Models API."""

    def __init__(self, delay: float) -> None:
        self._delay = delay

    def __aiter__(self) -> _SlowModelPage:
        return self

    async def __anext__(self) -> Any:
        # Block as if the control-plane hung, then end (never yields a model).
        await asyncio.sleep(self._delay)
        raise StopAsyncIteration


class _SlowModelsStub:
    """``client.models`` stand-in whose ``list()`` hangs for ``delay`` seconds."""

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self.list_calls = 0

    def list(self, **_kwargs: Any) -> _SlowModelPage:
        self.list_calls += 1
        return _SlowModelPage(self._delay)


class _CaptureAnthropicClient:
    """Records the kwargs passed to ``messages.create``.

    Minimal stand-in for ``anthropic.AsyncAnthropic`` — a sanctioned SDK
    stand-in (no dataknobs testing construct returns a real Anthropic
    request/response). Exercises the real ``AnthropicProvider.complete``
    wiring (``adapt_messages`` → ``_build_api_kwargs`` → ``messages.create``
    → ``adapt_response``) without a live API or the ``anthropic`` package. The
    ``models`` sub-stub scripts the Models-API ``list()`` used by the dynamic
    ``max_tokens``-ceiling resolution.
    """

    def __init__(self) -> None:
        self.captured_kwargs: dict[str, Any] = {}
        # ``provider._client.messages.create`` → this object's ``create``.
        self.messages = self
        # ``provider._client.models.list`` → the scripted models stub.
        self.models = _ModelsStub()

    async def create(self, **kwargs: Any) -> object:
        self.captured_kwargs = kwargs
        return make_anthropic_response([{"type": "text", "text": "ok"}])


def _provider_with_capture(
    model: str, **config_kwargs: Any
) -> tuple[AnthropicProvider, _CaptureAnthropicClient]:
    """Build an initialised ``AnthropicProvider`` backed by a capture client."""
    provider = AnthropicProvider(
        LLMConfig(provider="anthropic", model=model, **config_kwargs)
    )
    client = _CaptureAnthropicClient()
    provider._client = client
    provider._is_initialized = True
    return provider, client
