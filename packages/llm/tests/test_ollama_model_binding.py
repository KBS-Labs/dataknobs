"""Tests for the Ollama model-metadata binding (live-first, local, no pricing).

Ollama is the first local, live-first binding: the server authoritatively
reports each installed model's capabilities + context window via ``/api/show``,
so the binding sources those facets from the live API (with a name-based
heuristic fallback) and folds availability into the substrate — replacing the
pre-binding hardcoded capability-substring lists.

Ollama has no SDK — it speaks HTTP over ``aiohttp`` — so these drive the real
``OllamaProvider`` and real ``LiveApiSource`` through a routing async-session
stand-in at the transport edge (``RoutingSession`` below: real async context
managers returning canned ``/api/tags`` + ``/api/show`` + ``/api/chat`` JSON,
not a ``MagicMock``). Reproduce-first: the matcher + capability tests FAIL
against the pre-binding code and pass after.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_common.testing import (
    assert_no_blocking,
    is_blockbuster_available,
)
from dataknobs_llm.llm.base import LLMConfig, ModelCapability
from dataknobs_llm.llm.model_profile import (
    LiveApiSource,
    ModelProfile,
    match_family_key,
)
from dataknobs_llm.llm.providers.ollama import (
    OllamaProvider,
    _ollama_live_extractor,
    ollama_match_key,
)

from _aiohttp_error_stub import FakeResponse


# --------------------------------------------------------------------------
# Routing aiohttp session stand-in (GET /api/tags, POST /api/show + /api/chat)
# --------------------------------------------------------------------------


class _Ctx:
    """Async-context-manager stand-in yielding a scripted response (or raising)."""

    def __init__(
        self, resp: FakeResponse | None = None, exc: Exception | None = None
    ) -> None:
        self._resp = resp
        self._exc = exc

    async def __aenter__(self) -> FakeResponse:
        if self._exc is not None:
            raise self._exc
        assert self._resp is not None
        return self._resp

    async def __aexit__(self, *exc: object) -> bool:
        return False


class RoutingSession:
    """Minimal ``aiohttp.ClientSession`` stand-in routing Ollama endpoints.

    ``GET /api/tags`` returns the installed set; ``POST /api/show`` returns the
    per-model ``show`` payload (404 for an unlisted model); ``POST /api/chat``
    returns a canned completion. Connection failures are simulated via
    ``tags_exc`` / ``show_exc`` raised on context entry.
    """

    def __init__(
        self,
        *,
        installed: tuple[str, ...] = (),
        show: dict[str, dict[str, Any]] | None = None,
        tags_status: int = 200,
        tags_exc: Exception | None = None,
        show_exc: Exception | None = None,
        chat: dict[str, Any] | None = None,
    ) -> None:
        self._installed = list(installed)
        self._show = show or {}
        self._tags_status = tags_status
        self._tags_exc = tags_exc
        self._show_exc = show_exc
        self._chat = chat
        self.get_urls: list[str] = []
        self.post_urls: list[tuple[str, Any]] = []

    def get(self, url: str) -> _Ctx:
        self.get_urls.append(url)
        if self._tags_exc is not None:
            return _Ctx(exc=self._tags_exc)
        body = {"models": [{"name": n} for n in self._installed]}
        return _Ctx(resp=FakeResponse(self._tags_status, json_data=body))

    def post(self, url: str, json: Any = None) -> _Ctx:
        self.post_urls.append((url, json))
        if url.endswith("/api/show"):
            if self._show_exc is not None:
                return _Ctx(exc=self._show_exc)
            entry = self._show.get((json or {}).get("model"))
            if entry is None:
                return _Ctx(resp=FakeResponse(404, json_data={}))
            return _Ctx(resp=FakeResponse(200, json_data=entry))
        if url.endswith("/api/chat"):
            return _Ctx(
                resp=FakeResponse(
                    200,
                    json_data=self._chat
                    or {"message": {"content": "ok"}, "done": True},
                )
            )
        return _Ctx(resp=FakeResponse(200, json_data={}))


def _provider(
    session: RoutingSession, model: str = "llama3.1:8b", **cfg: Any
) -> OllamaProvider:
    provider = OllamaProvider(LLMConfig(provider="ollama", model=model, **cfg))
    provider._session = session  # type: ignore[assignment]
    provider._is_initialized = True
    return provider


def _show(
    capabilities: list[str] | None = None,
    *,
    arch: str = "llama",
    context_length: int | None = None,
) -> dict[str, Any]:
    """Build an ``/api/show`` payload with optional capabilities + context."""
    payload: dict[str, Any] = {}
    if capabilities is not None:
        payload["capabilities"] = capabilities
    if context_length is not None:
        payload["model_info"] = {
            "general.architecture": arch,
            f"{arch}.context_length": context_length,
        }
    return payload


# --------------------------------------------------------------------------
# The additive LiveApiSource match= seam (reproduce-first)
# --------------------------------------------------------------------------


class TestOllamaMatchKeySeam:
    """The substrate default matcher mis-resolves Ollama ids; the injected
    ``ollama_match_key`` fixes it — without changing the default.
    """

    def test_substrate_default_matcher_reintroduces_prefix_bug(self) -> None:
        """Documents the greedy-substring hazard the seam exists to close."""
        assert (
            match_family_key(
                "nomic-embed-text", ["nomic-embed-text-v2-moe:latest"]
            )
            == "nomic-embed-text-v2-moe:latest"
        )
        assert (
            match_family_key("llama2", ["llama2-uncensored:latest"])
            == "llama2-uncensored:latest"
        )

    def test_ollama_match_key_rejects_prefix_collision(self) -> None:
        assert (
            ollama_match_key(
                "nomic-embed-text", ["nomic-embed-text-v2-moe:latest"]
            )
            is None
        )
        assert ollama_match_key("llama2", ["llama2-uncensored:latest"]) is None

    def test_ollama_match_key_base_name_matches_tagged(self) -> None:
        assert (
            ollama_match_key("llama3.1", ["llama3.1:8b", "mistral:latest"])
            == "llama3.1:8b"
        )

    def test_ollama_match_key_exact_and_family_fallback(self) -> None:
        keys = ["llama3.1:8b", "llama3.1:70b"]
        # An exact id resolves exactly.
        assert ollama_match_key("llama3.1:8b", keys) == "llama3.1:8b"
        # A requested tag that is absent falls back to the base family (the
        # documented _find_matching_models tag-fallback), not a false-match.
        assert ollama_match_key("llama3.1:405b", keys) == "llama3.1:8b"
        # A genuinely different family does not match at all.
        assert ollama_match_key("mistral:7b", keys) is None

    def _seeded_source(self, match: Any) -> LiveApiSource:
        source = LiveApiSource(
            lambda: [],  # walker unused — seeded directly
            _ollama_live_extractor,
            match=match,
        )
        source.seed(
            "nomic-embed-text-v2-moe:latest",
            ModelProfile(
                capabilities=frozenset({ModelCapability.EMBEDDINGS}),
                available=True,
            ),
        )
        return source

    def test_live_source_default_match_false_resolves(self) -> None:
        """With the substrate default, the live cache false-resolves (the bug)."""
        source = self._seeded_source(match_family_key)
        assert source.resolve("nomic-embed-text").available is True

    def test_live_source_ollama_match_no_false_resolve(self) -> None:
        """With the injected matcher, an absent bare id does not false-resolve."""
        source = self._seeded_source(ollama_match_key)
        assert source.resolve("nomic-embed-text").available is None


# --------------------------------------------------------------------------
# Live capabilities (reproduce-first: FAIL on pre-binding, PASS after)
# --------------------------------------------------------------------------


class TestOllamaCapabilitiesLive:
    async def _caps(
        self, session: RoutingSession, model: str
    ) -> set[ModelCapability]:
        provider = _provider(session, model=model)
        await provider._live_source.refresh_if_stale()
        return set(provider._detect_capabilities())

    async def test_completion_tools_vision(self) -> None:
        session = RoutingSession(
            installed=("qwen2.5vl:7b",),
            show={"qwen2.5vl:7b": _show(["completion", "tools", "vision"])},
        )
        caps = await self._caps(session, "qwen2.5vl:7b")
        assert caps == {
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
            ModelCapability.JSON_MODE,
            ModelCapability.EMBEDDINGS,
        }

    async def test_modern_family_pre_binding_missed(self) -> None:
        """A family the pre-binding substring lists missed is now live-detected."""
        session = RoutingSession(
            installed=("llama4:latest",),
            show={"llama4:latest": _show(["completion", "tools", "vision"])},
        )
        caps = await self._caps(session, "llama4:latest")
        assert ModelCapability.FUNCTION_CALLING in caps
        assert ModelCapability.VISION in caps

    async def test_code_capability_name_derived(self) -> None:
        session = RoutingSession(
            installed=("codellama:13b",),
            show={"codellama:13b": _show(["completion"])},
        )
        caps = await self._caps(session, "codellama:13b")
        assert ModelCapability.CODE in caps
        assert ModelCapability.CHAT in caps

    async def test_embedding_only_model_is_disjoint(self) -> None:
        """A server reporting only ``embedding`` resolves EMBEDDINGS-only."""
        session = RoutingSession(
            installed=("nomic-embed-text:latest",),
            show={"nomic-embed-text:latest": _show(["embedding"])},
        )
        caps = await self._caps(session, "nomic-embed-text:latest")
        assert caps == {ModelCapability.EMBEDDINGS}


class TestOllamaCapabilitiesHeuristicFallback:
    async def _caps(
        self, session: RoutingSession, model: str
    ) -> set[ModelCapability]:
        provider = _provider(session, model=model)
        await provider._live_source.refresh_if_stale()
        return set(provider._detect_capabilities())

    async def test_show_without_capabilities_falls_back(self) -> None:
        """An older server (no ``capabilities`` field) → name-based heuristic."""
        session = RoutingSession(
            installed=("mistral:7b",),
            show={"mistral:7b": _show(context_length=32768)},
        )
        caps = await self._caps(session, "mistral:7b")
        # Heuristic classifies the tool family; context still comes from show.
        assert ModelCapability.FUNCTION_CALLING in caps
        assert ModelCapability.JSON_MODE in caps

    async def test_show_error_falls_back(self) -> None:
        """A ``/api/show`` transport error degrades to the heuristic (no crash)."""
        import aiohttp

        session = RoutingSession(
            installed=("llava:13b",),
            show_exc=aiohttp.ClientConnectionError("show failed"),
        )
        caps = await self._caps(session, "llava:13b")
        assert ModelCapability.VISION in caps  # name-derived fallback


class TestOllamaContextWindow:
    async def test_context_window_populates_max_input_tokens(self) -> None:
        """Reproduce-first: max_input_tokens was always None for Ollama."""
        session = RoutingSession(
            installed=("llama3.1:8b",),
            show={
                "llama3.1:8b": _show(["completion", "tools"], context_length=131072)
            },
        )
        provider = _provider(session, model="llama3.1:8b")
        await provider._live_source.refresh_if_stale()
        assert provider.get_constraints().max_input_tokens == 131072

    async def test_no_context_length_yields_none(self) -> None:
        session = RoutingSession(
            installed=("phi3:mini",),
            show={"phi3:mini": _show(["completion"])},
        )
        provider = _provider(session, model="phi3:mini")
        await provider._live_source.refresh_if_stale()
        assert provider.get_constraints().max_input_tokens is None


class TestOllamaPricing:
    """Ollama sources no pricing itself, but a consumer override lights it up."""

    def test_no_pricing_by_default(self) -> None:
        provider = _provider(RoutingSession(), model="llama3.1:8b")
        assert provider.get_pricing() is None

    def test_pricing_override_lights_up_get_pricing(self) -> None:
        provider = _provider(
            RoutingSession(),
            model="llama3.1:8b",
            model_profile_overrides={
                "pricing": {"input_per_mtok": 0.5, "output_per_mtok": 1.5}
            },
        )
        pricing = provider.get_pricing()
        assert pricing is not None
        assert pricing.input_per_mtok == 0.5
        assert pricing.output_per_mtok == 1.5


class TestOllamaValidateModel:
    async def test_installed_model_validates_true(self) -> None:
        session = RoutingSession(
            installed=("llama3.1:8b",),
            show={"llama3.1:8b": _show(["completion"])},
        )
        provider = _provider(session, model="llama3.1")  # bare alias
        assert await provider.validate_model() is True

    async def test_not_installed_model_validates_false(self) -> None:
        session = RoutingSession(
            installed=("mistral:7b",), show={"mistral:7b": _show(["completion"])}
        )
        provider = _provider(session, model="llama3.1:8b")
        assert await provider.validate_model() is False

    async def test_unreachable_server_validates_false(self) -> None:
        import aiohttp

        session = RoutingSession(tags_exc=aiohttp.ClientConnectionError("down"))
        provider = _provider(session, model="llama3.1:8b")
        assert await provider.validate_model() is False


class TestOllamaShapingParity:
    """Request-shaping choke point: no-op by default, honors consumer overrides.

    Ollama has no output ceiling (``num_predict: -1`` = unlimited) and no
    rejected params by default, so shaping is a byte-identical no-op unless a
    consumer declares ``constraints``.
    """

    def test_default_options_byte_identical_noop(self) -> None:
        session = RoutingSession()
        provider = _provider(
            session, model="llama3.1:8b", temperature=0.7, max_tokens=999999
        )
        # Shaped == unshaped: no clamp (no ceiling), no drop, no remap.
        assert provider._build_shaped_options(
            provider.config
        ) == provider._build_options(provider.config)
        # A large num_predict passes through un-clamped (no output ceiling).
        assert (
            provider._build_shaped_options(provider.config)["num_predict"]
            == 999999
        )

    def test_rejected_params_override_drops_param(self) -> None:
        session = RoutingSession()
        provider = _provider(
            session,
            model="llama3.1:8b",
            temperature=0.7,
            constraints={"rejected_params": ["temperature"]},
        )
        options = provider._build_shaped_options(provider.config)
        assert "temperature" not in options

    def test_ceiling_override_clamps_num_predict(self) -> None:
        session = RoutingSession()
        provider = _provider(
            session,
            model="llama3.1:8b",
            max_tokens=8192,
            constraints={"max_tokens_ceiling": 4096},
        )
        options = provider._build_shaped_options(provider.config)
        assert options["num_predict"] == 4096


@pytest.mark.skipif(
    not is_blockbuster_available(), reason="blockbuster not installed"
)
class TestOllamaNoBlocking:
    async def test_request_path_does_not_block_the_loop(self) -> None:
        session = RoutingSession(
            installed=("llama3.1:8b",),
            show={"llama3.1:8b": _show(["completion", "tools"], context_length=131072)},
        )
        provider = _provider(session, model="llama3.1:8b")
        with assert_no_blocking():
            await provider.complete("hello")
