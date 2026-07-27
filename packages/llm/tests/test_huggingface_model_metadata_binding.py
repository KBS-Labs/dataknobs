"""Reproduce-first tests for the HuggingFace model-metadata binding.

Covers the two workstreams of the binding:

- **W-A** — the declarative heuristic-primary binding replacing the pre-binding
  inline capability substring lists and the hardcoded ``max_new_tokens=100``,
  plus the request-shaping choke point, the config-override path for every
  facet, and the ``validate_model`` availability pin.
- **W-B** — the additive ``match=`` seam on
  :class:`~dataknobs_llm.llm.model_profile.ConfigOverrideSource`, injected by HF
  as an exact repo-id matcher so a per-repo override map does not resolve a base
  repo to a prefix-sharing variant's override (the substring prefix-collision).

HuggingFace has no SDK — it speaks HTTP over ``aiohttp`` — so the boundary is
the HTTP session. Tests that touch the wire use a capturing session stand-in
(records the posted ``parameters`` / routes ``get`` to a scripted status),
following the sanctioned no-SDK transport-edge pattern of
``test_huggingface_error_handling.py`` (not a ``MagicMock``).
"""

from __future__ import annotations

from typing import Any

from dataknobs_llm.llm.base import LLMConfig, ModelCapability
from dataknobs_llm.llm.model_profile import (
    ConfigOverrideSource,
    match_family_key,
)
from dataknobs_llm.llm.providers.huggingface import (
    _HF_DEFAULT_MAX_NEW_TOKENS,
    HuggingFaceProvider,
    hf_match_key,
)


# ---------------------------------------------------------------------------
# HTTP session stand-in (captures POST payloads; scripts GET status)
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status: int = 200, json_data: Any = None) -> None:
        self.status = status
        self._json = json_data if json_data is not None else [{"generated_text": "ok"}]

    async def text(self) -> str:
        return ""

    async def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        return None


class _Ctx:
    def __init__(self, resp: _Resp) -> None:
        self._resp = resp

    async def __aenter__(self) -> _Resp:
        return self._resp

    async def __aexit__(self, *exc: object) -> None:
        return None


class CapturingSession:
    """Records POST bodies and answers GET with a scripted status.

    ``post`` captures each JSON payload (so a test can assert the shaped
    ``parameters`` dict) and returns a canned generation response; ``get``
    returns a response whose ``status`` is the scripted ``get_status`` (for
    ``validate_model``). ``get_calls`` records the probed URLs so a test can
    prove the probe was (or was not) reached.
    """

    def __init__(self, *, get_status: int = 200) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.get_calls: list[str] = []
        self._get_status = get_status

    def post(self, url: str, json: Any = None) -> _Ctx:
        self.payloads.append(json)
        return _Ctx(_Resp(200))

    def get(self, url: str) -> _Ctx:
        self.get_calls.append(url)
        return _Ctx(_Resp(self._get_status))


def _provider(session: Any = None, **config_kwargs: Any) -> HuggingFaceProvider:
    config_kwargs.setdefault("model", "gpt2")
    provider = HuggingFaceProvider(
        LLMConfig(provider="huggingface", **config_kwargs)
    )
    if session is not None:
        provider._session = session
        provider._is_initialized = True
    return provider


# ---------------------------------------------------------------------------
# W-B — the ConfigOverrideSource match= seam (resolves W2-OLLAMA-FU1)
# ---------------------------------------------------------------------------


class TestConfigOverrideMatcherSeam:
    """A per-repo override map must not resolve a base repo to a variant override."""

    PER_REPO = {
        "meta-llama/Llama-3.1-8B-Instruct": {"context_window": 4096},
    }

    def test_default_matcher_reintroduces_the_prefix_collision(self) -> None:
        # Reproduce: with the substrate default (match_family_key), the base repo
        # id is a substring of the -Instruct key, so the bare-alias direction
        # fires and the base request wrongly resolves the -Instruct override.
        source = ConfigOverrideSource(self.PER_REPO)  # default match_family_key
        profile = source.resolve("meta-llama/Llama-3.1-8B")
        assert profile.context_window == 4096  # the collision — wrong override

    def test_hf_matcher_fixes_the_collision(self) -> None:
        # The exact matcher returns no match for the base repo (not a declared
        # key), so it resolves an empty profile instead of the -Instruct facets.
        source = ConfigOverrideSource(self.PER_REPO, match=hf_match_key)
        profile = source.resolve("meta-llama/Llama-3.1-8B")
        assert profile.context_window is None  # no false match

    def test_hf_matcher_resolves_an_exact_repo_id(self) -> None:
        source = ConfigOverrideSource(self.PER_REPO, match=hf_match_key)
        profile = source.resolve("meta-llama/Llama-3.1-8B-Instruct")
        assert profile.context_window == 4096

    def test_flat_single_model_override_sidesteps_the_matcher(self) -> None:
        # A facet-keyed flat mapping applies to the configured model regardless
        # of matcher (the flat branch does no key matching).
        flat = {"context_window": 8192}
        assert (
            ConfigOverrideSource(flat, match=hf_match_key)
            .resolve("any/repo")
            .context_window
            == 8192
        )
        assert (
            ConfigOverrideSource(flat).resolve("any/repo").context_window == 8192
        )

    def test_default_matcher_is_match_family_key(self) -> None:
        # The default keeps every existing adopter (openai/bedrock/ollama/
        # anthropic — none pass match=) byte-identical.
        source = ConfigOverrideSource(None)
        assert source._match is match_family_key


class TestHFMatchKey:
    """The exact repo-id matcher used by HuggingFace."""

    def test_exact_match_returns_key(self) -> None:
        assert hf_match_key("a/b", ["a/b", "a/b-instruct"]) == "a/b"

    def test_no_substring_match(self) -> None:
        assert hf_match_key("a/b", ["a/b-instruct"]) is None

    def test_no_match_returns_none(self) -> None:
        assert hf_match_key("x/y", ["a/b", "c/d"]) is None


# ---------------------------------------------------------------------------
# W-A — capabilities (reproduce-first: HEAD's bare 'embedding' test missed these)
# ---------------------------------------------------------------------------


class TestCapabilities:
    def _caps(self, model: str, **kw: Any) -> set[ModelCapability]:
        return set(_provider(model=model, **kw).get_capabilities())

    def test_sentence_transformers_resolves_embeddings_disjoint_from_chat(
        self,
    ) -> None:
        caps = self._caps("sentence-transformers/all-MiniLM-L6-v2")
        assert ModelCapability.EMBEDDINGS in caps  # the correctness widening
        assert ModelCapability.CHAT not in caps  # embedding-only, disjoint
        assert ModelCapability.TEXT_GENERATION in caps

    def test_feature_extraction_repo_resolves_embeddings(self) -> None:
        caps = self._caps("intfloat/e5-base-v2-feature-extraction")
        assert ModelCapability.EMBEDDINGS in caps

    def test_instruct_repo_resolves_text_and_chat(self) -> None:
        caps = self._caps("mistralai/Mistral-7B-Instruct-v0.2")
        assert caps == {ModelCapability.TEXT_GENERATION, ModelCapability.CHAT}

    def test_base_repo_resolves_text_only(self) -> None:
        assert self._caps("gpt2") == {ModelCapability.TEXT_GENERATION}

    def test_override_lights_up_vision(self) -> None:
        caps = self._caps(
            "llava-hf/llava-1.5-7b-hf",
            model_profile_overrides={
                "capabilities": ["text_generation", "chat", "vision"]
            },
        )
        assert ModelCapability.VISION in caps  # the non-heuristic override path

    def test_no_repo_resolves_function_calling_or_streaming(self) -> None:
        # The Inference API supports neither; the heuristic must never assert them.
        for model in (
            "gpt2",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "meta-llama/Llama-3.1-8B-Instruct",
        ):
            caps = self._caps(model)
            assert ModelCapability.FUNCTION_CALLING not in caps
            assert ModelCapability.STREAMING not in caps


class TestCapabilityHeuristicCorrections:
    """Root-cause corrections to the repo-name heuristic (deep-review 🟡s).

    Reproduce-first: the embed/reranker/disjointness cases are mis-resolved by the
    pre-correction additive heuristic (``instruct`` as a bare substring, ``bge-``
    with no reranker guard, independent embed/chat markers that could both fire).
    The two fused-name cases (``chatglm3`` / ``openchat``) guard the opposite
    edge — that correcting the embed side did not regress chat detection for real
    repos whose name fuses the ``chat`` marker into a single token.
    """

    def _caps(self, model: str) -> set[ModelCapability]:
        return set(_provider(model=model).get_capabilities())

    def test_instructor_family_is_embeddings_not_chat(self) -> None:
        # `instruct` is a *substring* of `instructor`, so under additive markers
        # the Instructor embedding family resolved CHAT and missed EMBEDDINGS.
        # The `instructor` embed token + embed-suppresses-chat ordering fixes both
        # directions: embed is resolved first, so the chat substring never runs.
        caps = self._caps("hkunlp/instructor-large")
        assert ModelCapability.EMBEDDINGS in caps
        assert ModelCapability.CHAT not in caps

    def test_e5_instruct_embedding_is_embeddings_not_chat(self) -> None:
        # An e5 embedding model whose repo name ends in `-instruct` (real
        # embedding-family naming) resolved CHAT-only. The `e5` embed marker plus
        # embed-suppresses-chat classifies it as the embedding model it is.
        caps = self._caps("intfloat/e5-mistral-7b-instruct")
        assert ModelCapability.EMBEDDINGS in caps
        assert ModelCapability.CHAT not in caps

    def test_reranker_is_not_an_embedding_model(self) -> None:
        # A cross-encoder reranker matches the `bge` family prefix but is not an
        # embedding model; the `reranker` token suppresses the embed classification.
        caps = self._caps("BAAI/bge-reranker-large")
        assert ModelCapability.EMBEDDINGS not in caps

    def test_embeddings_and_chat_are_structurally_disjoint(self) -> None:
        # A repo matching BOTH an embed marker and a chat token resolves only
        # EMBEDDINGS — disjointness is guaranteed by the logic (embed suppresses
        # chat), not merely by the tested names.
        caps = self._caps("my-org/all-minilm-chat")
        assert ModelCapability.EMBEDDINGS in caps
        assert ModelCapability.CHAT not in caps

    def test_instruct_token_still_resolves_chat(self) -> None:
        # Guard against over-correction: a genuine `instruct` *token* (whole word)
        # on a non-embedding repo still resolves CHAT.
        caps = self._caps("mistralai/Mistral-7B-Instruct-v0.2")
        assert caps == {ModelCapability.TEXT_GENERATION, ModelCapability.CHAT}

    def test_chatglm_fused_name_still_resolves_chat(self) -> None:
        # Reproduce-first: `chatglm3` fuses the `chat` marker into one token, so a
        # whole-token chat match would DROP CHAT for this real, widely-used family.
        # Substring chat matching keeps it — the marker is `chat`, not a token.
        caps = self._caps("THUDM/chatglm3-6b")
        assert caps == {ModelCapability.TEXT_GENERATION, ModelCapability.CHAT}

    def test_openchat_fused_name_still_resolves_chat(self) -> None:
        # Reproduce-first: `openchat` is a single token containing `chat`; a
        # token-boundary match would lose CHAT. Substring matching preserves it.
        caps = self._caps("openchat/openchat-3.5")
        assert caps == {ModelCapability.TEXT_GENERATION, ModelCapability.CHAT}


# ---------------------------------------------------------------------------
# W-A — request-shaping choke point parity (no-op default + override wiring)
# ---------------------------------------------------------------------------


class TestRequestShaping:
    async def test_default_parameters_are_byte_identical_to_head(self) -> None:
        session = CapturingSession()
        provider = _provider(session)
        await provider.complete("hi")
        assert session.payloads[0]["parameters"] == {
            "max_new_tokens": _HF_DEFAULT_MAX_NEW_TOKENS,
            "return_full_text": False,
        }

    async def test_caller_max_tokens_flows_through(self) -> None:
        session = CapturingSession()
        provider = _provider(session, max_tokens=512, temperature=0.5, top_p=0.9)
        await provider.complete("hi")
        params = session.payloads[0]["parameters"]
        assert params["max_new_tokens"] == 512
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9

    async def test_rejected_param_is_dropped_before_the_call(self) -> None:
        session = CapturingSession()
        provider = _provider(
            session,
            temperature=0.7,
            constraints={"rejected_params": ["temperature"]},
        )
        await provider.complete("hi")
        assert "temperature" not in session.payloads[0]["parameters"]

    async def test_param_remap_renames_the_wire_key(self) -> None:
        session = CapturingSession()
        provider = _provider(
            session,
            max_tokens=64,
            constraints={"param_remaps": {"max_new_tokens": "max_length"}},
        )
        await provider.complete("hi")
        params = session.payloads[0]["parameters"]
        assert "max_new_tokens" not in params
        assert params["max_length"] == 64


# ---------------------------------------------------------------------------
# W-A — context window (dead on HEAD → override lights up max_input_tokens)
# ---------------------------------------------------------------------------


class TestContextWindow:
    def test_override_context_window_sets_max_input_tokens(self) -> None:
        provider = _provider(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            model_profile_overrides={"context_window": 32768},
        )
        assert provider.get_constraints().max_input_tokens == 32768

    def test_absent_context_window_is_none(self) -> None:
        provider = _provider(model="mistralai/Mistral-7B-Instruct-v0.2")
        assert provider.get_constraints().max_input_tokens is None


# ---------------------------------------------------------------------------
# W-A — pricing (D-HF-PRICING: None by default, override-only, inherited)
# ---------------------------------------------------------------------------


class TestPricing:
    def test_pricing_is_none_by_default(self) -> None:
        provider = _provider(model="mistralai/Mistral-7B-Instruct-v0.2")
        assert provider.get_pricing() is None

    def test_override_lights_up_pricing(self) -> None:
        provider = _provider(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            model_profile_overrides={
                "pricing": {"input_per_mtok": 0.2, "output_per_mtok": 0.6}
            },
        )
        pricing = provider.get_pricing()
        assert pricing is not None
        assert pricing.input_per_mtok == 0.2
        assert pricing.output_per_mtok == 0.6


# ---------------------------------------------------------------------------
# W-A — validate_model (live probe by default, override pin short-circuits)
# ---------------------------------------------------------------------------


class TestValidateModel:
    async def test_installed_repo_validates_true(self) -> None:
        session = CapturingSession(get_status=200)
        provider = _provider(session)
        assert await provider.validate_model() is True
        assert session.get_calls  # the probe ran

    async def test_missing_repo_validates_false(self) -> None:
        session = CapturingSession(get_status=404)
        provider = _provider(session)
        assert await provider.validate_model() is False

    async def test_available_true_override_short_circuits_the_probe(self) -> None:
        session = CapturingSession(get_status=404)  # would fail if probed
        provider = _provider(
            session, model_profile_overrides={"available": True}
        )
        assert await provider.validate_model() is True
        assert session.get_calls == []  # no HTTP call

    async def test_available_false_override_short_circuits_the_probe(self) -> None:
        session = CapturingSession(get_status=200)  # would pass if probed
        provider = _provider(
            session, model_profile_overrides={"available": False}
        )
        assert await provider.validate_model() is False
        assert session.get_calls == []
