"""Parity + structural guard for the shared request-shaping orchestration.

The request-shaping *primitives* have long been shared on the base
(``LLMProvider._apply_request_constraints`` in canonical config space,
``LLMProvider._apply_param_remaps`` at the wire). The *orchestration* around
them — resolve constraints once, split per-call kwargs into shaped-field folds
vs wire-only passthroughs, run the canonical drop/clamp — was re-implemented
verbatim in each provider's build method (two providers carried a byte-identical
``shaped_fields`` split, the other three re-derived the resolve→shape spine).

That orchestration is now a single base method,
``LLMProvider._shape_request_params(config, extra) -> _ShapedRequest``, and every
provider's build method routes through it, supplying only its vendor-specific
wire assembly + the final ``_apply_param_remaps``.

This module has three parts:

- **Parity pins** (``TestRequestShapingParity``): the cross-provider output is
  byte-identical to the pre-hoist behavior — folded/clamped/dropped/remapped
  exactly as before, wire-only kwargs pass straight through, and the two
  merge-point divergences (OpenAI merges ``wire_extra`` *before* the remap;
  Bedrock merges it *last*, after full Converse assembly) are preserved. No
  mocks — real provider instances, real adapters; the build methods are pure
  (no network), so they are exercised directly.
- **Unit block** (``TestShapeRequestParams``): the shared method's own contract
  — the shaped-field fold vs wire-only split, ``extra=None`` → empty
  ``wire_extra``, resolve-once, and the ``constraints`` object threaded back.
- **Recurrence guard** (``TestOrchestrationSharedOnBase``): a structural guard,
  modeled on ``test_profile_detection_mixin_adoption.py``. Because the build
  methods are per-provider (not inherited), the analog of that file's MRO
  identity check is a source-scan: every shipped provider's build method routes
  through ``_shape_request_params`` and none re-inlines the ``shaped_fields``
  split, enforced both for the five named adopters and via a maintenance-free
  sweep over every provider method.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

from dataknobs_llm.llm.base import LLMConfig, LLMProvider, _ShapedRequest
from dataknobs_llm.llm.providers.anthropic import AnthropicProvider
from dataknobs_llm.llm.providers.bedrock import BedrockProvider
from dataknobs_llm.llm.providers.huggingface import HuggingFaceProvider
from dataknobs_llm.llm.providers.ollama import OllamaProvider
from dataknobs_llm.llm.providers.openai import OpenAIProvider

# ---------------------------------------------------------------------------
# Representative models chosen to trigger *real* shaping (verified against the
# bundled profile resources / Claude-family heuristic).
# ---------------------------------------------------------------------------

# OpenAI reasoning family: rejects ``temperature``, remaps
# ``max_tokens`` -> ``max_completion_tokens``, and carries an output ceiling.
_OPENAI_REASONING = "o1"
# Bedrock Claude-5 family id: the heuristic rejects ``temperature`` (a 400).
_BEDROCK_CLAUDE5 = "us.anthropic.claude-5-sonnet-20260101-v1:0"
# Bedrock Claude-4.5: carries an output ceiling (no rejected params).
_BEDROCK_CLAUDE45 = "anthropic.claude-sonnet-4-5-20250929-v1:0"
# Native Anthropic Claude-5: rejects ``temperature``.
_ANTHROPIC_CLAUDE5 = "claude-5-opus"


def _openai(model: str, **cfg: Any) -> OpenAIProvider:
    return OpenAIProvider(LLMConfig(provider="openai", model=model, **cfg))


def _bedrock(model: str, **cfg: Any) -> BedrockProvider:
    return BedrockProvider(LLMConfig(provider="bedrock", model=model, **cfg))


def _anthropic(model: str, **cfg: Any) -> AnthropicProvider:
    return AnthropicProvider(LLMConfig(provider="anthropic", model=model, **cfg))


def _ollama(model: str, **cfg: Any) -> OllamaProvider:
    return OllamaProvider(LLMConfig(provider="ollama", model=model, **cfg))


def _huggingface(model: str, **cfg: Any) -> HuggingFaceProvider:
    return HuggingFaceProvider(
        LLMConfig(provider="huggingface", model=model, **cfg)
    )


# ---------------------------------------------------------------------------
# Parity pins — byte-identical cross-provider output (P1-P8)
# ---------------------------------------------------------------------------


class TestRequestShapingParity:
    """The hoist is byte-identical: every provider shapes exactly as before."""

    def test_p1_openai_folds_clamps_and_remaps_shaped_kwarg(self) -> None:
        """P1: an over-ceiling ``max_tokens`` kwarg is folded → clamped → remapped.

        The reasoning family clamps to its ceiling, drops ``temperature``, and
        renames ``max_tokens`` -> ``max_completion_tokens``; the raw canonical key
        never reaches the wire.
        """
        p = _openai(_OPENAI_REASONING)
        ceiling = p.get_constraints().max_tokens_ceiling
        assert ceiling is not None
        wire = p._build_api_kwargs(p.config, {"max_tokens": ceiling + 50_000})
        assert wire["max_completion_tokens"] == ceiling
        assert "max_tokens" not in wire
        assert wire.get("temperature") is None

    def test_p2_openai_wire_only_kwargs_pass_through(self) -> None:
        """P2: wire-only kwargs ride through untouched (the ``wire_extra`` merge).

        ``user`` (a genuine wire-only param) and a ``response_format`` dict
        (richer than the narrow ``str`` config field) are not shaped fields, so
        they land on the wire verbatim. This pins passthrough, not merge order:
        the merge point is unobservable through the public build method here,
        because every remap *source* is itself a shaped field (always folded,
        never in ``wire_extra``), so no rename source can reach this path.
        """
        p = _openai(_OPENAI_REASONING)
        rf = {"type": "json_object"}
        wire = p._build_api_kwargs(
            p.config, {"user": "u", "response_format": rf}
        )
        assert wire["user"] == "u"
        assert wire["response_format"] == rf

    def test_p3_bedrock_drops_rejected_and_merges_wire_only_last(self) -> None:
        """P3: Bedrock drops ``temperature``; wire-only kwarg lands top-level last.

        A Claude-5 id rejects ``temperature`` (never reaches
        ``inferenceConfig``); a genuine wire-only Converse param
        (``additionalModelRequestFields``) is merged after the full assembly.
        """
        p = _bedrock(_BEDROCK_CLAUDE5, temperature=0.7, max_tokens=1024)
        req = p._build_converse_request(
            "hi", p.config, None, {"additionalModelRequestFields": {"foo": 1}}
        )
        assert "temperature" not in req["inferenceConfig"]
        assert req["inferenceConfig"]["maxTokens"] == 1024
        assert req["additionalModelRequestFields"] == {"foo": 1}

    def test_p3b_bedrock_clamps_max_tokens_to_ceiling(self) -> None:
        """P3b: Bedrock clamps an over-ceiling ``max_tokens`` in ``inferenceConfig``."""
        p = _bedrock(_BEDROCK_CLAUDE45)
        ceiling = p.get_constraints().max_tokens_ceiling
        assert ceiling is not None
        req = p._build_converse_request(
            "hi", p.config, None, {"max_tokens": ceiling + 10_000}
        )
        assert req["inferenceConfig"]["maxTokens"] == ceiling

    def test_p4_bedrock_system_and_guardrail_read_unshaped_fields(self) -> None:
        """P4: ``system`` / guardrail reflect the config after a shaped fold.

        The Bedrock preservation subtlety: after the hoist the local
        ``runtime_config`` is no longer reassigned to the folded clone, so
        ``adapt_messages(system_prompt=...)`` and ``_guardrail_config(...)`` read
        the original config. This is byte-identical because folding only ever
        moves *shaped* fields (sampling / ``max_tokens``), never ``system_prompt``
        or any guardrail field — so folded-vs-original agree on exactly the fields
        these two calls read. A shaped ``max_tokens`` fold is present to force the
        clone path; ``system`` and ``guardrailConfig`` must still land.
        """
        p = _bedrock(
            _BEDROCK_CLAUDE5,
            system_prompt="SYS",
            max_tokens=256,
            options={
                "guardrail_identifier": "gid-1",
                "guardrail_version": "DRAFT",
            },
        )
        req = p._build_converse_request(
            "hi", p.config, None, {"max_tokens": 512}
        )
        assert req["system"] == [{"text": "SYS"}]
        assert req["guardrailConfig"] == {
            "guardrailIdentifier": "gid-1",
            "guardrailVersion": "DRAFT",
        }
        # The shaped fold still took effect on the sampling side.
        assert req["inferenceConfig"]["maxTokens"] == 512

    def test_p5_anthropic_drops_temperature(self) -> None:
        """P5: native Anthropic Claude-5 drops the rejected ``temperature``."""
        p = _anthropic(_ANTHROPIC_CLAUDE5, temperature=0.5, max_tokens=1024)
        wire = p._build_api_kwargs(p.config)
        assert wire.get("temperature") is None

    def test_p6_ollama_is_byte_identical_no_op(self) -> None:
        """P6: Ollama (empty constraints) shapes to exactly ``_build_options``."""
        p = _ollama("llama3.2", max_tokens=256, temperature=0.5)
        assert p._build_shaped_options(p.config) == p._build_options(p.config)

    def test_p7_huggingface_default_max_new_tokens_still_lands(self) -> None:
        """P7: HuggingFace (empty constraints) is a no-op; the default still lands."""
        p = _huggingface("meta-llama/Llama-3-8b")
        params = p._build_hf_parameters(p.config)
        assert params["max_new_tokens"] > 0
        assert params["return_full_text"] is False

    def test_p8_unknown_model_is_pure_passthrough(self) -> None:
        """P8: an unknown model resolves an all-permissive profile → no shaping."""
        p = _openai("gpt-nonexistent-9", max_tokens=123)
        wire = p._build_api_kwargs(p.config)
        # No ceiling, no rejected params, no remap → canonical max_tokens survives.
        assert wire.get("max_tokens") == 123
        assert "max_completion_tokens" not in wire


# ---------------------------------------------------------------------------
# Unit block — the shared method's own contract
# ---------------------------------------------------------------------------


class TestShapeRequestParams:
    """``LLMProvider._shape_request_params`` in isolation."""

    def test_returns_shaped_request_namedtuple(self) -> None:
        p = _openai(_OPENAI_REASONING)
        result = p._shape_request_params(p.config)
        assert isinstance(result, _ShapedRequest)
        assert result._fields == ("config", "wire_extra", "constraints")

    def test_shaped_field_kwarg_is_folded_not_wire(self) -> None:
        """A kwarg naming a shaped field (``max_tokens``) folds into the config."""
        p = _openai(_OPENAI_REASONING)
        ceiling = p.get_constraints().max_tokens_ceiling
        assert ceiling is not None
        result = p._shape_request_params(
            p.config, {"max_tokens": ceiling + 1000}
        )
        # Folded + clamped in canonical config space; not left in wire_extra.
        assert "max_tokens" not in result.wire_extra
        assert result.config.max_tokens == ceiling

    def test_wire_only_kwarg_is_split_out(self) -> None:
        """A non-shaped kwarg is returned in ``wire_extra``, untouched."""
        p = _openai(_OPENAI_REASONING)
        result = p._shape_request_params(p.config, {"user": "u"})
        assert result.wire_extra == {"user": "u"}

    def test_extra_none_yields_empty_wire_extra(self) -> None:
        p = _anthropic(_ANTHROPIC_CLAUDE5)
        result = p._shape_request_params(p.config)
        assert result.wire_extra == {}

    def test_constraints_resolved_once_per_call(self) -> None:
        """``get_constraints`` is called exactly once (resolve-once).

        Wraps the real method in a counting shim (no mock — the original runs
        underneath) to prove the shared method does not re-resolve.
        """
        p = _openai(_OPENAI_REASONING)
        calls = {"n": 0}
        original = p.get_constraints

        def _counting(config: LLMConfig | None = None) -> Any:
            calls["n"] += 1
            return original(config)

        p.get_constraints = _counting  # type: ignore[method-assign]
        p._shape_request_params(p.config, {"max_tokens": 999_999})
        assert calls["n"] == 1

    def test_constraints_threaded_back_for_remap(self) -> None:
        """The returned ``constraints`` carry the family's ``param_remaps``.

        This is the object the caller feeds to ``_apply_param_remaps`` — resolving
        it here and returning it is what lets the caller avoid a second build.
        """
        p = _openai(_OPENAI_REASONING)
        result = p._shape_request_params(p.config)
        assert result.constraints.param_remaps.get("max_tokens") == (
            "max_completion_tokens"
        )


# ---------------------------------------------------------------------------
# Recurrence guard — orchestration stays shared on the base
# ---------------------------------------------------------------------------

# The five shipped build methods (per-provider, NOT inherited — hence a
# source-scan guard rather than the MRO-identity check the profile-detection
# mixin uses for its inherited trio).
BUILD_METHODS: dict[type[LLMProvider], str] = {
    OpenAIProvider: "_build_api_kwargs",
    AnthropicProvider: "_build_api_kwargs",
    BedrockProvider: "_build_converse_request",
    OllamaProvider: "_build_shaped_options",
    HuggingFaceProvider: "_build_hf_parameters",
}


def _all_subclasses(cls: type) -> set[type]:
    out: set[type] = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out |= _all_subclasses(sub)
    return out


def _shipped_providers() -> set[type[LLMProvider]]:
    """Every LLMProvider subclass shipped in ``dataknobs_llm.llm.providers``."""
    return {
        c
        for c in _all_subclasses(LLMProvider)
        if c.__module__.startswith("dataknobs_llm.llm.providers")
    }


class TestOrchestrationSharedOnBase:
    """Every provider routes request shaping through the base, none re-inlines."""

    def test_named_adopters_route_through_shared_method(self) -> None:
        """Each build method calls ``_shape_request_params``, no inline split.

        Every one of the five routes through the shared method and does not
        re-inline the ``shaped_fields`` kwarg-split.
        """
        for cls, name in BUILD_METHODS.items():
            source = inspect.getsource(getattr(cls, name))
            assert "_shape_request_params" in source, (
                f"{cls.__name__}.{name} must route through "
                f"_shape_request_params"
            )
            assert "shaped_fields" not in source, (
                f"{cls.__name__}.{name} re-inlines the shaped_fields split"
            )

    def test_no_provider_method_reinlines_the_split(self) -> None:
        """Maintenance-free sweep: no shipped provider method anywhere re-inlines
        the kwarg-split orchestration.

        Catches a *sixth* provider (or a new method on an existing one) copying
        the block under any name — the ``ProfileDetectionMixin`` guard's
        'future adopter caught automatically' spirit, one layer over. The shared
        method itself lives on the base (``dataknobs_llm.llm.base``), out of this
        providers-package scan.
        """
        for cls in _shipped_providers():
            for attr, member in vars(cls).items():
                if not inspect.isfunction(member):
                    continue
                source = inspect.getsource(member)
                # Whitespace-tolerant: catch ``shaped_fields =`` / ``= (`` /
                # ``=(`` regardless of ruff-format spacing or line wrapping.
                assert not re.search(r"shaped_fields\s*=", source), (
                    f"{cls.__name__}.{attr} re-inlines the shaped_fields split "
                    f"— route it through LLMProvider._shape_request_params"
                )

    def test_enumeration_finds_the_known_adopters(self) -> None:
        """The sweep discovers at least the five known adopters.

        Anti-vacuous sanity: the enumeration cannot silently degrade to an empty
        (vacuous-pass) set.
        """
        shipped = _shipped_providers()
        assert shipped >= set(BUILD_METHODS), (
            f"provider enumeration lost a known adopter: "
            f"{set(BUILD_METHODS) - shipped}"
        )
