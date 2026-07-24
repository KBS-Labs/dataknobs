"""Tests for model-family request-shape constraints (S1).

Bug: ``AnthropicProvider`` forwards ``temperature`` (and other sampling
params) to any Anthropic model, but the Claude 5 model family rejects
``temperature`` with a hard 400. There was no model-family awareness — a
Claude-5-family bot config carrying ``temperature:`` produced a provider 400
the moment it ran.

Fix: a config-overridable ``ModelConstraints`` surface. A provider
auto-detects the family's request-shape rules (Claude 5 → rejects
``temperature``) and the base overlays any ``LLMConfig.constraints`` override,
so a consumer can declare/withdraw a rule at runtime without a dataknobs
release. The provider drops rejected params before the call — drop-and-warn,
never silently.

The end-to-end reproduce-first tests capture the kwargs the provider passes to
``messages.create`` via a minimal stand-in for ``anthropic.AsyncAnthropic``
(the sanctioned narrow case: no dataknobs testing construct produces a real
Anthropic request, and the stand-in exercises the real provider wiring end to
end). They FAIL against HEAD (``temperature`` forwarded for Claude 5) and pass
after the fix.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Self

import pytest

from dataknobs_llm.llm.base import LLMConfig, ModelConstraints
from dataknobs_llm.llm.providers import anthropic as anthropic_mod
from dataknobs_llm.llm.providers.anthropic import AnthropicProvider
from dataknobs_llm.tooling import model_limits

from test_anthropic_param_handling import make_anthropic_response


# ---------------------------------------------------------------------------
# Reusable stand-in for the Anthropic SDK client
# ---------------------------------------------------------------------------


class _ScriptedModel:
    """A minimal ``anthropic`` ``ModelInfo`` stand-in (id + max_tokens)."""

    def __init__(self, model_id: str, max_tokens: int | None) -> None:
        self.id = model_id
        self.max_tokens = max_tokens


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


@pytest.fixture(autouse=True)
def _reset_model_limits_cache() -> Any:
    """Isolate the module-level process caches between tests.

    The dynamic ceiling cache, per-loop last-fetch timestamps, refresh locks,
    and the discovered-rejected-params cache are process-global; clearing them
    before each test keeps state from leaking across tests.
    """
    for cache in (
        anthropic_mod._MODEL_LIMITS_CACHE,
        anthropic_mod._MODEL_LIMITS_LAST_FETCH,
        anthropic_mod._MODEL_LIMITS_LOCKS,
        anthropic_mod._DISCOVERED_REJECTED_PARAMS,
    ):
        cache.clear()
    yield
    for cache in (
        anthropic_mod._MODEL_LIMITS_CACHE,
        anthropic_mod._MODEL_LIMITS_LAST_FETCH,
        anthropic_mod._MODEL_LIMITS_LOCKS,
        anthropic_mod._DISCOVERED_REJECTED_PARAMS,
    ):
        cache.clear()


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


# ---------------------------------------------------------------------------
# End-to-end reproduce-first: rejected params are dropped before the API call
# ---------------------------------------------------------------------------


class TestModelFamilyParamRejection:
    """The provider drops family-rejected params before ``messages.create``."""

    async def test_temperature_dropped_for_claude_5(self) -> None:
        """Claude 5 rejects ``temperature`` → it must not reach the API."""
        provider, client = _provider_with_capture(
            "claude-sonnet-5", temperature=0.3
        )
        await provider.complete("hi")
        assert "temperature" not in client.captured_kwargs

    async def test_temperature_kept_for_claude_4_5(self) -> None:
        """Claude 4.5 still accepts ``temperature`` → forwarded unchanged."""
        provider, client = _provider_with_capture(
            "claude-haiku-4-5-20251001", temperature=0.3
        )
        await provider.complete("hi")
        assert client.captured_kwargs["temperature"] == 0.3

    async def test_temperature_kept_for_claude_opus_4_8(self) -> None:
        """Opus 4.8 is not Claude 5 → ``temperature`` forwarded."""
        provider, client = _provider_with_capture(
            "claude-opus-4-8", temperature=0.5
        )
        await provider.complete("hi")
        assert client.captured_kwargs["temperature"] == 0.5

    async def test_stream_drops_temperature_for_claude_5(self) -> None:
        """The streaming path shares the same choke point."""
        provider, _ = _provider_with_capture("claude-sonnet-5", temperature=0.3)

        captured: dict[str, Any] = {}

        class _StreamCtx:
            async def __aenter__(self) -> Self:
                return self

            async def __aexit__(self, *exc: object) -> None:
                return None

            def __aiter__(self) -> Self:
                return self

            async def __anext__(self) -> object:
                raise StopAsyncIteration

            async def get_final_message(self) -> object:
                return make_anthropic_response([{"type": "text", "text": "ok"}])

        def _stream(**kwargs: Any) -> _StreamCtx:
            captured.update(kwargs)
            return _StreamCtx()

        provider._client.messages.stream = _stream  # type: ignore[attr-defined]

        async for _ in provider.stream_complete("hi"):
            pass

        assert "temperature" not in captured

    def test_drop_emits_warning(self, caplog) -> None:
        """A dropped param must be logged, never silent."""
        provider = AnthropicProvider(
            LLMConfig(provider="anthropic", model="claude-sonnet-5", temperature=0.3)
        )
        with caplog.at_level(logging.WARNING):
            params = provider._build_api_kwargs(provider.config)
        assert "temperature" not in params
        assert any(
            "temperature" in rec.message and "claude-sonnet-5" in rec.message
            for rec in caplog.records
        )

    def test_no_warning_when_nothing_dropped(self, caplog) -> None:
        """No spurious warning for a family with no rejected params."""
        provider = AnthropicProvider(
            LLMConfig(
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                temperature=0.3,
            )
        )
        with caplog.at_level(logging.WARNING):
            params = provider._build_api_kwargs(provider.config)
        assert params["temperature"] == 0.3
        assert not caplog.records


# ---------------------------------------------------------------------------
# S1 surface: ModelConstraints detection + config override
# ---------------------------------------------------------------------------


class TestModelConstraintsResolution:
    """The resolved ``ModelConstraints`` surface and its config override."""

    def test_claude_5_detected_rejects_temperature(self) -> None:
        provider = AnthropicProvider(
            LLMConfig(provider="anthropic", model="claude-sonnet-5")
        )
        constraints = provider.get_constraints()
        assert "temperature" in constraints.rejected_params

    def test_claude_4_5_detected_rejects_nothing(self) -> None:
        provider = AnthropicProvider(
            LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
        )
        constraints = provider.get_constraints()
        assert constraints.rejected_params == frozenset()

    def test_anthropic_never_accepts_inline_system(self) -> None:
        """Every Anthropic model hoists system messages (read by #187)."""
        for model in ("claude-sonnet-5", "claude-haiku-4-5-20251001", "claude-3-opus"):
            provider = AnthropicProvider(
                LLMConfig(provider="anthropic", model=model)
            )
            assert provider.get_constraints().accepts_inline_system is False

    def test_config_override_adds_rejected_param(self) -> None:
        """A consumer can declare an extra rejected param without a release."""
        provider = AnthropicProvider(
            LLMConfig(
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                constraints={"rejected_params": ["top_p"]},
            )
        )
        constraints = provider.get_constraints()
        assert constraints.rejected_params == frozenset({"top_p"})

    def test_config_override_withdraws_stale_rule(self) -> None:
        """Passing an empty list withdraws the auto-detected rejection."""
        provider = AnthropicProvider(
            LLMConfig(
                provider="anthropic",
                model="claude-sonnet-5",
                constraints={"rejected_params": []},
            )
        )
        constraints = provider.get_constraints()
        assert constraints.rejected_params == frozenset()

    async def test_config_override_withdrawal_forwards_param(self) -> None:
        """Withdrawing the rule means the param reaches the API again."""
        provider, client = _provider_with_capture(
            "claude-sonnet-5",
            temperature=0.3,
            constraints={"rejected_params": []},
        )
        await provider.complete("hi")
        assert client.captured_kwargs["temperature"] == 0.3


class TestPerCallModelOverride:
    """Constraints resolve from the per-call runtime config, not self.config.

    The base config pins a Claude 4.x model (which accepts ``temperature``),
    but a call overrides the model to a Claude 5 model (which rejects it). The
    drop must reflect the model **actually being sent** — otherwise the request
    carries ``temperature`` and pays a wasted 400 round-trip. Reproduce-first:
    FAILS before the fix (constraints read ``self.config`` → Claude 4.x → keeps
    ``temperature``), passes after.
    """

    async def test_override_to_claude_5_drops_temperature_up_front(self) -> None:
        provider, client = _provider_with_capture(
            "claude-haiku-4-5-20251001", temperature=0.3
        )
        await provider.complete(
            "hi", config_overrides={"model": "claude-sonnet-5"}
        )
        assert client.captured_kwargs.get("model") == "claude-sonnet-5"
        assert "temperature" not in client.captured_kwargs

    def test_get_constraints_honors_passed_config(self) -> None:
        """``get_constraints(config)`` resolves the passed config's family."""
        provider = AnthropicProvider(
            LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
        )
        # Default (self.config) → Claude 4.5 rejects nothing.
        assert provider.get_constraints().rejected_params == frozenset()
        # Per-call config for a Claude 5 model → rejects temperature.
        claude5 = LLMConfig(provider="anthropic", model="claude-opus-5")
        assert "temperature" in provider.get_constraints(claude5).rejected_params


class TestMaxTokensCeilingClamp:
    """``max_tokens`` is clamped down to the family ceiling — clamp-and-warn.

    Substrate S1 declared ``max_tokens_ceiling`` but left it inert (no read
    site). The clamp wires it in at the same ``_build_api_kwargs`` choke point
    as the rejected-param drop: when a request asks for more output tokens than
    the family grants, reduce to the ceiling and warn (never silent). Clamping
    *down* is always a valid request (asking for fewer tokens never 400s), so
    unlike the rejected-param path this needs no retry net.

    The clamp is proven via a **config override** (``max_tokens_ceiling`` set on
    ``LLMConfig.constraints``), independent of the dynamically/resource-resolved
    ceiling that could go stale — the override always wins. Reproduce-first:
    the discriminating test FAILS against HEAD (no clamp → ``max_tokens`` stays
    at the requested value).
    """

    def test_clamp_via_config_override(self, caplog) -> None:
        """A ceiling below the request clamps ``max_tokens`` and warns.

        Discriminating reproduce-first test — FAILS on HEAD (max_tokens stays
        500), passes after the clamp lands. Seed-independent: the ceiling comes
        from the config override, not the seed table.
        """
        provider = AnthropicProvider(
            LLMConfig(
                provider="anthropic",
                model="claude-sonnet-5",
                max_tokens=500,
                constraints={"max_tokens_ceiling": 100},
            )
        )
        with caplog.at_level(logging.WARNING):
            params = provider._build_api_kwargs(provider.config)
        assert params["max_tokens"] == 100
        assert any(
            "max_tokens" in rec.message and "claude-sonnet-5" in rec.message
            for rec in caplog.records
        )

    def test_no_clamp_when_under_ceiling(self, caplog) -> None:
        """A request at or below the ceiling passes through, no warning."""
        provider = AnthropicProvider(
            LLMConfig(
                provider="anthropic",
                model="claude-sonnet-5",
                max_tokens=100,
                constraints={"max_tokens_ceiling": 4096},
            )
        )
        with caplog.at_level(logging.WARNING):
            params = provider._build_api_kwargs(provider.config)
        assert params["max_tokens"] == 100
        assert not caplog.records

    def test_no_clamp_when_no_ceiling(self, caplog) -> None:
        """No ceiling (unresolved family, none overridden) → unchanged, no warning.

        Guards the "overwhelming majority of requests are byte-identical"
        backward-compat claim: with no resolvable family ceiling (a model absent
        from both the dynamic cache and the fallback resource), ``max_tokens``
        flows through untouched.
        """
        provider = AnthropicProvider(
            LLMConfig(
                provider="anthropic",
                model="claude-unseeded-model",
                max_tokens=100_000,
            )
        )
        assert provider.get_constraints().max_tokens_ceiling is None
        with caplog.at_level(logging.WARNING):
            params = provider._build_api_kwargs(provider.config)
        assert params["max_tokens"] == 100_000
        assert not caplog.records

    async def test_clamp_applied_in_live_request_path(self) -> None:
        """The clamp fires end-to-end through ``complete`` → ``messages.create``.

        Proves the clamp sits in the real request path (not only reachable by
        calling ``_build_api_kwargs`` directly), mirroring the rejected-param
        drop's end-to-end coverage.
        """
        provider, client = _provider_with_capture(
            "claude-sonnet-5",
            max_tokens=500,
            constraints={"max_tokens_ceiling": 100},
        )
        await provider.complete("hi")
        assert client.captured_kwargs["max_tokens"] == 100

    def test_per_call_override_clamps_to_runtime_ceiling(self) -> None:
        """The clamp reads the per-call runtime config, not ``self.config``.

        Base config has no ceiling; a per-call ``constraints`` override supplies
        one below the requested ``max_tokens``. The clamp must reflect the
        config actually being sent.
        """
        provider = AnthropicProvider(
            LLMConfig(
                provider="anthropic",
                model="claude-sonnet-5",
                max_tokens=500,
            )
        )
        runtime = provider.config.clone(
            constraints={"max_tokens_ceiling": 100}
        )
        params = provider._build_api_kwargs(runtime)
        assert params["max_tokens"] == 100
        # self.config (no ceiling) is unaffected → passes through.
        assert provider._build_api_kwargs(provider.config)["max_tokens"] == 500

    def test_detect_constraints_permissive_by_default(self) -> None:
        """Resolution is permissive (``None``) for an unknown model.

        A model absent from both the dynamic cache and the fallback resource
        resolves to ``None`` → no clamp, identical to pre-clamp behavior. This
        documents the safe default for an unrecognized model.
        """
        provider = AnthropicProvider(
            LLMConfig(provider="anthropic", model="some-unknown-model")
        )
        assert provider.get_constraints().max_tokens_ceiling is None


class TestModelConstraintsDataclass:
    """Unit tests for the ``ModelConstraints`` value type."""

    def test_defaults_are_permissive(self) -> None:
        c = ModelConstraints()
        assert c.rejected_params == frozenset()
        assert c.accepts_inline_system is True
        assert c.max_tokens_ceiling is None

    def test_with_overrides_is_pure(self) -> None:
        base = ModelConstraints(rejected_params=frozenset({"temperature"}))
        overridden = base.with_overrides({"rejected_params": ["top_p"]})
        # Original is unchanged (frozen, pure overlay).
        assert base.rejected_params == frozenset({"temperature"})
        assert overridden.rejected_params == frozenset({"top_p"})

    def test_with_overrides_none_rejected_params_clears(self) -> None:
        base = ModelConstraints(rejected_params=frozenset({"temperature"}))
        assert base.with_overrides(
            {"rejected_params": None}
        ).rejected_params == frozenset()

    def test_with_overrides_absent_key_preserved(self) -> None:
        base = ModelConstraints(
            rejected_params=frozenset({"temperature"}),
            accepts_inline_system=False,
        )
        overridden = base.with_overrides({"max_tokens_ceiling": 8192})
        assert overridden.rejected_params == frozenset({"temperature"})
        assert overridden.accepts_inline_system is False
        assert overridden.max_tokens_ceiling == 8192


class TestDynamicMaxTokensResolution:
    """The ``max_tokens`` ceiling resolves from the live Models API, cached.

    Precedence per model: config override → dynamic (live Models API,
    cached, TTL-refreshed) → bundled fallback resource → ``None``. These
    reproduce-first tests drive the real ``AnthropicProvider`` and script the
    Models-API ``list()`` via the sanctioned SDK stand-in. Each FAILS against
    the pre-FU7 state (no dynamic path; the fallback table shipped empty):
    ``refresh_model_limits`` did not exist and the ceiling resolved to ``None``.
    """

    async def test_dynamic_value_overrides_resource(self) -> None:
        """A live Models-API value wins over the bundled resource value.

        The resource seeds ``claude-sonnet-5`` at 128000; a live 200000 must
        take precedence once fetched.
        """
        provider, client = _provider_with_capture("claude-sonnet-5")
        client.models.models = [_ScriptedModel("claude-sonnet-5", 200000)]
        await provider.refresh_model_limits()
        assert provider.get_constraints().max_tokens_ceiling == 200000

    async def test_resource_used_when_dynamic_fails(self) -> None:
        """A failed refresh falls back to the bundled resource value."""
        provider, client = _provider_with_capture("claude-sonnet-5")
        client.models.raise_on_list = True
        await provider.refresh_model_limits()  # swallows the error
        assert provider.get_constraints().max_tokens_ceiling == 128000

    async def test_absent_model_is_permissive(self) -> None:
        """A model in neither the cache nor the resource resolves to ``None``."""
        provider, client = _provider_with_capture("mystery-model-x")
        client.models.raise_on_list = True
        await provider.refresh_model_limits()
        assert provider.get_constraints().max_tokens_ceiling is None

    async def test_dynamic_disabled_uses_resource_without_api_call(self) -> None:
        """``model_limits_dynamic=False`` → resource-only, no Models-API call."""
        provider, client = _provider_with_capture(
            "claude-sonnet-5", options={"model_limits_dynamic": False}
        )
        await provider.complete("hi")
        assert client.models.list_calls == 0
        assert provider.get_constraints().max_tokens_ceiling == 128000

    async def test_ttl_zero_refreshes_each_call(self) -> None:
        """TTL≈0 → each request re-polls the Models API."""
        provider, client = _provider_with_capture(
            "claude-sonnet-5", options={"model_limits_ttl": 0}
        )
        client.models.models = [_ScriptedModel("claude-sonnet-5", 200000)]
        await provider.complete("hi")
        await provider.complete("hi")
        assert client.models.list_calls == 2

    async def test_long_ttl_refreshes_once(self) -> None:
        """A long TTL → one poll shared across requests (bounded, not per-call)."""
        provider, client = _provider_with_capture(
            "claude-sonnet-5", options={"model_limits_ttl": 3600}
        )
        client.models.models = [_ScriptedModel("claude-sonnet-5", 200000)]
        await provider.complete("hi")
        await provider.complete("hi")
        assert client.models.list_calls == 1

    async def test_dynamic_value_not_degraded_on_failed_refresh(self) -> None:
        """A known-good dynamic value survives a later failed refresh.

        Source-aware non-degradation: a transient Models-API failure must not
        drop a cached dynamic value back to the (possibly rounded-down)
        resource.
        """
        provider, client = _provider_with_capture("claude-sonnet-5")
        client.models.models = [_ScriptedModel("claude-sonnet-5", 200000)]
        await provider.refresh_model_limits()
        assert provider.get_constraints().max_tokens_ceiling == 200000
        # A subsequent forced refresh fails — the dynamic value must persist.
        client.models.raise_on_list = True
        await provider.refresh_model_limits()
        assert provider.get_constraints().max_tokens_ceiling == 200000

    async def test_concurrent_requests_dedup_refresh(self) -> None:
        """N concurrent requests on a cold cache coalesce into one poll."""
        provider, client = _provider_with_capture("claude-sonnet-5")
        client.models.models = [_ScriptedModel("claude-sonnet-5", 200000)]
        await asyncio.gather(*[provider.complete("hi") for _ in range(8)])
        assert client.models.list_calls == 1

    async def test_clamp_uses_dynamic_ceiling_end_to_end(self) -> None:
        """The clamp fires against a dynamically-sourced ceiling, end to end.

        Discriminating reproduce-first test — over-ceiling ``max_tokens`` clamps
        to the live value through ``complete`` → ``messages.create``. FAILS on
        pre-FU7 HEAD (no dynamic ceiling → no clamp → ``max_tokens`` stays 500000).
        """
        provider, client = _provider_with_capture(
            "claude-sonnet-5", max_tokens=500_000
        )
        client.models.models = [_ScriptedModel("claude-sonnet-5", 128000)]
        await provider.complete("hi")
        assert client.captured_kwargs["max_tokens"] == 128000

    def test_resource_ships_and_loads(self) -> None:
        """The bundled fallback resource is importable and non-empty.

        Guards package-data inclusion via ``importlib.resources`` — a
        missing-from-package regression makes this fail.
        """
        limits = anthropic_mod._load_model_limits_resource()
        assert limits
        assert "claude-opus-4-8" in limits


class TestModelLimitsTooling:
    """The ``--check``/``--update`` reconciliation tool (no network).

    Driven with the sanctioned SDK stand-in and a temp resource path — no live
    API. The key-gated live drift test lives separately (skips without a key).
    """

    def test_check_without_key_is_noop(self, monkeypatch, capsys) -> None:
        """Keyless ``--check`` is a clean no-op (exit 0), never a failure."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert model_limits.main(["--check"]) == 0
        assert "skipped" in capsys.readouterr().out

    async def test_fetch_live_limits_skips_models_without_max_tokens(self) -> None:
        """Live fetch collects ``max_tokens`` and skips models lacking it."""
        client = _CaptureAnthropicClient()
        client.models.models = [
            _ScriptedModel("claude-a", 100),
            _ScriptedModel("claude-b", None),
        ]
        limits = await model_limits.fetch_live_limits(client)
        assert limits == {"claude-a": 100}

    def test_diff_detects_drift(self) -> None:
        assert model_limits.diff_limits({"a": 100}, {"a": 200}) == [("a", 100, 200)]
        assert model_limits.diff_limits({"a": 100}, {"a": 100}) == []

    def test_check_exits_nonzero_on_drift(self, tmp_path) -> None:
        path = tmp_path / "limits.yaml"
        path.write_text("models:\n  claude-a: 999\n", encoding="utf-8")
        client = _CaptureAnthropicClient()
        client.models.models = [_ScriptedModel("claude-a", 100)]
        assert model_limits.main(["--check"], client=client, resource_path=path) == 1

    def test_check_zero_when_matching(self, tmp_path) -> None:
        path = tmp_path / "limits.yaml"
        path.write_text("models:\n  claude-a: 100\n", encoding="utf-8")
        client = _CaptureAnthropicClient()
        client.models.models = [_ScriptedModel("claude-a", 100)]
        assert model_limits.main(["--check"], client=client, resource_path=path) == 0

    def test_update_rewrites_resource_from_live(self, tmp_path) -> None:
        path = tmp_path / "limits.yaml"
        client = _CaptureAnthropicClient()
        client.models.models = [
            _ScriptedModel("claude-a", 100),
            _ScriptedModel("claude-z", 200),
        ]
        rc = model_limits.main(
            ["--update"],
            client=client,
            resource_path=path,
            verified_date="2026-07-23",
        )
        assert rc == 0
        assert model_limits.load_resource_limits(path) == {
            "claude-a": 100,
            "claude-z": 200,
        }
        assert "2026-07-23" in path.read_text(encoding="utf-8")

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="requires ANTHROPIC_API_KEY for a live Models-API drift check",
    )
    def test_bundled_resource_matches_live_api(self) -> None:
        """Key-gated drift alarm: the bundled resource matches live values.

        Skips without a key (normal / keyless CI). On a keyed nightly/local run
        it fails if the bundled resource has drifted from the live Models API —
        run ``bin/update-model-limits.sh --update`` to refresh it.
        """
        assert model_limits.main(["--check"]) == 0
