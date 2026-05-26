"""Tests for ``DynaBotConfig.reasoning`` polymorphic-section validation.

A parsed ``DynaBotConfig`` can be validated (without constructing the bot)
so a malformed or unknown ``reasoning`` section is caught at config-lint
time. The ``reasoning`` resolver delegates to the reasoning strategy
registry, reading ``CONFIG_CLS`` off the registered strategy class for the
``strategy`` discriminator, so all five built-in strategies resolve to a
real ``StructuredConfig`` — the skip sentinel is reached only by a 3rd-party
bare-callable strategy.

Because the grounded/hybrid strategy configs carry nested sub-config trees,
one ``DynaBotConfig.from_dict(raw).validate()`` descends into them (the
base's recursion through dry-run-built children). The reasoning configs are
deliberately lenient (no field-value validators that raise), so the deep
surface available today is a wizard section missing its required
``wizard_config`` — that raises from the dry-run build.

These config-level tests construct the config dataclass directly — a
legitimate use (they test config internals, not bot flows), so no
``BotTestHarness`` is needed.

The registry modules are imported for their side effect (registering the
``reasoning`` resolver, eager on import); ``dataknobs-data`` is imported so
the nested ``vector_store`` resolver is registered too, and ``dataknobs_llm``
so the ``llm`` resolver is registered.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

# Required side-effect imports: registering the nested resolvers in
# config_registries that validate() resolves the sections against. Do NOT
# remove as "unused" — without them the bindings are unregistered and
# validate() degrades to a no-op skip. (The ``reasoning`` resolver is
# registered by importing ``dataknobs_bots.reasoning.registry``, already loaded
# via the ``get_registry`` import below, so it needs no explicit line here.)
import dataknobs_data.vector.stores  # noqa: F401
import dataknobs_llm  # noqa: F401
from dataknobs_bots.bot.config import DynaBotConfig
from dataknobs_bots.reasoning.registry import get_registry
from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.structured_config import config_registries
from dataknobs_common.testing import assert_polymorphic_bindings_resolve

# --- binding resolves (wiring guard) ---


def test_reasoning_binding_registered() -> None:
    # The resolver registers eagerly on import; this fails in CI if the
    # registration is ever dropped or renamed.
    assert config_registries.has("reasoning")


def test_dynabot_reasoning_binding_resolves_guard() -> None:
    # DynaBotConfig declares the ``reasoning`` binding (alongside llm / memory /
    # knowledge_base); this fails if its resolver registration drops.
    assert_polymorphic_bindings_resolve(DynaBotConfig)


# --- happy paths per strategy ---


@pytest.mark.parametrize(
    "section",
    [
        {"strategy": "simple"},
        {"strategy": "react", "max_iterations": 3, "verbose": True},
        {
            "strategy": "grounded",
            "intent": {"mode": "extract", "num_queries": 3},
            "retrieval": {"top_k": 5},
            "synthesis": {"style": "conversational"},
        },
        {
            "strategy": "hybrid",
            "grounded": {"intent": {"mode": "extract"}, "retrieval": {"top_k": 4}},
            "react": {"max_iterations": 4},
            "store_provenance": True,
        },
        {"strategy": "wizard", "wizard_config": "wizards/onboarding.yaml"},
    ],
    ids=["simple", "react", "grounded", "hybrid", "wizard"],
)
def test_good_reasoning_section_validates(section: dict) -> None:
    DynaBotConfig.from_dict({"reasoning": section}).validate()


def test_default_strategy_validates() -> None:
    # No ``strategy`` key -> registry default ``simple``.
    DynaBotConfig.from_dict({"reasoning": {}}).validate()


def test_absent_reasoning_is_noop() -> None:
    # reasoning defaults to None; an empty/absent section is a clean no-op.
    DynaBotConfig.from_dict({}).validate()
    DynaBotConfig.from_dict({"reasoning": None}).validate()


# --- unknown discriminator -> ConfigurationError ---


def test_unknown_strategy_raises() -> None:
    cfg = DynaBotConfig.from_dict({"reasoning": {"strategy": "bogus_strategy"}})
    with pytest.raises(ConfigurationError) as exc:
        cfg.validate()
    msg = str(exc.value)
    assert "reasoning" in msg
    assert "DynaBotConfig" in msg


# --- recursion: a malformed deep field surfaces from one top-level validate() ---


def test_wizard_missing_required_config_raises_via_recursion() -> None:
    # A single DynaBotConfig.validate() descends: reasoning resolves to
    # WizardReasoningConfig (strategy=wizard), whose ``wizard_config`` field is
    # required, so a wizard section that omits it surfaces from the dry-run
    # build. (The reasoning configs carry no field-value validators, so a
    # missing-required-field is the deep failure available today.)
    #
    # The error type is coupled to ``wizard_config`` being a required field with
    # no default: the dataclass constructor raises ``TypeError`` for the missing
    # argument today. Accept ``ValueError`` too so a future refactor that gives
    # the field a default and moves the check into ``__post_init__`` (raising
    # ``ValueError``) still exercises the recursion-surfaces-deep-errors intent
    # rather than silently passing.
    cfg = DynaBotConfig.from_dict({"reasoning": {"strategy": "wizard"}})
    with pytest.raises((TypeError, ValueError), match="wizard_config"):
        cfg.validate()


def test_deeply_nested_grounded_section_validates_cleanly() -> None:
    # Recursion runs through hybrid -> grounded -> intent/retrieval/synthesis
    # without spurious errors: a single top-level validate() dry-run-builds the
    # whole nested tree.
    DynaBotConfig.from_dict(
        {
            "reasoning": {
                "strategy": "hybrid",
                "grounded": {
                    "intent": {
                        "mode": "extract",
                        "num_queries": 5,
                        "domain_context": "support",
                    },
                    "retrieval": {"top_k": 8, "score_threshold": 0.25},
                    "synthesis": {"style": "concise"},
                },
                "react": {"max_iterations": 6, "store_trace": True},
            }
        }
    ).validate()


# --- skip sentinel: bare-callable strategy skipped, not raised ---


def test_bare_callable_strategy_skipped_without_raising(
    register_untyped_backend: Callable[..., str],
) -> None:
    # A custom strategy registered as a bare callable has no CONFIG_CLS; the
    # resolver returns SKIP_VALIDATION, so validate() skips its section rather
    # than false-positive-raising on a valid, constructible strategy. This is
    # the only path that reaches the sentinel (all five built-ins are typed).
    register_untyped_backend(get_registry(), name="untyped_test_strategy")
    DynaBotConfig.from_dict(
        {"reasoning": {"strategy": "untyped_test_strategy", "anything": 1}}
    ).validate()
