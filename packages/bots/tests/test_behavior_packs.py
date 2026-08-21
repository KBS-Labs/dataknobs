"""Tests for the bot-flavored pack vocabulary and its install rail.

The generic composition machinery is exercised in
``packages/common/tests/test_packs.py``. What is under test here is the
*binding*: that ``BehaviorPackSpec``'s five declared rules produce the
right shape for a bot, that a synthesizer name a pack declares is checked
rather than silently dropped, and — the point of the whole feature — that a
composed pack reaches a live bot and its middleware actually runs on a turn.

No mocks: real ``PackRegistry``, real middleware classes, and a real
``DynaBot`` built through ``BotTestHarness`` (``EchoProvider`` underneath).
"""

from __future__ import annotations

import pathlib
import re
from typing import Any, ClassVar

import pytest

from dataknobs_bots import (
    BehaviorPackRegistry,
    BehaviorPackSpec,
    build_conversation_middleware,
    build_middleware,
    verify_stage_synthesizers,
)
from dataknobs_bots.middleware.base import Middleware
from dataknobs_bots.reasoning import (
    register_stage_synthesizer,
    unregister_stage_synthesizer,
)
from dataknobs_bots.testing import BotTestHarness
from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.packs import (
    PackRegistry,
    PackResolutionError,
    PackResolutionReason,
    PackWarningCode,
)
from dataknobs_common.testing import assert_structured_config_roundtrip
from dataknobs_llm.conversations import ConversationMiddleware

_LOGGING_MW = "dataknobs_bots.middleware.logging.LoggingMiddleware"
_COST_MW = "dataknobs_bots.middleware.cost.CostTrackingMiddleware"
_REDACTION_MW = "dataknobs_llm.conversations.HistoryRedactionMiddleware"


@pytest.fixture
def registry() -> BehaviorPackRegistry:
    """A fresh, empty behavior-pack registry.

    Per-test rather than module-level: DK ships no packs and provides no
    singleton, precisely because a pack binding is a per-deployment
    decision.
    """
    return PackRegistry("behavior_packs", BehaviorPackSpec)


# ---------------------------------------------------------------------------
# Composition — the five declared rules, together
# ---------------------------------------------------------------------------


def test_two_packs_compose_across_every_field(
    registry: BehaviorPackRegistry,
) -> None:
    """Each field folds under its own declared rule, in priority order."""
    registry.register_pack(
        BehaviorPackSpec(
            name="observability",
            priority=20,
            middleware=({"class": _COST_MW},),
            conversation_middleware=({"class": _REDACTION_MW, "params": {"redactions": []}},),
            strategy_overrides={"max_iterations": 5, "verbose": True},
            stage_synthesizers=("intent_confirm", "vendor_select"),
        )
    )
    registry.register_pack(
        BehaviorPackSpec(
            name="baseline",
            priority=10,
            required_strategy="wizard",
            middleware=({"class": _LOGGING_MW},),
            strategy_overrides={"max_iterations": 3},
            stage_synthesizers=("intent_confirm",),
        )
    )

    resolution = registry.resolve({"baseline": {}, "observability": {}})
    spec = resolution.spec

    # Ascending priority: baseline (10) folds before observability (20),
    # regardless of registration or binding order.
    assert resolution.packs == ("baseline", "observability")

    # CONCAT — order preserved, both flavors.
    assert [m["class"] for m in spec.middleware] == [_LOGGING_MW, _COST_MW]
    assert [m["class"] for m in spec.conversation_middleware] == [_REDACTION_MW]

    # MERGE — later (higher-priority) pack wins the contested key.
    assert spec.strategy_overrides == {"max_iterations": 5, "verbose": True}

    # UNANIMOUS — one pack sets it, the other leaves it at its default.
    assert spec.required_strategy == "wizard"

    # CONCAT_UNIQUE — the shared name appears once.
    assert spec.stage_synthesizers == ("intent_confirm", "vendor_select")


def test_the_contested_override_key_is_reported(
    registry: BehaviorPackRegistry,
) -> None:
    """A silently-won MERGE collision would be the defect; it warns."""
    registry.register_pack(
        BehaviorPackSpec(name="low", priority=1, strategy_overrides={"max_iterations": 3})
    )
    registry.register_pack(
        BehaviorPackSpec(name="high", priority=2, strategy_overrides={"max_iterations": 5})
    )

    resolution = registry.resolve({"low": {}, "high": {}})

    assert resolution.spec.strategy_overrides == {"max_iterations": 5}
    assert [w.code for w in resolution.warnings] == ["key_override"]
    assert resolution.warnings[0].field == "strategy_overrides"
    assert "max_iterations" in str(resolution.warnings[0])


def test_conflicting_required_strategy_raises(
    registry: BehaviorPackRegistry,
) -> None:
    """Two packs demanding different strategies is unsatisfiable.

    Keeping one silently would ship a bot that violates the other pack's
    stated requirement — which is the failure this whole mechanism exists
    to make loud.
    """
    registry.register_pack(BehaviorPackSpec(name="wiz", priority=1, required_strategy="wizard"))
    registry.register_pack(BehaviorPackSpec(name="react", priority=2, required_strategy="react"))

    with pytest.raises(PackResolutionError) as excinfo:
        registry.resolve({"wiz": {}, "react": {}})

    assert excinfo.value.reason == "field_conflict"
    assert "required_strategy" in str(excinfo.value)


def test_agreeing_required_strategy_reconciles(
    registry: BehaviorPackRegistry,
) -> None:
    """Two packs asserting the *same* requirement is not a conflict."""
    registry.register_pack(BehaviorPackSpec(name="a", priority=1, required_strategy="wizard"))
    registry.register_pack(BehaviorPackSpec(name="b", priority=2, required_strategy="wizard"))

    assert registry.resolve({"a": {}, "b": {}}).spec.required_strategy == "wizard"


def test_a_disabled_pack_contributes_nothing(
    registry: BehaviorPackRegistry,
) -> None:
    """``enabled: false`` excludes the pack from every field."""
    registry.register_pack(
        BehaviorPackSpec(
            name="audit",
            priority=1,
            required_strategy="wizard",
            middleware=({"class": _LOGGING_MW},),
        )
    )
    registry.register_pack(
        BehaviorPackSpec(name="cost", priority=2, middleware=({"class": _COST_MW},))
    )

    resolution = registry.resolve({"audit": {"enabled": False}, "cost": {}})

    assert resolution.packs == ("cost",)
    assert [m["class"] for m in resolution.spec.middleware] == [_COST_MW]
    assert resolution.spec.required_strategy is None


# ---------------------------------------------------------------------------
# Config-authoring path — YAML shapes, not hand-built tuples
# ---------------------------------------------------------------------------


def test_yaml_shaped_from_dict_produces_a_working_spec() -> None:
    """Lists and nested dicts, as a config loader would deliver them.

    ``StructuredConfig.from_dict`` assigns non-config values verbatim, so a
    YAML list lands in a ``tuple[...]``-annotated field as a ``list`` unless
    the spec base normalizes it. Without that normalization the value would
    also *alias* the caller's data. Both are checked here on the real bot
    vocabulary, not just on the generic base.
    """
    raw: dict[str, Any] = {
        "name": "from_yaml",
        "priority": 7,
        "required_strategy": "wizard",
        "strategy_overrides": {"max_iterations": 4, "nested": {"a": 1}},
        "middleware": [{"class": _LOGGING_MW, "params": {"log_level": "DEBUG"}}],
        "conversation_middleware": [{"class": _REDACTION_MW}],
        "stage_synthesizers": ["intent_confirm"],
    }

    spec = BehaviorPackSpec.from_dict(raw)

    assert isinstance(spec.middleware, tuple)
    assert isinstance(spec.conversation_middleware, tuple)
    assert isinstance(spec.stage_synthesizers, tuple)
    assert spec.priority == 7
    assert spec.strategy_overrides["nested"] == {"a": 1}

    # The frozen spec does not alias the caller's mapping.
    raw["strategy_overrides"]["max_iterations"] = 99
    assert spec.strategy_overrides["max_iterations"] == 4


def test_spec_round_trips_through_to_dict() -> None:
    """``from_dict(to_dict())`` is an identity on the bot vocabulary."""
    spec = BehaviorPackSpec(
        name="rt",
        priority=3,
        required_strategy="react",
        strategy_overrides={"k": "v"},
        middleware=({"class": _LOGGING_MW},),
        conversation_middleware=({"class": _REDACTION_MW},),
        stage_synthesizers=("intent_confirm",),
    )

    assert_structured_config_roundtrip(spec)


def test_a_binding_tunes_a_pack_without_editing_it(
    registry: BehaviorPackRegistry,
) -> None:
    """The deployment-side half: overrides compose under the same rules."""
    registry.register_pack(
        BehaviorPackSpec(
            name="baseline",
            priority=10,
            middleware=({"class": _LOGGING_MW},),
            strategy_overrides={"max_iterations": 3},
        )
    )

    resolution = registry.resolve(
        {
            "baseline": {
                "priority": 1,
                "middleware": [{"class": _COST_MW}],
                "strategy_overrides": {"max_iterations": 9},
            }
        }
    )

    # CONCAT: the binding appends to the pack's own middleware.
    assert [m["class"] for m in resolution.spec.middleware] == [_LOGGING_MW, _COST_MW]
    # MERGE: the binding wins the key.
    assert resolution.spec.strategy_overrides == {"max_iterations": 9}


# ---------------------------------------------------------------------------
# verify_stage_synthesizers
# ---------------------------------------------------------------------------


class _ProbeSynthesizer:
    """A minimal real synthesizer — the registry keys off ``field``."""

    field = "behavior_pack_probe"

    def synthesize(self, stage: dict[str, Any]) -> None:  # pragma: no cover
        """Not exercised: verification never invokes a synthesizer."""
        stage.pop(self.field, None)


@pytest.fixture
def probe_synthesizer():
    """Register the probe for one test, then remove it.

    The synthesizer registry is process-global (registration is an
    import-time act), so a test that registers must unregister.
    """
    register_stage_synthesizer(_ProbeSynthesizer())
    yield _ProbeSynthesizer.field
    unregister_stage_synthesizer(_ProbeSynthesizer.field)


def test_verify_passes_for_a_registered_synthesizer(probe_synthesizer: str) -> None:
    """A declared name that is registered is silently fine.

    The assertion is the absence of a raise — verification is a guard, not
    a query, so there is nothing else to check.
    """
    verify_stage_synthesizers([probe_synthesizer])


def test_verify_raises_for_an_unregistered_synthesizer() -> None:
    """The typo hole a fail-soft lookup would leave open."""
    with pytest.raises(ConfigurationError) as excinfo:
        verify_stage_synthesizers(["definitely_not_registered"])

    assert "definitely_not_registered" in str(excinfo.value)
    assert excinfo.value.context["missing"] == ["definitely_not_registered"]


def test_verify_reports_every_missing_name_at_once(probe_synthesizer: str) -> None:
    """One call reports the whole gap, not the first of it."""
    with pytest.raises(ConfigurationError) as excinfo:
        verify_stage_synthesizers([probe_synthesizer, "missing_b", "missing_a"])

    assert excinfo.value.context["missing"] == ["missing_a", "missing_b"]
    assert probe_synthesizer in excinfo.value.context["available"]


def test_verify_accepts_an_empty_declaration() -> None:
    """A pack that declares no synthesizers verifies trivially."""
    verify_stage_synthesizers(())


def test_verify_consumes_a_resolution_directly(
    registry: BehaviorPackRegistry, probe_synthesizer: str
) -> None:
    """The documented call shape — resolution field straight in."""
    registry.register_pack(BehaviorPackSpec(name="p", stage_synthesizers=(probe_synthesizer,)))

    resolution = registry.resolve({"p": {}})

    verify_stage_synthesizers(resolution.spec.stage_synthesizers)


# ---------------------------------------------------------------------------
# The full rail — packs to a live bot
# ---------------------------------------------------------------------------


class _RecordingMiddleware(Middleware):
    """Records that a bot turn actually ran through it.

    A real ``Middleware`` subclass rather than a mock: the point of the
    test is that the class resolves by dotted path, instantiates with
    ``params``, and receives real turn callbacks.
    """

    #: Instances are found through this module-level list because the spec
    #: names a class, not an object — the factory constructs it.
    seen: ClassVar[list[str]] = []

    def __init__(self, tag: str = "default") -> None:
        self.tag = tag

    async def after_turn(self, turn: Any) -> None:
        _RecordingMiddleware.seen.append(self.tag)


class _RecordingConversationMiddleware(ConversationMiddleware):
    """The LLM-call-wrap analogue, recording that it wrapped a request."""

    seen: ClassVar[list[str]] = []

    def __init__(self, tag: str = "default") -> None:
        self.tag = tag

    async def process_request(self, messages: Any, state: Any) -> Any:
        _RecordingConversationMiddleware.seen.append(self.tag)
        return messages

    async def process_response(self, response: Any, state: Any) -> Any:
        return response


_RECORDING_MW = f"{__name__}._RecordingMiddleware"
_RECORDING_CONV_MW = f"{__name__}._RecordingConversationMiddleware"


@pytest.fixture(autouse=True)
def _clear_recordings():
    """Class-level recorders are shared; reset around every test."""
    _RecordingMiddleware.seen.clear()
    _RecordingConversationMiddleware.seen.clear()
    yield
    _RecordingMiddleware.seen.clear()
    _RecordingConversationMiddleware.seen.clear()


async def test_composed_packs_reach_a_live_bot_and_run(
    registry: BehaviorPackRegistry,
) -> None:
    """The end-to-end rail this whole item exists to close.

    Pack specs -> ``resolve()`` -> ``build_middleware()`` ->
    ``platform_middleware=`` -> the middleware runs on a real turn. Every
    link is the shipped one; nothing here reimplements DK's spec
    resolution, which is precisely the coupling the feature removes.
    """
    registry.register_pack(
        BehaviorPackSpec(
            name="inner",
            priority=1,
            middleware=({"class": _RECORDING_MW, "params": {"tag": "inner"}},),
            conversation_middleware=(
                {"class": _RECORDING_CONV_MW, "params": {"tag": "conv-inner"}},
            ),
        )
    )
    registry.register_pack(
        BehaviorPackSpec(
            name="outer",
            priority=2,
            middleware=({"class": _RECORDING_MW, "params": {"tag": "outer"}},),
        )
    )

    resolution = registry.resolve({"inner": {}, "outer": {}})

    platform = build_middleware(resolution.spec.middleware)
    platform_conv = build_conversation_middleware(resolution.spec.conversation_middleware)

    assert [type(mw).__name__ for mw in platform] == [
        "_RecordingMiddleware",
        "_RecordingMiddleware",
    ]
    assert [mw.tag for mw in platform] == ["inner", "outer"]

    async with await BotTestHarness.create(
        bot_config={
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "simple"},
        },
        main_responses=["hello back"],
        platform_middleware=platform,
        platform_conversation_middleware=platform_conv,
    ) as harness:
        result = await harness.chat("hello")

    assert result.response == "hello back"
    # Pack priority order survived all the way to turn execution.
    assert _RecordingMiddleware.seen == ["inner", "outer"]
    assert _RecordingConversationMiddleware.seen == ["conv-inner"]


async def test_a_disabled_pack_does_not_install_its_middleware(
    registry: BehaviorPackRegistry,
) -> None:
    """Suppression is observable at the bot, not just in the resolution."""
    registry.register_pack(
        BehaviorPackSpec(
            name="optional_audit",
            priority=1,
            middleware=({"class": _RECORDING_MW, "params": {"tag": "audit"}},),
        )
    )
    registry.register_pack(
        BehaviorPackSpec(
            name="always",
            priority=2,
            middleware=({"class": _RECORDING_MW, "params": {"tag": "always"}},),
        )
    )

    resolution = registry.resolve({"optional_audit": {"enabled": False}, "always": {}})

    async with await BotTestHarness.create(
        bot_config={
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "simple"},
        },
        main_responses=["ok"],
        platform_middleware=build_middleware(resolution.spec.middleware),
    ) as harness:
        await harness.chat("hi")

    assert _RecordingMiddleware.seen == ["always"]


async def test_pack_middleware_is_additive_to_the_bots_own(
    registry: BehaviorPackRegistry,
) -> None:
    """A pack never replaces what the bot config already declares."""
    registry.register_pack(
        BehaviorPackSpec(
            name="platform",
            middleware=({"class": _RECORDING_MW, "params": {"tag": "platform"}},),
        )
    )
    resolution = registry.resolve({"platform": {}})

    async with await BotTestHarness.create(
        bot_config={
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {"strategy": "simple"},
            "middleware": [{"class": _LOGGING_MW}],
        },
        main_responses=["ok"],
        platform_middleware=build_middleware(resolution.spec.middleware),
    ) as harness:
        names = [type(mw).__name__ for mw in harness.bot.middleware]

    assert "LoggingMiddleware" in names
    assert "_RecordingMiddleware" in names
    # Additive means appended: the pack's middleware runs after the bot's.
    assert names.index("LoggingMiddleware") < names.index("_RecordingMiddleware")


# The vocabularies themselves live in dataknobs-common and are guarded
# against that package's guide. This guide is their *second* documented
# copy, hand-maintained and in a different package, so it needs its own
# guard — a reason removed upstream would otherwise keep being documented
# here, and a bots consumer builds their escalation table from this table.


_BEHAVIOR_PACKS_GUIDE = pathlib.Path(__file__).parents[1] / "docs" / "BEHAVIOR_PACKS.md"


def _assert_documented_vocabulary_is_real(text: str) -> None:
    """Every vocabulary member this guide names must still exist.

    Only this direction. The guide's table describes *bot* failure modes,
    not a reproduction of the upstream vocabulary, so a member it does not
    happen to mention is not drift — but one it mentions that no longer
    exists is a case a consumer will write unreachable code for.

    Both patterns are exact rather than prose-scraped: a reason appears as
    ``reason="..."`` inside a constructor illustration, and a code as
    ```...` warning``.
    """
    reasons = set(re.findall(r'reason="([a-z_]+)"', text))
    codes = set(re.findall(r"`([a-z_]+)` warning", text))

    # Both patterns are exact, so a reflow that changed how the guide spells
    # them would leave nothing to check and this guard would pass on an
    # empty set — the vacuity it exists to prevent.
    assert reasons and codes, (
        f"extracted {len(reasons)} reasons and {len(codes)} codes from the "
        f"guide; it documents both, so an empty side means the spelling "
        f"changed and the patterns above need updating"
    )

    unknown_reasons = sorted(reasons - {m.value for m in PackResolutionReason})
    unknown_codes = sorted(codes - {m.value for m in PackWarningCode})

    assert not unknown_reasons, (
        f"guide documents PackResolutionError reasons that do not exist: {unknown_reasons}"
    )
    assert not unknown_codes, (
        f"guide documents PackWarning codes that do not exist: {unknown_codes}"
    )


def test_guide_documents_only_real_vocabulary_members() -> None:
    """The bots-side copy of both vocabularies is checked, not assumed."""
    assert _BEHAVIOR_PACKS_GUIDE.is_file(), (
        f"{_BEHAVIOR_PACKS_GUIDE} is missing; its absence disables a drift "
        f"guard rather than making one inapplicable."
    )
    _assert_documented_vocabulary_is_real(_BEHAVIOR_PACKS_GUIDE.read_text())


@pytest.mark.parametrize(
    "drift",
    [
        pytest.param('reason="retired_reason"', id="reason"),
        pytest.param("`retired_code` warning", id="code"),
    ],
)
def test_guard_rejects_a_documented_member_that_no_longer_exists(drift: str) -> None:
    """The guard fires on a member this guide outlived."""
    text = _BEHAVIOR_PACKS_GUIDE.read_text() + f"\n| x | {drift} |\n"
    with pytest.raises(AssertionError):
        _assert_documented_vocabulary_is_real(text)
