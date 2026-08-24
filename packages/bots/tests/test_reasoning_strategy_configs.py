"""Tests for the non-wizard reasoning-strategy ``StructuredConfig`` family.

Covers the four configs converted/created in the core reasoning-config
migration: ``SimpleReasoningConfig``, ``ReActReasoningConfig``,
``GroundedReasoningConfig`` (+ its sub-configs), and
``HybridReasoningConfig``.  Verifies round-trip serialization, frozen
immutability, the dict-loading shape quirks preserved via
``_normalize_dict`` (legacy ``query_generation`` alias, empty
``result_processing``, nested ``react`` flattening), and that each
strategy class declares ``CONFIG_CLS`` pointing at its config.

No validation binding is exercised here — that lands with the resolver
in a later PR.  These configs are typed/redacted/round-trippable on
their own.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging

import pytest

from dataknobs_common.testing import assert_structured_config_roundtrip

from dataknobs_bots.reasoning.config_base import ReasoningConfig
from dataknobs_bots.reasoning.grounded import GroundedReasoning
from dataknobs_bots.reasoning.grounded_config import (
    GroundedIntentConfig,
    GroundedReasoningConfig,
    GroundedResultProcessingConfig,
    GroundedRetrievalConfig,
    GroundedSourceConfig,
    GroundedSynthesisConfig,
)
from dataknobs_bots.reasoning.hybrid import HybridReasoning
from dataknobs_bots.reasoning.hybrid_config import HybridReasoningConfig
from dataknobs_bots.reasoning.react import ReActReasoning
from dataknobs_bots.reasoning.react_config import ReActReasoningConfig
from dataknobs_bots.reasoning.simple import SimpleReasoning
from dataknobs_bots.reasoning.simple_config import SimpleReasoningConfig
from dataknobs_bots.reasoning.wizard_config import WizardReasoningConfig

#: Every ``ReasoningConfig`` subclass shipped as a strategy config.  Named
#: once because three tests below assert over the whole family, and a
#: sixth strategy config added to only two of the three lists would be
#: the exact silent gap this module exists to close.
_FAMILY = [
    SimpleReasoningConfig,
    ReActReasoningConfig,
    GroundedReasoningConfig,
    HybridReasoningConfig,
    WizardReasoningConfig,
]
_FAMILY_IDS = ["simple", "react", "grounded", "hybrid", "wizard"]


class TestConfigClsPointers:
    """Each strategy declares ``CONFIG_CLS`` → its config (a pointer only)."""

    def test_simple_pointer(self) -> None:
        assert SimpleReasoning.CONFIG_CLS is SimpleReasoningConfig

    def test_react_pointer(self) -> None:
        assert ReActReasoning.CONFIG_CLS is ReActReasoningConfig

    def test_grounded_pointer(self) -> None:
        assert GroundedReasoning.CONFIG_CLS is GroundedReasoningConfig

    def test_hybrid_pointer(self) -> None:
        assert HybridReasoning.CONFIG_CLS is HybridReasoningConfig


class TestSimpleReasoningConfig:
    def test_default_roundtrip(self) -> None:
        assert_structured_config_roundtrip(SimpleReasoningConfig())

    def test_populated_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            SimpleReasoningConfig(greeting_template="Hi {{ user_name }}!")
        )

    def test_frozen(self) -> None:
        cfg = SimpleReasoningConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.greeting_template = "nope"  # type: ignore[misc]

    def test_from_dict_ignores_unknown_keys(self) -> None:
        cfg = SimpleReasoningConfig.from_dict({"greeting_template": "hello", "bogus": 1})
        assert cfg.greeting_template == "hello"


class TestReActReasoningConfig:
    def test_default_roundtrip(self) -> None:
        assert_structured_config_roundtrip(ReActReasoningConfig())

    def test_populated_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            ReActReasoningConfig(
                max_iterations=8,
                verbose=True,
                store_trace=True,
                greeting_template="Hello",
            )
        )

    def test_defaults_match_from_config_defaults(self) -> None:
        # The defaults mirror ReActReasoning.from_config's dict .get defaults.
        cfg = ReActReasoningConfig()
        assert cfg.max_iterations == 5
        assert cfg.verbose is False
        assert cfg.store_trace is False
        assert cfg.greeting_template is None

    def test_frozen(self) -> None:
        cfg = ReActReasoningConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.max_iterations = 99  # type: ignore[misc]


class TestGroundedReasoningConfig:
    def test_default_roundtrip(self) -> None:
        assert_structured_config_roundtrip(GroundedReasoningConfig())

    def test_nested_roundtrip(self) -> None:
        cfg = GroundedReasoningConfig(
            intent=GroundedIntentConfig(mode="static", num_queries=2),
            retrieval=GroundedRetrievalConfig(top_k=10, deduplicate=False),
            synthesis=GroundedSynthesisConfig(style="hybrid", require_citations=False),
            result_processing=GroundedResultProcessingConfig(
                normalize_strategy="min_max", min_results=2
            ),
            sources=[
                GroundedSourceConfig(source_type="vector_kb", name="docs"),
                GroundedSourceConfig(source_type="database", name="sql"),
            ],
            store_provenance=False,
            greeting_template="Hello",
        )
        assert_structured_config_roundtrip(cfg)

    def test_subconfigs_are_structured(self) -> None:
        # The four sub-configs (plus the already-converted source config)
        # all round-trip independently.
        assert_structured_config_roundtrip(GroundedIntentConfig())
        assert_structured_config_roundtrip(GroundedRetrievalConfig())
        assert_structured_config_roundtrip(GroundedSynthesisConfig())
        assert_structured_config_roundtrip(GroundedResultProcessingConfig())
        assert_structured_config_roundtrip(GroundedSourceConfig())

    def test_frozen(self) -> None:
        cfg = GroundedReasoningConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.store_provenance = False  # type: ignore[misc]

    def test_from_dict_builds_nested(self) -> None:
        cfg = GroundedReasoningConfig.from_dict(
            {
                "intent": {"mode": "static", "num_queries": 4},
                "retrieval": {"top_k": 7},
                "synthesis": {"require_citations": False},
                "sources": [{"type": "vector_kb", "name": "kb"}],
            }
        )
        assert isinstance(cfg.intent, GroundedIntentConfig)
        assert cfg.intent.mode == "static"
        assert cfg.intent.num_queries == 4
        assert isinstance(cfg.retrieval, GroundedRetrievalConfig)
        assert cfg.retrieval.top_k == 7
        assert cfg.synthesis.require_citations is False
        assert len(cfg.sources) == 1
        # Source's ``type`` alias normalized to ``source_type``.
        assert cfg.sources[0].source_type == "vector_kb"
        assert cfg.sources[0].name == "kb"

    def test_legacy_query_generation_alias(self) -> None:
        # The legacy ``query_generation`` key maps to ``intent`` when
        # ``intent`` is absent.
        cfg = GroundedReasoningConfig.from_dict({"query_generation": {"num_queries": 9}})
        assert cfg.intent.num_queries == 9
        # The alias property still mirrors intent.
        assert cfg.query_generation is cfg.intent

    def test_intent_takes_precedence_over_legacy_alias(self) -> None:
        cfg = GroundedReasoningConfig.from_dict(
            {
                "intent": {"num_queries": 1},
                "query_generation": {"num_queries": 9},
            }
        )
        assert cfg.intent.num_queries == 1

    def test_empty_result_processing_disables_pipeline(self) -> None:
        # An empty/None ``result_processing`` must default to None (pipeline
        # disabled), not a default-constructed config.
        assert (
            GroundedReasoningConfig.from_dict({"result_processing": {}}).result_processing is None
        )
        assert (
            GroundedReasoningConfig.from_dict({"result_processing": None}).result_processing is None
        )
        assert GroundedReasoningConfig.from_dict({}).result_processing is None

    def test_result_processing_present_builds_config(self) -> None:
        cfg = GroundedReasoningConfig.from_dict(
            {"result_processing": {"normalize_strategy": "z_score"}}
        )
        assert isinstance(cfg.result_processing, GroundedResultProcessingConfig)
        assert cfg.result_processing.normalize_strategy == "z_score"


class TestHybridReasoningConfig:
    def test_default_roundtrip(self) -> None:
        assert_structured_config_roundtrip(HybridReasoningConfig())

    def test_nested_roundtrip(self) -> None:
        cfg = HybridReasoningConfig(
            grounded=GroundedReasoningConfig(
                intent=GroundedIntentConfig(mode="static"),
                retrieval=GroundedRetrievalConfig(top_k=3),
            ),
            react_max_iterations=7,
            react_verbose=True,
            react_store_trace=True,
            store_provenance=True,
            greeting_template="Hello",
        )
        assert_structured_config_roundtrip(cfg)

    def test_frozen(self) -> None:
        cfg = HybridReasoningConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.react_max_iterations = 1  # type: ignore[misc]

    def test_from_dict_flattens_nested_react(self) -> None:
        cfg = HybridReasoningConfig.from_dict(
            {
                "grounded": {
                    "intent": {"mode": "static"},
                    "retrieval": {"top_k": 10},
                },
                "react": {
                    "max_iterations": 8,
                    "verbose": True,
                    "store_trace": True,
                },
            }
        )
        assert cfg.react_max_iterations == 8
        assert cfg.react_verbose is True
        assert cfg.react_store_trace is True
        assert isinstance(cfg.grounded, GroundedReasoningConfig)
        assert cfg.grounded.intent.mode == "static"
        assert cfg.grounded.retrieval.top_k == 10

    def test_from_dict_react_defaults_when_absent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="dataknobs_bots.reasoning.hybrid_config"):
            cfg = HybridReasoningConfig.from_dict({"grounded": {}})
        assert cfg.react_max_iterations == 5
        assert cfg.react_verbose is False
        assert cfg.react_store_trace is False
        # Both grounded and hybrid default ``store_provenance`` to True, so the
        # provenance-mismatch warning must stay silent on the defaults path. A
        # future default shift that introduced a spurious warning would trip here.
        assert not [rec for rec in caplog.records if "store_provenance" in rec.message]

    def test_provenance_mismatch_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="dataknobs_bots.reasoning.hybrid_config"):
            HybridReasoningConfig(
                grounded=GroundedReasoningConfig(store_provenance=True),
                store_provenance=False,
            )
        assert any(
            "store_provenance" in rec.message and "effective" in rec.message
            for rec in caplog.records
        )

    def test_matching_provenance_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="dataknobs_bots.reasoning.hybrid_config"):
            HybridReasoningConfig(
                grounded=GroundedReasoningConfig(store_provenance=True),
                store_provenance=True,
            )
        assert not [rec for rec in caplog.records if "store_provenance" in rec.message]


# ===================================================================
# TestReasoningConfigBase
# ===================================================================


class TestReasoningConfigBase:
    """The family's shared base, and the one field it exists to declare.

    ``ReasoningStrategy`` calls ``greeting_template`` universal.  Before
    this base, every strategy config re-declared it, and a config that
    forgot to lost the field silently -- ``StructuredConfig`` ignores an
    undeclared key rather than reporting it.  Declaring it once removes
    the opportunity rather than guarding against taking it.
    """

    def test_the_base_declares_the_universal_field(self) -> None:
        names = {f.name for f in dataclasses.fields(ReasoningConfig)}
        assert names == {"greeting_template"}

    @pytest.mark.parametrize("config_cls", _FAMILY, ids=_FAMILY_IDS)
    def test_every_strategy_config_inherits_it(self, config_cls: type[ReasoningConfig]) -> None:
        """Both halves of "declared once": inherited, and not re-declared.

        The second half is the recurrence guard.  A subclass that
        re-declares the field still *works* -- which is exactly why
        nothing would notice the family drifting back to five
        declarations, and why this asserts on ``__annotations__`` (the
        class's own — that is what ``inspect.get_annotations`` returns)
        rather than on ``fields()``, which includes inherited.
        """
        assert issubclass(config_cls, ReasoningConfig)
        assert "greeting_template" in {f.name for f in dataclasses.fields(config_cls)}
        assert "greeting_template" not in inspect.get_annotations(config_cls), (
            f"{config_cls.__name__} re-declares greeting_template. It is "
            f"declared once on ReasoningConfig; a second declaration is the "
            f"state this base was introduced to end."
        )

    def test_a_subclass_may_declare_a_required_field(self) -> None:
        """T2 -- what ``kw_only=True`` on the base buys.

        Without it, ``@dataclass`` refuses the whole *class*, at import:
        ``TypeError: non-default argument 'wizard_config' follows default
        argument``.  The base's field is defaulted, so any subclass field
        that is not would follow a default -- and the wizard has one.
        Delete ``kw_only=True`` and this fails at collection, not here.
        """

        @dataclasses.dataclass(frozen=True)
        class RequiresSomething(ReasoningConfig):
            needed: str

        cfg = RequiresSomething(needed="x")
        assert cfg.needed == "x"
        assert cfg.greeting_template is None

        with pytest.raises(TypeError):
            RequiresSomething()  # type: ignore[call-arg]

    def test_the_shipped_required_field_survives_both_paths(self) -> None:
        """T6 -- the wizard is the real instance of the case above.

        Both construction paths, because they are different code: the
        generated ``__init__`` and ``StructuredConfig.from_dict``.
        """
        direct = WizardReasoningConfig(wizard_config="w.yaml", greeting_template="Hi {{ who }}!")
        assert direct.wizard_config == "w.yaml"
        assert direct.greeting_template == "Hi {{ who }}!"

        loaded = WizardReasoningConfig.from_dict(
            {"wizard_config": "w.yaml", "greeting_template": "Hi {{ who }}!"}
        )
        assert loaded == direct

    def test_the_field_is_keyword_only(self) -> None:
        """Keyword-only on the base is forced by inheritance.

        A base field is declared before every subclass field, so a
        defaulted one would sit ahead of the wizard's required
        ``wizard_config`` -- rejected at import with ``non-default
        argument follows default argument``.  ``kw_only`` moves it behind
        the ``*`` instead, which is what makes the shared base possible
        at all.
        """
        (field,) = dataclasses.fields(ReasoningConfig)
        assert field.kw_only is True

    @pytest.mark.parametrize("config_cls", _FAMILY, ids=_FAMILY_IDS)
    def test_the_whole_family_is_keyword_only(self, config_cls: type) -> None:
        """Every field of every strategy config, not just the shared one.

        Inheriting ``greeting_template`` moves it behind the ``*``, which
        shifts each subclass's own fields one position left.  Four of the
        five configs raise on a call that relied on the old order.  The
        wizard did not: its second positional was ``greeting_template``
        and became ``config_base_path``, both ``str | None``, so
        ``WizardReasoningConfig(cfg, "Hello!")`` still *constructed* --
        with no greeting and a nonsense base path.  A silent rebind is
        the failure this whole change exists to remove, so the family is
        keyword-only throughout and every such call now raises.

        This is the guard for that decision: a subclass that drops
        ``kw_only`` re-opens the shift for its own fields.
        """
        for field in dataclasses.fields(config_cls):
            assert field.kw_only is True, (
                f"{config_cls.__name__}.{field.name} is positionally reachable; "
                "the family is keyword-only so a stale positional call raises "
                "instead of rebinding to the neighbouring field"
            )

    def test_a_positional_call_raises_rather_than_rebinding(self) -> None:
        """The wizard case specifically, since it is the one that was quiet.

        ``str | None`` next to ``str | None`` is invisible to the
        dataclass machinery -- nothing about the value distinguishes a
        greeting from a path.  Only the signature can reject it.
        """
        with pytest.raises(TypeError, match="positional"):
            WizardReasoningConfig({"stages": {}}, "Hello!")  # type: ignore[misc]

        ok = WizardReasoningConfig(wizard_config={"stages": {}}, greeting_template="Hello!")
        assert ok.greeting_template == "Hello!"
        assert ok.config_base_path is None
