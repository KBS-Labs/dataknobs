"""Construction-path tests for the reasoning-strategy consumer-mixin adoption.

The non-wizard reasoning strategies — ``SimpleReasoning``,
``ReActReasoning``, ``GroundedReasoning``, ``HybridReasoning`` — are built
from a frozen ``StructuredConfig`` via
:class:`~dataknobs_common.structured_config.StructuredConfigConsumer`.
Each gains the uniform construction surface: a typed-config ctor, a
dict-dispatch ``cls.from_config({...})`` (the entry point the reasoning
registry calls), loose-kwarg construction (``cls(greeting_template=...)``),
and a typed read-only ``self.config`` — replacing the per-strategy bespoke
``from_config`` classmethod and hand-rolled ``__init__``.

These tests pin that contract per adopter: the typed-config, dict, and
``from_config`` paths reach identical config state; ``self.config`` is the
typed config (not a dict); mixing a typed config with loose kwargs raises
``TypeError``; the derived ``_setup`` attributes are computed; injected
collaborators travel through the ``components`` channel (not the config);
and the parity guard (:func:`assert_structured_config_consumer`) holds —
including the MRO-ordering check (the mixin must precede ``ReasoningStrategy``
so its ``__init__`` is the construction entry point).

Collaborators are *not* config: ReAct's ``prompt_refresher`` / registries
and grounded's ``prompt_resolver`` / ``query_provider`` are opaque runtime
objects threaded through ``from_config(config, collaborator=...)``. They land
on ``self.components`` and are bound in ``_setup`` — never folded into
``self.config`` (a ``from_dict`` would silently drop them). The strategies'
behavior (retrieval pipeline, ReAct loop, KB auto-wrap) is covered by the
existing reasoning + DynaBot suites; these tests cover only the construction
surface the mixin standardizes.

No external service is required — construction only.
"""

from __future__ import annotations

import pytest
from dataknobs_common.testing import (
    assert_structured_config_consumer,
    assert_structured_config_roundtrip,
)

from dataknobs_bots.reasoning.grounded import GroundedReasoning
from dataknobs_bots.reasoning.grounded_config import GroundedReasoningConfig
from dataknobs_bots.reasoning.hybrid import HybridReasoning
from dataknobs_bots.reasoning.hybrid_config import HybridReasoningConfig
from dataknobs_bots.reasoning.react import ReActReasoning
from dataknobs_bots.reasoning.react_config import ReActReasoningConfig
from dataknobs_bots.reasoning.simple import SimpleReasoning
from dataknobs_bots.reasoning.simple_config import SimpleReasoningConfig

# (consumer_cls, config_cls, typed_config, equivalent_dict)
#
# Every config in this matrix is all-default-constructible, so the
# default-construction and round-trip checks apply uniformly.
ADOPTERS = [
    (
        SimpleReasoning,
        SimpleReasoningConfig,
        SimpleReasoningConfig(greeting_template="Hi {{ name }}!"),
        {"greeting_template": "Hi {{ name }}!"},
    ),
    (
        ReActReasoning,
        ReActReasoningConfig,
        ReActReasoningConfig(max_iterations=3, store_trace=True),
        {"max_iterations": 3, "store_trace": True},
    ),
    (
        GroundedReasoning,
        GroundedReasoningConfig,
        GroundedReasoningConfig(greeting_template="Hello!"),
        {"greeting_template": "Hello!"},
    ),
    (
        HybridReasoning,
        HybridReasoningConfig,
        HybridReasoningConfig(react_max_iterations=4),
        {"react": {"max_iterations": 4}},
    ),
]

_IDS = [a[0].__name__ for a in ADOPTERS]


@pytest.mark.parametrize("consumer_cls", [a[0] for a in ADOPTERS], ids=_IDS)
def test_parity_guard(consumer_cls: type) -> None:
    """The unified parity contract holds (CONFIG_CLS, field/ctor match, MRO).

    Its MRO-ordering check (the mixin must precede ``ReasoningStrategy``)
    subsumes a bare ``issubclass(consumer_cls, StructuredConfigConsumer)``
    assertion — that the class mixes in the consumer is a precondition of
    the ordering check — so no separate subclass test is kept.

    No ``ignore_params``: the strategies drop their custom ``__init__``, so
    the inspected ctor is the mixin's ``(config, *, _components, **kwargs)``
    and the dataclass-field-vs-ctor check is trivially satisfied. The
    collaborators ReAct/grounded once took as ctor params now travel through
    the ``components`` channel, off the ctor surface entirely.
    """
    assert_structured_config_consumer(consumer_cls)


@pytest.mark.parametrize(
    "consumer_cls, config_cls, typed_config, equivalent_dict",
    ADOPTERS,
    ids=_IDS,
)
def test_typed_dict_and_from_config_reach_same_state(
    consumer_cls, config_cls, typed_config, equivalent_dict
) -> None:
    """Typed-config ctor, dict ctor, and from_config reach identical config."""
    via_typed = consumer_cls(typed_config)
    via_dict = consumer_cls(equivalent_dict)
    via_from_config = consumer_cls.from_config(equivalent_dict)

    # ``self.config`` is the typed config (not a dict).
    assert isinstance(via_typed.config, config_cls)
    assert isinstance(via_dict.config, config_cls)
    assert isinstance(via_from_config.config, config_cls)

    # All three paths produce an equal config.
    assert via_typed.config == typed_config
    assert via_dict.config == typed_config
    assert via_from_config.config == typed_config


def test_hybrid_nested_grounded_reaches_same_state() -> None:
    """A non-default nested ``grounded`` sub-config reconstructs identically.

    The ADOPTERS matrix entry for Hybrid varies only ``react_max_iterations``,
    leaving ``grounded`` at its default — so a ``from_dict`` regression in the
    nested ``grounded`` tree would pass silently there (both sides would fall
    back to the same grounded default). This pins the nested-grounded
    reconstruction explicitly: the typed-config ctor, the dict ctor, and
    ``from_config`` must all reach the same state with a non-default
    ``grounded.greeting_template``.
    """
    typed = HybridReasoningConfig(
        grounded=GroundedReasoningConfig(greeting_template="Hey"),
        react_max_iterations=4,
    )
    equivalent_dict = {
        "grounded": {"greeting_template": "Hey"},
        "react": {"max_iterations": 4},
    }
    via_typed = HybridReasoning(typed)
    via_dict = HybridReasoning(equivalent_dict)
    via_from_config = HybridReasoning.from_config(equivalent_dict)

    assert via_typed.config == typed
    assert via_dict.config == typed
    assert via_from_config.config == typed
    # The nested grounded sub-config specifically survived reconstruction
    # (not merely defaulted equal on both sides).
    assert via_dict.config.grounded.greeting_template == "Hey"
    assert via_from_config.config.grounded.greeting_template == "Hey"


@pytest.mark.parametrize(
    "consumer_cls, _config_cls, typed_config, _equivalent_dict",
    ADOPTERS,
    ids=_IDS,
)
def test_mixing_typed_config_with_kwargs_raises(
    consumer_cls, _config_cls, typed_config, _equivalent_dict
) -> None:
    """A typed config plus loose kwargs is a construction error."""
    with pytest.raises(TypeError):
        consumer_cls(typed_config, some_unexpected_kwarg=1)


@pytest.mark.parametrize(
    "consumer_cls, config_cls, _typed_config, _equivalent_dict",
    ADOPTERS,
    ids=_IDS,
)
def test_default_construction(
    consumer_cls, config_cls, _typed_config, _equivalent_dict
) -> None:
    """Every strategy config is all-default — construction with no args works."""
    obj = consumer_cls()
    assert isinstance(obj.config, config_cls)
    assert obj.config == config_cls()


@pytest.mark.parametrize(
    "_consumer_cls, _config_cls, typed_config, _equivalent_dict",
    ADOPTERS,
    ids=_IDS,
)
def test_config_roundtrip(
    _consumer_cls, _config_cls, typed_config, _equivalent_dict
) -> None:
    """Each strategy config round-trips through from_dict/to_dict."""
    assert_structured_config_roundtrip(typed_config)


@pytest.mark.parametrize(
    "consumer_cls, _config_cls, typed_config, _equivalent_dict",
    ADOPTERS,
    ids=_IDS,
)
def test_config_property_is_read_only(
    consumer_cls, _config_cls, typed_config, _equivalent_dict
) -> None:
    """``config`` is a read-only property — reassignment raises."""
    obj = consumer_cls(typed_config)
    with pytest.raises(AttributeError):
        obj.config = typed_config


# --- Per-strategy derived-state (_setup) checks --------------------------


def test_simple_setup_binds_greeting_template() -> None:
    strat = SimpleReasoning.from_config({"greeting_template": "Hi {{ name }}!"})
    assert strat._greeting_template == "Hi {{ name }}!"
    # Loose-kwarg construction reaches the same field (greeting_template is a
    # config field, so the mixin merges it through from_dict).
    assert SimpleReasoning(greeting_template="Hi {{ name }}!").config == strat.config


def test_react_setup_binds_scalars_and_defaults_collaborators() -> None:
    strat = ReActReasoning.from_config(
        {"max_iterations": 7, "verbose": True, "store_trace": True}
    )
    assert strat.max_iterations == 7
    assert strat.verbose is True
    assert strat.store_trace is True
    # No collaborators injected → all default to None.
    assert strat._artifact_registry is None
    assert strat._review_executor is None
    assert strat._context_builder is None
    assert strat._extra_context is None
    assert strat._prompt_refresher is None


def test_react_collaborators_travel_through_components_not_config() -> None:
    def refresher() -> str:
        return "fresh prompt"

    registry = object()
    strat = ReActReasoning.from_config(
        {"max_iterations": 2},
        prompt_refresher=refresher,
        artifact_registry=registry,
        extra_context={"k": "v"},
    )
    # Collaborators bound from the components channel.
    assert strat._prompt_refresher is refresher
    assert strat._artifact_registry is registry
    assert strat._extra_context == {"k": "v"}
    # Scalars still come from the typed config.
    assert strat.max_iterations == 2
    # Collaborators are NOT folded into the config (a from_dict would drop them).
    assert strat.components["prompt_refresher"] is refresher
    assert not hasattr(strat.config, "prompt_refresher")


def test_react_scalar_loose_kwargs_still_construct() -> None:
    """The vast-majority scalar call shape keeps working via from_dict merge."""
    strat = ReActReasoning(max_iterations=9, verbose=True)
    assert strat.config == ReActReasoningConfig(max_iterations=9, verbose=True)
    assert strat.max_iterations == 9


def test_grounded_setup_binds_config_and_collaborators() -> None:
    query_provider = object()
    prompt_resolver = object()
    strat = GroundedReasoning.from_config(
        {"greeting_template": "Hi!"},
        query_provider=query_provider,
        prompt_resolver=prompt_resolver,
    )
    assert strat._greeting_template == "Hi!"
    assert strat._query_provider is query_provider
    assert strat._prompt_resolver is prompt_resolver
    # No knowledge_base / sources injected → empty source list.
    assert strat._sources == []


def test_grounded_typed_config_only_construction() -> None:
    """The dominant test-construction shape ``GroundedReasoning(config=cfg)``."""
    cfg = GroundedReasoningConfig(greeting_template="Hey")
    strat = GroundedReasoning(config=cfg)
    assert strat.config is cfg
    assert strat._greeting_template == "Hey"


def test_hybrid_setup_builds_children() -> None:
    strat = HybridReasoning.from_config(
        {"grounded": {"greeting_template": "Hi"}, "react": {"max_iterations": 6}}
    )
    # Children are built via their own mixin entry points.
    assert isinstance(strat._grounded, GroundedReasoning)
    assert isinstance(strat._react, ReActReasoning)
    assert strat._react.max_iterations == 6
    # The grounded child carries the nested grounded config.
    assert strat._grounded.config == strat.config.grounded


def test_hybrid_forwards_prompt_resolver_to_grounded_child() -> None:
    prompt_resolver = object()
    strat = HybridReasoning.from_config({}, prompt_resolver=prompt_resolver)
    assert strat._grounded._prompt_resolver is prompt_resolver
