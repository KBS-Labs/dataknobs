"""The universal ``greeting_template`` contract, driven over the registry.

:class:`~dataknobs_bots.reasoning.base.ReasoningStrategy` declares
``greeting_template`` universal in its own class docstring:

    *"All strategies support an optional ``greeting_template`` … an override
    is expected to honour the field rather than discard it, which is what
    makes it universal."*

Nothing enforced that sentence.  The family has no shared config base, so the
field is re-declared by every strategy config and re-bound by every strategy,
and a strategy that skips either half fails **silently** — an undeclared key is
dropped by ``StructuredConfig``'s ``_UNKNOWN_KEYS = "ignore"`` policy rather
than reported.  The existing parity guard
(``assert_structured_config_consumer``) does not cover it: it compares a
config class against a constructor signature, and for a mixin adopter that
signature is the mixin's variadic one, so the comparison is structurally
empty.

These tests are that enforcement, in two tiers:

* **Tier 1 — construction.**  ``from_config({"greeting_template": …})``
  constructs, and the value is readable back off the instance.  This is the
  tier that matters, because *both* known failures are construction-time: a
  config class that does not declare the field drops it silently, and a
  directly-subclassed strategy that does not accept the constructor keyword
  raises ``TypeError``.
* **Tier 2 — effect.**  ``greet()`` renders it, observed across the
  config-dict → ``from_config`` → ``greet()`` boundary rather than asserted
  against a dataclass's shape.

Tier 2 has one legitimate exception, and the tree ships it: a strategy whose
``greet()`` deliberately raises (``ErrorRaisingStrategy``) discards the field
by design.  What survives that exception — and what tier 1 enforces for every
strategy without exception — is that the field must still be *accepted* at
construction.

The tiers are driven over :func:`list_strategies`, not over a hard-coded list,
so a strategy a consumer registers at import time in their own session is held
to the same contract.  Anchoring to the registry rather than to a field list
is deliberate: the defect this guards against was produced by deriving a
config envelope from *what each strategy's* ``from_config`` *read*, which
omits a field whose contract lives in a base-class docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from dataknobs_common.structured_config import StructuredConfig

from dataknobs_bots.reasoning import (
    ReasoningStrategy,
    SimpleReasoning,
    get_registry,
    get_strategy_factory,
    list_strategies,
    register_strategy,
)
from dataknobs_bots.testing import (
    ErrorRaisingStrategy,
    StubManager,
    WizardConfigBuilder,
)


GREETING = "Hello {{ who }}!"
RENDERED = "Hello Ada!"


def _minimal_wizard_config() -> dict[str, Any]:
    """Smallest wizard definition that loads — one stage, start and end."""
    return (
        WizardConfigBuilder("greeting-contract")
        .stage("only", is_start=True, is_end=True, prompt="Hi.")
        .build()
    )


def _required_config(name: str) -> dict[str, Any]:
    """Config keys a strategy requires beyond the greeting itself.

    Keyed by registered name.  A strategy absent from this map is expected to
    construct from the greeting alone; if a future strategy adds a required
    field, tier 1 fails loudly at construction and the fix is to add its
    minimum here — which is the intended prompt to think about whether the new
    strategy honours the universal field.
    """
    if name == "wizard":
        return {"wizard_config": _minimal_wizard_config()}
    return {}


def assert_accepts_universal_greeting(
    factory: Any,
    *,
    extra_config: dict[str, Any] | None = None,
) -> Any:
    """Tier 1 — the strategy accepts the universal greeting at construction.

    Returns the constructed strategy so a caller can drive tier 2 against the
    same instance.

    Reads the public :attr:`ReasoningStrategy.greeting_template` property,
    which resolves both routes — the typed config on a mixin adopter, the
    constructor keyword on a directly-subclassed strategy.  This assertion
    was written against the private ``_greeting_template`` while the guard
    ran ahead of the consolidation it protects; that attribute meant
    different things on the two populations and agreed only by accident of
    five separate bindings.  Those bindings are gone and the property is
    now the single read, so the compromise is discharged, not carried.
    """
    config: dict[str, Any] = {"greeting_template": GREETING}
    config.update(extra_config or {})

    strategy = factory.from_config(config)

    bound = strategy.greeting_template
    assert bound == GREETING, (
        f"{factory!r} did not accept the universal greeting_template: "
        f"expected {GREETING!r}, got {bound!r}. Either its config class does "
        f"not declare the field (so from_dict dropped it), or nothing binds "
        f"it onto the instance."
    )
    return strategy


# ------------------------------------------------------------------
# The shipped construct that fails the contract today
# ------------------------------------------------------------------


class TestShippedStrategyAcceptsTheKeyword:
    """``ErrorRaisingStrategy`` is buildable from config.

    It is a direct ``ReasoningStrategy`` subclass, so it inherits the base
    ``from_config``, which calls ``cls(greeting_template=…)``.  A constructor
    that does not accept that keyword makes the inherited classmethod raise
    ``TypeError`` — so a strategy shipped for consumers to build from config
    has to accept the base class's own universal field, whether or not it
    does anything with it.  This one does not: its ``greet()`` raises by
    design, which is the whole construct.
    """

    def test_from_config_accepts_the_universal_greeting(self) -> None:
        """Tier 1 for the shipped error strategy.

        Construction only.  Its ``greet()`` raises by design, so asserting a
        rendered greeting here would assert the opposite of what the construct
        is for.
        """
        assert_accepts_universal_greeting(ErrorRaisingStrategy)

    def test_the_configured_error_is_still_what_it_raises(self) -> None:
        """Accepting the keyword does not disturb what the construct is for."""
        boom = RuntimeError("boom")
        strategy = ErrorRaisingStrategy(boom)
        assert strategy._error is boom

    async def test_greet_still_raises_when_a_greeting_is_configured(self) -> None:
        """The greet() override still raises — that is the exception tier 2 skips."""
        strategy = assert_accepts_universal_greeting(ErrorRaisingStrategy)
        with pytest.raises(ValueError, match="test error"):
            await strategy.greet(StubManager(), None, initial_context={"who": "Ada"})


# ------------------------------------------------------------------
# The driver — every registered strategy, both tiers
# ------------------------------------------------------------------


class TestEveryRegisteredStrategyHonoursTheUniversalField:
    """The recurrence guard.

    This passes today for all five built-ins.  Its value is not what it
    reports now — it is that it fails on strategy six, and on any edit to one
    of the five that stops honouring the field.
    """

    @pytest.mark.parametrize("name", list_strategies())
    def test_accepts_it_at_construction(self, name: str) -> None:
        """Tier 1 — over what is registered, not over a hard-coded list."""
        factory = get_strategy_factory(name)
        assert factory is not None, f"{name} is listed but has no factory"
        assert_accepts_universal_greeting(factory, extra_config=_required_config(name))

    @pytest.mark.parametrize("name", list_strategies())
    async def test_renders_it_from_greet(self, name: str) -> None:
        """Tier 2 — the effect, across the config-dict → from_config → greet boundary.

        Covers the three strategies the existing greeting suite's
        config-factory round-trip never reached (grounded, hybrid, wizard);
        for the wizard the strategy-level template is the start stage's
        default, so it renders on the greeting turn like any other.
        """
        factory = get_strategy_factory(name)
        assert factory is not None
        strategy = assert_accepts_universal_greeting(factory, extra_config=_required_config(name))

        result = await strategy.greet(StubManager(), None, initial_context={"who": "Ada"})

        assert result is not None, f"{name}.greet() returned None despite a template"
        assert result.content == RENDERED


# ------------------------------------------------------------------
# The counterfactuals — proof the guard is not vacuous
# ------------------------------------------------------------------


class TestTheGuardCatchesEachWayOfFailingTheContract:
    """Three ways to fail the contract, and the guard rejects all three.

    Without these, tier 1 passing over five compliant strategies would be
    equally consistent with a guard that asserts nothing.
    """

    def test_a_config_class_that_omits_the_field_drops_it_silently(self) -> None:
        """The defect the wizard's config actually had.

        Config class does not declare the field and nothing binds it, so
        ``from_dict``'s ``"ignore"`` policy drops the key: construction
        succeeds, ``greet()`` returns ``None``, and nothing reports it.  This
        is the shape that survived five months.
        """

        @dataclass(frozen=True)
        class ForgetfulConfig(StructuredConfig):
            pass

        class SilentlyDroppingReasoning(SimpleReasoning):
            CONFIG_CLS = ForgetfulConfig

        # The defect itself: it constructs, and the greeting is gone.
        dropped = SilentlyDroppingReasoning.from_config({"greeting_template": GREETING})
        assert dropped.greeting_template is None

        with pytest.raises(AssertionError, match="did not accept the universal"):
            assert_accepts_universal_greeting(SilentlyDroppingReasoning)

    def test_a_config_class_that_omits_the_field_while_binding_it(self) -> None:
        """The louder variant — the binding survives the declaration's removal.

        The parity guard that exists passes on a config class declaring
        nothing at all, so this shape is not hypothetical; it fails at
        construction rather than silently, but it is still a strategy whose
        config cannot carry the universal field.
        """

        @dataclass(frozen=True)
        class ForgetfulConfig(StructuredConfig):
            pass

        class UnboundReasoning(SimpleReasoning):
            CONFIG_CLS = ForgetfulConfig

            def _setup(self) -> None:
                # Declared here rather than inherited so this counterfactual
                # pins the shape it names, and does not turn red merely
                # because a real strategy's binding changed.
                self._greeting_template = self.config.greeting_template

        with pytest.raises(AttributeError, match="greeting_template"):
            assert_accepts_universal_greeting(UnboundReasoning)

    def test_a_direct_subclass_that_omits_the_constructor_keyword(self) -> None:
        """The second population — the one ``CUSTOM_STRATEGIES.md`` teaches.

        A directly-subclassed strategy inherits the base ``from_config``,
        which calls ``cls(greeting_template=…)``.  Omitting the keyword makes
        that inherited classmethod raise — which is exactly what the shipped
        error strategy did until it accepted the keyword.
        """

        class ForgetfulStrategy(ReasoningStrategy):
            def __init__(self) -> None:
                super().__init__()

            async def generate(
                self,
                manager: Any,
                llm: Any,
                tools: list[Any] | None = None,
                **kwargs: Any,
            ) -> Any:
                return None

        with pytest.raises(TypeError, match="greeting_template"):
            assert_accepts_universal_greeting(ForgetfulStrategy)


# ------------------------------------------------------------------
# The guard reaches consumer strategies, not only the built-ins
# ------------------------------------------------------------------


@pytest.fixture()
def registered_consumer_strategy() -> Any:
    """Register a consumer-style strategy, then remove it again.

    The driver above parametrizes over :func:`list_strategies` at collection
    time, so a strategy a consumer registers at import time in their own
    session is picked up by it.  This fixture demonstrates that reach without
    depending on collection order, and unregisters so the module-level
    registry is left as it was found.
    """

    class ConsumerStrategy(ReasoningStrategy):
        """The three-site pattern ``CUSTOM_STRATEGIES.md`` teaches."""

        def __init__(self, *, greeting_template: str | None = None) -> None:
            super().__init__(greeting_template=greeting_template)

        async def generate(
            self,
            manager: Any,
            llm: Any,
            tools: list[Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            return None

    name = "_greeting_contract_consumer"
    register_strategy(name, ConsumerStrategy)
    try:
        yield name
    finally:
        get_registry().unregister(name)
        # A leak here would silently widen every other module's view of the
        # registry, so the restoration is asserted rather than assumed.
        assert name not in list_strategies()


class TestTheGuardReachesConsumerStrategies:
    """A registry-driven guard covers strategies dataknobs never sees."""

    def test_a_registered_consumer_strategy_is_held_to_the_contract(
        self, registered_consumer_strategy: str
    ) -> None:
        name = registered_consumer_strategy
        assert name in list_strategies()

        factory = get_strategy_factory(name)
        assert factory is not None
        assert_accepts_universal_greeting(factory)

    async def test_it_renders_the_greeting_too(self, registered_consumer_strategy: str) -> None:
        """Also the property's fallback branch, end to end.

        This strategy carries no typed config, so ``greet()`` reaches its
        template only through the constructor-keyword half of
        :attr:`ReasoningStrategy.greeting_template`.  Before that property
        existed the assertion was weaker — ``greet()`` read the attribute
        directly, so nothing here could have distinguished the two routes.
        """
        factory = get_strategy_factory(registered_consumer_strategy)
        assert factory is not None
        strategy = assert_accepts_universal_greeting(factory)

        result = await strategy.greet(StubManager(), None, initial_context={"who": "Ada"})

        assert result is not None
        assert result.content == RENDERED


# ------------------------------------------------------------------
# The property that resolves the two routes
# ------------------------------------------------------------------


class TestTheStrategyResolvesBothRoutesToOneField:
    """``ReasoningStrategy.greeting_template`` is the single read.

    Two routes supply the field and one property resolves them, which is
    what let the five per-strategy bindings be deleted.  Tested directly
    because the guard above exercises each route in isolation and never
    the choice between them.
    """

    def test_the_typed_config_route(self) -> None:
        strategy = SimpleReasoning.from_config({"greeting_template": GREETING})
        assert strategy.greeting_template == GREETING

    def test_the_constructor_keyword_route(self) -> None:
        class Direct(ReasoningStrategy):
            async def generate(
                self,
                manager: Any,
                llm: Any,
                tools: list[Any] | None = None,
                **kwargs: Any,
            ) -> Any:
                return None

        assert Direct(greeting_template=GREETING).greeting_template == GREETING

    def test_neither_route_supplied_reads_none(self) -> None:
        assert SimpleReasoning.from_config({}).greeting_template is None

    def test_a_configured_empty_template_is_not_treated_as_absent(self) -> None:
        """Why the property tests ``is not None`` rather than truthiness.

        An empty template is a configured value: it renders to an empty
        greeting, which is what it did before the property existed. Falling
        back on falsiness would silently substitute a different route's
        value for one the config set deliberately — and for every value
        other than ``""`` the two spellings agree, so nothing else would
        have caught the difference.
        """
        strategy = SimpleReasoning.from_config({"greeting_template": ""})
        assert strategy.greeting_template == ""
