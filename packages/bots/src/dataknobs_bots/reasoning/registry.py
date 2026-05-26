"""Strategy registry for reasoning strategy discovery and construction.

Provides a plugin registry that allows 3rd parties to register custom
reasoning strategies without modifying core DynaBot code.  Built-in
strategies are registered lazily at first access.

Usage::

    from dataknobs_bots import register_strategy
    from dataknobs_bots.reasoning import create_reasoning_from_config

    # Register a custom strategy
    register_strategy("my_strategy", MyStrategy)

    # Or register a factory function
    register_strategy("my_strategy", my_factory_fn)

    # Then use it via config
    config = {"strategy": "my_strategy", ...}
    strategy = create_reasoning_from_config(config)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from dataknobs_common.registry import PluginRegistry
from dataknobs_common.structured_config import (
    SKIP_VALIDATION,
    StructuredConfig,
    _SkipValidation,
    config_registries,
)

from .base import ReasoningStrategy

logger = logging.getLogger(__name__)

# A strategy factory is either a ReasoningStrategy subclass (which has
# ``from_config``) or a callable with the same signature.
StrategyFactory = type[ReasoningStrategy] | Callable[..., ReasoningStrategy]


def _register_builtins(
    registry: PluginRegistry[ReasoningStrategy],
) -> None:
    """Register the 5 built-in strategies via lazy imports.

    Imports are deferred to avoid circular imports and to keep
    startup cost low for consumers that only use a subset.
    """
    from .grounded import GroundedReasoning
    from .hybrid import HybridReasoning
    from .react import ReActReasoning
    from .simple import SimpleReasoning
    from .wizard import WizardReasoning

    for name, cls in [
        ("simple", SimpleReasoning),
        ("react", ReActReasoning),
        ("wizard", WizardReasoning),
        ("grounded", GroundedReasoning),
        ("hybrid", HybridReasoning),
    ]:
        registry.register(name, cls)


# Module-level singleton — configured PluginRegistry replaces
# the former StrategyRegistry class.
_registry: PluginRegistry[ReasoningStrategy] = PluginRegistry(
    "reasoning_strategies",
    validate_type=ReasoningStrategy,
    canonicalize_keys=True,
    config_key="strategy",
    config_key_default="simple",
    on_first_access=_register_builtins,
)


# ------------------------------------------------------------------
# Polymorphic-section validation resolver
# ------------------------------------------------------------------


def _resolve_reasoning_config_cls(
    raw: Mapping[str, Any],
) -> type[StructuredConfig] | _SkipValidation | None:
    """Resolve a ``reasoning`` section's dict to its strategy config class.

    The resolver registered for the ``"reasoning"`` binding in
    :data:`~dataknobs_common.structured_config.config_registries`, used by
    :meth:`StructuredConfig.validate
    <dataknobs_common.structured_config.StructuredConfig.validate>` to
    validate a raw ``reasoning`` section (and, by the base's recursion, the
    nested grounded/hybrid sub-config trees) without constructing the
    strategy.

    Delegates to the reasoning :class:`PluginRegistry` — the same registry the
    construction path uses — by reading ``CONFIG_CLS`` off the registered
    strategy class for the ``"strategy"`` discriminator (defaulting to
    ``"simple"``, the registry's own default). Holding no independent
    strategy→config-class table is the no-drift guarantee. Returns ``None``
    for an unknown strategy, which ``validate`` surfaces as a
    ``ConfigurationError``.

    Like ``memory``/``knowledge_base``, ``register_strategy`` accepts a bare
    callable factory (``StrategyFactory = type[ReasoningStrategy] |
    Callable[..., ReasoningStrategy]``). Such a strategy has no
    ``StructuredConfig`` ``CONFIG_CLS``, so the resolver returns
    :data:`SKIP_VALIDATION`: the strategy is valid and constructible but has
    no typed schema to dry-run against. All five built-ins carry a
    ``CONFIG_CLS``, so this branch is reached only by a 3rd-party
    bare-callable strategy registered out of band.
    """
    strategy = raw.get("strategy", "simple")  # registry's own default
    factory = _registry.get_factory(strategy)
    if factory is None:
        return None  # unknown strategy -> validate() raises ConfigurationError
    config_cls = getattr(factory, "CONFIG_CLS", None)
    if isinstance(config_cls, type) and issubclass(config_cls, StructuredConfig):
        return config_cls
    return SKIP_VALIDATION  # registered bare callable, no CONFIG_CLS -> skip


# Eager registration (mirroring the memory / knowledge_base resolvers and
# ``dataknobs-data``'s ``vector_store``). The package ``__init__`` imports the
# reasoning package, so ``import dataknobs_bots.reasoning`` fires this.
# ``config_registries`` is a plain ``Registry`` -> ``allow_overwrite`` (NOT
# ``override=``, which is this module's ``register_strategy`` /
# ``PluginRegistry`` param — do not cross the two) keeps re-import idempotent.
config_registries.register(
    "reasoning", _resolve_reasoning_config_cls, allow_overwrite=True
)


# ------------------------------------------------------------------
# Public API — thin wrappers around the singleton
# ------------------------------------------------------------------


def register_strategy(
    name: str,
    factory: StrategyFactory,
    *,
    override: bool = False,
) -> None:
    """Register a custom reasoning strategy.

    Args:
        name: Strategy name (used in ``reasoning.strategy`` config).
        factory: ``ReasoningStrategy`` subclass or factory callable.
        override: Replace existing registration if ``True``.

    Raises:
        OperationError: If ``name`` is already registered and
            ``override`` is ``False``.
        TypeError: If ``factory`` is not a ``ReasoningStrategy``
            subclass or callable.

    Example::

        from dataknobs_bots.reasoning.registry import register_strategy

        class MyStrategy(ReasoningStrategy):
            ...

        register_strategy("my_strategy", MyStrategy)
    """
    _registry.register(name, factory, override=override)


def get_strategy_factory(name: str) -> StrategyFactory | None:
    """Return the factory for a strategy name, or ``None``."""
    return _registry.get_factory(name)


def is_strategy_registered(name: str) -> bool:
    """Check whether a strategy name is registered."""
    return _registry.is_registered(name)


def list_strategies() -> list[str]:
    """Return sorted list of all registered strategy names."""
    return sorted(_registry.list_keys())


def get_registry() -> PluginRegistry[ReasoningStrategy]:
    """Return the module-level strategy registry singleton.

    Useful for advanced scenarios (e.g. bulk registration, testing).
    """
    return _registry
