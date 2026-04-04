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
from collections.abc import Callable

from dataknobs_common.registry import PluginRegistry

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
# the former StrategyRegistry class (consumer-gaps plan Item 65).
_registry: PluginRegistry[ReasoningStrategy] = PluginRegistry(
    "reasoning_strategies",
    validate_type=ReasoningStrategy,
    canonicalize_keys=True,
    config_key="strategy",
    config_key_default="simple",
    on_first_access=_register_builtins,
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
    return _registry.list_keys()


def get_registry() -> PluginRegistry[ReasoningStrategy]:
    """Return the module-level strategy registry singleton.

    Useful for advanced scenarios (e.g. bulk registration, testing).
    """
    return _registry
