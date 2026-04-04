"""Strategy registry for reasoning strategy discovery and construction.

Provides a plugin registry that allows 3rd parties to register custom
reasoning strategies without modifying core DynaBot code.  Built-in
strategies are registered lazily at first access.

Usage::

    from dataknobs_bots.reasoning.registry import register_strategy

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
import threading
from collections.abc import Callable
from typing import Any

from .base import ReasoningStrategy

logger = logging.getLogger(__name__)

# A strategy factory is either a ReasoningStrategy subclass (which has
# ``from_config``) or a callable with the same signature.
StrategyFactory = type[ReasoningStrategy] | Callable[..., ReasoningStrategy]


class StrategyRegistry:
    """Registry mapping strategy names to their factories.

    Unlike ``PluginRegistry`` (which caches singleton instances and has
    a ``(key, config)`` factory signature), ``StrategyRegistry`` creates
    a fresh instance per call — strategies are per-bot, not singletons.
    """

    def __init__(self) -> None:
        self._factories: dict[str, StrategyFactory] = {}
        self._initialized = False
        self._lock = threading.RLock()

    def _ensure_builtins(self) -> None:
        """Register built-in strategies on first access."""
        with self._lock:
            if self._initialized:
                return
            self._initialized = True
            _register_builtins(self)

    def register(
        self,
        name: str,
        factory: StrategyFactory,
        *,
        override: bool = False,
    ) -> None:
        """Register a strategy factory under the given name.

        Args:
            name: Strategy name (used in config ``strategy`` field).
            factory: A ``ReasoningStrategy`` subclass or callable
                ``(config, **kwargs) -> ReasoningStrategy``.
            override: If ``True``, silently replace an existing
                registration.  Otherwise raise ``ValueError``.

        Raises:
            ValueError: If ``name`` is already registered and
                ``override`` is ``False``.
        """
        self._ensure_builtins()
        canonical = name.lower()
        with self._lock:
            if canonical in self._factories and not override:
                raise ValueError(
                    f"Strategy '{canonical}' is already registered. "
                    f"Use override=True to replace it."
                )
            self._factories[canonical] = factory
        logger.debug("Registered strategy '%s'", canonical)

    def create(
        self,
        config: dict[str, Any],
        **kwargs: Any,
    ) -> ReasoningStrategy:
        """Create a strategy instance from a config dict.

        Extracts ``config["strategy"]`` (default ``"simple"``), looks up
        the factory, and calls it.  For ``ReasoningStrategy`` subclasses
        the factory is ``cls.from_config(config, **kwargs)``.  For plain
        callables the factory is called as ``factory(config, **kwargs)``.

        Args:
            config: Strategy configuration dict (must contain
                ``strategy`` key).
            **kwargs: Forwarded to the factory (e.g. ``knowledge_base``).

        Returns:
            Configured strategy instance.

        Raises:
            ValueError: If the strategy name is not registered.
        """
        self._ensure_builtins()
        name = config.get("strategy", "simple").lower()
        factory = self._factories.get(name)
        if factory is None:
            available = ", ".join(sorted(self._factories))
            raise ValueError(
                f"Unknown reasoning strategy: '{name}'. "
                f"Available strategies: {available}"
            )

        if isinstance(factory, type) and issubclass(factory, ReasoningStrategy):
            return factory.from_config(config, **kwargs)
        return factory(config, **kwargs)

    def get_factory(self, name: str) -> StrategyFactory | None:
        """Return the factory for a strategy name, or ``None``."""
        self._ensure_builtins()
        return self._factories.get(name.lower())

    def is_registered(self, name: str) -> bool:
        """Check whether a strategy name is registered."""
        self._ensure_builtins()
        return name.lower() in self._factories

    def list_keys(self) -> list[str]:
        """Return sorted list of registered strategy names."""
        self._ensure_builtins()
        return sorted(self._factories)


def _register_builtins(registry: StrategyRegistry) -> None:
    """Register the 5 built-in strategies via lazy imports.

    Imports are deferred to avoid circular imports and to keep
    startup cost low for consumers that only use a subset.
    """
    from .grounded import GroundedReasoning
    from .hybrid import HybridReasoning
    from .react import ReActReasoning
    from .simple import SimpleReasoning
    from .wizard import WizardReasoning

    # _initialized is already True (set by caller), so register()
    # won't re-enter _ensure_builtins.
    registry.register("simple", SimpleReasoning)
    registry.register("react", ReActReasoning)
    registry.register("wizard", WizardReasoning)
    registry.register("grounded", GroundedReasoning)
    registry.register("hybrid", HybridReasoning)


# Module-level singleton
_registry = StrategyRegistry()


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


def get_registry() -> StrategyRegistry:
    """Return the module-level strategy registry singleton.

    Useful for advanced scenarios (e.g. bulk registration, testing).
    """
    return _registry
