"""Reasoning strategies for DynaBot."""

from typing import Any

from .base import ReasoningStrategy
from .react import ReActReasoning
from .simple import SimpleReasoning
from .wizard import WizardReasoning, WizardStageContext, WizardState
from .wizard_fsm import WizardFSM
from .wizard_loader import WizardConfigLoader, load_wizard_config

__all__ = [
    "ReasoningStrategy",
    "SimpleReasoning",
    "ReActReasoning",
    "WizardReasoning",
    "WizardStageContext",
    "WizardState",
    "WizardFSM",
    "WizardConfigLoader",
    "load_wizard_config",
    "create_reasoning_from_config",
]


def create_reasoning_from_config(config: dict[str, Any]) -> ReasoningStrategy:
    """Create reasoning strategy from configuration.

    Args:
        config: Reasoning configuration with:
            - strategy: Strategy type ('simple', 'react', 'wizard')
            - max_iterations: For ReAct, max reasoning loops (default: 5)
            - verbose: Enable debug logging for reasoning steps (default: False)
            - store_trace: Store reasoning trace in conversation metadata (default: False)
            - wizard_config: For wizard, path to wizard YAML config
            - extraction_config: For wizard, extraction configuration dict
            - strict_validation: For wizard, enforce schema validation (default: True)

    Returns:
        Configured reasoning strategy instance

    Raises:
        ValueError: If strategy type is not supported

    Example:
        ```python
        # Simple reasoning
        config = {"strategy": "simple"}
        strategy = create_reasoning_from_config(config)

        # ReAct reasoning with trace storage
        config = {
            "strategy": "react",
            "max_iterations": 5,
            "verbose": True,
            "store_trace": True
        }
        strategy = create_reasoning_from_config(config)

        # Wizard reasoning
        config = {
            "strategy": "wizard",
            "wizard_config": "wizards/onboarding.yaml",
            "extraction_config": {
                "provider": "ollama",
                "model": "qwen3-coder",
            },
            "strict_validation": True
        }
        strategy = create_reasoning_from_config(config)
        ```
    """
    strategy_type = config.get("strategy", "simple").lower()

    if strategy_type == "simple":
        return SimpleReasoning()

    elif strategy_type == "react":
        return ReActReasoning(
            max_iterations=config.get("max_iterations", 5),
            verbose=config.get("verbose", False),
            store_trace=config.get("store_trace", False),
        )

    elif strategy_type == "wizard":
        return WizardReasoning.from_config(config)

    else:
        raise ValueError(
            f"Unknown reasoning strategy: {strategy_type}. "
            f"Available strategies: simple, react, wizard"
        )
