"""Testing utilities for dataknobs-bots.

This module provides capture-replay infrastructure for testing bot interactions
with grounded LLM responses captured from real provider runs.

Key components:
- ``CaptureReplay``: Loads a capture JSON file and creates pre-loaded
  EchoProviders for deterministic replay of recorded conversations.
- ``inject_providers``: Injects LLM providers into a DynaBot instance,
  replacing both the main LLM and the extraction LLM used by wizard reasoning.

Example:
    ```python
    from dataknobs_bots.testing import CaptureReplay

    # Load a previously captured wizard conversation
    replay = CaptureReplay.from_file("captures/configbot_basic.json")

    # Create bot and inject captured responses
    bot = DynaBot(config=config, llm=real_provider, ...)
    replay.inject_into_bot(bot)

    # Run the conversation — bot uses captured LLM responses
    for turn in replay.turns:
        if turn["type"] == "greet":
            await bot.greet(context)
        else:
            await bot.chat(turn["user_message"], context)
    ```
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dataknobs_llm import EchoProvider
from dataknobs_llm.llm.base import AsyncLLMProvider, LLMResponse
from dataknobs_llm.testing import llm_response_from_dict

logger = logging.getLogger(__name__)


def inject_providers(
    bot: Any,
    main_provider: AsyncLLMProvider | None = None,
    extraction_provider: AsyncLLMProvider | None = None,
    **role_providers: AsyncLLMProvider,
) -> None:
    """Inject LLM providers into a DynaBot instance for testing.

    For ``main_provider``, directly replaces ``bot.llm`` (the ``"main"``
    role is always served from this attribute, not the registry catalog).

    For ``extraction_provider`` and ``**role_providers``, updates both the
    registry catalog and the actual subsystem wiring via ``set_provider()``.

    **Lifecycle contract:** Injected providers are NOT owned by the bot.
    The caller retains responsibility for closing both the injected
    provider and any displaced provider.  This follows the
    originator-owns-lifecycle principle — ``bot.close()`` will not close
    providers it did not create.

    If ``bot`` does not implement ``register_provider``, catalog
    registration is skipped; only subsystem wiring via ``set_provider()``
    is performed.

    Args:
        bot: A DynaBot instance (or any object with ``llm`` and
            ``reasoning_strategy`` attributes).
        main_provider: Provider to use for main LLM calls. If None,
            the existing provider is kept.
        extraction_provider: Provider to use for schema extraction.
            If None, the existing provider is kept.
        **role_providers: Additional providers keyed by role name
            (e.g. ``memory_embedding=echo_provider``).  Each provider
            is registered in the catalog AND wired into the owning
            subsystem via ``set_provider()``.

    Example:
        ```python
        from dataknobs_llm import EchoProvider
        from dataknobs_bots.testing import inject_providers

        main = EchoProvider()
        extraction = EchoProvider()
        inject_providers(bot, main, extraction)
        ```
    """
    if main_provider is not None:
        bot.llm = main_provider

    if extraction_provider is not None:
        from dataknobs_bots.bot.base import PROVIDER_ROLE_EXTRACTION

        # Update the registry entry
        if hasattr(bot, "register_provider"):
            bot.register_provider(PROVIDER_ROLE_EXTRACTION, extraction_provider)

        # Also update the actual extractor so subsystem calls use it
        strategy = getattr(bot, "reasoning_strategy", None)
        if strategy is None:
            logger.warning(
                "Bot has no reasoning_strategy — skipping extraction provider injection"
            )
        elif hasattr(strategy, "set_provider"):
            strategy.set_provider(PROVIDER_ROLE_EXTRACTION, extraction_provider)
        else:
            # Fallback for strategies without set_provider (e.g. test stubs)
            extractor = getattr(strategy, "_extractor", None)
            if extractor is None:
                logger.warning(
                    "Reasoning strategy has no _extractor — "
                    "skipping extraction provider injection"
                )
            else:
                extractor.provider = extraction_provider

    # Wire role-based providers into catalog AND subsystems
    for role, provider in role_providers.items():
        if hasattr(bot, "register_provider"):
            bot.register_provider(role, provider)

        # Wire into the actual subsystem that owns this role
        _wire_role_provider(bot, role, provider)


def _wire_role_provider(bot: Any, role: str, provider: AsyncLLMProvider) -> None:
    """Wire a role provider into the subsystem that owns it.

    Iterates over the bot's subsystems (memory, knowledge_base,
    reasoning_strategy) and calls ``set_provider(role, provider)``
    on the first one that claims the role.

    Args:
        bot: DynaBot instance (or compatible stub).
        role: Provider role name.
        provider: Replacement provider instance.
    """
    subsystems = [
        getattr(bot, "memory", None),
        getattr(bot, "knowledge_base", None),
        getattr(bot, "reasoning_strategy", None),
    ]
    for subsystem in subsystems:
        if subsystem is not None and hasattr(subsystem, "set_provider"):
            if subsystem.set_provider(role, provider):
                return
    logger.debug(
        "Role %r registered in catalog but no subsystem claimed it", role
    )


class CaptureReplay:
    """Loads a capture JSON file and creates pre-loaded EchoProviders.

    Capture files contain serialized LLM request/response pairs from real
    provider runs, organized by turn. CaptureReplay deserializes these and
    creates EchoProviders queued with the correct responses, enabling
    deterministic replay of captured conversations.

    Attributes:
        metadata: Capture session metadata (description, model info, timestamps)
        turns: List of turn dicts with wizard state, user messages, bot responses
        format_version: Capture file format version

    Example:
        ```python
        replay = CaptureReplay.from_file("captures/quiz_basic.json")

        # Get providers for replay
        main = replay.main_provider()
        extraction = replay.extraction_provider()

        # Or inject directly into a bot
        replay.inject_into_bot(bot)
        ```
    """

    def __init__(
        self,
        data: dict[str, Any],
    ) -> None:
        self.format_version: str = data.get("format_version", "1.0")
        self.metadata: dict[str, Any] = data.get("metadata", {})
        self.turns: list[dict[str, Any]] = data.get("turns", [])
        self._data = data

        # Pre-separate LLM calls by role for provider creation
        self._main_responses: list[LLMResponse] = []
        self._extraction_responses: list[LLMResponse] = []
        self._parse_calls()

    def _parse_calls(self) -> None:
        """Parse all LLM calls from turns and separate by role."""
        for turn in self.turns:
            for call in turn.get("llm_calls", []):
                response = llm_response_from_dict(call["response"])
                role = call.get("role", "main")
                if role == "extraction":
                    self._extraction_responses.append(response)
                else:
                    self._main_responses.append(response)

    @classmethod
    def from_file(cls, path: str | Path) -> CaptureReplay:
        """Load a capture replay from a JSON file.

        Args:
            path: Path to the capture JSON file

        Returns:
            CaptureReplay instance

        Raises:
            FileNotFoundError: If the file does not exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        with open(path) as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaptureReplay:
        """Create a CaptureReplay from a dict (e.g., already-parsed JSON).

        Args:
            data: Capture data dict

        Returns:
            CaptureReplay instance
        """
        return cls(data)

    def main_provider(self) -> EchoProvider:
        """Create an EchoProvider queued with main-role responses.

        Returns:
            EchoProvider with responses in capture order
        """
        provider = EchoProvider({"provider": "echo", "model": "capture-replay"})
        if self._main_responses:
            provider.set_responses(self._main_responses)
        return provider

    def extraction_provider(self) -> EchoProvider:
        """Create an EchoProvider queued with extraction-role responses.

        Returns:
            EchoProvider with responses in capture order
        """
        provider = EchoProvider({"provider": "echo", "model": "capture-replay"})
        if self._extraction_responses:
            provider.set_responses(self._extraction_responses)
        return provider

    def inject_into_bot(self, bot: Any) -> None:
        """Replace providers on a DynaBot with capture-replay EchoProviders.

        Creates main and extraction EchoProviders from the captured data
        and injects them into the bot using ``inject_providers``.

        Args:
            bot: A DynaBot instance
        """
        inject_providers(
            bot,
            main_provider=self.main_provider(),
            extraction_provider=self.extraction_provider() if self._extraction_responses else None,
        )
