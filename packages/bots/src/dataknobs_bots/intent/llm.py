"""LLM-based intent classifier.

Args:
    llm: AsyncLLMProvider used for classification. May also be
        supplied per-call via ``classify(llm=...)``.
    prompt_template: Prompt template with ``{message}``,
        ``{intent_list}``, ``{extract_intents}`` placeholders.
        Defaults to :data:`DEFAULT_LLM_PROMPT_TEMPLATE`.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from dataknobs_bots.intent.defaults import DEFAULT_LLM_PROMPT_TEMPLATE
from dataknobs_bots.intent.protocol import (
    IntentClassifier,
    IntentMatchResult,
    IntentSpec,
)
from dataknobs_llm import LLMMessage
from dataknobs_llm.llm.base import AsyncLLMProvider

logger = logging.getLogger(__name__)


class LLMIntentClassifier(IntentClassifier):
    def __init__(
        self,
        *,
        llm: AsyncLLMProvider | None = None,
        prompt_template: str | None = None,
    ) -> None:
        self._llm = llm
        self._prompt_template = (
            prompt_template or DEFAULT_LLM_PROMPT_TEMPLATE
        )

    async def classify(
        self,
        message: str,
        intents: Sequence[IntentSpec],
        *,
        llm: AsyncLLMProvider | None = None,
        **_: Any,
    ) -> IntentMatchResult:
        provider = llm or self._llm
        if provider is None:
            return IntentMatchResult(
                intent=None, extracted=None,
                rule_based=False, raw_reply=message,
            )

        valid_names = {s.name for s in intents}
        extract_intent_names = sorted(
            {s.name for s in intents if s.extract},
        )
        prompt = self._prompt_template.format(
            message=message,
            intent_list=", ".join(f'"{n}"' for n in valid_names),
            extract_intents=", ".join(
                f'"{n}"' for n in extract_intent_names
            ),
        )
        try:
            response = await provider.complete(
                messages=[LLMMessage(role="user", content=prompt)],
            )
        except Exception as exc:
            logger.warning(
                "LLM intent classification call failed (%s): %s",
                type(exc).__name__, exc,
            )
            return IntentMatchResult(
                intent=None, extracted=None,
                rule_based=False, raw_reply=message,
            )

        if not response or not response.content:
            return IntentMatchResult(
                intent=None, extracted=None,
                rule_based=False, raw_reply=message,
            )

        raw = response.content.strip()
        name: str | None = None
        extracted: Any = None

        # Preferred: parse the JSON `{"intent": ..., "extracted": ...}`
        # shape produced by DEFAULT_LLM_PROMPT_TEMPLATE.
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                name = data.get("intent")
                extracted = data.get("extracted")
        except json.JSONDecodeError:
            # Back-compat: accept a bare intent ID (the shape produced
            # by the legacy wizard intent_detection: {method: llm} prompt).
            bare = raw.strip().strip('"').strip("'")
            if bare in valid_names:
                name = bare

        if name in valid_names:
            spec = next(s for s in intents if s.name == name)
            return IntentMatchResult(
                intent=spec, extracted=extracted,
                rule_based=False, raw_reply=message,
            )
        return IntentMatchResult(
            intent=None, extracted=None,
            rule_based=False, raw_reply=message,
        )
