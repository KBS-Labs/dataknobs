"""LLM-based intent classifier.

Args:
    llm: AsyncLLMProvider used for classification. May also be
        supplied per-call via ``classify(llm=...)``.
    prompt_template: Prompt template with ``{message}``,
        ``{intent_list}``, ``{extract_intents}`` placeholders.
        Defaults to :data:`DEFAULT_LLM_PROMPT_TEMPLATE`.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Any

from dataknobs_llm import LLMMessage
from dataknobs_llm.intent.defaults import DEFAULT_LLM_PROMPT_TEMPLATE
from dataknobs_llm.intent.protocol import (
    IntentClassifier,
    IntentMatchResult,
    IntentSpec,
)
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

        # Preserve caller order for prompt reproducibility: prompt
        # cache hit rates and LLM eval determinism both depend on the
        # rendered prompt being byte-stable across runs. Set iteration
        # would shuffle the intent list. Dedup conservatively, keeping
        # first occurrence — and build the extract-intents list inside
        # the SAME dedup loop so duplicate caller-supplied names don't
        # bleed into the prompt's extract list.
        seen: set[str] = set()
        ordered_names: list[str] = []
        extract_intent_names: list[str] = []
        for spec in intents:
            if spec.name in seen:
                continue
            seen.add(spec.name)
            ordered_names.append(spec.name)
            if spec.extract:
                extract_intent_names.append(spec.name)
        prompt = self._prompt_template.format(
            message=message,
            intent_list=", ".join(f'"{n}"' for n in ordered_names),
            extract_intents=", ".join(
                f'"{n}"' for n in extract_intent_names
            ),
        )
        try:
            response = await provider.complete(
                messages=[LLMMessage(role="user", content=prompt)],
            )
        except asyncio.CancelledError:
            # Never swallow cancellation — callers and the asyncio
            # event loop both need to see it.
            raise
        except Exception as exc:
            # Provider errors (network failure, auth, rate limit,
            # malformed response) are deliberately absorbed: an LLM
            # classifier is one signal among many (often in a
            # CompositeIntentClassifier chain), and an LLM outage
            # should not crash the wizard turn. Caller observes
            # `intent=None`; the warning makes the silent absorption
            # auditable.
            logger.warning(
                "LLMIntentClassifier absorbing provider error "
                "(%s); returning no-match. exc=%s",
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
            if bare in seen:
                name = bare

        if name in seen:
            spec = next(s for s in intents if s.name == name)
            # Normalize the extracted payload to the documented shape
            # (``IntentMatchResult.extracted: str | None``). Models
            # occasionally return lists/dicts/numbers; downstream
            # consumers (the wizard writes this into a schema-declared
            # ``string`` property) require ``str`` or ``None``.
            extracted_str = _coerce_extracted(extracted)
            return IntentMatchResult(
                intent=spec, extracted=extracted_str,
                rule_based=False, raw_reply=message,
            )
        return IntentMatchResult(
            intent=None, extracted=None,
            rule_based=False, raw_reply=message,
        )


def _coerce_extracted(value: Any) -> str | None:
    """Normalize an LLM-returned ``extracted`` value to ``str | None``.

    Accepts the documented shape (``str``, ``None``, or absent), and
    tolerates models that hallucinate a wrapper type by collapsing it:

    * ``None`` / missing → ``None``
    * Empty string → ``None`` (treat as "no payload")
    * ``str`` → returned as-is (with surrounding whitespace stripped)
    * ``int`` / ``float`` / ``bool`` → ``str(value)`` (the natural
      string rendering — preserves the model's intent)
    * ``list`` / ``tuple`` of one element → recurse on the element
    * Anything else (multi-element list, dict, etc.) → ``None``
      (cannot be safely stringified into a single schema field)
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _coerce_extracted(value[0])
    logger.debug(
        "LLMIntentClassifier dropping non-coercible extracted value "
        "of type %s",
        type(value).__name__,
    )
    return None
