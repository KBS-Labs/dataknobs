"""Synthesizer for the ``intent_confirm:`` stage primitive.

Pure load-time YAML transformation. ``intent_confirm:`` expands to:

* ``mode: conversation`` — first-render template, then detect_intent
  on subsequent turns
* ``response_template: <proposal_template>`` — the first-turn proposal
* ``clarification_template: <on_no_match.clarification_template>`` —
  shown on subsequent renders when no intent matched (optional)
* ``intent_detection: {classifier: keyword, intents: [...],
  per_intent_booleans: true, ...}`` — the existing detect_intent path
* ``schema: {...}`` — auto-synthesized from intents
* ``transitions: [...]`` — auto-synthesized from intent targets +
  fallback for on_no_match

Zero new runtime dispatch — every step runs through the existing
wizard machinery.
"""
from __future__ import annotations

from typing import Any

from .stage_synthesizers import (
    register_stage_synthesizer,
    validate_no_conflicting_fields,
)


class IntentConfirmSynthesizer:
    """Expand ``intent_confirm:`` into existing wizard primitives."""

    field = "intent_confirm"

    def validate(self, stage: dict[str, Any]) -> None:
        validate_no_conflicting_fields(
            stage,
            self.field,
            ["schema", "response_template", "transitions"],
        )
        block = stage[self.field]
        if not block.get("intents"):
            from dataknobs_common.exceptions import ConfigurationError

            stage_name = stage.get("name", "<unnamed>")
            raise ConfigurationError(
                f"Stage '{stage_name}': intent_confirm: declares no "
                f"intents. At least one intent (with target) is required.",
                context={"stage": stage_name},
            )

    def synthesize(self, stage: dict[str, Any]) -> None:
        block = stage[self.field]
        intents = block["intents"]
        on_no_match = block.get("on_no_match", {})

        # 1) Conversation-mode first-turn template
        stage["mode"] = "conversation"
        stage["response_template"] = block["proposal_template"]

        # 2) Optional clarification template (consulted by template-mode
        #    response branch on re-render when set)
        if on_no_match.get("clarification_template"):
            stage["clarification_template"] = on_no_match[
                "clarification_template"
            ]

        # 3) intent_detection block (consumed by classifier-backend dispatch)
        intent_detection_intents: list[dict[str, Any]] = []
        for name, intent in intents.items():
            entry: dict[str, Any] = {"id": name}
            if intent.get("keywords") is not None:
                entry["keywords"] = list(intent["keywords"])
            if intent.get("extract"):
                entry["extract"] = intent["extract"]
            intent_detection_intents.append(entry)

        # Synthesize the classifier shape. If the primitive declares
        # llm_fallback=true, expand to a composite chain (keyword first,
        # LLM second). Otherwise the plain keyword classifier.
        if block.get("llm_fallback", False):
            classifier_name = "composite"
            classifier_config: dict[str, Any] = {
                "classifiers": [
                    {"classifier": "keyword", "config": {}},
                    # llm provider injected at dispatch
                    {"classifier": "llm", "config": {}},
                ],
                "strategy": "first_match",
            }
        elif block.get("classifier"):
            classifier_name = block["classifier"]
            classifier_config = dict(block.get("classifier_config", {}))
        else:
            classifier_name = "keyword"
            classifier_config = {}

        stage["intent_detection"] = {
            "classifier": classifier_name,
            "classifier_config": classifier_config,
            "intents": intent_detection_intents,
            "per_intent_booleans": True,
            "use_default_vocabulary": True,
            "negation_filter": block.get("negation_filter", False),
        }

        # 4) Synthesized schema
        properties: dict[str, Any] = {}
        for name, intent in intents.items():
            properties[name] = {"type": "boolean"}
            if intent.get("extract"):
                properties[intent["extract"]] = {"type": "string"}
        stage["schema"] = {"type": "object", "properties": properties}

        # 5) Synthesized transitions
        transitions = [
            {
                "target": intent["target"],
                "condition": f"data.get({name!r}) == True",
            }
            for name, intent in intents.items()
        ]
        if on_no_match.get("target"):
            transitions.append({
                "target": on_no_match["target"],
                "condition": (
                    "not any(data.get(k) for k in "
                    f"{sorted(intents.keys())!r})"
                ),
            })
        stage["transitions"] = transitions


# Auto-register on module import — synthesizer is in-tree.
register_stage_synthesizer(IntentConfirmSynthesizer())
