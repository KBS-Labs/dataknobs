"""Stage-synthesizer registry for wizard stage primitives.

Lets dataknobs and consumers ship higher-level stage primitives that
expand into existing wizard machinery at load time, with zero new
runtime dispatch. The wizard loader iterates the registry during a
dedicated synthesis phase after validation and before FSM translation.

Reference adopter: :class:`IntentConfirmSynthesizer`
(``wizard_intent_confirm.py``), which expands ``intent_confirm:``
into ``mode: conversation`` + ``response_template`` +
``intent_detection`` + ``schema`` + ``transitions``.

Consumer extensibility::

    from dataknobs_bots.reasoning.stage_synthesizers import (
        StageSynthesizer,
        register_stage_synthesizer,
        validate_no_conflicting_fields,
    )

    class VendorSelectSynthesizer:
        field = "vendor_select"

        def validate(self, stage):
            validate_no_conflicting_fields(
                stage, "vendor_select",
                ["schema", "response_template", "transitions"],
            )

        def synthesize(self, stage):
            block = stage["vendor_select"]
            # ... expand into wizard primitives in place ...

    register_stage_synthesizer(VendorSelectSynthesizer())
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from dataknobs_common.exceptions import ConfigurationError


@runtime_checkable
class StageSynthesizer(Protocol):
    """Protocol for a stage primitive's load-time synthesizer.

    A synthesizer claims a stage field name (``field``) and provides
    two methods:

    * ``validate(stage)`` — raise ``ConfigurationError`` for any
      load-time invariant violation (e.g. co-existence with
      conflicting fields). **Optional**: synthesizers may omit
      ``validate`` if they have no load-time invariants beyond what
      ``synthesize`` itself enforces.
    * ``synthesize(stage)`` — mutate the stage dict IN PLACE,
      expanding the primitive's declarative block into existing
      wizard primitives (``response_template``, ``intent_detection``,
      ``schema``, ``transitions``, etc.).

    Synthesizers run during a dedicated loader phase BEFORE
    ``_validate_config`` and ``_translate_to_fsm`` — downstream
    validator and FSM build code paths see only the normalized shape.
    """

    field: str

    def validate(self, stage: dict[str, Any]) -> None: ...

    def synthesize(self, stage: dict[str, Any]) -> None: ...


_STAGE_SYNTHESIZERS: dict[str, Any] = {}


def register_stage_synthesizer(synthesizer: Any) -> None:
    """Register a stage-primitive synthesizer.

    Args:
        synthesizer: Object with a ``field: str`` attribute and a
            ``synthesize(stage)`` method. May optionally provide a
            ``validate(stage)`` method.
    """
    _STAGE_SYNTHESIZERS[synthesizer.field] = synthesizer


def unregister_stage_synthesizer(field: str) -> None:
    """Remove a registered synthesizer (test/cleanup utility)."""
    _STAGE_SYNTHESIZERS.pop(field, None)


def iter_stage_synthesizers() -> Mapping[str, Any]:
    """Read-only view of the registered synthesizers."""
    return dict(_STAGE_SYNTHESIZERS)


def validate_no_conflicting_fields(
    stage: Mapping[str, Any],
    primitive_field: str,
    conflicting_fields: list[str],
) -> None:
    """Reusable load-time validation: primitive vs conflicting fields.

    Raises :class:`ConfigurationError` if the stage declares
    ``primitive_field`` AND any of ``conflicting_fields``. Provides
    a consistent error message every primitive synthesizer can use.

    Args:
        stage: Stage configuration dict.
        primitive_field: The synthesizer's declared field
            (e.g. ``"intent_confirm"``).
        conflicting_fields: Fields whose presence indicates a
            collision with the primitive's responsibility (typically
            ``["schema", "response_template", "transitions"]`` for
            synthesizers that auto-generate those).
    """
    if primitive_field not in stage:
        return
    collisions = [k for k in conflicting_fields if stage.get(k) is not None]
    if collisions:
        stage_name = stage.get("name", "<unnamed>")
        raise ConfigurationError(
            f"Stage '{stage_name}' declares '{primitive_field}:' alongside "
            f"{collisions}. '{primitive_field}' is the source of truth for "
            f"this stage; remove the conflicting block(s) or replace "
            f"the '{primitive_field}' with hand-rolled equivalents.",
            context={
                "stage": stage_name,
                "primitive_field": primitive_field,
                "collisions": collisions,
            },
        )
