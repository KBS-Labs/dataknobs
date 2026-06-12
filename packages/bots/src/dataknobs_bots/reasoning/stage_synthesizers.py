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

Registration semantics: re-registering the same ``field`` overwrites
the prior synthesizer (the registry calls
:meth:`Registry.register` with ``allow_overwrite=True``). This is
intentionally distinct from sibling registries like
``intent_classifier_backends``, which raise on duplicate registration
by default — stage synthesizers are typically auto-registered at
module import, and consumers often want to replace the in-tree
synthesizer with a customized one for the same field name.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.registry import Registry


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
      ``schema``, ``transitions``, etc.). The synthesizer SHOULD
      ``del stage[self.field]`` at the end so the FSM-metadata layer
      does not carry the un-expanded primitive block as a parallel
      source of truth.

    Synthesizers run during a dedicated loader phase BEFORE
    ``_validate_config`` and ``_translate_to_fsm`` — downstream
    validator and FSM build code paths see only the normalized shape.
    """

    field: str

    def validate(self, stage: dict[str, Any]) -> None: ...

    def synthesize(self, stage: dict[str, Any]) -> None: ...


stage_synthesizer_backends: Registry[StageSynthesizer] = Registry(
    name="stage_synthesizer_backends",
)
"""Registry of stage-primitive synthesizers, keyed by ``field`` name.

Structurally mirrors :data:`intent_classifier_backends`,
:data:`event_bus_backends`, and :data:`lock_backends`. Thread-safe
register / lookup / metrics for free.

Registration semantics: re-registering the same ``field`` overwrites
(see module docstring for rationale).
"""


def register_stage_synthesizer(synthesizer: StageSynthesizer) -> None:
    """Register a stage-primitive synthesizer.

    The synthesizer's ``field`` attribute is its registry key.
    Re-registering the same field overwrites the prior synthesizer.

    Args:
        synthesizer: Object satisfying the :class:`StageSynthesizer`
            protocol (``field: str`` attribute, ``synthesize(stage)``
            method, optional ``validate(stage)``).
    """
    stage_synthesizer_backends.register(
        synthesizer.field, synthesizer, allow_overwrite=True,
    )


def unregister_stage_synthesizer(field: str) -> None:
    """Remove a registered synthesizer (test/cleanup utility).

    No-op when ``field`` is not registered.
    """
    if stage_synthesizer_backends.has(field):
        stage_synthesizer_backends.unregister(field)


def iter_stage_synthesizers() -> Mapping[str, StageSynthesizer]:
    """Read-only snapshot of the registered synthesizers."""
    return {
        key: stage_synthesizer_backends.get(key)
        for key in stage_synthesizer_backends.list_keys()
    }


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
