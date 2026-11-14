# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""An authored vocabulary's entities with their variants admitted as aliases.

:func:`~dataknobs_common.text.admit` decides which variants of which forms are
kept; this module is where that decision meets :class:`Entity`. Two entry
points, one of which invokes the other:

- :func:`with_variants`, for a caller building a
  :class:`~dataknobs_common.ontology.MappingEntitySource` in code;
- :func:`apply_variants`, which every door binding an authored vocabulary
  calls with the rules a document's ``variants:`` section names and the
  door's own fold. There are three such doors -- the two module-level loaders
  and ``dataknobs_data``'s ``OntologyRegistry`` -- and one function between
  them is what keeps them from disagreeing about the fold, the scope, or what
  is logged.

**The scope a collision is counted over is one vocabulary.** Every form every
entity in the mapping declares is held, so a variant of one entity meeting
another entity's id, name or alias is refused; a form in some other
vocabulary is not seen.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from dataknobs_common.text import DEFAULT_RULES, Declared, Variants, admit, default_normalizer

if TYPE_CHECKING:
    from dataknobs_common.ontology.model import Entity
    from dataknobs_common.ontology.values import OntologyParts
    from dataknobs_common.text import Rule

__all__ = ["apply_variants", "with_variants"]

logger = logging.getLogger(__name__)


def with_variants(
    entities: Mapping[str, Entity[str]],
    rules: Sequence[Rule] = DEFAULT_RULES,
    *,
    normalizer: Callable[[str], str] | None = None,
) -> tuple[dict[str, Entity[str]], Variants[str]]:
    """The entities with every admitted variant appended to their aliases, and the report.

    Variants are made from each entity's name and aliases, with the entity's
    ``metadata`` as the context a rule reads. An entity's id is held -- a
    variant meeting another entity's id is refused -- but is not varied: it
    is a key, not a form anyone wrote to be read.

    The mapping passed in is not changed. An entity with nothing admitted is
    returned as it was; one with variants is a copy.

    Args:
        entities: The vocabulary, keyed by entity id. The whole scope a
            collision is counted over.
        rules: The rules, in the order they compose.
        normalizer: The fold of the index the entities will be bound into.
            Defaults to :func:`~dataknobs_common.text.default_normalizer`, which
            is the default of every source that takes one.

    Returns:
        The entities to bind, keyed as given, and what was admitted and refused.
    """
    declared: list[Declared[str]] = []
    for entity_id, entity in entities.items():
        declared.append(Declared(entity_id, entity_id, entity.metadata, generates=False))
        for form in dict.fromkeys((entity.name, *entity.aliases)):
            if form:
                declared.append(Declared(entity_id, form, entity.metadata))
    report = admit(declared, rules, fold=normalizer or default_normalizer)
    added: dict[str, list[str]] = {}
    for variant in report.admitted:
        added.setdefault(variant.key, []).append(variant.form)
    varied: dict[str, Entity[str]] = {}
    for entity_id, entity in entities.items():
        forms = added.get(entity_id)
        varied[entity_id] = replace(entity, aliases=(*entity.aliases, *forms)) if forms else entity
    return varied, report


def apply_variants(
    parts: OntologyParts,
    *,
    normalizer: Callable[[str], str] | None = None,
) -> tuple[Mapping[str, Entity[str]], Variants[str] | None]:
    """The entities a door binds for ``parts``, and the variants report it carries.

    Where the document has no ``variants:`` section this is
    ``parts.declared_entities`` and ``None``, unchanged. Otherwise it is
    :func:`with_variants` over them with the section's rules and the door's
    fold, and the outcome is logged: one INFO line with the counts, and one
    DEBUG line per refusal naming the variant and what it would have met.

    ``parts`` is not changed, so ``parts.declared_entities`` stays as
    authored.

    Args:
        parts: The validated document.
        normalizer: The fold the door's entity source is built with.

    Returns:
        The entities to bind, and the report -- ``None`` with no section.
    """
    if parts.variant_rules is None:
        return parts.declared_entities, None
    entities, report = with_variants(
        parts.declared_entities, parts.variant_rules, normalizer=normalizer
    )
    logger.info(
        "ontology %r: %d variant(s) admitted, %d refused",
        parts.id,
        len(report.admitted),
        len(report.refused),
        extra={
            "ontology_id": parts.id,
            "variants_admitted": len(report.admitted),
            "variants_refused": len(report.refused),
        },
    )
    for refusal in report.refused:
        made = refusal.variant
        logger.debug(
            "ontology %r: variant %r of entity %r (from %r by %s) refused; it would also name %s",
            parts.id,
            made.form,
            made.key,
            made.source,
            made.rule,
            list(refusal.others),
        )
    return entities, report
