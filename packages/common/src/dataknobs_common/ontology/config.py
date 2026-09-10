"""The typed form of an ``ontology:`` document.

Leaf sections stay raw mappings on purpose. ``sources:`` and ``resolver:`` are
discriminated by a ``kind:`` their entries carry, and the set of kinds is a
registry read rather than a list this module could close over -- so typing them
here would mean naming, in ``dataknobs-common``, kinds that other packages
register. The loader validates what it needs and hands the rest on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dataknobs_common.structured_config import StructuredConfig

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class OntologyConfig(StructuredConfig):
    """What a door must be handed to load an ontology.

    **Compared field-wise, and declared unhashable.**
    :class:`~dataknobs_common.structured_config.StructuredConfig` declares
    ``type(cfg).from_dict(cfg.to_dict()) == cfg`` as a property of every
    subclass, so equality is not a choice here -- it is the base class's
    contract -- and a config carrying eight mappings and lists cannot honour a
    hash over its fields as well.

    ``__hash__ = None`` rather than ``frozen=False``, because the base class
    is frozen and a dataclass may not unfreeze one. It reaches the same
    contract by the one route that leaves: equality field-wise,
    :class:`collections.abc.Hashable` answering False, and a caller who asks
    getting the truth instead of a ``TypeError``. The class body sets it
    directly, which ``dataclasses`` honours -- an explicit ``__hash__`` with
    no explicit ``__eq__`` beside it is left alone rather than regenerated.

    Attributes:
        id: The ontology's namespace. Reserved value ``dk`` is refused, and so
            is any id containing ``:``
        version: The vocabulary's own version, not this schema's
        imports: Other ontology ids whose namespaces come into scope
        entity_types: Type declarations, each optionally naming an ``isa``
        relation_types: Relation declarations, loaded into the same store as
            ``entities`` under a typed discriminator
        entities: Inline entity rows, each carrying its own ``id``
        assertions: Inline assertion rows over those entities
        sources: Unbound source specs, discriminated by ``kind:``
        overlay: A default rather than a binding
        taxonomies: Axis definitions -- the definition, never the built axis
        index: The semantic index's configuration, raw
        resolver: The placement cascade's configuration, raw, because its
            ``rungs`` are themselves discriminated by ``kind:``
    """

    id: str
    version: str = "1.0"
    imports: list[str] = field(default_factory=list)
    entity_types: list[Mapping[str, Any]] = field(default_factory=list)
    relation_types: list[Mapping[str, Any]] = field(default_factory=list)
    entities: list[Mapping[str, Any]] = field(default_factory=list)
    assertions: list[Mapping[str, Any]] = field(default_factory=list)
    sources: list[Mapping[str, Any]] = field(default_factory=list)
    overlay: Mapping[str, Any] | None = None
    taxonomies: list[Mapping[str, Any]] = field(default_factory=list)
    index: Mapping[str, Any] | None = None
    resolver: Mapping[str, Any] | None = None

    # Declared unhashable, because every field above but two is a list or a
    # mapping. See the class docstring for why this spelling and not
    # `frozen=False`.
    #
    # The directive is mypy's, not this class's: `__hash__ = None` is the data
    # model's own way of saying a type is unhashable, and mypy reads it against
    # `object.__hash__`'s signature instead. Same shape and same code as the
    # one in `dataknobs_fsm.core.data_wrapper`, which declares a mutable
    # mapping unhashable for the same reason.
    __hash__ = None  # type: ignore[assignment]
