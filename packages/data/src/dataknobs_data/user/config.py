"""Typed configuration for the per-user cross-session state coordinator.

The coordinator (:class:`~dataknobs_data.user.store.UserStateStore` and its
async sibling) stores per-user state in named *sections*. Each section is a
:class:`UserStateSectionSpec` describing the section's storage shape
(``document`` — one record per user — or ``collection`` — many), governance
tags (``sensitivity``, ``consent_scope``), a retention window
(``retention_days``), and a schema ``version``. :class:`UserStateStoreConfig`
groups the sections with the backing store's ``backend`` and a ``namespace``
that isolates one coordinator's records from another's on a shared store.

Both are frozen :class:`~dataknobs_common.structured_config.StructuredConfig`
subclasses, so a raw dict projects onto typed, validated, round-trippable
objects. The coordinator ships **zero** domain sections — the consumer declares
every section it needs (e.g. ``preferences``, ``alerts``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.structured_config import StructuredConfig


class SectionKind(str, Enum):
    """Storage shape of a user-state section.

    ``DOCUMENT`` sections hold exactly one record per user (settings, a
    profile, a consent ledger) addressed by a derived deterministic id.
    ``COLLECTION`` sections hold many records per user (events, notes,
    interactions) addressed by backend-generated ids and read by filter.
    """

    DOCUMENT = "document"
    COLLECTION = "collection"


class Sensitivity(str, Enum):
    """Governance classification of a section's contents.

    Drives which sections a whole-user :meth:`~dataknobs_data.user.store\
.AsyncUserStateStore.snapshot` surfaces: ``SENSITIVE`` sections are omitted
    from the default snapshot view (opt in with ``include_sensitive=True``).
    ``PUBLIC`` / ``INTERNAL`` sections are always surfaced. Section payload
    values are never emitted in delta events or logs regardless of
    classification (events carry metadata only).
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"


@dataclass(frozen=True)
class UserStateSectionSpec(StructuredConfig):
    """Declaration of one named per-user state section.

    Attributes:
        name: Section name — the key the coordinator's API addresses
            (``get_document(user_id, "preferences")``). Must be unique
            within a store.
        kind: :class:`SectionKind` — ``DOCUMENT`` (one record per user) or
            ``COLLECTION`` (many). Determines which API methods are valid
            for the section.
        schema: Optional field schema for the section's records. Kept a raw
            mapping — it documents the payload shape for the consumer; the
            coordinator does not enforce it in v0.
        sensitivity: :class:`Sensitivity` classification driving snapshot
            visibility (see :class:`Sensitivity`). Defaults to ``INTERNAL``.
        version: Section schema version stamped onto every written record
            (``_section_version``). Reserved for lazy on-read migration;
            not acted on in the base coordinator.
        consent_scope: Optional named consent scope this section belongs to.
            Reserved for consent-gated access enforcement; not acted on in
            the base coordinator.
        retention_days: Optional retention window (collection sections).
            Reserved for retention pruning; not acted on in the base
            coordinator.
    """

    name: str = ""
    kind: SectionKind = SectionKind.DOCUMENT
    schema: Mapping[str, Any] = field(default_factory=dict)
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    version: int = 1
    consent_scope: str | None = None
    retention_days: int | None = None


@dataclass(frozen=True)
class UserStateStoreConfig(StructuredConfig):
    """Configuration for the per-user cross-session state coordinator.

    Attributes:
        backend: Backing database backend key (``"memory"``, ``"sqlite"``,
            ``"postgres"``, …) used only when the coordinator builds its own
            store. An injected database wins and this key is ignored.
        namespace: Logical namespace isolating this coordinator's records
            from another's on a shared backend. Feeds the derived
            document-id and the default single-tenant context's ``domain_id``.
        sections: The declared sections. Every section the coordinator's API
            addresses must appear here; an unknown section name raises.
        enable_event_log: Reserved flag for a persisted append-only audit
            section. Inert in the base coordinator (delta events are emitted
            through the in-process callback registry regardless).
    """

    backend: str = "memory"
    namespace: str = "user_state"
    sections: tuple[UserStateSectionSpec, ...] = ()
    enable_event_log: bool = False

    def __post_init__(self) -> None:
        """Validate the declared sections at config-load time.

        The section map the coordinator builds is keyed by ``name``, so a
        duplicate or empty name silently collapses sections (last wins) and
        surfaces later as a confusing wrong-kind / not-found runtime error far
        from the config typo. Failing here — at construction, on both the
        typed and ``from_dict`` paths — turns that data-integrity footgun into
        an immediate, actionable error. An empty ``sections`` tuple is allowed
        (an inert store); every *declared* section must be named and unique.
        """
        seen: set[str] = set()
        for spec in self.sections:
            if not spec.name:
                raise ConfigurationError(
                    "User-state section names must be non-empty.",
                    context={"namespace": self.namespace},
                )
            if spec.name in seen:
                raise ConfigurationError(
                    f"Duplicate user-state section name: {spec.name!r}. "
                    "Section names must be unique within a store.",
                    context={"namespace": self.namespace, "name": spec.name},
                )
            seen.add(spec.name)
