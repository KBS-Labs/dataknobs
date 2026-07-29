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

#: Name of the reserved document section the coordinator auto-manages to store
#: per-user consent grants. Defined here (the lower module) so the store can
#: import it without a circular dependency and the reserved-name guard below
#: cannot drift from the store's usage.
RESERVED_CONSENT_SECTION = "consent"

#: Name of the reserved collection section the coordinator auto-manages to hold
#: the persisted append-only audit log when ``enable_event_log`` is set. Defined
#: here (the lower module) so the store imports it without a circular dependency
#: and the reserved-name guard below cannot drift from the store's usage.
RESERVED_EVENTS_SECTION = "events"

#: Section names the coordinator reserves for its own auto-managed sections. A
#: consumer that declares a section with one of these names collides with the
#: reserved section (which shares the same name map), so it is rejected at
#: config-load time rather than silently shadowed.
RESERVED_SECTION_NAMES: frozenset[str] = frozenset(
    {RESERVED_CONSENT_SECTION, RESERVED_EVENTS_SECTION}
)


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
            A non-``None`` scope gates the section behind a consent grant
            (:meth:`~dataknobs_data.user.store.AsyncUserStateStore.grant_consent`).
        retention_days: Optional retention window, in days, for a
            **collection** section: records whose ``_written_at`` stamp is
            older than the window are removed by
            :meth:`~dataknobs_data.user.store.AsyncUserStateStore.prune`
            (or lazily on ``query`` when ``prune_on_query`` is set). A
            document section holds one evolving record per user and never
            expires, so a ``retention_days`` on a document section is
            rejected at config-load time. When set it must be a **positive**
            number of days; a zero or negative window is rejected at
            config-load time (it would mark live records as already expired
            and delete them on the next ``prune``).
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
        enable_event_log: When ``True``, the coordinator auto-registers a
            reserved ``events`` collection section and appends one
            metadata-only record to it after every data write and scoped
            deletion (a persisted per-user audit trail, distinct from — and in
            addition to — the in-process delta events emitted through the
            callback registry). The log is read through
            :meth:`~dataknobs_data.user.store.AsyncUserStateStore.query_events`;
            it is never written or read through the generic content API. Off by
            default.
        event_log_retention_days: Optional retention window, in days, for the
            reserved ``events`` audit section (only meaningful with
            ``enable_event_log``). When set, a section-less
            :meth:`~dataknobs_data.user.store.AsyncUserStateStore.prune` sweeps
            expired audit records alongside the consumer's own windowed
            sections; when ``None`` (the default) the log is unbounded until the
            consumer erases the user with
            :meth:`~dataknobs_data.user.store.AsyncUserStateStore.clear`. When
            set it must be a positive number of days (a zero or negative window
            is rejected at config-load time).
        prune_on_query: When ``True``, a ``query`` of a collection section
            carrying a ``retention_days`` window first prunes that section's
            expired records for the queried user (lazy retention enforcement).
            Off by default — retention is otherwise enforced only by an
            explicit
            :meth:`~dataknobs_data.user.store.AsyncUserStateStore.prune`.
    """

    backend: str = "memory"
    namespace: str = "user_state"
    sections: tuple[UserStateSectionSpec, ...] = ()
    enable_event_log: bool = False
    event_log_retention_days: int | None = None
    prune_on_query: bool = False

    def __post_init__(self) -> None:
        """Validate the declared sections at config-load time.

        The section map the coordinator builds is keyed by ``name``, so a
        duplicate or empty name silently collapses sections (last wins) and
        surfaces later as a confusing wrong-kind / not-found runtime error far
        from the config typo. Failing here — at construction, on both the
        typed and ``from_dict`` paths — turns that data-integrity footgun into
        an immediate, actionable error. An empty ``sections`` tuple is allowed
        (an inert store); every *declared* section must be named and unique,
        and may not use a name reserved for a coordinator-managed section
        (:data:`RESERVED_SECTION_NAMES`).

        A ``retention_days`` window on a ``DOCUMENT`` section is also rejected:
        a document section holds one evolving record per user and never
        expires, so a retention window on it is a configuration mistake caught
        here rather than silently ignored. A non-positive ``retention_days``
        (zero or negative) on any section is likewise rejected — such a window
        would mark live records as already expired and delete them, so a
        mis-signed window is caught here rather than silently destroying data.
        The same non-positive guard applies to ``event_log_retention_days``.
        """
        if (
            self.event_log_retention_days is not None
            and self.event_log_retention_days < 1
        ):
            raise ConfigurationError(
                "event_log_retention_days="
                f"{self.event_log_retention_days} is invalid: a retention "
                "window must be a positive number of days. A zero or negative "
                "window would mark live audit records as already expired and "
                "delete them on the next prune.",
                context={"namespace": self.namespace},
            )
        seen: set[str] = set()
        for spec in self.sections:
            if not spec.name:
                raise ConfigurationError(
                    "User-state section names must be non-empty.",
                    context={"namespace": self.namespace},
                )
            if (
                spec.retention_days is not None
                and spec.kind == SectionKind.DOCUMENT
            ):
                raise ConfigurationError(
                    f"Section {spec.name!r} is a document section and cannot "
                    "carry retention_days: a document section holds one "
                    "evolving record per user and never expires. Retention "
                    "applies to collection sections.",
                    context={"namespace": self.namespace, "name": spec.name},
                )
            if (
                spec.retention_days is not None
                and spec.retention_days < 1
            ):
                raise ConfigurationError(
                    f"Section {spec.name!r} declares retention_days="
                    f"{spec.retention_days}: a retention window must be a "
                    "positive number of days. A zero or negative window "
                    "would mark live records as already expired and delete "
                    "them on the next prune.",
                    context={"namespace": self.namespace, "name": spec.name},
                )
            if spec.name in RESERVED_SECTION_NAMES:
                raise ConfigurationError(
                    f"Section name {spec.name!r} is reserved for a "
                    "coordinator-managed section and may not be declared. "
                    f"Reserved names: {sorted(RESERVED_SECTION_NAMES)}.",
                    context={"namespace": self.namespace, "name": spec.name},
                )
            if spec.name in seen:
                raise ConfigurationError(
                    f"Duplicate user-state section name: {spec.name!r}. "
                    "Section names must be unique within a store.",
                    context={"namespace": self.namespace, "name": spec.name},
                )
            seen.add(spec.name)
