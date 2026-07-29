"""Per-user cross-session state coordinator.

:class:`UserStateStore` and :class:`AsyncUserStateStore` coordinate a user's
state across sessions. They scope an injected
:class:`~dataknobs_data.SyncDatabase` / :class:`~dataknobs_data.AsyncDatabase`
by ``(namespace, tenant, user_id, section)`` over two section shapes:

- **document** sections — one record per user, addressed by a deterministic
  id derived from the scope tuple (opacity-safe: the opaque ``user_id`` is only
  ever a hash input or a filter value, never split into a delimited key);
- **collection** sections — many records per user, addressed by
  backend-generated ids and read by filter.

Writes are optimistic-concurrency aware (``expected_version`` compare-and-set),
tenant-scoped when a :class:`~dataknobs_common.tenancy.BoundTenantContext` is
injected, and emit a metadata-only delta event through an in-process
:class:`~dataknobs_common.callbacks.CallbackRegistry` (optionally fanned out to
an :class:`~dataknobs_common.events.EventBus`).

The sync and async variants share a set of pure helpers (id derivation, scope
stamping, read-filter composition, snapshot visibility) so their behaviour
cannot drift; each variant owns only its ``await`` / non-``await`` I/O. The
coordinator ships **zero** domain sections — the consumer declares every
section (see :class:`~dataknobs_data.user.config.UserStateStoreConfig`).

Example:
    ```python
    store = await AsyncUserStateStore.from_config(
        {
            "backend": "memory",
            "namespace": "acme",
            "sections": [
                {"name": "preferences", "kind": "document"},
                {"name": "alerts", "kind": "collection"},
            ],
        }
    )
    try:
        await store.put_document("user-42", "preferences", {"theme": "dark"})
        await store.add_record("user-42", "alerts", {"text": "welcome"})
        view = await store.snapshot("user-42")
    finally:
        await store.close()
    ```
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar

from dataknobs_common.callbacks import CallbackRegistry
from dataknobs_common.capabilities import (
    Capability,
    CapabilityLike,
    CapabilityMixin,
)
from dataknobs_common.exceptions import ConfigurationError, ConsentRequiredError
from dataknobs_common.lifecycle import close_if_owned, close_if_owned_sync
from dataknobs_common.structured_config import StructuredConfigConsumer
from dataknobs_common.tenancy import SingleTenantContext, TenantContext
from dataknobs_data.factory import async_database_factory, database_factory
from dataknobs_data.query import Filter, Operator, Query
from dataknobs_data.records import Record
from dataknobs_data.user.config import (
    RESERVED_CONSENT_SECTION,
    SectionKind,
    Sensitivity,
    UserStateSectionSpec,
    UserStateStoreConfig,
)

#: Topic fired on the callback registry (and any composed EventBus) after a
#: successful write. Payloads are metadata-only — a section's *values* are
#: never emitted, so a SENSITIVE section's contents cannot leak into an
#: observer. The payload does carry the ``user_id`` (an opaque identifier) for
#: routing/filtering — it is the only identity in the event; a consumer whose
#: ``user_id`` is itself PII (an email, an OIDC subject) should treat the event
#: stream with the same care as the identifier.
SECTION_WRITTEN_TOPIC = "user_state:section_written"

#: Coordinator-owned fields stamped onto every record. Stripped from the
#: whole-user :meth:`snapshot` view so a consumer sees only its own payload,
#: and skipped by opacity-safe scope comparison.
_RESERVED_FIELDS: frozenset[str] = frozenset(
    {"user_id", "section", "tenant_id", "_section_version", "_written_at"}
)

#: Payload keys that would participate in backend storage-id resolution
#: (:attr:`~dataknobs_data.Record.id` falls back to a ``id`` / ``record_id``
#: data field, and ``storage_id`` / ``_id`` name the id attributes). The
#: coordinator owns record identity — document ids derive from the scope tuple,
#: collection ids are backend-generated — so a caller payload carrying one of
#: these is rejected at the write boundary. This also closes a latent sync/async
#: divergence: the sync memory backend keys a collection ``create`` off a
#: payload ``id`` field while the async backend mints a fresh UUID, so the same
#: payload would land under different ids across variants.
_ID_KEYING_FIELDS: frozenset[str] = frozenset(
    {"id", "storage_id", "_id", "record_id"}
)


# --------------------------------------------------------------------- #
# Pure helpers — the shared, synchronous core both variants call.
# --------------------------------------------------------------------- #


def _document_id(
    namespace: str, tenant_id: str | None, user_id: str, section: str
) -> str:
    """Derive a deterministic document id from the scope tuple.

    Each component is length-prefixed before hashing so no two distinct
    tuples collide (a raw-separator concatenation would collide when a
    component contains the separator). The opaque ``user_id`` is only ever a
    hash *input* here — it is never split on a delimiter — so an id
    containing ``/`` or ``://`` is structurally safe.
    """
    digest = hashlib.blake2b(digest_size=16)
    for component in (namespace, tenant_id or "", user_id, section):
        encoded = component.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _scope_fields(
    user_id: str, section: str, tenant_id: str | None, section_version: int
) -> dict[str, Any]:
    """Return the coordinator-owned scope fields stamped onto a record.

    Pure: the wall-clock ``_written_at`` stamp is added by the write methods
    at the I/O boundary, not here, so this stays deterministic and testable.
    ``tenant_id`` is stamped only when tenant-bound (single-tenant records
    carry no ``tenant_id`` field, matching the read filter).
    """
    fields: dict[str, Any] = {
        "user_id": user_id,
        "section": section,
        "_section_version": section_version,
    }
    if tenant_id is not None:
        fields["tenant_id"] = tenant_id
    return fields


def _read_filter(
    query: Query | None, user_id: str, section: str, tenant_id: str | None
) -> Query:
    """AND-compose the scope filters into a caller's query.

    ``user_id`` and ``section`` are coordinator-owned: any caller-supplied
    filter on those fields is dropped and replaced, so a caller cannot broaden
    scope to another user. ``tenant_id`` is **explicit-filter-wins** — a
    caller passing an explicit ``tenant_id`` filter reads across tenants (the
    admin escape hatch, mirroring ``RAGKnowledgeBase._resolve_read_filter``);
    otherwise the bound tenant is AND-composed. Returns a fresh
    :class:`~dataknobs_data.Query` (the caller's is never mutated), preserving
    its sort / limit / offset / projection.
    """
    base = query if query is not None else Query()
    caller_filters = [
        f for f in base.filters if f.field not in ("user_id", "section")
    ]
    caller_has_tenant = any(f.field == "tenant_id" for f in caller_filters)
    scoped = [
        Filter("user_id", Operator.EQ, user_id),
        Filter("section", Operator.EQ, section),
    ]
    if tenant_id is not None and not caller_has_tenant:
        scoped.append(Filter("tenant_id", Operator.EQ, tenant_id))
    return replace(base, filters=scoped + caller_filters)


def _in_scope(
    record: Record, user_id: str, section: str, tenant_id: str | None
) -> bool:
    """Return whether ``record`` belongs to the given ``(user, section, tenant)``.

    The pure predicate behind both the raising :func:`_verify_scope` guard
    (collection mutation-by-id) and the non-raising scope check in
    :meth:`~AsyncUserStateStore.record_version` (which returns ``None`` for an
    out-of-scope id rather than leaking its existence via an exception).
    """
    return (
        record.get_value("user_id") == user_id
        and record.get_value("section") == section
        and record.get_value("tenant_id") == tenant_id
    )


def _verify_scope(
    record: Record, user_id: str, section: str, tenant_id: str | None
) -> None:
    """Raise if ``record`` does not belong to the given scope.

    Guards collection mutation-by-id: a record id from another user's scope
    cannot be updated (which would re-stamp it into the caller's scope) or
    deleted through the coordinator. Document sections need no such check —
    their id is derived from the scope tuple, so an id mismatch simply reads
    a different record.
    """
    if not _in_scope(record, user_id, section, tenant_id):
        raise ValueError(
            "Target record does not belong to the given user/section scope."
        )


def _visible_sections(
    specs: tuple[UserStateSectionSpec, ...], include_sensitive: bool
) -> list[UserStateSectionSpec]:
    """Select the sections a :meth:`snapshot` surfaces.

    ``SENSITIVE`` sections are omitted from the default view; pass
    ``include_sensitive=True`` to include them.
    """
    if include_sensitive:
        return list(specs)
    return [s for s in specs if s.sensitivity != Sensitivity.SENSITIVE]


def _public_data(record: Record) -> dict[str, Any]:
    """Return a record's payload with the coordinator-owned fields stripped."""
    return {k: v for k, v in record.data.items() if k not in _RESERVED_FIELDS}


def _is_expired(
    record: Record, retention_days: int | None, now: datetime
) -> bool:
    """Return whether ``record`` has aged past its ``retention_days`` window.

    Pure and fail-safe: an unbounded window (``retention_days is None``) never
    expires, and a record whose ``_written_at`` stamp is missing or
    unparseable is treated as *not* expired — the coordinator never deletes a
    record it cannot confidently date. A record is expired when its
    ``_written_at`` is strictly older than ``now - retention_days``.
    """
    if retention_days is None:
        return False
    written_at = record.get_value("_written_at")
    if not isinstance(written_at, str):
        return False
    try:
        written = datetime.fromisoformat(written_at)
    except ValueError:
        return False
    return written < now - timedelta(days=retention_days)


#: The single payload key the reserved consent document nests its grant map
#: under. Grants are stored as ``{_GRANTS_KEY: {scope: {...}}}`` — one isolated
#: namespace — rather than as top-level payload keys, so a ``consent_scope``
#: whose name collides with a coordinator-owned field (:data:`_RESERVED_FIELDS`)
#: is still storable and grantable (a top-level layout would let the reserved
#: stamp shadow the grant, locking the section permanently and silently).
_GRANTS_KEY: str = "grants"


def _grants_of(consent_record: Record | None) -> dict[str, Any]:
    """Return the grant map nested in a consent document (empty when absent).

    The reserved consent document stores every grant under the single
    :data:`_GRANTS_KEY` key, isolated from the coordinator's scope stamps, so
    the grant namespace can never collide with a reserved field name.
    """
    if consent_record is None:
        return {}
    grants = consent_record.get_value(_GRANTS_KEY)
    return dict(grants) if isinstance(grants, Mapping) else {}


def _consent_satisfied(consent_record: Record | None, scope: str) -> bool:
    """Return whether ``scope`` is granted in the consent document.

    Fail-closed (the security rule): a missing consent document, a missing
    scope entry, or a non-``True`` ``granted`` flag all deny access.
    """
    grant = _grants_of(consent_record).get(scope)
    return isinstance(grant, Mapping) and grant.get("granted") is True


def _granted_scopes(consent_record: Record | None) -> frozenset[str]:
    """Return the set of currently-granted scope names in a consent document."""
    return frozenset(
        scope
        for scope, grant in _grants_of(consent_record).items()
        if isinstance(grant, Mapping) and grant.get("granted") is True
    )


def _grant_map(
    grants: Mapping[str, Any], scope: str, *, granted: bool, now: str
) -> dict[str, Any]:
    """Return a new grants map with ``scope`` set granted / revoked at ``now``.

    Pure: the caller reads the current grants, this composes the next map, the
    caller writes it back. A grant records ``granted_at``; a revoke records
    ``revoked_at`` and only flips the flag (block-only — the section's stored
    data is untouched, so a later re-grant surfaces it again).
    """
    updated = dict(grants)
    if granted:
        updated[scope] = {"granted": True, "granted_at": now}
    else:
        updated[scope] = {"granted": False, "revoked_at": now}
    return updated


def _snapshot_sections(
    specs: tuple[UserStateSectionSpec, ...],
    granted_scopes: frozenset[str],
    include_sensitive: bool,
) -> list[UserStateSectionSpec]:
    """Select the sections a :meth:`snapshot` surfaces.

    Composes the sensitivity filter (:func:`_visible_sections`) with the
    consent filter: a section carrying a ``consent_scope`` the user has not
    granted is omitted. The snapshot is the already-filtered surface, so it
    *omits* an ungranted section (mirroring the ``SENSITIVE`` omission) rather
    than raising the way a direct :meth:`get_document` / :meth:`query` does.
    """
    visible = _visible_sections(specs, include_sensitive)
    return [
        s
        for s in visible
        if s.consent_scope is None or s.consent_scope in granted_scopes
    ]


# --------------------------------------------------------------------- #
# Shared non-I/O logic (mixed in LAST so StructuredConfigConsumer stays first).
# --------------------------------------------------------------------- #


class _UserStateStoreCommon:
    """Shared setup / validation / record-building for both variants.

    Mixed in after ``StructuredConfigConsumer`` and ``CapabilityMixin`` so the
    consumer mixin remains the first base (its ``__init__`` is the entry
    point). Holds only synchronous, transport-agnostic logic; the sync and
    async stores own their I/O and their database build.
    """

    # Whether this variant can safely fan a section-written event out to an
    # async :class:`~dataknobs_common.events.EventBus`. Only the async store
    # can — ``EventBus.publish`` is a coroutine, and the sync ``fire`` path
    # cannot drive it from within a running loop (it would raise *after* the
    # write already persisted). Overridden to ``True`` on the async variant.
    _SUPPORTS_ASYNC_FANOUT: ClassVar[bool] = False

    # Attributes established by :meth:`_bind_common` (declared for typing).
    config: UserStateStoreConfig
    components: Mapping[str, Any]
    _db: Any
    _owns_db: bool
    _tenant: TenantContext
    _sections: dict[str, UserStateSectionSpec]
    _reserved_sections: set[str]
    _callbacks: CallbackRegistry
    _now: Callable[[], datetime]

    def _bind_common(self) -> None:
        """Bind the tenant, section map, callbacks, and any injected db.

        Called from each variant's ``_setup``. An injected ``db`` / ``tenant``
        / ``event_bus`` collaborator (passed through the components channel)
        is bound here; the async variant builds its own db later in
        ``_ainit`` when none was injected, the sync variant in its ``_setup``.

        An ``event_bus`` injected into the **sync** store is rejected here (at
        construction) rather than silently doing a per-write ``asyncio.run`` or
        raising *after* a write under a running loop — see
        :attr:`_SUPPORTS_ASYNC_FANOUT`. The sync store still fully supports
        in-process sync callbacks registered on :attr:`_callbacks`.
        """
        self._owns_db = False
        self._db = self.components.get("db")
        tenant = self.components.get("tenant")
        self._tenant = (
            tenant
            if tenant is not None
            else SingleTenantContext(domain_id=self.config.namespace)
        )
        self._sections = {s.name: s for s in self.config.sections}
        self._reserved_sections = set()
        self._register_reserved_sections()
        # The clock is an injected collaborator (a ``Callable`` is not config
        # data — it round-trips a frozen config by identity only), defaulting
        # to wall-clock UTC. It stamps ``_written_at`` and drives retention
        # expiry, so a test can advance a fake clock instead of sleeping.
        injected_now = self.components.get("now")
        self._now = (
            injected_now
            if injected_now is not None
            else (lambda: datetime.now(timezone.utc))
        )
        self._callbacks = CallbackRegistry()
        event_bus = self.components.get("event_bus")
        if event_bus is not None:
            if not self._SUPPORTS_ASYNC_FANOUT:
                raise ConfigurationError(
                    f"{type(self).__name__} (synchronous) cannot fan "
                    "section-written events out to an EventBus: "
                    "EventBus.publish is asynchronous and the sync fire path "
                    "cannot drive it safely from within a running event loop. "
                    "Use AsyncUserStateStore for bus fan-out, or register sync "
                    "callbacks on the callback registry directly.",
                    context={"store": type(self).__name__},
                )
            self._callbacks.also_publish_to(event_bus)

    def _require_kind(
        self, section: str, kind: SectionKind
    ) -> UserStateSectionSpec:
        """Return the spec for ``section``, or raise if unknown / wrong kind.

        A coordinator-managed reserved section (the ``consent`` grant ledger)
        is never a valid content-API target: it is reached only through the
        consent helpers' private read/write path, so addressing it via
        ``get_document`` / ``put_document`` / ``query`` / etc. is refused here
        — the single chokepoint every public content method routes through.
        Without this a caller could ``put_document(user, "consent", …)`` to
        forge a grant or clobber the ledger.
        """
        if section in self._reserved_sections:
            raise ConfigurationError(
                f"Section {section!r} is reserved and coordinator-managed; "
                "it cannot be read or written through the content API. Use "
                "grant_consent / revoke_consent / has_consent.",
                context={"section": section},
            )
        spec = self._sections.get(section)
        if spec is None:
            raise ConfigurationError(
                f"Unknown user-state section: {section!r}. Declared "
                f"sections: {sorted(self._sections)}.",
                context={"section": section, "known": sorted(self._sections)},
            )
        if spec.kind != kind:
            raise ValueError(
                f"Section {section!r} is a {spec.kind.value} section, "
                f"not a {kind.value} section."
            )
        return spec

    def _register_reserved_sections(self) -> None:
        """Register coordinator-managed sections not declared by the consumer.

        The reserved ``consent`` document section is registered whenever at
        least one declared section carries a ``consent_scope`` (no
        consent-scoped sections → not registered → zero overhead). It joins
        ``self._sections`` so the consent helpers and :meth:`clear` see it, and
        ``self._reserved_sections`` so it is distinguishable from a consumer
        section; it never surfaces in :meth:`snapshot` (which iterates only
        ``config.sections``). The frozen config is left untouched — the
        reserved entry is a runtime-only addition.
        """
        if any(s.consent_scope is not None for s in self.config.sections):
            self._sections[RESERVED_CONSENT_SECTION] = UserStateSectionSpec(
                name=RESERVED_CONSENT_SECTION,
                kind=SectionKind.DOCUMENT,
                sensitivity=Sensitivity.INTERNAL,
            )
            self._reserved_sections.add(RESERVED_CONSENT_SECTION)

    def _consent_enabled(self) -> bool:
        """Whether a reserved ``consent`` section is registered for this store."""
        return RESERVED_CONSENT_SECTION in self._sections

    def _consent_spec(self) -> UserStateSectionSpec:
        """Return the reserved consent spec, or raise if consent is unavailable.

        Consent management is only meaningful when at least one declared
        section carries a ``consent_scope``; calling a consent helper otherwise
        is a configuration error, surfaced here with an actionable message
        rather than a confusing ``unknown section`` deep in a write.
        """
        spec = self._sections.get(RESERVED_CONSENT_SECTION)
        if spec is None:
            raise ConfigurationError(
                "Consent management is unavailable: no declared section "
                "carries a consent_scope, so there is nothing to gate.",
                context={"namespace": self.config.namespace},
            )
        return spec

    def _sections_to_prune(
        self, section: str | None
    ) -> list[UserStateSectionSpec]:
        """Resolve which collection sections a :meth:`prune` pass considers.

        With ``section=None`` every collection section carrying a
        ``retention_days`` window is pruned (document sections never expire and
        unwindowed collections are skipped). With an explicit ``section`` the
        spec is resolved through :meth:`_require_kind`, so a document section,
        an unknown section, or the reserved consent section raises rather than
        silently pruning nothing.
        """
        if section is not None:
            return [self._require_kind(section, SectionKind.COLLECTION)]
        return [
            spec
            for spec in self._sections.values()
            if spec.kind == SectionKind.COLLECTION
            and spec.retention_days is not None
        ]

    def _doc_id(self, user_id: str, section: str) -> str:
        return _document_id(
            self.config.namespace, self._tenant.tenant_id, user_id, section
        )

    def _consent_record(
        self, user_id: str, grants: Mapping[str, Any]
    ) -> tuple[str, Record, UserStateSectionSpec]:
        """Build the ``(doc_id, record, spec)`` for the reserved consent document.

        The shared, transport-agnostic half of the consent write path: each
        variant's :meth:`_write_consent` does only the ``await`` / non-``await``
        upsert and the (metadata-only) delta-event fire. Nests the grant map
        under :data:`_GRANTS_KEY` so the grant namespace stays isolated from the
        coordinator's scope stamps. Raises via :meth:`_consent_spec` when no
        section declares a ``consent_scope``. Returns the resolved spec so the
        writer can fire the section-written event without a second lookup.
        """
        spec = self._consent_spec()
        doc_id = self._doc_id(user_id, RESERVED_CONSENT_SECTION)
        record = self._build_record(
            {_GRANTS_KEY: dict(grants)},
            user_id,
            RESERVED_CONSENT_SECTION,
            spec,
            storage_id=doc_id,
        )
        return doc_id, record, spec

    def _build_record(
        self,
        data: Mapping[str, Any],
        user_id: str,
        section: str,
        spec: UserStateSectionSpec,
        *,
        storage_id: str | None = None,
    ) -> Record:
        """Compose a record: caller payload + owned scope fields + stamps.

        Owned identity wins — the scope fields are applied *after* the caller
        payload, so a caller-supplied ``user_id`` / ``section`` / ``tenant_id``
        in ``data`` cannot override the coordinator's. Storage-identity keys
        (:data:`_ID_KEYING_FIELDS`) are rejected outright: the coordinator owns
        record identity, and honouring a caller-supplied one would both break
        that ownership and diverge sync vs async (the sync memory backend keys
        a collection ``create`` off a payload ``id`` while the async one mints
        a UUID).
        """
        conflicting = _ID_KEYING_FIELDS.intersection(data)
        if conflicting:
            raise ValueError(
                "User-state section payloads may not carry storage-identity "
                f"keys {sorted(conflicting)}: the coordinator owns record "
                "identity (document ids derive from the scope tuple; "
                "collection ids are backend-generated). Rename the field(s)."
            )
        payload = dict(data)
        payload.update(
            _scope_fields(
                user_id, section, self._tenant.tenant_id, spec.version
            )
        )
        payload["_written_at"] = self._now().isoformat()
        return Record(payload, storage_id=storage_id)

    def _written_payload(
        self, user_id: str, section: str, spec: UserStateSectionSpec, op: str
    ) -> dict[str, Any]:
        """Build the metadata-only delta-event payload (never section values)."""
        return {
            "namespace": self.config.namespace,
            "tenant_id": self._tenant.tenant_id,
            "user_id": user_id,
            "section": section,
            "kind": spec.kind.value,
            "op": op,
        }


# --------------------------------------------------------------------- #
# Async variant.
# --------------------------------------------------------------------- #


class AsyncUserStateStore(
    StructuredConfigConsumer[UserStateStoreConfig],
    CapabilityMixin,
    _UserStateStoreCommon,
):
    """Async coordinator for per-user cross-session state.

    Build from config (``await AsyncUserStateStore.from_config({...})`` — builds
    the backing database when none is injected) or from a pre-built database
    (``AsyncUserStateStore.from_components(db=…)``). An injected database is
    caller-owned and left open by :meth:`close`; a config-built one is owned
    and closed.
    """

    CONFIG_CLS: ClassVar[type[UserStateStoreConfig]] = UserStateStoreConfig

    # The async store drives fan-out through ``fire_async`` / awaited
    # ``EventBus.publish``, so composing an EventBus is safe here.
    _SUPPORTS_ASYNC_FANOUT: ClassVar[bool] = True

    # Structural advertisement: the class HAS the conditional-write
    # (``expected_version``) and tenant-scoping code paths. Whether a given
    # instance is currently tenant-scoping is the binding check
    # ``store._tenant.tenant_id is not None``.
    SUPPORTED_CAPABILITIES: ClassVar[frozenset[CapabilityLike]] = frozenset(
        {Capability.CONDITIONAL_WRITE, Capability.TENANT_SCOPED_STATE}
    )

    def _setup(self) -> None:
        self._bind_common()

    @classmethod
    async def from_config(  # type: ignore[override]
        cls, config: Any, **components: Any
    ) -> AsyncUserStateStore:
        """Create the coordinator from configuration (async build).

        Accepts a config dict or a typed :class:`UserStateStoreConfig`. Builds
        the backing database from ``config.backend`` unless a ``db``
        collaborator is injected. Routes through :meth:`from_config_async`.
        """
        return await cls.from_config_async(config, **components)

    async def _ainit(
        self,
        *,
        db: Any = None,
        event_bus: Any = None,
        tenant: Any = None,
        now: Any = None,
    ) -> None:
        if self._prebuilt:
            return
        # ``db`` / ``event_bus`` / ``tenant`` / ``now`` were already bound from
        # the components channel in ``_bind_common``; the only async-only work
        # is building a database when none was injected.
        if self._db is None:
            self._db = async_database_factory.create(backend=self.config.backend)
            self._owns_db = True

    def _adopt_components(
        self,
        *,
        db: Any = None,
        event_bus: Any = None,
        tenant: Any = None,
        now: Any = None,
    ) -> None:
        if db is None:
            raise TypeError(
                "AsyncUserStateStore.from_components requires a `db` "
                "collaborator."
            )
        self._db = db
        self._owns_db = False

    # ----- consent ----- #

    async def _read_consent(self, user_id: str) -> Record | None:
        """Read the reserved consent document for ``user_id`` (or None)."""
        return await self._db.read(
            self._doc_id(user_id, RESERVED_CONSENT_SECTION)
        )

    async def _write_consent(
        self, user_id: str, grants: Mapping[str, Any], op: str
    ) -> None:
        """Persist the reserved consent document (the private write path).

        Bypasses the public ``put_document`` — which now refuses the reserved
        ``consent`` section — so grants can only be written through
        :meth:`grant_consent` / :meth:`revoke_consent`, never forged or
        clobbered through the content API. Still fires the same metadata-only
        ``section_written`` delta event every other write fires (carrying
        ``op`` — ``"grant_consent"`` / ``"revoke_consent"`` — and never the
        scope name or grant status), so consent writes stay observable.
        """
        doc_id, record, spec = self._consent_record(user_id, grants)
        await self._db.upsert(doc_id, record)
        await self._fire_written(
            user_id, RESERVED_CONSENT_SECTION, spec, op
        )

    async def _require_consent(
        self, user_id: str, spec: UserStateSectionSpec
    ) -> None:
        """Refuse access to a consent-scoped section that is not granted.

        A no-op for a section with no ``consent_scope`` (including the reserved
        consent section itself, so reading / writing the consent document is
        never gated — no recursion). Otherwise reads the consent document and
        raises :class:`~dataknobs_common.exceptions.ConsentRequiredError` when
        the scope is not granted (fail-closed).
        """
        if spec.consent_scope is None:
            return
        if not _consent_satisfied(
            await self._read_consent(user_id), spec.consent_scope
        ):
            raise ConsentRequiredError(
                f"Access to section {spec.name!r} requires consent scope "
                f"{spec.consent_scope!r}, which has not been granted.",
                scope=spec.consent_scope,
                user_id=user_id,
            )

    async def grant_consent(self, user_id: str, scope: str) -> None:
        """Grant ``scope`` for ``user_id``, unlocking every section tagged with it."""
        self._consent_spec()
        grants = _grants_of(await self._read_consent(user_id))
        await self._write_consent(
            user_id,
            _grant_map(grants, scope, granted=True, now=self._now().isoformat()),
            "grant_consent",
        )

    async def revoke_consent(self, user_id: str, scope: str) -> None:
        """Revoke ``scope`` for ``user_id``.

        Block-only: future access to sections tagged with ``scope`` is refused,
        but their stored data is left in place (a later :meth:`grant_consent`
        surfaces it again). Erasure remains the explicit :meth:`clear`.
        """
        self._consent_spec()
        grants = _grants_of(await self._read_consent(user_id))
        await self._write_consent(
            user_id,
            _grant_map(grants, scope, granted=False, now=self._now().isoformat()),
            "revoke_consent",
        )

    async def has_consent(self, user_id: str, scope: str) -> bool:
        """Return whether ``user_id`` has granted ``scope``."""
        self._consent_spec()
        return _consent_satisfied(await self._read_consent(user_id), scope)

    # ----- document sections ----- #

    async def get_document(self, user_id: str, section: str) -> Record | None:
        """Read a document section's single record for ``user_id`` (or None).

        A section carrying a ``consent_scope`` raises
        :class:`~dataknobs_common.exceptions.ConsentRequiredError` when the
        user has not granted that scope.
        """
        spec = self._require_kind(section, SectionKind.DOCUMENT)
        await self._require_consent(user_id, spec)
        return await self._db.read(self._doc_id(user_id, section))

    async def document_version(
        self, user_id: str, section: str
    ) -> str | None:
        """Return the compare-and-set token for a document (or None if absent).

        Pass the returned token as ``expected_version`` to :meth:`put_document`
        for a conditional write.
        """
        self._require_kind(section, SectionKind.DOCUMENT)
        return await self._db.get_version(self._doc_id(user_id, section))

    async def put_document(
        self,
        user_id: str,
        section: str,
        data: Mapping[str, Any],
        *,
        expected_version: str | None = None,
    ) -> str:
        """Create or replace a document section's record for ``user_id``.

        When ``expected_version`` is provided the write is a compare-and-set:
        it proceeds only if the stored token still matches, else raises
        :class:`~dataknobs_common.exceptions.ConcurrencyError`.
        """
        spec = self._require_kind(section, SectionKind.DOCUMENT)
        await self._require_consent(user_id, spec)
        doc_id = self._doc_id(user_id, section)
        record = self._build_record(
            data, user_id, section, spec, storage_id=doc_id
        )
        result_id = await self._db.upsert(
            doc_id, record, expected_version=expected_version
        )
        await self._fire_written(user_id, section, spec, "put_document")
        return result_id

    # ----- collection sections ----- #

    async def add_record(
        self, user_id: str, section: str, data: Mapping[str, Any]
    ) -> str:
        """Append a record to a collection section for ``user_id``.

        Returns the backend-generated record id.
        """
        spec = self._require_kind(section, SectionKind.COLLECTION)
        await self._require_consent(user_id, spec)
        record = self._build_record(data, user_id, section, spec)
        record_id = await self._db.create(record)
        await self._fire_written(user_id, section, spec, "add_record")
        return record_id

    async def query(
        self, user_id: str, section: str, query: Query | None = None
    ) -> list[Record]:
        """Read a collection section's records for ``user_id``.

        The optional ``query`` adds payload filters / sort / pagination; the
        user + section (+ bound tenant) scope is AND-composed automatically. A
        section carrying a ``consent_scope`` raises
        :class:`~dataknobs_common.exceptions.ConsentRequiredError` when the
        user has not granted that scope. When the store is configured with
        ``prune_on_query`` and the section carries a ``retention_days`` window,
        the user's expired records in the section are pruned before the read.
        """
        spec = self._require_kind(section, SectionKind.COLLECTION)
        await self._require_consent(user_id, spec)
        if self.config.prune_on_query and spec.retention_days is not None:
            await self.prune(user_id, section)
        return await self._db.search(
            _read_filter(query, user_id, section, self._tenant.tenant_id)
        )

    async def prune(self, user_id: str, section: str | None = None) -> int:
        """Delete ``user_id``'s records past their section's retention window.

        With ``section=None`` every collection section carrying a
        ``retention_days`` window is pruned; with an explicit ``section`` only
        that one is (a document / unknown / reserved section raises through
        :meth:`_require_kind`). Expiry is measured against the injected clock,
        so a caller schedules pruning on its own cadence — the coordinator is a
        library, not a daemon. Not consent-gated: pruning is data minimization,
        which must always be possible (mirroring :meth:`clear`). Returns the
        number of records deleted.
        """
        now = self._now()
        ids: list[str] = []
        for spec in self._sections_to_prune(section):
            records = await self._db.search(
                _read_filter(None, user_id, spec.name, self._tenant.tenant_id)
            )
            ids.extend(
                r.storage_id
                for r in records
                if r.storage_id
                and _is_expired(r, spec.retention_days, now)
            )
        if not ids:
            return 0
        results = await self._db.delete_batch(ids)
        return sum(1 for ok in results if ok)

    async def record_version(
        self, user_id: str, section: str, record_id: str
    ) -> str | None:
        """Return the compare-and-set token for a collection record.

        Scope-checked: a ``record_id`` that is absent, or belongs to another
        user / section / tenant, returns ``None`` (indistinguishable from a
        missing record) rather than leaking its existence or version. Pass the
        returned token as ``expected_version`` to :meth:`update_record` /
        :meth:`delete_record` for a conditional write.
        """
        self._require_kind(section, SectionKind.COLLECTION)
        existing = await self._db.read(record_id)
        if existing is None or not _in_scope(
            existing, user_id, section, self._tenant.tenant_id
        ):
            return None
        return await self._db.get_version(record_id)

    async def update_record(
        self,
        user_id: str,
        section: str,
        record_id: str,
        data: Mapping[str, Any],
        *,
        expected_version: str | None = None,
    ) -> bool:
        """Replace a collection record owned by ``user_id`` (scope-checked)."""
        spec = self._require_kind(section, SectionKind.COLLECTION)
        await self._require_consent(user_id, spec)
        existing = await self._db.read(record_id)
        if existing is None:
            return False
        _verify_scope(existing, user_id, section, self._tenant.tenant_id)
        record = self._build_record(
            data, user_id, section, spec, storage_id=record_id
        )
        updated = await self._db.update(
            record_id, record, expected_version=expected_version
        )
        if updated:
            await self._fire_written(user_id, section, spec, "update_record")
        return updated

    async def delete_record(
        self,
        user_id: str,
        section: str,
        record_id: str,
        *,
        expected_version: str | None = None,
    ) -> bool:
        """Delete a collection record owned by ``user_id`` (scope-checked)."""
        self._require_kind(section, SectionKind.COLLECTION)
        existing = await self._db.read(record_id)
        if existing is None:
            return False
        _verify_scope(existing, user_id, section, self._tenant.tenant_id)
        return await self._db.delete(
            record_id, expected_version=expected_version
        )

    # ----- whole-user ----- #

    async def snapshot(
        self, user_id: str, *, include_sensitive: bool = False
    ) -> dict[str, Any]:
        """Return a whole-user view keyed by section name.

        Document sections map to their payload dict (or ``None`` when unset);
        collection sections map to a list of payload dicts. Coordinator-owned
        fields are stripped. ``SENSITIVE`` sections are omitted unless
        ``include_sensitive=True``; a consent-scoped section the user has not
        granted is likewise omitted (the snapshot omits rather than raises).
        """
        granted = (
            _granted_scopes(await self._read_consent(user_id))
            if self._consent_enabled()
            else frozenset()
        )
        view: dict[str, Any] = {}
        for spec in _snapshot_sections(
            self.config.sections, granted, include_sensitive
        ):
            if spec.kind == SectionKind.DOCUMENT:
                record = await self.get_document(user_id, spec.name)
                view[spec.name] = (
                    _public_data(record) if record is not None else None
                )
            else:
                records = await self.query(user_id, spec.name)
                view[spec.name] = [_public_data(r) for r in records]
        return view

    async def clear(self, user_id: str) -> int:
        """Delete every record for ``user_id`` across all sections.

        The right-to-erasure primitive. Returns the number of records deleted.
        Every id is collected first and removed through a single
        :meth:`~dataknobs_data.AsyncDatabase.delete_batch`, so a backend that
        implements atomic batch deletion erases the user in one operation; the
        in-memory default deletes them sequentially.
        """
        ids = await self._clear_ids(user_id)
        if not ids:
            return 0
        results = await self._db.delete_batch(ids)
        return sum(1 for ok in results if ok)

    async def _clear_ids(self, user_id: str) -> list[str]:
        """Collect every storage id for ``user_id`` across all sections.

        Iterates ``self._sections`` (declared **and** reserved) so erasure
        removes coordinator-managed state — the consent document included —
        not just the consumer's declared sections.
        """
        ids: list[str] = []
        for spec in self._sections.values():
            if spec.kind == SectionKind.DOCUMENT:
                ids.append(self._doc_id(user_id, spec.name))
            else:
                records = await self._db.search(
                    _read_filter(
                        None, user_id, spec.name, self._tenant.tenant_id
                    )
                )
                ids.extend(r.storage_id for r in records if r.storage_id)
        return ids

    async def close(self) -> None:
        """Release the backing database when this coordinator owns it."""
        await close_if_owned(self._db, self._owns_db)

    async def _fire_written(
        self, user_id: str, section: str, spec: UserStateSectionSpec, op: str
    ) -> None:
        await self._callbacks.fire_async(
            SECTION_WRITTEN_TOPIC,
            self._written_payload(user_id, section, spec, op),
        )


# --------------------------------------------------------------------- #
# Sync variant.
# --------------------------------------------------------------------- #


class UserStateStore(
    StructuredConfigConsumer[UserStateStoreConfig],
    CapabilityMixin,
    _UserStateStoreCommon,
):
    """Synchronous coordinator for per-user cross-session state.

    The sync mirror of :class:`AsyncUserStateStore`. Build from config
    (``UserStateStore.from_config({...})`` — builds the backing database when
    none is injected) or from a pre-built database
    (``UserStateStore.from_components(db=…)``). Ownership / teardown semantics
    match the async variant.
    """

    CONFIG_CLS: ClassVar[type[UserStateStoreConfig]] = UserStateStoreConfig

    SUPPORTED_CAPABILITIES: ClassVar[frozenset[CapabilityLike]] = frozenset(
        {Capability.CONDITIONAL_WRITE, Capability.TENANT_SCOPED_STATE}
    )

    def _setup(self) -> None:
        self._bind_common()
        # Sync construction has no async hook, so the database (when not
        # injected) is built here.
        if self._db is None:
            self._db = database_factory.create(backend=self.config.backend)
            self._owns_db = True

    def _adopt_components(
        self,
        *,
        db: Any = None,
        event_bus: Any = None,
        tenant: Any = None,
        now: Any = None,
    ) -> None:
        if db is None:
            raise TypeError(
                "UserStateStore.from_components requires a `db` collaborator."
            )
        self._db = db
        self._owns_db = False

    # ----- consent ----- #

    def _read_consent(self, user_id: str) -> Record | None:
        """Read the reserved consent document for ``user_id`` (or None)."""
        return self._db.read(self._doc_id(user_id, RESERVED_CONSENT_SECTION))

    def _write_consent(
        self, user_id: str, grants: Mapping[str, Any], op: str
    ) -> None:
        """Persist the reserved consent document (sync mirror).

        Bypasses the public ``put_document`` (which refuses the reserved
        ``consent`` section) so grants are writable only through the consent
        helpers — never forged or clobbered through the content API. Still fires
        the same metadata-only ``section_written`` delta event every other write
        fires (carrying ``op``, never the scope name or grant status).
        """
        doc_id, record, spec = self._consent_record(user_id, grants)
        self._db.upsert(doc_id, record)
        self._fire_written(user_id, RESERVED_CONSENT_SECTION, spec, op)

    def _require_consent(
        self, user_id: str, spec: UserStateSectionSpec
    ) -> None:
        """Refuse access to a consent-scoped section that is not granted.

        Sync mirror of the async twin: a no-op for a section with no
        ``consent_scope`` (including the reserved consent section — no
        recursion); otherwise reads the consent document and raises
        :class:`~dataknobs_common.exceptions.ConsentRequiredError` when the
        scope is not granted (fail-closed).
        """
        if spec.consent_scope is None:
            return
        if not _consent_satisfied(
            self._read_consent(user_id), spec.consent_scope
        ):
            raise ConsentRequiredError(
                f"Access to section {spec.name!r} requires consent scope "
                f"{spec.consent_scope!r}, which has not been granted.",
                scope=spec.consent_scope,
                user_id=user_id,
            )

    def grant_consent(self, user_id: str, scope: str) -> None:
        """Grant ``scope`` for ``user_id``, unlocking every section tagged with it."""
        self._consent_spec()
        grants = _grants_of(self._read_consent(user_id))
        self._write_consent(
            user_id,
            _grant_map(grants, scope, granted=True, now=self._now().isoformat()),
            "grant_consent",
        )

    def revoke_consent(self, user_id: str, scope: str) -> None:
        """Revoke ``scope`` for ``user_id`` (block future access; data untouched)."""
        self._consent_spec()
        grants = _grants_of(self._read_consent(user_id))
        self._write_consent(
            user_id,
            _grant_map(grants, scope, granted=False, now=self._now().isoformat()),
            "revoke_consent",
        )

    def has_consent(self, user_id: str, scope: str) -> bool:
        """Return whether ``user_id`` has granted ``scope``."""
        self._consent_spec()
        return _consent_satisfied(self._read_consent(user_id), scope)

    # ----- document sections ----- #

    def get_document(self, user_id: str, section: str) -> Record | None:
        """Read a document section's single record for ``user_id`` (or None).

        A consent-scoped section raises
        :class:`~dataknobs_common.exceptions.ConsentRequiredError` when the
        user has not granted that scope.
        """
        spec = self._require_kind(section, SectionKind.DOCUMENT)
        self._require_consent(user_id, spec)
        return self._db.read(self._doc_id(user_id, section))

    def document_version(self, user_id: str, section: str) -> str | None:
        """Return the compare-and-set token for a document (or None if absent)."""
        self._require_kind(section, SectionKind.DOCUMENT)
        return self._db.get_version(self._doc_id(user_id, section))

    def put_document(
        self,
        user_id: str,
        section: str,
        data: Mapping[str, Any],
        *,
        expected_version: str | None = None,
    ) -> str:
        """Create or replace a document section's record for ``user_id``."""
        spec = self._require_kind(section, SectionKind.DOCUMENT)
        self._require_consent(user_id, spec)
        doc_id = self._doc_id(user_id, section)
        record = self._build_record(
            data, user_id, section, spec, storage_id=doc_id
        )
        result_id = self._db.upsert(
            doc_id, record, expected_version=expected_version
        )
        self._fire_written(user_id, section, spec, "put_document")
        return result_id

    # ----- collection sections ----- #

    def add_record(
        self, user_id: str, section: str, data: Mapping[str, Any]
    ) -> str:
        """Append a record to a collection section for ``user_id``."""
        spec = self._require_kind(section, SectionKind.COLLECTION)
        self._require_consent(user_id, spec)
        record = self._build_record(data, user_id, section, spec)
        record_id = self._db.create(record)
        self._fire_written(user_id, section, spec, "add_record")
        return record_id

    def query(
        self, user_id: str, section: str, query: Query | None = None
    ) -> list[Record]:
        """Read a collection section's records for ``user_id``.

        A consent-scoped section raises
        :class:`~dataknobs_common.exceptions.ConsentRequiredError` when the
        user has not granted that scope. When the store is configured with
        ``prune_on_query`` and the section carries a ``retention_days`` window,
        the user's expired records in the section are pruned before the read.
        """
        spec = self._require_kind(section, SectionKind.COLLECTION)
        self._require_consent(user_id, spec)
        if self.config.prune_on_query and spec.retention_days is not None:
            self.prune(user_id, section)
        return self._db.search(
            _read_filter(query, user_id, section, self._tenant.tenant_id)
        )

    def prune(self, user_id: str, section: str | None = None) -> int:
        """Delete ``user_id``'s records past their section's retention window.

        Sync mirror of the async twin: with ``section=None`` every windowed
        collection section is pruned; an explicit document / unknown / reserved
        section raises through :meth:`_require_kind`. Expiry is measured against
        the injected clock; not consent-gated (data minimization, like
        :meth:`clear`). Returns the number of records deleted.
        """
        now = self._now()
        ids: list[str] = []
        for spec in self._sections_to_prune(section):
            records = self._db.search(
                _read_filter(None, user_id, spec.name, self._tenant.tenant_id)
            )
            ids.extend(
                r.storage_id
                for r in records
                if r.storage_id
                and _is_expired(r, spec.retention_days, now)
            )
        if not ids:
            return 0
        results = self._db.delete_batch(ids)
        return sum(1 for ok in results if ok)

    def record_version(
        self, user_id: str, section: str, record_id: str
    ) -> str | None:
        """Return the compare-and-set token for a collection record.

        Scope-checked mirror of the async twin: an absent or out-of-scope
        ``record_id`` returns ``None`` rather than leaking its existence.
        """
        self._require_kind(section, SectionKind.COLLECTION)
        existing = self._db.read(record_id)
        if existing is None or not _in_scope(
            existing, user_id, section, self._tenant.tenant_id
        ):
            return None
        return self._db.get_version(record_id)

    def update_record(
        self,
        user_id: str,
        section: str,
        record_id: str,
        data: Mapping[str, Any],
        *,
        expected_version: str | None = None,
    ) -> bool:
        """Replace a collection record owned by ``user_id`` (scope-checked)."""
        spec = self._require_kind(section, SectionKind.COLLECTION)
        self._require_consent(user_id, spec)
        existing = self._db.read(record_id)
        if existing is None:
            return False
        _verify_scope(existing, user_id, section, self._tenant.tenant_id)
        record = self._build_record(
            data, user_id, section, spec, storage_id=record_id
        )
        updated = self._db.update(
            record_id, record, expected_version=expected_version
        )
        if updated:
            self._fire_written(user_id, section, spec, "update_record")
        return updated

    def delete_record(
        self,
        user_id: str,
        section: str,
        record_id: str,
        *,
        expected_version: str | None = None,
    ) -> bool:
        """Delete a collection record owned by ``user_id`` (scope-checked)."""
        self._require_kind(section, SectionKind.COLLECTION)
        existing = self._db.read(record_id)
        if existing is None:
            return False
        _verify_scope(existing, user_id, section, self._tenant.tenant_id)
        return self._db.delete(record_id, expected_version=expected_version)

    # ----- whole-user ----- #

    def snapshot(
        self, user_id: str, *, include_sensitive: bool = False
    ) -> dict[str, Any]:
        """Return a whole-user view keyed by section name (see async twin)."""
        granted = (
            _granted_scopes(self._read_consent(user_id))
            if self._consent_enabled()
            else frozenset()
        )
        view: dict[str, Any] = {}
        for spec in _snapshot_sections(
            self.config.sections, granted, include_sensitive
        ):
            if spec.kind == SectionKind.DOCUMENT:
                record = self.get_document(user_id, spec.name)
                view[spec.name] = (
                    _public_data(record) if record is not None else None
                )
            else:
                records = self.query(user_id, spec.name)
                view[spec.name] = [_public_data(r) for r in records]
        return view

    def clear(self, user_id: str) -> int:
        """Delete every record for ``user_id`` across all sections.

        Sync mirror of the async twin: ids are collected first and removed
        through a single :meth:`~dataknobs_data.SyncDatabase.delete_batch`.
        """
        ids = self._clear_ids(user_id)
        if not ids:
            return 0
        results = self._db.delete_batch(ids)
        return sum(1 for ok in results if ok)

    def _clear_ids(self, user_id: str) -> list[str]:
        """Collect every storage id for ``user_id`` across all sections.

        Iterates ``self._sections`` (declared **and** reserved) so erasure
        removes coordinator-managed state — the consent document included.
        """
        ids: list[str] = []
        for spec in self._sections.values():
            if spec.kind == SectionKind.DOCUMENT:
                ids.append(self._doc_id(user_id, spec.name))
            else:
                records = self._db.search(
                    _read_filter(
                        None, user_id, spec.name, self._tenant.tenant_id
                    )
                )
                ids.extend(r.storage_id for r in records if r.storage_id)
        return ids

    def close(self) -> None:
        """Release the backing database when this coordinator owns it."""
        close_if_owned_sync(self._db, self._owns_db)

    def _fire_written(
        self, user_id: str, section: str, spec: UserStateSectionSpec, op: str
    ) -> None:
        self._callbacks.fire(
            SECTION_WRITTEN_TOPIC,
            self._written_payload(user_id, section, spec, op),
        )
