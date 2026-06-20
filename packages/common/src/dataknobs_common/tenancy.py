"""Tenant-scoped backend operations Protocol and reference implementations.

A :class:`TenantContext` describes the tenant scope of a backend operation
(typically a knowledge-base backend or per-tenant resource operation).
Backends receive a ``TenantContext`` alongside operation parameters and use
it to compute distributed-lock keys, state-storage prefixes, and isolation
boundaries.

The Protocol exposes two scalar accessors and three projection methods::

    tenant_id          — None for single-tenant deployments
    domain_id          — always required; the KB / corpus identifier
    lock_key(op)       — projects (tenant_id, domain_id, op) to a
                         ``DistributedLock.acquire``-compatible string
    state_key_prefix() — projects the tenant scope to a state-storage
                         path prefix ("", "tenants/{t}/_state/", ...)
    matches(other)     — explicit ``__eq__``-like comparison for
                         cache-key / isolation-boundary use

Backends with no tenancy awareness MUST still accept a ``TenantContext``
but MAY treat ``tenant_id`` as informational only — pass a
:class:`SingleTenantContext` to preserve the single-tenant behavior of code
written before the context surface existed.

Reference implementations:

- :class:`SingleTenantContext` — backwards-compatible default. Lock keys +
  empty state prefix byte-identical to the pre-context single-tenant code.
- :class:`BoundTenantContext` — strict per-tenant isolation; locks + state
  both tenant-scoped.
- :class:`PrefixedTenantContext` — consumer-controlled prefix convention via
  a format-string template.
- :class:`SharedCorpusTenantContext` — tenant-isolated state on a shared
  content corpus.

Construct a context from a config mapping with :func:`create_tenant_context`
or from the environment with :func:`tenant_context_from_env`.

Capability advertisement (e.g. ``TENANT_SCOPED_STATE`` /
``TENANT_SCOPED_LOCKS``) is the responsibility of the *consuming* backend,
not the context. A context only projects identity into keys; the backend
decides whether it honors strict per-tenant locking or snapshot isolation and
declares the matching capability.

Extend the family by writing a frozen, hashable class satisfying
:class:`TenantContext` (and, if config-driven construction is needed, your own
small factory). The four reference impls plus :class:`PrefixedTenantContext`'s
format-string escape hatch cover the common conventions, so there is
intentionally no plugin registry here.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "BoundTenantContext",
    "PrefixedTenantContext",
    "SharedCorpusTenantContext",
    "SingleTenantContext",
    "TenantContext",
    "create_tenant_context",
    "tenant_context_from_env",
]


@runtime_checkable
class TenantContext(Protocol):
    """Tenant scope of a backend operation.

    Implementations project ``(tenant_id, domain_id, operation)`` tuples
    into lock keys and state-storage prefixes. The Protocol is
    backwards-compatible at the single-tenant boundary via
    :class:`SingleTenantContext`, whose projections are byte-identical to
    the pre-context code.

    Implementations MUST be:

    - **immutable** (frozen) — contexts are passed across method boundaries
      and into caches; mutation would break consistency
    - **deterministic** — same context → same lock keys + state prefixes
      across calls
    - **hashable** — contexts are used as cache keys and set members

    The provided reference impls are frozen dataclasses; consumer impls
    should follow the same shape.
    """

    @property
    def tenant_id(self) -> str | None:
        """The tenant identifier, or ``None`` for single-tenant operations.

        Returns ``None`` for :class:`SingleTenantContext`; a ``str`` for
        tenant-scoped impls. Backends without tenancy awareness MAY ignore
        this value entirely.
        """
        ...

    @property
    def domain_id(self) -> str:
        """The KB / corpus / domain identifier. Always required.

        Even multi-tenant configurations have a domain (the KB id).
        Single-tenant impls treat the domain as the primary identity;
        tenant-scoped impls compose tenant + domain into lock keys and
        state prefixes.
        """
        ...

    def lock_key(self, operation: str) -> str:
        """Compute the distributed-lock key for this context + operation.

        The returned string is passed to ``DistributedLock.acquire(...)``
        directly. The Protocol does the ``(tenant_id, domain_id,
        operation)`` → ``str`` projection; the lock backend does not see
        the structure.

        Reference impl conventions::

            SingleTenantContext       — "{operation}:{domain_id}"
            BoundTenantContext        — "{operation}:{tenant_id}:{domain_id}"
            PrefixedTenantContext     — "{operation}:{prefix}:{domain_id}"
            SharedCorpusTenantContext — "{operation}:{tenant_id}:{corpus}"
        """
        ...

    def state_key_prefix(self) -> str:
        """Prefix for state-key storage (e.g. ``_state/`` files).

        Returns ``""`` for single-tenant deployments (backwards-compat).
        Tenant-scoped impls return non-empty values like
        ``"tenants/{tenant_id}/_state/"`` so concurrent ingest of the same
        domain by different tenants does not race on state writes.

        The trailing separator is part of the convention (concatenation,
        not path-join) — implementations include the trailing ``"/"`` so
        backend code can do ``prefix + filename`` without inserting it.
        """
        ...

    def matches(self, other: TenantContext) -> bool:
        """True iff ``self`` and ``other`` describe the same tenant scope.

        Used for cache-key comparison + state-isolation enforcement.
        Implementations MAY treat scopes as equivalent under projection
        (e.g. :class:`SharedCorpusTenantContext` treats the same
        ``shared_corpus_id`` + ``tenant_id`` as a content-key match even
        when ``domain_id`` differs).

        For frozen-dataclass impls, ``matches`` typically defers to value
        equality — but it is explicit so consumers do not rely on
        ``__eq__`` semantics being identical to ``matches`` semantics.
        """
        ...


# --------------------------------------------------------------------- #
# Reference implementations
# --------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class SingleTenantContext:
    """Backwards-compat default for single-tenant deployments.

    Lock keys follow the canonical single-tenant format
    ``"{operation}:{domain_id}"`` — the stable shape backends adopting the
    context surface route their single-tenant locks through. (The contract
    tests pin this format so it cannot drift once backends depend on it.)

    State-key prefix is ``""`` — state-write paths are unchanged from the
    single-tenant on-disk layout used before the context surface existed, so
    existing state files stay readable.

    The ``tenant_id`` property returns ``None``; backends use this as the
    "no tenant context" signal to skip per-tenant routing.

    Use case: code written before the context surface existed that just
    wants tenant-aware backends to do the right thing. Wrapping a call
    site's ``domain_id`` in ``SingleTenantContext(domain_id)`` is the
    migration path.
    """

    domain_id: str

    @property
    def tenant_id(self) -> str | None:
        return None

    def lock_key(self, operation: str) -> str:
        return f"{operation}:{self.domain_id}"

    def state_key_prefix(self) -> str:
        return ""

    def matches(self, other: TenantContext) -> bool:
        return (
            isinstance(other, SingleTenantContext)
            and other.domain_id == self.domain_id
        )

    def __eq__(self, other: object) -> bool:
        """Accept ``str`` equality for incremental consumer-code migration.

        Consumer code with ``ctx == "domain_id"`` continues to compare
        positively against the matching context during the migration
        window. Multi-tenant consumers MUST migrate to
        ``ctx.matches(other_ctx)`` — no ``str``-equality shim is possible
        once ``tenant_id`` is part of the identity.
        """
        if isinstance(other, str):
            return self.domain_id == other
        if isinstance(other, SingleTenantContext):
            return self.domain_id == other.domain_id
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("SingleTenantContext", self.domain_id))


@dataclass(frozen=True)
class BoundTenantContext:
    """Strict per-tenant isolation.

    Lock keys include the ``tenant_id`` segment, so concurrent operations
    on the same domain by different tenants do not collide.

    State-key prefix is ``"tenants/{tenant_id}/_state/"`` — each tenant's
    state files live under a per-tenant subtree, isolating metadata +
    snapshot writes across tenants on the same backend.

    Use case: the standard default for multi-tenant deployments where
    tenants must not share state.
    """

    tenant_id: str
    domain_id: str

    def lock_key(self, operation: str) -> str:
        return f"{operation}:{self.tenant_id}:{self.domain_id}"

    def state_key_prefix(self) -> str:
        return f"tenants/{self.tenant_id}/_state/"

    def matches(self, other: TenantContext) -> bool:
        return (
            isinstance(other, BoundTenantContext)
            and other.tenant_id == self.tenant_id
            and other.domain_id == self.domain_id
        )


@dataclass(frozen=True)
class PrefixedTenantContext:
    """Consumer-controlled prefix convention via a format-string template.

    ``prefix_pattern`` is a Python format string accepting ``tenant_id``,
    ``domain_id``, and ``operation`` placeholders. The formatted string
    becomes the lock-key middle segment and the state-key prefix.

    Use case: a consumer migrating from a legacy convention (e.g.
    ``"{tenant_id}-{domain_id}/"`` rather than the
    :class:`BoundTenantContext` default ``"tenants/{tenant_id}/_state/"``)
    supplies their own pattern via this impl rather than waiting for a
    per-convention reference impl.

    The pattern is formatted once per accessor call; implementations MUST
    NOT mutate the pattern after construction. The pattern is validated at
    construction time (an unknown placeholder or malformed braces raise
    ``ValueError`` immediately, rather than deep inside a later lock /
    state-key call).
    """

    tenant_id: str
    domain_id: str
    prefix_pattern: str

    def __post_init__(self) -> None:
        # Fail fast on a bad template instead of at first lock_key /
        # state_key_prefix call. Dry-run the format with the instance's own
        # values across both placeholder sets the accessors use.
        try:
            self.prefix_pattern.format(
                tenant_id=self.tenant_id,
                domain_id=self.domain_id,
                operation="",
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                f"Invalid prefix_pattern {self.prefix_pattern!r}: only "
                f"{{tenant_id}}, {{domain_id}}, and {{operation}} placeholders "
                f"are supported ({exc})."
            ) from exc

    def lock_key(self, operation: str) -> str:
        return (
            f"{operation}:"
            + self.prefix_pattern.format(
                tenant_id=self.tenant_id,
                domain_id=self.domain_id,
                operation=operation,
            )
            + f":{self.domain_id}"
        )

    def state_key_prefix(self) -> str:
        return self.prefix_pattern.format(
            tenant_id=self.tenant_id,
            domain_id=self.domain_id,
            operation="",
        )

    def matches(self, other: TenantContext) -> bool:
        # Compares the raw pattern string, not its projected output: two
        # textually different patterns that happen to project identically do
        # NOT match. Textual identity is the conservative choice for an
        # isolation boundary.
        return (
            isinstance(other, PrefixedTenantContext)
            and other.tenant_id == self.tenant_id
            and other.domain_id == self.domain_id
            and other.prefix_pattern == self.prefix_pattern
        )


@dataclass(frozen=True, eq=False)
class SharedCorpusTenantContext:
    """Tenant-isolated state on a shared content corpus.

    Consumers running multiple tenants over a shared content corpus (the
    same documents indexed for all tenants) but with per-tenant ingest
    state (snapshot + checkpoint) use this context: content operations key
    on ``shared_corpus_id`` so tenants share index space, but state
    operations key on ``tenant_id`` so each tenant's metadata + snapshot
    writes are isolated.

    Equality, hashing, and :meth:`matches` are all keyed on
    ``(tenant_id, shared_corpus_id)`` — the ``domain_id`` field may differ
    between per-tenant "views" of the same shared corpus, and such views
    compare equal and hash equal. This is deliberate: it lets content-keyed
    caches keyed by the context (``dict`` / ``set`` membership) share a single
    entry across views, which is the whole point of the shared corpus. Keeping
    ``__eq__`` / ``__hash__`` aligned with ``matches`` avoids the trap where
    two views ``matches()`` each other yet land in different cache buckets.

    Use case: legal / regulatory deployments where each tenant has its own
    snapshot lineage but the underlying corpus is shared (cost + cache
    efficiency).
    """

    tenant_id: str
    domain_id: str
    shared_corpus_id: str

    def lock_key(self, operation: str) -> str:
        # Lock on (tenant, shared_corpus) — state writes serialize
        # per-tenant within the shared corpus, not per-domain_id.
        return f"{operation}:{self.tenant_id}:{self.shared_corpus_id}"

    def state_key_prefix(self) -> str:
        return f"tenants/{self.tenant_id}/_state/"

    def matches(self, other: TenantContext) -> bool:
        # Equivalence is on (tenant_id, shared_corpus_id), not domain_id —
        # different per-tenant views of the same shared corpus match.
        # Defers to __eq__ so matches / equality / hashing cannot diverge.
        return self == other

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SharedCorpusTenantContext):
            return (
                self.tenant_id == other.tenant_id
                and self.shared_corpus_id == other.shared_corpus_id
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash(
            (
                "SharedCorpusTenantContext",
                self.tenant_id,
                self.shared_corpus_id,
            )
        )


# --------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------- #

_CONTEXT_KINDS = {
    "single": SingleTenantContext,
    "bound": BoundTenantContext,
    "prefixed": PrefixedTenantContext,
    "shared_corpus": SharedCorpusTenantContext,
}


def _infer_context_kind(config: Mapping[str, Any]) -> str:
    """Infer the context kind from which keys are present.

    Most-specific wins: ``shared_corpus_id`` and ``prefix_pattern`` are
    mutually exclusive discriminators (both also require ``tenant_id``); a
    bare ``tenant_id`` selects ``bound``; neither selects ``single``.
    """
    has_corpus = bool(config.get("shared_corpus_id"))
    has_prefix = bool(config.get("prefix_pattern"))
    if has_corpus and has_prefix:
        raise ValueError(
            "Ambiguous tenant-context config: 'shared_corpus_id' and "
            "'prefix_pattern' are mutually exclusive. Set an explicit "
            "'kind' to disambiguate."
        )
    if has_corpus:
        return "shared_corpus"
    if has_prefix:
        return "prefixed"
    if config.get("tenant_id"):
        return "bound"
    return "single"


def create_tenant_context(config: Mapping[str, Any]) -> TenantContext:
    """Build a :class:`TenantContext` from a config mapping.

    ``domain_id`` is always required. The context kind is taken from an
    explicit ``"kind"`` key (one of ``"single"``, ``"bound"``,
    ``"prefixed"``, ``"shared_corpus"``) when present, otherwise inferred
    from which keys are supplied:

    - ``shared_corpus_id`` present → :class:`SharedCorpusTenantContext`
    - ``prefix_pattern`` present → :class:`PrefixedTenantContext`
    - ``tenant_id`` present → :class:`BoundTenantContext`
    - none of the above → :class:`SingleTenantContext`

    Extra keys in ``config`` are ignored, so a single deployment-config
    section can carry knobs the chosen impl does not consume.

    Inference is by truthiness: a present-but-falsy discriminator (e.g.
    ``tenant_id: None`` or ``prefix_pattern: ""``) is treated as absent, so a
    config section that always carries an optional ``tenant_id`` key still
    yields a single-tenant context when that value is ``None``/empty. To force
    a specific impl — and get a clear "requires a non-empty ..." error when a
    required field is falsy — set an explicit ``kind``.

    Raises ``ValueError`` for a missing ``domain_id``, an unknown explicit
    ``kind``, or a kind whose required fields are absent.
    """
    domain_id = config.get("domain_id")
    if not domain_id:
        raise ValueError(
            "tenant-context config requires a non-empty 'domain_id'."
        )

    kind = config.get("kind") or _infer_context_kind(config)
    if kind not in _CONTEXT_KINDS:
        raise ValueError(
            f"Unknown tenant-context kind: {kind!r}. "
            f"Available kinds: {sorted(_CONTEXT_KINDS)}."
        )

    if kind == "single":
        return SingleTenantContext(domain_id=domain_id)

    tenant_id = config.get("tenant_id")
    if not tenant_id:
        raise ValueError(
            f"tenant-context kind {kind!r} requires a non-empty "
            f"'tenant_id'."
        )

    if kind == "bound":
        return BoundTenantContext(tenant_id=tenant_id, domain_id=domain_id)

    if kind == "prefixed":
        prefix_pattern = config.get("prefix_pattern")
        if not prefix_pattern:
            raise ValueError(
                "tenant-context kind 'prefixed' requires a non-empty "
                "'prefix_pattern'."
            )
        return PrefixedTenantContext(
            tenant_id=tenant_id,
            domain_id=domain_id,
            prefix_pattern=prefix_pattern,
        )

    # kind == "shared_corpus"
    shared_corpus_id = config.get("shared_corpus_id")
    if not shared_corpus_id:
        raise ValueError(
            "tenant-context kind 'shared_corpus' requires a non-empty "
            "'shared_corpus_id'."
        )
    return SharedCorpusTenantContext(
        tenant_id=tenant_id,
        domain_id=domain_id,
        shared_corpus_id=shared_corpus_id,
    )


def tenant_context_from_env(
    *,
    environ: Mapping[str, str] | None = None,
) -> TenantContext:
    """Build a :class:`TenantContext` from environment variables.

    Reads ``DOMAIN_ID`` (required), ``TENANT_ID`` (optional),
    ``TENANT_PREFIX_PATTERN`` (optional), and ``TENANT_SHARED_CORPUS_ID``
    (optional), then dispatches through :func:`create_tenant_context`. With
    only ``DOMAIN_ID`` set the result is a :class:`SingleTenantContext`;
    adding ``TENANT_ID`` yields a :class:`BoundTenantContext` (or the
    prefixed / shared-corpus impl when the matching variable is also set).

    Pass ``environ`` to read from a mapping other than ``os.environ`` (for
    tests or layered config). Raises ``ValueError`` when ``DOMAIN_ID`` is
    unset.

    Note that these variable names are unprefixed (``DOMAIN_ID`` / ``TENANT_ID``
    rather than a namespaced ``DK_*`` form). In a deployment whose environment
    already defines a generic ``DOMAIN_ID`` / ``TENANT_ID`` for an unrelated
    purpose, read the tenant scope from explicit config via
    :func:`create_tenant_context` and pass a scoped ``environ`` mapping here
    rather than relying on the ambient process environment.
    """
    env = environ if environ is not None else os.environ
    config: dict[str, Any] = {}
    domain_id = env.get("DOMAIN_ID")
    if domain_id:
        config["domain_id"] = domain_id
    if env.get("TENANT_ID"):
        config["tenant_id"] = env["TENANT_ID"]
    if env.get("TENANT_PREFIX_PATTERN"):
        config["prefix_pattern"] = env["TENANT_PREFIX_PATTERN"]
    if env.get("TENANT_SHARED_CORPUS_ID"):
        config["shared_corpus_id"] = env["TENANT_SHARED_CORPUS_ID"]
    if "domain_id" not in config:
        raise ValueError(
            "tenant_context_from_env requires the DOMAIN_ID environment "
            "variable to be set."
        )
    return create_tenant_context(config)
