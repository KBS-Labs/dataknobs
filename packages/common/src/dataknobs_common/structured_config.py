"""Typed-configuration base class and consumer-side dispatch mixin.

``StructuredConfig`` is the typed successor to
``dataknobs_config.builders.ConfigurableBase``. It enforces
dataclass-ness, provides auto-derived ``from_dict`` / ``to_dict`` via
``dataclasses.fields()`` introspection, and offers a per-class
``_normalize_dict`` override hook for cases that need pre-projection
dict normalization (e.g., Postgres connection-string assembly).

``StructuredConfigConsumer[ConfigT]`` is the mixin for classes
constructed from a ``StructuredConfig`` subclass. It provides
typed/dict/loose-kwarg dispatch in a single ``__init__``, a typed
``self.config`` property, a one-line ``cls.from_config`` classmethod,
and a ``_setup()`` hook for subclass-specific initialization.

These primitives generalize the per-backend hand-rolled pattern
shipped for the four event-bus backends. Downstream consumers in the
data, vector, and bots packages plug into them in place of the
``ConfigurableBase`` kwarg-splat predecessor.

Relationship to ``Serializable``:
    ``StructuredConfig`` instances structurally satisfy the
    ``Serializable`` protocol (``to_dict`` / ``from_dict`` are both
    present). The two are complementary — ``Serializable`` is the
    right tool for data interchange (``Record``, ``Field``,
    provenance, audit logs); ``StructuredConfig`` is the right tool
    for configuration. No nominal inheritance — the structural
    relationship is sufficient.

Relationship to ``ConfigurableBase``:
    ``StructuredConfigConsumer`` is the typed successor. Existing
    ``ConfigurableBase`` adopters keep working until that class is
    removed in a future release.

Environment-variable substitution:
    Consumers that want ``${VAR}`` expansion before field projection
    call ``substitute_env_vars(raw_dict)`` (from ``dataknobs_config``)
    themselves and pass the result to ``cls.from_dict(...)``. The
    helper is intentionally not invoked inside ``StructuredConfig``
    to keep ``dataknobs-common`` from depending on
    ``dataknobs-config``.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import logging
import threading
import types
from collections.abc import Callable, Mapping
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.registry import Registry
from dataknobs_common.serialization import jsonify

if TYPE_CHECKING:
    # ``Self`` is referenced only in (lazy, ``from __future__``-stringized)
    # annotations, so it is never evaluated at runtime. Guarding the import
    # keeps this zero-dependency package from importing ``typing_extensions``
    # at module load. ``typing_extensions`` (not ``typing``) is the source
    # because mypy's ``python_version = "3.10"`` target predates
    # ``typing.Self``; mypy resolves the guarded import regardless.
    from typing_extensions import Self

logger = logging.getLogger(__name__)

# ``X | None`` (PEP 604) yields ``types.UnionType``; ``Optional[X]`` /
# ``Union[...]`` yield ``typing.Union``. Both must be recognised as unions
# when decomposing field annotations.
_UNION_ORIGINS = (Union, types.UnionType)
# Container origins whose element type(s) may themselves be (or contain) a
# ``StructuredConfig`` subclass and therefore need element-wise coercion.
_SEQUENCE_ORIGINS = (list, tuple, set, frozenset)

# Interior ``Mapping``/``list`` keys whose (truthy) values ``_redacted_repr``
# masks regardless of the parent field name. Polymorphic/discriminated config
# sections (``vector_store``, ``embedding``, ``llm``, ...) are held as raw
# mappings whose schema is owned by another package and dispatched by a
# discriminator key, so the parent has no typed field to redact — key-name is
# the only handle available at that layer (exactly as the subsystem registries
# dispatch on a key-name discriminator). These are matched case-insensitively
# and *exactly* (not as a substring), so a benign ``token_count`` is never
# masked.
#
# Why these names and not the bare generics ``token`` / ``secret``: interior
# matching targets raw dict sections whose schema is owned by *another* package
# (``vector_store``, ``embedding``, ``llm``). A false positive there masks a
# benign key the consumer cannot rename (it's a third-party schema) and cannot
# remove from this set (the default set is frozen, ``_SENSITIVE_FIELDS`` only
# adds). So the default set holds only names that are almost never benign. Bare
# ``token`` (NLP tokens, pagination/page tokens, CSRF tokens) and bare
# ``secret`` (a ``secret`` boolean flag, a ``secret_name`` *reference* to a
# vault) are deliberately excluded in favour of the unambiguous compound names
# below.
#
# A class needing a non-default interior key masked adds it to its
# ``_SENSITIVE_FIELDS``; a process that needs one masked globally across opaque
# sections calls :func:`register_sensitive_interior_key`. Both only *add* —
# there is no remove (redaction is fail-closed). Display-only: ``to_dict`` is
# never redacted.
_DEFAULT_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "password",
        "connection_string",
        "client_secret",
        "secret_key",
        "secret_access_key",
        "access_token",
        "refresh_token",
        "auth_token",
        "bearer_token",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
    }
)

# Process-global interior keys registered at runtime via
# :func:`register_sensitive_interior_key`. Add-only (never removed) so a
# consumer cannot configuration-disable credential masking. Stored lowercase to
# match the case-insensitive interior comparison; folded into the per-class
# redaction set by :func:`_interior_sensitive_keys`, whose cache is cleared on
# every registration.
_EXTRA_SENSITIVE_INTERIOR_KEYS: set[str] = set()

# Guards the read-check-update-clear of ``_EXTRA_SENSITIVE_INTERIOR_KEYS`` in
# ``register_sensitive_interior_key`` and the snapshot read in
# ``_interior_sensitive_keys``. Registration is typically a startup-time call,
# but a ``repr`` (via ``_interior_sensitive_keys``) can run on any thread; the
# lock keeps a concurrent registration from raising ``RuntimeError: Set changed
# size during iteration`` while the snapshot ``frozenset(...)`` iterates the
# live set. (The add-only invariant holds without the lock — ``update`` happens
# before ``cache_clear``, so a recompute always sees the new keys — but the
# iteration race is a real, if narrow, robustness defect in shared infra.)
_sensitive_keys_lock = threading.Lock()


def register_sensitive_interior_key(*names: str) -> None:
    """Add interior key name(s) to the process-global redaction set.

    Extends the set of keys masked inside raw ``Mapping``/``list`` config
    sections (see :func:`_redact_value`) for *every* ``StructuredConfig``
    subclass — the global counterpart to a per-class ``_SENSITIVE_FIELDS``
    entry. Intended for a consumer with custom opaque sections whose
    credential key is not one of the unambiguous defaults (e.g. a private
    ``vault_ref``).

    Add-only by design: there is no corresponding removal. Redaction is a
    fail-closed security feature — configuration must never be able to turn
    credential masking *off*. Names are stored lowercase (interior matching is
    case-insensitive) and matched exactly, not as substrings. Affects display
    only; ``to_dict`` is never redacted.

    Idempotent and cheap: registering an already-present name is a no-op, and
    the per-class :func:`_interior_sensitive_keys` cache is invalidated only
    when the set actually changes.
    """
    added = {name.lower() for name in names if name}
    with _sensitive_keys_lock:
        if added - _EXTRA_SENSITIVE_INTERIOR_KEYS:
            _EXTRA_SENSITIVE_INTERIOR_KEYS.update(added)
            _interior_sensitive_keys.cache_clear()


@functools.cache
def _interior_sensitive_keys(cls: type) -> frozenset[str]:
    """Per-class interior-key redaction set, computed once and cached.

    Unions the frozen module defaults, any runtime-registered extras
    (:func:`register_sensitive_interior_key`), and the class's own
    ``_SENSITIVE_FIELDS`` (lower-cased so interior matching is
    case-insensitive). ``_SENSITIVE_FIELDS`` is a class-definition-time
    ``ClassVar`` and the defaults are frozen, so the result is stable per class
    except across a runtime registration — which clears this cache. Mirrors the
    per-class caching of :func:`_resolved_hints`, keeping the union off the
    ``repr`` hot path.
    """
    sensitive_fields = getattr(cls, "_SENSITIVE_FIELDS", frozenset())
    # Snapshot the live mutable set under the lock so a concurrent
    # ``register_sensitive_interior_key`` cannot mutate it mid-iteration.
    with _sensitive_keys_lock:
        extras = frozenset(_EXTRA_SENSITIVE_INTERIOR_KEYS)
    return (
        _DEFAULT_SENSITIVE_KEYS
        | extras
        | frozenset(name.lower() for name in sensitive_fields)
    )


# Default (and minimum) safety bound on ``_redact_value`` recursion. Config
# sections are shallow (the deepest known shape nests ~2-3 levels); 6 is ample
# headroom while bounding pathological or cyclic structures. Beyond the bound
# the value renders via the caller's plain ``repr`` — a stop, not an expected
# path. A subclass with an unusually deep raw section may *raise* its effective
# bound via the ``_MAX_REDACT_DEPTH`` ClassVar; it can never go below this floor
# (lowering would reduce credential protection — redaction is fail-closed), a
# constraint enforced in :meth:`StructuredConfig.__init_subclass__`.
_DEFAULT_MAX_REDACT_DEPTH = 6


def _redact_value(
    value: Any, sensitive_keys: frozenset[str], depth: int, max_depth: int
) -> Any:
    """Build a display copy of ``value`` with sensitive interior keys masked.

    Descends into ``Mapping`` and ``list``/``tuple`` values, replacing the
    value under any key in ``sensitive_keys`` (matched case-insensitively and
    exactly, not as a substring) whose value is truthy with ``'***'``, and
    recursing into every other value. A falsy nested credential (``None`` /
    ``""``) renders verbatim — absence is not a secret. Scalars (and anything
    that is not a mapping/sequence) are returned unchanged, so a config with
    no sensitive interior keys renders byte-for-byte as the plain-dataclass
    repr would.

    The real config value is never mutated — the returned copy is for
    ``repr`` only. Recursion stops at ``max_depth`` (the calling config's
    ``_MAX_REDACT_DEPTH``), beyond which ``value`` is returned as-is (the
    caller renders it via plain ``repr``).
    """
    if depth >= max_depth:
        return value
    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and k.lower() in sensitive_keys and v:
                redacted[k] = "***"
            else:
                redacted[k] = _redact_value(v, sensitive_keys, depth + 1, max_depth)
        return redacted
    if isinstance(value, list):
        return [_redact_value(v, sensitive_keys, depth + 1, max_depth) for v in value]
    # Only a plain ``tuple`` is rebuilt as a tuple; a ``tuple`` subclass
    # (e.g. a namedtuple) can't be faithfully reconstructed from an iterable,
    # so it is left verbatim rather than risk altering its repr.
    if type(value) is tuple:
        return tuple(
            _redact_value(v, sensitive_keys, depth + 1, max_depth) for v in value
        )
    return value


@functools.cache
def _resolved_hints(cls: type) -> Mapping[str, Any]:
    """Cache of ``typing.get_type_hints(cls)``, degrading to ``{}``.

    Because the package uses ``from __future__ import annotations``,
    ``dataclasses.fields(cls)[i].type`` is a *string*; type-aware field
    coercion needs the resolved runtime types. ``get_type_hints`` resolves
    them against the class's module globals. If resolution fails (an
    annotation referencing a name unavailable at runtime — e.g. a
    ``TYPE_CHECKING``-only import), we degrade to ``{}`` so ``from_dict``
    falls back to the pre-coercion pass-through behaviour rather than
    crashing.
    """
    try:
        return get_type_hints(cls)
    except (NameError, TypeError, AttributeError) as exc:
        # An annotation that can't be resolved at runtime — typically a
        # name available only under ``TYPE_CHECKING`` — surfaces as
        # ``NameError``; malformed/incompatible annotations as
        # ``TypeError`` / ``AttributeError``. Degrade to ``{}`` so
        # ``from_dict`` falls back to the pre-coercion pass-through
        # behaviour rather than crashing, and log at debug so the dropped
        # coercion is diagnosable.
        logger.debug(
            "Could not resolve type hints for %s (%s); nested-config "
            "coercion will be skipped for its fields.",
            getattr(cls, "__qualname__", cls),
            exc,
        )
        return {}


def _type_is_coercible(declared: Any) -> bool:
    """True if ``declared`` is (or its type graph contains) a coercible type.

    A type is *coercible* when projecting a raw value onto it can produce a
    typed instance: a ``StructuredConfig`` subclass (rebuilt from its dict
    shape) or an ``Enum`` subclass (rebuilt from its member value). Both
    recurse through container element types and union arms.

    Used as a fast gate in ``from_dict``: a field whose declared type does
    not reference any coercible type is assigned verbatim, preserving the
    flat-projection behaviour for plain scalars (no new objects, identical
    identity for pass-through values). Only fields that actually nest a
    config or an enum incur the recursive coercion path.
    """
    if isinstance(declared, type):
        return issubclass(declared, (StructuredConfig, Enum))
    origin = get_origin(declared)
    if origin is None:
        return False
    return any(
        _type_is_coercible(arg)
        for arg in get_args(declared)
        if arg is not type(None)
    )


def _coerce_field(declared: Any, value: Any) -> Any:
    """Project a raw ``value`` onto its declared field type, recursively.

    Handles the nesting shapes documented on :meth:`StructuredConfig.from_dict`
    — a ``StructuredConfig`` subclass, ``Optional`` of one, and homogeneous
    ``list``/``tuple``/``set``/``dict`` containers of them (including
    ``dict[K, list[SubCfg]]``). Anything that is already a typed instance
    passes through untouched (idempotence), as does any shape the coercion
    rules don't recognise — the dataclass ctor then validates it.
    """
    # Direct ``StructuredConfig`` subclass.
    if isinstance(declared, type) and issubclass(declared, StructuredConfig):
        if isinstance(value, declared):
            return value
        if isinstance(value, Mapping):
            return declared.from_dict(value)
        return value

    # Direct ``Enum`` subclass: map a raw member value (e.g. the string
    # ``"retry"`` loaded from YAML/JSON) to its member. An existing member
    # passes through untouched (idempotence, including ``StrEnum`` /
    # ``IntEnum`` whose members are themselves str/int). An unrecognised
    # value falls through unchanged so the dataclass ctor / ``__post_init__``
    # surfaces it, rather than this raising a less-contextual ``ValueError``.
    if isinstance(declared, type) and issubclass(declared, Enum):
        if isinstance(value, declared):
            return value
        try:
            return declared(value)
        except (ValueError, KeyError):
            return value

    origin = get_origin(declared)
    if origin is None:
        return value

    args = get_args(declared)

    # ``Optional[SubCfg]`` / ``Mode | None``. Only auto-coerce when exactly
    # one non-``None`` arm is coercible (a ``StructuredConfig`` or ``Enum``).
    # A union with several coercible arms is a discriminated/polymorphic
    # shape — selecting among them by inspecting the data is deliberately
    # out of scope here (it belongs in the subsystem registry /
    # object-graph layer), so the value passes through untouched rather
    # than silently binding to whichever arm happens to be declared first.
    if origin in _UNION_ORIGINS:
        if value is None:
            return value
        coercible_arms = [
            arg
            for arg in args
            if arg is not type(None)
            and _type_is_coercible(arg)
        ]
        if len(coercible_arms) != 1:
            return value
        return _coerce_field(coercible_arms[0], value)

    # ``list``/``tuple``/``set``/``frozenset`` of configs.
    if origin in _SEQUENCE_ORIGINS:
        if not args or not isinstance(value, (list, tuple, set, frozenset)):
            return value
        if origin is tuple and not (len(args) == 2 and args[1] is Ellipsis):
            # Fixed-length tuple: coerce positionally.
            coerced_items = [
                _coerce_field(args[i], v) if i < len(args) else v
                for i, v in enumerate(value)
            ]
        else:
            elem_type = args[0]
            coerced_items = [_coerce_field(elem_type, v) for v in value]
        # Build the container from the *declared* origin (list/tuple/set/
        # frozenset), not the raw value's type — a JSON list projected onto
        # a ``tuple[...]`` field must come back a tuple.
        return origin(coerced_items)

    # ``dict[K, SubCfg]`` / ``dict[K, list[SubCfg]]``.
    if isinstance(origin, type) and issubclass(origin, Mapping):
        if len(args) != 2 or not isinstance(value, Mapping):
            return value
        val_type = args[1]
        return {k: _coerce_field(val_type, v) for k, v in value.items()}

    return value


@dataclasses.dataclass(frozen=True)
class StructuredConfig:
    """Base class for typed, dict-loadable, frozen configuration dataclasses.

    Subclasses are themselves ``@dataclass(frozen=True)``; this base
    provides auto-derived ``from_dict`` / ``to_dict`` via
    ``dataclasses.fields()`` introspection. Drift between the dataclass
    fields and the dict-shape projection becomes structurally
    impossible — the field set IS the projection.

    Per-class invariants stay in ``__post_init__`` (e.g., validating a
    field is non-empty). Per-class dict-shape normalization (e.g.,
    routing a config dict through
    ``normalize_postgres_connection_config``) goes in
    ``_normalize_dict``; do NOT override ``from_dict``.

    Nested-config composition: ``from_dict`` recurses into a field whose
    declared type is (or contains) a ``StructuredConfig`` subclass —
    ``SubCfg``, ``SubCfg | None``, ``list[SubCfg]``,
    ``tuple[SubCfg, ...]``, ``set[SubCfg]``, ``frozenset[SubCfg]``,
    ``dict[K, SubCfg]``, and ``dict[K, list[SubCfg]]`` are all rebuilt
    from their raw dict shape. Recursion is bounded by the static
    field-type graph (not by runtime data), so a config whose fields are
    all scalars terminates immediately. A field already holding a typed
    instance passes through unchanged. ``Enum``-typed fields coerce the
    same way — a raw member value (``"fast"`` from YAML) becomes its
    member (``Mode.FAST``), through the same container/``| None`` shapes;
    :meth:`to_json_dict` is the symmetric output side. Polymorphic/discriminated
    selection (a section whose concrete type is chosen by a discriminator
    key) is deliberately NOT handled here — that stays in the subsystem
    registry / object-graph layer; type a polymorphic field as its
    abstract sub-config and dispatch in the consumer's factory. A union
    of several concrete sub-configs (``A | B`` where both are
    ``StructuredConfig`` subclasses) is likewise left untouched — only a
    single-config ``SubCfg | None`` is auto-coerced.

    Secret redaction: a subclass carrying a credential-bearing field
    (an API key, a connection string with an embedded password) lists
    those field names in ``_SENSITIVE_FIELDS``::

        _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset({"api_key"})

    That declaration is the *only* thing required — redaction is
    automatic. A ``__repr__`` that masks every listed field (and renders
    every other field exactly as the dataclass would) is installed on
    each subclass by :meth:`__init_subclass__`, which runs *before* the
    subclass's ``@dataclass`` decorator and writes the repr into the
    subclass ``__dict__`` so the decorator's own ``repr=True`` generation
    is suppressed for it. This sidesteps both failure modes of the naive
    approaches: a dataclass-*generated* repr cannot be redacted, and an
    *inherited* redacting repr would be shadowed by the generated repr of
    any intermediate ``@dataclass`` base in the MRO. A non-empty value of
    a listed field renders as ``'***'``; ``None`` and ``""`` (an unset
    credential) render verbatim, since absence is not a secret.
    ``to_dict`` is deliberately *not* redacted (it must round-trip back
    into an identical config) — redaction is display-only, keeping
    secrets out of logs that interpolate ``repr(config)``, tracebacks,
    and pytest failure output.

    Redaction also descends into raw ``Mapping``/``list`` field values:
    a credential nested inside an intentionally-untyped polymorphic
    section (``vector_store``, ``embedding``, ``llm`` — sections held as
    raw dicts because their schema is owned by another package and
    dispatched by a discriminator key) is masked by *interior key name*.
    The interior key set is ``_DEFAULT_SENSITIVE_KEYS`` (``api_key``,
    ``password``, ``connection_string``, the compound ``*_secret*`` /
    ``*_token`` credential names, and the AWS keys) unioned with the
    class's ``_SENSITIVE_FIELDS`` and any process-global keys registered
    via :func:`register_sensitive_interior_key` — so the known leaks are
    closed with zero per-class configuration. The default set holds only
    unambiguous credential names: the bare generics ``token`` and
    ``secret`` are excluded because a false positive masks a benign key
    inside a third-party opaque section that the consumer can neither
    rename nor remove from the (frozen, add-only) set. Interior matching is
    exact and case-insensitive (a benign ``token_count`` is untouched) and
    truthy-only, depth-bounded, and likewise display-only. The depth bound
    defaults to ``_DEFAULT_MAX_REDACT_DEPTH`` (6, ample for the ~2-3-level
    real shapes); a subclass with an unusually deep raw section may raise it
    via the ``_MAX_REDACT_DEPTH`` ClassVar but never lower it below the floor
    (lowering would shrink the masked region — fail-closed). This is the
    redaction-side complement to the subsystem registries: both dispatch
    on a key-name handle because the section schema is opaque at the
    parent layer. See :func:`_redact_value`.

    A subclass that needs a genuinely custom ``__repr__`` may still
    define one in its body; :meth:`__init_subclass__` leaves an
    explicitly-defined repr untouched.
    """

    #: Field names masked by :meth:`_redacted_repr`. Empty by default,
    #: so a config with no secrets behaves exactly like a plain
    #: dataclass.
    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset()

    #: Maximum interior-redaction recursion depth (see :func:`_redact_value`).
    #: Defaults to the module floor; a subclass with an unusually deep raw
    #: section may *raise* it (``_MAX_REDACT_DEPTH: ClassVar[int] = 8``) but
    #: never lower it below the floor — lowering would reduce credential
    #: protection, which :meth:`__init_subclass__` rejects.
    _MAX_REDACT_DEPTH: ClassVar[int] = _DEFAULT_MAX_REDACT_DEPTH

    #: Maps a raw-dict field name -> the binding name registered in
    #: :data:`config_registries`. Empty by default, so a config with no
    #: polymorphic sections has a no-op :meth:`validate`. String-valued by
    #: design: the parent declares *which registry* governs the field
    #: WITHOUT importing the child config type, so adopting validation adds
    #: no cross-package type-surface coupling — the binding is a string
    #: literal plus a runtime resolver registration in the owning package.
    #: A frozen ``MappingProxyType`` default so an accidental
    #: ``Base._polymorphic_fields[...] = ...`` raises instead of silently
    #: mutating the shared base default for every non-overriding subclass.
    #: See :meth:`validate`.
    _polymorphic_fields: ClassVar[Mapping[str, str]] = types.MappingProxyType({})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Install the redacting ``__repr__`` and validate ``_MAX_REDACT_DEPTH``.

        Runs at class-creation time, *before* the subclass's
        ``@dataclass`` decorator is applied. Writing ``__repr__`` into the
        subclass ``__dict__`` here makes ``dataclasses``' ``repr=True``
        generation a no-op for it — ``dataclasses._set_new_attribute``
        skips an attribute already present in the class dict — so every
        ``StructuredConfig`` subclass renders through
        :meth:`_redacted_repr` and honours ``_SENSITIVE_FIELDS`` with no
        per-leaf boilerplate. A subclass that defines its own ``__repr__``
        in its body keeps it (it is already in ``cls.__dict__`` when this
        runs). See the class docstring for why neither an inherited repr
        nor the generated one can do this on their own.

        Also enforces the raise-only floor on ``_MAX_REDACT_DEPTH``: a
        subclass may raise its effective interior-redaction depth above the
        module default but never lower it below
        ``_DEFAULT_MAX_REDACT_DEPTH``. Lowering would shrink the masked
        region — a configuration-time reduction of credential protection,
        which redaction's fail-closed posture forbids. A non-int or
        below-floor value raises ``ValueError`` at class definition, so the
        misuse surfaces immediately rather than as a silent leak at repr time.
        """
        super().__init_subclass__(**kwargs)
        if "__repr__" not in cls.__dict__:
            # Assigning ``__repr__`` here puts an entry in ``cls.__dict__``
            # *before* the ``@dataclass`` decorator runs, so dataclass sees a
            # user-defined repr and skips generating one — that is what lets the
            # redacting repr win over the generated one.
            cls.__repr__ = StructuredConfig._redacted_repr
        # Validate an explicitly-overridden depth only (an inherited value is
        # already known-valid). ``bool`` is an ``int`` subclass but a
        # nonsensical depth, so reject it explicitly.
        if "_MAX_REDACT_DEPTH" in cls.__dict__:
            depth = cls.__dict__["_MAX_REDACT_DEPTH"]
            if (
                not isinstance(depth, int)
                or isinstance(depth, bool)
                or depth < _DEFAULT_MAX_REDACT_DEPTH
            ):
                raise ValueError(
                    f"{cls.__qualname__}._MAX_REDACT_DEPTH must be an int "
                    f">= {_DEFAULT_MAX_REDACT_DEPTH} (the fail-closed floor); "
                    f"got {depth!r}. The depth bound may be raised for an "
                    "unusually deep raw section but never lowered — lowering "
                    "reduces credential masking."
                )

    def _redacted_repr(self) -> str:
        """Dataclass-style repr that masks sensitive values, scalar and nested.

        Mirrors the auto-generated dataclass repr for every ``repr=True``
        field, redacting at two levels so credentials never reach logs
        through ``repr(config)`` or an f-string:

        1. **Scalar field-name redaction** — a field whose *name* is in
           ``_SENSITIVE_FIELDS`` and whose value is truthy renders as
           ``'***'`` (the original mechanism). A falsy value (``None`` /
           ``""``) is shown verbatim — an unset credential is not a secret,
           and masking ``""`` would falsely imply one is configured.
        2. **Nested interior-key redaction** — for every other field, the
           value is rendered through :func:`_redact_value`, which descends
           into raw ``Mapping``/``list`` values and masks any interior key in
           the per-class set from :func:`_interior_sensitive_keys`
           (``_DEFAULT_SENSITIVE_KEYS``, the class's ``_SENSITIVE_FIELDS``,
           and any runtime-registered extras — exact, case-insensitive,
           truthy-only). This reaches credentials nested
           inside the intentionally-untyped polymorphic sections
           (``vector_store``/``embedding``/``llm``) that field-name redaction
           alone cannot see — the display-side complement to the subsystem
           registries on the construction side.

        Display-only: the real field values and ``to_dict`` are untouched, so
        round-trip is preserved exactly. ``type(self).__qualname__`` (matching
        the dataclass-generated repr) keeps this a byte-for-byte drop-in for
        configs with no sensitive content. Installed as the ``__repr__`` of
        every subclass by :meth:`__init_subclass__`.
        """
        sensitive = _interior_sensitive_keys(type(self))
        max_depth = self._MAX_REDACT_DEPTH
        parts: list[str] = []
        for f in dataclasses.fields(self):
            if not f.repr:
                continue
            value = getattr(self, f.name)
            if f.name in self._SENSITIVE_FIELDS and value:
                rendered: Any = "***"
            else:
                rendered = _redact_value(value, sensitive, 0, max_depth)
            parts.append(f"{f.name}={rendered!r}")
        return f"{type(self).__qualname__}({', '.join(parts)})"

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> Self:
        """Build an instance from a config dict via field introspection.

        Each ``@dataclass`` field on ``cls`` is read from ``config`` by
        name. Fields absent from ``config`` use their declared default
        (or ``default_factory``). Keys in ``config`` that don't match
        any field are ignored — registry-routing keys like
        ``"backend"`` pass through cleanly.

        A field whose declared type is (or contains) a
        ``StructuredConfig`` subclass is rebuilt recursively from its raw
        dict shape (see the class docstring's nested-composition note). A
        field whose declared type is (or contains) an ``Enum`` subclass is
        rebuilt from its member value — so ``{"mode": "fast"}`` loaded from
        YAML/JSON becomes ``Mode.FAST``, not the bare string. An
        unrecognised enum value passes through unchanged (the ctor /
        ``__post_init__`` surfaces it). All other fields are assigned
        verbatim.

        Pre-projection normalization happens in
        :meth:`_normalize_dict`; override that, not ``from_dict``.

        Args:
            config: Source dict. Shallow-copied before normalization;
                the caller's dict is not mutated.

        Returns:
            A new ``cls`` instance with fields populated from
            ``config`` (and defaults where keys are absent).
        """
        normalized = cls._normalize_dict(dict(config))
        hints = _resolved_hints(cls)
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name not in normalized:
                continue
            declared = hints.get(f.name)
            value = normalized[f.name]
            if declared is not None and _type_is_coercible(declared):
                kwargs[f.name] = _coerce_field(declared, value)
            else:
                kwargs[f.name] = value
        return cls(**kwargs)

    @classmethod
    def _normalize_dict(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """Per-class hook for dict-shape normalization before projection.

        Default identity. Override when the input dict's shape differs
        from the field set (e.g., Postgres backends that accept
        ``DATABASE_URL`` and ``POSTGRES_*`` env-var fallbacks and need
        to be routed through ``normalize_postgres_connection_config``
        before field projection).

        The argument is a shallow copy of the caller's dict — overrides
        may mutate it freely.
        """
        return raw

    def to_dict(self) -> dict[str, Any]:
        """Symmetric serialization. Delegates to ``dataclasses.asdict``.

        Round-trip property: ``type(cfg).from_dict(cfg.to_dict()) == cfg``
        holds for flat configs and for nested configs alike. ``asdict``
        recurses into nested dataclasses and ``from_dict`` recurses back
        into the matching field types (see the class docstring's
        nested-composition note), so the two are symmetric for every
        statically-typed nesting shape. Verified by
        :func:`dataknobs_common.testing.assert_structured_config_roundtrip`.

        This is the *in-process* serialization: ``Enum`` fields render as
        their members (not their ``.value``) and any live callables/types
        round-trip by identity, so the output is not necessarily
        JSON-serialisable. Note that ``IntEnum`` / ``StrEnum`` members *are*
        instances of their ``int`` / ``str`` base and so survive
        ``json.dumps`` as-is; only a plain ``Enum`` member is non-native.
        Use :meth:`to_json_dict` when any field may hold a plain ``Enum``
        (or when the field type is uncertain) and you need a dict that
        survives ``json.dumps`` and reloads via ``from_dict``.
        """
        return dataclasses.asdict(self)

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-serialisable projection: like :meth:`to_dict`, enums as values.

        Identical to :meth:`to_dict` except every ``Enum`` member (at any
        nesting depth, including inside list/tuple/dict containers and
        nested configs) is replaced by its ``.value``. Because
        :meth:`from_dict` coerces those raw values back to members, the
        round-trip property holds through JSON too:
        ``type(cfg).from_dict(json.loads(json.dumps(cfg.to_json_dict()))) == cfg``
        — for configs whose remaining field values are JSON-native.

        Configs carrying live callables or ``type`` objects (e.g. a
        ``fallback_function`` hook or an exception-type list) still hold
        those verbatim and are no more JSON-serialisable here than via
        :meth:`to_dict`; enum normalisation is the one transformation this
        adds. ``set`` / ``frozenset`` fields are likewise left untouched
        (JSON has no set type). Delegates the enum normalisation to the
        shared :func:`dataknobs_common.serialization.jsonify` utility.
        """
        return jsonify(dataclasses.asdict(self))

    def validate(self) -> None:
        """Validate polymorphic raw-dict sections by dry-run construction.

        ``from_dict`` is deliberately tolerant: it ignores keys that match
        no field (registry-routing keys like ``backend`` pass through) and
        does not look inside a raw-dict section whose schema is owned by
        another package. ``validate`` is the opt-in companion that closes
        that gap *without constructing the runtime objects* — the use case
        is the gap between parse and construction (CI config-linting,
        write-time validation in a multi-tenant config store, config
        editors). A flow that builds the object immediately after parsing
        already gets a fast failure from the construction factory; this is
        for flows that parse but do not (yet) build.

        For each field named in :attr:`_polymorphic_fields` whose value is
        a non-empty mapping (or a list/tuple of mappings), the concrete
        config class is resolved via :data:`config_registries` and its
        ``from_dict`` is called purely to surface field-level errors — the
        result is discarded (the field stays a raw dict). The dry-run child
        is then itself validated, so a single ``parent.validate()``
        validates the whole polymorphic tree (e.g. a bot config's
        knowledge-base section down to its vector-store section).
        Statically-typed nested ``StructuredConfig`` fields are likewise
        recursed into, so the same single call covers typed nesting too.

        Behavior:

        * ``_polymorphic_fields`` empty (a non-adopter) — no-op.
        * a section value that is empty (``{}`` / ``None``) — skipped
          (it is the default-constructible / ``from_components`` path).
        * the binding name has no registered resolver — skipped with a
          ``logger.debug`` (best-effort: the cause is import order, and the
          test-time guard
          :func:`dataknobs_common.testing.assert_polymorphic_bindings_resolve`
          catches genuine wiring drift in CI rather than at runtime).
        * the resolver returns ``None`` (an unknown discriminator value) —
          raises :class:`~dataknobs_common.exceptions.ConfigurationError`.
          This is the headline win: a typo'd discriminator is caught at
          config-lint time, not at first use.
        * the resolver returns :data:`SKIP_VALIDATION` (the discriminator is
          recognized but the variant exposes no typed config — e.g. a
          bare-callable backend with no ``CONFIG_CLS``) — skipped with a
          ``logger.debug``, never raised. Rejecting a valid, constructible
          backend's config would be a false positive, so this stays
          fail-soft like the unregistered-binding case.
        * the resolver returns a config class — its ``from_dict`` runs
          (surfacing any field-level error from the child's
          ``__post_init__``) and the built child is validated recursively.

        Does not mutate ``self`` and never auto-runs inside ``from_dict``
        or construction — adopters/tools opt in by calling
        ``Cfg.from_dict(raw).validate()``.

        Raises:
            ConfigurationError: If a polymorphic section's discriminator
                names no known variant. Field-level errors from a child
                config's own validation (e.g. a ``ValueError`` from its
                ``__post_init__``) propagate unchanged.
        """
        self._validate(set())

    def _validate(self, visited: set[int]) -> None:
        """Recursive worker for :meth:`validate` with a cycle guard.

        ``visited`` tracks the ``id()`` of every persistent, statically-typed
        nested instance reached from the original ``self``, so a config object
        graph that shares a child instance via multiple paths (or, defensively,
        a cycle built outside ``from_dict``) is validated once instead of
        recursing without bound. The ephemeral dry-run children built inside
        :meth:`_validate_polymorphic_section` deliberately do NOT share this
        set: each starts a fresh ``validate()`` scope, because a discarded
        child's ``id()`` can be reused by the next one and a shared set would
        then skip a section that must be validated.
        """
        marker = id(self)
        if marker in visited:
            return
        visited.add(marker)
        for field_name, binding in self._polymorphic_fields.items():
            value = getattr(self, field_name, None)
            if isinstance(value, Mapping):
                self._validate_polymorphic_section(field_name, binding, value)
            elif isinstance(value, (list, tuple)):
                for element in value:
                    if isinstance(element, Mapping):
                        self._validate_polymorphic_section(
                            field_name, binding, element
                        )
        # Recurse through statically-typed nested ``StructuredConfig``
        # fields (and their containers) so one ``parent.validate()`` covers
        # the whole config tree, not only the polymorphic raw-dict sections.
        for f in dataclasses.fields(self):
            if f.name in self._polymorphic_fields:
                continue
            self._validate_nested(getattr(self, f.name, None), visited)

    @staticmethod
    def _validate_nested(value: Any, visited: set[int]) -> None:
        """Recurse :meth:`validate` into nested ``StructuredConfig`` values."""
        if isinstance(value, StructuredConfig):
            value._validate(visited)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for element in value:
                if isinstance(element, StructuredConfig):
                    element._validate(visited)
        elif isinstance(value, Mapping):
            for element in value.values():
                if isinstance(element, StructuredConfig):
                    element._validate(visited)

    def _validate_polymorphic_section(
        self, field_name: str, binding: str, raw: Mapping[str, Any]
    ) -> None:
        """Dry-run-build one polymorphic section to surface its errors.

        See :meth:`validate` for the full contract. Resolves the section's
        config class via :data:`config_registries`, builds it from ``raw``
        purely to validate (discarding the result), and recurses into the
        built child so the whole tree is checked.
        """
        if not raw:
            return
        resolver = config_registries.get_optional(binding)
        if resolver is None:
            logger.debug(
                "No config resolver registered for binding %r (field "
                "%s.%s); skipping validation of this section. Import the "
                "package that owns the binding to register its resolver.",
                binding,
                type(self).__name__,
                field_name,
            )
            return
        config_cls = resolver(raw)
        if config_cls is SKIP_VALIDATION:
            logger.debug(
                "Resolver for binding %r recognizes the discriminator in "
                "field %s.%s but exposes no typed config; skipping validation "
                "of this section.",
                binding,
                type(self).__name__,
                field_name,
            )
            return
        if config_cls is None:
            raise ConfigurationError(
                f"{type(self).__name__}.{field_name}: this configuration "
                f"does not match any variant registered for '{binding}'; "
                "check the section's discriminator key (e.g. 'backend' / "
                "'provider').",
                context={
                    "config": type(self).__name__,
                    "field": field_name,
                    "binding": binding,
                    "section_keys": sorted(str(k) for k in raw),
                },
            )
        # Dry-run build to surface field-level errors, then recurse. The
        # built child is discarded — the field stays a raw dict, so
        # round-trip and construction paths are unaffected.
        config_cls.from_dict(raw).validate()


ConfigT = TypeVar("ConfigT", bound=StructuredConfig)


class _SkipValidation:
    """Sentinel type for :data:`SKIP_VALIDATION` (see it for semantics).

    Enforced as a singleton: ``validate`` distinguishes the sentinel with an
    ``is SKIP_VALIDATION`` identity check, so a second instance would silently
    fall through to the ``None`` (unknown-discriminator → raise) branch. The
    ``__new__`` guard makes ``_SkipValidation()`` impossible once the module
    constant exists, so the only way to obtain one is the public
    :data:`SKIP_VALIDATION`. This type is private; to annotate a resolver's
    return, use the public :data:`ConfigClassResolution` alias rather than
    importing this name.
    """

    __slots__ = ()
    _instance: ClassVar[_SkipValidation | None] = None

    def __new__(cls) -> Self:
        if cls._instance is not None:
            raise TypeError(
                "Use the SKIP_VALIDATION singleton; do not instantiate "
                "_SkipValidation directly."
            )
        instance = super().__new__(cls)
        cls._instance = instance
        return instance

    def __repr__(self) -> str:
        return "SKIP_VALIDATION"


#: Sentinel a :data:`ConfigClassResolver` returns to mean "I recognize this
#: section's discriminator, but it has no typed :class:`StructuredConfig` to
#: validate against — skip it (do not raise)." This is distinct from a ``None``
#: return, which means "unknown discriminator" and *does* raise. The case
#: arises for construction registries that accept bare-callable backends (no
#: ``CONFIG_CLS`` to read): such a backend is valid and constructible, so
#: rejecting its config at :meth:`StructuredConfig.validate` time would be a
#: false positive. Skipping keeps ``validate`` fail-soft — the same posture it
#: takes for an unregistered binding or an empty section — while preserving the
#: typo-catching ``None`` → raise path.
SKIP_VALIDATION: _SkipValidation = _SkipValidation()


#: The result a :data:`ConfigClassResolver` produces (its return type): the
#: concrete :class:`StructuredConfig` subclass that validates the section;
#: ``None`` when the discriminator names no known variant (surfaced by
#: :meth:`StructuredConfig.validate` as a
#: :class:`~dataknobs_common.exceptions.ConfigurationError`); or
#: :data:`SKIP_VALIDATION` when the discriminator *is* recognized but exposes no
#: typed config to check (skipped, not raised). Exported so a resolver function
#: can annotate its return without importing the private :class:`_SkipValidation`
#: sentinel type.
ConfigClassResolution = type[StructuredConfig] | _SkipValidation | None

#: A resolver maps a polymorphic section's raw dict to a
#: :data:`ConfigClassResolution`. A resolver MUST delegate to the section's own
#: construction registry (e.g. read ``CONFIG_CLS`` off the registered store
#: class) rather than holding an independent discriminator→type table, so
#: validation and construction cannot drift.
ConfigClassResolver = Callable[[Mapping[str, Any]], ConfigClassResolution]

#: Process-global registry of section resolvers, keyed by binding name (the
#: value side of a :attr:`StructuredConfig._polymorphic_fields` entry). The
#: registry-of-registries seam: ``dataknobs-common`` owns it but
#: (``dependencies = []``) cannot populate it, so each package that owns a
#: polymorphic section registers its resolver eagerly at import (e.g.
#: ``dataknobs-data`` registers ``"vector_store"``). The string binding plus a
#: runtime registration is what keeps adoption coupling-free — a parent config
#: names the registry without importing the child config type.
config_registries: Registry[ConfigClassResolver] = Registry(
    "config_section_resolvers"
)


@functools.cache
def _hook_keyword_spec(func: Callable[..., Any]) -> tuple[bool, frozenset[str]]:
    """Keyword-binding spec for a collaborator hook (``_ainit`` / ``_adopt_components``).

    Returns ``(accepts_var_keyword, bindable_names)`` for the unbound hook
    function. ``self`` and positional-only / var-positional params are
    excluded; ``bindable_names`` lists only the parameters a caller can
    supply by keyword. Cached per function (one per consumer class).

    A callable whose signature can't be introspected degrades to
    ``(True, frozenset())`` so delivery falls back to the historical
    pass-everything behaviour rather than silently dropping collaborators.
    (When ``accepts_var_keyword`` is ``True`` the ``names`` set is unused —
    :func:`_components_for_hook` passes everything — so this fallback is
    intentionally indistinguishable from a genuine ``**kwargs`` hook.)
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return True, frozenset()
    accepts_var_keyword = False
    names: set[str] = set()
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_var_keyword = True
        elif param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            names.add(name)
    return accepts_var_keyword, frozenset(names)


def _components_for_hook(
    func: Callable[..., Any], components: Mapping[str, Any]
) -> dict[str, Any]:
    """Select the collaborators a hook can bind by keyword.

    Delivers every component when the hook declares ``**kwargs``; otherwise
    only those whose names match a keyword-bindable parameter. A no-arg
    hook (``_ainit(self)``) receives ``{}``. Either way the full set stays
    on ``self.components`` — this governs only the convenience keyword
    delivery, so a hook that narrows or omits the collaborator parameters
    cannot be crashed by an injected component it does not declare. (A hook
    that declares a collaborator param *without* a default still fails the
    zero-injection path; that drift is pinned by
    :func:`dataknobs_common.testing.assert_structured_config_consumer`.)
    """
    if not components:
        return {}
    accepts_var_keyword, names = _hook_keyword_spec(func)
    if accepts_var_keyword:
        return dict(components)
    selected = {k: v for k, v in components.items() if k in names}
    if len(selected) != len(components):
        # A collaborator the caller injected is not declared by the hook
        # (commonly a misspelled parameter name). It stays on
        # ``self.components`` but is not delivered to the hook — log at
        # debug so the silent drop is diagnosable, mirroring the
        # hint-resolution degrade in ``_resolved_hints``.
        dropped = sorted(set(components) - selected.keys())
        logger.debug(
            "Hook %s does not declare injected collaborator(s) %s; they "
            "remain on self.components but are not delivered to the hook.",
            getattr(func, "__qualname__", func),
            dropped,
        )
    return selected


class StructuredConfigConsumer(Generic[ConfigT]):
    """Mixin for classes constructed from a ``StructuredConfig`` subclass.

    Provides:

    - ``__init__(config: ConfigT | Mapping | None, **kwargs)`` with
      built-in typed/loose/None dispatch. Mixing typed ``config=``
      with loose ``**kwargs`` raises ``TypeError``. Calls
      ``super().__init__()`` so the mixin composes into a cooperative
      multiple-inheritance hierarchy (data backends, vector stores).
    - ``cls.from_config(config, **components) -> Self`` classmethod that
      runs the input through ``CONFIG_CLS.from_dict`` (when given a
      Mapping) then ``cls``.
    - ``cls.from_config_async(config, **components) -> Self`` classmethod:
      the same sync assembly followed by ``await obj._ainit(...)`` with
      signature-aware collaborator delivery — the canonical entry point
      for consumers whose initialization is asynchronous (DBs that connect
      eagerly, LLM-backed bots, KB warmup).
    - ``cls.from_components(config=None, **collaborators) -> Self``
      classmethod for the dual-input shape: assemble from pre-built
      collaborators (skipping the config-driven build) instead of from
      config alone.
    - ``self.config: ConfigT`` typed read-only property — the one
      ``config`` notion across every adopter.
    - ``self.components: Mapping[str, Any]`` read-only view of injected
      collaborators (empty for config-only construction).
    - ``_setup()`` sync hook called after ``self._config`` is
      established; ``_ainit(**components)`` async hook called by
      ``from_config_async`` after ``_setup``; ``_adopt_components(**…)``
      sync hook called by ``from_components``.

    Collaborator injection:

    The construction contract distinguishes two kinds of inputs.
    *Config* (scalar knobs, nested sub-configs) flows through
    ``CONFIG_CLS`` and lands on ``self.config``. *Injected collaborators*
    are objects the orchestrating parent supplies — a shared knowledge
    base threaded into several strategies, a bot's main LLM passed as a
    memory fallback, a pre-built store for a test. They are NOT config
    data and never enter ``self.config``; they travel through the
    keyword ``components``/``collaborators`` channel and land on
    ``self.components``, with async consumers receiving them as ``_ainit``
    keyword arguments. The collaborator-free call sites
    (``from_config(config)``, ``cls(config)``) are unchanged.

    Subclass requirements:

    - Set ``CONFIG_CLS: ClassVar[type[ConfigT]]`` to the concrete
      ``StructuredConfig`` subclass.
    - Implement ``_setup()`` (sync) and/or ``_ainit()`` (async) for
      derived-attribute computation, connection placeholders, async
      collaborator construction, etc. (Both default to no-ops.) Hook
      delivery is signature-aware: an ``_ainit`` (or
      ``_adopt_components``) override receives only the collaborators it
      declares (or all, if it declares ``**kwargs``), so a no-arg or
      narrowly-typed override is never crashed by an undeclared injected
      collaborator. A consumed collaborator should be declared
      keyword-only with a default
      (``async def _ainit(self, *, knowledge_base=None)``) so the
      zero-injection path is safe.
    - When mixing in alongside other bases, list
      ``StructuredConfigConsumer`` **first** so its ``__init__`` is the
      construction entry point; the remaining bases must accept
      ``__init__()`` with no required args (expose ``_setup``, not a
      competing config ctor).
    - Do NOT override ``__init__`` to re-implement dispatch — that
      duplication is exactly what this mixin eliminates. The one
      legitimate exception is preserving a back-compat positional
      shortcut (see ``PostgresEventBus`` for an example).

    Async-canonical construction:

    ``from_config_async`` is the canonical entry point for an object
    whose initialization is asynchronous; it is the only path that runs
    :meth:`_ainit`. ``from_config`` stays synchronous on this base and
    never runs ``_ainit`` — synchronous construction is opt-in to the
    sync lifecycle only. An object whose canonical construction is async
    AND that must keep a public ``await X.from_config(...)`` API may
    override ``from_config`` to a one-line async delegator — the blessed
    counterpart to the ``__init__`` back-compat exception above::

        @classmethod
        async def from_config(cls, config, **components) -> Self:
            return await cls.from_config_async(config, **components)

    This is not a divergence: it routes straight through
    ``from_config_async``, so ``_coerce_config``, ``__init__(config)``,
    ``_setup``, and ``_ainit`` all run. A consumer with explicit
    injection kwargs keeps them and forwards them as components (e.g. a
    bot threading its main LLM and middleware through). Note the
    footgun: an async ``from_config`` override *removes* the synchronous
    half-built path for that class — ``X.from_config(...)`` now returns a
    coroutine — which is exactly what an async-built object wants. A
    caller that forgets the ``await`` is not left without a signal: a
    static type checker flags any use of the result as an instance (its
    inferred type is ``Coroutine[..., Self]``, not ``Self``), and the
    interpreter emits ``RuntimeWarning: coroutine '...from_config' was
    never awaited`` when the orphaned coroutine is garbage-collected. The
    parity guard
    (:func:`dataknobs_common.testing.assert_structured_config_consumer`)
    requires an async ``from_config`` override to delegate to
    ``from_config_async`` and a sync override to route through
    ``_coerce_config``.

    See :func:`dataknobs_common.testing.assert_structured_config_consumer`
    for the parity guard that pins these contracts at test time.
    """

    CONFIG_CLS: ClassVar[type[StructuredConfig]]

    def __init__(
        self,
        config: ConfigT | Mapping[str, Any] | None = None,
        *,
        _components: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, self.CONFIG_CLS):
            if kwargs:
                raise TypeError(
                    f"{type(self).__name__}: cannot mix typed "
                    f"`config={type(config).__name__}` with loose "
                    f"kwargs ({sorted(kwargs)})."
                )
            self._config: ConfigT = config
        elif config is not None and not isinstance(config, Mapping):
            raise TypeError(
                f"{type(self).__name__}: `config` must be "
                f"{self.CONFIG_CLS.__name__} or Mapping, got "
                f"{type(config).__name__}."
            )
        else:
            merged: dict[str, Any] = dict(config or {})
            merged.update(kwargs)
            self._config = cast("ConfigT", self.CONFIG_CLS.from_dict(merged))
        # Continue the cooperative multiple-inheritance chain before
        # derived setup. For a single-base consumer (e.g. an event-bus
        # backend) ``super()`` is ``object`` and this is a no-op —
        # behaviour is identical to a plain consumer. For a
        # multiple-inheritance consumer
        # (data backends, vector stores) this runs the remaining
        # non-config bases' ``__init__``; under the unified construction
        # model those bases take no construction args (they expose
        # ``_setup``, not a competing config ctor). The mixin must be
        # listed FIRST among the bases so its ``__init__`` is the entry
        # point — :func:`assert_structured_config_consumer` pins this.
        super().__init__()
        # Injected collaborators (objects supplied by the orchestrating
        # parent, NOT config data). Empty for config-only construction —
        # every existing event-bus / data-backend consumer sees ``{}``.
        # ``_components`` is a keyword-only, underscore-prefixed internal
        # channel: it cannot collide with a config field and direct/loose
        # construction never sets it. See :meth:`from_config` /
        # :meth:`from_components` for the public entry points.
        self._components: dict[str, Any] = dict(_components or {})
        # True only when assembled from pre-built collaborators via
        # :meth:`from_components`; lets an async ``_ainit`` short-circuit
        # the config-driven collaborator build.
        self._prebuilt: bool = False
        self._setup()

    @property
    def config(self) -> ConfigT:
        """Typed read-only view of the construction parameters."""
        return self._config

    @property
    def components(self) -> Mapping[str, Any]:
        """Read-only view of injected collaborators.

        Empty (``{}``) for config-only construction. Populated when
        collaborators are passed through :meth:`from_config`,
        :meth:`from_config_async`, or :meth:`from_components` — objects
        supplied by the orchestrating parent (a shared knowledge base, a
        bot's main LLM, a prompt resolver, pre-built test doubles) that
        are *not* part of this object's own config. The sync
        :meth:`_setup` hook may read this; the async :meth:`_ainit` hook
        also receives the collaborators it declares as keyword arguments
        (signature-aware delivery — see :meth:`_ainit`).
        """
        return types.MappingProxyType(self._components)

    #: Names of injected collaborators THIS class consumes itself.
    #: Excluded from :meth:`forwardable_components` so children built
    #: by composing strategies do not inherit their parent's own
    #: collaborators (FSM handles, internal caches, etc.). Defaults to
    #: empty — only composing strategies need override.
    INTERNAL_COMPONENTS: ClassVar[frozenset[str]] = frozenset()

    def forwardable_components(self) -> dict[str, Any]:
        """Return collaborators safe to forward to child consumers.

        Returns ``self.components`` minus the names declared in
        :attr:`INTERNAL_COMPONENTS`. The fresh dict is suitable to
        spread into a child's construction call::

            class MyComposingStrategy(
                StructuredConfigConsumer[MyConfig], ReasoningStrategy,
            ):
                INTERNAL_COMPONENTS = frozenset(
                    {"my_internal_collaborator"}
                )

                def _build_child(self, child_config):
                    return get_registry().create(
                        config=child_config,
                        **self.forwardable_components(),
                    )

        The wizard reference adopter declares
        ``INTERNAL_COMPONENTS = frozenset({"wizard_fsm"})`` so per-stage
        sub-strategies do not receive the outer wizard's FSM handle;
        every other collaborator threaded through
        ``WizardReasoning.from_config(config, **components)`` flows
        through opaquely.
        """
        if not self.INTERNAL_COMPONENTS:
            return dict(self._components)
        return {
            k: v
            for k, v in self._components.items()
            if k not in self.INTERNAL_COMPONENTS
        }

    def set_component(
        self, name: str, value: Any, *, allow_overwrite: bool = True
    ) -> None:
        """Inject or replace an injected collaborator after construction.

        Construction-time injection (:meth:`from_config`,
        :meth:`from_components`) covers collaborators that exist when the
        consumer is built. Some do not: a collaborator that itself depends
        on the fully-built consumer (a circular dependency), or a resource
        built later at application-lifespan / setup time. ``set_component``
        is the supported write path for those — it writes the private
        backing store behind the read-only :attr:`components` view.

        The value is immediately visible via :attr:`components` and included
        in :meth:`forwardable_components` (unless ``name`` is in
        :attr:`INTERNAL_COMPONENTS`, which is never forwarded to children —
        setting such a name affects only this consumer).

        **Read-once boundary.** Whether this write *reaches* a consuming
        subsystem depends on when that subsystem reads its collaborators. A
        consumer that re-reads :attr:`components` / :meth:`forwardable_components`
        afresh per operation (e.g. a per-turn rebuild) observes the update on
        its next read. A consumer that reads a collaborator **once** in
        :meth:`_setup` / :meth:`_ainit` / :meth:`_adopt_components` folds it
        into derived state at construction and does **not** re-consume it — for
        those, ``set_component`` must be called before that first read. Note
        this is impossible for a genuine circular dependency (the collaborator
        does not exist when ``_ainit`` runs); such a consumer must consume the
        collaborator lazily per-operation instead.

        **Threading.** The write is *not* synchronized. It mutates the plain
        ``dict`` that :attr:`components` / :meth:`forwardable_components` read,
        so it must occur on the same thread that reads those — inject before the
        consumer begins serving, or from the same event-loop thread. Calling
        ``set_component`` from a background thread while another thread iterates
        :meth:`forwardable_components` can raise ``RuntimeError: dictionary
        changed size during iteration``. Cross-thread injection is not supported.

        Args:
            name: Collaborator key (as it would appear in :attr:`components`).
            value: The collaborator object.
            allow_overwrite: When ``False``, raise ``ValueError`` if ``name``
                is already present (inject-only). Defaults to ``True``
                (inject-or-replace).

        Raises:
            ValueError: ``allow_overwrite=False`` and ``name`` already present.
        """
        if not allow_overwrite and name in self._components:
            raise ValueError(
                f"Component {name!r} already present on {type(self).__name__}; "
                "pass allow_overwrite=True to replace it."
            )
        self._components[name] = value

    def set_components(
        self, values: Mapping[str, Any], *, allow_overwrite: bool = True
    ) -> None:
        """Inject or replace several injected collaborators at once.

        Bulk form of :meth:`set_component` — the shape a caller previously
        expressed as ``self._components.update(values)`` (reaching past the
        read-only view). The read-once boundary documented on
        :meth:`set_component` applies identically.

        With ``allow_overwrite=False`` the write is **all-or-nothing**: if
        *any* key in ``values`` is already present, a ``ValueError`` is raised
        and **no** entries are applied — a partial bulk write never leaves the
        consumer half-wired.

        Args:
            values: Collaborator key → object mapping.
            allow_overwrite: When ``False``, raise ``ValueError`` if any key in
                ``values`` is already present. Defaults to ``True``.

        Raises:
            ValueError: ``allow_overwrite=False`` and any key already present.
        """
        if not allow_overwrite:
            clashes = [name for name in values if name in self._components]
            if clashes:
                raise ValueError(
                    f"Component(s) {clashes!r} already present on "
                    f"{type(self).__name__}; pass allow_overwrite=True to "
                    "replace them."
                )
        self._components.update(values)

    @classmethod
    def _coerce_config(
        cls, config: Mapping[str, Any] | StructuredConfig
    ) -> ConfigT:
        """Coerce a dict-or-typed ``config`` argument to a typed ``ConfigT``.

        Shared by :meth:`from_config` and any subclass that overrides
        ``from_config`` to deliver the typed config through a non-default
        ctor slot (e.g. :class:`PostgresEventBus`, whose first positional
        is the legacy ``connection_string``). Centralizing the guard here
        means every ``from_config`` path rejects a config of the wrong
        ``StructuredConfig`` subclass with the same clear ``TypeError``
        rather than the opaque ``dict()``-on-dataclass crash that
        ``from_dict`` would otherwise raise.
        """
        if isinstance(config, cls.CONFIG_CLS):
            return cast("ConfigT", config)
        if not isinstance(config, Mapping):
            raise TypeError(
                f"{cls.__name__}.from_config: `config` must be "
                f"{cls.CONFIG_CLS.__name__} or Mapping, got "
                f"{type(config).__name__}."
            )
        return cast("ConfigT", cls.CONFIG_CLS.from_dict(config))

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | StructuredConfig,
        **components: Any,
    ) -> Self:
        """Build an instance from a config dict or typed config.

        The recommended programmatic-construction entry point. Registry
        factories collapse to one-line wrappers over this. A typed
        ``config`` of the wrong ``StructuredConfig`` subclass (not
        ``CONFIG_CLS``) raises ``TypeError``.

        Injected collaborators (objects supplied by the orchestrating
        parent — a shared knowledge base, a prompt resolver, a bot's main
        LLM) are passed as keyword ``components``. They are stored on
        ``self._components`` (see :attr:`components`) and made available
        to :meth:`_setup`; they are *not* folded into ``self.config``.
        The collaborator-free call ``from_config(config)`` is unchanged.
        """
        return cls(cls._coerce_config(config), _components=components or None)

    @classmethod
    async def from_config_async(
        cls,
        config: Mapping[str, Any] | StructuredConfig,
        **components: Any,
    ) -> Self:
        """Async construction: sync assemble, then ``await _ainit()``.

        The canonical way to build a consumer whose initialization is
        asynchronous — databases/vector stores that connect eagerly,
        LLM-backed bots, knowledge-base warmup. Synchronous consumers
        ignore this and use :meth:`from_config`. Accepts dict or typed
        config through the same :meth:`_coerce_config` guard
        ``from_config`` uses, so a wrong-subclass typed config raises the
        same clear ``TypeError``.

        Injected collaborators are passed as keyword ``components`` and
        delivered to :meth:`_ainit` as keyword arguments (in addition to
        being stored on ``self._components``). The collaborator-free call
        ``from_config_async(config)`` delivers nothing to ``_ainit`` and
        is unchanged.

        A subclass whose async factory consumes the typed config through
        a non-default ctor slot may override this; the override must
        still route through :meth:`_coerce_config` (pinned by
        :func:`dataknobs_common.testing.assert_structured_config_consumer`).
        """
        obj = cls(cls._coerce_config(config), _components=components or None)
        await obj._ainit(**_components_for_hook(cls._ainit, obj._components))
        return obj

    @classmethod
    def from_components(
        cls,
        config: ConfigT | Mapping[str, Any] | None = None,
        **collaborators: Any,
    ) -> Self:
        """Assemble from pre-built collaborators, skipping config-driven build.

        The companion to :meth:`from_config` for the dual-input shape
        where a parent already holds fully-built collaborators (a
        pre-built vector store, an LLM provider, child sub-objects) and
        wants to inject them directly rather than have the consumer build
        them from config. Recurs across the object graph: an orchestrator
        builds children, then hands them to the parent.

        ``config`` is an optional snapshot of the scalar knobs; when
        omitted it defaults to ``CONFIG_CLS()`` (valid only when every
        config field has a default — pass a ``config`` for configs with
        required fields). The collaborators are stored on
        ``self._components``, ``self._prebuilt`` is set ``True``, and
        :meth:`_adopt_components` is called so the subclass binds its
        collaborator attributes. A consumer's :meth:`_ainit` should
        short-circuit when ``self._prebuilt`` is ``True`` so a later
        :meth:`from_config_async` does not rebuild what is already wired.
        """
        if config is not None:
            cfg: ConfigT = cls._coerce_config(config)
        else:
            try:
                cfg = cast("ConfigT", cls.CONFIG_CLS())
            except TypeError as exc:
                raise ValueError(
                    f"{cls.__name__}.from_components: CONFIG_CLS "
                    f"{cls.CONFIG_CLS.__name__} has required fields and "
                    "cannot be default-constructed. Pass a `config=` "
                    "snapshot covering the required fields."
                ) from exc
        obj = cls(cfg, _components=collaborators or None)
        obj._prebuilt = True
        obj._adopt_components(
            **_components_for_hook(cls._adopt_components, obj._components)
        )
        return obj

    def _setup(self) -> None:
        """Subclass hook: derived-attribute computation after ``self._config``.

        Default no-op. Override to initialize derived attributes
        computed from ``self.config.*`` (connection-pool placeholders,
        lock/handle initialization, etc.). Field normalization belongs
        in the config dataclass (``_normalize_dict`` / ``__post_init__``),
        not here — ``_setup`` runs after ``self._config`` is frozen.
        Called once during ``__init__`` (both sync and async paths).

        Injected collaborators (if any) are already on ``self.components``
        when ``_setup`` runs, so a sync consumer can bind them here. Note
        that ``self._prebuilt`` is always ``False`` in ``_setup`` — it is
        set after ``__init__`` returns, so the ``_prebuilt`` short-circuit
        is only observable in :meth:`_ainit` / :meth:`_adopt_components`,
        not here.
        """

    async def _ainit(self, **components: Any) -> None:
        """Subclass hook: async initialization after sync construction.

        Default no-op. Override for connection establishment, provider
        warmup, KB ingest, async collaborator construction, or any
        awaitable setup that cannot run in the synchronous
        :meth:`_setup`. Runs exactly once, after ``_setup``, when the
        object is built via :meth:`from_config_async`. Objects built via
        the synchronous path (``__init__`` / :meth:`from_config`) do NOT
        run ``_ainit`` automatically — async setup is opt-in via the
        async entry point.

        Injected collaborators passed to :meth:`from_config_async` are
        delivered here as keyword arguments (the same objects are always
        on ``self._components`` regardless). Delivery is signature-aware:
        a hook declaring ``**kwargs`` receives every collaborator; one
        declaring named params receives only the matching subset; a no-arg
        override receives none (and reads ``self.components`` instead). An
        override that consumes a collaborator should declare it
        keyword-only with a default — e.g.
        ``async def _ainit(self, *, knowledge_base=None)`` — so the
        zero-injection call is safe; an undeclared collaborator is dropped
        from delivery rather than crashing the hook. A collaborator param
        *without* a default (or a required positional) still breaks the
        zero-injection path and is rejected by
        :func:`dataknobs_common.testing.assert_structured_config_consumer`.
        A consumer that also supports :meth:`from_components` should
        ``return`` early when ``self._prebuilt`` is ``True``.
        """

    def _adopt_components(self, **collaborators: Any) -> None:
        """Subclass hook: bind pre-built collaborators from :meth:`from_components`.

        Default no-op. Override to assign the collaborator attributes a
        config-driven :meth:`_ainit` would otherwise build (e.g.
        ``self._vector_store = vector_store``). Called once, synchronously,
        immediately after construction when the object is assembled via
        :meth:`from_components`. Delivery is signature-aware exactly as for
        :meth:`_ainit`: ``**kwargs`` receives every collaborator, named
        params receive the matching subset, a no-arg override receives
        none (collaborators stay on ``self.components``). Consumed
        parameters should be keyword-only with defaults so the
        zero-injection path is safe; a param without a default is rejected
        by :func:`dataknobs_common.testing.assert_structured_config_consumer`.
        """


__all__ = [
    "SKIP_VALIDATION",
    "ConfigClassResolution",
    "ConfigClassResolver",
    "ConfigT",
    "StructuredConfig",
    "StructuredConfigConsumer",
    "config_registries",
    "register_sensitive_interior_key",
]
