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
import types
from collections.abc import Mapping
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

if TYPE_CHECKING:
    # ``Self`` is referenced only in (lazy, ``from __future__``-stringized)
    # annotations, so it is never evaluated at runtime. Guarding the import
    # keeps this zero-dependency package from importing ``typing_extensions``
    # at module load. ``typing_extensions`` (not ``typing``) is the source
    # because mypy's ``python_version = "3.10"`` target predates
    # ``typing.Self``; mypy resolves the guarded import regardless.
    from typing_extensions import Self

# ``X | None`` (PEP 604) yields ``types.UnionType``; ``Optional[X]`` /
# ``Union[...]`` yield ``typing.Union``. Both must be recognised as unions
# when decomposing field annotations.
_UNION_ORIGINS = (Union, types.UnionType)
# Container origins whose element type(s) may themselves be (or contain) a
# ``StructuredConfig`` subclass and therefore need element-wise coercion.
_SEQUENCE_ORIGINS = (list, tuple, set, frozenset)


@functools.cache
def _resolved_hints(cls: type) -> Mapping[str, Any]:
    """Cache of ``typing.get_type_hints(cls)``, degrading to ``{}``.

    Because the package uses ``from __future__ import annotations``,
    ``dataclasses.fields(cls)[i].type`` is a *string*; type-aware field
    coercion needs the resolved runtime types. ``get_type_hints`` resolves
    them against the class's module globals. If resolution fails (an
    annotation referencing a name unavailable at runtime — e.g. a
    ``TYPE_CHECKING``-only import), we degrade to ``{}`` so ``from_dict``
    falls back to the pre-146b pass-through behaviour rather than crashing.
    """
    try:
        return get_type_hints(cls)
    except Exception:
        return {}


def _type_contains_structured_config(declared: Any) -> bool:
    """True if ``declared`` is (or its type graph contains) a config subclass.

    Used as a fast gate in ``from_dict``: a field whose declared type does
    not reference any ``StructuredConfig`` subclass is assigned verbatim,
    preserving the exact Item-146 projection (no new container objects,
    identical identity for pass-through values). Only fields that actually
    nest configs incur the recursive coercion path.
    """
    if isinstance(declared, type):
        return issubclass(declared, StructuredConfig)
    origin = get_origin(declared)
    if origin is None:
        return False
    return any(
        _type_contains_structured_config(arg)
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

    origin = get_origin(declared)
    if origin is None:
        return value

    args = get_args(declared)

    # ``Optional[SubCfg]`` / ``SubCfg | None`` (and wider unions).
    if origin in _UNION_ORIGINS:
        if value is None:
            return value
        for arg in args:
            if arg is type(None):
                continue
            coerced = _coerce_field(arg, value)
            if coerced is not value:
                return coerced
        return value

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
    ``SubCfg``, ``SubCfg | None``, ``list[SubCfg]``, ``dict[K, SubCfg]``,
    and ``dict[K, list[SubCfg]]`` are all rebuilt from their raw dict
    shape. Recursion is bounded by the static field-type graph (not by
    runtime data), so a config whose fields are all scalars terminates
    immediately. A field already holding a typed instance passes through
    unchanged. Polymorphic/discriminated selection (a section whose
    concrete type is chosen by a discriminator key) is deliberately NOT
    handled here — that stays in the subsystem registry / object-graph
    layer; type a polymorphic field as its abstract sub-config and
    dispatch in the consumer's factory.
    """

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
        dict shape (see the class docstring's nested-composition note);
        all other fields are assigned verbatim.

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
            if declared is not None and _type_contains_structured_config(
                declared
            ):
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
        """
        return dataclasses.asdict(self)


ConfigT = TypeVar("ConfigT", bound=StructuredConfig)


class StructuredConfigConsumer(Generic[ConfigT]):
    """Mixin for classes constructed from a ``StructuredConfig`` subclass.

    Provides:

    - ``__init__(config: ConfigT | Mapping | None, **kwargs)`` with
      built-in typed/loose/None dispatch. Mixing typed ``config=``
      with loose ``**kwargs`` raises ``TypeError``. Calls
      ``super().__init__()`` so the mixin composes into a cooperative
      multiple-inheritance hierarchy (data backends, vector stores).
    - ``cls.from_config(config) -> Self`` classmethod that runs the
      input through ``CONFIG_CLS.from_dict`` (when given a Mapping)
      then ``cls``.
    - ``cls.from_config_async(config) -> Self`` classmethod: the same
      sync assembly followed by ``await obj._ainit()`` — the canonical
      entry point for consumers whose initialization is asynchronous
      (DBs that connect eagerly, LLM-backed bots, KB warmup).
    - ``self.config: ConfigT`` typed read-only property — the one
      ``config`` notion across every adopter.
    - ``_setup()`` sync hook called after ``self._config`` is
      established; ``_ainit()`` async hook called by
      ``from_config_async`` after ``_setup``.

    Subclass requirements:

    - Set ``CONFIG_CLS: ClassVar[type[ConfigT]]`` to the concrete
      ``StructuredConfig`` subclass.
    - Implement ``_setup()`` (sync) and/or ``_ainit()`` (async) for
      derived-attribute computation, connection placeholders, async
      collaborator construction, etc. (Both default to no-ops.)
    - When mixing in alongside other bases, list
      ``StructuredConfigConsumer`` **first** so its ``__init__`` is the
      construction entry point; the remaining bases must accept
      ``__init__()`` with no required args (expose ``_setup``, not a
      competing config ctor).
    - Do NOT override ``__init__`` to re-implement dispatch — that
      duplication is exactly what this mixin eliminates. The one
      legitimate exception is preserving a back-compat positional
      shortcut (see ``PostgresEventBus`` for an example).

    See :func:`dataknobs_common.testing.assert_structured_config_consumer`
    for the parity guard that pins these contracts at test time.
    """

    CONFIG_CLS: ClassVar[type[StructuredConfig]]

    def __init__(
        self,
        config: ConfigT | Mapping[str, Any] | None = None,
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
        self._setup()

    @property
    def config(self) -> ConfigT:
        """Typed read-only view of the construction parameters."""
        return self._config

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
        cls, config: Mapping[str, Any] | StructuredConfig
    ) -> Self:
        """Build an instance from a config dict or typed config.

        The recommended programmatic-construction entry point. Registry
        factories collapse to one-line wrappers over this. A typed
        ``config`` of the wrong ``StructuredConfig`` subclass (not
        ``CONFIG_CLS``) raises ``TypeError``.
        """
        return cls(cls._coerce_config(config))

    @classmethod
    async def from_config_async(
        cls, config: Mapping[str, Any] | StructuredConfig
    ) -> Self:
        """Async construction: sync assemble, then ``await _ainit()``.

        The canonical way to build a consumer whose initialization is
        asynchronous — databases/vector stores that connect eagerly,
        LLM-backed bots, knowledge-base warmup. Synchronous consumers
        ignore this and use :meth:`from_config`. Accepts dict or typed
        config through the same :meth:`_coerce_config` guard
        ``from_config`` uses, so a wrong-subclass typed config raises the
        same clear ``TypeError``.

        A subclass whose async factory consumes the typed config through
        a non-default ctor slot may override this; the override must
        still route through :meth:`_coerce_config` (pinned by
        :func:`dataknobs_common.testing.assert_structured_config_consumer`).
        """
        obj = cls(cls._coerce_config(config))
        await obj._ainit()
        return obj

    def _setup(self) -> None:
        """Subclass hook: derived-attribute computation after ``self._config``.

        Default no-op. Override to initialize derived attributes
        computed from ``self.config.*`` (connection-pool placeholders,
        lock/handle initialization, etc.). Field normalization belongs
        in the config dataclass (``_normalize_dict`` / ``__post_init__``),
        not here — ``_setup`` runs after ``self._config`` is frozen.
        Called once during ``__init__`` (both sync and async paths).
        """

    async def _ainit(self) -> None:
        """Subclass hook: async initialization after sync construction.

        Default no-op. Override for connection establishment, provider
        warmup, KB ingest, async collaborator construction, or any
        awaitable setup that cannot run in the synchronous
        :meth:`_setup`. Runs exactly once, after ``_setup``, when the
        object is built via :meth:`from_config_async`. Objects built via
        the synchronous path (``__init__`` / :meth:`from_config`) do NOT
        run ``_ainit`` automatically — async setup is opt-in via the
        async entry point.
        """


__all__ = [
    "ConfigT",
    "StructuredConfig",
    "StructuredConfigConsumer",
]
