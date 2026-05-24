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
import types
from collections.abc import Callable, Mapping
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

logger = logging.getLogger(__name__)

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


def _type_contains_structured_config(declared: Any) -> bool:
    """True if ``declared`` is (or its type graph contains) a config subclass.

    Used as a fast gate in ``from_dict``: a field whose declared type does
    not reference any ``StructuredConfig`` subclass is assigned verbatim,
    preserving the exact flat-projection behaviour (no new container
    objects, identical identity for pass-through values). Only fields that
    actually nest configs incur the recursive coercion path.
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

    # ``Optional[SubCfg]`` / ``SubCfg | None``. Only auto-coerce when
    # exactly one non-``None`` arm is (or contains) a ``StructuredConfig``.
    # A union with several config arms is a discriminated/polymorphic
    # shape — selecting among them by inspecting the data is deliberately
    # out of scope here (it belongs in the subsystem registry /
    # object-graph layer), so the value passes through untouched rather
    # than silently binding to whichever arm happens to be declared first.
    if origin in _UNION_ORIGINS:
        if value is None:
            return value
        config_arms = [
            arg
            for arg in args
            if arg is not type(None)
            and _type_contains_structured_config(arg)
        ]
        if len(config_arms) != 1:
            return value
        return _coerce_field(config_arms[0], value)

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
    instance passes through unchanged. Polymorphic/discriminated
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

    A subclass that needs a genuinely custom ``__repr__`` may still
    define one in its body; :meth:`__init_subclass__` leaves an
    explicitly-defined repr untouched.
    """

    #: Field names masked by :meth:`_redacted_repr`. Empty by default,
    #: so a config with no secrets behaves exactly like a plain
    #: dataclass.
    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Install the redacting ``__repr__`` on every subclass.

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
        """
        super().__init_subclass__(**kwargs)
        if "__repr__" not in cls.__dict__:
            # ``setattr`` (not ``cls.__repr__ =``) keeps mypy from flagging
            # a method reassignment; the effect — an entry in
            # ``cls.__dict__`` — is what suppresses dataclass generation.
            cls.__repr__ = StructuredConfig._redacted_repr

    def _redacted_repr(self) -> str:
        """Dataclass-style repr that masks ``_SENSITIVE_FIELDS`` values.

        Mirrors the auto-generated dataclass repr for every ``repr=True``
        field, substituting ``'***'`` for a non-empty sensitive value so
        credentials never reach logs through ``repr(config)`` or an
        f-string. A falsy value (``None`` / ``""``) is shown verbatim — an
        unset credential is not a secret, and masking ``""`` would falsely
        imply one is configured. ``type(self).__qualname__`` (matching the
        dataclass-generated repr) makes this a byte-for-byte drop-in for
        configs with no sensitive fields. Installed as the ``__repr__`` of
        every subclass by :meth:`__init_subclass__`.
        """
        parts: list[str] = []
        for f in dataclasses.fields(self):
            if not f.repr:
                continue
            value = getattr(self, f.name)
            if f.name in self._SENSITIVE_FIELDS and value:
                value = "***"
            parts.append(f"{f.name}={value!r}")
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
    "ConfigT",
    "StructuredConfig",
    "StructuredConfigConsumer",
]
