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
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

if TYPE_CHECKING:
    # ``Self`` is referenced only in (lazy, ``from __future__``-stringized)
    # annotations, so it is never evaluated at runtime. Guarding the import
    # keeps this zero-dependency package from importing ``typing_extensions``
    # at module load. ``typing_extensions`` (not ``typing``) is the source
    # because mypy's ``python_version = "3.10"`` target predates
    # ``typing.Self``; mypy resolves the guarded import regardless.
    from typing_extensions import Self


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

    Nested-config composition: if a field's declared type is itself a
    ``StructuredConfig`` subclass, ``from_dict`` does NOT automatically
    recurse — assign the typed value directly, or normalize the nested
    dict in ``_normalize_dict``. Auto-projection of nested fields was
    intentionally scoped out to keep type introspection bounded.
    """

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> Self:
        """Build an instance from a config dict via field introspection.

        Each ``@dataclass`` field on ``cls`` is read from ``config`` by
        name. Fields absent from ``config`` use their declared default
        (or ``default_factory``). Keys in ``config`` that don't match
        any field are ignored — registry-routing keys like
        ``"backend"`` pass through cleanly.

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
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name in normalized:
                kwargs[f.name] = normalized[f.name]
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

        Round-trip property: for any ``cfg: StructuredConfig``,
        ``type(cfg).from_dict(cfg.to_dict()) == cfg`` holds. Verified
        by
        :func:`dataknobs_common.testing.assert_structured_config_roundtrip`.
        """
        return dataclasses.asdict(self)


ConfigT = TypeVar("ConfigT", bound=StructuredConfig)


class StructuredConfigConsumer(Generic[ConfigT]):
    """Mixin for classes constructed from a ``StructuredConfig`` subclass.

    Provides:

    - ``__init__(config: ConfigT | Mapping | None, **kwargs)`` with
      built-in typed/loose/None dispatch. Mixing typed ``config=``
      with loose ``**kwargs`` raises ``TypeError``.
    - ``cls.from_config(config) -> Self`` classmethod that runs the
      input through ``CONFIG_CLS.from_dict`` (when given a Mapping)
      then ``cls``.
    - ``self.config: ConfigT`` typed read-only property.
    - ``_setup()`` hook called after ``self._config`` is established.

    Subclass requirements:

    - Set ``CONFIG_CLS: ClassVar[type[ConfigT]]`` to the concrete
      ``StructuredConfig`` subclass.
    - Implement ``_setup()`` for derived-attribute computation,
      connection placeholders, etc. (Default ``_setup`` is a no-op.)
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
        self._setup()

    @property
    def config(self) -> ConfigT:
        """Typed read-only view of the construction parameters."""
        return self._config

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | StructuredConfig
    ) -> Self:
        """Build an instance from a config dict or typed config.

        The recommended programmatic-construction entry point. Registry
        factories collapse to one-line wrappers over this.
        """
        if isinstance(config, cls.CONFIG_CLS):
            return cls(config)
        # ``config`` is a Mapping here (the typed branch returned). mypy
        # cannot narrow against the ``type[StructuredConfig]`` ClassVar,
        # so the residual union is asserted away explicitly.
        return cls(
            cls.CONFIG_CLS.from_dict(cast("Mapping[str, Any]", config))
        )

    def _setup(self) -> None:
        """Subclass hook: derived-attribute computation after ``self._config``.

        Default no-op. Override to initialize derived attributes
        computed from ``self.config.*`` (connection-pool placeholders,
        post-validation re-sanitization, etc.). Called once during
        ``__init__``.
        """


__all__ = [
    "ConfigT",
    "StructuredConfig",
    "StructuredConfigConsumer",
]
