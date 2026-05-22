"""Drift-guard helpers for factory ↔ ctor parity in dataknobs registries.

These helpers exist because the SQS event bus shipped with a ctor knob
(``require_topic_attribute``) that its registry factory didn't forward.
The specific gap was closed; these helpers prevent the *class* of bug
from recurring.

Four structural patterns exist across the dataknobs registries:

1. **Typed dataclass + ctor.** The bus (or provider) ctor consumes a
   frozen dataclass. Drift = the dataclass has a field the ctor doesn't
   accept, or vice versa. Guard:
   :func:`assert_dataclass_config_matches_ctor`.

2. **Free-function factory that names its kwargs.** A registry entry
   is a callable that does ``return cls(kwarg=config.get("k"), ...)``.
   Drift = a kwarg name in the factory body doesn't exist on the
   target ctor (or vice versa, when an allowlist silently drops a
   knob). Guard: :func:`assert_factory_kwargs_match_ctor`.

3. **Whole-dict ctor reading documented keys.** The ctor takes a
   dict and reads ``config.get("X")`` / ``config["X"]`` internally.
   Drift = a documented key is no longer read. Guard:
   :func:`assert_ctor_reads_documented_keys`.

4. **``StructuredConfigConsumer[ConfigT]`` adopter.** The class mixes
   in
   :class:`~dataknobs_common.structured_config.StructuredConfigConsumer`
   and declares ``CONFIG_CLS``. Drift = the declaration is missing,
   ``CONFIG_CLS`` is not a ``StructuredConfig`` subclass, or the
   dataclass field set drifts from the consumer ctor surface. Guard:
   :func:`assert_structured_config_consumer`. Bundles patterns 1 and
   2 for adopters of the structured-config primitives.

The AST-walking helpers (1, 2, 3) do not instantiate the target class,
so backends with optional dependencies (asyncpg, aioboto3, ...) can be
parity-tested without those dependencies installed.
:func:`assert_structured_config_roundtrip` exercises a config instance
directly — it asserts a serialization property, not a structural
contract.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
from collections.abc import Callable
from typing import Any


def assert_dataclass_config_matches_ctor(
    config_cls: type,
    target_cls: type,
    *,
    ignore_params: set[str] | None = None,
) -> None:
    """Assert ``config_cls`` fields match ``target_cls.__init__`` params.

    Use for registries where the ctor consumes a typed dataclass
    (event buses with structured configs, LLM providers, future
    structured-config consumers).

    Args:
        config_cls: A frozen ``@dataclass`` config class. Its
            ``dataclasses.fields()`` are the expected ctor surface.
        target_cls: The class whose ``__init__`` should mirror the
            dataclass. The implicit ``self`` and the ``config`` kwarg
            (used to receive the dataclass itself) are ignored.
        ignore_params: Additional ctor param names to ignore (for
            internal-only kwargs that aren't intended to be config
            fields — pass them explicitly so the omission is
            documented).

    Raises:
        AssertionError: If the dataclass has a field the ctor lacks, or
            the ctor has a non-ignored param the dataclass lacks.
    """
    ignore_params = ignore_params or set()
    config_fields = {f.name for f in dataclasses.fields(config_cls)}
    sig = inspect.signature(target_cls.__init__)
    accepts_var_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    ctor_params = {
        name
        for name, p in sig.parameters.items()
        if p.kind is not inspect.Parameter.VAR_KEYWORD
        and p.kind is not inspect.Parameter.VAR_POSITIONAL
        and name not in {"self", "config"} | ignore_params
    }
    missing_in_config = ctor_params - config_fields
    # When the ctor accepts ``**kwargs`` (the
    # ``StructuredConfigConsumer`` pattern), every dataclass field is
    # accepted by construction — the variadic forwards into
    # ``from_dict`` for field projection. Drift in the
    # ctor-missing-field direction is structurally impossible.
    missing_in_ctor: set[str] = (
        set() if accepts_var_kwargs else config_fields - ctor_params
    )
    failures: list[str] = []
    if missing_in_config:
        failures.append(
            f"{config_cls.__name__} is missing fields for "
            f"{target_cls.__name__}.__init__ params: "
            f"{sorted(missing_in_config)}"
        )
    if missing_in_ctor:
        failures.append(
            f"{target_cls.__name__}.__init__ is missing params for "
            f"{config_cls.__name__} fields: {sorted(missing_in_ctor)}"
        )
    if failures:
        raise AssertionError(" | ".join(failures))


def assert_factory_kwargs_match_ctor(
    factory: Callable[..., Any],
    target_cls: type,
    *,
    ignore_kwargs: set[str] | None = None,
) -> None:
    """Assert every kwarg the factory body passes to ``target_cls`` is valid.

    AST-walks the factory function body for ``target_cls(...)`` or
    ``target_cls.from_config(...)`` call sites and asserts every keyword
    argument is a valid parameter of ``target_cls.__init__``. Catches
    the allowlist-drift failure mode in any registry: a factory adds (or
    drops) a kwarg without a matching ctor change.

    This is the only helper that *catches* an allowlist factory dropping
    a ctor kwarg — by walking the factory body and noting which kwargs
    it provides, it detects the inverse direction (factory missing a
    kwarg that exists on the ctor) via comparison against the ctor's
    parameter list. Symmetric check.

    Args:
        factory: The factory callable. Must be inspectable via
            ``inspect.getsource``.
        target_cls: The class the factory constructs. Its
            ``__init__`` parameter list is the expected kwarg set.
        ignore_kwargs: Ctor param names that the factory is allowed
            to omit (typically required positionals the consumer is
            expected to supply, or knobs without a sensible
            config-dict default).

    Raises:
        AssertionError: If the factory passes a kwarg that's not on
            the ctor, or if the ctor exposes a non-ignored param the
            factory doesn't forward.
    """
    ignore_kwargs = ignore_kwargs or set()
    try:
        src = textwrap.dedent(inspect.getsource(factory))
    except (OSError, TypeError) as e:
        raise AssertionError(
            f"Cannot read source of factory {factory!r}: {e}"
        ) from e
    tree = ast.parse(src)

    factory_kwargs: set[str] = set()
    # ``"cls"`` is the conventional alias for the target class inside a
    # classmethod, so the walker treats ``cls(...)`` / ``cls.from_config(...)``
    # as equivalent to ``Target(...)`` / ``Target.from_config(...)``. This
    # assumes factory functions do not also use a *local* identifier named
    # ``cls`` for an unrelated callable — a convention the registry
    # factories in this codebase follow, but worth documenting because a
    # future factory that violates it would yield false positives.
    target_names = {target_cls.__name__, "cls"}
    delegates_to_from_config = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # ``Target.from_config(...)`` / ``cls.from_config(...)`` is the
        # structured-config delegation path. When the factory uses it,
        # the dataclass IS the kwarg-coverage source of truth, so the
        # "missing kwargs" check is satisfied by the
        # ``assert_dataclass_config_matches_ctor`` companion test.
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in target_names
            and func.attr == "from_config"
        ):
            delegates_to_from_config = True
            continue
        # Direct ``Target(...)`` ctor call — enumerate its kwargs.
        if isinstance(func, ast.Name) and func.id in target_names:
            for kw in node.keywords:
                if kw.arg is None:
                    # **kwargs splat — opaque; skip rather than false-positive.
                    continue
                factory_kwargs.add(kw.arg)

    sig = inspect.signature(target_cls.__init__)
    accepts_var_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    ctor_params = {
        name
        for name, p in sig.parameters.items()
        if p.kind is not inspect.Parameter.VAR_KEYWORD
        and p.kind is not inspect.Parameter.VAR_POSITIONAL
        and name != "self"
    }
    # When the ctor declares ``**kwargs`` (the
    # ``StructuredConfigConsumer`` pattern), every named kwarg the
    # factory passes is structurally accepted — drift in the
    # unknown-to-ctor direction is impossible. The
    # missing-from-factory direction stays meaningful only when the
    # factory hand-rolls calls; ``from_config`` delegation covers
    # that orthogonally.
    unknown_in_factory: set[str] = (
        set() if accepts_var_kwargs else factory_kwargs - ctor_params
    )
    missing_in_factory = (
        ctor_params - factory_kwargs - ignore_kwargs - {"config"}
    )
    failures: list[str] = []
    if unknown_in_factory:
        failures.append(
            f"Factory {factory.__name__} passes kwargs to "
            f"{target_cls.__name__} that are not on its ctor: "
            f"{sorted(unknown_in_factory)}"
        )
    # The "missing kwargs" check only applies when the factory hand-rolls
    # ctor calls. ``from_config`` delegation is a stronger guarantee
    # (every dataclass field is consumed wholesale) and is verified by
    # the companion ``assert_dataclass_config_matches_ctor`` parity test.
    if missing_in_factory and not delegates_to_from_config:
        failures.append(
            f"Factory {factory.__name__} does not forward ctor kwargs "
            f"of {target_cls.__name__}: {sorted(missing_in_factory)}. "
            "Add the missing kwarg or include it in ignore_kwargs."
        )
    if failures:
        raise AssertionError(" | ".join(failures))


def assert_ctor_reads_documented_keys(
    target_cls: type,
    documented_keys: set[str],
    *,
    config_param: str = "config",
) -> None:
    """Assert ``target_cls.__init__`` reads every key in ``documented_keys``.

    Use for registries where the ctor takes a dict and reads keys via
    ``config.get("X")`` or ``config["X"]`` inside its body (vector
    stores, postgres lock, data backends post-merge-into-base-init).

    Args:
        target_cls: The class to check.
        documented_keys: The set of dict keys the docs/registry-row
            promise are accepted by this backend.
        config_param: Name of the ``__init__`` parameter that holds
            the config dict (default ``"config"``).

    Raises:
        AssertionError: If a documented key is not read in the
            ctor body.
    """
    try:
        src = textwrap.dedent(inspect.getsource(target_cls.__init__))
    except (OSError, TypeError) as e:
        raise AssertionError(
            f"Cannot read source of {target_cls.__name__}.__init__: {e}"
        ) from e
    tree = ast.parse(src)

    read_keys: set[str] = set()
    for node in ast.walk(tree):
        # ``config.get("X", ...)`` / ``config.get("X")``
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == config_param
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            read_keys.add(node.args[0].value)
        # ``config["X"]``
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == config_param
        ):
            slice_node = node.slice
            if (
                isinstance(slice_node, ast.Constant)
                and isinstance(slice_node.value, str)
            ):
                read_keys.add(slice_node.value)
        # ``config.pop("X", ...)`` — same effect as get for documentation purposes
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "pop"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == config_param
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            read_keys.add(node.args[0].value)

    missing = documented_keys - read_keys
    if missing:
        raise AssertionError(
            f"{target_cls.__name__}.__init__ does not read these "
            f"documented keys from `{config_param}`: {sorted(missing)}"
        )


def assert_structured_config_consumer(
    consumer_cls: type,
    *,
    expected_factory: Callable[..., Any] | None = None,
    ignore_params: set[str] | None = None,
) -> None:
    """Assert ``consumer_cls`` correctly applies the structured-config pattern.

    Combines these checks for adopters of
    :class:`~dataknobs_common.structured_config.StructuredConfigConsumer`:

    1. ``consumer_cls`` declares a ``CONFIG_CLS`` ClassVar.
    2. ``CONFIG_CLS`` is a ``StructuredConfig`` subclass.
    3. ``CONFIG_CLS`` field set matches ``consumer_cls.__init__``
       parameter set (delegates to
       :func:`assert_dataclass_config_matches_ctor`). The implicit
       ``self`` and ``config`` kwarg are ignored; the variadic
       ``**kwargs`` channel that the mixin provides is also ignored.
    4. **MRO ordering.** If ``consumer_cls`` does not define its own
       ``__init__`` (i.e. it relies on the mixin's), the inherited
       ``__init__`` must resolve to ``StructuredConfigConsumer.__init__``
       — proving the mixin precedes any other base that defines a
       competing ``__init__``. A consumer that overrides ``__init__``
       (the documented back-compat shortcut) is exempt.
    5. **Async-entry symmetry.** If ``consumer_cls`` overrides
       ``from_config_async``, the override must route through
       ``_coerce_config`` (the same guard ``from_config`` uses) — catches
       an async factory that bypasses the typed-config dispatch.
    6. (Optional) If ``expected_factory`` is supplied, its body
       delegates to ``consumer_cls.from_config(config)`` (delegates to
       :func:`assert_factory_kwargs_match_ctor`) — proves the registry
       factory hasn't regressed to a per-kwarg allowlist.

    Args:
        consumer_cls: A class mixing in
            :class:`~dataknobs_common.structured_config.StructuredConfigConsumer`.
        expected_factory: Optional registry-factory callable; when
            supplied, asserts it delegates to ``from_config``.
        ignore_params: Additional ctor params to ignore (forwarded to
            :func:`assert_dataclass_config_matches_ctor`). Use for
            back-compat positional shortcuts that intentionally live
            outside the dataclass surface.

    Raises:
        AssertionError: If any of the four checks fails.
    """
    # Lazy import to keep ``testing.factory_parity`` from transitively
    # importing the abstraction it helps test.
    from dataknobs_common.structured_config import (
        StructuredConfig,
        StructuredConfigConsumer,
    )

    config_cls = getattr(consumer_cls, "CONFIG_CLS", None)
    if config_cls is None:
        raise AssertionError(
            f"{consumer_cls.__name__} does not declare CONFIG_CLS "
            "(required by StructuredConfigConsumer)."
        )
    if not (
        isinstance(config_cls, type)
        and issubclass(config_cls, StructuredConfig)
    ):
        raise AssertionError(
            f"{consumer_cls.__name__}.CONFIG_CLS is {config_cls!r}, "
            "not a StructuredConfig subclass."
        )

    assert_dataclass_config_matches_ctor(
        config_cls,
        consumer_cls,
        ignore_params=ignore_params or set(),
    )

    # Check 4 — MRO ordering. When the consumer relies on the mixin's
    # ``__init__`` (does not define its own), the resolved ``__init__``
    # must be the mixin's. If another base precedes the mixin in the MRO
    # and shadows it, construction never runs the typed-config dispatch.
    if "__init__" not in consumer_cls.__dict__:
        if consumer_cls.__init__ is not StructuredConfigConsumer.__init__:
            resolved = getattr(
                consumer_cls.__init__, "__qualname__", consumer_cls.__init__
            )
            raise AssertionError(
                f"{consumer_cls.__name__}: StructuredConfigConsumer must "
                "precede other bases so its __init__ is the construction "
                f"entry point, but the resolved __init__ is {resolved}. "
                "List StructuredConfigConsumer first among the bases."
            )

    # Check 5 — async-entry symmetry. An overridden ``from_config_async``
    # must route through ``_coerce_config`` like ``from_config`` does.
    if "from_config_async" in consumer_cls.__dict__:
        raw = consumer_cls.__dict__["from_config_async"]
        if isinstance(raw, classmethod):
            raw = raw.__func__
        try:
            src = textwrap.dedent(inspect.getsource(raw))
        except (OSError, TypeError) as e:
            raise AssertionError(
                f"Cannot read source of "
                f"{consumer_cls.__name__}.from_config_async: {e}"
            ) from e
        routes_through_guard = any(
            isinstance(node, ast.Attribute)
            and node.attr == "_coerce_config"
            for node in ast.walk(ast.parse(src))
        )
        if not routes_through_guard:
            raise AssertionError(
                f"{consumer_cls.__name__}.from_config_async does not route "
                "through _coerce_config; it must use the same typed-config "
                "guard as from_config so a wrong-subclass config is rejected."
            )

    if expected_factory is not None:
        assert_factory_kwargs_match_ctor(
            expected_factory,
            consumer_cls,
            ignore_kwargs=ignore_params or set(),
        )


def assert_structured_config_roundtrip(config: Any) -> None:
    """Assert ``type(cfg).from_dict(cfg.to_dict()) == cfg``.

    Property assertion for a
    :class:`~dataknobs_common.structured_config.StructuredConfig`
    instance. Eliminates the per-consumer round-trip boilerplate that
    every adopter of the abstraction would otherwise duplicate.

    The property holds for flat configs and for nested configs alike:
    ``to_dict`` recurses via ``asdict`` and ``from_dict`` recurses back
    into the matching field types, so the two are symmetric for every
    statically-typed nesting shape (``SubCfg``, ``SubCfg | None``,
    ``list[SubCfg]``, ``dict[K, SubCfg]``, ``dict[K, list[SubCfg]]``). No
    ``_normalize_dict`` override is required for nesting alone.

    Args:
        config: A ``StructuredConfig`` instance to round-trip.

    Raises:
        AssertionError: If the recovered config is not equal to the
            original. The mismatch is included in the failure message.
    """
    recovered = type(config).from_dict(config.to_dict())
    if recovered != config:
        raise AssertionError(
            f"Round-trip mismatch for {type(config).__name__}:\n"
            f"  original={config!r}\n  recovered={recovered!r}"
        )


__all__ = [
    "assert_ctor_reads_documented_keys",
    "assert_dataclass_config_matches_ctor",
    "assert_factory_kwargs_match_ctor",
    "assert_structured_config_consumer",
    "assert_structured_config_roundtrip",
]
