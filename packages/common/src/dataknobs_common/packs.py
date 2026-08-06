"""Ordered, precedence-resolved composition of named declaration bundles.

A *pack* is a named, frozen, partial declaration. A *binding* selects
packs for one deployment and may tune them. Resolution folds the selected
packs — in ascending priority order — into a single composed declaration,
field by field, under rules the spec class *declares*.

The three moving parts:

- :class:`PackSpec` — the base a consumer subclasses to name its fields.
  Every non-meta field declares how it composes in ``_COMPOSITION``.
- :class:`PackRegistry` — a :class:`~dataknobs_common.registry.Registry`
  of specs keyed by ``spec.name``, plus the pure :meth:`PackRegistry.resolve`.
- :class:`PackResolution` — what resolution produced: the applied specs in
  order, the composed spec, and any :class:`PackWarning` diagnostics.

The module is deliberately vocabulary-free: it knows ``name``, ``priority``,
``enabled``, ``locked``, and a per-field merge rule. It knows nothing about
what the fields *mean*. Domain packages define the vocabulary by subclassing
:class:`PackSpec`.

Precedence semantics, and how they differ from the neighbours:

    Three distinct merge semantics now live in the tree, and confusing them
    is the easy mistake:

    - :class:`~dataknobs_common.resolver.CompositeResolver` — first-*record*
      wins. The first resolver returning a non-``None`` value supplies the
      whole answer.
    - ``dataknobs_llm.llm.model_profile.merge_partials`` — first-*facet*
      wins. Sources are merged attribute-by-attribute, first non-``None``
      per attribute.
    - **This module** — an *ordered fold under per-field declared rules*.
      Every participating pack contributes to every field it sets, and the
      field's own rule decides what "contribute" means: concatenate,
      shallow-merge, take the last value, take the first, or require
      unanimity.

    Order is **ascending priority** — the lower ``priority`` value folds
    first — matching :class:`~dataknobs_common.callbacks.PriorityOrdering`
    ("lower fires first") and the ``priority=-100`` guard idiom. Ties break
    by registration order (FIFO), which is the same key
    ``CompositeOrdering(PriorityOrdering(), FIFOOrdering())`` computes; that
    family is not reused directly only because its ``compare()`` is pinned
    to the callback-specific ``CallbackEntry``.

    Because the fold runs low-to-high, ``LAST_WINS``/``MERGE`` give the
    **highest** priority the final say, while ``FIRST_WINS``/``UNANIMOUS``
    let the **lowest** (earliest) pack pin a value.

Failure posture — two error families, deliberately distinct:

- **Declaration/wiring errors** raise :class:`ConfigurationError`: a field
  with no declared rule, a rule for a field that does not exist, a non-meta
  field with no default, a registered spec whose class adds fields the
  registry's class cannot compose, a *spec's* value whose shape contradicts
  its rule, a custom reducer that raises. These are authoring bugs. Most
  surface when a :class:`PackRegistry` is constructed or a spec registered;
  a value's shape and a reducer's behaviour are only knowable once there is
  a value, so those surface at :meth:`PackRegistry.resolve`.
- **Resolution errors** raise :class:`PackResolutionError` (a
  ``ConfigurationError`` subclass) carrying ``context["reason"]``: an unknown
  pack name, an unknown binding key, a locked-but-disabled pack, a conflict
  under ``UNANIMOUS``, a malformed binding, or a *binding's* value whose
  shape contradicts its rule.

The split follows the value's **origin**, not its symptom: the same
malformed value is a programmer error inside a spec and operator input
inside a binding, so only the latter carries a machine-readable ``reason``.

Nothing here does I/O, and :meth:`PackRegistry.resolve` is pure: it mutates
neither the registry, the bindings mapping, nor the registered specs.

Example:
    ```python
    from dataclasses import dataclass, field
    from types import MappingProxyType
    from dataknobs_common.packs import MergeKind, PackRegistry, PackSpec

    @dataclass(frozen=True)
    class PolicyPack(PackSpec):
        tier: str | None = None
        checks: tuple[str, ...] = ()
        limits: dict[str, int] = field(default_factory=dict)

        _COMPOSITION = MappingProxyType({
            "tier": MergeKind.UNANIMOUS,
            "checks": MergeKind.CONCAT_UNIQUE,
            "limits": MergeKind.MERGE,
        })

    registry = PackRegistry("policies", PolicyPack)
    registry.register_pack(PolicyPack(name="base", checks=("pii",), limits={"rps": 10}))
    registry.register_pack(
        PolicyPack(name="strict", priority=10, checks=("pii", "toxicity"),
                   limits={"rps": 2})
    )

    resolution = registry.resolve({"base": {}, "strict": {"locked": True}})
    resolution.packs                 # ('base', 'strict')
    resolution.spec.checks           # ('pii', 'toxicity')
    resolution.spec.limits           # {'rps': 2}  — highest priority wins the key
    ```
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Generic, NamedTuple, TypeAlias, TypeVar

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.registry import Registry
from dataknobs_common.structured_config import StructuredConfig

SpecT = TypeVar("SpecT", bound="PackSpec")

#: Binding keys that are flags rather than field overrides.
_BINDING_FLAGS: frozenset[str] = frozenset({"enabled", "locked"})


class _Unset:
    """The type of :data:`UNSET`. Not for instantiation by consumers."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        """Falsy, so ``spec.field or fallback`` reads the way ``None`` does."""
        return False

    def __copy__(self) -> _Unset:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _Unset:
        return self

    def __reduce__(self) -> str:
        return "UNSET"


#: Sentinel default marking a field as *not contributed*.
#:
#: Participation compares a contributed value against that field's **declared
#: default** (see :func:`_fold`), so a field defaulting to ``None`` cannot
#: distinguish "did not mention this" from "explicitly none". Declaring
#: ``UNSET`` as the default moves the "absent" marker off the domain's own
#: value space, making ``None`` — and every other domain value — an ordinary
#: participating contribution::
#:
#:     @dataclass(frozen=True)
#:     class IngestPack(PackSpec):
#:         chunker: str | None = UNSET
#:         _COMPOSITION = MappingProxyType({"chunker": MergeKind.LAST_WINS})
#:
#: The cost is local and visible: a field no pack ever set reads back as
#: ``UNSET`` rather than a natural empty value, so consumers handle one extra
#: case at the point of use. Worth it for ``LAST_WINS`` / ``FIRST_WINS`` /
#: ``UNANIMOUS``, where a default-valued contribution is meaningful; pointless
#: for ``CONCAT`` / ``CONCAT_UNIQUE`` / ``MERGE``, where an empty contribution
#: is already a no-op.
#:
#: Typed ``Any`` so it can stand as the default of a narrowly-typed field
#: without a per-field ``type: ignore``. Identity is the contract — it is a
#: module-level singleton preserved across copy, deepcopy, and pickle — so
#: test it with ``is UNSET``. Being an object rather than a string, it does
#: not survive a JSON round-trip; a spec that must serialize to JSON should
#: keep a natural default and encode "explicitly off" as a domain value.
UNSET: Any = _Unset()


class MergeKind(Enum):
    """Built-in per-field composition rules.

    Every rule is applied as a left fold over the participating packs in
    ascending-priority order, so "last" means *highest priority* and
    "first" means *lowest priority*.
    """

    #: The last participating pack's value wins.
    LAST_WINS = "last_wins"
    #: The first participating pack's value wins — lets an early (low
    #: priority) pack *pin* a value a later one cannot change.
    FIRST_WINS = "first_wins"
    #: Every participating pack must agree; disagreement is a
    #: :class:`PackResolutionError`.
    UNANIMOUS = "unanimous"
    #: Sequences concatenated in fold order, duplicates kept.
    CONCAT = "concat"
    #: Sequences concatenated in fold order, order-preserving dedup.
    CONCAT_UNIQUE = "concat_unique"
    #: Shallow mapping merge; later packs win per key.
    MERGE = "merge"


#: A pure two-argument fold step: ``(accumulated, next) -> value``. The
#: escape hatch for semantics the built-in :class:`MergeKind` vocabulary
#: does not cover. ``dataknobs_config.inheritance.deep_merge`` matches this
#: signature exactly, so recursive merge is available through the hatch
#: without this package depending on ``dataknobs-config``.
Reducer: TypeAlias = Callable[[Any, Any], Any]

#: What a spec class declares for each of its fields in ``_COMPOSITION``:
#: either a built-in :class:`MergeKind` or a custom :data:`Reducer`.
CompositionRule: TypeAlias = MergeKind | Reducer


@dataclass(frozen=True)
class PackWarning:
    """A non-fatal resolution diagnostic.

    Structured rather than a bare string so a deployment can escalate a
    specific ``code`` to a hard failure without pattern-matching prose.

    Attributes:
        code: Stable machine-readable discriminator — one of
            ``"priority_tie"`` (selected packs share a priority *and*
            contend for a field), ``"value_override"`` (``LAST_WINS`` /
            ``FIRST_WINS`` discarded a differing value), ``"key_override"``
            (``MERGE`` overrode keys another pack had set), or
            ``"binding_override_ignored"`` (a binding named a field whose
            declared rule then discarded the binding's value).
        message: Human-readable description.
        packs: Names of the packs involved, in fold order.
        field: The spec field the warning concerns, or ``None`` for
            whole-resolution diagnostics such as a priority tie.
    """

    code: str
    message: str
    packs: tuple[str, ...] = ()
    field: str | None = None

    def __str__(self) -> str:
        return self.message


class PackResolutionError(ConfigurationError):
    """Raised for every fail-closed rejection during resolution.

    ``context["reason"]`` (also available as :attr:`reason`) is one of
    ``"unknown_pack"``, ``"unknown_binding_key"``, ``"locked_pack_disabled"``,
    ``"field_conflict"``, ``"invalid_binding"``.

    Distinct from a plain :class:`ConfigurationError`, which this module
    raises for *declaration* problems (a spec class whose composition plan
    is incoherent) rather than *resolution* problems (a binding that cannot
    be honoured).
    """

    def __init__(self, message: str, *, reason: str, **context: Any) -> None:
        super().__init__(message, context={"reason": reason, **context})
        self.reason = reason


@dataclass(frozen=True)
class PackSpec(StructuredConfig):
    """Base for a named, precedence-ordered, composable declaration.

    A subclass names its fields and declares how each composes::

        @dataclass(frozen=True)
        class MyPack(PackSpec):
            steps: tuple[str, ...] = ()
            options: dict[str, Any] = field(default_factory=dict)

            _COMPOSITION = MappingProxyType({
                "steps": MergeKind.CONCAT,
                "options": MergeKind.MERGE,
            })

    Every non-meta field **must** appear in ``_COMPOSITION`` and **must**
    have a default (a pack is a *partial* contribution; an unset field is
    one the pack does not speak to). Both rules are enforced when a
    :class:`PackRegistry` is constructed — a field added later without a
    rule fails loudly instead of being silently dropped from composition.

    Subclasses must be ``@dataclass(frozen=True)`` — the frozen-ness is
    inherited as a constraint, and immutability is what makes resolution
    safe to run repeatedly against a shared registry.

    Value normalization: :meth:`__post_init__` converts ``list`` values to
    ``tuple``, ``set`` values to ``frozenset``, and copies ``Mapping``
    values. This is necessary because
    :meth:`~dataknobs_common.structured_config.StructuredConfig.from_dict`
    assigns non-config, non-enum field values verbatim — a YAML list would
    otherwise land in a ``tuple[...]``-annotated field as a ``list``, and a
    caller's dict would alias into the "frozen" spec.

    Normalization is *shallow*, and ``frozen=True`` blocks only rebinding a
    field, never writing *through* one. So a ``dict``-typed field value and
    every container element are **read-only by convention**: writing to one
    mutates the registered spec in place and changes what every later
    resolution composes. Copy before mutating
    (``dict(spec.limits)``). Sequence and set fields are immutable by
    construction; mappings are not, deliberately — see
    :func:`_normalize_value`.

    A subclass defining its own ``__post_init__`` **must** call
    ``super().__post_init__()``.
    """

    name: str
    priority: int = 0

    #: Fields that describe the pack rather than contribute to composition.
    #: Excluded from ``_COMPOSITION`` and from the fold.
    _META_FIELDS: ClassVar[frozenset[str]] = frozenset({"name", "priority"})

    #: Per-field composition rules. Keyed by field name; every non-meta
    #: field must appear. A frozen ``MappingProxyType`` default so an
    #: accidental ``PackSpec._COMPOSITION[...] = ...`` raises rather than
    #: mutating the shared base default.
    _COMPOSITION: ClassVar[Mapping[str, CompositionRule]] = MappingProxyType({})

    def __post_init__(self) -> None:
        """Normalize and copy container field values in place (see class docs).

        Note the asymmetry with the instruction this class gives *its*
        subclasses: no ``super().__post_init__()`` call here, because
        :class:`StructuredConfig` defines none and calling it would raise
        ``AttributeError``. Subclasses of ``PackSpec`` do have one to call —
        this method — so they must.
        """
        for f in dataclasses.fields(self):
            current = getattr(self, f.name)
            normalized = _normalize_value(current)
            if normalized is not current:
                object.__setattr__(self, f.name, normalized)


@dataclass(frozen=True)
class PackResolution(Generic[SpecT]):
    """The outcome of resolving a binding mapping against a registry.

    Attributes:
        applied: The binding-applied specs in fold order (ascending
            priority, FIFO tie-break). Disabled packs are absent.
        spec: The composed declaration — an instance of the registry's spec
            class, so consumers read typed fields rather than an untyped
            bag. Its ``name`` is the caller-supplied ``composed_name`` and
            its ``priority`` is the class default; neither meta field is
            meaningful on a composed result.
        warnings: Non-fatal diagnostics, in the order they were produced.
    """

    applied: tuple[SpecT, ...]
    spec: SpecT
    warnings: tuple[PackWarning, ...] = ()

    @property
    def packs(self) -> tuple[str, ...]:
        """Names of the applied packs, in fold order."""
        return tuple(spec.name for spec in self.applied)


@dataclass(frozen=True)
class _CompositionPlan:
    """Validated, cached composition metadata for one spec class.

    Deliberately holds **no** reference to the spec class: the plan is the
    value in a :class:`weakref.WeakKeyDictionary` keyed on that class, and a
    value referencing its own key would pin the entry forever. Code needing
    the class takes it as a separate argument.
    """

    #: Non-meta field name -> declared rule, in dataclass field order.
    rules: Mapping[str, CompositionRule]
    #: Non-meta field name -> its normalized declared default.
    defaults: Mapping[str, Any]
    #: Every key a binding body may legally carry.
    binding_keys: frozenset[str]


def _normalize_value(value: Any) -> Any:
    """Return a normalized, non-aliasing copy of a field value.

    ``list`` -> ``tuple``, ``set`` -> ``frozenset``, ``Mapping`` -> a fresh
    ``dict``. Everything else — scalars, tuples, frozensets, live objects —
    passes through by identity. Shallow by design: elements are shared.

    Sequences and sets end up genuinely immutable; **mappings do not**, and
    that asymmetry is deliberate rather than an oversight. Wrapping a
    mapping in ``MappingProxyType`` would make it unwritable but also
    un-``deepcopy``-able, and
    :meth:`~dataknobs_common.structured_config.StructuredConfig.to_dict`
    delegates to ``dataclasses.asdict``, which deep-copies every field
    value — so freezing would trade a documented read-only *convention* for
    a broken serialization *contract*, and would booby-trap any consumer
    calling ``dataclasses.asdict`` on a spec directly. A dict field is
    therefore copied (never aliased to the caller's) but must be treated as
    read-only, exactly as its elements must be.
    """
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, (set, frozenset)):
        return frozenset(value)
    return value


def _field_default(f: dataclasses.Field[Any]) -> Any:
    """The declared default of a field, or ``dataclasses.MISSING``."""
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return dataclasses.MISSING


#: Memo for :func:`_composition_plan`, keyed *weakly* on the spec class.
#:
#: A plain ``functools.cache`` would key a process-lifetime dict on the class
#: itself, making every spec class permanently reachable along with its
#: module globals. Harmless for module-scope specs, but a process that builds
#: spec classes dynamically — or a test suite defining them per-test — would
#: leak one per class.
#:
#: This only works because ``_CompositionPlan`` holds no strong reference back
#: to the class: a value that referenced its own key would keep the entry
#: alive forever, which is why ``_assert_composable`` takes the class as a
#: separate argument rather than reading it off the plan.
_PLAN_CACHE: weakref.WeakKeyDictionary[type[PackSpec], _CompositionPlan] = (
    weakref.WeakKeyDictionary()
)


def _composition_plan(spec_cls: type[PackSpec]) -> _CompositionPlan:
    """Validate a spec class's composition declaration and cache the plan.

    Caching is a pure memo: concurrent first calls may both build a plan and
    the last write wins, which is harmless because the result depends only
    on the class and nothing compares plans by identity. Failures are not
    cached — an invalid class re-raises on every call.

    Four checks, all fail-closed:

    1. every non-meta dataclass field appears in ``_COMPOSITION``;
    2. every ``_COMPOSITION`` key is a real non-meta field;
    3. every non-meta field has a default or ``default_factory`` — reachable
       only for a ``kw_only`` spec class, since ordinary field ordering
       already makes a defaultless field after ``priority`` a ``TypeError``
       from ``dataclasses`` itself;
    4. every rule is a :class:`MergeKind` member or a callable.

    Deliberately *not* run from ``__init_subclass__``: that hook fires
    before the subclass's ``@dataclass`` decorator, so the field set is not
    yet observable there (``dataclasses.fields`` would return the *base's*
    fields, inherited via ``__dataclass_fields__``, and report a subclass
    with no rules as valid). Validation is instead lazy and cached, driven
    from :class:`PackRegistry` construction and :func:`compose_packs`.

    Raises:
        ConfigurationError: If the class is not a frozen ``PackSpec``
            dataclass, or any of the four checks fails.
    """
    if not (isinstance(spec_cls, type) and issubclass(spec_cls, PackSpec)):
        raise ConfigurationError(
            f"Expected a PackSpec subclass, got {spec_cls!r}",
            context={"spec_cls": repr(spec_cls)},
        )
    cached = _PLAN_CACHE.get(spec_cls)
    if cached is not None:
        return cached
    # NOT ``dataclasses.is_dataclass``: that tests for ``__dataclass_fields__``,
    # which every subclass inherits from the decorated base — so it is true even
    # for an undecorated subclass, whose own annotations are then invisible to
    # ``dataclasses.fields``. The failure would surface below as "_COMPOSITION
    # keys that are not non-meta fields", pointing at the wrong thing entirely.
    # (Non-frozen needs no check: ``dataclasses`` itself refuses to inherit a
    # non-frozen dataclass from a frozen one.)
    if "__dataclass_fields__" not in spec_cls.__dict__:
        raise ConfigurationError(
            f"{spec_cls.__qualname__} must be decorated with "
            "@dataclass(frozen=True) to be used as a pack spec; without it "
            "its own fields are invisible to composition",
            context={"spec_cls": spec_cls.__qualname__},
        )

    meta = frozenset(spec_cls._META_FIELDS)
    declared = dict(spec_cls._COMPOSITION)
    all_fields = {f.name for f in dataclasses.fields(spec_cls)}

    rules: dict[str, CompositionRule] = {}
    defaults: dict[str, Any] = {}
    missing_rule: list[str] = []
    missing_default: list[str] = []
    bad_rule: list[str] = []

    for f in dataclasses.fields(spec_cls):
        if f.name in meta:
            continue
        default = _field_default(f)
        if default is dataclasses.MISSING:
            missing_default.append(f.name)
        if f.name not in declared:
            missing_rule.append(f.name)
            continue
        rule = declared[f.name]
        if not isinstance(rule, MergeKind) and not callable(rule):
            bad_rule.append(f.name)
            continue
        if default is not dataclasses.MISSING:
            rules[f.name] = rule
            defaults[f.name] = _normalize_value(default)

    unknown_rule = sorted(set(declared) - (all_fields - meta))

    if missing_rule or missing_default or bad_rule or unknown_rule:
        problems: list[str] = []
        if missing_rule:
            problems.append(f"fields with no _COMPOSITION rule: {sorted(missing_rule)}")
        if unknown_rule:
            problems.append(f"_COMPOSITION keys that are not non-meta fields: {unknown_rule}")
        if missing_default:
            problems.append(
                f"fields with no default (a pack is a partial contribution): "
                f"{sorted(missing_default)}"
            )
        if bad_rule:
            problems.append(f"rules that are neither a MergeKind nor callable: {sorted(bad_rule)}")
        raise ConfigurationError(
            f"{spec_cls.__qualname__} has an incoherent composition plan — " + "; ".join(problems),
            context={
                "spec_cls": spec_cls.__qualname__,
                "missing_rule": sorted(missing_rule),
                "unknown_rule": unknown_rule,
                "missing_default": sorted(missing_default),
                "invalid_rule": sorted(bad_rule),
            },
        )

    plan = _CompositionPlan(
        rules=MappingProxyType(rules),
        defaults=MappingProxyType(defaults),
        # Composable fields plus the two flags plus ``priority`` — NOT every
        # non-``name`` field. A spec class that extends ``_META_FIELDS`` gets
        # no fold handling for the extra descriptor, so advertising it as a
        # binding key would accept it and then silently ignore it.
        binding_keys=frozenset(_BINDING_FLAGS | set(rules) | {"priority"}),
    )
    _PLAN_CACHE[spec_cls] = plan
    return plan


def _assert_composable(
    spec_type: type[PackSpec], spec_cls: type[PackSpec], plan: _CompositionPlan
) -> None:
    """Reject a spec whose own class adds fields the plan cannot compose.

    ``isinstance`` admits a subclass, but composition is defined entirely by
    the registry's declared class: the composed result is an instance of
    *that* class, built from *its* rules. A field only the subclass declares
    therefore has no rule, no plan default, and nowhere to land in the
    composed spec — it would be dropped silently, yielding a plausible-
    looking result that is missing data. Two subclasses in one registry
    could not produce a single composed type at all.

    Subclassing for *behaviour* stays legal; it is extra fields that cannot
    be honoured, so that is exactly what this rejects.
    """
    if spec_type is spec_cls:
        return
    extra = sorted(
        {f.name for f in dataclasses.fields(spec_type)}
        - {f.name for f in dataclasses.fields(spec_cls)}
    )
    if not extra:
        return
    raise ConfigurationError(
        f"{spec_type.__qualname__} declares field(s) {extra} that "
        f"{spec_cls.__qualname__} does not, so composition — which "
        f"builds a {spec_cls.__qualname__} from its own rules — cannot "
        "carry them. Declare the fields on the registry's spec class.",
        context={
            "spec_cls": spec_cls.__qualname__,
            "got": spec_type.__qualname__,
            "uncomposable_fields": extra,
        },
    )


def _rule_label(rule: CompositionRule) -> str:
    """A stable string for a rule, for error context.

    A custom :data:`Reducer` has no ``.value``, so fall back to its name.
    """
    if isinstance(rule, MergeKind):
        return rule.value
    return getattr(rule, "__name__", repr(rule))


def _shape_error(
    message: str, field_name: str, source: str, rule: CompositionRule, from_binding: bool
) -> ConfigurationError:
    """Build the right error family for a bad contributed value.

    The two families answer different questions. A value inside a *spec* is
    written by the pack author in code, so a bad shape there is a
    programmer error that should never reach a request — plain
    :class:`ConfigurationError`. The same bad shape inside a *binding* is
    operator input, which a deployment has to be able to catch and report
    against a stable ``reason`` rather than by matching prose.
    """
    context = {"field": field_name, "source": source, "rule": _rule_label(rule)}
    if from_binding:
        return PackResolutionError(message, reason="invalid_binding", **context)
    return ConfigurationError(message, context=context)


def _check_value_shape(
    rule: CompositionRule,
    field_name: str,
    value: Any,
    source: str,
    *,
    from_binding: bool = False,
) -> None:
    """Reject a contributed value whose shape contradicts its declared rule.

    ``CONCAT``/``CONCAT_UNIQUE`` need a non-string sequence and ``MERGE``
    needs a mapping. Without this, ``CONCAT`` on a ``str`` would silently
    explode it into characters and ``MERGE`` on a scalar would raise an
    opaque ``TypeError`` deep in the fold. The check runs on every
    contribution — including the first — so a one-pack resolution and a
    two-pack resolution reject the same bad values.

    ``from_binding`` selects the error family (see :func:`_shape_error`).
    """
    if rule is MergeKind.MERGE:
        if not isinstance(value, Mapping):
            raise _shape_error(
                f"Field '{field_name}' declares MergeKind.MERGE but "
                f"'{source}' contributed a {type(value).__name__}; "
                "MERGE requires a mapping",
                field_name,
                source,
                rule,
                from_binding,
            )
    elif rule in (MergeKind.CONCAT, MergeKind.CONCAT_UNIQUE):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise _shape_error(
                f"Field '{field_name}' declares {rule} but '{source}' "
                f"contributed a {type(value).__name__}; concatenation "
                "requires a non-string sequence",
                field_name,
                source,
                rule,
                from_binding,
            )


def _concat_unique(acc: Sequence[Any], nxt: Sequence[Any]) -> tuple[Any, ...]:
    """Concatenate, dropping later duplicates, preserving first-seen order.

    Equality-based rather than hash-based so it is total over unhashable
    elements (a sequence of raw config mappings, for instance).
    """
    out: list[Any] = []
    for item in (*acc, *nxt):
        if item not in out:
            out.append(item)
    return tuple(out)


class _Reduction(NamedTuple):
    """One fold step's outcome.

    ``contributed`` records whether the incoming pack influenced ``value``.
    It is false only where a rule discards the incoming value outright
    (``FIRST_WINS`` against an already-pinned field), so that pack is left
    out of the field's provenance and cannot appear in a later warning's
    source list.
    """

    value: Any
    warnings: tuple[PackWarning, ...]
    contributed: bool


def _reduce_field(
    rule: CompositionRule,
    field_name: str,
    acc: Any,
    nxt: Any,
    acc_packs: tuple[str, ...],
    next_pack: str,
) -> _Reduction:
    """Fold one further contribution into a field's accumulated value.

    Returns the new value, any diagnostics, and whether ``next_pack``
    actually contributed to the result — the last so provenance names packs
    that influenced the value rather than packs that merely tried. Raises
    :class:`PackResolutionError` only for a ``UNANIMOUS`` conflict — the one
    case where two packs make incompatible claims that no rule can
    reconcile.

    Why the built-in kinds are a dispatch here rather than a
    ``MergeKind -> Reducer`` lookup table: ``UNANIMOUS`` is a *constraint*,
    not a combiner (a pure ``(acc, next) -> value`` reducer for it could
    only return ``acc`` and swallow the disagreement), and the diagnostics
    need the field name and contributing pack names — context the
    :data:`Reducer` signature deliberately does not carry. A table would
    therefore need a parallel per-kind diagnostics branch, giving two
    dispatch points that can drift. One branch decides both.
    """
    if not isinstance(rule, MergeKind):
        # A custom reducer gets no _check_value_shape pass — the module cannot
        # know what shapes it accepts — so the wrap is what supplies the field
        # and pack context that check exists to guarantee. Without it a
        # mismatched value surfaces as a bare TypeError from inside the fold,
        # naming neither the field nor the pack that contributed it.
        try:
            return _Reduction(rule(acc, nxt), (), True)
        except Exception as exc:
            raise ConfigurationError(
                f"The custom reducer for '{field_name}' failed folding "
                f"'{next_pack}' into {list(acc_packs)}: {exc}",
                context={
                    "field": field_name,
                    "source": next_pack,
                    "rule": _rule_label(rule),
                    "packs": [*acc_packs, next_pack],
                },
            ) from exc

    if rule is MergeKind.UNANIMOUS:
        if acc != nxt:
            raise PackResolutionError(
                f"Packs disagree on '{field_name}': "
                f"{acc_packs} declare {acc!r} but '{next_pack}' declares {nxt!r}. "
                "The field is declared UNANIMOUS — every pack that sets it must agree.",
                reason="field_conflict",
                field=field_name,
                packs=[*acc_packs, next_pack],
                values=[repr(acc), repr(nxt)],
            )
        # Agreement is a contribution: both packs assert this value.
        return _Reduction(acc, (), True)

    if rule in (MergeKind.LAST_WINS, MergeKind.FIRST_WINS):
        winner, loser = (nxt, acc) if rule is MergeKind.LAST_WINS else (acc, nxt)
        if acc == nxt:
            return _Reduction(winner, (), True)
        # Under FIRST_WINS the later value is thrown away, so this pack did not
        # contribute: recording it would make a *subsequent* pack's warning cite
        # a pack whose value was already discarded.
        contributed = rule is MergeKind.LAST_WINS
        return _Reduction(
            winner,
            (
                PackWarning(
                    code="value_override",
                    message=(
                        f"'{field_name}' set by multiple packs "
                        f"({[*acc_packs, next_pack]}); "
                        f"{rule} keeps {winner!r} and discards {loser!r}"
                    ),
                    packs=(*acc_packs, next_pack),
                    field=field_name,
                ),
            ),
            contributed,
        )

    if rule is MergeKind.CONCAT:
        return _Reduction(tuple(acc) + tuple(nxt), (), True)

    if rule is MergeKind.CONCAT_UNIQUE:
        return _Reduction(_concat_unique(acc, nxt), (), True)

    # MergeKind.MERGE — the only remaining member.
    overridden = sorted(str(key) for key in nxt if key in acc and acc[key] != nxt[key])
    merged = {**acc, **nxt}
    if not overridden:
        return _Reduction(merged, (), True)
    return _Reduction(
        merged,
        (
            PackWarning(
                code="key_override",
                message=(
                    f"'{next_pack}' overrides keys {overridden} of '{field_name}' "
                    f"previously set by {list(acc_packs)}"
                ),
                packs=(*acc_packs, next_pack),
                field=field_name,
            ),
        ),
        True,
    )


@dataclass(frozen=True)
class _Contribution:
    """One labelled set of field values entering the fold.

    ``origin`` records where the values came from, which is what decides
    whether *presence of a key* is meaningful — a property of the source,
    not of the content:

    - A **spec** is a frozen dataclass, so every field is always present and
      "did not mention it" is unrecoverable from the mapping alone. Absence
      has to be inferred by comparing against the declared default.
    - A **binding body** is a partial mapping written by hand, so naming a
      field *is* the contribution. Comparing against the default there would
      discard information the mapping actually carries, and would silently
      ignore an operator explicitly clearing a field.
    """

    label: str
    values: Mapping[str, Any]
    #: ``"spec"`` for a registered declaration, ``"binding"`` for a
    #: deployment body. One fact with two consequences — it decides both
    #: how participation is measured and which error family a bad value
    #: raises — so it is stored once rather than as two booleans that could
    #: be set inconsistently.
    origin: str = "spec"

    @property
    def explicit(self) -> bool:
        """Whether *presence of a key* is meaningful for this contribution."""
        return self.origin == "binding"


def _fold(
    plan: _CompositionPlan,
    contributions: Sequence[_Contribution],
) -> tuple[dict[str, Any], tuple[PackWarning, ...]]:
    """Left-fold labelled field contributions under the plan's declared rules.

    A field from a non-``explicit`` contribution **participates** only when
    its value differs from that field's declared default — "unset" is "still
    at the default". That is what makes ``LAST_WINS``/``UNANIMOUS`` behave: a
    pack that does not mention a field must not clobber a pack that does.
    (The same family as the first-non-``None``-per-facet rule in the LLM
    profile resolver, generalized from ``None`` to the declared default.)
    ``CONCAT``/``MERGE`` are unaffected — an empty contribution is already a
    no-op. Declare :data:`UNSET` as a field's default to opt that field out
    of default-comparison entirely.

    An ``explicit`` contribution participates on presence alone. Note this
    decides *participation*, not the outcome: the field's declared rule still
    applies, so an explicit contribution cannot change a ``FIRST_WINS`` field
    already pinned by an earlier one, and disagreeing with a ``UNANIMOUS``
    field still raises.

    Shared by :func:`compose_packs` (spec after spec) and by binding
    application (spec first, binding second), so the two cannot drift.
    """
    values: dict[str, Any] = {}
    sources: dict[str, list[str]] = {}
    warnings: list[PackWarning] = []

    for contribution in contributions:
        label = contribution.label
        contributed = contribution.values
        for field_name, rule in plan.rules.items():
            if field_name not in contributed:
                continue
            value = _normalize_value(contributed[field_name])
            if not contribution.explicit and value == plan.defaults[field_name]:
                continue
            _check_value_shape(rule, field_name, value, label, from_binding=contribution.explicit)
            if field_name not in values:
                # Seed concatenation fields as a tuple so the composed shape
                # does not depend on how many packs contributed: two or more
                # go through ``tuple(acc) + tuple(nxt)``, and a lone
                # contribution would otherwise keep whatever sequence type it
                # arrived as.
                if rule in (MergeKind.CONCAT, MergeKind.CONCAT_UNIQUE):
                    value = tuple(value)
                values[field_name] = value
                sources[field_name] = [label]
                continue
            reduction = _reduce_field(
                rule,
                field_name,
                values[field_name],
                value,
                tuple(sources[field_name]),
                label,
            )
            values[field_name] = reduction.value
            if reduction.contributed:
                sources[field_name].append(label)
            warnings.extend(reduction.warnings)

    return values, tuple(warnings)


def compose_packs(
    specs: Sequence[SpecT],
    spec_cls: type[SpecT],
    *,
    composed_name: str = "composed",
) -> tuple[SpecT, tuple[PackWarning, ...]]:
    """Fold an ordered sequence of specs into a single composed spec.

    The caller supplies the order; this function does not sort. Use
    :meth:`PackRegistry.resolve` for the priority-ordered, binding-aware
    path — this is the lower-level primitive it delegates to, exposed for
    consumers composing specs they already hold.

    Args:
        specs: Participating specs, already in fold order (lowest
            precedence first).
        spec_cls: The class to instantiate for the composed result. Every
            entry of ``specs`` must be an instance of it.
        composed_name: ``name`` for the composed spec. Neither meta field
            (``name``, ``priority``) is meaningful on a composed result.

    Returns:
        The composed spec and any diagnostics, in production order.

    Raises:
        ConfigurationError: If ``spec_cls``'s composition plan is
            incoherent, or a spec is not an instance of it.
        PackResolutionError: On a ``UNANIMOUS`` field conflict.
    """
    plan = _composition_plan(spec_cls)
    contributions: list[_Contribution] = []
    for spec in specs:
        if not isinstance(spec, spec_cls):
            raise ConfigurationError(
                f"Expected every spec to be a {spec_cls.__qualname__}, got "
                f"{type(spec).__qualname__}",
                context={"spec_cls": spec_cls.__qualname__, "got": type(spec).__qualname__},
            )
        _assert_composable(type(spec), spec_cls, plan)
        contributions.append(
            _Contribution(spec.name, {name: getattr(spec, name) for name in plan.rules})
        )
    values, warnings = _fold(plan, contributions)
    return spec_cls(name=composed_name, **values), warnings


def merge_bindings(*layers: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ordered binding layers into one binding mapping, later winning.

    Bindings arrive in layers whenever more than one authority has a say —
    a platform baseline plus a per-tenant overlay being the usual shape.
    Without this, each deployment hand-merges those dicts at wiring time,
    which is the same lossy ad-hoc merging this module exists to replace.

    It is also what makes ``locked`` more than a tautology. Within a single
    body, ``locked: true`` next to ``enabled: false`` is a contradiction the
    same author wrote and could simply not have written. Across layers it is
    load-bearing::

        platform = {"audit": {"locked": True}}
        tenant = {"audit": {"enabled": False}}
        registry.resolve(merge_bindings(platform, tenant))
        # PackResolutionError(reason="locked_pack_disabled")

    Merging is per pack and *shallow* within a body: a later layer replaces
    the keys it names and leaves the rest. Field overrides are values in
    that mapping, so a later layer's ``middleware`` replaces an earlier
    layer's rather than appending — layer precedence is a different
    question from the per-field composition a pack's ``_COMPOSITION``
    governs, and conflating them would make a binding's meaning depend on
    which layer it came from. Compose *packs* to accumulate; layer
    *bindings* to override.

    Args:
        layers: Binding mappings in ascending precedence (the last wins).

    Returns:
        A new mapping; no input layer is mutated or aliased.
    """
    merged: dict[str, Any] = {}
    for layer in layers:
        if not isinstance(layer, Mapping):
            raise PackResolutionError(
                f"Each binding layer must be a mapping; got {type(layer).__name__}",
                reason="invalid_binding",
            )
        for pack_name, body in layer.items():
            if not isinstance(body, Mapping):
                # Left for resolve() to reject with full registry context.
                merged[pack_name] = body
                continue
            existing = merged.get(pack_name)
            if isinstance(existing, Mapping):
                merged[pack_name] = {**existing, **body}
            else:
                merged[pack_name] = dict(body)
    return merged


class PackRegistry(Registry[SpecT], Generic[SpecT]):
    """A registry of named :class:`PackSpec` declarations plus resolution.

    Inherits thread-safety, optional metrics, and structural conformance to
    :class:`~dataknobs_common.registry.BackendRegistry` from
    :class:`~dataknobs_common.registry.Registry`. Packs are eagerly
    registered declarations — not lazily-constructed backends — so this
    extends ``Registry``, not ``PluginRegistry``.

    Registration order is the tie-break for equal priorities, so it is part
    of the contract, not an implementation detail.

    No module-level singleton is provided, and consumers should not create
    one: a pack binding is a *per-deployment* decision, and a process-global
    pack registry is a multi-tenant hazard. Construct one and own it.

    Args:
        name: Registry name, for logging and error context.
        spec_cls: The :class:`PackSpec` subclass this registry holds. Its
            composition plan is validated immediately, so an incoherent
            declaration fails at wiring time rather than at first resolve.
        enable_metrics: Forwarded to :class:`Registry`.

    Raises:
        ConfigurationError: If ``spec_cls``'s composition plan is incoherent.
    """

    def __init__(
        self,
        name: str,
        spec_cls: type[SpecT],
        *,
        enable_metrics: bool = False,
    ) -> None:
        super().__init__(name, enable_metrics=enable_metrics)
        _composition_plan(spec_cls)
        self._spec_cls = spec_cls

    @property
    def spec_cls(self) -> type[SpecT]:
        """The spec class this registry holds."""
        return self._spec_cls

    def register_pack(self, spec: SpecT, *, allow_overwrite: bool = False) -> None:
        """Register a spec under its own ``name``.

        The idiomatic entry point — mirrors the ``register_*(item)`` helper
        shape used by the other attribute-keyed registries in the tree.
        """
        self.register(spec.name, spec, allow_overwrite=allow_overwrite)

    def register(
        self,
        key: str,
        item: SpecT,
        metadata: dict[str, Any] | None = None,
        allow_overwrite: bool = False,
    ) -> None:
        """Register a spec, rejecting a foreign type or a key/name mismatch.

        Raises:
            ConfigurationError: If ``item`` is not an instance of this
                registry's spec class, or ``key`` differs from ``item.name``
                (which would make :meth:`resolve`'s binding lookup and the
                composed provenance disagree).
            OperationError: If the key is taken and ``allow_overwrite`` is
                ``False`` (inherited behaviour).
        """
        if not isinstance(item, self._spec_cls):
            raise ConfigurationError(
                f"{self.name} holds {self._spec_cls.__qualname__} specs; got "
                f"{type(item).__qualname__}",
                context={
                    "registry": self.name,
                    "spec_cls": self._spec_cls.__qualname__,
                    "got": type(item).__qualname__,
                },
            )
        if key != item.name:
            raise ConfigurationError(
                f"Pack registration key '{key}' does not match spec name "
                f"'{item.name}'; a pack is addressed by its own name",
                context={"registry": self.name, "key": key, "name": item.name},
            )
        _assert_composable(type(item), self._spec_cls, _composition_plan(self._spec_cls))
        super().register(key, item, metadata, allow_overwrite)

    def resolve(
        self,
        bindings: Mapping[str, Any],
        *,
        composed_name: str = "composed",
    ) -> PackResolution[SpecT]:
        """Resolve a binding mapping into an ordered, composed declaration.

        Pure and synchronous: the registry, the registered specs, and the
        ``bindings`` mapping are all left untouched, and resolving the same
        bindings twice yields equal results.

        A binding body is a *partial spec plus two flags*. ``enabled``
        (default ``True``) selects the pack; ``locked`` (default ``False``)
        asserts that a deployment must not turn it off — ``locked: true``
        together with ``enabled: false`` is the contradiction this whole
        mechanism exists to catch, and it raises. Every other key is a field
        override, composed **spec first, binding second** under that field's
        declared rule — so ``MERGE`` fields merge over the pack's values and
        ``CONCAT`` fields append to them. ``priority`` is meta and is simply
        replaced.

        A consequence of using the declared rule for binding overrides: a
        ``UNANIMOUS`` field can be re-asserted by a binding but not
        *changed*. That is intentional — a pack author's declared mechanism
        is not a deployment knob.

        Unknown binding keys are rejected rather than ignored. A typo'd
        ``lockd: true`` that parsed as an unknown key would silently
        disable the safety contract.

        Args:
            bindings: ``{pack_name: binding_body}``. An empty mapping
                resolves to an all-default composed spec with no applied
                packs — packs are opt-in.
            composed_name: ``name`` for the composed spec.

        Returns:
            The applied specs in fold order, the composed spec, and any
            diagnostics.

        Raises:
            PackResolutionError: On an unknown pack, an unknown or malformed
                binding key, a locked-but-disabled pack, or a ``UNANIMOUS``
                field conflict.
        """
        if not isinstance(bindings, Mapping):
            raise PackResolutionError(
                f"Bindings must be a mapping of pack name to binding body; got "
                f"{type(bindings).__name__}",
                reason="invalid_binding",
                registry=self.name,
            )

        plan = _composition_plan(self._spec_cls)
        snapshot = self.items()
        registered = dict(snapshot)
        registration_order = {name: index for index, (name, _) in enumerate(snapshot)}

        selected: list[tuple[int, int, SpecT]] = []
        binding_warnings: list[PackWarning] = []
        for pack_name, binding in bindings.items():
            spec = registered.get(pack_name)
            if spec is None:
                raise PackResolutionError(
                    f"Unknown pack '{pack_name}' in {self.name} bindings",
                    reason="unknown_pack",
                    registry=self.name,
                    pack=pack_name,
                    available=sorted(registered),
                )
            body = self._validated_binding(pack_name, binding, plan)
            if body is None:
                continue
            applied, applied_warnings = self._apply_binding(spec, body, plan)
            binding_warnings.extend(applied_warnings)
            selected.append((applied.priority, registration_order[pack_name], applied))

        selected.sort(key=lambda entry: (entry[0], entry[1]))
        ordered = [spec for _, _, spec in selected]

        warnings = binding_warnings + list(_priority_tie_warnings(selected, plan))
        composed, compose_warnings = compose_packs(
            ordered, self._spec_cls, composed_name=composed_name
        )
        warnings.extend(compose_warnings)
        return PackResolution(applied=tuple(ordered), spec=composed, warnings=tuple(warnings))

    def _validated_binding(
        self,
        pack_name: str,
        binding: Any,
        plan: _CompositionPlan,
    ) -> Mapping[str, Any] | None:
        """Validate one binding body; ``None`` means the pack is disabled.

        Raises:
            PackResolutionError: For a non-mapping body, an unknown key, a
                non-boolean flag, or the locked-but-disabled contradiction.
        """
        if not isinstance(binding, Mapping):
            raise PackResolutionError(
                f"Binding for pack '{pack_name}' must be a mapping (use '{{}}' to "
                f"enable with no overrides); got {type(binding).__name__}",
                reason="invalid_binding",
                registry=self.name,
                pack=pack_name,
            )

        unknown = sorted(set(binding) - plan.binding_keys)
        if unknown:
            raise PackResolutionError(
                f"Unknown binding key(s) {unknown} for pack '{pack_name}'; "
                f"allowed: {sorted(plan.binding_keys)}",
                reason="unknown_binding_key",
                registry=self.name,
                pack=pack_name,
                keys=unknown,
            )

        for flag in sorted(_BINDING_FLAGS):
            if flag in binding and not isinstance(binding[flag], bool):
                raise PackResolutionError(
                    f"Binding key '{flag}' for pack '{pack_name}' must be a boolean; "
                    f"got {binding[flag]!r}",
                    reason="invalid_binding",
                    registry=self.name,
                    pack=pack_name,
                    key=flag,
                )

        enabled = bool(binding.get("enabled", True))
        locked = bool(binding.get("locked", False))
        if locked and not enabled:
            raise PackResolutionError(
                f"Pack '{pack_name}' is locked but the binding disables it; a locked "
                "pack cannot be turned off",
                reason="locked_pack_disabled",
                registry=self.name,
                pack=pack_name,
            )
        return binding if enabled else None

    def _apply_binding(
        self,
        spec: SpecT,
        binding: Mapping[str, Any],
        plan: _CompositionPlan,
    ) -> tuple[SpecT, tuple[PackWarning, ...]]:
        """Return ``spec`` with the binding's overrides applied, plus warnings.

        The fold's own ``value_override`` warnings are dropped: a binding
        overriding its own pack's declared value is an intentional
        deployment act, not a diagnostic. The reverse is *not* — a rule that
        discards the binding's value (``FIRST_WINS`` already pinned by this
        pack) means an operator typed a key that did nothing, and whether it
        does nothing depends on the pack's content rather than on anything
        visible in the binding. That is reported.

        A ``UNANIMOUS`` conflict still raises (see :meth:`resolve`).
        """
        overrides = {key: value for key, value in binding.items() if key in plan.rules}
        priority = self._binding_priority(spec, binding)
        if not overrides and priority == spec.priority:
            # Rebuild anyway: returning the registered instance here and a new
            # one otherwise would make ``resolution.applied`` alias the
            # registry depending on the binding's content.
            return dataclasses.replace(spec), ()

        spec_values = {name: getattr(spec, name) for name in plan.rules}
        values, _ = _fold(
            plan,
            [
                _Contribution(spec.name, spec_values),
                # A binding names its fields, so presence is the contribution
                # — including naming one to clear it back to its default.
                _Contribution(f"{spec.name}[binding]", overrides, origin="binding"),
            ],
        )
        # ``replace`` rather than ``type(spec)(name=..., **values)``: the plan
        # covers only composable fields, so reconstructing from them alone
        # would silently reset any field it cannot see — an extra
        # ``_META_FIELDS`` descriptor, for instance.
        applied = dataclasses.replace(spec, priority=priority, **values)
        return applied, self._ignored_override_warnings(spec, overrides, values, plan)

    @staticmethod
    def _ignored_override_warnings(
        spec: SpecT,
        overrides: Mapping[str, Any],
        values: Mapping[str, Any],
        plan: _CompositionPlan,
    ) -> tuple[PackWarning, ...]:
        """Warn for each override whose value the field's rule threw away.

        Detected by outcome rather than by rule name: an override was
        discarded exactly when the resolved value is the pack's own and is
        not the one the binding asked for. ``CONCAT``/``MERGE`` results
        differ from both inputs, so they never match.
        """
        ignored = [
            key
            for key, raw in overrides.items()
            if key in values
            and values[key] == getattr(spec, key)
            and values[key] != _normalize_value(raw)
        ]
        return tuple(
            PackWarning(
                code="binding_override_ignored",
                message=(
                    f"Binding for '{spec.name}' sets '{key}' to "
                    f"{overrides[key]!r}, but the field's declared "
                    f"{_rule_label(plan.rules[key])} rule keeps the pack's "
                    f"{getattr(spec, key)!r}"
                ),
                packs=(spec.name,),
                field=key,
            )
            for key in ignored
        )

    def _binding_priority(self, spec: SpecT, binding: Mapping[str, Any]) -> int:
        """The binding's ``priority`` override, or the spec's own."""
        if "priority" not in binding:
            return spec.priority
        priority = binding["priority"]
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise PackResolutionError(
                f"Binding key 'priority' for pack '{spec.name}' must be an integer; "
                f"got {priority!r}",
                reason="invalid_binding",
                registry=self.name,
                pack=spec.name,
                key="priority",
            )
        return priority


def _participating_fields(spec: PackSpec, plan: _CompositionPlan) -> frozenset[str]:
    """The non-meta fields this spec actually contributes to the fold."""
    return frozenset(
        name for name in plan.rules if _normalize_value(getattr(spec, name)) != plan.defaults[name]
    )


def _priority_tie_warnings(
    selected: Sequence[tuple[int, int, PackSpec]],
    plan: _CompositionPlan,
) -> Iterator[PackWarning]:
    """Warn for each priority group whose members contend for a field.

    A tie resolves deterministically (registration order), but silently — so
    a tie that *decides something* is surfaced rather than left for a future
    registration-order change to alter composition invisibly.

    Restricted to packs that actually co-contribute to at least one field.
    ``priority`` defaults to ``0``, so a bare group-by-priority fires on the
    ordinary "bind two packs, don't bother with priorities" case, where no
    field is contended and the order changes nothing. A diagnostic that
    fires when nothing is wrong trains consumers to filter its code out,
    which costs them the case it exists for.
    """
    grouped: dict[int, list[PackSpec]] = {}
    for priority, _, spec in selected:
        grouped.setdefault(priority, []).append(spec)
    for priority, specs in grouped.items():
        if len(specs) < 2:
            continue
        seen: set[str] = set()
        contended: set[str] = set()
        for spec in specs:
            fields = _participating_fields(spec, plan)
            contended |= fields & seen
            seen |= fields
        if not contended:
            continue
        names = [spec.name for spec in specs]
        yield PackWarning(
            code="priority_tie",
            message=(
                f"Packs {names} share priority {priority} and both set "
                f"{sorted(contended)}; composition order falls back to "
                "registration order"
            ),
            packs=tuple(names),
        )


__all__ = [
    "UNSET",
    "CompositionRule",
    "MergeKind",
    "PackRegistry",
    "PackResolution",
    "PackResolutionError",
    "PackSpec",
    "PackWarning",
    "Reducer",
    "compose_packs",
    "merge_bindings",
]
