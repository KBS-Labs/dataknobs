"""Unified model-metadata substrate — one ``ModelProfile`` per ``(provider, model)``.

Four kinds of model-keyed fact — **capabilities, request-shape validation, token
ceilings / context windows, and pricing** — used to be hand-maintained as
scattered Python literals inside each provider, and every one went stale on each
vendor release (a single new model family could be mis-capability-detected,
wrongly rejected by ``validate_model``, and left with no token ceiling all at
once). They are not four problems: they are **one** operation — *resolve a fact
about ``(provider, model)`` from a layered, refreshable, config-overridable set
of sources* — applied M facets by N providers.

This module builds that operation once:

- :class:`ModelProfile` — one frozen record holding every model-keyed facet.
  ``None`` on a facet means "unknown"; a *present* value (including an empty
  ``frozenset()`` / ``{}``) means "authoritatively known" — the distinction that
  lets a source assert "this model has no capabilities" separately from "I don't
  classify this model."
- :class:`ModelMetadataSource` — a source of *partial* profiles (only the facets
  it knows are non-``None``). Built-ins here plus consumer-registered sources via
  :data:`model_metadata_sources`.
- :class:`LayeredModelProfileResolver` — composes an ordered list of sources and
  merges their partials **facet-by-facet, highest precedence first**
  (:func:`merge_partials`). Distinct from
  :class:`~dataknobs_common.resolver.CompositeResolver`, which is first-*record*
  wins; this is first-*facet* wins, so config can pin one facet while the live
  API supplies another and the bundled resource supplies a third.

A provider becomes a *binding*: an ordered source list + a small extractor,
after which capability / constraint / ceiling detection is a one-line profile
read. The substrate leans entirely on ``dataknobs-common`` primitives
(:class:`~dataknobs_common.registry.PluginRegistry`,
:class:`~dataknobs_common.structured_config.StructuredConfig`) — no new
dependency.

**Synchronous resolve, out-of-band refresh.** :meth:`ModelMetadataSource.resolve`
is *synchronous and I/O-free* because a provider's capability / constraint
detection runs on a synchronous, per-request (and even construction-time) path
that must never touch the event loop. A source backed by a live API keeps a
process cache and refreshes it out-of-band (a provider drives the async refresh
at its request boundary, exactly as the native Anthropic provider already does);
``resolve`` only ever reads that cache. Simple sources (bundled resource, config
override, heuristic rule) have no cache and no I/O at all.
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, fields
from typing import Any, Protocol, runtime_checkable

from dataknobs_common.registry import PluginRegistry
from dataknobs_common.structured_config import StructuredConfig

from .base import CAPABILITY_NAMES, ModelCapability

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPricing(StructuredConfig):
    """Per-model USD pricing, unifying the previously-scattered price tables.

    Every field is per-million-tokens (``per_mtok``) and ``None`` when unknown.
    No vendor serves pricing live, so this facet is always sourced from a bundled
    resource (or a config override). ``StructuredConfig`` gives ``from_dict`` /
    ``to_dict`` for free.

    Attributes:
        input_per_mtok: USD per 1M input (prompt) tokens.
        output_per_mtok: USD per 1M output (completion) tokens.
        cached_input_per_mtok: USD per 1M cached-input tokens (prompt caching),
            or ``None`` when the model has no cache tier.
        batch_input_per_mtok: USD per 1M input tokens on the batch API tier.
        batch_output_per_mtok: USD per 1M output tokens on the batch API tier.
    """

    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    cached_input_per_mtok: float | None = None
    batch_input_per_mtok: float | None = None
    batch_output_per_mtok: float | None = None


@dataclass(frozen=True)
class ModelProfile(StructuredConfig):
    """Every model-keyed facet in one home. ``None`` per facet means "unknown".

    A *partial* profile is a ``ModelProfile`` a single source produces, with only
    the facets it knows set non-``None`` (see :data:`PartialModelProfile`). The
    resolver merges partials into one profile via :func:`merge_partials`. There is
    no structural difference between a "partial" and a "full" profile — a fully
    resolved profile may still carry ``None`` facets that no source knew.

    The **uniform ``None``-is-unknown sentinel** on every facet (scalar and
    collection alike) is load-bearing for the merge (:func:`merge_partials`): a
    present empty ``frozenset()`` / ``{}`` is an *authoritative* "known none" that
    wins over a lower-precedence guess, distinct from ``None`` = "I don't know."

    Attributes:
        context_window: Input/context-window size in tokens (→
            ``ModelConstraints.max_input_tokens``), or ``None``.
        max_output_tokens: Output ``max_tokens`` ceiling (→
            ``ModelConstraints.max_tokens_ceiling``), or ``None``.
        capabilities: The model's capability set, or ``None`` when unclassified.
            An empty ``frozenset()`` authoritatively asserts "no capabilities."
        rejected_params: Sampling/generation parameter names the family rejects
            at the request boundary (e.g. the Claude 5 family rejects
            ``temperature``), or ``None``.
        param_remaps: Request-param renames the family requires (e.g. the OpenAI
            o-series ``max_tokens`` → ``max_completion_tokens``), or ``None``.
        pricing: Per-model USD pricing, or ``None``.
        available: Whether the model is available to this account/endpoint
            (``validate_model`` result), or ``None`` when unknown.
        aliases: Additional ids that resolve to this same profile, or ``None``.
    """

    context_window: int | None = None
    max_output_tokens: int | None = None
    capabilities: frozenset[ModelCapability] | None = None
    rejected_params: frozenset[str] | None = None
    param_remaps: Mapping[str, str] | None = None
    pricing: ModelPricing | None = None
    available: bool | None = None
    aliases: tuple[str, ...] | None = None


#: A *partial* profile is just a :class:`ModelProfile` a single source produces
#: (only the facets it knows are non-``None``). It is a documented alias rather
#: than a separate record so the two can never drift on the field set — the merge
#: (:func:`merge_partials`) treats every facet uniformly, and "partial" vs "full"
#: is a resolution-stage distinction, not a structural one.
PartialModelProfile = ModelProfile

#: Field names of :class:`ModelProfile`, computed once (the facet list the merge
#: and loose-dict parser iterate). Kept in sync with the record automatically.
_PROFILE_FACETS: tuple[str, ...] = tuple(f.name for f in fields(ModelProfile))

#: Canonical capability ordering for reconstructing an ordered capability *list*
#: from a resolved ``frozenset`` facet, so a provider whose ``get_capabilities``
#: historically returned an ordered list keeps byte-identical order.
CAPABILITY_ORDER: tuple[ModelCapability, ...] = (
    ModelCapability.TEXT_GENERATION,
    ModelCapability.CHAT,
    ModelCapability.EMBEDDINGS,
    ModelCapability.STREAMING,
    ModelCapability.CODE,
    ModelCapability.VISION,
    ModelCapability.FUNCTION_CALLING,
    ModelCapability.JSON_MODE,
)


def merge_partials(partials: Iterable[ModelProfile]) -> ModelProfile:
    """Merge source partials into one profile, **per facet, first non-``None`` wins**.

    The crux of the substrate (design decision D-MERGE): *override, not union*.
    ``partials`` is consumed in **precedence order, highest first** — for each
    facet the first partial with a non-``None`` value wins, and no later
    (lower-precedence) partial can displace it. ``None`` = "unknown" (skipped); a
    present value — *including* an empty ``frozenset()`` / ``{}`` — is
    "authoritatively known" and wins. So a config pinning
    ``capabilities=frozenset()`` ("this model has none") replaces a lower layer's
    guess, matching config-always-wins intent.

    Uniform across scalar and collection facets; there is no per-facet special
    case. (If a consumer ever needs capability *union* across layers, that is a
    future per-facet merge policy — this ships override-wins.)

    Args:
        partials: Source partials in descending precedence order.

    Returns:
        A single merged :class:`ModelProfile` (facets no source knew stay
        ``None``).
    """
    merged: dict[str, Any] = dict.fromkeys(_PROFILE_FACETS)
    for partial in partials:
        for facet in _PROFILE_FACETS:
            if merged[facet] is None:
                value = getattr(partial, facet)
                if value is not None:
                    merged[facet] = value
    return ModelProfile(**merged)


# ---------------------------------------------------------------------------
# Loose-dict parsing (shared by the config-override and bundled-resource sources)
# ---------------------------------------------------------------------------


def profile_from_loose(data: Mapping[str, Any]) -> ModelProfile:
    """Parse a loose config/resource mapping into a :class:`ModelProfile` partial.

    Shared by :class:`ConfigOverrideSource` (reads ``LLMConfig.model_profile_overrides``)
    and :class:`BundledResourceSource` (reads a per-model resource entry) so the
    two can never drift on how a loose facet is coerced. Every key is optional; an
    absent key leaves that facet ``None`` (unknown). Unknown keys are ignored.

    Coercions:

    - ``capabilities`` — a list of :class:`ModelCapability` value strings
      (``"vision"``, ``"function_calling"``) → ``frozenset[ModelCapability]``.
      Unknown names are dropped with a warning. An empty list → an authoritative
      empty ``frozenset()``.
    - ``rejected_params`` / ``aliases`` — lists of strings → ``frozenset`` /
      ``tuple``.
    - ``param_remaps`` — a ``{from: to}`` string mapping.
    - ``pricing`` — a nested mapping → :class:`ModelPricing` (via its
      ``from_dict``); an existing ``ModelPricing`` passes through.
    - ``context_window`` / ``max_output_tokens`` — ``int`` (``None``-tolerant).
    - ``available`` — ``bool``.
    """
    parsed: dict[str, Any] = {}

    caps = data.get("capabilities")
    if caps is not None:
        resolved: list[ModelCapability] = []
        for name in caps:
            cap = CAPABILITY_NAMES.get(str(name))
            if cap is not None:
                resolved.append(cap)
            else:
                logger.warning("Unknown capability name in model profile: %s", name)
        parsed["capabilities"] = frozenset(resolved)

    rejected = data.get("rejected_params")
    if rejected is not None:
        parsed["rejected_params"] = frozenset(str(p) for p in rejected)

    remaps = data.get("param_remaps")
    if remaps is not None:
        parsed["param_remaps"] = {str(k): str(v) for k, v in dict(remaps).items()}

    aliases = data.get("aliases")
    if aliases is not None:
        parsed["aliases"] = tuple(str(a) for a in aliases)

    pricing = data.get("pricing")
    if pricing is not None:
        parsed["pricing"] = (
            pricing
            if isinstance(pricing, ModelPricing)
            else ModelPricing.from_dict(dict(pricing))
        )

    for facet in ("context_window", "max_output_tokens"):
        value = data.get(facet)
        if value is not None:
            parsed[facet] = int(value)

    if data.get("available") is not None:
        parsed["available"] = bool(data["available"])

    return ModelProfile(**parsed)


def match_family_key(model_lower: str, keys: Iterable[str]) -> str | None:
    """Resolve *model_lower* to the best-matching key in *keys*, or ``None``.

    The family-alias rule shared with the token-ceiling matcher
    (:func:`~dataknobs_llm.llm.providers._claude_shared.match_ceiling`), returning
    the matched *key* (so the caller fetches that key's profile) rather than a
    value. In precedence order:

    1. **Exact** id match (returns immediately).
    2. **Family-alias**: the longest key that is a substring of the request — a
       short family key (``claude-sonnet-5``) covers a longer dated request.
    3. **Bare-alias**: the longest key of which the request is a substring — a
       bare request resolves against a longer dated key.

    Among substring matches the longest overlap wins; ``None`` when nothing
    matches. For two distinct strings only one substring direction can hold, so
    cases 2 and 3 never both fire for one key.
    """
    best: str | None = None
    best_len = -1
    for key in keys:
        if key == model_lower:
            return key
        if key in model_lower:
            if len(key) > best_len:
                best, best_len = key, len(key)
        elif model_lower in key and len(model_lower) > best_len:
            best, best_len = key, len(model_lower)
    return best


# ---------------------------------------------------------------------------
# The source Protocol + registry
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelMetadataSource(Protocol):
    """A source of partial :class:`ModelProfile` records for a provider's models.

    A source contributes the facets it knows and leaves the rest ``None``; the
    resolver merges an ordered list of sources (:class:`LayeredModelProfileResolver`).

    :meth:`resolve` is **synchronous and I/O-free** — it runs on the provider's
    per-request (and construction-time) detect path, which must never touch the
    event loop. A source backed by a live API keeps a process cache refreshed
    out-of-band and only *reads* it here (see the module docstring); simple
    sources do no I/O at all.
    """

    @property
    def name(self) -> str:
        """Stable identifier for logging / debugging (e.g. ``"bundled_resource"``)."""
        ...

    def resolve(self, model: str) -> ModelProfile:
        """Return the facets this source knows for *model* (rest ``None``)."""
        ...


#: Consumer-extensible registry of named :class:`ModelMetadataSource` factories.
#: An in-house gateway / vLLM / proxy registers a source
#: (``model_metadata_sources.register("my_gateway", factory)``) without a
#: dataknobs release; the provider composes it into its resolver. Mirrors
#: ``intent_classifier_backends`` / ``event_bus_backends``.
model_metadata_sources: PluginRegistry[ModelMetadataSource] = PluginRegistry(
    name="model_metadata_sources",
    validate_type=ModelMetadataSource,
    not_found_kind="model metadata source",
    not_found_exception=ValueError,
)


# ---------------------------------------------------------------------------
# Built-in sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallableModelMetadataSource:
    """A source wrapping a synchronous ``(model) -> ModelProfile`` callable.

    The universal adapter for a source whose logic is a small function: a
    family-substring *heuristic* (capabilities / rejected params), a thin reader
    over an existing cache or resource, or any consumer rule. The callable must be
    synchronous and I/O-free (see :class:`ModelMetadataSource`).
    """

    name: str
    fn: Any  # Callable[[str], ModelProfile]

    def resolve(self, model: str) -> ModelProfile:
        return self.fn(model)


class ConfigOverrideSource:
    """Highest-precedence source reading ``LLMConfig.model_profile_overrides``.

    Extends the config-override philosophy (a consumer declares a fact without a
    dataknobs release) from request-shape constraints to *every* profile facet.
    The override is a loose mapping parsed by :func:`profile_from_loose`; absent →
    an all-``None`` partial (no effect). Model-keyed: the override may be a single
    flat mapping (applies to the configured model) or a ``{model_id: {...}}``
    mapping (per-model), matched by :func:`match_family_key`.
    """

    name = "config_override"

    def __init__(self, overrides: Mapping[str, Any] | None) -> None:
        self._overrides = overrides or {}

    def resolve(self, model: str) -> ModelProfile:
        if not self._overrides:
            return ModelProfile()
        # Per-model mapping ({model_id: {facets}}) vs a flat single-model mapping.
        # Heuristic: a value that is itself a mapping keyed by a facet name is a
        # flat override; otherwise treat top-level keys as model ids.
        if any(k in _PROFILE_FACETS for k in self._overrides):
            return profile_from_loose(self._overrides)
        key = match_family_key(model.lower(), (str(k).lower() for k in self._overrides))
        if key is None:
            return ModelProfile()
        # Recover the original-cased key whose lowercase matched.
        for original in self._overrides:
            if str(original).lower() == key:
                return profile_from_loose(self._overrides[original])
        return ModelProfile()


class BundledResourceSource:
    """A source reading a bundled per-provider profile resource (YAML/JSON).

    The maintained-fallback layer: a ``{models: {id: {facets}}}`` resource shipped
    in the package, read **once at construction** (off the event loop) into a
    ``{lowercased-id: ModelProfile}`` map and resolved by
    :func:`match_family_key`. Each per-model entry is parsed by
    :func:`profile_from_loose`, so a resource can carry any facet
    (capabilities / ceilings / pricing / rejected_params). ``None`` when a model
    is unknown → the resolver falls through to a lower-precedence source.

    A malformed / missing resource degrades to an empty map (logged) so a data
    issue never breaks import; a packaging regression is caught by a resource
    test instead.
    """

    def __init__(
        self,
        profiles: Mapping[str, ModelProfile],
        *,
        name: str = "bundled_resource",
    ) -> None:
        self.name = name
        self._profiles = {k.lower(): v for k, v in profiles.items()}

    @classmethod
    def from_resource(
        cls,
        package: str,
        resource: str,
        *,
        name: str = "bundled_resource",
    ) -> BundledResourceSource:
        """Build from a packaged ``{models: {id: {facets}}}`` YAML/JSON resource.

        Reads via :mod:`importlib.resources` (so the disk read is at construction,
        never on the event loop). Degrades to an empty source on any error.
        """
        import importlib.resources

        from dataknobs_common.config_loading import load_yaml_or_json

        profiles: dict[str, ModelProfile] = {}
        try:
            ref = importlib.resources.files(package) / resource
            with importlib.resources.as_file(ref) as path:
                data = load_yaml_or_json(path, require_dict=True)
            section = data.get("models") or {}
            for model_id, entry in section.items():
                profiles[str(model_id).lower()] = profile_from_loose(
                    entry if isinstance(entry, Mapping) else {}
                )
        except Exception:  # pragma: no cover - guarded by a resource test
            logger.exception(
                "Failed to load bundled model-profile resource %s/%s; "
                "profiles will resolve from other sources",
                package,
                resource,
            )
        return cls(profiles, name=name)

    def resolve(self, model: str) -> ModelProfile:
        key = match_family_key(model.lower(), self._profiles.keys())
        if key is None:
            return ModelProfile()
        return self._profiles[key]


def _default_model_id(model_obj: Any) -> str | None:
    """Read a live-API model object's id (the ``getattr(obj, "id")`` default).

    The default ``model_id`` extractor for :class:`LiveApiSource`. A vendor
    whose model objects key their id under a different attribute passes its own
    ``model_id`` callable.
    """
    value = getattr(model_obj, "id", None)
    return str(value) if value is not None else None


def _has_known_facet(profile: ModelProfile) -> bool:
    """Whether *profile* carries at least one non-``None`` facet.

    The generic form of the "cache the entry only when the model reports a
    ceiling" gate: a live model object the extractor projects to an all-``None``
    partial contributes nothing, so it is not cached (a lower-precedence source
    supplies the fallback).
    """
    return any(getattr(profile, facet) is not None for facet in _PROFILE_FACETS)


@dataclass
class _LiveEntry:
    """One cached per-model partial profile fetched from a live vendor API.

    ``source`` tags provenance (``"dynamic"`` — from the live API): the cache
    holds only live-sourced entries, so the **non-degradation** guarantee (a
    transient refresh failure never drops a known-good live value) is achieved by
    leaving the cache untouched on failure — a lower-precedence source (the
    bundled resource) supplies the fallback separately, never overwriting a live
    entry.
    ``fetched_at`` is a :func:`time.monotonic` stamp (provenance/debugging; the
    refresh cadence is gated per-loop, not per entry).
    """

    profile: ModelProfile
    source: str  # "dynamic"
    fetched_at: float


class LiveApiSource:
    """A :class:`ModelMetadataSource` backed by a live vendor Models API, cached.

    Generalizes the Anthropic live Models-API ceiling cache into a reusable
    source any provider serving live model metadata can compose. It wraps:

    - ``list_models`` — an async ``() -> Iterable[api_object]`` collecting every
      model the vendor lists (the provider already owns one — e.g. Anthropic's
      auto-paged ``client.models.list()`` walker).
    - ``extractor`` — a synchronous ``(api_object) -> ModelProfile`` projecting
      one vendor model object into the facets it reports (a *partial* profile;
      the facets the API does not serve stay ``None``).

    **Synchronous resolve, out-of-band refresh** (the substrate contract):
    :meth:`resolve` is a pure cache read (no I/O), safe on the provider's
    per-request / construction-time detect path; the provider drives
    :meth:`refresh_if_stale` / :meth:`force_refresh` from its async request
    boundary to keep the cache current.

    **The refresh carries three properties (lifted from the Anthropic cache):**

    - **TTL-gated** — a fresh cache is a no-op (no I/O); a refresh fires at most
      once per ``ttl`` per event loop, never per request (:meth:`is_stale`).
    - **Per-loop-locked** — concurrent callers on a cold/stale cache coalesce
      into a single ``list_models()`` (the double-check after the lock returns
      the losers early). Locks + last-fetch timestamps are keyed on the loop
      *object* via a :class:`weakref.WeakKeyDictionary`, so a collected loop's
      state is evicted (no leak) and a fresh loop is always a distinct key
      (closing the ``id(loop)``-reuse mis-skip hole where a new loop could
      inherit a dead loop's stale timestamp).
    - **Source-aware non-degradation** — the last-fetch timer is re-armed
      *before* the poll (success or failure), so a Models-API outage cannot
      busy-retry (bounded to one attempt per TTL); on failure the cache is left
      intact, so a known-good live value is never dropped back to the bundled
      fallback (which a lower-precedence source supplies). A model absent from
      the cache resolves via that fallback until a poll returns it, at which
      point its ``dynamic`` entry outranks the fallback in the layered resolver.

    The cache is **per-instance**: each provider owns its live source (D-KEY —
    provider identity is structural), so two providers on distinct accounts keep
    isolated caches rather than sharing one keyed only by model id.
    """

    def __init__(
        self,
        list_models: Callable[[], Awaitable[Iterable[Any]]],
        extractor: Callable[[Any], ModelProfile],
        *,
        name: str = "live_api",
        ttl: float = 3600.0,
        refresh_timeout: float = 10.0,
        enabled: bool = True,
        model_id: Callable[[Any], str | None] = _default_model_id,
    ) -> None:
        """Build a live source.

        Args:
            list_models: Async ``() -> Iterable[api_object]`` collecting every
                model the vendor lists.
            extractor: Sync ``(api_object) -> ModelProfile`` projecting one
                model object into the facets it reports.
            name: Stable identifier for logging / debugging.
            ttl: Seconds between Models-API refreshes per loop. A fresh cache is
                a no-op; ``0`` re-polls each stale check (maximal freshness).
            refresh_timeout: Hard timeout (seconds) on a single poll — the lock
                is held across it, so a *hung* control-plane is bounded here
                rather than stalling every cold-cache caller.
            enabled: When ``False``, both refresh entries are no-ops (the source
                resolves from an empty cache — resource-only via the resolver).
            model_id: ``(api_object) -> str | None`` reading a model object's id
                (default: ``getattr(obj, "id")``).
        """
        self.name = name
        self._list_models = list_models
        self._extractor = extractor
        self._ttl = ttl
        self._refresh_timeout = refresh_timeout
        self._enabled = enabled
        self._model_id = model_id
        self._cache: dict[str, _LiveEntry] = {}
        self._last_fetch: weakref.WeakKeyDictionary[Any, float] = (
            weakref.WeakKeyDictionary()
        )
        self._locks: weakref.WeakKeyDictionary[Any, asyncio.Lock] = (
            weakref.WeakKeyDictionary()
        )

    # -- read path (synchronous, I/O-free) --------------------------------

    def resolve(self, model: str) -> ModelProfile:
        """Return the facets the live cache knows for *model* (rest ``None``).

        Resolved **per facet** by the shared family-alias matcher
        (:func:`match_family_key`): for each facet, the best-matching cache key
        *among entries that know that facet* wins, so a bare-alias request
        (``claude-sonnet-5``) resolves against a dated cache key
        (``claude-sonnet-5-<snapshot>``) fetched from the API — independently per
        facet, so a model reporting only its input window contributes its input
        facet without fabricating an output value. Pure cache read (no I/O).
        """
        if not self._cache:
            return ModelProfile()
        key = model.lower()
        merged: dict[str, Any] = {}
        for facet in _PROFILE_FACETS:
            known = {
                cache_key: value
                for cache_key, entry in self._cache.items()
                if (value := getattr(entry.profile, facet)) is not None
            }
            if not known:
                continue
            matched = match_family_key(key, known.keys())
            if matched is not None:
                merged[facet] = known[matched]
        return ModelProfile(**merged)

    # -- refresh (async, out-of-band) -------------------------------------

    def is_stale(self) -> bool:
        """Whether the running loop is due for a refresh (per ``ttl``).

        Must be called from within a running event loop (the refresh state is
        keyed on the loop object); raises :class:`RuntimeError` otherwise. The
        refresh entries (:meth:`refresh_if_stale` / :meth:`force_refresh`) are
        the intended drivers — this is exposed for callers already on the loop
        that want to gate their own work on freshness.
        """
        last = self._last_fetch.get(asyncio.get_running_loop())
        if last is None:
            return True
        return (time.monotonic() - last) >= self._ttl

    async def refresh_if_stale(self) -> None:
        """Refresh the cache if this loop's TTL expired (the hot-path entry).

        A no-op (no lock, no I/O) when disabled or the cache is fresh, so it is
        cheap to call on every request; a stale/cold cache coalesces concurrent
        callers into one poll under the per-loop lock. Best-effort — never raises.
        """
        if not self._enabled or not self.is_stale():
            return
        await self._locked_refresh(force=False)

    async def force_refresh(self) -> None:
        """Poll now, bypassing the TTL gate (honors the disabled switch).

        The public force-refresh entry for a consumer driving freshness on its
        own schedule instead of relying on the TTL. Best-effort — never raises.
        """
        if not self._enabled:
            return
        await self._locked_refresh(force=True)

    async def _locked_refresh(self, *, force: bool) -> None:
        """Serialize the poll per loop; re-arm the timer before the call.

        The double-check after acquiring the lock returns the losers of a
        cold-cache race early (one shared poll). The timer is re-armed *before*
        the API call — success or failure — so an outage is bounded to one
        attempt per TTL and never busy-retries; on any error the cache is left
        intact (non-degradation) and the request proceeds on the cached/fallback
        value.
        """
        async with self._lock():
            if not force and not self.is_stale():
                return
            self._last_fetch[asyncio.get_running_loop()] = time.monotonic()
            try:
                # Bound the poll independently of the client's request timeout:
                # the lock is held across it, so a *hung* control-plane would
                # otherwise stall every cold-cache caller.
                models = await asyncio.wait_for(
                    self._list_models(), timeout=self._refresh_timeout
                )
            except Exception as exc:  # never fatal — serve cached/fallback value
                logger.debug(
                    "%s live model refresh failed, using fallback: %s",
                    self.name,
                    exc,
                )
                return
            now = time.monotonic()
            for model_obj in models:
                profile = self._extractor(model_obj)
                model_id = self._model_id(model_obj)
                if model_id and _has_known_facet(profile):
                    self._cache[model_id.lower()] = _LiveEntry(
                        profile=profile, source="dynamic", fetched_at=now,
                    )

    def _lock(self) -> asyncio.Lock:
        """Return (lazily creating) the refresh lock for the running loop.

        Lazy creation is race-free within a loop: no ``await`` separates the
        ``get`` from the assignment, so a single loop cannot interleave two
        creations, and distinct loops key on distinct loop objects.
        """
        loop = asyncio.get_running_loop()
        lock = self._locks.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[loop] = lock
        return lock

    # -- seeding / lifecycle helpers --------------------------------------

    def seed(
        self, model_id: str, profile: ModelProfile, *, source: str = "dynamic"
    ) -> None:
        """Seed a cache entry directly (manual-priming / test helper)."""
        self._cache[model_id.lower()] = _LiveEntry(
            profile=profile, source=source, fetched_at=time.monotonic()
        )

    def clear(self) -> None:
        """Drop all cached entries + per-loop refresh state."""
        self._cache.clear()
        self._last_fetch.clear()
        self._locks.clear()


# ---------------------------------------------------------------------------
# The layered resolver
# ---------------------------------------------------------------------------


class LayeredModelProfileResolver:
    """Resolve a :class:`ModelProfile` by merging sources **facet-by-facet**.

    Composes an ordered ``list[ModelMetadataSource]`` (highest precedence first)
    and merges each source's partial via :func:`merge_partials` (design decision
    D-MERGE — first non-``None`` per facet wins). Distinct from
    :class:`~dataknobs_common.resolver.CompositeResolver`, which is first-*record*
    wins; this is first-*facet* wins, so config can pin one facet while the live
    API supplies another and the bundled resource a third.

    The resolver is **per-provider** (design decision D-KEY): each provider
    composes its own ordered source list and keys by ``model: str``. Provider
    identity is structural — you ask *a provider's* resolver — mirroring how each
    provider already owns its own live cache; there is no global god-resolver.

    :meth:`resolve` is synchronous and I/O-free (every source's ``resolve`` is).
    A live-backed source is refreshed out-of-band by the provider before the
    detect path reads it.
    """

    def __init__(self, sources: Iterable[ModelMetadataSource]) -> None:
        self._sources: tuple[ModelMetadataSource, ...] = tuple(sources)

    @property
    def sources(self) -> tuple[ModelMetadataSource, ...]:
        return self._sources

    def resolve(self, model: str) -> ModelProfile:
        """Merge every source's partial for *model*, highest precedence first."""
        return merge_partials(source.resolve(model) for source in self._sources)


__all__ = [
    "CAPABILITY_ORDER",
    "BundledResourceSource",
    "CallableModelMetadataSource",
    "ConfigOverrideSource",
    "LayeredModelProfileResolver",
    "LiveApiSource",
    "ModelMetadataSource",
    "ModelPricing",
    "ModelProfile",
    "PartialModelProfile",
    "match_family_key",
    "merge_partials",
    "model_metadata_sources",
    "profile_from_loose",
]
