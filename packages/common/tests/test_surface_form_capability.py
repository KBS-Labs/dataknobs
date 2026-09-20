# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The capability a rung reads, asked before the rung can ask for it.

``by_surface_form`` became a **partial** member when a source over a live
table arrived: a vocabulary whose forms were folded when they were written
answers it, and one reading a column holding the form as it was typed cannot.
The protocol says so, and says a source that cannot fold withholds
:attr:`~dataknobs_common.capabilities.Capability.SURFACE_FORM_LOOKUP` and
raises when asked anyway.

That half shipped with a writer and no reader. Three of the four rungs here
call the member unguarded, so a cascade composed over such a source learned at
**query time** that a rung could not run -- and a document declaring no
``resolver:`` at all is the worst case rather than the safe one, because
silence is the default composition and two of its three rungs read the member.

The refusal is at **rung construction**, which is where
:class:`~dataknobs_common.entity_resolution.LexicalSignal` already refuses a
source that cannot hand over its forms. The argument is that one's argument:
an empty answer is what a working rung returns when the query matched nothing,
so a misconfigured source must not be able to produce one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common.capabilities import (
    Capability,
    CapabilityContract,
    CapabilityNotSupportedError,
    require_capability,
    supports_capability,
)
from dataknobs_common.entity_resolution import (
    AliasSignal,
    AsyncAliasSignal,
    AsyncExactNormalizedSignal,
    AsyncLexicalSignal,
    AsyncScanningSignal,
    ExactNormalizedSignal,
    LexicalSignal,
    ScanningSignal,
    async_signal_backends,
    declared_signal_metadata,
    signal_backends,
)
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import (
    AsyncMappingEntitySource,
    Entity,
    EntitySource,
    MappingEntitySource,
    async_load_ontology,
    load_ontology,
)
from dataknobs_common.ontology.loader import async_build_resolver, build_resolver

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from dataknobs_common.ontology import SourceDescription, SourceRef
    from dataknobs_common.records import Record

VOCABULARY = {"dog": Entity(id="dog", type="Species", name="Dog", aliases=["hound"])}


class UnfoldedSource:
    """A source over forms held as they were written, which cannot fold.

    A real implementation of the protocol rather than a mock, and the only
    way to ask this question inside this package: every source that ships
    here folds, and the one that cannot -- a projection over a live table --
    lives in ``dataknobs-data``, which depends on this package rather than
    the other way about. What it stands in for is that class over a binding
    declaring no folded lookup, which is two lines: withhold the capability,
    raise when asked anyway.

    It **does** publish :meth:`surface_forms`, so a near-spelling rung's
    existing refusal is not what fires here. Without that the lexical case
    would pass against the guard that was already there.
    """

    def __init__(self, entities: dict[str, Entity] | None = None) -> None:
        self._inner = MappingEntitySource(VOCABULARY if entities is None else entities)

    def get(self, entity_id: str) -> Entity | None:
        return self._inner.get(entity_id)

    def get_many(self, entity_ids: Sequence[str]) -> dict[str, Entity]:
        return self._inner.get_many(entity_ids)

    def fetch_origin(self, ref: SourceRef) -> Record | None:
        return self._inner.fetch_origin(ref)

    def fetch_origins(self, refs: Sequence[SourceRef]) -> list[Record | None]:
        return self._inner.fetch_origins(refs)

    def describe(self) -> SourceDescription:
        described = self._inner.describe()
        return type(described)(
            source_id=described.source_id,
            backend=described.backend,
            table=described.table,
            projection=described.projection,
            capabilities=frozenset(
                capability
                for capability in described.capabilities
                if capability is not Capability.SURFACE_FORM_LOOKUP
            ),
            declares=described.declares,
        )

    def by_surface_form(self, form: str) -> frozenset[str]:
        raise CapabilityNotSupportedError(Capability.SURFACE_FORM_LOOKUP, self)

    def by_type(self, type_id: str) -> frozenset[str]:
        return self._inner.by_type(type_id)

    def surface_forms(self) -> Iterable[str]:
        return ("dog", "hound")

    def longest_form_tokens(self) -> int | None:
        return None


class AsyncUnfoldedSource(UnfoldedSource):
    """:class:`UnfoldedSource`' twin, refusing at the same point."""

    async def get(self, entity_id: str) -> Entity | None:  # type: ignore[override]
        return self._inner.get(entity_id)

    async def get_many(  # type: ignore[override]
        self, entity_ids: Sequence[str]
    ) -> dict[str, Entity]:
        return self._inner.get_many(entity_ids)

    async def fetch_origin(self, ref: SourceRef) -> Record | None:  # type: ignore[override]
        return None

    async def fetch_origins(self, refs: Sequence[SourceRef]) -> list[Record | None]:
        return [None] * len(refs)

    async def by_surface_form(self, form: str) -> frozenset[str]:  # type: ignore[override]
        raise CapabilityNotSupportedError(Capability.SURFACE_FORM_LOOKUP, self)

    async def by_type(self, type_id: str) -> frozenset[str]:  # type: ignore[override]
        return self._inner.by_type(type_id)

    async def surface_forms(self) -> Iterable[str]:  # type: ignore[override]
        return ("dog", "hound")


def _swap(ontology: Any, entities: Any) -> Any:
    """The same vocabulary, read through another source."""
    return type(ontology)(
        id=ontology.id,
        version=ontology.version,
        entity_types=ontology.entity_types,
        relation_types=ontology.relation_types,
        entities=entities,
        assertions=ontology.assertions,
        taxonomies=ontology.taxonomies,
        describes=ontology.describes,
        codec=ontology.codec,
    )


# --------------------------------------------------------------------------
# The source that folds says so
# --------------------------------------------------------------------------


def test_an_authored_vocabulary_declares_the_capability_it_answers() -> None:
    """The index folds ids, names and aliases -- that *is* the capability.

    ``describe().capabilities`` is what the contract instructs a consumer to
    guard with, so a source answering the member while reporting that it
    cannot sends every guarded caller down a fallback path -- and the most
    common source in the tree was reporting exactly that.
    """
    source = MappingEntitySource(VOCABULARY)

    assert Capability.SURFACE_FORM_LOOKUP in source.describe().capabilities
    assert source.by_surface_form("Hound") == frozenset({"dog"})


async def test_the_async_authored_vocabulary_declares_it_too() -> None:
    """The twin, because a capability that differs by flavour is a bug in one of them."""
    source = AsyncMappingEntitySource(VOCABULARY)

    assert Capability.SURFACE_FORM_LOOKUP in source.describe().capabilities
    assert await source.by_surface_form("Hound") == frozenset({"dog"})


def test_the_double_conforms_to_the_protocol_it_stands_in_for() -> None:
    """A double that has stopped conforming reports nothing on its own."""
    assert isinstance(UnfoldedSource(), EntitySource)


@pytest.mark.parametrize("source_cls", [MappingEntitySource, AsyncMappingEntitySource])
def test_the_authored_twins_answer_the_guard_their_description_declares(
    source_cls: type[Any],
) -> None:
    """``describe()`` said yes and the guard the contract names said no.

    :func:`~dataknobs_common.capabilities.require_capability` is the pre-call
    guard this subsystem tells a consumer to use, and it duck-types on
    ``supports`` -- which neither authored twin had, so an object absent that
    member answers ``False`` for every capability including the one it is
    built around. A consumer following the documented pattern was refused by
    the source that always folds.
    """
    source = source_cls(VOCABULARY)

    assert isinstance(source, CapabilityContract)
    assert source_cls.supported_capabilities() == frozenset({Capability.SURFACE_FORM_LOOKUP})
    assert source.instance_capabilities() == source.describe().capabilities
    assert supports_capability(source, Capability.SURFACE_FORM_LOOKUP)
    require_capability(source, Capability.SURFACE_FORM_LOOKUP)
    assert not supports_capability(source, Capability.ORIGIN_FETCH)


# --------------------------------------------------------------------------
# Every rung that reads the member refuses the source that withholds it
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rung", [ExactNormalizedSignal, ScanningSignal, LexicalSignal], ids=lambda cls: cls.key
)
def test_a_rung_reading_surface_forms_refuses_a_source_that_withholds_them(
    rung: type[Any],
) -> None:
    """At construction, on the near-spelling rung's precedent and for its reason."""
    with pytest.raises(ValidationError) as raised:
        rung(UnfoldedSource())

    message = str(raised.value)
    assert "surface_form_lookup" in message
    assert "UnfoldedSource" in message


@pytest.mark.parametrize(
    "rung",
    [AsyncExactNormalizedSignal, AsyncScanningSignal, AsyncLexicalSignal],
    ids=lambda cls: cls.key,
)
def test_the_async_rungs_refuse_it_identically(rung: type[Any]) -> None:
    """The twins, refusing at the same point with the same message."""
    with pytest.raises(ValidationError) as raised:
        rung(AsyncUnfoldedSource())

    assert "surface_form_lookup" in str(raised.value)


@pytest.mark.parametrize("rung", [AliasSignal, AsyncAliasSignal], ids=lambda cls: cls.__name__)
def test_the_alias_rung_does_not_refuse_it(rung: type[Any]) -> None:
    """The other direction, without which the refusal could be "refuse everything".

    ``alias`` reads ``by_alias_form``, never ``by_surface_form``, so this
    source's withholding is none of its business. A rung built here and left
    unbuilt there is what makes the set a *derived* answer rather than a
    second list somebody keeps in step.
    """
    source = AsyncUnfoldedSource() if rung is AsyncAliasSignal else UnfoldedSource()

    assert rung(source) is not None


@pytest.mark.parametrize(
    ("registry", "rungs"),
    [
        (signal_backends, {"exact", "scan", "lexical"}),
        (async_signal_backends, {"exact", "scan", "lexical"}),
    ],
    ids=["sync", "async"],
)
def test_each_rung_kind_declares_whether_it_reads_surface_forms(
    registry: Any, rungs: set[str]
) -> None:
    """The fact a *document* can be refused against, before anything is built.

    A registry is the only place a caller holding a composition and no
    instances can ask, which is what the loader that binds a live source
    needs -- so the fact is declared per kind rather than discovered per
    instance. Asserted in both directions so a kind added without it is
    caught here rather than by the consumer it fails to refuse for.
    """
    reading = {
        key for key in registry.list_keys() if registry.get_metadata(key).get("reads_surface_forms")
    }

    assert reading == rungs
    assert registry.get_metadata("alias").get("reads_surface_forms") is False


# --------------------------------------------------------------------------
# The composition nobody wrote, which is the common one
# --------------------------------------------------------------------------


def test_the_default_composition_over_a_source_that_cannot_fold_is_refused(
    mammals_path: Path,
) -> None:
    """The case no inspection of a document can see.

    A document with no ``resolver:`` section has said nothing, and silence is
    the *default* composition -- ``exact``, ``alias``, ``scan`` -- two of
    whose three rungs read surface forms. A guard reading the document sees
    no rungs and passes it; the door that builds them is holding both the
    composition and the source, and is the one place the question can be put.
    """
    ontology = _swap(load_ontology(mammals_path), UnfoldedSource())

    with pytest.raises(ValidationError) as raised:
        build_resolver(mammals_path, ontology)

    assert "surface_form_lookup" in str(raised.value)


async def test_the_async_default_composition_is_refused_the_same_way(
    mammals_path: Path,
) -> None:
    """The twin, and the door a live binding actually reaches."""
    ontology = _swap(await async_load_ontology(mammals_path), AsyncUnfoldedSource())

    with pytest.raises(ValidationError) as raised:
        await async_build_resolver(mammals_path, ontology)

    assert "surface_form_lookup" in str(raised.value)


def test_the_default_composition_still_builds_over_a_source_that_folds(
    mammals_path: Path,
) -> None:
    """The refusal reaches the source that cannot fold and no further."""
    resolver = build_resolver(mammals_path, load_ontology(mammals_path))

    resolved = resolver.resolve("beagles")

    assert [candidate.entity_id for candidate in resolved.candidates] == ["beagle"]


# --------------------------------------------------------------------------
# The extension point, where the fact is restated or derived
# --------------------------------------------------------------------------


class _BareProtocolRung:
    """A rung written against the protocol, declaring neither fact.

    Not a stand-in for anything: ``AuthoritySignal`` in ``dataknobs-xization``
    is this shape for a stated reason -- its backing is an authority stack
    rather than a dictionary, so it subclasses no base that would supply these
    attributes. A helper that reached them by attribute access would be
    unusable by the one registration in the tree that is not written here.
    """

    key = "bare"


class _ConsumerRung:
    """A rung that declares, without inheriting the base that usually does."""

    key = "consumer"
    reads_surface_forms = True
    bounded_by_longest_form = True


def test_a_rung_that_declares_nothing_is_read_as_declaring_nothing() -> None:
    """The default is the reading that refuses nothing.

    A rung with no opinion must not be refused a lookup it never reads, nor a
    bound it never spends -- so the absent attribute answers ``False`` rather
    than raising, which is what lets a bare-protocol rung use this at all.
    """
    metadata = declared_signal_metadata(_BareProtocolRung, {"flavour": "sync"})

    assert metadata == {
        "flavour": "sync",
        "reads_surface_forms": False,
        "bounded_by_longest_form": False,
    }


def test_a_consumers_rung_declares_on_the_class_and_the_registry_reports_it() -> None:
    """The asymmetry this function is published to close.

    A consumer hand-writing the keys can write them wrongly, and the cost is
    silent: the class-level guard still fires when the rung is constructed,
    while every *load-time* refusal reading this registry passes them over.
    Derived from the class, the two cannot disagree.
    """
    metadata = declared_signal_metadata(_ConsumerRung, {"flavour": "sync", "needs_io": True})

    assert metadata["reads_surface_forms"] is True
    assert metadata["bounded_by_longest_form"] is True
    assert metadata["needs_io"] is True


def test_the_base_is_copied_so_one_base_serves_both_flavours() -> None:
    """Both flavours are registered from one base, which must survive the first."""
    base = {"flavour": "sync", "needs_io": False}

    declared_signal_metadata(_ConsumerRung, base)

    assert base == {"flavour": "sync", "needs_io": False}


@pytest.mark.parametrize(
    ("registry", "flavour"),
    [(signal_backends, "sync"), (async_signal_backends, "async")],
    ids=["sync", "async"],
)
def test_a_rung_registered_from_another_distribution_declares_both_facts(
    registry: Any, flavour: str
) -> None:
    """``authority`` ships in ``dataknobs-xization`` and is the live case.

    It is the one registration in the tree written outside the module that
    holds the four built-ins, so it is the one that had to restate these keys
    by hand -- and it did not, which is what a consumer's registration would
    also do. The keys being *present* is the assertion; their values are the
    rung's own answer and are ``False`` because an authority stack holds no
    folded form table.
    """
    pytest.importorskip("dataknobs_xization.entity_resolution")

    metadata = registry.get_metadata("authority")

    assert metadata["flavour"] == flavour
    assert metadata["reads_surface_forms"] is False
    assert metadata["bounded_by_longest_form"] is False
