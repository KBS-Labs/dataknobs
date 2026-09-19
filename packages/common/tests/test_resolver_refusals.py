"""Refusing a rung this flavour cannot build, before anything is built.

The refusal is computed from the declared **kind**, which is what lets it hold
with no rung constructed. That property is the whole criterion, and it is not
observable from the message -- so the assertions below reach past the message
to what was and was not reached.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from dataknobs_common.entity_resolution import EntityCandidate, signal_backends
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import async_load_ontology, load_ontology
from dataknobs_common.ontology.loader import build_resolver

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path


def semantic_document(source: Path, destination: Path) -> Path:
    """The fixture's vocabulary, with a ``resolver:`` naming an async-only rung."""
    document = yaml.safe_load(source.read_text())
    document["ontology"]["resolver"] = {"rungs": [{"kind": "semantic"}]}
    destination.write_text(yaml.safe_dump(document))
    return destination


class CountingSignal:
    """A synchronous rung that records that it was built.

    A real implementation of the protocol rather than a mock: the question
    these tests ask is *was anything constructed*, and a class that answers it
    by existing is both the cheapest way to ask and the one that also proves
    the registry accepts a consumer's own rung.
    """

    built = 0

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        type(self).built += 1

    @property
    def name(self) -> str:
        return "semantic"

    def narrows(self) -> bool:
        return False

    def candidates(
        self, query: str, k: int, *, filter: dict[str, Any] | None = None
    ) -> list[EntityCandidate]:
        return []

    def candidates_many(
        self, queries: Sequence[str], k: int, *, filter: dict[str, Any] | None = None
    ) -> list[list[EntityCandidate]]:
        return [[] for _ in queries]


def make_counting_signal(config: dict[str, Any] | None = None) -> CountingSignal:
    """A **callable** factory for :class:`CountingSignal`.

    A named function rather than a lambda over the class, and not the class
    itself: registering the class takes the class-factory path, which is
    judged at registration. This is the other path, judged on the instance,
    and keeping the two apart is what one of the tests below is about.
    """
    return CountingSignal(config)


@pytest.fixture
def restored_registry() -> Iterator[None]:
    """Undo a registration, so one test cannot decide another's outcome.

    ``signal_backends`` is module-global by design -- a consumer registers
    into it at import -- so a test that registers must put the declaration
    back or every later test runs against a registry that can build the thing
    they assert is unbuildable.

    **The metadata is restored as well as the reason, and it has to be.**
    ``declare_unavailable`` with no ``metadata=`` leaves whatever the key
    already carried, so restoring by reason alone put the *sentence* back and
    left the *facts* wherever the test moved them. That was invisible while
    nothing read them: it stopped being invisible when
    :func:`~dataknobs_common.ontology.loader._refuse_async_only_rungs` began
    choosing its remedy off ``needs_io``, at which point a test that had set
    that key decided the message every later test in the process saw.
    """
    reason = signal_backends.unavailable_reason("semantic")
    metadata = dict(signal_backends.get_metadata("semantic"))
    assert reason is not None, "the declaration under test is absent before the test runs"
    try:
        yield
    finally:
        signal_backends.declare_unavailable("semantic", reason=reason, metadata=metadata)


def test_a_rung_the_sync_flavour_cannot_build_is_refused(
    mammals_path: Path, tmp_path: Path
) -> None:
    """Refused by the **build** door, naming the kind and the way out.

    **The way out used to be one sentence for every kind, and it was false
    for half of them.** This assertion pinned ``async_load_ontology``, on the
    reasoning that whatever the synchronous door refuses the asynchronous one
    accepts. That holds for a rung whose asynchrony is its own; it does not
    hold for a rung constructed over a **live handle**, which no door taking a
    document and a vocabulary can conjure. Following the old sentence for
    ``semantic`` produced a second error rather than a rung.

    So what is asserted here is the remedy for *this* kind, which the mark's
    own ``needs_io`` picks out -- and the sibling below asserts that a rung
    without it still gets the original sentence, because that half was always
    true and losing it would be the other way to make the message wrong.
    """
    assert signal_backends.get_metadata("semantic").get("needs_io") is True, (
        "the mark's `needs_io` is what picks the remedy, so a test that moved it "
        "and did not put it back decides this assertion -- see `restored_registry`"
    )
    path = semantic_document(mammals_path, tmp_path / "semantic.yaml")
    ontology = load_ontology(path)

    with pytest.raises(ValidationError) as raised:
        build_resolver(path, ontology)

    message = str(raised.value)
    assert "semantic" in message
    assert "no synchronous form" in message
    assert "OntologyRegistry" in message
    assert "handles=" in message
    assert "async_load_ontology" not in message, (
        "the door that builds an ontology cannot build a rung that needs an index"
    )


def test_a_rung_that_needs_no_handle_is_still_sent_to_the_async_door(
    mammals_path: Path, tmp_path: Path, restored_registry: None
) -> None:
    """The half of the message that was always true, kept.

    A kind declared unavailable **without** ``needs_io`` is a rung this
    distribution could build in the other flavour, and
    :func:`~dataknobs_common.ontology.loader.async_build_resolver` really is
    where such a caller goes. Asserted over a mark stood up for the purpose
    rather than over ``semantic``, because ``semantic`` is the kind that made
    the distinction necessary and cannot demonstrate its other side.
    """
    signal_backends.declare_unavailable(
        "semantic",
        reason="a stand-in rung with no backing to reach",
        metadata={"flavour": "async", "needs_io": False},
    )
    path = semantic_document(mammals_path, tmp_path / "semantic.yaml")

    with pytest.raises(ValidationError) as raised:
        build_resolver(path, load_ontology(path))

    message = str(raised.value)
    assert "async_build_resolver" in message
    assert "async_load_ontology" in message
    assert "OntologyRegistry" not in message


def test_the_refusal_constructs_nothing(mammals_path: Path, tmp_path: Path) -> None:
    """*At validation time* means the registry's ``create`` is never reached.

    The distinction a message cannot show: a door that built the rung and
    reported the failure would produce a message just as good and a refusal
    that is not a refusal. Asserted by counting constructions, because that is
    the only side of it that differs.
    """
    path = semantic_document(mammals_path, tmp_path / "semantic.yaml")
    before = CountingSignal.built

    with pytest.raises(ValidationError):
        build_resolver(path, load_ontology(path))

    assert CountingSignal.built == before


def test_the_refusal_holds_with_dataknobs_data_imported(mammals_path: Path, tmp_path: Path) -> None:
    """The answer does not depend on what else has been imported.

    A declaration left to the package that implements the rung would be
    present in an application and absent here -- so the criterion would pass
    in production and fail in its own test file, with nothing saying which.
    ``common`` declares the fact itself, and importing the package that owns
    the class does not move it: that package registers an *asynchronous* rung,
    and the synchronous registry's mark is untouched by it.
    """
    import dataknobs_data  # noqa: F401  -- imported for its side effects, which is the point

    path = semantic_document(mammals_path, tmp_path / "semantic.yaml")

    with pytest.raises(ValidationError):
        build_resolver(path, load_ontology(path))


def test_registering_a_synchronous_rung_makes_the_door_accept_its_kind(
    mammals_path: Path, tmp_path: Path, restored_registry: None
) -> None:
    """The other side, and without it the criterion asserts only a refusal.

    The cheapest way to pass a refusal-only criterion is a permanent refusal,
    which would break the promise that a consumer's own rung is buildable.
    Registering clears the unavailable mark, so supplying the rung is exactly
    what makes the door accept the kind it had been refusing.
    """
    path = semantic_document(mammals_path, tmp_path / "semantic.yaml")
    signal_backends.register("semantic", CountingSignal)

    ontology = load_ontology(path)
    resolver = build_resolver(path, ontology)

    assert [type(rung) for rung in resolver.rungs] == [CountingSignal]


def test_unavailable_reason_reports_why_without_building(restored_registry: None) -> None:
    """The reader the refusal needs, and the two nothings it keeps apart.

    ``None`` for a creatable key and ``None`` for an unknown one -- which is
    not an ambiguity, because ``is_known`` answers that question and is the
    reader for it. One reader per question rather than a sentinel doing two
    jobs.
    """
    assert signal_backends.unavailable_reason("semantic") == (
        "SemanticSignal has no synchronous form"
    )
    assert signal_backends.is_known("semantic")

    assert signal_backends.unavailable_reason("exact") is None
    assert signal_backends.is_registered("exact")

    assert signal_backends.unavailable_reason("no-such-rung") is None
    assert not signal_backends.is_known("no-such-rung")


def test_the_scanning_rung_is_registered_in_both_flavours() -> None:
    """A rung with two flavours needs no ``declare_unavailable`` mark.

    ``semantic`` above is the asymmetric case and is the reason this one is
    worth stating beside it: asking either registry for ``scan`` builds
    something, so neither has a reason to explain itself. The classes are
    asserted rather than only the keys, because a factory registered under the
    wrong flavour is exactly what the two registries exist to separate -- and
    a key that builds *a* rung is not the claim.
    """
    from dataknobs_common.entity_resolution import (
        AsyncScanningSignal,
        ScanningSignal,
        async_signal_backends,
    )
    from dataknobs_common.ontology import AsyncMappingEntitySource, Entity, MappingEntitySource

    assert signal_backends.is_registered("scan")
    assert signal_backends.unavailable_reason("scan") is None
    assert async_signal_backends.is_registered("scan")
    assert async_signal_backends.unavailable_reason("scan") is None

    declared = {"beagle": Entity(id="beagle", type="Breed", name="Beagle")}
    built = signal_backends.create(
        config={"kind": "scan", "entities": MappingEntitySource(declared)}
    )
    twin = async_signal_backends.create(
        config={"kind": "scan", "entities": AsyncMappingEntitySource(declared)}
    )

    assert isinstance(built, ScanningSignal)
    assert isinstance(twin, AsyncScanningSignal)
    assert built.name == twin.name == "scan"


def test_the_two_registries_refuse_each_other_s_flavour() -> None:
    """Why there are two registries rather than one, asserted rather than stated.

    A ``@runtime_checkable`` protocol compares member *names* and nothing
    about their async-ness, so a single guard could not tell the twins apart:
    both flavours have ``candidates``. Two guards can, and the refusal names
    the member and the direction rather than surfacing as a ``TypeError`` at
    the caller's ``await``.

    Both registration paths are exercised because they fail at different
    moments. A class is judged when registered; a callable cannot be, since
    nothing yet knows what it returns, so it is judged on the instance.
    """
    from dataknobs_common.entity_resolution import async_signal_backends

    with pytest.raises(TypeError) as at_registration:
        async_signal_backends.register("sync-class", CountingSignal)

    assert "AsyncMatchSignal" in str(at_registration.value)
    assert "candidates" in str(at_registration.value)

    async_signal_backends.register("sync-callable", make_counting_signal)
    try:
        with pytest.raises(Exception) as at_create:
            async_signal_backends.create(config={"kind": "sync-callable"})
        assert "AsyncMatchSignal" in str(at_create.value)
    finally:
        async_signal_backends.unregister("sync-callable")


def test_a_conforming_rung_is_accepted_by_its_own_registry(restored_registry: None) -> None:
    """The other direction, so the guard is not merely a refusal machine."""
    signal_backends.register("semantic", CountingSignal)

    built = signal_backends.create(config={"kind": "semantic"})

    assert isinstance(built, CountingSignal)


def test_a_loader_does_not_refuse_a_rung_it_never_builds(
    mammals_path: Path, tmp_path: Path
) -> None:
    """Loading a vocabulary is not proposing to build a cascade over it.

    The refusal's subject is *this door cannot build that rung*, which is
    false of a door that builds no rung: an ``Ontology`` is a value, and a
    caller reading one for its entity types -- a validator, an exporter, a
    migration -- was being refused over a ``resolver:`` section it never read.

    The cost is stated rather than hidden: such a caller is no longer told
    early. It is told in full at the first call that proposes to build the
    thing, which the test above asserts, and which is the only call for which
    the message is true.

    Both loaders, because the claim ``async_load_ontology`` makes in its own
    docstring -- the same file and the same refusals -- is only worth as much
    as something checking it.
    """
    path = semantic_document(mammals_path, tmp_path / "semantic.yaml")

    ontology = load_ontology(path)
    assert ontology.entity("beagle").name == "Beagle"

    asynchronous = asyncio.run(async_load_ontology(path))
    assert asynchronous.id == ontology.id


def test_a_rung_that_ships_elsewhere_says_so_rather_than_reading_as_a_typo() -> None:
    """The second mark, and it is marked for a different condition than the first.

    ``semantic`` is withdrawn because no synchronous form of it exists
    anywhere. ``authority`` exists in both flavours and ships in
    ``dataknobs-xization``, which ``dataknobs-common`` cannot import and must
    not -- so the key is declared here and the class is not. Both registries
    carry it, and importing ``dataknobs_xization.entity_resolution`` clears
    both marks by registering over them.

    **The pre-import state is what is asserted, so the test checks that it is
    still the pre-import state.** ``bin/test.sh`` gives each package its own
    pytest process and nothing in ``common`` depends on ``xization``, so under
    the runner of record nothing here has imported it. That is a property of
    *that runner* and not of this test, and the workspace supports another:
    the root ``pytest.ini`` sets ``testpaths = packages tests`` -- deliberately,
    so a bare ``pytest`` at the root does not skip the workspace guards -- and
    under it ``xization``'s own suite imports the module and registers the kind
    in these same process-global registries. ``pytest-randomly`` then decides
    which suite runs first, so the assertion below held or failed by seed.

    Skipping rather than asserting a weaker thing: once the module is imported
    the mark is *correctly* gone, and there is no version of these assertions
    that is true in both worlds. The condition is named so a reader who meets
    the skip learns why rather than assuming the test was disabled.

    The reason names the distribution *and* the import, because a reader
    stuck on this key has two questions and a package name answers only the
    first: registration happens at a module's import, not at a distribution's
    presence.
    """
    if "dataknobs_xization.entity_resolution" in sys.modules:
        pytest.skip(
            "dataknobs_xization.entity_resolution is imported in this process, "
            "so it has registered over both marks -- which is the behaviour "
            "xization's own suite asserts. The mark is only observable before "
            "that import, and bin/test.sh is the runner that guarantees it."
        )

    from dataknobs_common.entity_resolution import async_signal_backends

    for registry in (signal_backends, async_signal_backends):
        assert registry.is_known("authority")
        assert not registry.is_registered("authority")

        reason = registry.unavailable_reason("authority")
        assert reason is not None
        assert "dataknobs-xization" in reason
        assert "dataknobs_xization.entity_resolution" in reason

        assert registry.get_metadata("authority")["requires_install"] == (
            "pip install dataknobs-xization"
        )

    # The flavour a door refuses a composition by, which is per registry and
    # is the one piece of this metadata the two marks do not share.
    assert signal_backends.get_metadata("authority")["flavour"] == "sync"
    assert async_signal_backends.get_metadata("authority")["flavour"] == "async"
