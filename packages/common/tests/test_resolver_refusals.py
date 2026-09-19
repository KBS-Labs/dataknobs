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
from dataknobs_common.ontology import (
    async_build_resolver,
    async_load_ontology,
    load_ontology,
)
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

    A kind declared unavailable without ``needs_io`` **and whose flavour here
    is** ``async`` is a rung this distribution could build in the other
    flavour, and
    :func:`~dataknobs_common.ontology.loader.async_build_resolver` really is
    where such a caller goes. Asserted over a mark stood up for the purpose
    rather than over ``semantic``, because ``semantic`` is the kind that made
    the distinction necessary and cannot demonstrate its other side.

    **The flavour in the stand-in is load-bearing and used not to be.** This
    mark said only ``needs_io: False``, which made it a stand-in for a
    population of one that does not exist: the real kind reaching the
    ``needs_io`` branch's ``else`` is ``authority``, whose mark *here*
    declares ``flavour: "sync"`` and which this sentence is wrong for. The
    sibling below is that case.
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


def test_a_rung_that_ships_elsewhere_is_sent_to_the_import_and_not_to_a_door(
    mammals_path: Path, tmp_path: Path
) -> None:
    """The third remedy, for the only kind that ever reached the second one.

    The two-way split read ``needs_io`` and sent everything else to
    :func:`~dataknobs_common.ontology.loader.async_build_resolver`. Measured
    over the marks this registry actually carries, *everything else* is
    ``authority`` and nothing more --- and that door carries the **identical**
    mark, so an author who followed the sentence got the same refusal back
    from the door it named. A remedy whose remedy builds nothing is the
    failure this whole paragraph of the module exists to have fixed once.

    The real remedy is the reason's own: import the module. After that this
    door builds the kind, because ``AuthoritySignal`` has a synchronous form
    --- which is exactly what the mark says in the one key the two flavours'
    marks do not share, ``flavour``. So that is the discriminator now.

    Skipped rather than weakened where the import has already happened, for
    the reason the mark's own test gives: once ``xization`` is imported the
    mark is *correctly* gone and there is no version of this that holds in
    both worlds.
    """
    if "dataknobs_xization.entity_resolution" in sys.modules:
        pytest.skip(
            "dataknobs_xization.entity_resolution is imported in this process, so "
            "the kind is registered and there is no refusal to read"
        )
    assert signal_backends.get_metadata("authority")["flavour"] == "sync", (
        "the mark must say a synchronous form exists, or this asserts the wrong branch"
    )

    document = yaml.safe_load(mammals_path.read_text())
    document["ontology"]["resolver"] = {"rungs": [{"kind": "authority"}]}
    path = tmp_path / "authority.yaml"
    path.write_text(yaml.safe_dump(document))

    with pytest.raises(ValidationError) as raised:
        build_resolver(path, load_ontology(path))

    message = str(raised.value)
    assert "dataknobs_xization.entity_resolution" in message
    assert "import the module that registers it" in message
    assert "async_build_resolver" not in message, (
        "that door carries the same mark, so naming it sends the author in a circle"
    )
    assert "async_load_ontology" not in message


@pytest.mark.parametrize(
    ("section", "says"),
    [
        ({"rungs": ["exact"]}, "not a mapping"),
        ({"rungs": [{"threshold": 0.5}]}, "names no `kind:`"),
        ({"rungs": [{"kind": "exatc"}]}, "does not build"),
    ],
)
def test_every_way_a_composition_can_be_unbuildable_is_one_exception_type(
    section: dict[str, Any], says: str, mammals_path: Path, tmp_path: Path
) -> None:
    """Both doors' ``Raises:`` promise one type, and three shapes escaped it.

    A bare string in ``rungs:`` came out of the spec merge as ``TypeError``,
    an entry with no ``kind:`` out of the registry's key resolution as
    ``ValueError`` -- raised *before* the wrapper that would have converted
    it -- and a misspelled kind as ``NotFoundError``. A caller catching what
    the docstring named caught none of the three.

    The one consumer that noticed converted them on **its** side, which left
    every other caller of these doors holding the original types. They are
    normalized at the door now, which is the layer the promise is made at.
    """
    document = yaml.safe_load(mammals_path.read_text())
    document["ontology"]["resolver"] = section
    path = tmp_path / "composition.yaml"
    path.write_text(yaml.safe_dump(document))
    ontology = load_ontology(path)

    with pytest.raises(ValidationError) as raised:
        build_resolver(path, ontology)
    assert says in str(raised.value)

    asynchronous = asyncio.run(async_load_ontology(path))
    with pytest.raises(ValidationError) as raised_async:
        asyncio.run(async_build_resolver(path, asynchronous))
    assert says in str(raised_async.value)


def test_the_synchronous_door_forwards_handles_as_its_twin_does(
    mammals_path: Path, tmp_path: Path, restored_registry: None
) -> None:
    """The channel, on the door that used to lack it.

    The asymmetry was argued from the shipped rungs -- *the synchronous door
    builds no rung that needs a handle, because the one rung that does has no
    synchronous form* -- which is true and is not the question. The registry
    both doors read is a published extension point, so the rung with a
    synchronous form and a live backing is a consumer's to write, and a
    channel they cannot reach is one they reimplement: a ``CascadingResolver``
    assembled beside this door, which is a second copy of it.

    Asserted with a rung that records what it was handed, so this is the
    merge order rather than the parameter's existence: a handle beats a key
    the document spelled the same way, and ``entities`` beats both.
    """
    seen: dict[str, Any] = {}

    class _Recording(CountingSignal):
        def __init__(self, config: dict[str, Any] | None = None) -> None:
            super().__init__(config)
            seen.update(config or {})

    signal_backends.register("semantic", _Recording, override=True)
    document = yaml.safe_load(mammals_path.read_text())
    document["ontology"]["resolver"] = {
        "rungs": [{"kind": "semantic", "backing": "what the document wrote"}]
    }
    path = tmp_path / "handled.yaml"
    path.write_text(yaml.safe_dump(document))
    ontology = load_ontology(path)
    live = object()

    resolver = build_resolver(path, ontology, handles={"backing": live})

    assert [type(rung) for rung in resolver.rungs] == [_Recording]
    assert seen["backing"] is live, "a handle must beat the key a document spelled"
    assert seen["entities"] is ontology.entities, "and `entities` must beat the handle"


def test_a_handle_may_not_be_named_kind(mammals_path: Path, tmp_path: Path) -> None:
    """The one key that would make two readers disagree about one composition.

    Every refusal that runs before construction reads ``kind:`` off the
    document; the registry resolves it off the **merged** config. A handle
    spelled ``kind`` therefore has a composition checked as one thing and
    built as another, with nothing between them noticing -- which is exactly
    the class of silent redirection the merge order's published rule exists to
    rule out, in the one position that rule cannot cover.

    Asserted on both doors, because the merge is one body and a guard on one
    of them would be a guard neither reader could rely on.
    """
    document = yaml.safe_load(mammals_path.read_text())
    document["ontology"]["resolver"] = {"rungs": [{"kind": "exact"}, {"kind": "alias"}]}
    path = tmp_path / "kinded.yaml"
    path.write_text(yaml.safe_dump(document))

    with pytest.raises(ValidationError, match="may not carry 'kind'"):
        build_resolver(path, load_ontology(path), handles={"kind": "scan"})

    asynchronous = asyncio.run(async_load_ontology(path))
    with pytest.raises(ValidationError, match="may not carry 'kind'"):
        asyncio.run(async_build_resolver(path, asynchronous, handles={"kind": "scan"}))


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
