"""Refusing a rung this flavour cannot build, before anything is built.

The refusal is computed from the declared **kind**, which is what lets it hold
with no rung constructed. That property is the whole criterion, and it is not
observable from the message -- so the assertions below reach past the message
to what was and was not reached.
"""

from __future__ import annotations

import asyncio
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
    """
    reason = signal_backends.unavailable_reason("semantic")
    assert reason is not None, "the declaration under test is absent before the test runs"
    try:
        yield
    finally:
        signal_backends.declare_unavailable("semantic", reason=reason)


def test_a_rung_the_sync_flavour_cannot_build_is_refused(
    mammals_path: Path, tmp_path: Path
) -> None:
    """Refused by the **build** door, naming the kind and the way out."""
    path = semantic_document(mammals_path, tmp_path / "semantic.yaml")
    ontology = load_ontology(path)

    with pytest.raises(ValidationError) as raised:
        build_resolver(path, ontology)

    message = str(raised.value)
    assert "semantic" in message
    assert "async_load_ontology" in message
    assert "no synchronous form" in message


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
