"""The key codec: why it exists, why it has no default, and what it round-trips.

Three claims, and the first is the measurement that decided the shape:

* **a rendering cannot be defaulted.** A key is addressed by *equality* and
  ``object.__repr__`` is a function of *identity*, so two equal keys render to
  two strings -- and one node then addresses two entities, type-checked and
  silent. That is asserted here rather than argued, because it is the reason
  the codec is an argument the type checker will not let you omit;
* **the grammar was never the obstacle.** A local id keeps its colons and the
  source segment is decided by set membership, so a rendered key round-trips
  through ``qualify`` and back through ``localize`` -- over a frozen dataclass,
  the same key carrying a colon, a tuple, and an integer;
* **the ``str`` path is unchanged**, which is what makes the widening free for
  every caller who was not asking for it.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from dataknobs_common.ontology import (
    AsyncMappingAssertionSource,
    AsyncMappingEntitySource,
    AsyncOntology,
    KeyCodec,
    Ontology,
    SourceDescription,
    StrCodec,
    async_load_ontology,
    load_ontology,
)
from dataknobs_common.ontology.sources import MappingAssertionSource, MappingEntitySource


@dataclass(frozen=True)
class Sku:
    """A consumer's key: hashable, value-equal, and not a string."""

    plant: str
    line: int


class SkuCodec:
    """``Sku`` in one direction and back. Two functions, both the consumer's."""

    def to_id(self, key: Sku, /) -> str:
        return f"{key.plant}/{key.line}"

    def from_id(self, rendered: str, /) -> Sku:
        plant, _, line = rendered.rpartition("/")
        return Sku(plant=plant, line=int(line))


def _onto(codec: object) -> Ontology:
    """An ontology with empty sources -- the two doors read neither."""
    return Ontology(
        id="acme",
        version="1.0",
        entity_types={},
        relation_types={},
        entities=MappingEntitySource({}),
        assertions=MappingAssertionSource([]),
        taxonomies={},
        describes=(),
        codec=codec,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# Why there is no default
# --------------------------------------------------------------------------


def test_two_equal_keys_render_to_two_strings_under_the_default_repr() -> None:
    """The measurement that refuses ``repr()`` as the default rendering.

    The structure axis is perfect over such a key -- equal, hashing alike,
    answering alike as a mapping key -- and the content axis would address two
    different entities from one node, because the rendering is a function of
    identity and the addressing is a function of equality. Nothing reports it:
    it type-checks, and every test holding a single key object passes.
    """

    class OpaqueKey:
        """Value-equal and hashable, with the ``__repr__`` it inherited."""

        def __init__(self, part: str) -> None:
            self.part = part

        def __eq__(self, other: object) -> bool:
            return isinstance(other, OpaqueKey) and other.part == self.part

        def __hash__(self) -> int:
            return hash(self.part)

    first, second = OpaqueKey("a-1"), OpaqueKey("a-1")

    assert first == second
    assert hash(first) == hash(second)
    assert {first: "one"}[second] == "one", "the structure axis is perfect over this key"

    assert repr(first) != repr(second), (
        "and the rendering is not -- so a default rendering would let one node address two entities"
    )


def test_a_bound_cannot_say_has_a_string_representation() -> None:
    """The other refused shape: a structural bound admits everything.

    *A type that has a string representation* is, in Python, every type. A
    bound narrow enough to discriminate would have to be nominal, and ``str``
    itself would then fail it -- destroying the default that makes the whole
    widening free.
    """
    for candidate in ("x", 3, [1], object(), Sku, Sku("ACME", 1)):
        assert hasattr(candidate, "__str__")


def test_the_codec_cannot_be_omitted() -> None:
    """Required rather than defaulted, which is the guard.

    A defaulted codec would let an ontology over a consumer's key be built
    carrying the identity -- which type-checks, and renders their key as its
    ``repr``. The type checker reports the omission as a missing argument; this
    is the same refusal at runtime, so the property is asserted rather than
    merely declared.
    """
    with pytest.raises(TypeError, match="codec"):
        Ontology(  # type: ignore[call-arg]
            id="acme",
            version="1.0",
            entity_types={},
            relation_types={},
            entities=MappingEntitySource({}),
            assertions=MappingAssertionSource([]),
            taxonomies={},
            describes=(),
        )


# --------------------------------------------------------------------------
# What it round-trips
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "codec"),
    [
        pytest.param(Sku("ACME", 3), SkuCodec(), id="frozen-dataclass"),
        pytest.param("has:a:colon", StrCodec(), id="colons-in-the-local-id"),
        pytest.param(
            ("a", 2),
            type(
                "TupleCodec",
                (),
                {
                    "to_id": lambda self, key, /: f"{key[0]}|{key[1]}",
                    "from_id": lambda self, rendered, /: (
                        rendered.split("|")[0],
                        int(rendered.split("|")[1]),
                    ),
                },
            )(),
            id="tuple",
        ),
        pytest.param(
            41,
            type(
                "IntCodec",
                (),
                {"to_id": lambda self, key, /: str(key), "from_id": lambda self, r, /: int(r)},
            )(),
            id="int",
        ),
    ],
)
def test_a_rendered_key_round_trips_through_both_doors(key: object, codec: object) -> None:
    """``qualify`` out, ``localize`` back, and the same key at the end.

    The grammar carries this rather than tolerating it: a local id keeps its
    colons and the source segment is decided by membership in the declared set,
    so the local position is not parsed by guessing. The colon case is the one
    that would have failed a grammar that split on the last separator.
    """
    onto = _onto(codec)

    qualified = onto.qualify(key)

    assert qualified.startswith("acme:")
    assert onto.localize(qualified) == key


def _every_spelling(onto: Ontology, key: object) -> list[str]:
    """Every id ``qualify`` can be asked to build for ``key``.

    Derived from what the member *declares* rather than listed here, so the
    test below states a law over the whole surface and needs no edit when the
    surface changes. A spelling nobody can write is a spelling nothing has to
    read back.
    """
    spellings = [onto.qualify(key)]
    if "source_id" in inspect.signature(type(onto).qualify).parameters:
        spellings.append(onto.qualify(key, "plant_a"))  # type: ignore[call-arg]
    return spellings


def test_every_id_qualify_builds_localize_reads_back() -> None:
    """The law the codec is a **pair** for, over a vocabulary binding two sources.

    ``localize`` hands its result to ``from_id``, which is documented as the
    inverse of ``to_id`` -- so every string ``to_id`` never produced is a
    string ``from_id`` has no contract to parse, and parsing it anyway is
    silent. ``SkuCodec.from_id`` splits on the last ``/``, so a source segment
    in front of a rendered key comes back as a ``Sku`` whose plant is
    ``'plant_a:ACME'``: no exception, and a key that addresses nothing.

    The multi-source ontology is the whole point. ``localize`` keeps the source
    segment where one applies -- that is the space ``entities`` speaks -- so
    the segment is *inside* what the codec parses, and anything that composes
    one behind the codec's back breaks the pair.
    """
    onto = replace(
        _onto(SkuCodec()),
        describes=(
            SourceDescription("plant_a", "memory", None, {}, frozenset()),
            SourceDescription("plant_b", "memory", None, {}, frozenset()),
        ),
    )
    key = Sku("ACME", 3)

    for qualified in _every_spelling(onto, key):
        assert onto.localize(qualified) == key, (
            f"{qualified!r} was built by qualify and does not read back"
        )


def test_the_str_path_is_the_identity_and_reads_as_it_always_did(mammals_path: Path) -> None:
    """A loaded vocabulary is ``str``-keyed, and its codec changes nothing.

    An authored document's ids are the strings its author typed, so this door
    binds the parameter rather than passing one through -- and the two members
    that spend the codec answer exactly what they answered before it existed.
    """
    onto = load_ontology(mammals_path)

    assert isinstance(onto.codec, StrCodec)
    assert isinstance(onto.codec, KeyCodec)
    assert onto.qualify("beagle") == "mammals:beagle"
    assert onto.localize("mammals:beagle") == "beagle"
    assert onto.entity("beagle") is not None


@pytest.mark.asyncio
async def test_the_async_door_binds_it_too(mammals_path: Path) -> None:
    """The twin, because a field on one half is the drift this closes."""
    onto = await async_load_ontology(mammals_path)

    assert isinstance(onto.codec, StrCodec)
    assert onto.localize("mammals:beagle") == "beagle"

    built: AsyncOntology = AsyncOntology(
        id="acme",
        version="1.0",
        entity_types={},
        relation_types={},
        entities=AsyncMappingEntitySource({}),
        assertions=AsyncMappingAssertionSource([]),
        taxonomies={},
        describes=(),
        codec=SkuCodec(),  # type: ignore[arg-type]
    )

    assert built.localize(built.qualify(Sku("ACME", 3))) == Sku("ACME", 3)
