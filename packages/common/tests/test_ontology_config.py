"""``OntologyConfig`` inherits a property from its base class, and nothing asserted it.

:class:`~dataknobs_common.structured_config.StructuredConfig` declares that
``type(cfg).from_dict(cfg.to_dict()) == cfg``, and
:func:`~dataknobs_common.testing.assert_structured_config_roundtrip` is the
shared statement of it. Every other config type in this package that carries
the property says so in its own tests; this one did not, so the property held
by luck rather than by check.

That gap is not theoretical. The roundtrip is an *equality* property, so it is
exactly what a change to the type's ``eq``/``frozen`` declaration would break,
and a type with no assertion is the one such a change passes through in
silence. The tests below are the assertion the type was missing, not a report
of anything currently wrong: the property holds today, bare and fully
populated, which is why they are green.
"""

from __future__ import annotations

from collections.abc import Hashable

from dataknobs_common.ontology.config import OntologyConfig
from dataknobs_common.structured_config import StructuredConfig
from dataknobs_common.testing import assert_structured_config_roundtrip


def _populated() -> OntologyConfig:
    """A config with every field set, including the three that default to None."""
    return OntologyConfig(
        id="creatures",
        version="1",
        imports=["base"],
        entity_types=[{"id": "animal"}],
        relation_types=[{"id": "isa"}],
        entities=[{"id": "dog", "type": "animal"}],
        assertions=[{"subject": "dog", "relation": "isa", "object": "animal"}],
        sources=[{"source_id": "s", "backend": "memory"}],
        overlay={"note": "kept"},
        taxonomies=[{"id": "t", "relation": "isa"}],
        index={"backend": "memory"},
        resolver={"kind": "mapping"},
    )


class TestOntologyConfigStructured:
    """The structured-config contract, stated for this type as it is for its peers."""

    def test_it_is_a_structured_config(self) -> None:
        """The property below is inherited, so the inheritance is worth pinning."""
        assert issubclass(OntologyConfig, StructuredConfig)

    def test_construction_parity(self) -> None:
        """``from_dict`` and the constructor reach the same value."""
        assert OntologyConfig.from_dict({"id": "creatures", "version": "1"}) == (
            OntologyConfig(id="creatures", version="1")
        )

    def test_roundtrip_bare(self) -> None:
        """Only the two required fields, so the defaults travel the whole way."""
        assert_structured_config_roundtrip(OntologyConfig(id="creatures", version="1"))

    def test_roundtrip_populated(self) -> None:
        """Every field set, so nested mappings and sequences travel too.

        The bare case alone would pass on a type that dropped every optional
        field, which is the roundtrip failure most worth catching.
        """
        assert_structured_config_roundtrip(_populated())


class TestOntologyConfigIsHonestlyUnhashable:
    """Equality and unhashability, which this type has to hold at once.

    The base class is frozen, so ``frozen=False`` -- the ordinary way to say
    *compared field-wise, not hashable* -- is unavailable: a dataclass may not
    unfreeze an inherited one. The type reaches the same contract by setting
    ``__hash__`` in its class body, and ``dataclasses`` honours that only
    while no explicit ``__eq__`` sits beside it. That is a quiet condition, so
    it is pinned rather than trusted: a later hand-written ``__eq__`` would
    regenerate the hash and put the type back in the state it just left.
    """

    def test_the_type_does_not_claim_to_be_hashable(self) -> None:
        """The check is how a caller asks, so it has to answer honestly."""
        assert not issubclass(OntologyConfig, Hashable)
        assert not isinstance(_populated(), Hashable)

    def test_hashing_one_raises_the_ordinary_error(self) -> None:
        """Unhashable in the way every other unhashable value is."""
        try:
            hash(_populated())
        except TypeError:
            return
        raise AssertionError("hash() succeeded on a type declaring __hash__ = None")

    def test_equality_still_compares_the_fields(self) -> None:
        """The half the base class's roundtrip property depends on.

        Asserted here as well as through the roundtrip because the two fail
        for different reasons: the roundtrip breaks if ``to_dict`` and
        ``from_dict`` disagree, this breaks if the type stops comparing.
        """
        assert _populated() == _populated()
        assert _populated() != OntologyConfig(id="creatures", version="1")
