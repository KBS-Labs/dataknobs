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
