"""The vocabulary documents this suite's fixtures and guards both load.

**A module pytest does not collect**, which is the whole reason it exists.
``conftest.py`` is imported by pytest under a name of pytest's choosing, so a
test module importing it back by the bare name ``conftest`` binds a *second*
module object for the same file -- module-level code run twice, and which copy
a name resolves to decided by ``sys.path`` ordering nobody declared. That is
what ``declare_import_root``'s own docstring warns about and names the remedy
for: an underscore-prefixed module, imported by both.

Measured before this file existed: ``packages/common/tests/conftest.py`` was
live in a run as ``tests.conftest`` *and* as ``conftest``, two objects, and the
guard comparing a published fence against a constant was reading the second one
while every fixture around it came from the first. Outside pytest the same
import resolves to the repository root's ``conftest.py`` instead and raises.
``tests/test_pytest_collection_integrity.py`` fails on such an import now.

The documents live here rather than in ``conftest.py`` because both a fixture
and a guard need them; the *fixtures* stay there, where pytest looks for them.

**Not only documents.** :class:`Sku` and :class:`SkuCodec` are here for the
same reason the documents are: two suites need the one vocabulary whose keys
are not strings, and a second spelling of a two-field key and its codec is two
fixtures drifting under suites that each believe they are testing the same
thing.
"""

from __future__ import annotations

from dataclasses import dataclass

MAMMALS_DOCUMENT = """\
ontology:
  id: mammals
  version: "1.0"

  entity_types:
    - id: Species
      attributes:
        - {name: latin_name, type: string, required: true}
    - id: Breed
      isa: Species

  relation_types:
    - id: isa
      transitive: true

  entities:
    - {id: mammal, type: Species, name: Mammal}
    - {id: dog,    type: Species, name: Dog, aliases: [Canine, "Domestic dog"]}
    - {id: beagle, type: Breed,   name: Beagle, aliases: [Beagles],
       source: {source_id: clinic_db, table: species, key: "sp-2291"}}

  assertions:
    - {subject: dog,    relation: isa, object: mammal}
    - {subject: beagle, relation: isa, object: dog}
"""
"""The worked example, as a hand-edited file would carry it.

Shared by the ontology suites rather than repeated in each: they assert
different things *about the same document*, and six near-copies of a YAML
block drift until the thing they are all supposedly loading is six things.

**A seventh copy is published**, as the ``worked-input`` fence in
``packages/common/docs/guides/ontology.md``, and it has to be: a reader copies
the fence and this constant is not reachable from a guide. The two are held
identical -- less the ``# mammals.yaml`` line a published file names itself
with -- by ``test_worked_input_fences.py`` beside this file, so neither may be edited
alone.
"""


MAMMALS_V11_DOCUMENT = """\
ontology:
  id: mammals
  version: "1.1"

  entity_types:
    - id: Species
      attributes:
        - {name: latin_name, type: string, required: true}
        - {name: lifespan_years, type: number, field_type: float}
    - id: Breed
      isa: Species                        # <-- the TYPE lattice
      attributes:
        - {name: akc_group, type: string}

  relation_types:
    - id: isa
      transitive: true

  entities:
    - {id: mammal, type: Species, name: Mammal,
       description: "Warm-blooded, milk-producing vertebrates."}
    - {id: dog, type: Species, name: Dog, aliases: [Canine, "Domestic dog"],
       description: "A domesticated carnivoran."}
    - {id: retriever, type: Breed, name: Retriever}
    - {id: golden_retriever, type: Breed, name: Golden Retriever, aliases: [Goldie]}
    - {id: beagle, type: Breed, name: Beagle, aliases: [Beagles],
       source: {source_id: clinic_db, table: species, key: "sp-2291"}}

  assertions:
    - {subject: dog, relation: isa, object: mammal}            # <-- the INSTANCE
    - {subject: retriever, relation: isa, object: dog}         #     lattice, same
    - {subject: golden_retriever, relation: isa, object: retriever}   # relation id
    - {subject: beagle, relation: isa, object: dog}
    # an attribute value is an assertion whose object is a Literal
    - {subject: dog, relation: lifespan_years, object: 12}

  taxonomies:
    - {id: species, name: Species, relation: isa}
"""
"""The same vocabulary at version 1.1: three more entities, and a declared axis.

Separate from :data:`MAMMALS_DOCUMENT` rather than replacing it. The v1.0
document is the one an acceptance criterion requires to run *verbatim*, so it
is not editable; and the axis questions need a ``taxonomies:`` section, a
multi-child node and a grandchild before they can be asked at all. Both are the
same hand-edited file at two points in its life, which is what the two versions
say.

**Published too**, as the ``worked-input`` fence in
``packages/common/docs/guides/entity-resolution.md``, and held identical to
this constant by ``test_worked_input_fences.py`` beside this file. That matters more here
than for the v1.0 pair: ``retriever`` and ``golden_retriever`` are the two
overlapping forms the span and coverage assertions are *about*, and they are
asserted against this constant in the package suite and against the fence in
the workspace runner. A form added to one copy alone leaves both suites
agreeing on offsets that no longer describe one document.
"""


MAMMALS_GUIDE_DOCUMENT = """\
ontology:
  id: mammals
  version: "1.1"

  entity_types:
    - id: Species
      attributes:
        - {name: latin_name, type: string}
        - {name: lifespan_years, type: number, field_type: float}
    - id: Breed
      isa: Species                      # the TYPE lattice
      attributes:
        - {name: akc_group, type: string}

  relation_types:
    - id: isa
      transitive: true

  entities:
    - {id: mammal, type: Species, name: Mammal,
       description: "Warm-blooded, milk-producing vertebrates."}
    - {id: dog, type: Species, name: Dog, aliases: [Canine, "Domestic dog"],
       description: "A domesticated carnivoran."}
    - {id: retriever, type: Breed, name: Retriever}
    - {id: golden_retriever, type: Breed, name: Golden Retriever, aliases: [Goldie]}
    - {id: beagle, type: Breed, name: Beagle, aliases: [Beagles],
       source: {source_id: clinic_db, table: species, key: "sp-2291"}}

  assertions:
    - {subject: dog, relation: isa, object: mammal}
    - {subject: retriever, relation: isa, object: dog}
    - {subject: golden_retriever, relation: isa, object: retriever}
    - {subject: beagle, relation: isa, object: dog}

  taxonomies:
    - {id: species, name: Species, relation: isa}
"""
"""The v1.1 vocabulary as the *guides* publish it, which is a third document.

Nearest is :data:`MAMMALS_V11_DOCUMENT`, and the two differ substantively:
``latin_name`` is not required here, and the literal-object assertion
(``lifespan_years``) that one carries is absent. So this is not a drifted copy
of either other constant --- it is the document a reader of a published guide
copies to disk and runs the page against.

**It was a single published copy until a second page published it**, and that
is why it is here at all. ``anchored-view.md`` published it alone, guarded by
being *executed* rather than by being compared; ``content-tags.md`` publishes
the same vocabulary deliberately, so that a fourth service-free acceptance is a
property of the substrate rather than a coincidence. Two copies of one document
with nothing comparing them is the defect ``test_worked_input_fences.py`` was
written for, so the two fences are held to this constant and to each other.
"""


def _with_materialization(axis: str) -> str:
    """:data:`MAMMALS_V11_DOCUMENT` with one axis of the block flipped.

    Derived from the v1.1 document rather than written out again: the one thing
    under test is the ``materialization:`` block, and a second full copy of the
    document would let the two drift on everything else.

    One axis at a time, because each is refused *naming the axis* -- a document
    asking for both proves only that one of the two refusals fired first.
    """
    return MAMMALS_V11_DOCUMENT.replace(
        "    - {id: species, name: Species, relation: isa}\n",
        f"""\
    - id: species
      name: Species
      relation: isa
      materialization:
        {axis}: materialized
""",
    )


MATERIALIZED_CONTENT_DOCUMENT = _with_materialization("content")
"""The v1.1 vocabulary whose axis asks for a copy nothing here can hold."""

MATERIALIZED_STRUCTURE_DOCUMENT = _with_materialization("structure")
"""The v1.1 vocabulary whose axis asks for a snapshot nothing here builds."""


@dataclass(frozen=True)
class Sku:
    """A consumer's key: hashable, value-equal, and not a string.

    The smallest thing that is not a ``str`` and is still a legitimate entity
    key --- two fields, frozen, so it hashes and compares by value the way a
    node id must. Everything a non-``str`` vocabulary is needed to demonstrate
    is demonstrable over this: that ``qualify`` renders it, that ``localize``
    parses it back, and that a tag's ``node_id`` is a string while what a
    cursor takes is not.
    """

    plant: str
    line: int


class SkuCodec:
    """``Sku`` in one direction and back. Two functions, both the consumer's.

    Deliberately not a subclass of anything: the codec is a structural
    protocol, so a consumer satisfies it by having the two methods rather than
    by importing a base. A test fixture that inherited one would be asserting
    a requirement that does not exist.
    """

    def to_id(self, key: Sku, /) -> str:
        return f"{key.plant}/{key.line}"

    def from_id(self, rendered: str, /) -> Sku:
        plant, _, line = rendered.rpartition("/")
        return Sku(plant=plant, line=int(line))
