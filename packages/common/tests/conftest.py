"""Shared fixtures for ``dataknobs-common`` tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import pytest

from dataknobs_common.testing import live_dk_daemon_threads


@pytest.fixture
def new_dk_daemon_threads() -> Iterator[Callable[..., list[str]]]:
    """Report dataknobs daemon threads *this test* created and left alive.

    Thread assertions in this package used to compare against an absolute
    zero, which quietly made them a report on the whole process: a thread
    leaked by any other test in a multi-package run turned them red and
    named the wrong file as the culprit. Measuring against a per-test
    baseline scopes each assertion to the test that owns it.

    The baseline is captured over *every* watched name, so the returned
    callable can narrow to one name per call without needing a matching
    baseline per name::

        def test_something(new_dk_daemon_threads):
            bridge = SyncLoopBridge()
            assert new_dk_daemon_threads(DK_SYNC_BRIDGE_THREAD)
            bridge.close()
            assert new_dk_daemon_threads(DK_SYNC_BRIDGE_THREAD) == []

    Lives here rather than in each test module because the same eight-line
    idiom had been copied into three of them — which is the duplication
    ``dataknobs_common.testing.threads`` was extracted to end. Prefer
    ``assert_no_leaked_bridge_threads`` when a whole block should leak
    nothing; reach for this only when a test needs to assert *mid-run* that
    a thread does or does not exist.
    """
    baseline = set(live_dk_daemon_threads())

    def _still_alive(names: Iterable[str] | str | None = None) -> list[str]:
        watched = [names] if isinstance(names, str) else names
        return sorted(t.name for t in live_dk_daemon_threads(watched) if t not in baseline)

    yield _still_alive


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
"""


@pytest.fixture
def mammals_path(tmp_path: Path) -> Path:
    """:data:`MAMMALS_DOCUMENT` written to disk."""
    path = tmp_path / "mammals.yaml"
    path.write_text(MAMMALS_DOCUMENT)
    return path


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
"""


@pytest.fixture
def mammals_v11_path(tmp_path: Path) -> Path:
    """:data:`MAMMALS_V11_DOCUMENT` written to disk."""
    path = tmp_path / "mammals.yaml"
    path.write_text(MAMMALS_V11_DOCUMENT)
    return path


MATERIALIZED_CONTENT_DOCUMENT = MAMMALS_V11_DOCUMENT.replace(
    "    - {id: species, name: Species, relation: isa}\n",
    """\
    - id: species
      name: Species
      relation: isa
      materialization:
        structure: materialized
        content: materialized
""",
)
"""The v1.1 vocabulary whose axis asks for a copy nothing here can hold.

Derived from :data:`MAMMALS_V11_DOCUMENT` rather than written out again: the
one thing under test is the ``materialization:`` block, and a second full copy
of the document would let the two drift on everything else.
"""


@pytest.fixture
def materialized_content_path(tmp_path: Path) -> Path:
    """:data:`MATERIALIZED_CONTENT_DOCUMENT` written to disk."""
    path = tmp_path / "mammals.yaml"
    path.write_text(MATERIALIZED_CONTENT_DOCUMENT)
    return path
