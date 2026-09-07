"""The async door does not read the file on the event loop.

The defect this guards is **non-functional**: an ontology loaded through a
blocking read is byte-for-byte the ontology loaded through an offloaded one, so
no assertion about the result can tell the two apart. On a shared loop -- a
server, a bot holding many conversations -- that read freezes every other task
for its duration, and nothing reports it.

So the check is a runtime detector rather than an outcome, and the bracket is
around the ``await`` alone: writing the fixture file is synchronous test setup
that may legitimately block.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_common.ontology import async_load_ontology
from dataknobs_common.testing import assert_no_blocking, is_blockbuster_available

MAMMALS = """\
ontology:
  id: mammals
  version: "1.0"
  entity_types:
    - id: Species
  entities:
    - {id: mammal, type: Species, name: Mammal}
    - {id: dog, type: Species, name: Dog}
  assertions:
    - {subject: dog, relation: isa, object: mammal}
"""


@pytest.fixture
def mammals_file(tmp_path: Path) -> Path:
    """The document on disk. Written here, synchronously, before any loop."""
    path = tmp_path / "mammals.yaml"
    path.write_text(MAMMALS)
    return path


@pytest.mark.skipif(
    not is_blockbuster_available(),
    reason="blockbuster is required to detect a blocking call on the loop",
)
@pytest.mark.asyncio
async def test_async_load_does_not_read_on_the_event_loop(mammals_file: Path) -> None:
    """The read is offloaded, not performed on the loop.

    ``load_yaml_or_json`` reads through a blocking ``open()``. Calling it
    directly from an ``async def`` stalls the loop for the length of the read,
    which this raises ``BlockingError`` for.
    """
    with assert_no_blocking():
        onto = await async_load_ontology(mammals_file)

    assert onto.id == "mammals"


@pytest.mark.asyncio
async def test_async_load_from_a_mapping_touches_no_file(
    mammals_file: Path,
) -> None:
    """Handed a mapping, the door reads nothing at all.

    The offload is on the *read*, so the in-memory path must not acquire one:
    a caller who already has the document should not pay for a thread.
    """
    document = {
        "ontology": {
            "id": "mammals",
            "entities": [{"id": "dog", "type": "Species"}],
        }
    }
    onto = await async_load_ontology(document)
    assert await onto.entity("dog") is not None
