"""A transcription of the ``worked-bridge`` fence in ``common``'s resolution guide.

Everything below the end of this docstring is the fence, character for
character, and :mod:`tests.test_worked_entity_resolution_call_site` holds the
two identical in both directions.

**A second marker is a second program**, not a continuation of the first. The
guard runs each executed copy through ``runpy.run_path`` in its own namespace,
so nothing this file needs can be inherited from the call-site copy beside it
--- which is why the block builds its own cascade rather than reaching for one
the page bound earlier.
"""

import asyncio
from pathlib import Path

from dataknobs_common.entity_resolution import BridgedEntityResolver
from dataknobs_common.ontology import async_build_resolver, async_load_ontology


async def cascade():
    onto = await async_load_ontology(Path("mammals.yaml"))
    return await async_build_resolver(Path("mammals.yaml"), onto)


async_resolver = asyncio.run(cascade())

# Synchronous from here down, which is the point: no `await`, no loop of your
# own, and no rewriting the cascade you already have.
with BridgedEntityResolver(async_resolver) as bridged:
    result = bridged.resolve("beagles", k=5)

assert [c.entity_id for c in result.candidates] == ["beagle"]
assert [e.signal for e in result.explain("beagle")] == ["exact", "alias", "scan"]
