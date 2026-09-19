"""A transcription of the ``worked-call-site`` fence in ``data``'s registry guide.

Everything below the end of this docstring is the fence, character for
character, and :mod:`tests.test_worked_ontology_registry_call_site` holds the
two identical in both directions. Edit one and copy it to the other; never one
alone.
"""

import asyncio

from dataknobs_common.entity_resolution import ResolutionResult
from dataknobs_common.ontology import OntologyConfig
from dataknobs_data.ontology import OntologyRegistry
from dataknobs_data.testing import DeterministicEmbedder

document = {
    "id": "catalog",
    "version": "1.0",
    "entity_types": [{"id": "Product"}, {"id": "Brand"}],
    "entities": [
        {
            "id": "sku-4471",
            "type": "Product",
            "name": "Acme Widget",
            "description": "a widget",
            "aliases": ["Widget"],
        },
        {"id": "sku-8802", "type": "Product", "name": "Bolt", "description": "a threaded fastener"},
        {
            "id": "acme",
            "type": "Brand",
            "name": "Acme Corp",
            "description": "the maker of the widget",
        },
    ],
    "index": {
        "store": {"backend": "memory", "dimensions": 32},
        "fields": ["name", "description"],
    },
    "resolver": {"rungs": [{"kind": "exact"}, {"kind": "semantic"}]},
}


async def main() -> ResolutionResult:
    registry = OntologyRegistry.from_components(
        config=OntologyConfig.from_dict(document),
        embedder=DeterministicEmbedder(dimensions=32),
    )
    await registry.load()
    try:
        # `load()` assembles the index; filling it is yours. Nothing here
        # re-embeds a vocabulary on every reload behind your back.
        index = registry.index("catalog")
        assert index is not None  # `None` where a document declares no `index:`
        await index.build()

        resolver = registry.resolver("catalog")
        assert resolver is not None  # and `None` where it declares none
        result = await resolver.resolve("Acme Widget", k=5)

        best = result.candidates[0]
        assert best.entity_id == "sku-4471"  # local, not "catalog:sku-4471"
        assert best.declared  # a form the vocabulary carries
        assert [e.signal for e in result.explain("sku-4471")] == ["exact", "semantic"]

        tail = result.candidates[-1]
        assert not tail.declared  # only the embedding proposed it
        assert [e.signal for e in tail.evidence] == ["semantic"]

        return result
    finally:
        await registry.close()


result = asyncio.run(main())
