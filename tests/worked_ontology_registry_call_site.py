"""A transcription of the ``worked-call-site`` fence in ``data``'s registry guide.

Everything below the end of this docstring is the fence, character for
character, and :mod:`tests.test_worked_ontology_registry_call_site` holds the
two identical in both directions. Edit one and copy it to the other; never one
alone.
"""

import asyncio
from typing import Any

from dataknobs_common.entity_resolution import ResolutionResult
from dataknobs_common.ontology import (
    NODE_ID_KEY,
    ONTOLOGY_ID_KEY,
    TAXONOMY_ID_KEY,
    OntologyConfig,
    read_node_tags_many,
)
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

# A second vocabulary, loaded into the same registry. One is the case where the
# question below has a single answer and nobody has to ask it -- the registry
# instance is the unit of sharing, so a deployment holding several holds them
# here, in one flat id space.
suppliers = {
    "id": "suppliers",
    "version": "1.0",
    "entity_types": [{"id": "Supplier"}],
    "entities": [{"id": "acme-co", "type": "Supplier", "name": "Acme Corporation"}],
}

# What your own retrieval handed back. This page wrote none of it: rows out of
# somebody else's store, carrying whatever tags were written on them. Row 1
# touched no vocabulary at all, which most of a corpus does.
rows: list[dict[str, Any]] = [
    {ONTOLOGY_ID_KEY: "catalog", TAXONOMY_ID_KEY: "kinds", NODE_ID_KEY: ["sku-4471"]},
    {"invoice_id": "2291"},
    {ONTOLOGY_ID_KEY: "suppliers", TAXONOMY_ID_KEY: "trade", NODE_ID_KEY: ["acme-co"]},
    {ONTOLOGY_ID_KEY: "catalog", TAXONOMY_ID_KEY: "kinds", NODE_ID_KEY: ["sku-8802"]},
]


async def main() -> ResolutionResult:
    registry = OntologyRegistry.from_components(
        config=OntologyConfig.from_dict(document),
        embedder=DeterministicEmbedder(dimensions=32),
    )
    await registry.load()
    await registry.load(suppliers)
    try:
        # Which of the vocabularies this registry holds is that page of results
        # even about? Ranked by how many rows named each, so the answer is also
        # a measure of how much of it, and the rows ride along so that picking
        # one costs no second pass over the corpus. The axis id on each row is
        # not read here -- rolling a corpus up onto one axis is what needs it.
        in_play = registry.ontologies_in_play(read_node_tags_many(rows).tags)
        assert [(s.ontology_id, s.rows) for s in in_play] == [
            ("catalog", (0, 3)),
            ("suppliers", (2,)),
        ]
        about = in_play[0].ontology_id  # "catalog", on two rows against one

        # `load()` assembles the index; filling it is yours. Nothing here
        # re-embeds a vocabulary on every reload behind your back.
        index = registry.index(about)
        assert index is not None  # `None` where a document declares no `index:`
        await index.build()

        resolver = registry.resolver(about)
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
