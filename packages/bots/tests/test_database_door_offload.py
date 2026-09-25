"""Both ``bots`` doors that build a database from config build it off the loop.

A grounded ``database`` source and the registry adapter each turn a config
into a database inside an ``async def``. Resolving a backend name imports the
backend implementation through ``PluginRegistry``'s ``on_first_access`` hook,
which reads a module off disk, and a file backend's config also normalizes its
path while it is built. The ``data`` doors that do the same --
``AsyncUserStateStore.from_config_async`` and the ontology registry's -- carry
the offload already; these two did not.

The harm is the ordinary one: a multi-tenant server building a bot for one
request stalls every other task on that loop for the duration. Cheap and once
per process is why it goes unnoticed, not why it is allowed.

The bracket covers the whole door, because the build happens inside the
awaited call. Against the un-offloaded code these FAIL with
``blockbuster.BlockingError``; with the offload they PASS.

**They prove it only against a cold backend registry.** The import happens
once per process, and measured on this tree neither door blocks once both
backends have been resolved -- the sqlite path work included. So in a process
where an earlier test already built a memory or sqlite database, these pass
whether or not the build is offloaded. Run alone, they are the reproduce-first
pair; in a full run they are not a regression guard. The ``data`` offload
suites this file is modelled on share that limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from dataknobs_common.testing import assert_no_blocking, requires_blockbuster, requires_package

from dataknobs_bots.knowledge.sources.factory import _create_database_source
from dataknobs_bots.reasoning.grounded_config import GroundedSourceConfig
from dataknobs_bots.registry.adapter import DataKnobsRegistryAdapter
from dataknobs_data import Record
from dataknobs_data.sources.base import RetrievalIntent

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.asyncio]

FIELDS = {"title": "string", "summary": "text"}


def _source_config(**options: Any) -> GroundedSourceConfig:
    return GroundedSourceConfig(name="case_studies", source_type="database", options=options)


# --------------------------------------------------------------------------
# The grounded ``database`` source
# --------------------------------------------------------------------------


@requires_blockbuster
async def test_a_grounded_source_resolves_its_backend_off_the_loop() -> None:
    with assert_no_blocking():
        source = await _create_database_source(
            _source_config(backend="memory", schema={"fields": FIELDS})
        )
    await source.close()


@requires_blockbuster
@requires_package("aiosqlite")
async def test_a_grounded_file_source_is_neither_resolved_nor_opened_on_the_loop(
    tmp_path: Path,
) -> None:
    """The backend whose config also normalizes a path, and which really connects."""
    with assert_no_blocking():
        source = await _create_database_source(
            _source_config(
                backend="sqlite",
                path=str(tmp_path / "cases.db"),
                table="cases",
                content_field="summary",
                schema={"fields": FIELDS},
            )
        )

    try:
        await source._db.create(Record({"title": "Widget recall", "summary": "Recalled."}))
        results = await source.query(RetrievalIntent(text_queries=[]))
        assert [r.content for r in results] == ["Recalled."]
    finally:
        await source.close()


async def test_a_refused_backend_option_still_names_the_source() -> None:
    """The offload keeps the wrapper that says which source a refusal came from.

    ``asyncio.to_thread`` re-raises in the awaiting task, so the ``except``
    around it still sees the factory's error. A bot config can declare several
    sources, and the key alone does not say which one was wrong.
    """
    with pytest.raises(ValueError, match=r"^Source 'case_studies': .*content_fields"):
        await _create_database_source(_source_config(backend="memory", content_fields="summary"))


# --------------------------------------------------------------------------
# The registry adapter
# --------------------------------------------------------------------------


@requires_blockbuster
async def test_the_registry_adapter_resolves_its_backend_off_the_loop() -> None:
    adapter = DataKnobsRegistryAdapter(backend_type="memory")
    with assert_no_blocking():
        await adapter.initialize()
    await adapter.close()


@requires_blockbuster
@requires_package("aiosqlite")
async def test_the_registry_adapter_neither_resolves_nor_opens_a_file_on_the_loop(
    tmp_path: Path,
) -> None:
    adapter = DataKnobsRegistryAdapter(
        backend_type="sqlite", backend_config={"path": str(tmp_path / "registry.db")}
    )
    with assert_no_blocking():
        await adapter.initialize()

    try:
        await adapter.register("bot-1", {"llm": {"provider": "echo"}})
        registration = await adapter.get("bot-1")
        assert registration is not None
        assert registration.config == {"llm": {"provider": "echo"}}
    finally:
        await adapter.close()
