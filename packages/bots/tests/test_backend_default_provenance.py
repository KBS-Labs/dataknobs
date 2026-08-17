"""A config that named no backend should still be able to say so.

``dataknobs_data`` reports an absent ``backend`` key at WARNING, because a
config that arrives empty and a config asking for an in-process store build
the same object and are not the same event. That report is only reachable
if the absence survives the trip to the factory.

It did not. Every caller in this package read the key with its own
``"memory"`` default and wrote the result back into the dict it passed
down, so the factory saw an explicit choice and logged INFO. The absence
was consumed one frame above the only code positioned to report it -- and
the case that produces an empty config in the first place is a ``$resource``
reference to a resource the environment does not define, which is exactly
what the WARNING text tells the reader to go and check.

These pin the provenance, not the object: every case below builds the same
in-process store either way.
"""

from __future__ import annotations

import logging

import pytest


#: The module that decides the backend, and therefore the only one that can
#: report having guessed. Named here rather than asserted against "any
#: logger" so an unrelated WARNING elsewhere cannot satisfy these.
SELECTION_LOGGER = "dataknobs_data.backend_selection"


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == SELECTION_LOGGER
    ]


class TestArtifactBankCatalog:
    def test_an_empty_config_reports_the_fallback(self, caplog: pytest.LogCaptureFixture) -> None:
        from dataknobs_bots.memory.catalog import ArtifactBankCatalog

        with caplog.at_level(logging.DEBUG, logger=SELECTION_LOGGER):
            ArtifactBankCatalog.from_config({})

        assert len(_warnings(caplog)) == 1
        assert "falling back to 'memory'" in _warnings(caplog)[0]

    def test_an_explicit_backend_does_not(self, caplog: pytest.LogCaptureFixture) -> None:
        from dataknobs_bots.memory.catalog import ArtifactBankCatalog

        with caplog.at_level(logging.DEBUG, logger=SELECTION_LOGGER):
            ArtifactBankCatalog.from_config({"backend": "memory"})

        assert _warnings(caplog) == []

    def test_a_backend_named_inside_backend_config_is_not_overwritten(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The outer default used to clobber an inner explicit choice.

        ``backend_config["backend"] = config.get("backend", "memory")``
        overwrote whatever ``backend_config`` already said, so a config
        naming its backend in the inner dict silently got the outer
        default instead.
        """
        from dataknobs_bots.memory.catalog import ArtifactBankCatalog

        with caplog.at_level(logging.DEBUG, logger=SELECTION_LOGGER):
            ArtifactBankCatalog.from_config({"backend_config": {"backend": "memory"}})

        assert _warnings(caplog) == []


class TestRegistryAdapter:
    @pytest.mark.asyncio
    async def test_an_empty_config_reports_the_fallback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from dataknobs_bots.registry.adapter import DataKnobsRegistryAdapter

        adapter = DataKnobsRegistryAdapter.from_config({})
        with caplog.at_level(logging.DEBUG, logger=SELECTION_LOGGER):
            await adapter.initialize()

        assert len(_warnings(caplog)) == 1

    @pytest.mark.asyncio
    async def test_an_explicit_backend_does_not(self, caplog: pytest.LogCaptureFixture) -> None:
        from dataknobs_bots.registry.adapter import DataKnobsRegistryAdapter

        adapter = DataKnobsRegistryAdapter.from_config({"backend": "memory"})
        with caplog.at_level(logging.DEBUG, logger=SELECTION_LOGGER):
            await adapter.initialize()

        assert _warnings(caplog) == []


class TestDatabaseGroundedSource:
    @pytest.mark.asyncio
    async def test_options_naming_no_backend_report_the_fallback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from dataknobs_bots.knowledge.sources.factory import _create_database_source
        from dataknobs_bots.reasoning.grounded_config import GroundedSourceConfig

        config = GroundedSourceConfig(
            name="docs",
            source_type="database",
            options={"schema": {"fields": {"body": {"type": "string"}}}},
        )

        with caplog.at_level(logging.DEBUG, logger=SELECTION_LOGGER):
            await _create_database_source(config)

        assert len(_warnings(caplog)) == 1

    @pytest.mark.asyncio
    async def test_options_naming_a_backend_do_not(self, caplog: pytest.LogCaptureFixture) -> None:
        from dataknobs_bots.knowledge.sources.factory import _create_database_source
        from dataknobs_bots.reasoning.grounded_config import GroundedSourceConfig

        config = GroundedSourceConfig(
            name="docs",
            source_type="database",
            options={
                "backend": "memory",
                "schema": {"fields": {"body": {"type": "string"}}},
            },
        )

        with caplog.at_level(logging.DEBUG, logger=SELECTION_LOGGER):
            await _create_database_source(config)

        assert _warnings(caplog) == []
