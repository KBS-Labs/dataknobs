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

#: Where a ``PluginRegistry`` reports its own defaulting, which is a
#: different decision made by different code -- see
#: :class:`TestARegistryDefaultThatCostsNothing`.
REGISTRY_LOGGER = "dataknobs_common.registry"


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == SELECTION_LOGGER
    ]


def _wizard():
    """The minimal wizard the bank-db tests need, as ``test_sqlite_backend``
    builds one -- through ``WizardConfigLoader``, so the FSM is real.
    """
    from dataknobs_bots.reasoning.wizard import WizardReasoning
    from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

    fsm = WizardConfigLoader().load_from_dict(
        {
            "name": "test-wizard",
            "version": "1.0",
            "settings": {},
            "stages": [{"name": "start", "is_start": True, "is_end": True, "prompt": "test"}],
        }
    )
    return WizardReasoning(wizard_fsm=fsm, strict_validation=False)


def _registry_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == REGISTRY_LOGGER
    ]


class TestARegistryDefaultThatCostsNothing:
    """The other half: not every omitted key is worth a WARNING.

    ``dataknobs_data``'s backend default is consequential -- an unpersisted
    store answering every query with zero results -- so it is reported. The
    registries in this package default to the *recommended* answer, and
    every documented config for them omits the key on purpose:
    ``reasoning:`` without ``strategy:`` means ``simple``, ``memory:``
    without ``type:`` means ``buffer``, ``knowledge_base:`` without
    ``type:`` means ``rag``. Each example in this package's own docs is
    written that way.

    Reporting those at WARNING fires on correct usage -- per turn, where the
    resolve is per turn -- and drowns the reports that mean something.
    """

    def test_a_reasoning_config_naming_no_strategy_is_quiet(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from dataknobs_bots.reasoning.registry import get_registry

        with caplog.at_level(logging.DEBUG, logger=REGISTRY_LOGGER):
            get_registry().create(config={})

        assert _registry_warnings(caplog) == []

    def test_a_memory_config_naming_no_type_is_quiet(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from dataknobs_bots.memory.registry import memory_backends

        with caplog.at_level(logging.DEBUG, logger=REGISTRY_LOGGER):
            memory_backends.create(config={})

        assert _registry_warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_a_knowledge_base_config_naming_no_type_is_quiet(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from dataknobs_bots.knowledge.registry import knowledge_base_backends

        with caplog.at_level(logging.DEBUG, logger=REGISTRY_LOGGER):
            await knowledge_base_backends.create_async(config={})

        assert _registry_warnings(caplog) == []

    def test_the_fallback_is_still_recorded_at_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Quieter, not silent. The provenance is still there to find."""
        from dataknobs_bots.reasoning.registry import get_registry

        with caplog.at_level(logging.DEBUG, logger=REGISTRY_LOGGER):
            get_registry().create(config={})

        debug = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.DEBUG and record.name == REGISTRY_LOGGER
        ]
        assert any("simple" in message and "strategy" in message for message in debug), (
            f"expected the defaulted strategy recorded at DEBUG, got {debug}"
        )


class TestABankConfigThatNamesNoBackend:
    """The sites that branch *before* a factory, and why they stay quiet.

    A wizard memory bank naming no backend is asking for
    conversation-scoped storage: the documented default and the
    recommended answer, not the "a config arrived empty" case the
    factory's WARNING exists for. Nothing reaches a factory on that
    branch, so there is no provenance to lose either -- these sites read
    the key to decide whether to build a database at all.

    Pinned rather than left implicit, because the same shape one frame
    further along *was* a defect: the three sites in the classes below
    passed a laundered default into a factory, where the distinction was
    real and reportable. What separates them is whether a factory is
    involved, and that is worth a test rather than a reader's inference.
    """

    def test_it_gets_in_process_storage(self) -> None:
        from dataknobs_data.backends.memory import SyncMemoryDatabase

        db, mode = _wizard()._create_bank_db("notes", {})

        assert isinstance(db, SyncMemoryDatabase)
        assert mode == "inline"

    def test_it_reports_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG):
            _wizard()._create_bank_db("notes", {})

        assert _warnings(caplog) == []
        assert _registry_warnings(caplog) == []

    def test_a_bank_naming_a_real_backend_still_builds_one(self, tmp_path) -> None:
        """The other branch, so the predicate swap is covered both ways."""
        db, _mode = _wizard()._create_bank_db(
            "notes",
            {"backend": "file", "backend_config": {"path": str(tmp_path / "notes.json")}},
        )

        assert type(db).__name__ == "SyncFileDatabase"

    def test_a_bank_naming_an_unusable_backend_says_so(self) -> None:
        """``.get("backend", "memory")`` let ``None`` through to a factory."""
        with pytest.raises(ValueError, match="present but null"):
            _wizard()._create_bank_db("notes", {"backend": None})


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


class TestEverySpellingOfTheDefault:
    """``mem`` is ``memory``; a caller branching on the answer must agree.

    ``_create_bank_db`` routes the default to an inline ``SyncMemoryDatabase``
    and everything else through a factory, which also connects the database
    and gives the bank a table -- a different storage mode. Comparing the
    config's value against the literal ``"memory"`` made that choice depend
    on which spelling the author used for a backend the factory resolves to
    one class either way.
    """

    @pytest.mark.parametrize("spelling", ["memory", "mem", "MEM", "  Mem  "])
    def test_it_routes_to_inline_storage(self, spelling: str) -> None:
        from dataknobs_data.backends.memory import SyncMemoryDatabase

        db, mode = _wizard()._create_bank_db("notes", {"backend": spelling})

        assert isinstance(db, SyncMemoryDatabase)
        assert mode == "inline", f"{spelling!r} took the factory branch"

    def test_a_genuinely_different_backend_still_does_not(self, tmp_path) -> None:
        _db, mode = _wizard()._create_bank_db(
            "notes",
            {"backend": "file", "backend_config": {"path": str(tmp_path / "n.json")}},
        )

        assert mode == "external"


class TestAVectorMemoryThatNamedNoBackend:
    """The same laundering, in the typed-dataclass spelling.

    ``VectorMemoryConfig.backend`` defaulted to ``"memory"`` and was written
    into the dict handed to ``VectorStoreFactory``, so a config that named
    nothing reached the factory as a choice. An unpersisted vector store
    loses every embedding on restart, which is exactly the consequence the
    factory's report exists to name.
    """

    @pytest.mark.asyncio
    async def test_an_absent_backend_is_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        from dataknobs_bots.memory.config import VectorMemoryConfig
        from dataknobs_bots.memory.vector import VectorMemory

        config = VectorMemoryConfig.from_dict({"dimension": 8})
        with caplog.at_level(logging.DEBUG, logger=SELECTION_LOGGER):
            await VectorMemory.from_config_async(config)

        assert len(_warnings(caplog)) == 1

    @pytest.mark.asyncio
    async def test_a_named_backend_is_not(self, caplog: pytest.LogCaptureFixture) -> None:
        from dataknobs_bots.memory.config import VectorMemoryConfig
        from dataknobs_bots.memory.vector import VectorMemory

        config = VectorMemoryConfig.from_dict({"dimension": 8, "backend": "memory"})
        with caplog.at_level(logging.DEBUG, logger=SELECTION_LOGGER):
            await VectorMemory.from_config_async(config)

        assert _warnings(caplog) == []
