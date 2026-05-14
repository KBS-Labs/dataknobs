"""Postgres integration tests for the three sibling registries.

Pin the metadata-column routing contract on PostgreSQL for the
registries that are now reachable through the ``filter_metadata``
channel.  A consumer that needs tenant / audit / feature-flag
filtering on any of these registries gets end-to-end JSONB pushdown
without further changes.

Three sibling registries covered here:

* :class:`ArtifactRegistry` — ``query(filter_metadata=...)``
* :class:`RubricRegistry`   — ``get_for_target`` and ``list_all``
* :class:`GeneratorRegistry` — ``list_definitions(filter_metadata=...)``

Each test would have **failed** against earlier registry implementations
that built ``Record(data=...)`` inline and never populated the metadata
column; queries on ``metadata.X`` returned an empty list.

Skipped automatically when PostgreSQL is unavailable via
``@requires_postgres``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from dataknobs_bots.artifacts.registry import ArtifactRegistry
from dataknobs_bots.generators.base import (
    Generator,
    GeneratorContext,
    GeneratorOutput,
)
from dataknobs_bots.generators.registry import GeneratorRegistry
from dataknobs_bots.rubrics.models import (
    Rubric,
    RubricCriterion,
    RubricLevel,
    ScoringMethod,
    ScoringType,
)
from dataknobs_bots.rubrics.registry import RubricRegistry
from dataknobs_common.testing import requires_postgres
from dataknobs_data import AsyncDatabase
from dataknobs_data.backends.postgres import AsyncPostgresDatabase

pytestmark = requires_postgres


@pytest.fixture
async def postgres_db(make_postgres_test_db) -> AsyncGenerator[AsyncDatabase, None]:
    """Yield a clean ``AsyncPostgresDatabase`` per test.

    Uses the shared ``make_postgres_test_db`` factory fixture so the
    per-test table is unique and dropped on teardown.
    """
    for pg in make_postgres_test_db("test_sibling_registries_"):
        db = AsyncPostgresDatabase({
            "host": pg["host"],
            "port": pg["port"],
            "database": pg["database"],
            "user": pg["user"],
            "password": pg["password"],
            "table": pg["table"],
        })
        await db.connect()
        try:
            yield db
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# ArtifactRegistry on Postgres
# ---------------------------------------------------------------------------


class TestArtifactRegistryPostgres:
    """``ArtifactRegistry`` metadata routing against the JSONB column."""

    @pytest.mark.asyncio
    async def test_create_and_query_filter_metadata_postgres(
        self, postgres_db: AsyncDatabase,
    ) -> None:
        """``query(filter_metadata=...)`` is pushed into the JSONB column."""
        registry = ArtifactRegistry(postgres_db)

        await registry.create(
            artifact_type="content",
            name="Acme Doc",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )
        await registry.create(
            artifact_type="content",
            name="Globex Doc",
            content={"v": 1},
            metadata={"tenant_id": "globex"},
        )
        await registry.create(
            artifact_type="content",
            name="Acme Doc 2",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )

        acme = await registry.query(filter_metadata={"tenant_id": "acme"})
        assert sorted(a.name for a in acme) == ["Acme Doc", "Acme Doc 2"]

    @pytest.mark.asyncio
    async def test_combined_type_and_metadata_filter_postgres(
        self, postgres_db: AsyncDatabase,
    ) -> None:
        """``artifact_type`` (data column) AND-combines with ``filter_metadata`` (JSONB)."""
        registry = ArtifactRegistry(postgres_db)

        await registry.create(
            artifact_type="content",
            name="Acme Content",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )
        await registry.create(
            artifact_type="config",
            name="Acme Config",
            content={"v": 1},
            metadata={"tenant_id": "acme"},
        )
        await registry.create(
            artifact_type="content",
            name="Globex Content",
            content={"v": 1},
            metadata={"tenant_id": "globex"},
        )

        results = await registry.query(
            artifact_type="content",
            filter_metadata={"tenant_id": "acme"},
        )
        assert [a.name for a in results] == ["Acme Content"]

    @pytest.mark.asyncio
    async def test_round_trip_metadata_postgres(
        self, postgres_db: AsyncDatabase,
    ) -> None:
        """``create(..., metadata=...)`` round-trips through ``get()`` on PG."""
        registry = ArtifactRegistry(postgres_db)

        created = await registry.create(
            artifact_type="content",
            name="Round Trip",
            content={"v": 1},
            metadata={"tenant_id": "acme", "audit": {"by": "alice"}},
        )

        retrieved = await registry.get(created.id)
        assert retrieved is not None
        assert retrieved.metadata == {
            "tenant_id": "acme",
            "audit": {"by": "alice"},
        }


# ---------------------------------------------------------------------------
# RubricRegistry on Postgres
# ---------------------------------------------------------------------------


def _make_rubric(
    rubric_id: str = "rubric_001",
    version: str = "1.0.0",
    target_type: str = "content",
    metadata: dict[str, Any] | None = None,
) -> Rubric:
    """Helper: build a minimal valid Rubric for testing."""
    return Rubric(
        id=rubric_id,
        name=f"Rubric {rubric_id}",
        description="A test rubric",
        version=version,
        target_type=target_type,
        criteria=[
            RubricCriterion(
                id="c1",
                name="Criterion 1",
                description="Test criterion",
                weight=1.0,
                levels=[
                    RubricLevel(
                        id="fail", label="Fail", description="Fail", score=0.0
                    ),
                    RubricLevel(
                        id="pass", label="Pass", description="Pass", score=1.0
                    ),
                ],
                scoring_method=ScoringMethod(
                    type=ScoringType.DETERMINISTIC,
                    function_ref="test:func",
                ),
            ),
        ],
        pass_threshold=0.7,
        metadata=dict(metadata or {}),
    )


class TestRubricRegistryPostgres:
    """``RubricRegistry`` metadata routing against the JSONB column."""

    @pytest.mark.asyncio
    async def test_get_for_target_filter_metadata_postgres(
        self, postgres_db: AsyncDatabase,
    ) -> None:
        """``get_for_target`` AND-combines ``target_type`` with ``filter_metadata`` on PG."""
        registry = RubricRegistry(postgres_db)

        await registry.register(
            _make_rubric("r1", target_type="content", metadata={"tenant_id": "acme"})
        )
        await registry.register(
            _make_rubric("r2", target_type="content", metadata={"tenant_id": "globex"})
        )
        await registry.register(
            _make_rubric("r3", target_type="content", metadata={"tenant_id": "acme"})
        )

        acme = await registry.get_for_target(
            "content", filter_metadata={"tenant_id": "acme"}
        )
        assert {r.id for r in acme} == {"r1", "r3"}

    @pytest.mark.asyncio
    async def test_list_all_filter_metadata_postgres(
        self, postgres_db: AsyncDatabase,
    ) -> None:
        """``list_all`` honors ``filter_metadata`` against the JSONB column on PG."""
        registry = RubricRegistry(postgres_db)

        await registry.register(_make_rubric("r1", metadata={"tenant_id": "acme"}))
        await registry.register(_make_rubric("r2", metadata={"tenant_id": "globex"}))

        acme = await registry.list_all(filter_metadata={"tenant_id": "acme"})
        assert {r.id for r in acme} == {"r1"}

    @pytest.mark.asyncio
    async def test_round_trip_metadata_postgres(
        self, postgres_db: AsyncDatabase,
    ) -> None:
        """``register`` → ``get`` round-trips the metadata channel on PG."""
        registry = RubricRegistry(postgres_db)

        rubric = _make_rubric(
            "round-trip",
            metadata={"tenant_id": "acme", "author": "alice"},
        )
        await registry.register(rubric)

        retrieved = await registry.get("round-trip")
        assert retrieved is not None
        assert retrieved.metadata == {"tenant_id": "acme", "author": "alice"}


# ---------------------------------------------------------------------------
# GeneratorRegistry on Postgres
# ---------------------------------------------------------------------------


class _NamedGenerator(Generator):
    """Minimal generator stub used to test registry metadata flow.

    Generators normally carry code/templates and are not persisted; the
    registry stores a :class:`GeneratorDefinition` snapshot.  This stub
    is fine for testing the registry's persistence behavior.
    """

    def __init__(self, gen_id: str) -> None:
        self._id = gen_id

    @property
    def id(self) -> str:
        return self._id

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def parameter_schema(self) -> dict[str, Any]:
        return {"type": "object"}

    @property
    def output_schema(self) -> dict[str, Any]:
        return {"type": "object"}

    async def generate(
        self,
        parameters: dict[str, Any],
        context: GeneratorContext | None = None,
    ) -> GeneratorOutput:
        from dataknobs_bots.artifacts.provenance import create_provenance

        return GeneratorOutput(
            content={},
            provenance=create_provenance(
                created_by=f"system:generator:{self._id}",
                creation_method="generator",
            ),
        )


class TestGeneratorRegistryPostgres:
    """``GeneratorRegistry`` metadata routing against the JSONB column."""

    @pytest.mark.asyncio
    async def test_list_definitions_filter_metadata_postgres(
        self, postgres_db: AsyncDatabase,
    ) -> None:
        """``list_definitions(filter_metadata=...)`` reaches the JSONB column on PG."""
        registry = GeneratorRegistry(postgres_db)

        await registry.register(_NamedGenerator("gen_a"), metadata={"tenant_id": "acme"})
        await registry.register(_NamedGenerator("gen_b"), metadata={"tenant_id": "globex"})

        acme = await registry.list_definitions(filter_metadata={"tenant_id": "acme"})
        assert {d.generator_id for d in acme} == {"gen_a"}

        globex = await registry.list_definitions(filter_metadata={"tenant_id": "globex"})
        assert {d.generator_id for d in globex} == {"gen_b"}

    @pytest.mark.asyncio
    async def test_get_definition_round_trip_postgres(
        self, postgres_db: AsyncDatabase,
    ) -> None:
        """``register(..., metadata=...)`` round-trips through ``get_definition`` on PG.

        Also closes the historical shadow-bug at this site: a local
        variable named ``metadata`` was passed positionally to
        ``Record(metadata)`` and silently routed into the data column
        under a misleading name.  The migration to
        ``AsyncKeyedRecordStore`` makes this defect structurally
        impossible — the metadata channel is part of the serializer's
        type signature.
        """
        registry = GeneratorRegistry(postgres_db)

        await registry.register(
            _NamedGenerator("gen_x"),
            metadata={"tenant_id": "acme", "feature_flag": True},
        )

        defn = await registry.get_definition("gen_x")
        assert defn is not None
        assert defn.metadata == {"tenant_id": "acme", "feature_flag": True}

    @pytest.mark.asyncio
    async def test_empty_filter_metadata_is_no_filter_postgres(
        self, postgres_db: AsyncDatabase,
    ) -> None:
        """``filter_metadata={}`` is equivalent to ``filter_metadata=None`` on PG."""
        registry = GeneratorRegistry(postgres_db)

        await registry.register(_NamedGenerator("gen_a"), metadata={"tenant_id": "acme"})
        await registry.register(_NamedGenerator("gen_b"))

        with_empty = await registry.list_definitions(filter_metadata={})
        with_none = await registry.list_definitions()
        assert {d.generator_id for d in with_empty} == {
            d.generator_id for d in with_none
        }
        assert len(with_empty) == 2
