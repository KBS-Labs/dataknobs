"""What the bots subsystem helpers show of a backend factory's failure.

``create_memory_from_config``, ``create_knowledge_base_from_config`` and
``create_source_from_config`` used to unwrap any ``ValueError`` or
``DataknobsError`` off the registry's bounded error and raise it as itself, so
a backend's text -- the connection URL it was handed, credentials included --
reached the caller, and through the bots API the HTTP response.

Two halves are pinned here. A backend's own words stay bounded. And
dataknobs' own diagnostics, which are most of what a config author sees,
still arrive word for word: they are raised as deep as ``dataknobs_data`` and
``dataknobs_llm``, and each is marked where it is raised.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from dataknobs_bots.knowledge import create_knowledge_base_from_config, knowledge_base_backends
from dataknobs_bots.knowledge.sources.factory import create_source_from_config, source_backends
from dataknobs_bots.memory import create_memory_from_config, memory_backends
from dataknobs_bots.reasoning.grounded_config import GroundedSourceConfig
from dataknobs_common.exceptions import (
    ConfigurationError,
    DottedPathError,
    FactoryValueError,
    NotFoundError,
    ResourceError,
    TimeoutError,
    ValidationError,
)
from dataknobs_common.registry import PluginRegistry

DSN = "postgres://admin:hunter2@db.internal:5432/prod"


async def _build(subsystem: str, key: str) -> Any:
    if subsystem == "memory":
        return await create_memory_from_config({"type": key})
    if subsystem == "knowledge_base":
        return await create_knowledge_base_from_config({"type": key})
    return await create_source_from_config(
        GroundedSourceConfig(source_type=key, name="cases", options={})
    )


_REGISTRIES: dict[str, PluginRegistry[Any]] = {
    "memory": memory_backends,
    "knowledge_base": knowledge_base_backends,
    "source": source_backends,
}
_SUBJECTS = {
    "memory": "memory backend",
    "knowledge_base": "knowledge base",
    "source": "grounded source",
}


@pytest.fixture
def registered() -> Iterator[Callable[[str, BaseException], str]]:
    added: list[tuple[PluginRegistry[Any], str]] = []

    def register(subsystem: str, error: BaseException) -> str:
        def factory(config: Any, **_: Any) -> Any:
            raise error

        registry = _REGISTRIES[subsystem]
        registry.register("test_leaky", factory)
        added.append((registry, "test_leaky"))
        return "test_leaky"

    yield register
    for registry, key in added:
        registry.unregister(key)


@pytest.mark.parametrize("subsystem", sorted(_REGISTRIES))
class TestABackendsOwnWordsStayBounded:
    async def test_a_value_error(self, subsystem: str, registered: Any) -> None:
        key = registered(subsystem, ValueError(f"cannot connect to {DSN}"))

        with pytest.raises(ValueError) as caught:
            await _build(subsystem, key)

        assert isinstance(caught.value, FactoryValueError)
        assert DSN not in str(caught.value)
        assert DSN not in repr(caught.value.context)
        assert str(caught.value) == f"Failed to create {_SUBJECTS[subsystem]} '{key}' (ValueError)"
        assert DSN in str(caught.value.__cause__)

    async def test_a_validation_error(self, subsystem: str, registered: Any) -> None:
        """It rendered as a 422 with the DSN in the body; it is still a 422, bounded."""
        key = registered(subsystem, ValidationError(f"refused {DSN}", context={"url": DSN}))

        with pytest.raises(FactoryValueError) as caught:
            await _build(subsystem, key)

        assert caught.value.context["cause_type"] == "ValidationError"
        assert DSN not in str(caught.value)
        assert DSN not in repr(caught.value.context)

    @pytest.mark.parametrize("family", [ResourceError, TimeoutError])
    async def test_a_connection_failure_keeps_its_family(
        self, subsystem: str, registered: Any, family: type
    ) -> None:
        key = registered(subsystem, family(f"unreachable: {DSN}", context={"url": DSN}))

        with pytest.raises(family) as caught:
            await _build(subsystem, key)

        assert type(caught.value) is family
        assert DSN not in str(caught.value)
        assert DSN not in repr(caught.value.context)

    async def test_a_backends_own_not_found_error_is_not_an_unknown_type(
        self, subsystem: str, registered: Any
    ) -> None:
        """The helper's "Unknown … type" is for the registry's miss, not a backend's."""
        key = registered(subsystem, NotFoundError(f"no table at {DSN}"))

        with pytest.raises(NotFoundError) as caught:
            await _build(subsystem, key)

        assert DSN not in str(caught.value)
        assert "Unknown" not in str(caught.value)


# ---------------------------------------------------------------------------
# The census: dataknobs' own diagnostics, word for word
# ---------------------------------------------------------------------------


class TestDataknobsOwnDiagnosticsKeepTheirWords:
    async def test_summary_memory_with_no_llm(self) -> None:
        with pytest.raises(ValueError) as caught:
            await create_memory_from_config({"type": "summary"})
        assert str(caught.value).startswith(
            "Summary memory requires an LLM provider. Either include an 'llm' section"
        )
        assert not isinstance(caught.value, FactoryValueError)

    async def test_composite_memory_with_no_strategies(self) -> None:
        with pytest.raises(ValueError) as caught:
            await create_memory_from_config({"type": "composite", "strategies": []})
        assert str(caught.value).startswith("Composite memory requires at least one strategy")

    async def test_composite_memory_with_an_unknown_child(self) -> None:
        with pytest.raises(ValueError) as caught:
            await create_memory_from_config({"type": "composite", "strategies": [{"type": "nope"}]})
        assert str(caught.value).startswith("Unknown memory type: nope. Available types: ")

    async def test_an_unknown_memory_type(self) -> None:
        with pytest.raises(ValueError, match=r"^Unknown memory type: nope\. Available types: "):
            await create_memory_from_config({"type": "nope"})

    async def test_a_rag_knowledge_base_with_an_unknown_vector_backend(self) -> None:
        with pytest.raises(ValueError) as caught:
            await create_knowledge_base_from_config(
                {
                    "type": "rag",
                    "vector_store": {"backend": "nope", "dimensions": 3},
                    "embedding_provider": "echo",
                    "embedding_model": "test",
                }
            )
        assert str(caught.value).startswith("Unknown backend type: nope. Available backends: ")

    async def test_a_rag_knowledge_base_with_an_unknown_embedding_provider(self) -> None:
        with pytest.raises(ValueError) as caught:
            await create_knowledge_base_from_config(
                {
                    "type": "rag",
                    "vector_store": {"backend": "memory", "dimensions": 3},
                    "embedding_provider": "nope",
                    "embedding_model": "test",
                }
            )
        assert str(caught.value).startswith("Unknown provider: nope. Available providers: ")

    async def test_a_vector_kb_source_with_no_knowledge_base(self) -> None:
        with pytest.raises(ValueError) as caught:
            await create_source_from_config(
                GroundedSourceConfig(source_type="vector_kb", name="docs", options={})
            )
        assert str(caught.value).startswith(
            "Source 'docs' has type 'vector_kb' but no knowledge_base was provided."
        )

    async def test_a_database_source_with_an_unknown_backend(self) -> None:
        with pytest.raises(ValueError) as caught:
            await create_source_from_config(
                GroundedSourceConfig(
                    source_type="database", name="cases", options={"backend": "nope"}
                )
            )
        assert str(caught.value).startswith("Source 'cases': ")
        assert "nope" in str(caught.value)
        assert not isinstance(caught.value, FactoryValueError)

    async def test_a_database_source_with_an_unreadable_schema(self) -> None:
        with pytest.raises(ValidationError) as caught:
            await create_source_from_config(
                GroundedSourceConfig(
                    source_type="database",
                    name="cases",
                    options={"backend": "memory", "schema": {"fields": {"title": "money"}}},
                )
            )
        assert str(caught.value).startswith("source 'cases': ")
        assert "'money'" in str(caught.value)

    async def test_a_vector_kb_source_with_an_unresolvable_callable(self) -> None:
        with pytest.raises(DottedPathError) as caught:
            await create_source_from_config(
                GroundedSourceConfig(
                    source_type="vector_kb",
                    name="docs",
                    options={"dedup_key": "nope.module:fn"},
                ),
                knowledge_base=object(),
            )
        assert "'docs': cannot resolve dedup_key='nope.module:fn'" in str(caught.value)


_FIELDS = {"fields": {"id": {"type": "string"}, "title": {"type": "string"}}}


class TestABackendsConfigRefusalNamesTheSource:
    """A data backend's refusal of its own config reaches the bot config's author.

    These are dataknobs' words about what the author wrote, raised while the
    backend is built inside the ``database`` source factory, so they cross the
    source registry and come back through ``create_source_from_config``.
    """

    @pytest.mark.parametrize(
        ("options", "error", "words"),
        [
            pytest.param(
                {
                    "backend": "postgres",
                    "layout": "native",
                    "table": "cases",
                    "id_column": "case_id",
                    "schema": _FIELDS,
                },
                ValidationError,
                "id_column 'case_id' is not a declared column",
                id="native-undeclared-id-column",
            ),
            pytest.param(
                {"backend": "postgres", "layout": "native", "table": "cases", "id_column": "id"},
                ValidationError,
                "declares none",
                id="native-no-columns",
            ),
            pytest.param(
                {"backend": "postgres", "layout": "bogus"},
                ValidationError,
                "`layout:` is one of ['jsonb', 'native'], got 'bogus'",
                id="unknown-layout",
            ),
            pytest.param(
                {"backend": "postgres", "table": "bad name;"},
                ConfigurationError,
                "Invalid Postgres 'table' identifier",
                id="postgres-identifier",
            ),
            pytest.param(
                {
                    "backend": "sqlite",
                    "layout": "native",
                    "id_column": "id",
                    "schema": _FIELDS,
                    "journal_mode": "WAL",
                    "path": "cases.db",
                },
                ConfigurationError,
                "journal_mode: WAL",
                id="refused-under-native",
            ),
            pytest.param(
                {"backend": "elasticsearch", "max_result_window": 0},
                ValueError,
                "Elasticsearch 'max_result_window' must be a positive integer, got 0",
                id="elasticsearch-positive-integer",
            ),
            pytest.param(
                {"backend": "s3"},
                ValueError,
                "S3 backend requires 'bucket' in configuration",
                id="s3-bucket",
            ),
        ],
    )
    async def test_the_refusal_reaches_the_author_naming_the_source(
        self, options: dict[str, Any], error: type[BaseException], words: str
    ) -> None:
        with pytest.raises(error) as caught:
            await create_source_from_config(
                GroundedSourceConfig(source_type="database", name="cases", options=options)
            )
        message = str(caught.value)
        assert "cases" in message.split(":", 1)[0]
        assert words in message
        assert not isinstance(caught.value, FactoryValueError)
