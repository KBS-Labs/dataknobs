"""Every refusal a built-in source raises about itself carries ``source``.

A bot config can declare several grounded sources. The message named which one
was refused, and ``context`` -- what a program reads, and what the API renders
as ``detail`` -- did not.

The second half pins what that context must *not* pick up: a backend's own
text, which can hold the connection URL it was handed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from dataknobs_bots.knowledge.sources import SourceConfigError, create_source_from_config
from dataknobs_bots.knowledge.sources.factory import _create_database_source
from dataknobs_bots.reasoning.grounded_config import GroundedSourceConfig
from dataknobs_common.exceptions import ConfigFieldError, PluginConfigError, ValidationError
from dataknobs_common.structured_config import StructuredConfig
from dataknobs_data.backends import async_backends

DSN = "postgres://admin:hunter2@db.internal:5432/prod"


def _config(**options: Any) -> GroundedSourceConfig:
    return GroundedSourceConfig(source_type="database", name="cases", options=options)


async def _direct(config: GroundedSourceConfig) -> Any:
    return await _create_database_source(config)


async def _helper(config: GroundedSourceConfig) -> Any:
    return await create_source_from_config(config)


LANES = pytest.mark.parametrize("build", [_direct, _helper], ids=["direct", "helper"])

#: A key the backend's config class refuses.
REFUSED = [
    pytest.param({"backend": "memory", "nonsense_key": 1}, "'nonsense_key'", id="unknown-key"),
    pytest.param({"backend": "elasticsearch", "layout": "native"}, "'layout'", id="layout"),
]

#: Refused in dataknobs' words, though not as a config class's key.
MARKED = [
    pytest.param({"backend": "nope"}, "nope", id="unknown-backend"),
    pytest.param({"backend": "s3"}, "bucket", id="no-bucket"),
]


@pytest.fixture
def leaky_backend() -> Iterator[Any]:
    """Registers an async backend whose construction raises what it is given."""

    def register(build: Any) -> str:
        class Leaky:
            @classmethod
            def from_config(cls, config: Any, **_: Any) -> Any:
                return build(config)

        async_backends.register("test_leaky", Leaky)
        return "test_leaky"

    yield register
    if async_backends.is_registered("test_leaky"):
        async_backends.unregister("test_leaky")


class TestSourceIsInContext:
    @pytest.mark.parametrize(("options", "word"), [*REFUSED, *MARKED])
    async def test_called_directly(self, options: dict[str, Any], word: str) -> None:
        with pytest.raises(SourceConfigError) as caught:
            await _direct(_config(**options))

        assert caught.value.context["source"] == "cases"
        assert str(caught.value).startswith("Source 'cases': ")
        assert word in str(caught.value)

    @pytest.mark.parametrize(("options", "word"), MARKED)
    async def test_through_the_helper(self, options: dict[str, Any], word: str) -> None:
        with pytest.raises(SourceConfigError) as caught:
            await _helper(_config(**options))

        assert caught.value.context["source"] == "cases"
        assert str(caught.value).startswith("Source 'cases': ")
        assert word in str(caught.value)

    @pytest.mark.parametrize(("options", "word"), REFUSED)
    async def test_a_refused_key_through_the_helper(
        self, options: dict[str, Any], word: str
    ) -> None:
        with pytest.raises(PluginConfigError) as caught:
            await _helper(_config(**options))

        assert caught.value.context["source"] == "cases"
        assert caught.value.context["key"] == "database"
        assert caught.value.context["unknown"]
        assert str(caught.value).startswith("Failed to create plugin 'database': Source 'cases': ")
        assert word in str(caught.value)

    async def test_a_refused_keys_own_context_is_kept(self) -> None:
        with pytest.raises(SourceConfigError) as caught:
            await _direct(_config(backend="memory", nonsense_key=1))

        assert caught.value.context["unknown"] == ["nonsense_key"]

    async def test_a_vector_kb_source_with_no_knowledge_base(self) -> None:
        config = GroundedSourceConfig(source_type="vector_kb", name="docs", options={})

        with pytest.raises(SourceConfigError) as caught:
            await create_source_from_config(config)

        assert caught.value.context == {"source": "docs"}

    @pytest.mark.parametrize("source_type", ["nope", ""])
    async def test_an_unknown_source_type(self, source_type: str) -> None:
        config = GroundedSourceConfig(source_type=source_type, name="docs", options={})

        with pytest.raises(ValueError) as caught:
            await create_source_from_config(config)

        assert str(caught.value).startswith(f"Unknown grounded source type: {source_type!r}. ")
        assert isinstance(caught.value, SourceConfigError)
        assert caught.value.context == {"source": "docs"}

    def test_it_is_both_a_validation_error_and_a_value_error(self) -> None:
        assert issubclass(SourceConfigError, ValidationError)
        assert issubclass(SourceConfigError, ValueError)


class TestABackendsOwnTextStaysOut:
    @LANES
    async def test_a_backend_value_error_is_named_by_type(
        self, build: Any, leaky_backend: Any
    ) -> None:
        def fail(_: Any) -> Any:
            raise ValueError(f"cannot reach {DSN}")

        with pytest.raises(ValueError) as caught:
            await build(_config(backend=leaky_backend(fail)))

        error = caught.value
        assert isinstance(error, SourceConfigError)
        assert str(error) == "Source 'cases': its database could not be opened (ValueError)"
        assert error.context == {"source": "cases", "cause_type": "ValueError"}
        assert DSN in str(error.__cause__)

    @LANES
    async def test_a_backend_errors_context_is_not_carried(
        self, build: Any, leaky_backend: Any
    ) -> None:
        class BackendError(ValidationError, ValueError):
            pass

        def fail(_: Any) -> Any:
            raise BackendError("refused", context={"url": DSN})

        with pytest.raises(SourceConfigError) as caught:
            await build(_config(backend=leaky_backend(fail)))

        assert caught.value.context == {"source": "cases", "cause_type": "BackendError"}

    @LANES
    async def test_a_refusal_a_backend_amended_carries_only_its_recorded_words(
        self, build: Any, leaky_backend: Any
    ) -> None:
        """The recorded context, never ``exc.context``, which a backend may since have written to."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class LeakyConfig(StructuredConfig):
            _UNKNOWN_KEYS = "raise"
            path: str = ""

        def fail(_: Any) -> Any:
            try:
                LeakyConfig.from_dict({"pth": "x"})
            except ConfigFieldError as exc:
                exc.args = (f"{exc} (opening {DSN})",)
                exc.context["url"] = DSN
                raise
            raise AssertionError("the config class accepted an unknown key")

        with pytest.raises(ConfigFieldError if build is _helper else SourceConfigError) as caught:
            await build(_config(backend=leaky_backend(fail)))

        assert caught.value.context["source"] == "cases"
        assert caught.value.context["unknown"] == ["pth"]
        assert "Source 'cases': LeakyConfig does not accept 'pth'" in str(caught.value)
        assert DSN not in str(caught.value)
        assert DSN not in repr(caught.value.context)


async def test_the_api_renders_the_source() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from dataknobs_bots.api.exceptions import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    async def boom() -> None:
        await create_source_from_config(_config(backend="nope"))

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/boom")

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "SourceConfigError"
    assert body["message"].startswith("Source 'cases': ")
    assert body["detail"] == {"source": "cases"}
