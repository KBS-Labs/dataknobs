"""Tests for testing utilities."""

import contextlib
import http.server
import json
import socket
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from dataknobs_common.testing import (
    create_test_json_files,
    create_test_markdown_files,
    get_test_bot_config,
    get_test_rag_config,
    is_chromadb_available,
    is_elasticsearch_available,
    is_faiss_available,
    is_ollama_available,
    is_ollama_model_available,
    is_ollama_model_usable,
    is_package_available,
    is_postgres_available,
    is_redis_available,
    requires_chromadb,
    requires_faiss,
    requires_ollama,
    requires_ollama_model,
    requires_ollama_usable_model,
    requires_package,
    requires_redis,
    safe_sql_ident,
)


class TestServiceAvailability:
    """Tests for service availability checks."""

    def test_is_ollama_available_returns_bool(self):
        """Test that is_ollama_available returns a boolean."""
        result = is_ollama_available()
        assert isinstance(result, bool)

    def test_is_ollama_model_available_returns_bool(self):
        """Test that is_ollama_model_available returns a boolean."""
        result = is_ollama_model_available("nomic-embed-text")
        assert isinstance(result, bool)

    def test_is_ollama_model_available_returns_false_if_ollama_unavailable(self):
        """Test that model check returns False if Ollama is not available."""
        # If Ollama is not available, model check should also return False
        if not is_ollama_available():
            assert is_ollama_model_available("any-model") is False

    def test_is_ollama_model_usable_returns_bool(self):
        """is_ollama_model_usable returns a boolean."""
        result = is_ollama_model_usable("nonexistent-model-xyz", timeout=5.0)
        assert isinstance(result, bool)

    def test_is_ollama_model_usable_false_for_nonexistent_model(self):
        """A model that cannot generate is not usable (deterministic).

        Reproduce-first for the readiness gap this closes: a model that is not
        installed (or a runtime that returns empty output) must read as *not
        usable*, so a live-model suite can gate on generation rather than mere
        presence. Independent of whether Ollama is running — an unreachable
        server, an HTTP error, or an empty body all resolve to False.
        """
        assert is_ollama_model_usable("nonexistent-model-xyz", timeout=5.0) is False

    def test_is_ollama_model_usable_false_when_unreachable(self):
        """An unreachable Ollama endpoint resolves to False, not an error."""
        result = is_ollama_model_usable("any-model", host="localhost", port=65432, timeout=2.0)
        assert result is False

    def test_is_faiss_available_returns_bool(self):
        """Test that is_faiss_available returns a boolean."""
        result = is_faiss_available()
        assert isinstance(result, bool)

    def test_is_chromadb_available_returns_bool(self):
        """Test that is_chromadb_available returns a boolean."""
        result = is_chromadb_available()
        assert isinstance(result, bool)

    def test_is_redis_available_returns_bool(self):
        """Test that is_redis_available returns a boolean."""
        result = is_redis_available()
        assert isinstance(result, bool)

    def test_is_redis_available_with_custom_host_port(self):
        """Test is_redis_available with custom host and port."""
        # Test with unlikely port - should return False
        result = is_redis_available(host="localhost", port=65432)
        assert result is False


class TestServiceProbeHostResolution:
    """Host/port resolution of the socket-probe availability checks.

    The probe must resolve the same host its paired ``*_connection_params``
    fixture does: arg -> ``$<SVC>_HOST`` -> Docker-aware default. Previously
    the probes hard-coded ``localhost``, so inside a container (service at its
    compose hostname, not localhost) the probe reported "unavailable" and the
    ``requires_*`` marker false-skipped tests that would actually run.
    """

    @staticmethod
    def _install_addr_capture(monkeypatch) -> list[tuple[str, int]]:
        """Patch ``socket.socket`` to record connect_ex addresses; report closed.

        Returns the list that accumulates every probed ``(host, port)``.
        """
        import socket

        captured: list[tuple[str, int]] = []

        class _AddrCapturingSocket:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def settimeout(self, _seconds: float) -> None:
                pass

            def connect_ex(self, addr: tuple[str, int]) -> int:
                captured.append(addr)
                return 1  # nonzero => closed; probe returns False

            def close(self) -> None:
                pass

        monkeypatch.setattr(socket, "socket", lambda *_a, **_k: _AddrCapturingSocket())
        return captured

    def test_probe_resolves_docker_host_inside_container(self, monkeypatch):
        """With DOCKER_CONTAINER set and no host env, the probe targets the
        compose service hostname, not localhost.
        """
        captured = self._install_addr_capture(monkeypatch)
        monkeypatch.setenv("DOCKER_CONTAINER", "1")
        for var in ("POSTGRES_HOST", "ELASTICSEARCH_HOST", "REDIS_HOST"):
            monkeypatch.delenv(var, raising=False)

        assert is_postgres_available() is False  # closed socket
        assert is_elasticsearch_available() is False
        assert is_redis_available() is False

        hosts = [addr[0] for addr in captured]
        assert "postgres" in hosts
        assert "elasticsearch" in hosts
        assert "redis" in hosts
        assert "localhost" not in hosts

    def test_probe_resolves_localhost_on_host(self, monkeypatch):
        """Outside Docker (no /.dockerenv, no DOCKER_CONTAINER), host is
        localhost.
        """
        captured = self._install_addr_capture(monkeypatch)
        monkeypatch.delenv("DOCKER_CONTAINER", raising=False)
        for var in ("POSTGRES_HOST", "ELASTICSEARCH_HOST"):
            monkeypatch.delenv(var, raising=False)
        real_exists = __import__("os").path.exists
        monkeypatch.setattr(
            "dataknobs_common.testing._core.os.path.exists",
            lambda p: False if p == "/.dockerenv" else real_exists(p),
        )

        is_postgres_available()
        is_elasticsearch_available()

        assert {addr[0] for addr in captured} == {"localhost"}

    def test_explicit_host_and_env_win_over_docker_default(self, monkeypatch):
        """An explicit host arg and ``$<SVC>_HOST`` both beat the Docker
        default.
        """
        captured = self._install_addr_capture(monkeypatch)
        monkeypatch.setenv("DOCKER_CONTAINER", "1")

        # Explicit arg wins.
        monkeypatch.delenv("POSTGRES_HOST", raising=False)
        is_postgres_available(host="db.internal", port=5432)
        # Env var wins over the Docker default.
        monkeypatch.setenv("ELASTICSEARCH_HOST", "es.internal")
        is_elasticsearch_available()

        hosts = [addr[0] for addr in captured]
        assert "db.internal" in hosts
        assert "es.internal" in hosts
        assert "postgres" not in hosts
        assert "elasticsearch" not in hosts

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", ""])
    def test_a_negative_docker_container_value_does_not_mean_docker(self, monkeypatch, value):
        """``DOCKER_CONTAINER=false`` must read as "not in Docker".

        The check was a bare truthiness test, so every string but ``""``
        was Docker — and ``false`` is the value a developer writes when
        they mean the opposite. On a host that exports it, every probe
        and every ``*_connection_params`` default silently retargets from
        ``localhost`` to a compose hostname that does not resolve, and
        the ``requires_*`` markers false-skip the suite rather than
        failing it.
        """
        captured = self._install_addr_capture(monkeypatch)
        monkeypatch.setenv("DOCKER_CONTAINER", value)
        for var in ("POSTGRES_HOST", "ELASTICSEARCH_HOST"):
            monkeypatch.delenv(var, raising=False)
        real_exists = __import__("os").path.exists
        monkeypatch.setattr(
            "dataknobs_common.testing._core.os.path.exists",
            lambda p: False if p == "/.dockerenv" else real_exists(p),
        )

        is_postgres_available()
        is_elasticsearch_available()

        assert {addr[0] for addr in captured} == {"localhost"}

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_an_affirmative_docker_container_value_still_means_docker(self, monkeypatch, value):
        """The affirmative spellings must keep working, case-insensitively."""
        captured = self._install_addr_capture(monkeypatch)
        monkeypatch.setenv("DOCKER_CONTAINER", value)
        monkeypatch.delenv("POSTGRES_HOST", raising=False)

        is_postgres_available()

        assert {addr[0] for addr in captured} == {"postgres"}

    def test_is_package_available_returns_true_for_installed(self):
        """Test that is_package_available returns True for installed packages."""
        # pytest is definitely installed since we're running tests
        assert is_package_available("pytest") is True

    def test_is_package_available_returns_false_for_missing(self):
        """Test that is_package_available returns False for missing packages."""
        assert is_package_available("nonexistent_package_xyz") is False


def _health_report(**indicators: str) -> dict[str, Any]:
    """A ``/_health_report`` document whose indicators carry *indicators*.

    Defaults are the healthy single-node shape: every gating indicator green
    except ``shards_availability``, which sits at yellow whenever an index
    asks for a replica the one node cannot host -- the ordinary resting state
    of a dev cluster, and so the shape the probe must still call available.
    """
    statuses = {
        "master_is_stable": "green",
        "disk": "green",
        "shards_capacity": "green",
        "shards_availability": "yellow",
        **indicators,
    }
    return {
        "status": "yellow",
        "indicators": {name: {"status": value} for name, value in statuses.items()},
    }


@contextlib.contextmanager
def _stub_http_json(routes: dict[str, tuple[int, Any]]) -> Iterator[tuple[str, int]]:
    """A real local HTTP server answering probe paths with JSON.

    Not a mock of any dataknobs interface -- an actual ``http.server`` on an
    ephemeral port, the same construct ``test_elasticsearch_sweep.py`` uses --
    so a probe runs its genuine request and JSON-parsing path against a
    controllable endpoint that needs no service behind it.

    ``POST`` is answered from the same table as ``GET``: the probes differ by
    method and by nothing else a stub here would model, and a second server
    class for the one that carries a body would be the duplication the shared
    request helper was just written to remove.

    Args:
        routes: Path (``"/_health_report"``, ``"/api/chat"``) to
            ``(status, json_body)``. A path with no entry answers 404, which
            is what a service predating an endpoint returns.

    Yields:
        The ``(host, port)`` the server is listening on.
    """

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            status, payload = routes.get(self.path, (404, {"error": "no handler"}))
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            # Drain the request body before answering: an unread body leaves
            # bytes in the socket and the client reads them as the next
            # response, which would make this stub flaky rather than wrong.
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.do_GET()

        def log_message(self, *_args: object) -> None:  # silence stderr noise
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    # A short poll interval so shutdown is prompt: the default 0.5s is what
    # serve_forever waits to notice the stop flag, and it would put a flat
    # half-second on every test in this class.
    thread = threading.Thread(target=lambda: server.serve_forever(0.01), daemon=True)
    thread.start()
    try:
        yield server.server_address[0], server.server_address[1]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class TestElasticsearchReadinessProbe:
    """The probe gates on serving, not on a listening port.

    ``requires_elasticsearch`` exists so a suite the cluster cannot run
    *skips*. A cluster out of disk defeats a reachability-only probe
    completely: it accepts the connection and answers ``/`` in milliseconds,
    reports ``green`` cluster status while it still holds no unassigned
    shard, and then blocks the suite's first index creation until the client
    gives up -- turning the promised skip into four timeouts.
    """

    def test_a_cluster_low_on_disk_is_not_available(self):
        """The reproducer: disk over the watermark, nothing unassigned yet.

        This is the state a dev cluster is in *before* the first test index
        is requested -- cluster status still green, because no shard has been
        refused yet. The health report is the only place the coming refusal
        is visible, and it is visible there as a non-green ``disk``.
        """
        with _stub_http_json(
            {
                "/_health_report": (200, _health_report(disk="yellow")),
                "/_cluster/health": (200, {"status": "green"}),
            }
        ) as (host, port):
            assert is_elasticsearch_available(host, port) is False

    def test_a_cluster_with_unassigned_primaries_is_not_available(self):
        """Red ``shards_availability`` means primaries are unassigned."""
        with _stub_http_json(
            {
                "/_health_report": (200, _health_report(shards_availability="red")),
                "/_cluster/health": (200, {"status": "red"}),
            }
        ) as (host, port):
            assert is_elasticsearch_available(host, port) is False

    def test_a_cluster_out_of_room_for_shards_is_not_available(self):
        """``shards_capacity`` is the ceiling the session-start sweep reclaims.

        The sweep exists because accumulated ``test_*`` residue exhausts
        ``cluster.max_shards_per_node``; a cluster already at the ceiling
        when the suite starts refuses every new index, and the gate should
        say so rather than let the suite discover it.
        """
        with _stub_http_json({"/_health_report": (200, _health_report(shards_capacity="red"))}) as (
            host,
            port,
        ):
            assert is_elasticsearch_available(host, port) is False

    def test_a_healthy_single_node_cluster_is_available(self):
        """The positive control, and the one that matters most.

        A gate that over-skips reports green while testing nothing. A
        one-node cluster hosting any replicated index sits at yellow
        ``shards_availability`` permanently, so treating yellow there as
        unavailable would skip every ordinary dev run.
        """
        with _stub_http_json(
            {
                "/_health_report": (200, _health_report()),
                "/_cluster/health": (200, {"status": "yellow"}),
            }
        ) as (host, port):
            assert is_elasticsearch_available(host, port) is True

    def test_a_cluster_without_the_health_api_falls_back_to_cluster_health(self):
        """``/_health_report`` is 8.7+; older clusters must not all-skip.

        A consumer on an Elasticsearch predating the health API gets a 404
        here. Reading that as unavailable would silently skip their whole
        suite, so the probe falls back to the ``yellow``-or-``green``
        criterion ``wait_for_elasticsearch`` has always used.
        """
        with _stub_http_json({"/_cluster/health": (200, {"status": "yellow"})}) as (
            host,
            port,
        ):
            assert is_elasticsearch_available(host, port) is True

    def test_the_fallback_still_refuses_a_red_cluster(self):
        """The fallback is a weaker check, not an absent one."""
        with _stub_http_json({"/_cluster/health": (200, {"status": "red"})}) as (
            host,
            port,
        ):
            assert is_elasticsearch_available(host, port) is False

    def test_a_port_that_answers_nothing_useful_is_not_available(self):
        """TCP open and both probes failing is not a cluster to run against."""
        with _stub_http_json({}) as (host, port):
            assert is_elasticsearch_available(host, port) is False


@contextlib.contextmanager
def _stub_malformed_http() -> Iterator[tuple[str, int]]:
    """A listening socket that answers with something that is not HTTP.

    ``http.server`` cannot produce this shape -- it always writes a valid
    status line -- so the reproduction is a raw socket. What comes back
    raises ``http.client.BadStatusLine``, which is an ``HTTPException`` and
    **not** an ``OSError``: the one shape a catch set spelled
    ``(URLError, TimeoutError, ValueError, OSError)`` misses while reading as
    though it covers everything.

    Yields:
        The ``(host, port)`` the socket is listening on.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(5)
    server.settimeout(0.05)
    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            except OSError:  # the socket closed under us; we are done
                return
            with contextlib.closing(conn), contextlib.suppress(OSError):
                conn.recv(65536)
                conn.sendall(b"NOT-HTTP-AT-ALL\r\n\r\n")

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        host, port = server.getsockname()
        yield host, port
    finally:
        stop.set()
        thread.join(timeout=5)
        server.close()


class TestProbeFailureModes:
    """A probe answers "no". It does not raise.

    Every probe here backs a ``pytest.mark.skipif`` whose condition is
    evaluated the moment a consuming module applies the marker -- at
    *import*, where an escaping exception is not a failed probe but a
    collection error that takes the whole module with it. So the catch set
    has to cover every shape a listening port can answer with, and
    ``HTTPException`` is the one that looks covered and is not: it does not
    inherit ``OSError``, so the narrower spelling let it straight through.
    """

    def test_a_malformed_response_does_not_escape_the_usable_model_probe(self) -> None:
        """The reproducer.

        ``is_ollama_model_usable`` was the probe left out of the shared
        request helper, and it still carried the narrow catch set. Against a
        port answering a malformed status line it raised ``BadStatusLine``
        out of ``requires_ollama_usable_model``'s ``skipif`` instead of
        reporting the model unusable.
        """
        with _stub_malformed_http() as (host, port):
            assert is_ollama_model_usable("any-model", host=host, port=port, timeout=5) is False

    def test_a_malformed_response_does_not_escape_the_model_listing_probe(self) -> None:
        """Its sibling, which the shared helper already covered."""
        with _stub_malformed_http() as (host, port):
            assert is_ollama_model_available("any-model", host, port) is False

    def test_a_malformed_response_does_not_escape_the_elasticsearch_probe(self) -> None:
        """The union catch set, pinned on the probe this branch rewrote.

        Nothing else asserts that ``HTTPException`` is caught rather than
        merely listed, and a catch set is exactly the kind of claim that
        reads as true until something raises the member that is missing.
        """
        with _stub_malformed_http() as (host, port):
            assert is_elasticsearch_available(host, port) is False

    def test_a_body_of_the_wrong_shape_does_not_escape_the_usable_model_probe(self) -> None:
        """A second escape in the same function, found next to the first.

        The canary read ``body.get("message")`` off whatever JSON came back.
        A well-formed JSON *list* has no ``.get``, and the ``AttributeError``
        that raised is in no probe's catch set -- so a service answering the
        wrong shape crashed collection just as a malformed one did.
        """
        with _stub_http_json({"/api/chat": (200, ["not", "an", "object"])}) as (host, port):
            assert is_ollama_model_usable("any-model", host=host, port=port, timeout=5) is False

    def test_a_model_answering_with_content_is_usable(self) -> None:
        """The positive control, and the reason the others prove anything.

        A probe hard-wired to ``False`` would pass every assertion above.
        """
        with _stub_http_json({"/api/chat": (200, {"message": {"content": "ok"}})}) as (host, port):
            assert is_ollama_model_usable("any-model", host=host, port=port, timeout=5) is True

    def test_a_model_answering_nothing_is_not_usable(self) -> None:
        """The check this probe exists for: listed, loaded, and mute."""
        with _stub_http_json({"/api/chat": (200, {"message": {"content": "   "}})}) as (
            host,
            port,
        ):
            assert is_ollama_model_usable("any-model", host=host, port=port, timeout=5) is False


class TestPytestMarkers:
    """Tests for pytest markers."""

    def test_requires_ollama_is_marker(self):
        """Test that requires_ollama is a valid marker."""
        assert requires_ollama is not None
        # It should be a pytest.mark object
        assert hasattr(requires_ollama, "mark")

    def test_requires_faiss_is_marker(self):
        """Test that requires_faiss is a valid marker."""
        assert requires_faiss is not None
        assert hasattr(requires_faiss, "mark")

    def test_requires_chromadb_is_marker(self):
        """Test that requires_chromadb is a valid marker."""
        assert requires_chromadb is not None
        assert hasattr(requires_chromadb, "mark")

    def test_requires_redis_is_marker(self):
        """Test that requires_redis is a valid marker."""
        assert requires_redis is not None
        assert hasattr(requires_redis, "mark")

    def test_requires_package_returns_marker(self):
        """Test that requires_package returns a marker."""
        marker = requires_package("pytest")
        assert marker is not None
        assert hasattr(marker, "mark")

    def test_requires_ollama_model_returns_marker(self):
        """Test that requires_ollama_model returns a marker."""
        marker = requires_ollama_model("nomic-embed-text")
        assert marker is not None
        assert hasattr(marker, "mark")

    def test_requires_ollama_usable_model_returns_marker(self):
        """Test that requires_ollama_usable_model returns a marker."""
        marker = requires_ollama_usable_model("nomic-embed-text")
        assert marker is not None
        assert hasattr(marker, "mark")


class TestBotConfigFactory:
    """Tests for get_test_bot_config factory."""

    def test_default_config(self):
        """Test default configuration."""
        config = get_test_bot_config()

        assert "llm" in config
        assert config["llm"]["provider"] == "echo"
        assert config["llm"]["model"] == "test"
        assert "conversation_storage" in config
        assert config["conversation_storage"]["backend"] == "memory"

    def test_with_real_llm(self):
        """Test configuration with real LLM."""
        config = get_test_bot_config(use_echo_llm=False)

        assert config["llm"]["provider"] == "openai"
        assert config["llm"]["model"] == "gpt-4o-mini"

    def test_with_memory(self):
        """Test configuration with memory enabled."""
        config = get_test_bot_config(include_memory=True)

        assert "memory" in config
        assert config["memory"]["type"] == "buffer"
        assert config["memory"]["max_messages"] == 10

    def test_without_memory(self):
        """Test configuration without memory."""
        config = get_test_bot_config(include_memory=False)

        assert "memory" not in config

    def test_with_system_prompt(self):
        """Test configuration with system prompt."""
        prompt = "You are a helpful assistant."
        config = get_test_bot_config(system_prompt=prompt)

        assert "system_prompt" in config
        assert config["system_prompt"] == prompt

    def test_without_system_prompt(self):
        """Test configuration without system prompt."""
        config = get_test_bot_config()

        assert "system_prompt" not in config

    def test_file_storage(self):
        """Test configuration with file storage."""
        config = get_test_bot_config(use_in_memory_storage=False)

        assert config["conversation_storage"]["backend"] == "file"


class TestRAGConfigFactory:
    """Tests for get_test_rag_config factory."""

    def test_default_config(self):
        """Test default RAG configuration."""
        config = get_test_rag_config()

        assert config["type"] == "rag"
        assert config["vector_store"]["backend"] == "memory"
        assert config["embedding_provider"] == "ollama"
        assert config["embedding_model"] == "nomic-embed-text"
        assert "chunking" in config
        assert "retrieval" in config

    def test_with_faiss_backend(self):
        """Test RAG configuration with FAISS backend."""
        config = get_test_rag_config(use_in_memory_store=False)

        assert config["vector_store"]["backend"] == "faiss"

    def test_with_custom_embedding(self):
        """Test RAG configuration with custom embedding."""
        config = get_test_rag_config(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )

        assert config["embedding_provider"] == "openai"
        assert config["embedding_model"] == "text-embedding-3-small"

    def test_chunking_config(self):
        """Test that chunking configuration is present."""
        config = get_test_rag_config()

        assert config["chunking"]["max_chunk_size"] == 800
        assert "chunk_overlap" not in config["chunking"]

    def test_retrieval_config(self):
        """Test that retrieval configuration is present."""
        config = get_test_rag_config()

        assert config["retrieval"]["top_k"] == 5
        assert config["retrieval"]["score_threshold"] == 0.7


class TestFileHelpers:
    """Tests for file creation helpers."""

    def test_create_test_markdown_files(self, tmp_path: Path):
        """Test creating test markdown files."""
        files = create_test_markdown_files(tmp_path)

        assert len(files) == 2
        assert all(Path(f).exists() for f in files)
        assert all(f.endswith(".md") for f in files)

        # Check content
        for file_path in files:
            content = Path(file_path).read_text()
            assert len(content) > 0
            assert "# " in content  # Contains headers

    def test_create_test_json_files(self, tmp_path: Path):
        """Test creating test JSON files."""
        files = create_test_json_files(tmp_path)

        assert len(files) == 2
        assert all(Path(f).exists() for f in files)
        assert all(f.endswith(".json") for f in files)

        # Check content is valid JSON
        for file_path in files:
            content = Path(file_path).read_text()
            data = json.loads(content)
            assert "title" in data
            assert "items" in data
            assert "metadata" in data

    def test_markdown_files_in_correct_directory(self, tmp_path: Path):
        """Test that markdown files are created in the correct directory."""
        files = create_test_markdown_files(tmp_path)

        for file_path in files:
            assert Path(file_path).parent == tmp_path

    def test_json_files_in_correct_directory(self, tmp_path: Path):
        """Test that JSON files are created in the correct directory."""
        files = create_test_json_files(tmp_path)

        for file_path in files:
            assert Path(file_path).parent == tmp_path


class TestSafeSqlIdent:
    """Tests for the safe_sql_ident helper."""

    @pytest.mark.parametrize(
        "name",
        [
            "public",
            "test_records_abc12345",
            "_x",
            "X1",
            "schema_name",
            "T",
        ],
    )
    def test_valid_identifier_returned_unchanged(self, name: str) -> None:
        assert safe_sql_ident(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "1abc",
            "a-b",
            "a.b",
            "a; DROP TABLE x",
            "a b",
            'a"b',
            "a'b",
            "a;b",
        ],
    )
    def test_invalid_identifier_raises(self, name: str) -> None:
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            safe_sql_ident(name)

    @pytest.mark.parametrize("value", [None, 123, b"public", ["public"]])
    def test_non_string_input_raises(self, value: object) -> None:
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            safe_sql_ident(value)  # type: ignore[arg-type]
