"""Behavioural tests for the published Ollama probe surface.

Every test here drives the real functions against a **real local HTTP server**
speaking Ollama's wire protocol on an ephemeral port — not a mock of anything,
and not a hand-rolled stand-in for a dataknobs interface. That matters for this
particular surface: the defects these tests pin are all about *which endpoint
gets probed and how its answer is read*, which a mocked transport defines away.

The reproduce-first pairs:

- **Reach.** ``is_ollama_available`` shelled out to the local ``ollama`` CLI and
  took no host or port, so it could not see a service anywhere but this machine
  and ignored ``OLLAMA_HOST`` entirely. Its four siblings in the same module
  (Redis, Postgres, Elasticsearch, LocalStack) all resolve arg → env → default.
- **Env format.** ``OLLAMA_HOST`` is Ollama's own variable and carries a URL
  (``bin/check-ollama.sh`` defaults it to ``http://localhost:11434``). The
  hand-rolled copies in two conftests read it as a bare hostname and built
  ``http://http://localhost:11434:11434/api/tags``.
- **Match.** ``is_ollama_model_available`` tested ``model_name in stdout`` — a
  substring search over the whole ``ollama list`` table, so a request for
  ``gemma3`` was satisfied by ``gemma3-uncensored:latest``.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import socket
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from dataknobs_common.testing import (
    is_ollama_available,
    is_ollama_model_available,
    is_ollama_model_usable,
    list_ollama_models,
    ollama_env_params,
    wait_for_ollama,
)


@contextlib.contextmanager
def _ollama_stub(
    models: list[str] | None = None,
    *,
    chat_content: str = "ok",
) -> Iterator[tuple[str, int]]:
    """A real HTTP server answering Ollama's ``/api/tags`` and ``/api/chat``.

    Yields ``(host, port)`` for a server on an ephemeral port. The response
    bodies use Ollama's real shapes, so the code under test does its genuine
    request, parse and projection rather than meeting a stand-in halfway.

    Args:
        models: Installed model names to report from ``/api/tags``.
        chat_content: Message content ``/api/chat`` returns; ``""`` simulates
            the empty-output runtime the usability canary exists to catch.
    """
    tags_body = json.dumps(
        {"models": [{"name": name, "model": name, "size": 1} for name in (models or [])]}
    ).encode()
    chat_body = json.dumps({"message": {"role": "assistant", "content": chat_content}}).encode()

    class _Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # Names mandated by BaseHTTPRequestHandler's do_<METHOD> dispatch.
        def do_GET(self) -> None:
            self._send(tags_body if self.path.startswith("/api/tags") else b"{}")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            self._send(chat_body if self.path.startswith("/api/chat") else b"{}")

        def log_message(self, *_args: object) -> None:
            """Silence the handler's per-request stderr noise."""

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[0], server.server_address[1]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _closed_port() -> int:
    """A port with nothing listening on it.

    Bound and released so the number is one the OS just confirmed free, rather
    than a hardcoded guess that another process on a busy machine may hold.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(autouse=True)
def _clear_ollama_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from an environment that says nothing about Ollama.

    Otherwise a developer with ``OLLAMA_HOST`` exported — the very people this
    surface is for — gets different results than CI.
    """
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_PORT", raising=False)


class TestReach:
    """The probe answers about the configured endpoint, not about this machine."""

    def test_availability_follows_the_configured_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``OLLAMA_HOST``/``OLLAMA_PORT`` decide what gets probed.

        Both halves are asserted together on purpose. The CLI probe this
        replaces returned the same answer whichever way the environment
        pointed, so no machine state satisfies both directions at once — which
        is what makes this a reproduction rather than a coincidence.
        """
        with _ollama_stub(["llama3.1:8b"]) as (host, port):
            monkeypatch.setenv("OLLAMA_HOST", host)
            monkeypatch.setenv("OLLAMA_PORT", str(port))
            assert is_ollama_available() is True

            monkeypatch.setenv("OLLAMA_PORT", str(_closed_port()))
            assert is_ollama_available() is False

    def test_availability_takes_an_explicit_host_and_port(self) -> None:
        """Explicit arguments win over the environment, as the siblings do."""
        with _ollama_stub(["llama3.1:8b"]) as (host, port):
            assert is_ollama_available(host, port) is True
        assert is_ollama_available("127.0.0.1", _closed_port()) is False

    def test_unreachable_endpoint_is_false_not_an_exception(self) -> None:
        """A closed port resolves to ``False`` — the caller decides skip vs fail."""
        assert is_ollama_available("127.0.0.1", _closed_port()) is False
        assert list_ollama_models("127.0.0.1", _closed_port()) == []
        assert is_ollama_model_available("any", "127.0.0.1", _closed_port()) is False


class TestOllamaHostFormats:
    """``OLLAMA_HOST`` is Ollama's variable and arrives in several shapes."""

    def test_url_form_is_understood(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``http://host:port`` — the form this repo's own shell scripts use.

        ``bin/check-ollama.sh`` and ``bin/manage-services.sh`` both default
        ``OLLAMA_HOST`` to ``http://localhost:11434``, so a developer who
        exports it for those scripts was breaking the Python probes, which
        pasted the URL in where a hostname belonged.
        """
        with _ollama_stub(["llama3.1:8b"]) as (host, port):
            monkeypatch.setenv("OLLAMA_HOST", f"http://{host}:{port}")
            assert ollama_env_params() == {"host": host, "port": port}
            assert is_ollama_available() is True

    def test_host_port_form_is_understood(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``host:port`` — the form the ``ollama`` CLI itself documents."""
        with _ollama_stub(["llama3.1:8b"]) as (host, port):
            monkeypatch.setenv("OLLAMA_HOST", f"{host}:{port}")
            assert ollama_env_params() == {"host": host, "port": port}
            assert is_ollama_available() is True

    def test_bare_host_form_is_understood(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bare hostname keeps the default port."""
        monkeypatch.setenv("OLLAMA_HOST", "ollama.internal")
        assert ollama_env_params() == {"host": "ollama.internal", "port": 11434}

    def test_explicit_port_var_overrides_a_port_in_the_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``OLLAMA_PORT`` is the more specific statement, so it wins."""
        monkeypatch.setenv("OLLAMA_HOST", "http://ollama.internal:11434")
        monkeypatch.setenv("OLLAMA_PORT", "9999")
        assert ollama_env_params() == {"host": "ollama.internal", "port": 9999}

    def test_defaults_when_nothing_is_set(self) -> None:
        """No environment, no surprises."""
        assert ollama_env_params() == {"host": "localhost", "port": 11434}


class TestModelMatching:
    """Which installed models satisfy a request for a model."""

    def test_a_longer_family_name_does_not_satisfy_a_shorter_request(self) -> None:
        """``gemma3`` is not satisfied by ``gemma3-uncensored:latest``.

        The substring form this replaces answered yes here, so a suite gated on
        one model ran green against a different one nobody asked for.
        """
        with _ollama_stub(["gemma3-uncensored:latest"]) as (host, port):
            assert is_ollama_model_available("gemma3", host, port) is False

    def test_a_tagged_variant_satisfies_an_untagged_request(self) -> None:
        """``gemma3`` is satisfied by ``gemma3:1b`` — same model, explicit tag."""
        with _ollama_stub(["gemma3:1b"]) as (host, port):
            assert is_ollama_model_available("gemma3", host, port) is True

    def test_an_exact_name_matches(self) -> None:
        """A fully qualified request matches its own name."""
        with _ollama_stub(["gemma3:1b", "llama3.1:8b"]) as (host, port):
            assert is_ollama_model_available("llama3.1:8b", host, port) is True

    def test_a_request_with_a_tag_is_not_satisfied_by_another_tag(self) -> None:
        """``gemma3:1b`` is not satisfied by ``gemma3:27b``."""
        with _ollama_stub(["gemma3:27b"]) as (host, port):
            assert is_ollama_model_available("gemma3:1b", host, port) is False

    def test_metadata_cannot_satisfy_a_request(self) -> None:
        """Only names are matched.

        The CLI form searched the rendered ``ollama list`` table, where the ID,
        size and modified columns were as matchable as the name.
        """
        with _ollama_stub(["gemma3:1b"]) as (host, port):
            assert is_ollama_model_available("1", host, port) is False

    def test_list_reports_the_installed_names(self) -> None:
        """``list_ollama_models`` projects ``/api/tags`` to plain names."""
        with _ollama_stub(["gemma3:1b", "llama3.1:8b"]) as (host, port):
            assert list_ollama_models(host, port) == ["gemma3:1b", "llama3.1:8b"]


class TestUsabilityCanary:
    """The stronger check resolves its endpoint the same way as the rest."""

    def test_canary_defaults_to_the_configured_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no explicit host, the environment decides.

        Its ``host="localhost"`` default was hardcoded, so the one function
        that *did* take a host still could not be pointed anywhere by
        configuration alone.

        The model is named so that no real Ollama could satisfy it. A plausible
        name makes this test pass on a developer machine that happens to have
        it installed — measured, not guessed: the first draft used ``gemma3:1b``
        and reported success against the local server while the code under test
        ignored the environment entirely.
        """
        unhostable = "dataknobs-canary-not-a-real-model:v0"
        with _ollama_stub([unhostable]) as (host, port):
            monkeypatch.setenv("OLLAMA_HOST", f"http://{host}:{port}")
            assert is_ollama_model_usable(unhostable, timeout=5.0) is True

    def test_empty_output_is_not_usable(self) -> None:
        """A model that returns nothing fails the canary — the whole point."""
        with _ollama_stub(["gemma3:1b"], chat_content="") as (host, port):
            assert is_ollama_model_usable("gemma3:1b", host=host, port=port, timeout=5.0) is False


class TestWaitForOllama:
    """The retry loop both conftests had, published once."""

    def test_returns_true_when_reachable(self) -> None:
        """A reachable service returns immediately."""
        with _ollama_stub(["gemma3:1b"]) as (host, port):
            assert wait_for_ollama(host, port, max_retries=1, delay=0.0) is True

    def test_raises_after_the_last_attempt(self) -> None:
        """An unreachable service raises with the endpoint in the message.

        One copy of this loop returned ``False`` instead of raising whenever
        the server answered with a non-200 rather than refusing the connection,
        so a misconfigured endpoint read as a clean negative.
        """
        port = _closed_port()
        with pytest.raises(ConnectionError, match=f"127.0.0.1:{port}"):
            wait_for_ollama("127.0.0.1", port, max_retries=2, delay=0.0)


def test_env_params_are_not_shared_between_callers() -> None:
    """Each call returns a fresh dict, so a caller may mutate its copy."""
    first: dict[str, Any] = ollama_env_params()
    first["host"] = "mutated"
    assert ollama_env_params()["host"] == "localhost"
