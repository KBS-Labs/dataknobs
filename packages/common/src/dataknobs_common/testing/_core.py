"""Test utilities for dataknobs packages.

This module provides pytest utilities for service availability checking,
test configuration factories, and fixture helpers.

Example:
    ```python
    import pytest
    from dataknobs_common.testing import (
        is_ollama_available,
        requires_ollama,
        get_test_bot_config,
        safe_sql_ident,
    )

    # Skip test if Ollama not available
    @pytest.mark.skipif(not is_ollama_available(), reason="Ollama not available")
    def test_with_ollama():
        ...

    # Or use the marker
    @requires_ollama
    def test_with_ollama_marker():
        ...

    # Get test configuration
    config = get_test_bot_config(use_echo_llm=True)

    # Validate SQL identifiers built from env vars / hardcoded defaults /
    # uuid suffixes before f-string interpolation in test fixtures
    cursor.execute(f"DROP TABLE IF EXISTS {safe_sql_ident(table)}")
    ```
"""

import importlib.util
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# SQL Identifier Validation


_SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def safe_sql_ident(name: str) -> str:
    """Validate that ``name`` is a safe unquoted SQL identifier.

    Returns the name unchanged when valid; raises ``ValueError`` otherwise.
    Intended for test-fixture identifier interpolation where the value comes
    from environment variables, hardcoded defaults, or uuid-based suffixes —
    not for arbitrary user input. For production code use the database
    driver's quoting facility (e.g. ``psycopg2.extensions.quote_ident``).
    """
    if not isinstance(name, str) or not _SQL_IDENT_RE.fullmatch(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


# Service Availability Checks


_OLLAMA_DEFAULT_PORT = 11434

#: Cap on a probe response body, well above any real one (an ``/api/tags``
#: listing or a LocalStack health document is kilobytes). A probe reads from
#: whatever endpoint the environment names, and an unbounded read there turns
#: a misdirected variable into a hang: the probe never answers, so the suite
#: it gates neither runs nor skips. Over the cap is treated as a failed probe.
_MAX_PROBE_BODY_BYTES = 1024 * 1024


def _read_probe_json(response: Any) -> Any:
    """Decode a bounded JSON body from an open HTTP response.

    Raises:
        ValueError: If the body exceeds :data:`_MAX_PROBE_BODY_BYTES` or is not
            JSON — the same type every caller here already treats as "no".
    """
    body = response.read(_MAX_PROBE_BODY_BYTES + 1)
    if len(body) > _MAX_PROBE_BODY_BYTES:
        raise ValueError(f"probe response exceeded {_MAX_PROBE_BODY_BYTES} bytes")
    return json.loads(body)


def _parse_ollama_host(raw: str) -> tuple[str, int | None]:
    """Split an ``OLLAMA_HOST`` value into a hostname and an optional port.

    ``OLLAMA_HOST`` is Ollama's own variable, not one this project invented,
    and it is written three ways in the wild: ``http://host:port`` (what this
    repo's ``bin/check-ollama.sh`` and ``bin/manage-services.sh`` default it
    to), ``host:port`` (what the ``ollama`` CLI documents), and a bare
    hostname. All three are accepted here, because a reader that accepts only
    one of them turns a correctly-set variable into a wrong endpoint rather
    than an error — the URL form pasted into a hostname slot yields
    ``http://http://host:port:11434/api/tags``, which fails to connect while
    looking like the service is simply down.

    Args:
        raw: The raw environment value, already stripped and non-empty.

    Returns:
        ``(host, port)`` where port is ``None`` if the value carried none.
    """
    # urlsplit needs a scheme (or a leading '//') to read an authority; with
    # neither it puts everything in `path`. Supplying '//' also gets the
    # bracketed IPv6 form parsed for free.
    candidate = raw if "://" in raw else f"//{raw}"
    parsed = urllib.parse.urlsplit(candidate)
    if not parsed.hostname:
        return raw, None
    try:
        port = parsed.port
    except ValueError:
        # A non-numeric port: keep the hostname, let the default supply a port.
        return parsed.hostname, None
    return parsed.hostname, port


def ollama_env_params() -> dict[str, Any]:
    """Resolve the Ollama endpoint from the environment.

    The single definition of what "the test Ollama" means, so a probe, a
    fixture and a plain helper function all reach the same service. Its
    Postgres counterpart is :func:`postgres_env_params`; the reason both exist
    is the same one — sites that restate the resolution drift apart, and the
    drift shows up as a probe reporting a service down while the code beside
    it talks to that service happily.

    ``OLLAMA_HOST`` supplies the host and may carry a port (see
    :func:`_parse_ollama_host`); an explicit ``OLLAMA_PORT`` is the more
    specific statement and wins over a port embedded in the host.

    Unlike its siblings this has no Docker-aware default: there is no Ollama
    compose service to name, because Ollama runs on the host rather than in
    the dev stack. Inside a container, ``OLLAMA_HOST`` is the way to point at
    it — which is precisely why the URL form above has to be understood.

    Returns:
        A fresh dict with ``host`` and ``port`` (``int``).
    """
    host = "localhost"
    port: int | None = None

    raw_host = os.environ.get("OLLAMA_HOST", "").strip()
    if raw_host:
        host, port = _parse_ollama_host(raw_host)

    raw_port = os.environ.get("OLLAMA_PORT", "").strip()
    if raw_port:
        port = int(raw_port)

    return {"host": host, "port": _OLLAMA_DEFAULT_PORT if port is None else port}


def _resolve_ollama_endpoint(host: str | None, port: int | None) -> tuple[str, int]:
    """Resolve host/port as explicit argument → environment → default."""
    params = ollama_env_params()
    return (
        params["host"] if host is None else host,
        params["port"] if port is None else port,
    )


def _ollama_get_json(
    path: str,
    host: str | None,
    port: int | None,
    timeout: float,
) -> Any | None:
    """GET a JSON document from the resolved Ollama endpoint.

    Returns the decoded body, or ``None`` for any failure — unreachable,
    timeout, HTTP error, or a body that is not JSON. Standard library only:
    ``dataknobs-common`` installs with no required dependencies, so a probe
    that reached for ``requests`` would be unusable in exactly the minimal
    environment a probe is for.
    """
    resolved_host, resolved_port = _resolve_ollama_endpoint(host, port)
    request = urllib.request.Request(f"http://{resolved_host}:{resolved_port}{path}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _read_probe_json(response)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        logger.debug(
            "Ollama request to %s at %s:%s failed: %s", path, resolved_host, resolved_port, exc
        )
        return None


def is_ollama_available(
    host: str | None = None,
    port: int | None = None,
    *,
    timeout: float = 2.0,
) -> bool:
    """Check if the Ollama service is available.

    Resolves the endpoint as ``host``/``port`` args → ``$OLLAMA_HOST`` /
    ``$OLLAMA_PORT`` → ``localhost:11434``, then probes ``GET /api/tags``.

    This asks the service, not this machine. The previous implementation shelled
    out to the local ``ollama`` CLI and took no host or port, so it could not be
    aimed anywhere by a caller, and a container running the suite without the
    binary installed reported the service down while a perfectly reachable
    server answered on the configured endpoint — a silent skip of every
    ``requires_ollama`` test rather than a failure anyone would notice.

    Args:
        host: Ollama host (default: ``$OLLAMA_HOST`` or ``localhost``)
        port: Ollama HTTP port (default: ``$OLLAMA_PORT`` or ``11434``)
        timeout: Per-request timeout in seconds.

    Returns:
        True if Ollama answered, False otherwise
    """
    return _ollama_get_json("/api/tags", host, port, timeout) is not None


def list_ollama_models(
    host: str | None = None,
    port: int | None = None,
    *,
    timeout: float = 5.0,
) -> list[str]:
    """List the models installed on the resolved Ollama endpoint.

    Args:
        host: Ollama host (default: ``$OLLAMA_HOST`` or ``localhost``)
        port: Ollama HTTP port (default: ``$OLLAMA_PORT`` or ``11434``)
        timeout: Per-request timeout in seconds.

    Returns:
        Installed model names, tags included (e.g. ``["gemma3:1b"]``). Empty
        when Ollama is unreachable — indistinguishable from "none installed",
        which is why availability has its own check.
    """
    body = _ollama_get_json("/api/tags", host, port, timeout)
    if not isinstance(body, dict):
        return []
    return [
        str(entry["name"])
        for entry in body.get("models") or []
        if isinstance(entry, dict) and entry.get("name")
    ]


def _model_request_is_satisfied_by(requested: str, installed: str) -> bool:
    """Whether an ``installed`` model name satisfies a ``requested`` one.

    An untagged request accepts any tag of that model — ``gemma3`` is satisfied
    by ``gemma3:1b`` — but not a different model whose name merely begins the
    same way, so ``gemma3`` is *not* satisfied by ``gemma3-uncensored:latest``.
    A request that names a tag must match it exactly.
    """
    if installed == requested:
        return True
    if ":" in requested:
        return False
    return installed.startswith(f"{requested}:")


def is_ollama_model_available(
    model_name: str = "nomic-embed-text",
    host: str | None = None,
    port: int | None = None,
    *,
    timeout: float = 5.0,
) -> bool:
    """Check whether a specific Ollama model is installed.

    Matches against the model names ``/api/tags`` reports, by the rule in
    :func:`_model_request_is_satisfied_by`. The previous implementation
    substring-searched the rendered ``ollama list`` table, where every column
    was as matchable as the name: a request for ``mistral`` was satisfied by an
    installed ``mistral-small``, and requests for ``latest``, ``GB`` and even
    the table's own ``NAME`` header all reported available.

    Args:
        model_name: Name of the model to check, with or without a tag.
        host: Ollama host (default: ``$OLLAMA_HOST`` or ``localhost``)
        port: Ollama HTTP port (default: ``$OLLAMA_PORT`` or ``11434``)
        timeout: Per-request timeout in seconds.

    Returns:
        True if a matching model is installed, False otherwise
    """
    return any(
        _model_request_is_satisfied_by(model_name, installed)
        for installed in list_ollama_models(host, port, timeout=timeout)
    )


def wait_for_ollama(
    host: str | None = None,
    port: int | None = None,
    *,
    max_retries: int = 30,
    delay: float = 1.0,
) -> bool:
    """Block until Ollama answers on the resolved endpoint.

    Args:
        host: Ollama host (default: ``$OLLAMA_HOST`` or ``localhost``)
        port: Ollama HTTP port (default: ``$OLLAMA_PORT`` or ``11434``)
        max_retries: Number of probes before giving up.
        delay: Seconds between probes.

    Returns:
        True once Ollama answers.

    Raises:
        ConnectionError: If Ollama never answered, naming the endpoint tried —
            without which the failure reads as "Ollama is down" when the real
            cause is a probe aimed at the wrong host.
    """
    resolved_host, resolved_port = _resolve_ollama_endpoint(host, port)
    for attempt in range(max_retries):
        if is_ollama_available(resolved_host, resolved_port):
            return True
        if attempt < max_retries - 1:
            time.sleep(delay)
    raise ConnectionError(
        f"Could not connect to Ollama at {resolved_host}:{resolved_port} after "
        f"{max_retries} attempts. Please ensure Ollama is running and accessible."
    )


def is_ollama_model_usable(
    model_name: str,
    *,
    host: str | None = None,
    port: int | None = None,
    prompt: str = "Reply with the single word: ok",
    num_predict: int = 32,
    timeout: float = 60.0,
) -> bool:
    """Check that an Ollama model actually produces usable (non-empty) output.

    A stronger readiness signal than :func:`is_ollama_model_available`, which
    only verifies the model is *listed*. A model can be installed and loaded yet
    return empty output — e.g. a reasoning model exhausting its token budget on
    hidden thinking, or a runtime/template mismatch after an Ollama upgrade. Such
    a runtime passes the "available" check but then fails every live assertion
    with a misleading empty result, so a live-model test suite should gate on
    this stronger check and fall back (or fail with a clear reason) instead.

    Sends one trivial, deterministic (temperature 0) completion via Ollama's
    ``/api/chat`` HTTP endpoint and returns ``True`` only when the response
    carries non-empty message content. Standard-library only (no ``requests``
    dependency). Any error (unreachable, timeout, HTTP error, malformed body)
    returns ``False`` — the caller decides whether that is a skip or a failure.

    The endpoint resolves through :func:`ollama_env_params`, the same path
    :func:`is_ollama_available` uses. It did not, and the two disagreed: with
    ``OLLAMA_HOST`` pointing at an unreachable name the availability check
    reported the service down while this one, hardcoded to ``localhost``,
    reported a model on it ready — one process, two answers about one service.

    Args:
        model_name: Ollama model to probe (e.g. ``"llama3.1:8b"``).
        host: Ollama host (default: ``$OLLAMA_HOST`` or ``localhost``)
        port: Ollama HTTP port (default: ``$OLLAMA_PORT`` or ``11434``)
        prompt: Trivial prompt for the canary generation.
        num_predict: Output-token cap for the canary — kept small; the check is
            "did it produce anything", not "is the answer correct".
        timeout: Per-request timeout in seconds.

    Returns:
        ``True`` if the model returned non-empty content, ``False`` otherwise.
    """
    resolved_host, resolved_port = _resolve_ollama_endpoint(host, port)
    payload = json.dumps(
        {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": num_predict, "temperature": 0.0},
        }
    ).encode()
    request = urllib.request.Request(
        f"http://{resolved_host}:{resolved_port}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = _read_probe_json(response)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        logger.debug("Ollama usability canary for model %r failed: %s", model_name, exc)
        return False
    content = (body.get("message") or {}).get("content") or ""
    return bool(str(content).strip())


def is_faiss_available() -> bool:
    """Check if FAISS is available.

    Returns:
        True if FAISS can be imported, False otherwise
    """
    return importlib.util.find_spec("faiss") is not None


def is_chromadb_available() -> bool:
    """Check if ChromaDB is available.

    Returns:
        True if ChromaDB can be imported, False otherwise
    """
    return importlib.util.find_spec("chromadb") is not None


#: Spellings of ``DOCKER_CONTAINER`` that mean "yes". Anything else —
#: including ``false``, ``0``, ``no`` and ``off`` — means no.
#:
#: An explicit affirmative list rather than a truthiness test, because
#: ``bool(os.environ.get(...))`` reads *every* non-empty string as yes,
#: and ``false`` is precisely the value someone writes to mean the
#: opposite. An unrecognized value resolves to ``localhost``, which is
#: the recoverable direction: a wrong ``localhost`` fails to connect
#: where a wrong compose hostname fails to *resolve*, and a probe that
#: cannot resolve turns into a silent ``requires_*`` skip.
_DOCKER_AFFIRMATIVE = frozenset({"true", "1", "yes", "on"})


def _in_docker_container() -> bool:
    """Whether this process appears to be running inside a container.

    Two independent signals: ``/.dockerenv``, which the runtime creates
    and nobody sets by hand, and an affirmative ``DOCKER_CONTAINER``,
    which is how a compose file or a runner declares it.

    The single definition for the package — the four callers below each
    used to inline it, which is why a defect in the env check was a
    defect in four places at once.
    """
    if os.path.exists("/.dockerenv"):
        return True
    return os.environ.get("DOCKER_CONTAINER", "").strip().lower() in _DOCKER_AFFIRMATIVE


def _docker_aware_default_host(docker_host: str) -> str:
    """Return the compose service hostname inside Docker, else ``localhost``.

    Mirrors the Docker detection the ``*_connection_params`` fixtures use
    (see :func:`_in_docker_container`) so an availability probe and its
    paired fixture resolve the same host. Without this, a probe run
    inside a container — where a service lives at its compose hostname,
    not ``localhost`` — would report the service unavailable and its
    ``requires_*`` marker would false-skip tests that would actually run.

    Args:
        docker_host: The compose service hostname to use inside Docker.

    Returns:
        ``docker_host`` inside Docker, otherwise ``"localhost"``.
    """
    return docker_host if _in_docker_container() else "localhost"


def _is_tcp_service_available(
    host: str | None,
    port: int | None,
    *,
    host_env: str,
    port_env: str,
    docker_host: str,
    default_port: int,
) -> bool:
    """TCP-probe a service, resolving host/port arg → env var → Docker default.

    Shared body for the socket-probe availability checks. Host resolution:
    an explicit ``host`` wins; else ``$<host_env>``; else the Docker-aware
    default (``docker_host`` inside a container, ``localhost`` on the host).
    Port resolution: explicit ``port`` wins; else ``$<port_env>``; else
    ``default_port``.

    Args:
        host: Explicit host, or ``None`` to resolve from env / Docker default.
        port: Explicit port, or ``None`` to resolve from env / default.
        host_env: Environment variable naming the host.
        port_env: Environment variable naming the port.
        docker_host: Compose service hostname used inside Docker.
        default_port: Port used when neither ``port`` nor ``$<port_env>`` is set.

    Returns:
        True if a TCP connection to the resolved host:port succeeds.
    """
    import socket

    if host is None:
        host = os.environ.get(host_env) or _docker_aware_default_host(docker_host)
    if port is None:
        port = int(os.environ.get(port_env, str(default_port)))
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            result = sock.connect_ex((host, port))
        finally:
            sock.close()
        return result == 0
    except OSError:
        return False


def is_redis_available(host: str | None = None, port: int | None = None) -> bool:
    """Check if Redis service is available.

    Resolves the host as ``host`` arg → ``$REDIS_HOST`` → Docker-aware default
    (``redis`` inside a container, ``localhost`` otherwise); the port as
    ``port`` arg → ``$REDIS_PORT`` → ``6379``.

    Args:
        host: Redis host (default: ``$REDIS_HOST`` or the Docker-aware default)
        port: Redis port (default: ``$REDIS_PORT`` or ``6379``)

    Returns:
        True if Redis is available, False otherwise
    """
    return _is_tcp_service_available(
        host,
        port,
        host_env="REDIS_HOST",
        port_env="REDIS_PORT",
        docker_host="redis",
        default_port=6379,
    )


def is_postgres_available(host: str | None = None, port: int | None = None) -> bool:
    """Check if PostgreSQL service is available.

    Resolves the host as ``host`` arg → ``$POSTGRES_HOST`` → Docker-aware
    default (``postgres`` inside a container, ``localhost`` otherwise); the
    port as ``port`` arg → ``$POSTGRES_PORT`` → ``5432``.

    Args:
        host: PostgreSQL host (default: ``$POSTGRES_HOST`` or the Docker-aware
            default)
        port: PostgreSQL port (default: ``$POSTGRES_PORT`` or ``5432``)

    Returns:
        True if PostgreSQL is available, False otherwise
    """
    return _is_tcp_service_available(
        host,
        port,
        host_env="POSTGRES_HOST",
        port_env="POSTGRES_PORT",
        docker_host="postgres",
        default_port=5432,
    )


def is_elasticsearch_available(host: str | None = None, port: int | None = None) -> bool:
    """Check if the Elasticsearch service is available.

    Resolves the host as ``host`` arg → ``$ELASTICSEARCH_HOST`` → Docker-aware
    default (``elasticsearch`` inside a container, ``localhost`` otherwise);
    the port as ``port`` arg → ``$ELASTICSEARCH_PORT`` → ``9200``.

    Args:
        host: Elasticsearch host (default: ``$ELASTICSEARCH_HOST`` or the
            Docker-aware default)
        port: Elasticsearch port (default: ``$ELASTICSEARCH_PORT`` or ``9200``)

    Returns:
        True if Elasticsearch is available, False otherwise
    """
    return _is_tcp_service_available(
        host,
        port,
        host_env="ELASTICSEARCH_HOST",
        port_env="ELASTICSEARCH_PORT",
        docker_host="elasticsearch",
        default_port=9200,
    )


def get_localstack_endpoint(host: str | None = None, port: int | None = None) -> str:
    """Resolve the LocalStack edge endpoint URL.

    Returns the URL form (e.g. ``"http://localhost:4566"``) suitable
    for passing as ``endpoint_url=`` to ``boto3`` / ``aioboto3``
    clients. Pairs with :func:`is_localstack_available`, which uses
    the same resolution chain for its TCP probe.

    Resolution order — highest priority first:

    1. Explicit ``host``/``port`` args (each independent — one may be
       passed without the other).
    2. ``LOCALSTACK_ENDPOINT`` (full URL; scheme optional, normalized
       to ``http://`` if absent).
    3. ``AWS_ENDPOINT_URL`` (full URL; same scheme handling).
    4. ``LOCALSTACK_HOST`` + ``LOCALSTACK_PORT`` env vars.
    5. Default: ``http://localhost:4566``, or
       ``http://localstack:4566`` when running inside a Docker
       container (detected via ``/.dockerenv`` or ``DOCKER_CONTAINER``
       env var — same precedent as
       :func:`postgres_connection_params` and
       :func:`elasticsearch_connection_params`).

    A scheme-less ``LOCALSTACK_ENDPOINT`` / ``AWS_ENDPOINT_URL``
    (e.g. ``host:4566``) fails ``urlparse`` host/port extraction and
    falls through to the env / default arms, so the returned URL is
    always well-formed.

    Args:
        host: Override LocalStack host (skips env resolution for the
            host component).
        port: Override LocalStack edge port (skips env resolution for
            the port component).

    Returns:
        Fully-qualified endpoint URL, scheme included.
    """
    from urllib.parse import urlparse

    scheme = "http"

    if host is None or port is None:
        endpoint = os.environ.get("LOCALSTACK_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL")
        if endpoint:
            parsed = urlparse(endpoint)
            # Only honor the env-supplied scheme when the URL parses as
            # a proper URL (i.e. a hostname is recoverable). For a
            # scheme-less value like ``host:4566`` urlparse returns
            # ``scheme="host"`` / ``hostname=None`` — fall through to
            # the defaults rather than emit a malformed URL.
            if parsed.hostname:
                if host is None:
                    host = parsed.hostname
                if port is None and parsed.port:
                    port = parsed.port
                if parsed.scheme:
                    scheme = parsed.scheme

        if host is None:
            env_host = os.environ.get("LOCALSTACK_HOST")
            host = env_host if env_host else _docker_aware_default_host("localstack")

        if port is None:
            port = int(os.environ.get("LOCALSTACK_PORT", "4566"))

    return f"{scheme}://{host}:{port}"


def _localstack_service_enabled(endpoint: str, service: str) -> bool:
    """Return True if *service* reports running/available on LocalStack.

    Queries ``GET {endpoint}/_localstack/health`` and inspects the
    ``services`` map. A service enabled in the container's ``SERVICES``
    list reports ``"running"`` or ``"available"``; one omitted reports
    ``"disabled"`` (and a starting/erroring one reports something else).

    Any error — unreachable endpoint, unparseable body, unexpected shape,
    or a service not in the ``running``/``available`` state — returns
    ``False``, the same fail-soft "skip, never fail" contract as the TCP
    probe. So a service-specific suite *skips* rather than *errors* on a
    partially-configured LocalStack (e.g. one whose ``SERVICES`` omits the
    service under test).

    Args:
        endpoint: Fully-qualified LocalStack endpoint URL.
        service: LocalStack service key (e.g. ``"sqs"``, ``"s3"``).

    Returns:
        True only when the health endpoint reports the service ready.
    """
    from urllib.error import URLError
    from urllib.request import urlopen

    url = f"{endpoint.rstrip('/')}/_localstack/health"
    try:
        with urlopen(url, timeout=2) as response:
            payload = _read_probe_json(response)
    except (URLError, OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    services = payload.get("services")
    if not isinstance(services, dict):
        return False
    return services.get(service) in ("running", "available")


def is_localstack_available(
    host: str | None = None,
    port: int | None = None,
    *,
    service: str | None = None,
) -> bool:
    """Check if a LocalStack edge endpoint (and optionally a service) is ready.

    Uses :func:`get_localstack_endpoint` to resolve the
    ``(host, port)`` pair so the probe and the URL form share a
    single source of truth. See that function's docstring for the
    resolution chain — including the Docker-aware default that picks
    ``localstack:4566`` inside a container and ``localhost:4566``
    elsewhere.

    By default this only probes **edge-port TCP reachability**. Pass
    ``service`` to additionally require that a specific service is
    **enabled** in the running container (via ``/_localstack/health``):
    a LocalStack started with a restricted ``SERVICES`` list (e.g. ``s3``
    without ``sqs``) is reachable on the edge port but rejects calls to the
    disabled service, so an sqs-specific suite must *skip*, not *fail*.

    Any connection error returns ``False`` (skip, never fail) — the
    same fail-soft contract as the other service probes.

    Args:
        host: LocalStack host (overrides env resolution when given)
        port: LocalStack edge port (overrides env resolution when given)
        service: Optional LocalStack service key (e.g. ``"sqs"``). When
            given, the edge port must be reachable AND the service must
            report ``running``/``available`` at ``/_localstack/health``.

    Returns:
        True if the LocalStack edge port accepts a TCP connection and,
        when ``service`` is given, that service is enabled.
    """
    from urllib.parse import urlparse

    endpoint = get_localstack_endpoint(host, port)
    parsed = urlparse(endpoint)
    probe_host = parsed.hostname or "localhost"
    probe_port = parsed.port or 4566
    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((probe_host, probe_port))
        sock.close()
    except OSError:
        return False
    if result != 0:
        return False
    if service is None:
        return True
    return _localstack_service_enabled(endpoint, service)


async def ensure_localstack_s3_bucket(
    bucket: str,
    endpoint: str | None = None,
    *,
    region: str = "us-east-1",
) -> None:
    """Idempotently create an S3 bucket on LocalStack.

    Designed for integration tests that target the dataknobs dev
    LocalStack container. ``aioboto3`` is lazy-imported so the base
    install of ``dataknobs-common`` stays lean (same pattern as
    :class:`~dataknobs_common.events.SqsEventBus`); install the ``sqs``
    extra to pull it in.

    The helper is safe to call from any test setup:

    - ``head_bucket`` is attempted first; on success the bucket already
      exists and the helper returns immediately.
    - ``NoSuchBucket`` / ``404`` triggers ``create_bucket``.
    - ``BucketAlreadyOwnedByYou`` and ``BucketAlreadyExists`` raised by
      the create call (e.g. a concurrent setup racing this one) are
      swallowed — by the time the call returns, the bucket exists.

    Args:
        bucket: Bucket name to ensure exists.
        endpoint: LocalStack endpoint URL. Defaults to
            :func:`get_localstack_endpoint` (the same resolution chain
            used by :func:`is_localstack_available`).
        region: AWS region to create the bucket in. ``us-east-1`` is
            the LocalStack default and the only region that does NOT
            require a ``CreateBucketConfiguration`` block.

    Raises:
        ClientError: For unexpected S3 errors (network failures,
            permission denied on configured non-LocalStack endpoints).
            ``NoSuchBucket`` and the two "already exists" variants are
            handled internally.
    """
    import aioboto3
    from botocore.exceptions import ClientError

    if endpoint is None:
        endpoint = get_localstack_endpoint()

    session = aioboto3.Session()
    async with session.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    ) as s3:
        try:
            await s3.head_bucket(Bucket=bucket)
            return
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            # head_bucket reports a missing bucket as 404 / NoSuchBucket
            # depending on credentials and the S3 implementation. Treat
            # both as "create needed". Any other ClientError propagates.
            if code not in {"404", "NoSuchBucket", "NotFound"}:
                raise

        try:
            await s3.create_bucket(Bucket=bucket)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            # A concurrent setup may have won the race between the
            # head_bucket above and our create_bucket here. Both
            # variants mean the bucket exists and is usable, which is
            # the contract this helper provides.
            if code not in {
                "BucketAlreadyOwnedByYou",
                "BucketAlreadyExists",
            }:
                raise


def is_bedrock_available() -> bool:
    """Check whether live Amazon Bedrock tests should run.

    Conservative by design and never makes a paid API call: Bedrock has no
    faithful local emulator (LocalStack community / moto do not implement
    ``bedrock-runtime`` inference), so any "reachable" probe would hit the
    paid API. This gates on an explicit opt-in env var — ``DK_TEST_BEDROCK``
    truthy (``1`` / ``true`` / ``yes``) — AND resolvable AWS credentials via
    a lazy botocore session. Absent the opt-in, botocore, or credentials it
    returns ``False`` (skip, never fail) — the same fail-soft contract as
    the other service probes.

    Requiring the opt-in keeps CI from ever invoking Bedrock by accident.

    Returns:
        True if ``DK_TEST_BEDROCK`` is truthy and AWS credentials resolve.
    """
    if os.environ.get("DK_TEST_BEDROCK", "").lower() not in {"1", "true", "yes"}:
        return False
    try:
        import botocore.session
    except ImportError:
        return False
    try:
        credentials = botocore.session.get_session().get_credentials()
    except Exception:
        # Any credential-resolution error → treat as unavailable (skip).
        return False
    return credentials is not None


def is_package_available(package_name: str) -> bool:
    """Check if a Python package is available.

    Args:
        package_name: Name of the package to check

    Returns:
        True if package can be imported, False otherwise
    """
    return importlib.util.find_spec(package_name) is not None


# Pytest Markers


try:
    import pytest

    requires_ollama = pytest.mark.skipif(
        not is_ollama_available(),
        reason="Ollama service not available",
    )

    requires_faiss = pytest.mark.skipif(
        not is_faiss_available(),
        reason="FAISS not installed",
    )

    requires_chromadb = pytest.mark.skipif(
        not is_chromadb_available(),
        reason="ChromaDB not installed",
    )

    requires_redis = pytest.mark.skipif(
        not is_redis_available(),
        reason="Redis not available",
    )

    requires_postgres = pytest.mark.skipif(
        not is_postgres_available(),
        reason="PostgreSQL not available",
    )

    requires_elasticsearch = pytest.mark.skipif(
        not is_elasticsearch_available(),
        reason="Elasticsearch not available",
    )

    requires_localstack = pytest.mark.skipif(
        not is_localstack_available(),
        reason="LocalStack not available",
    )

    def requires_localstack_service(service: str) -> Any:
        """Create a skip marker requiring a specific LocalStack service.

        Unlike :data:`requires_localstack` (edge-port reachability only),
        this skips when the named service is not enabled in the running
        LocalStack (e.g. its ``SERVICES`` list omits it), so a
        service-specific suite *skips* rather than *fails* on a
        partially-configured container.

        Args:
            service: LocalStack service key (e.g. ``"sqs"``).

        Returns:
            pytest.mark.skipif marker.
        """
        return pytest.mark.skipif(
            not is_localstack_available(service=service),
            reason=f"LocalStack service {service!r} not available",
        )

    requires_bedrock = pytest.mark.skipif(
        not is_bedrock_available(),
        reason="Amazon Bedrock live tests require DK_TEST_BEDROCK=true and "
        "resolvable AWS credentials",
    )

    requires_real_postgres = pytest.mark.skipif(
        not is_postgres_available()
        or os.environ.get("TEST_POSTGRES", "").lower() != "true"
        or not is_package_available("asyncpg"),
        reason="real-Postgres behavioural test requires a reachable "
        "server, TEST_POSTGRES=true, and asyncpg installed",
    )

    def requires_package(package_name: str) -> Any:
        """Create a skip marker for a required package.

        Args:
            package_name: Name of the required package

        Returns:
            pytest.mark.skipif marker
        """
        return pytest.mark.skipif(
            not is_package_available(package_name),
            reason=f"{package_name} not installed",
        )

    def requires_ollama_model(
        model_name: str = "nomic-embed-text",
        *,
        host: str | None = None,
        port: int | None = None,
    ) -> Any:
        """Create a skip marker for a required Ollama model.

        Args:
            model_name: Name of the required model
            host: Ollama host (default: ``$OLLAMA_HOST`` or ``localhost``)
            port: Ollama HTTP port (default: ``$OLLAMA_PORT`` or ``11434``)

        Returns:
            pytest.mark.skipif marker
        """
        return pytest.mark.skipif(
            not is_ollama_model_available(model_name, host, port),
            reason=f"Ollama model {model_name} not available",
        )

    def requires_ollama_usable_model(
        model_name: str, *, host: str | None = None, port: int | None = None
    ) -> Any:
        """Create a skip marker requiring an Ollama model that produces output.

        Stronger than :func:`requires_ollama_model` — skips unless the model is
        not only installed but also returns non-empty output (see
        :func:`is_ollama_model_usable`), so a broken/empty-output runtime does
        not surface as misleading per-assertion failures.

        Args:
            model_name: Name of the required model
            host: Ollama host (default: ``$OLLAMA_HOST`` or ``localhost``)
            port: Ollama HTTP port (default: ``$OLLAMA_PORT`` or ``11434``)

        Returns:
            pytest.mark.skipif marker
        """
        return pytest.mark.skipif(
            not is_ollama_model_usable(model_name, host=host, port=port),
            reason=f"Ollama model {model_name} not producing usable output",
        )

except ImportError:
    # pytest not installed - provide placeholder markers.
    #
    # dataknobs-common installs with no required dependencies, so this branch is
    # reachable: pytest is a dev-group dependency and a consumer importing this
    # module without it lands here. Each name was bound to a MarkDecorator above,
    # so rebinding it to None is an `assignment` mismatch -- named on every line
    # rather than waived blank, because a bare directive would go on covering
    # whatever else these lines came to report.
    requires_ollama = None  # type: ignore[assignment]
    requires_faiss = None  # type: ignore[assignment]
    requires_chromadb = None  # type: ignore[assignment]
    requires_redis = None  # type: ignore[assignment]
    requires_postgres = None  # type: ignore[assignment]
    requires_elasticsearch = None  # type: ignore[assignment]
    requires_real_postgres = None  # type: ignore[assignment]
    requires_localstack = None  # type: ignore[assignment]
    requires_bedrock = None  # type: ignore[assignment]

    def requires_localstack_service(service: str) -> Any:
        return None

    def requires_package(package_name: str) -> Any:
        return None

    def requires_ollama_model(
        model_name: str = "nomic-embed-text",
        *,
        host: str | None = None,
        port: int | None = None,
    ) -> Any:
        return None

    def requires_ollama_usable_model(
        model_name: str, *, host: str | None = None, port: int | None = None
    ) -> Any:
        return None


# Test Configuration Factories


def get_test_bot_config(
    use_echo_llm: bool = True,
    use_in_memory_storage: bool = True,
    include_memory: bool = False,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Get a test bot configuration.

    Args:
        use_echo_llm: Use echo LLM instead of real LLM (default: True)
        use_in_memory_storage: Use in-memory conversation storage (default: True)
        include_memory: Include buffer memory configuration (default: False)
        system_prompt: Optional system prompt content

    Returns:
        Bot configuration dictionary suitable for DynaBot.from_config()

    Example:
        ```python
        config = get_test_bot_config(
            use_echo_llm=True,
            system_prompt="You are a test assistant."
        )
        bot = await DynaBot.from_config(config)
        ```
    """
    config: dict[str, Any] = {
        "llm": {
            "provider": "echo" if use_echo_llm else "openai",
            "model": "test" if use_echo_llm else "gpt-4o-mini",
            "temperature": 0.7,
        },
        "conversation_storage": {
            "backend": "memory" if use_in_memory_storage else "file",
        },
    }

    if include_memory:
        config["memory"] = {
            "type": "buffer",
            "max_messages": 10,
        }

    if system_prompt:
        config["system_prompt"] = system_prompt

    return config


def get_test_rag_config(
    use_in_memory_store: bool = True,
    embedding_provider: str = "ollama",
    embedding_model: str = "nomic-embed-text",
) -> dict[str, Any]:
    """Get a test RAG/knowledge base configuration.

    Args:
        use_in_memory_store: Use in-memory vector store (default: True)
        embedding_provider: Embedding provider (default: "ollama")
        embedding_model: Embedding model name (default: "nomic-embed-text")

    Returns:
        Knowledge base configuration dictionary

    Example:
        ```python
        config = get_test_rag_config(use_in_memory_store=True)
        bot_config = get_test_bot_config()
        bot_config["knowledge_base"] = config
        ```
    """
    return {
        "type": "rag",
        "vector_store": {
            "backend": "memory" if use_in_memory_store else "faiss",
            "dimensions": 768,
            "metric": "cosine",
        },
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "chunking": {
            "max_chunk_size": 800,
        },
        "retrieval": {
            "top_k": 5,
            "score_threshold": 0.7,
        },
    }


# Test File Helpers


def create_test_markdown_files(tmp_path: Path) -> list[str]:
    """Create test markdown files for ingestion.

    Args:
        tmp_path: Temporary directory path (from pytest fixture)

    Returns:
        List of created file paths as strings

    Example:
        ```python
        def test_ingestion(tmp_path):
            files = create_test_markdown_files(tmp_path)
            # files contains paths to test markdown documents
        ```
    """
    files = []

    # Create test markdown file 1
    md1 = tmp_path / "test_doc1.md"
    md1.write_text(
        """# Test Document 1

## Introduction

This is a test document for validating ingestion and retrieval.

### Key Points

1. First important point
2. Second important point
3. Third important point

## Details

More detailed information about the topic goes here.
"""
    )
    files.append(str(md1))

    # Create test markdown file 2
    md2 = tmp_path / "test_doc2.md"
    md2.write_text(
        """# Test Document 2

## Overview

Another test document with different content.

## Content

- Item A: Description of item A
- Item B: Description of item B
- Item C: Description of item C

## Summary

This concludes the second test document.
"""
    )
    files.append(str(md2))

    return files


def create_test_json_files(tmp_path: Path) -> list[str]:
    """Create test JSON files.

    Args:
        tmp_path: Temporary directory path (from pytest fixture)

    Returns:
        List of created file paths as strings
    """
    import json

    files = []

    # Create test JSON file 1
    json1 = tmp_path / "test_data1.json"
    json1.write_text(
        json.dumps(
            {
                "title": "Test Data 1",
                "items": [
                    {"id": 1, "name": "Item 1", "value": 100},
                    {"id": 2, "name": "Item 2", "value": 200},
                ],
                "metadata": {"version": "1.0", "created": "2024-01-01"},
            },
            indent=2,
        )
    )
    files.append(str(json1))

    # Create test JSON file 2
    json2 = tmp_path / "test_data2.json"
    json2.write_text(
        json.dumps(
            {
                "title": "Test Data 2",
                "items": [
                    {"id": 3, "name": "Item 3", "value": 300},
                    {"id": 4, "name": "Item 4", "value": 400},
                ],
                "metadata": {"version": "1.0", "created": "2024-01-02"},
            },
            indent=2,
        )
    )
    files.append(str(json2))

    return files
