# Security Constraints

Security violations block deployment. These rules are NON-NEGOTIABLE.

## 1. Input Validation

All external input MUST be validated before use (database queries, HTTP clients, file ops, subprocess calls). Use schema validation (Pydantic, dataclasses with validation, etc.) at system boundaries — public API surfaces, configuration loaders, and data ingestion points.

```python
# CORRECT: Validated at the boundary
class ProviderConfig(BaseModel):
    base_url: HttpUrl
    model: str = Field(min_length=1)
    timeout: float = Field(gt=0, le=300)

# WRONG: Unvalidated input passed through
async def complete(url: str, prompt: str):
    await client.post(url, json={"prompt": prompt})  # No validation!
```

Since dataknobs is a library consumed by many projects, input validation at public API surfaces is critical — downstream consumers inherit our vulnerabilities.

## 2. External HTTP Requests

All HTTP requests (LLM provider calls, HTTP utilities, webhook deliveries) MUST:

- Set explicit timeouts
- Limit response body size where applicable
- Log failures with context but without sensitive data

```python
# CORRECT
async with httpx.AsyncClient(timeout=30.0) as client:
    response = await client.get(url)
    content = response.content[:MAX_RESPONSE_SIZE]

# WRONG: No timeout, unbounded response
async with httpx.AsyncClient() as client:
    response = await client.get(url)
```

This applies to all dataknobs packages that make HTTP calls: `dataknobs-llm` (provider API calls), `dataknobs-utils` (HTTP utilities), `dataknobs-data` (Elasticsearch, S3 backends).

## 3. File Path Security

Never concatenate user input into file paths without validation. This is critical for `dataknobs-data` file backends and `dataknobs-utils` file utilities.

```python
# CORRECT: Validate paths are within expected directories
def safe_path(base: Path, user_input: str) -> Path:
    resolved = (base / user_input).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError("Path traversal attempt detected")
    return resolved

# WRONG: Direct concatenation
def get_file(base: str, name: str) -> Path:
    return Path(base) / name  # "../../../etc/passwd" would work!
```

## 4. Error Handling

- Never use bare `except:` — catch specific exceptions
- Avoid broad `except Exception:` unless re-raising or at a top-level boundary
- Log detailed errors server-side; return generic messages to callers
- Include correlation IDs in log messages when available
- Use lazy formatting in logger calls — f-strings in `logger.exception()` lose stack trace context

```python
# CORRECT: Specific exception, lazy formatting
try:
    result = await provider.complete(messages)
except ProviderConnectionError as e:
    logger.exception("Provider connection failed for request %s: %s", request_id, e.code)
    raise

# WRONG: Broad except, f-string loses stack trace, leaks internals
except Exception as e:
    logger.exception(f"Failed: {e}")
    raise RuntimeError(str(e))  # Leaks internals to caller
```

Use the `dataknobs_common.exceptions` hierarchy for consistent error types across packages.

## 5. Sensitive Data Protection

- **Never log**: API keys, tokens, passwords, credentials, full request bodies with sensitive fields
- **Always log**: correlation IDs, action, timestamp, sanitized context
- Never write credentials to databases — store only encrypted references
- Fail closed — if credentials are invalid/expired, reject the request

```python
# CORRECT: Sanitized logging
logger.info("Provider initialized", extra={
    "provider": config.provider,
    "model": config.model,
    "base_url": config.base_url,
    # api_key deliberately omitted
})

# WRONG: Logging the full config
logger.info("Config: %s", config.to_dict())  # May contain api_key!
```

## 6. SSRF Prevention

When making HTTP requests to URLs from user input or external configuration (e.g., user-configured LLM provider URLs, webhook targets):

- Block localhost, private ranges (10/8, 172.16/12, 192.168/16), link-local (169.254/16)
- Resolve hostnames and validate the resolved IP before connecting
- This is relevant to `dataknobs-llm` providers and `dataknobs-utils` HTTP utilities

```python
import ipaddress
import socket
from urllib.parse import urlparse

BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
]

def is_safe_url(url: str) -> bool:
    parsed = urlparse(url)
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(parsed.hostname))
        return not any(ip in network for network in BLOCKED_NETWORKS)
    except (socket.gaierror, ValueError):
        return False
```

Note: For Ollama (localhost:11434) and other local development services, SSRF checks should be configurable — allow localhost in development but enforce restrictions in production via configuration.

## 7. External Service Configuration

Configuration data that changes over time (service URLs, public keys, feature flags) must be fetched dynamically with cached fallback — never hardcoded.

```python
# CORRECT: Dynamic with cached fallback
async def refresh_config(self) -> None:
    try:
        response = await self._client.get(self._config_url, timeout=10.0)
        self._cached_config = self._parse(response.text)
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("Failed to refresh config, using cached: %s", e)

# WRONG: Hardcoded values that become stale
SUPPORTED_MODELS = ["llama3.2", "mistral", ...]  # Stale on next release!
```

## 8. Security Bypass Flags

When security checks need a bypass for testing or development:

- MUST log at WARNING level when enabled
- MUST default to secure behavior on invalid/missing values (fail closed)
- MUST be impossible to trigger in production via configuration safeguards

```python
# CORRECT: Explicit match, fail closed
bypass_env = os.getenv("DK_SKIP_URL_VALIDATION", "").lower()
if bypass_env == "true":
    logger.warning("URL validation bypassed — development mode only")
    skip_validation = True
else:
    skip_validation = False

# WRONG: "false" string is truthy!
if os.getenv("DK_SKIP_URL_VALIDATION"):
    skip_validation = True
```

## 9. Test-Mode Bypasses

- NEVER use spoofable identifiers (headers, IP strings, user agents) for test detection
- Use environment-based flags checked at startup, not per-request
