"""Vendor error text must not reach a message the API layer discloses.

Every provider's ``_translate_api_error`` folded the vendor's own rendering into
the message of the dataknobs exception it raised. Two of the types that
translation produces are rendered *with their message shown* at the HTTP
boundary — ``ValidationError`` (422) and ``RateLimitError`` (429) — so that
rendering reached the client:

- ``aiohttp.ClientResponseError`` (ollama, huggingface) renders as
  ``400, message='Bad Request', url='http://host:11434/api/chat'`` — the
  endpoint URL verbatim, which on a self-hosted deployment is an internal
  hostname and port.
- ``openai`` / ``anthropic`` render as ``Error code: 400 - <response body>`` —
  the vendor's echo of what it rejected.
- ``botocore`` renders the AWS operation name and the service's message.

The fix builds the message from what the *provider* knows — its family key and
the status — and keeps the vendor rendering for classification and
``__cause__`` only. These tests construct **real** vendor exception objects, so
they exercise the real rendering rather than a guess at it.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

import pytest

from dataknobs_common.exceptions import (
    OperationError,
    RateLimitError,
    ValidationError,
)
from dataknobs_llm.exceptions import ContextLengthExceededError
from dataknobs_llm.llm.base import LLMConfig, LLMProvider

from _aiohttp_error_stub import (
    FakeResponse,
    FakeSession,
    make_client_response_error,
)

# Appears only inside the vendor error's own rendering. Any occurrence in a
# translated message is vendor text that escaped.
SENTINEL = "sentinel-do-not-disclose"
SENTINEL_URL = f"http://{SENTINEL}.internal:65000/v1/generate"


def _provider(module: str, cls: str, key: str) -> LLMProvider:
    """Build a provider without initializing a client (no network)."""
    impl = getattr(importlib.import_module(f"dataknobs_llm.llm.providers.{module}"), cls)
    return impl(LLMConfig(provider=key, model="m", api_key="k"))


def _aiohttp_error(status: int = 400) -> Exception:
    return make_client_response_error(status, "Bad Request", url=SENTINEL_URL)


def _openai_error(status: int = 400) -> Exception:
    import httpx
    import openai

    response = httpx.Response(status, request=httpx.Request("POST", SENTINEL_URL))
    cls = openai.RateLimitError if status == 429 else openai.BadRequestError
    return cls(f"Error code: {status} - {SENTINEL}", response=response, body=None)


def _anthropic_error(status: int = 400) -> Exception:
    import anthropic
    import httpx

    response = httpx.Response(status, request=httpx.Request("POST", SENTINEL_URL))
    cls = anthropic.RateLimitError if status == 429 else anthropic.BadRequestError
    return cls(f"Error code: {status} - {SENTINEL}", response=response, body=None)


def _bedrock_error(status: int = 400) -> Exception:
    from botocore.exceptions import ClientError

    code = "ThrottlingException" if status == 429 else "ValidationException"
    return ClientError(
        {
            "Error": {"Code": code, "Message": SENTINEL},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "InvokeModel",
    )


# (family key, module, class, vendor-error factory)
PROVIDERS: list[tuple[str, str, str, Callable[[int], Exception]]] = [
    ("ollama", "ollama", "OllamaProvider", _aiohttp_error),
    ("huggingface", "huggingface", "HuggingFaceProvider", _aiohttp_error),
    ("openai", "openai", "OpenAIProvider", _openai_error),
    ("anthropic", "anthropic", "AnthropicProvider", _anthropic_error),
    ("bedrock", "bedrock", "BedrockProvider", _bedrock_error),
]

_IDS = [key for key, _, _, _ in PROVIDERS]


class TestNoVendorTextReachesTheMessage:
    """The sweep: every shipped provider, both disclosed statuses."""

    @pytest.mark.parametrize("key,module,cls,make", PROVIDERS, ids=_IDS)
    def test_a_rejected_request_discloses_nothing_of_the_vendors(
        self, key: str, module: str, cls: str, make: Callable[[int], Exception]
    ) -> None:
        """400 → ValidationError, which the API layer renders at 422 with its
        message shown.
        """
        provider = _provider(module, cls, key)
        translated = provider._translate_api_error(make(400))

        assert isinstance(translated, ValidationError)
        assert SENTINEL not in str(translated)
        assert SENTINEL not in str(translated.context)

    @pytest.mark.parametrize("key,module,cls,make", PROVIDERS, ids=_IDS)
    def test_a_throttle_discloses_nothing_of_the_vendors(
        self, key: str, module: str, cls: str, make: Callable[[int], Exception]
    ) -> None:
        """429 → RateLimitError, also rendered with its message shown (at 429)."""
        provider = _provider(module, cls, key)
        translated = provider._translate_api_error(make(429))

        assert isinstance(translated, RateLimitError)
        assert SENTINEL not in str(translated)
        assert SENTINEL not in str(translated.context)

    @pytest.mark.parametrize("key,module,cls,make", PROVIDERS, ids=_IDS)
    def test_the_message_names_the_provider_and_the_status(
        self, key: str, module: str, cls: str, make: Callable[[int], Exception]
    ) -> None:
        """Dropping the vendor text must not leave the message useless."""
        provider = _provider(module, cls, key)
        message = str(provider._translate_api_error(make(400)))

        assert key in message
        assert "400" in message


class TestTheDetailSurvivesWhereItIsSafe:
    """Removed from the response, not from the diagnosis."""

    async def test_the_vendor_rendering_is_still_on_the_cause(self) -> None:
        err = _aiohttp_error(400)
        session = FakeSession([FakeSession.responding(FakeResponse(400, raise_exc=err))])
        provider = _provider("ollama", "OllamaProvider", "ollama")
        provider._session = session
        provider._is_initialized = True

        with pytest.raises(ValidationError) as excinfo:
            await provider.complete("hi")

        assert excinfo.value.__cause__ is err
        assert SENTINEL in str(excinfo.value.__cause__)

    def test_context_length_is_still_classified_from_the_vendor_text(self) -> None:
        """The regression this fix risks.

        Overflow detection reads the vendor's own phrasing. Now that the
        phrasing no longer lands in the message, classification has to read the
        detail instead — otherwise every overflow silently degrades to a plain
        ``ValidationError`` and the caller loses the one 400 it can act on.
        """
        provider = _provider("ollama", "OllamaProvider", "ollama")
        err = make_client_response_error(400, "maximum context length exceeded", url=SENTINEL_URL)

        translated = provider._translate_api_error(err)

        assert isinstance(translated, ContextLengthExceededError)
        assert SENTINEL not in str(translated)

    def test_an_overflow_says_so_without_borrowing_the_vendors_words(self) -> None:
        """The one 400 the caller can act on should not read like the rest.

        The condition is worth naming — the caller can compact history and
        retry — and naming it needs none of the vendor's text: the type has
        already been decided by the time the message is written.
        """
        provider = _provider("ollama", "OllamaProvider", "ollama")
        err = make_client_response_error(400, "maximum context length exceeded", url=SENTINEL_URL)

        overflow = str(provider._translate_api_error(err))
        other_400 = str(provider._translate_api_error(_aiohttp_error(400)))

        assert "context window" in overflow
        assert overflow != other_400
        assert SENTINEL not in overflow

    def test_an_overflow_identified_by_machine_code_still_classifies(self) -> None:
        """OpenAI supplies a ``code``; that path never read the message."""
        import httpx
        import openai

        provider = _provider("openai", "OpenAIProvider", "openai")
        response = httpx.Response(400, request=httpx.Request("POST", SENTINEL_URL))
        err = openai.BadRequestError(
            f"Error code: 400 - {SENTINEL}",
            response=response,
            body={"code": "context_length_exceeded"},
        )

        translated = provider._translate_api_error(err)

        assert isinstance(translated, ContextLengthExceededError)
        assert SENTINEL not in str(translated)


class TestStatuslessFailures:
    """A connection error or timeout carries no status to report."""

    def test_a_connection_error_names_the_provider_without_a_status(self) -> None:
        import aiohttp

        provider = _provider("ollama", "OllamaProvider", "ollama")
        translated = provider._translate_api_error(
            aiohttp.ClientConnectionError(f"cannot connect to {SENTINEL_URL}")
        )

        assert isinstance(translated, OperationError)
        assert SENTINEL not in str(translated)
        assert "ollama" in str(translated)
        assert "HTTP" not in str(translated)


class TestTheMessageIsNotTheProvidersToWrite:
    """The structural half: a provider cannot influence the disclosed message.

    Before the fix each provider passed its own message string, so closing the
    leak in the five shipped providers would leave the sixth — a consumer's own
    provider subclass — free to reintroduce it. The disclosed message is now
    built by the shared dispatcher from the provider's family key and the
    status, and a provider supplies only classification material.
    """

    def test_a_consumer_provider_gets_a_clean_message_for_free(self) -> None:
        provider = _provider("ollama", "OllamaProvider", "ollama")

        translated = provider._dataknobs_error_for_status(
            400, f"whatever the vendor said, including {SENTINEL}"
        )

        assert SENTINEL not in str(translated)

    def test_a_declared_family_key_is_what_the_message_names(self) -> None:
        """``provider_name`` is assignable for a gateway whose family the config
        cannot name; the message follows it.
        """
        provider = _provider("ollama", "OllamaProvider", "ollama")
        provider.provider_name = "acme-gateway"

        assert "acme-gateway" in str(provider._dataknobs_error_for_status(400, "detail"))


class TestTranslationPolicyUnchanged:
    """The status→type mapping this fix must not disturb."""

    @pytest.mark.parametrize(
        "status,expected",
        [
            (400, ValidationError),
            (401, OperationError),
            (403, OperationError),
            (429, RateLimitError),
            (500, OperationError),
            (None, OperationError),
        ],
    )
    def test_status_maps_to_the_same_type_as_before(
        self, status: int | None, expected: type[Exception]
    ) -> None:
        provider = _provider("ollama", "OllamaProvider", "ollama")
        assert isinstance(provider._dataknobs_error_for_status(status, "detail"), expected)

    def test_retry_after_still_rides_along(self) -> None:
        provider = _provider("ollama", "OllamaProvider", "ollama")
        translated: Any = provider._dataknobs_error_for_status(429, "detail", retry_after=7.5)
        assert translated.retry_after == 7.5
