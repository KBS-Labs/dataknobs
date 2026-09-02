"""LLM resource provider for language model interactions.

Note: This module was migrated from dataknobs_fsm.resources.llm to
consolidate all LLM functionality in the dataknobs-llm package.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field as dataclass_field
from typing import TYPE_CHECKING, Any, Dict, List, Union, cast
from enum import Enum

from dataknobs_common.lifecycle import close_if_owned_sync
from dataknobs_common.ratelimit import InMemoryRateLimiter, RateLimit, RateLimiterConfig
from dataknobs_fsm.functions.base import ResourceError
from dataknobs_fsm.resources.base import (
    BaseResourceProvider,
    ResourceHealth,
    ResourceStatus,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dataknobs_llm.llm.base import AsyncLLMProvider
    from dataknobs_llm.llm.providers.base import SyncProviderAdapter

logger = logging.getLogger(__name__)


def _as_vectors(result: Any) -> List[List[float]]:
    """Normalize a provider's embed result to one vector per input text.

    A provider handed a single string may answer with one flat vector rather
    than a list holding one. Both classes here promise the nested shape, and
    both used to unpack it themselves.
    """
    if result and isinstance(result[0], (int, float)):
        return [cast("List[float]", result)]
    return cast("List[List[float]]", result)


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    HUGGINGFACE_INFERENCE = "huggingface_inference"  # HF Inference API
    CUSTOM = "custom"


@dataclass
class LLMSession:
    """LLM session with configuration and state."""

    provider: LLMProvider
    model_name: str
    api_key: str | None = None
    endpoint: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Rate limiting (mainly for commercial APIs)
    requests_per_minute: int = 60
    tokens_per_minute: int = 90000
    request_count: int = 0
    token_count: int = 0
    window_start: float = dataclass_field(default_factory=time.time)

    # Token tracking
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_requests: int = 0

    # Provider-specific settings
    provider_config: Dict[str, Any] = dataclass_field(default_factory=dict)

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage.

        Args:
            prompt_tokens: Number of prompt tokens used.
            completion_tokens: Number of completion tokens generated.
        """
        total_tokens = prompt_tokens + completion_tokens

        self.request_count += 1
        self.token_count += total_tokens
        self.total_requests += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens


class LLMResource(BaseResourceProvider):
    """LLM resource provider for language model operations.

    Supports multiple providers:
    - OpenAI: GPT models via OpenAI API
    - Anthropic: Claude models via Anthropic API
    - Ollama: Local models via Ollama
    - HuggingFace: Local transformers or Inference API
    """

    def __init__(
        self,
        name: str,
        provider: Union[str, LLMProvider] = "ollama",
        model: str = "llama2",
        api_key: str | None = None,
        endpoint: str | None = None,
        **config: Any,
    ) -> None:
        """Initialize LLM resource.

        Args:
            name: Resource name.
            provider: LLM provider (ollama, openai, anthropic, huggingface, etc).
            model: Model name/identifier.
            api_key: API key for commercial providers.
            endpoint: Custom endpoint URL.
            **config: Additional configuration.
        """
        super().__init__(name, config)

        self._set_provider_identity(provider)

        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint or self._get_default_endpoint()

        # Initialize provider-specific clients
        self._client = None
        self._providers: Dict[str, SyncProviderAdapter] = {}
        self._initialize_client()

        self._sessions: Dict[int, LLMSession] = {}
        self.status = ResourceStatus.IDLE

    def _set_provider_identity(self, provider: Union[str, LLMProvider]) -> None:
        """Record both spellings of the provider this resource speaks to.

        The FSM-side enum is a subset of what ``create_llm_provider``
        supports -- ``echo`` is a provider and not an enum member -- so the
        raw string is kept beside the enum rather than collapsed into
        ``CUSTOM``. Collapsing it is what sent a working provider to a
        fabricating else-branch: the enum could not name it, so nothing
        could delegate to it.
        """
        if isinstance(provider, str):
            self._provider_name = provider.lower()
            try:
                self.provider = LLMProvider(self._provider_name)
            except ValueError:
                self.provider = LLMProvider.CUSTOM
        else:
            self.provider = provider
            self._provider_name = provider.value

    def _get_default_endpoint(self) -> str | None:
        """Get default endpoint for provider.

        Returns:
            Default endpoint URL or None.
        """
        defaults = {
            LLMProvider.OPENAI: "https://api.openai.com/v1",
            LLMProvider.ANTHROPIC: "https://api.anthropic.com/v1",
            LLMProvider.OLLAMA: "http://localhost:11434",
            LLMProvider.HUGGINGFACE_INFERENCE: "https://api-inference.huggingface.co/models",
        }
        return defaults.get(self.provider)

    def _initialize_client(self) -> None:
        """Initialize provider-specific client."""
        try:
            if self.provider == LLMProvider.OLLAMA:
                # Ollama uses HTTP API, no special client needed
                # Just verify endpoint is accessible
                import urllib.request

                try:
                    req = urllib.request.Request(f"{self.endpoint}/api/tags")
                    with urllib.request.urlopen(req, timeout=5) as response:
                        if response.status == 200:
                            self.status = ResourceStatus.IDLE
                except Exception:
                    # Ollama might not be running yet, that's ok
                    self.status = ResourceStatus.IDLE

            elif self.provider == LLMProvider.HUGGINGFACE:
                # For local HuggingFace transformers
                # We'll lazy-load the model when needed
                self.status = ResourceStatus.IDLE

            elif self.provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC]:
                # Commercial APIs - just verify we have API key
                if not self.api_key:
                    raise ResourceError(
                        f"{self.provider.value} requires an API key",
                        resource_name=self.name,
                        operation="initialize",
                    )
                self.status = ResourceStatus.IDLE

            else:
                self.status = ResourceStatus.IDLE

        except Exception as e:
            self.status = ResourceStatus.ERROR
            # Bounded message: an SDK client constructor reports a bad
            # credential or an unreachable base URL by naming it. The provider
            # family is ours and the type name is a class name; the rest
            # travels on __cause__.
            raise ResourceError(
                f"Failed to initialize {self.provider.value} client ({type(e).__name__})",
                resource_name=self.name,
                operation="initialize",
            ) from e

    def acquire(self, **kwargs: Any) -> LLMSession:
        """Acquire an LLM session.

        Args:
            **kwargs: Session configuration overrides.

        Returns:
            LLMSession instance.

        Raises:
            ResourceError: If acquisition fails.
        """
        try:
            # Set provider-specific defaults
            if self.provider == LLMProvider.OLLAMA:
                # Ollama defaults
                kwargs.setdefault("temperature", 0.8)
                kwargs.setdefault("requests_per_minute", 0)  # No limit
                kwargs.setdefault("tokens_per_minute", 0)  # No limit

            elif self.provider == LLMProvider.HUGGINGFACE:
                # HuggingFace local defaults
                kwargs.setdefault("device", "cpu")  # or "cuda" if available
                kwargs.setdefault("requests_per_minute", 0)  # No limit

            session = LLMSession(
                provider=self.provider,
                model_name=kwargs.get("model", self.model),
                api_key=kwargs.get("api_key", self.api_key),
                endpoint=kwargs.get("endpoint", self.endpoint),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0),
                requests_per_minute=kwargs.get("requests_per_minute", 60),
                tokens_per_minute=kwargs.get("tokens_per_minute", 90000),
                provider_config=kwargs.get("provider_config", {}),
            )

            session_id = id(session)
            self._sessions[session_id] = session
            self._resources.append(session)

            self.status = ResourceStatus.ACTIVE
            return session

        except Exception as e:
            self.status = ResourceStatus.ERROR
            raise ResourceError(
                f"Failed to acquire LLM session ({type(e).__name__})",
                resource_name=self.name,
                operation="acquire",
            ) from e

    def release(self, resource: Any) -> None:
        """Release an LLM session.

        Args:
            resource: The LLMSession to release.
        """
        if isinstance(resource, LLMSession):
            session_id = id(resource)
            if session_id in self._sessions:
                del self._sessions[session_id]

            if resource in self._resources:
                self._resources.remove(resource)

        if not self._resources:
            self.status = ResourceStatus.IDLE

    def validate(self, resource: Any) -> bool:
        """Validate an LLM session.

        Args:
            resource: The LLMSession to validate.

        Returns:
            True if the session is valid.
        """
        if not isinstance(resource, LLMSession):
            return False

        # Check if API key is set for commercial providers
        if resource.provider in [
            LLMProvider.OPENAI,
            LLMProvider.ANTHROPIC,
            LLMProvider.HUGGINGFACE_INFERENCE,
        ]:
            if not resource.api_key:
                return False

        return True

    def health_check(self) -> ResourceHealth:
        """Check LLM service health.

        Returns:
            Health status.
        """
        session = None
        try:
            session = self.acquire()

            if session.provider == LLMProvider.OLLAMA:
                # Check Ollama API
                import urllib.request

                req = urllib.request.Request(f"{session.endpoint}/api/tags")
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status == 200:
                        self.metrics.record_health_check(True)
                        return ResourceHealth.HEALTHY

            elif session.provider == LLMProvider.HUGGINGFACE:
                # For local HF, just check if transformers is available
                try:
                    import importlib.util

                    if importlib.util.find_spec("transformers"):
                        self.metrics.record_health_check(True)
                        return ResourceHealth.HEALTHY
                    else:
                        self.metrics.record_health_check(False)
                        return ResourceHealth.UNHEALTHY
                except ImportError:
                    self.metrics.record_health_check(False)
                    return ResourceHealth.UNHEALTHY

            else:
                # For commercial APIs, assume healthy if session is valid
                if self.validate(session):
                    self.metrics.record_health_check(True)
                    return ResourceHealth.HEALTHY

        except Exception:
            self.metrics.record_health_check(False)
            return ResourceHealth.UNHEALTHY
        finally:
            if session:
                self.release(session)

        return ResourceHealth.UNKNOWN

    def _provider_config(self, model: str) -> Any:
        """Build the provider config this resource stands for.

        One place, so ``complete`` and ``embed`` cannot disagree about which
        credential, endpoint or width is in force. They did disagree: the
        OpenAI methods read ``kwargs`` and then the environment, so a
        resource built with an explicit ``api_key`` reported that no key was
        provided, and a configured ``endpoint`` never reached the client at
        all -- every call went to the vendor's default host.
        """
        from dataknobs_llm.llm.base import LLMConfig as ProviderLLMConfig

        return ProviderLLMConfig(
            provider=self._provider_name,
            model=model,
            api_key=self.api_key,
            api_base=self.endpoint,
            dimensions=self.config.get("dimensions"),
        )

    def _sync_provider(self, model: str | None = None) -> SyncProviderAdapter:
        """The provider this resource delegates to, built once and reused.

        ``AsyncLLMResource`` has held its provider across calls since it was
        written. The sync half built a fresh one per call, which is why the
        HuggingFace path re-loaded a tokenizer and a model every time it was
        asked for a vector.

        Keyed by model, which is almost always just this resource's own.
        A caller naming a different embedding model gets a second provider
        rather than a mutated first one, because ``embed`` has no per-call
        config surface: ``config_overrides`` is a completion-only parameter
        and every provider's ``embed`` reads ``self.config.model`` directly.
        Passing a model somewhere it is *not* read from would honour it in
        the signature and drop it in fact, which is the defect this class is
        being repaired for.

        Named apart from ``AsyncLLMResource._get_provider``, which is a
        coroutine: ``AsyncLLMResource`` does not override ``complete``, so
        the sync one runs on the async class too and would have resolved to
        the override and awaited nothing.
        """
        key = model or self.model
        provider = self._providers.get(key)
        if provider is None:
            from dataknobs_llm.llm.providers import create_llm_provider

            try:
                provider = create_llm_provider(self._provider_config(key), is_async=False)
                provider.initialize()
            except Exception as e:
                raise self._operation_error(e, "initialize") from e
            self._providers[key] = provider
        return provider

    def _operation_error(self, exc: Exception, operation: str) -> ResourceError:
        """Name the resource and the operation; let ``__cause__`` carry the rest.

        Bounded for the reason ``_initialize_client`` already records: an SDK
        client reports a bad credential or an unreachable base URL by naming
        it, and a message is not the place for either. The provider family
        and the exception's class name are ours to say.
        """
        logger.error(
            "LLM resource operation failed",
            extra={
                "resource": self.name,
                "operation": operation,
                "provider": self._provider_name,
                "model": self.model,
                "error_type": type(exc).__name__,
            },
        )
        return ResourceError(
            f"{self._provider_name} {operation} failed ({type(exc).__name__})",
            resource_name=self.name,
            operation=operation,
        )

    def _session_overrides(
        self, session: LLMSession, call_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Per-call overrides, so one cached provider can serve every session.

        A session may name a different model or sampling than the resource
        was built with -- ``acquire()`` gives Ollama a temperature of its
        own, for one. The provider is built once from the resource and told
        the difference per call, which is the shape ``generate`` has always
        used. Consumed keys are removed from ``call_kwargs`` in place so they
        are not also forwarded as raw provider arguments.
        """
        overrides: Dict[str, Any] = {}
        if session.model_name != self.model:
            overrides["model"] = session.model_name
        for key in ("temperature", "max_tokens", "top_p"):
            value = call_kwargs.pop(key, getattr(session, key, None))
            if value is not None:
                overrides[key] = value
        return overrides

    @staticmethod
    def _as_completion_dict(response: Any) -> Dict[str, Any]:
        """The dict shape this resource's callers read.

        Written out three times before this: once in each of the two
        per-provider completions, and once in ``AsyncLLMResource.generate``,
        which now calls this instead.
        """
        return {
            "choices": [
                {
                    "text": response.content,
                    "index": 0,
                    "finish_reason": response.finish_reason or "stop",
                }
            ],
            "model": response.model,
            "usage": response.usage,
        }

    def complete(
        self, prompt: str, session: LLMSession | None = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate a completion for the given prompt.

        Args:
            prompt: Input prompt.
            session: Optional session to use.
            **kwargs: ``temperature``, ``max_tokens`` and ``top_p`` override
                the session's; anything else is forwarded to the provider.
                Credentials are not read from here -- they come from the
                resource, which is the only place that can hold them
                consistently for both operations.

        Returns:
            Completion response with text and metadata.

        Raises:
            ResourceError: If the provider cannot serve the completion. The
                provider's own exception travels on ``__cause__``. This used
                to be returned instead, as ``{"choices": [{"text": "Error:
                ..."}]}`` -- a failure in the field a caller reads as the
                model's own words.
        """
        from dataknobs_llm.llm.base import LLMMessage

        if session is None:
            session = self.acquire()
            should_release = True
        else:
            should_release = False

        try:
            provider = self._sync_provider()
            call_kwargs = dict(kwargs)
            overrides = self._session_overrides(session, call_kwargs)

            try:
                response = provider.complete(
                    [LLMMessage(role="user", content=prompt)],
                    config_overrides=overrides or None,
                    **call_kwargs,
                )
            except Exception as e:
                raise self._operation_error(e, "complete") from e

            if response.usage:
                session.record_usage(
                    response.usage.get("prompt_tokens", 0),
                    response.usage.get("completion_tokens", 0),
                )
            return self._as_completion_dict(response)

        finally:
            if should_release:
                self.release(session)

    def embed(
        self,
        text: Union[str, List[str]],
        session: LLMSession | None = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings for text.

        Args:
            text: Text or list of texts to embed.
            session: Optional session to use.
            **kwargs: ``embed_model`` selects the embedding model for this
                call, overriding the resource's; ``dimensions`` states the
                width the vectors must be, and a stated width is honoured or
                refused, never ignored. Anything else is forwarded to the
                provider.

        Returns:
            One vector per input text, whether one text or many was given.

        Raises:
            ResourceError: If the provider cannot produce the embeddings.
                The provider's own exception travels on ``__cause__`` --
                including the ``NotImplementedError`` a provider with no
                embeddings API raises, which this method used to answer with
                invented constant vectors instead.
        """
        if session is None:
            session = self.acquire()
            should_release = True
        else:
            should_release = False

        try:
            call_kwargs = dict(kwargs)
            provider = self._sync_provider(call_kwargs.pop("embed_model", None))

            try:
                vectors = provider.embed(text, **call_kwargs)
            except Exception as e:
                raise self._operation_error(e, "embed") from e

            return _as_vectors(vectors)

        finally:
            if should_release:
                self.release(session)

    def close(self) -> None:
        """Release sessions, and close the providers this resource built.

        Ownership is unconditional on this half: the sync class has no
        parameter for handing one in, so every provider in the map was built
        here. The guard is still the shared one, so the sync half reads the
        same as every other owned-teardown site rather than inventing a
        second spelling for the same decision.

        ``AsyncLLMResource``'s async provider needs an await and is not
        touched here -- the asymmetry between this method and ``aclose()``,
        including the injected provider that class does accept, is its own
        question, and this change neither widens nor narrows it.
        """
        for provider in self._providers.values():
            close_if_owned_sync(provider, True, on_error=self._log_close_error)
        self._providers.clear()
        super().close()

    def _log_close_error(self, exc: Exception) -> None:
        """One provider failing to close must not strand the others."""
        logger.warning(
            "LLM provider close failed",
            extra={"resource": self.name, "error_type": type(exc).__name__},
        )

    def get_usage_stats(self, session: LLMSession) -> Dict[str, Any]:
        """Get usage statistics for a session.

        Args:
            session: LLM session.

        Returns:
            Usage statistics.
        """
        stats: Dict[str, Any] = {
            "provider": session.provider.value,
            "model": session.model_name,
            "total_requests": session.total_requests,
        }

        # Add token stats for providers that track them
        if session.provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.OLLAMA]:
            stats.update(
                {
                    "total_prompt_tokens": session.total_prompt_tokens,
                    "total_completion_tokens": session.total_completion_tokens,
                    "total_tokens": session.total_prompt_tokens + session.total_completion_tokens,
                }
            )

        # Add rate limit info for commercial providers
        if session.provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC]:
            stats["rate_limits"] = {
                "requests_per_minute": session.requests_per_minute,
                "tokens_per_minute": session.tokens_per_minute,
                "current_window": {
                    "requests": session.request_count,
                    "tokens": session.token_count,
                    "window_start": session.window_start,
                },
            }

        return stats


class AsyncLLMResource(LLMResource):
    """Async LLM resource with native async providers and rate limiting.

    Extends LLMResource with:
    - async ``generate()`` — the method ``LLMCaller.transform()`` expects
    - async ``embed()`` — the method ``EmbeddingGenerator.transform()`` expects
    - ``InMemoryRateLimiter`` integration for request rate limiting
    - Persistent async provider (created once, reused across calls)

    Usage::

        resource = AsyncLLMResource("llm", provider="ollama", model="llama3.2")
        await resource.ainitialize()
        response = await resource.generate(prompt="Hello")
        await resource.aclose()
    """

    def __init__(
        self,
        name: str,
        provider: Union[str, LLMProvider] = "ollama",
        model: str = "llama3.2",
        api_key: str | None = None,
        endpoint: str | None = None,
        async_provider: AsyncLLMProvider | None = None,
        **config: Any,
    ) -> None:
        """Initialize async LLM resource.

        Skips the blocking ``_initialize_client()`` from ``LLMResource`` — use
        ``ainitialize()`` instead to set up the async provider, or pass a
        pre-built provider via ``async_provider``.

        Args:
            name: Resource name.
            provider: LLM provider name. Accepts any string that
                ``create_llm_provider()`` supports (e.g. ``"ollama"``,
                ``"openai"``, ``"echo"``), or an ``LLMProvider`` enum value.
            model: Model name/identifier.
            api_key: API key for commercial providers.
            endpoint: Custom endpoint URL.
            async_provider: Pre-built ``AsyncLLMProvider`` instance. When
                provided, ``ainitialize()`` is skipped and this provider is
                used directly. Useful for testing (with ``EchoProvider``) or
                when the caller manages provider lifecycle externally.
            **config: Additional configuration (``requests_per_minute``, etc).
        """
        # Call BaseResourceProvider directly (skip LLMResource._initialize_client)
        BaseResourceProvider.__init__(self, name, config)

        self._set_provider_identity(provider)

        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint or self._get_default_endpoint()
        self._client = None
        self._providers: Dict[str, SyncProviderAdapter] = {}
        self._sessions: dict[int, LLMSession] = {}
        self.status = ResourceStatus.IDLE

        # Async provider — either injected or lazy-initialized via ainitialize
        self._async_provider: AsyncLLMProvider | None = async_provider

        # Rate limiter from config
        rpm = config.get("requests_per_minute", 0)
        if rpm and rpm > 0:
            limiter_config = RateLimiterConfig(
                default_rates=[RateLimit(limit=rpm, interval=60)],
            )
            self._rate_limiter: InMemoryRateLimiter | None = InMemoryRateLimiter(limiter_config)
        else:
            self._rate_limiter = None

    async def ainitialize(self) -> None:
        """Create and initialize the async LLM provider.

        Uses the raw provider string (not the enum value) so that any
        provider supported by ``create_llm_provider()`` works — including
        ``"echo"`` for testing.
        """
        self._async_provider = await self._build_async_provider()

    async def _build_async_provider(self) -> AsyncLLMProvider:
        """Build and initialize one async provider from this resource.

        Separate from ``ainitialize`` so the lazy path can hold the result
        as a value rather than reading back an attribute it just wrote --
        which is the difference between a return type the checker can
        narrow and one it has to be asserted into.
        """
        from dataknobs_llm.llm.providers import create_llm_provider

        provider = create_llm_provider(self._provider_config(self.model), is_async=True)
        await provider.initialize()
        logger.info(
            "AsyncLLMResource initialized",
            extra={"provider": self._provider_name, "model": self.model},
        )
        return provider

    async def _get_provider(self) -> AsyncLLMProvider:
        """Get or lazily initialize the async provider.

        Returns:
            The initialized ``AsyncLLMProvider``.
        """
        if self._async_provider is None:
            self._async_provider = await self._build_async_provider()
        return self._async_provider

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncIterator[Any]:
        """Generate a completion asynchronously.

        This is the method ``LLMCaller.transform()`` expects via
        ``await resource.generate()``.

        Args:
            prompt: Input prompt text.
            system_prompt: Optional system prompt.
            model: Override model for this request.
            temperature: Override temperature.
            max_tokens: Override max tokens.
            stream: If True, return an async iterator of streaming chunks.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Completion response dict (or async iterator when streaming).
        """
        from dataknobs_llm.llm.base import LLMMessage

        session = self.acquire()
        try:
            # Rate limit check
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire(self.provider.value)

            llm_provider = await self._get_provider()

            # Build messages
            messages: list[LLMMessage] = []
            if system_prompt:
                messages.append(LLMMessage(role="system", content=system_prompt))
            messages.append(LLMMessage(role="user", content=prompt))

            # Config overrides
            overrides: dict[str, Any] = {}
            if model:
                overrides["model"] = model
            if temperature is not None:
                overrides["temperature"] = temperature
            if max_tokens is not None:
                overrides["max_tokens"] = max_tokens

            if stream:
                return llm_provider.stream_complete(messages, config_overrides=overrides or None)

            response = await llm_provider.complete(messages, config_overrides=overrides or None)

            # Record usage on session
            if response.usage:
                session.record_usage(
                    response.usage.get("prompt_tokens", 0),
                    response.usage.get("completion_tokens", 0),
                )

            return self._as_completion_dict(response)
        finally:
            self.release(session)

    async def embed(  # type: ignore[override]
        self,
        text: Union[str, List[str]],
        session: LLMSession | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Generate embeddings asynchronously.

        Overrides the sync ``LLMResource.embed()`` with an async
        implementation using the async provider.

        Args:
            text: Text or list of texts to embed.
            session: Optional session to use.
            **kwargs: Additional parameters.

        Returns:
            List of embedding vectors.
        """
        if session is None:
            session = self.acquire()
            should_release = True
        else:
            should_release = False

        try:
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire(self.provider.value)

            llm_provider = await self._get_provider()
            result = await llm_provider.embed(text, **kwargs)

            return _as_vectors(result)
        finally:
            if should_release:
                self.release(session)

    async def aclose(self) -> None:
        """Close async provider and rate limiter resources."""
        if self._async_provider is not None:
            await self._async_provider.close()
            self._async_provider = None
        if self._rate_limiter is not None:
            await self._rate_limiter.close()
