"""LLM resource provider for language model interactions.

Note: This module was migrated from dataknobs_fsm.resources.llm to
consolidate all LLM functionality in the dataknobs-llm package.
"""

import json
import os
import time
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Union
from enum import Enum

from dataknobs_fsm.functions.base import ResourceError
from dataknobs_fsm.resources.base import (
    BaseResourceProvider,
    ResourceHealth,
    ResourceStatus,
)


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
    
    def check_rate_limits(self, estimated_tokens: int = 0) -> bool:
        """Check if request would exceed rate limits.
        
        Args:
            estimated_tokens: Estimated tokens for the request.
            
        Returns:
            True if request can proceed, False if rate limited.
        """
        # Local providers don't have rate limits
        if self.provider in [LLMProvider.OLLAMA, LLMProvider.HUGGINGFACE]:
            return True
        
        current_time = time.time()
        window_elapsed = current_time - self.window_start
        
        # Reset window if a minute has passed
        if window_elapsed >= 60:
            self.request_count = 0
            self.token_count = 0
            self.window_start = current_time
            return True
        
        # Check limits
        if self.request_count >= self.requests_per_minute:
            return False
        
        if self.token_count + estimated_tokens > self.tokens_per_minute:
            return False
        
        return True
    
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
        **config
    ):
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
        
        # Convert string to enum
        if isinstance(provider, str):
            try:
                self.provider = LLMProvider(provider.lower())
            except ValueError:
                self.provider = LLMProvider.CUSTOM
        else:
            self.provider = provider
        
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint or self._get_default_endpoint()
        
        # Initialize provider-specific clients
        self._client = None
        self._initialize_client()
        
        self._sessions = {}
        self.status = ResourceStatus.IDLE
    
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
                        operation="initialize"
                    )
                self.status = ResourceStatus.IDLE
                
            else:
                self.status = ResourceStatus.IDLE
                
        except Exception as e:
            self.status = ResourceStatus.ERROR
            raise ResourceError(
                f"Failed to initialize {self.provider.value} client: {e}",
                resource_name=self.name,
                operation="initialize"
            ) from e
    
    def acquire(self, **kwargs) -> LLMSession:
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
                provider_config=kwargs.get("provider_config", {})
            )
            
            session_id = id(session)
            self._sessions[session_id] = session
            self._resources.append(session)
            
            self.status = ResourceStatus.ACTIVE
            return session
            
        except Exception as e:
            self.status = ResourceStatus.ERROR
            raise ResourceError(
                f"Failed to acquire LLM session: {e}",
                resource_name=self.name,
                operation="acquire"
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
        if resource.provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, 
                                 LLMProvider.HUGGINGFACE_INFERENCE]:
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
                    if importlib.util.find_spec('transformers'):
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
    
    def complete(
        self,
        prompt: str,
        session: LLMSession | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a completion for the given prompt.
        
        Args:
            prompt: Input prompt.
            session: Optional session to use.
            **kwargs: Additional parameters.
            
        Returns:
            Completion response with text and metadata.
        """
        if session is None:
            session = self.acquire()
            should_release = True
        else:
            should_release = False
        
        try:
            # Route to appropriate provider
            if session.provider == LLMProvider.OLLAMA:
                response = self._ollama_complete(session, prompt, **kwargs)
            elif session.provider == LLMProvider.HUGGINGFACE:
                response = self._huggingface_complete(session, prompt, **kwargs)
            elif session.provider == LLMProvider.OPENAI:
                response = self._openai_complete(session, prompt, **kwargs)
            elif session.provider == LLMProvider.ANTHROPIC:
                response = self._anthropic_complete(session, prompt, **kwargs)
            else:
                response = self._custom_complete(session, prompt, **kwargs)
            
            # Record usage if available
            if "usage" in response:
                prompt_tokens = response["usage"].get("prompt_tokens", 0)
                completion_tokens = response["usage"].get("completion_tokens", 0)
                session.record_usage(prompt_tokens, completion_tokens)
            
            return response
            
        finally:
            if should_release:
                self.release(session)
    
    def _ollama_complete(
        self,
        session: LLMSession,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Ollama completion.
        
        Args:
            session: LLM session.
            prompt: Input prompt.
            **kwargs: Additional parameters.
            
        Returns:
            Completion response.
        """
        import urllib.request
        import urllib.parse
        
        data = {
            "model": session.model_name,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", session.temperature),
            "max_tokens": kwargs.get("max_tokens", session.max_tokens),
            "stream": False
        }
        
        req = urllib.request.Request(
            f"{session.endpoint}/api/generate",
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read())
            
        return {
            "choices": [{
                "text": result.get("response", ""),
                "index": 0,
                "finish_reason": "stop" if result.get("done") else "length"
            }],
            "model": session.model_name,
            "usage": {
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0),
                "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
            }
        }
    
    def _huggingface_complete(
        self,
        session: LLMSession,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """HuggingFace local completion.
        
        Args:
            session: LLM session.
            prompt: Input prompt.
            **kwargs: Additional parameters.
            
        Returns:
            Completion response.
        """
        # This would use transformers library for local inference
        # Placeholder for now
        try:
            from transformers import pipeline
            
            # Lazy load the model
            pipe = pipeline(
                "text-generation",
                model=session.model_name,
                device=session.provider_config.get("device", "cpu")
            )
            
            result = pipe(
                prompt,
                max_length=kwargs.get("max_tokens", session.max_tokens),
                temperature=kwargs.get("temperature", session.temperature),
                top_p=kwargs.get("top_p", session.top_p),
            )
            
            generated_text = result[0]["generated_text"]
            # Remove the prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            
            return {
                "choices": [{
                    "text": generated_text,
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "model": session.model_name
            }
            
        except ImportError as e:
            raise ResourceError(
                "HuggingFace transformers library not installed. "
                "Install with: pip install transformers torch",
                resource_name=self.name,
                operation="complete"
            ) from e
    
    def _openai_complete(
        self,
        session: LLMSession,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """OpenAI completion using provider system."""
        from dataknobs_llm.llm.base import LLMConfig, LLMMessage
        from dataknobs_llm.llm.providers import create_llm_provider as create_provider
        
        # Create config from session
        config = LLMConfig(
            provider="openai",
            model=session.model_name,
            api_key=kwargs.get('api_key', os.getenv('OPENAI_API_KEY')),
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens', 1000)
        )
        
        try:
            # Create provider and execute
            provider = create_provider(config, is_async=False)
            provider.initialize()
            
            # Convert prompt to message format
            if isinstance(prompt, str):
                messages = [LLMMessage(role="user", content=prompt)]
            else:
                messages = prompt  # type: ignore[unreachable]
                
            response = provider.complete(messages, **kwargs)
            provider.close()
            
            # Convert to expected format
            return {
                "choices": [{
                    "text": response.content,
                    "index": 0,
                    "finish_reason": response.finish_reason or "stop"
                }],
                "model": response.model,
                "usage": response.usage
            }
        except Exception as e:
            # Fallback to placeholder on error
            return {
                "choices": [{
                    "text": f"Error: {e!s}",
                    "index": 0,
                    "finish_reason": "error"
                }],
                "model": session.model_name
            }
    
    def _anthropic_complete(
        self,
        session: LLMSession,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Anthropic completion using provider system."""
        from dataknobs_llm.llm.base import LLMConfig, LLMMessage
        from dataknobs_llm.llm.providers import create_llm_provider as create_provider
        
        # Create config from session
        config = LLMConfig(
            provider="anthropic",
            model=session.model_name,
            api_key=kwargs.get('api_key', os.getenv('ANTHROPIC_API_KEY')),
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens', 1000)
        )
        
        try:
            # Create provider and execute
            provider = create_provider(config, is_async=False)
            provider.initialize()
            
            # Convert prompt to message format
            if isinstance(prompt, str):
                messages = [LLMMessage(role="user", content=prompt)]
            else:
                messages = prompt  # type: ignore[unreachable]
                
            response = provider.complete(messages, **kwargs)
            provider.close()
            
            # Convert to expected format
            return {
                "choices": [{
                    "text": response.content,
                    "index": 0,
                    "finish_reason": response.finish_reason or "stop"
                }],
                "model": response.model,
                "usage": response.usage
            }
        except Exception as e:
            # Fallback to placeholder on error
            return {
                "choices": [{
                    "text": f"Error: {e!s}",
                    "index": 0,
                    "finish_reason": "error"
                }],
                "model": session.model_name
            }
    
    def _custom_complete(
        self,
        session: LLMSession,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Custom provider completion.
        
        For custom/unknown providers.
        """
        raise NotImplementedError(
            f"Custom provider {session.provider.value} not implemented"
        )
    
    def embed(
        self,
        text: Union[str, List[str]],
        session: LLMSession | None = None,
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings for text.
        
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
            if isinstance(text, str):
                texts = [text]
            else:
                texts = text
            
            # Route to appropriate provider
            if session.provider == LLMProvider.OLLAMA:
                embeddings = self._ollama_embed(session, texts, **kwargs)
            elif session.provider == LLMProvider.HUGGINGFACE:
                embeddings = self._huggingface_embed(session, texts, **kwargs)
            elif session.provider == LLMProvider.OPENAI:
                embeddings = self._openai_embed(session, texts, **kwargs)
            else:
                # Fallback to fake embeddings
                embeddings = [[0.1] * 768 for _ in texts]
            
            return embeddings
            
        finally:
            if should_release:
                self.release(session)
    
    def _ollama_embed(
        self,
        session: LLMSession,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings using Ollama.
        
        Args:
            session: LLM session.
            texts: Texts to embed.
            **kwargs: Additional parameters.
            
        Returns:
            List of embeddings.
        """
        import urllib.request
        
        embeddings = []
        for text in texts:
            data = {
                "model": kwargs.get("embed_model", "nomic-embed-text"),
                "prompt": text
            }
            
            req = urllib.request.Request(
                f"{session.endpoint}/api/embeddings",
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read())
                embeddings.append(result.get("embedding", []))
        
        return embeddings
    
    def _huggingface_embed(
        self,
        session: LLMSession,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings using HuggingFace.
        
        Args:
            session: LLM session.
            texts: Texts to embed.
            **kwargs: Additional parameters.
            
        Returns:
            List of embeddings.
        """
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            model_name = kwargs.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            embeddings = []
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use mean pooling
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                embeddings.append(embedding)
            
            return embeddings
            
        except ImportError as e:
            raise ResourceError(
                "HuggingFace transformers library not installed",
                resource_name=self.name,
                operation="embed"
            ) from e
    
    def _openai_embed(
        self,
        session: LLMSession,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI provider system."""
        from dataknobs_fsm.llm.base import LLMConfig
        from dataknobs_llm.llm.providers import create_llm_provider as create_provider
        
        # Create config for embeddings
        config = LLMConfig(
            provider="openai",
            model=kwargs.get('embed_model', 'text-embedding-ada-002'),
            api_key=kwargs.get('api_key', os.getenv('OPENAI_API_KEY'))
        )
        
        try:
            # Create provider and generate embeddings
            provider = create_provider(config, is_async=False)
            provider.initialize()
            
            embeddings = provider.embed(texts, **kwargs)
            provider.close()
            
            # Ensure we return List[List[float]]
            if isinstance(embeddings[0], list):
                return embeddings
            else:
                return [embeddings]  # Single text case
                
        except Exception:
            # Fallback to placeholder dimensions on error
            return [[0.1] * 1536 for _ in texts]  # OpenAI ada-002 dimension
    
    def get_usage_stats(self, session: LLMSession) -> Dict[str, Any]:
        """Get usage statistics for a session.
        
        Args:
            session: LLM session.
            
        Returns:
            Usage statistics.
        """
        stats = {
            "provider": session.provider.value,
            "model": session.model_name,
            "total_requests": session.total_requests,
        }
        
        # Add token stats for providers that track them
        if session.provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, 
                                LLMProvider.OLLAMA]:
            stats.update({
                "total_prompt_tokens": session.total_prompt_tokens,
                "total_completion_tokens": session.total_completion_tokens,
                "total_tokens": session.total_prompt_tokens + session.total_completion_tokens,
            })
        
        # Add rate limit info for commercial providers
        if session.provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC]:
            stats["rate_limits"] = {
                "requests_per_minute": session.requests_per_minute,
                "tokens_per_minute": session.tokens_per_minute,
                "current_window": {
                    "requests": session.request_count,
                    "tokens": session.token_count,
                    "window_start": session.window_start
                }
            }
        
        return stats
