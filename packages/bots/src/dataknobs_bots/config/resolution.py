"""Resource resolution utilities for DynaBot configuration.

This module provides utilities to create a ConfigBindingResolver with
DynaBot-specific factories registered, enabling direct resource instantiation
from logical names.

Example:
    ```python
    from dataknobs_config import EnvironmentConfig
    from dataknobs_bots.config import create_bot_resolver

    # Load environment
    env = EnvironmentConfig.load("production")

    # Create resolver with all DynaBot factories registered
    resolver = create_bot_resolver(env)

    # Resolve resources by logical name
    llm = resolver.resolve("llm_providers", "default")
    db = await resolver.resolve_async("databases", "conversations")
    vector_store = resolver.resolve("vector_stores", "knowledge")
    embedding = resolver.resolve("embedding_providers", "default")
    ```
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataknobs_config import ConfigBindingResolver, EnvironmentConfig

logger = logging.getLogger(__name__)


def create_bot_resolver(
    environment: EnvironmentConfig,
    resolve_env_vars: bool = True,
    register_defaults: bool = True,
) -> ConfigBindingResolver:
    """Create a ConfigBindingResolver with DynaBot-specific factories.

    This resolver can instantiate resources directly from logical names
    defined in environment configuration. It registers factories for:

    - **llm_providers**: LLM providers (OpenAI, Anthropic, Ollama, etc.)
    - **databases**: Database backends (memory, sqlite, postgres, etc.)
    - **vector_stores**: Vector store backends (FAISS, Chroma, memory, etc.)
    - **embedding_providers**: Embedding providers (uses LLM providers with embed())

    Args:
        environment: Environment configuration for resource lookup
        resolve_env_vars: Whether to resolve environment variables in configs
        register_defaults: If True, register all default DynaBot factories.
            Set to False to manually register only needed factories.

    Returns:
        ConfigBindingResolver with registered factories

    Example:
        ```python
        from dataknobs_config import EnvironmentConfig
        from dataknobs_bots.config import create_bot_resolver

        # Auto-detect environment from DATAKNOBS_ENVIRONMENT
        env = EnvironmentConfig.load()
        resolver = create_bot_resolver(env)

        # Resolve an LLM provider
        llm = resolver.resolve("llm_providers", "default")
        await llm.initialize()

        # Resolve a database asynchronously
        db = await resolver.resolve_async("databases", "conversations")

        # Resolve a vector store
        vs = resolver.resolve("vector_stores", "knowledge")
        await vs.initialize()
        ```

    Note:
        The resolver caches created instances by default. Use
        `resolver.resolve(..., use_cache=False)` to create fresh instances.
    """
    from dataknobs_config import ConfigBindingResolver

    resolver = ConfigBindingResolver(environment, resolve_env_vars=resolve_env_vars)

    if register_defaults:
        register_llm_factory(resolver)
        register_database_factory(resolver)
        register_vector_store_factory(resolver)
        register_embedding_factory(resolver)
        logger.debug("Registered all DynaBot resource factories")

    return resolver


def register_llm_factory(resolver: ConfigBindingResolver) -> None:
    """Register LLM provider factory with the resolver.

    Args:
        resolver: ConfigBindingResolver to register with

    Example:
        ```python
        resolver = ConfigBindingResolver(env)
        register_llm_factory(resolver)

        # Now can resolve LLM providers
        llm = resolver.resolve("llm_providers", "default")
        ```
    """
    from dataknobs_llm.llm import LLMProviderFactory

    factory = LLMProviderFactory(is_async=True)
    resolver.register_factory("llm_providers", factory)
    logger.debug("Registered LLM provider factory")


def register_database_factory(resolver: ConfigBindingResolver) -> None:
    """Register async database factory with the resolver.

    Args:
        resolver: ConfigBindingResolver to register with

    Example:
        ```python
        resolver = ConfigBindingResolver(env)
        register_database_factory(resolver)

        # Now can resolve databases
        db = await resolver.resolve_async("databases", "conversations")
        ```
    """
    from dataknobs_data.factory import AsyncDatabaseFactory

    factory = AsyncDatabaseFactory()
    resolver.register_factory("databases", factory)
    logger.debug("Registered database factory")


def register_vector_store_factory(resolver: ConfigBindingResolver) -> None:
    """Register vector store factory with the resolver.

    Args:
        resolver: ConfigBindingResolver to register with

    Example:
        ```python
        resolver = ConfigBindingResolver(env)
        register_vector_store_factory(resolver)

        # Now can resolve vector stores
        vs = resolver.resolve("vector_stores", "knowledge")
        await vs.initialize()
        ```
    """
    from dataknobs_data.vector.stores import VectorStoreFactory

    factory = VectorStoreFactory()
    resolver.register_factory("vector_stores", factory)
    logger.debug("Registered vector store factory")


def register_embedding_factory(resolver: ConfigBindingResolver) -> None:
    """Register embedding provider factory with the resolver.

    Embedding providers use the LLM provider factory since most LLM
    providers (OpenAI, Ollama, etc.) support embedding via their
    embed() method.

    Args:
        resolver: ConfigBindingResolver to register with

    Example:
        ```python
        resolver = ConfigBindingResolver(env)
        register_embedding_factory(resolver)

        # Now can resolve embedding providers
        embedder = resolver.resolve("embedding_providers", "default")
        await embedder.initialize()
        embedding = await embedder.embed("Hello world")
        ```

    Note:
        The resolved provider should have an `embed()` method. Standard
        LLM providers like OpenAI, Anthropic, and Ollama support this.
    """
    from dataknobs_llm.llm import LLMProviderFactory

    factory = LLMProviderFactory(is_async=True)
    resolver.register_factory("embedding_providers", factory)
    logger.debug("Registered embedding provider factory")


class BotResourceResolver:
    """High-level resource resolver for DynaBot.

    Provides convenient async methods for resolving and initializing
    DynaBot resources. Wraps ConfigBindingResolver with DynaBot-specific
    initialization logic.

    Example:
        ```python
        from dataknobs_config import EnvironmentConfig
        from dataknobs_bots.config import BotResourceResolver

        env = EnvironmentConfig.load("production")
        resolver = BotResourceResolver(env)

        # Get initialized LLM provider
        llm = await resolver.get_llm("default")

        # Get initialized database
        db = await resolver.get_database("conversations")

        # Get initialized vector store
        vs = await resolver.get_vector_store("knowledge")
        ```
    """

    def __init__(
        self,
        environment: EnvironmentConfig,
        resolve_env_vars: bool = True,
    ):
        """Initialize the resource resolver.

        Args:
            environment: Environment configuration
            resolve_env_vars: Whether to resolve env vars in configs
        """
        self._resolver = create_bot_resolver(
            environment,
            resolve_env_vars=resolve_env_vars,
        )
        self._environment = environment

    @property
    def environment(self) -> EnvironmentConfig:
        """Get the environment configuration."""
        return self._environment

    @property
    def resolver(self) -> ConfigBindingResolver:
        """Get the underlying ConfigBindingResolver."""
        return self._resolver

    async def get_llm(
        self,
        name: str = "default",
        use_cache: bool = True,
        **overrides: Any,
    ) -> Any:
        """Get an initialized LLM provider.

        Args:
            name: Logical name of the LLM provider
            use_cache: Whether to return cached instance
            **overrides: Config overrides for this resolution

        Returns:
            Initialized AsyncLLMProvider instance
        """
        llm = self._resolver.resolve(
            "llm_providers", name, use_cache=use_cache, **overrides
        )
        await llm.initialize()
        return llm

    async def get_database(
        self,
        name: str = "default",
        use_cache: bool = True,
        **overrides: Any,
    ) -> Any:
        """Get an initialized database backend.

        Args:
            name: Logical name of the database
            use_cache: Whether to return cached instance
            **overrides: Config overrides for this resolution

        Returns:
            Initialized database backend instance
        """
        db = self._resolver.resolve(
            "databases", name, use_cache=use_cache, **overrides
        )
        if hasattr(db, "connect"):
            await db.connect()
        return db

    async def get_vector_store(
        self,
        name: str = "default",
        use_cache: bool = True,
        **overrides: Any,
    ) -> Any:
        """Get an initialized vector store.

        Args:
            name: Logical name of the vector store
            use_cache: Whether to return cached instance
            **overrides: Config overrides for this resolution

        Returns:
            Initialized VectorStore instance
        """
        vs = self._resolver.resolve(
            "vector_stores", name, use_cache=use_cache, **overrides
        )
        if hasattr(vs, "initialize"):
            await vs.initialize()
        return vs

    async def get_embedding_provider(
        self,
        name: str = "default",
        use_cache: bool = True,
        **overrides: Any,
    ) -> Any:
        """Get an initialized embedding provider.

        Args:
            name: Logical name of the embedding provider
            use_cache: Whether to return cached instance
            **overrides: Config overrides for this resolution

        Returns:
            Initialized provider with embed() method
        """
        provider = self._resolver.resolve(
            "embedding_providers", name, use_cache=use_cache, **overrides
        )
        await provider.initialize()
        return provider

    def clear_cache(self, resource_type: str | None = None) -> None:
        """Clear cached resource instances.

        Args:
            resource_type: Specific type to clear, or None for all
        """
        self._resolver.clear_cache(resource_type)

    def __repr__(self) -> str:
        """String representation."""
        types = self._resolver.get_registered_types()
        return f"BotResourceResolver(environment={self._environment.name!r}, types={types})"
