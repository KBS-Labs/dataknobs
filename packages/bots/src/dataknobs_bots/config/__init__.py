"""Configuration utilities for DynaBot.

This module provides utilities for resource resolution and configuration
binding with environment-aware support.

Example:
    ```python
    from dataknobs_config import EnvironmentConfig
    from dataknobs_bots.config import create_bot_resolver, BotResourceResolver

    # Low-level: Create resolver and resolve manually
    env = EnvironmentConfig.load("production")
    resolver = create_bot_resolver(env)
    llm = resolver.resolve("llm_providers", "default")

    # High-level: Use BotResourceResolver for initialized resources
    bot_resolver = BotResourceResolver(env)
    llm = await bot_resolver.get_llm("default")
    db = await bot_resolver.get_database("conversations")
    ```
"""

from .resolution import (
    BotResourceResolver,
    create_bot_resolver,
    register_database_factory,
    register_embedding_factory,
    register_llm_factory,
    register_vector_store_factory,
)

__all__ = [
    "create_bot_resolver",
    "BotResourceResolver",
    "register_llm_factory",
    "register_database_factory",
    "register_vector_store_factory",
    "register_embedding_factory",
]
