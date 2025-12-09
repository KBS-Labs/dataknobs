"""Core DynaBot implementation."""

from types import TracebackType
from typing import Any

from dataknobs_llm.conversations import ConversationManager, DataknobsConversationStorage
from dataknobs_llm.llm import AsyncLLMProvider
from dataknobs_llm.prompts import AsyncPromptBuilder
from dataknobs_llm.tools import ToolRegistry

from .context import BotContext
from ..memory.base import Memory


class DynaBot:
    """Configuration-driven chatbot leveraging the DataKnobs ecosystem.

    DynaBot provides a flexible, configuration-driven bot that can be customized
    for different use cases through YAML/JSON configuration files.

    Attributes:
        llm: LLM provider for generating responses
        prompt_builder: Prompt builder for managing prompts
        conversation_storage: Storage backend for conversations
        tool_registry: Registry of available tools
        memory: Optional memory implementation for context
        knowledge_base: Optional knowledge base for RAG
        reasoning_strategy: Optional reasoning strategy
        middleware: List of middleware for request/response processing
        system_prompt_name: Name of the system prompt template to use
        system_prompt_content: Inline system prompt content (alternative to name)
        system_prompt_rag_configs: RAG configurations for inline system prompts
        default_temperature: Default temperature for LLM generation
        default_max_tokens: Default max tokens for LLM generation
    """

    def __init__(
        self,
        llm: AsyncLLMProvider,
        prompt_builder: AsyncPromptBuilder,
        conversation_storage: DataknobsConversationStorage,
        tool_registry: ToolRegistry | None = None,
        memory: Memory | None = None,
        knowledge_base: Any | None = None,
        reasoning_strategy: Any | None = None,
        middleware: list[Any] | None = None,
        system_prompt_name: str | None = None,
        system_prompt_content: str | None = None,
        system_prompt_rag_configs: list[dict[str, Any]] | None = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 1000,
    ):
        """Initialize DynaBot.

        Args:
            llm: LLM provider instance
            prompt_builder: Prompt builder instance
            conversation_storage: Conversation storage backend
            tool_registry: Optional tool registry
            memory: Optional memory implementation
            knowledge_base: Optional knowledge base
            reasoning_strategy: Optional reasoning strategy
            middleware: Optional middleware list
            system_prompt_name: Name of system prompt template (mutually exclusive with content)
            system_prompt_content: Inline system prompt content (mutually exclusive with name)
            system_prompt_rag_configs: RAG configurations for inline system prompts
            default_temperature: Default temperature (0-1)
            default_max_tokens: Default max tokens to generate
        """
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.conversation_storage = conversation_storage
        self.tool_registry = tool_registry or ToolRegistry()
        self.memory = memory
        self.knowledge_base = knowledge_base
        self.reasoning_strategy = reasoning_strategy
        self.middleware = middleware or []
        self.system_prompt_name = system_prompt_name
        self.system_prompt_content = system_prompt_content
        self.system_prompt_rag_configs = system_prompt_rag_configs
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self._conversation_managers: dict[str, ConversationManager] = {}

    @classmethod
    async def from_config(cls, config: dict[str, Any]) -> "DynaBot":
        """Create DynaBot from configuration.

        Args:
            config: Configuration dictionary containing:
                - llm: LLM configuration (provider, model, etc.)
                - conversation_storage: Storage configuration
                - tools: Optional list of tool configurations
                - memory: Optional memory configuration
                - knowledge_base: Optional knowledge base configuration
                - reasoning: Optional reasoning strategy configuration
                - middleware: Optional middleware configurations
                - prompts: Optional prompts library (dict of name -> content)
                - system_prompt: Optional system prompt configuration (see below)

        Returns:
            Configured DynaBot instance

        System Prompt Formats:
            The system_prompt can be specified in multiple ways:

            - String: Smart detection - if the string exists as a template name
              in the prompt library, it's used as a template reference; otherwise
              it's treated as inline content.

            - Dict with name: `{"name": "template_name"}` - explicit template reference
            - Dict with name + strict: `{"name": "template_name", "strict": true}` -
              raises error if template doesn't exist
            - Dict with content: `{"content": "inline prompt text"}` - inline content
            - Dict with content + rag_configs: inline content with RAG enhancement

        Example:
            ```python
            # Smart detection: uses as template if it exists in prompts library
            config = {
                "llm": {"provider": "openai", "model": "gpt-4"},
                "conversation_storage": {"backend": "memory"},
                "prompts": {
                    "helpful_assistant": "You are a helpful AI assistant."
                },
                "system_prompt": "helpful_assistant"  # Found in prompts, used as template
            }

            # Smart detection: treated as inline content (not in prompts library)
            config = {
                "llm": {"provider": "openai", "model": "gpt-4"},
                "conversation_storage": {"backend": "memory"},
                "system_prompt": "You are a helpful assistant."  # Not a template name
            }

            # Explicit inline content with RAG enhancement
            config = {
                "llm": {"provider": "openai", "model": "gpt-4"},
                "conversation_storage": {"backend": "memory"},
                "system_prompt": {
                    "content": "You are a helpful assistant. Use this context: {{ CONTEXT }}",
                    "rag_configs": [{
                        "adapter_name": "docs",
                        "query": "assistant guidelines",
                        "placeholder": "CONTEXT",
                        "k": 3
                    }]
                }
            }

            # Strict mode: error if template doesn't exist
            config = {
                "llm": {"provider": "openai", "model": "gpt-4"},
                "conversation_storage": {"backend": "memory"},
                "system_prompt": {
                    "name": "my_template",
                    "strict": true  # Raises ValueError if my_template doesn't exist
                }
            }

            bot = await DynaBot.from_config(config)
            ```
        """
        from dataknobs_data.factory import AsyncDatabaseFactory
        from dataknobs_llm.llm import LLMProviderFactory
        from dataknobs_llm.prompts import AsyncPromptBuilder
        from dataknobs_llm.prompts.implementations import CompositePromptLibrary
        from ..memory import create_memory_from_config

        # Create LLM provider
        llm_config = config["llm"]
        factory = LLMProviderFactory(is_async=True)
        llm = factory.create(llm_config)
        await llm.initialize()

        # Create conversation storage
        storage_config = config["conversation_storage"].copy()

        # Create database backend using factory
        db_factory = AsyncDatabaseFactory()
        backend = db_factory.create(**storage_config)
        await backend.connect()
        conversation_storage = DataknobsConversationStorage(backend)

        # Create prompt builder
        # Support optional prompts configuration
        prompt_libraries = []
        if "prompts" in config:
            from dataknobs_llm.prompts.implementations import ConfigPromptLibrary

            prompts_config = config["prompts"]

            # If prompts are provided as a dict, create a config-based library
            if isinstance(prompts_config, dict):
                # Convert simple string prompts to proper template structure
                structured_config = {"system": {}, "user": {}}

                for prompt_name, prompt_content in prompts_config.items():
                    if isinstance(prompt_content, dict):
                        # Already structured - use as-is
                        # Assume it's a system prompt unless specified
                        prompt_type = prompt_content.get("type", "system")
                        if prompt_type in structured_config:
                            structured_config[prompt_type][prompt_name] = prompt_content
                    else:
                        # Simple string - treat as system prompt template
                        structured_config["system"][prompt_name] = {
                            "template": prompt_content
                        }

                library = ConfigPromptLibrary(structured_config)
                prompt_libraries.append(library)

        # Create composite library (empty if no prompts configured)
        library = CompositePromptLibrary(libraries=prompt_libraries)
        prompt_builder = AsyncPromptBuilder(library)

        # Create tools
        tool_registry = ToolRegistry()
        if "tools" in config:
            for tool_config in config["tools"]:
                tool = cls._resolve_tool(tool_config, config)
                if tool:
                    tool_registry.register_tool(tool)

        # Create memory
        memory = None
        if "memory" in config:
            memory = await create_memory_from_config(config["memory"])

        # Create knowledge base
        knowledge_base = None
        kb_config = config.get("knowledge_base", {})
        if kb_config.get("enabled"):
            from ..knowledge import create_knowledge_base_from_config
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Initializing knowledge base with config: {kb_config.get('type', 'unknown')}")
            knowledge_base = await create_knowledge_base_from_config(kb_config)
            logger.info("Knowledge base initialized successfully")

        # Create reasoning strategy
        reasoning_strategy = None
        if "reasoning" in config:
            from ..reasoning import create_reasoning_from_config

            reasoning_strategy = create_reasoning_from_config(config["reasoning"])

        # Create middleware
        middleware = []
        if "middleware" in config:
            for mw_config in config["middleware"]:
                mw = cls._create_middleware(mw_config)
                if mw:
                    middleware.append(mw)

        # Extract system prompt (supports template name or inline content)
        system_prompt_name = None
        system_prompt_content = None
        system_prompt_rag_configs = None
        if "system_prompt" in config:
            system_prompt_config = config["system_prompt"]
            if isinstance(system_prompt_config, dict):
                # Explicit dict format: {name: "template"} or {content: "inline..."}
                system_prompt_name = system_prompt_config.get("name")
                system_prompt_content = system_prompt_config.get("content")
                system_prompt_rag_configs = system_prompt_config.get("rag_configs")

                # If strict mode is enabled, require the template to exist
                if system_prompt_name and system_prompt_config.get("strict"):
                    if library.get_system_prompt(system_prompt_name) is None:
                        raise ValueError(
                            f"System prompt template not found: {system_prompt_name} "
                            "(strict mode enabled)"
                        )
            elif isinstance(system_prompt_config, str):
                # String format: smart detection
                # If it exists in the library, use as template name; otherwise treat as inline
                if library.get_system_prompt(system_prompt_config) is not None:
                    system_prompt_name = system_prompt_config
                else:
                    system_prompt_content = system_prompt_config

        return cls(
            llm=llm,
            prompt_builder=prompt_builder,
            conversation_storage=conversation_storage,
            tool_registry=tool_registry,
            memory=memory,
            knowledge_base=knowledge_base,
            reasoning_strategy=reasoning_strategy,
            middleware=middleware,
            system_prompt_name=system_prompt_name,
            system_prompt_content=system_prompt_content,
            system_prompt_rag_configs=system_prompt_rag_configs,
            default_temperature=llm_config.get("temperature", 0.7),
            default_max_tokens=llm_config.get("max_tokens", 1000),
        )

    async def chat(
        self,
        message: str,
        context: BotContext,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        rag_query: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Process a chat message.

        Args:
            message: User message to process
            context: Bot execution context
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            stream: Whether to stream the response
            rag_query: Optional explicit query for knowledge base retrieval.
                      If provided, this is used instead of the message for RAG.
                      Useful when the message contains literal text to analyze
                      (e.g., "Analyze this prompt: [prompt text]") but you want
                      to search for analysis techniques instead.
            **kwargs: Additional arguments

        Returns:
            Bot response as string

        Example:
            ```python
            context = BotContext(
                conversation_id="conv-123",
                client_id="client-456",
                user_id="user-789"
            )
            response = await bot.chat("Hello!", context)

            # With explicit RAG query
            response = await bot.chat(
                "Analyze this: Write a poem about cats",
                context,
                rag_query="prompt analysis techniques evaluation"
            )
            ```
        """
        # Apply middleware (before)
        for mw in self.middleware:
            if hasattr(mw, "before_message"):
                await mw.before_message(message, context)

        # Build message with context from memory and knowledge
        full_message = await self._build_message_with_context(message, rag_query=rag_query)

        # Get or create conversation manager
        manager = await self._get_or_create_conversation(context)

        # Add user message
        await manager.add_message(content=full_message, role="user")

        # Update memory
        if self.memory:
            await self.memory.add_message(message, role="user")

        # Generate response
        if self.reasoning_strategy:
            response = await self.reasoning_strategy.generate(
                manager=manager,
                llm=self.llm,
                tools=list(self.tool_registry),
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
            )
        else:
            response = await manager.complete(
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
            )

        # Extract response content
        response_content = response.content if hasattr(response, "content") else str(response)

        # Update memory
        if self.memory:
            await self.memory.add_message(response_content, role="assistant")

        # Apply middleware (after)
        for mw in self.middleware:
            if hasattr(mw, "after_message"):
                await mw.after_message(response, context)

        return response_content

    async def get_conversation(self, conversation_id: str) -> Any:
        """Retrieve conversation history.

        This method fetches the complete conversation state including all messages,
        metadata, and the message tree structure. Useful for displaying conversation
        history, debugging, analytics, or exporting conversations.

        Args:
            conversation_id: Unique identifier of the conversation to retrieve

        Returns:
            ConversationState object containing the full conversation history,
            or None if the conversation does not exist

        Example:
            ```python
            # Retrieve a conversation
            conv_state = await bot.get_conversation("conv-123")

            # Access messages
            messages = conv_state.message_tree

            # Access metadata
            print(conv_state.metadata)
            ```

        See Also:
            - clear_conversation(): Clear/delete a conversation
            - chat(): Add messages to a conversation
        """
        return await self.conversation_storage.load_conversation(conversation_id)

    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation's history.

        This method removes the conversation from both persistent storage and the
        internal cache. The next chat() call with this conversation_id will start
        a fresh conversation. Useful for:

        - Implementing "start over" functionality
        - Privacy/data deletion requirements
        - Testing and cleanup
        - Resetting conversation context

        Args:
            conversation_id: Unique identifier of the conversation to clear

        Returns:
            True if the conversation was deleted, False if it didn't exist

        Example:
            ```python
            # Clear a conversation
            deleted = await bot.clear_conversation("conv-123")

            if deleted:
                print("Conversation deleted")
            else:
                print("Conversation not found")

            # Next chat will start fresh
            response = await bot.chat("Hello!", context)
            ```

        Note:
            This operation is permanent and cannot be undone. The conversation
            cannot be recovered after deletion.

        See Also:
            - get_conversation(): Retrieve conversation before clearing
            - chat(): Will create new conversation after clearing
        """
        # Remove from cache if present
        if conversation_id in self._conversation_managers:
            del self._conversation_managers[conversation_id]

        # Delete from storage
        return await self.conversation_storage.delete_conversation(conversation_id)

    async def close(self) -> None:
        """Close the bot and clean up resources.

        This method closes the LLM provider, conversation storage backend,
        and releases associated resources like HTTP connections and database
        connections. Should be called when the bot is no longer needed,
        especially in testing or when creating temporary bot instances.

        Example:
            ```python
            bot = await DynaBot.from_config(config)
            try:
                response = await bot.chat("Hello", context)
            finally:
                await bot.close()
            ```

        Note:
            After calling close(), the bot should not be used for further operations.
            Create a new bot instance if needed.
        """
        # Close LLM provider
        if self.llm and hasattr(self.llm, 'close'):
            await self.llm.close()

        # Close conversation storage backend
        if self.conversation_storage and hasattr(self.conversation_storage, 'backend'):
            backend = self.conversation_storage.backend
            if backend and hasattr(backend, 'close'):
                await backend.close()

        # Close knowledge base (releases embedding provider HTTP sessions)
        if self.knowledge_base and hasattr(self.knowledge_base, 'close'):
            await self.knowledge_base.close()

        # Close memory store
        if self.memory and hasattr(self.memory, 'close'):
            await self.memory.close()

    async def __aenter__(self) -> "DynaBot":
        """Async context manager entry.

        Returns:
            Self for use in async with statement
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit - ensures cleanup.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        await self.close()

    async def _get_or_create_conversation(
        self, context: BotContext
    ) -> ConversationManager:
        """Get or create conversation manager for context.

        Args:
            context: Bot execution context

        Returns:
            ConversationManager instance
        """
        conv_id = context.conversation_id

        # Check cache
        if conv_id in self._conversation_managers:
            return self._conversation_managers[conv_id]

        # Try to resume existing conversation
        try:
            manager = await ConversationManager.resume(
                conversation_id=conv_id,
                llm=self.llm,
                prompt_builder=self.prompt_builder,
                storage=self.conversation_storage,
            )
        except Exception:
            # Create new conversation with specified conversation_id
            from dataknobs_llm.conversations import ConversationNode, ConversationState
            from dataknobs_llm.llm.base import LLMMessage
            from dataknobs_structures.tree import Tree

            metadata = {
                "client_id": context.client_id,
                "user_id": context.user_id,
                **context.session_metadata,
            }

            # Create initial state with specified conversation_id
            # Start with empty root node (will be replaced by system prompt if provided)
            root_message = LLMMessage(role="system", content="")
            root_node = ConversationNode(
                message=root_message,
                node_id="",
            )
            tree = Tree(root_node)
            state = ConversationState(
                conversation_id=conv_id,  # Use the conversation_id from context
                message_tree=tree,
                current_node_id="",
                metadata=metadata,
            )

            # Create manager with pre-initialized state
            manager = ConversationManager(
                llm=self.llm,
                prompt_builder=self.prompt_builder,
                storage=self.conversation_storage,
                state=state,
                metadata=metadata,
            )

            # Add system prompt if specified (either as template name or inline content)
            if self.system_prompt_name:
                # Use template name - will be rendered by prompt builder
                await manager.add_message(
                    prompt_name=self.system_prompt_name,
                    role="system",
                )
            elif self.system_prompt_content:
                # Use inline content - pass RAG configs if available
                await manager.add_message(
                    content=self.system_prompt_content,
                    role="system",
                    rag_configs=self.system_prompt_rag_configs,
                    include_rag=bool(self.system_prompt_rag_configs),
                )

        # Cache manager
        self._conversation_managers[conv_id] = manager
        return manager

    async def _build_message_with_context(
        self,
        message: str,
        rag_query: str | None = None,
    ) -> str:
        """Build message with knowledge and memory context.

        Args:
            message: Original user message
            rag_query: Optional explicit query for knowledge base retrieval.
                      If provided, this is used instead of the message for RAG.

        Returns:
            Message augmented with context
        """
        contexts = []

        # Add knowledge context
        if self.knowledge_base:
            # Use explicit rag_query if provided, otherwise use message
            search_query = rag_query if rag_query else message
            kb_results = await self.knowledge_base.query(search_query, k=5)
            if kb_results:
                # Use format_context if available (new RAG utilities)
                if hasattr(self.knowledge_base, "format_context"):
                    kb_context = self.knowledge_base.format_context(
                        kb_results, wrap_in_tags=True
                    )
                    contexts.append(kb_context)
                else:
                    # Fallback to legacy formatting
                    formatted_chunks = []
                    for i, r in enumerate(kb_results, 1):
                        text = r["text"]
                        source = r.get("source", "")
                        heading = r.get("heading_path", "")

                        chunk_text = f"[{i}] {heading}\n{text}"
                        if source:
                            chunk_text += f"\n(Source: {source})"
                        formatted_chunks.append(chunk_text)

                    kb_context = "\n\n---\n\n".join(formatted_chunks)
                    contexts.append(f"<knowledge_base>\n{kb_context}\n</knowledge_base>")

        # Add memory context
        if self.memory:
            mem_results = await self.memory.get_context(message)
            if mem_results:
                mem_context = "\n\n".join([r["content"] for r in mem_results])
                contexts.append(f"<conversation_history>\n{mem_context}\n</conversation_history>")

        # Build full message with clear separation
        if contexts:
            context_section = "\n\n".join(contexts)
            return f"{context_section}\n\n<question>\n{message}\n</question>"
        return message

    @staticmethod
    def _resolve_tool(tool_config: dict[str, Any] | str, config: dict[str, Any]) -> Any | None:
        """Resolve tool from configuration.

        Supports two patterns:
        1. Direct class instantiation: {"class": "module.ToolClass", "params": {...}}
        2. XRef resolution: "xref:tools[tool_name]" or {"xref": "tools[tool_name]"}

        Args:
            tool_config: Tool configuration (dict or string xref)
            config: Full bot configuration for xref resolution

        Returns:
            Tool instance or None if resolution fails

        Example:
            # Direct instantiation
            tool_config = {
                "class": "my_tools.CalculatorTool",
                "params": {"precision": 2}
            }

            # XRef to pre-defined tool
            tool_config = "xref:tools[calculator]"
            # Requires config to have:
            # {
            #     "tool_definitions": {
            #         "calculator": {
            #             "class": "my_tools.CalculatorTool",
            #             "params": {}
            #         }
            #     }
            # }
        """
        import importlib
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Handle xref string format
            if isinstance(tool_config, str):
                if tool_config.startswith("xref:"):
                    # Parse xref (e.g., "xref:tools[calculator]")
                    # Extract the reference name
                    import re

                    match = re.match(r"xref:tools\[([^\]]+)\]", tool_config)
                    if not match:
                        logger.error(f"Invalid xref format: {tool_config}")
                        return None

                    tool_name = match.group(1)

                    # Look up in tool_definitions
                    tool_definitions = config.get("tool_definitions", {})
                    if tool_name not in tool_definitions:
                        logger.error(
                            f"Tool definition not found: {tool_name}. "
                            f"Available: {list(tool_definitions.keys())}"
                        )
                        return None

                    # Recursively resolve the referenced config
                    return DynaBot._resolve_tool(tool_definitions[tool_name], config)
                else:
                    logger.error(f"String tool config must be xref format: {tool_config}")
                    return None

            # Handle dict with xref key
            if isinstance(tool_config, dict) and "xref" in tool_config:
                return DynaBot._resolve_tool(tool_config["xref"], config)

            # Handle dict with class key (direct instantiation)
            if isinstance(tool_config, dict) and "class" in tool_config:
                class_path = tool_config["class"]
                params = tool_config.get("params", {})

                # Import the tool class
                module_path, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                tool_class = getattr(module, class_name)

                # Instantiate the tool
                tool = tool_class(**params)

                # Validate it's a Tool instance
                from dataknobs_llm.tools import Tool

                if not isinstance(tool, Tool):
                    logger.error(
                        f"Resolved class {class_path} is not a Tool instance: {type(tool)}"
                    )
                    return None

                logger.info(f"Successfully loaded tool: {tool.name} ({class_path})")
                return tool
            else:
                logger.error(
                    f"Invalid tool config format. Expected dict with 'class' or 'xref' key, "
                    f"or xref string. Got: {type(tool_config)}"
                )
                return None

        except ImportError as e:
            logger.error(f"Failed to import tool class: {e}")
            return None
        except AttributeError as e:
            logger.error(f"Failed to find tool class: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to instantiate tool: {e}")
            return None

    @staticmethod
    def _create_middleware(config: dict[str, Any]) -> Any | None:
        """Create middleware from configuration.

        Args:
            config: Middleware configuration

        Returns:
            Middleware instance or None
        """
        try:
            import importlib

            module_path, class_name = config["class"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            middleware_class = getattr(module, class_name)
            return middleware_class(**config.get("params", {}))
        except Exception:
            return None
