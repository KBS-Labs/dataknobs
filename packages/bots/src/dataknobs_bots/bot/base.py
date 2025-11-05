"""Core DynaBot implementation."""

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
        system_prompt_name: Name of the system prompt to use
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
            system_prompt_name: Name of system prompt
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
                - system_prompt: Optional system prompt configuration

        Returns:
            Configured DynaBot instance

        Example:
            ```python
            config = {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.7
                },
                "conversation_storage": {
                    "backend": "memory"
                },
                "memory": {
                    "type": "buffer",
                    "max_messages": 10
                },
                "prompts": {
                    "helpful_assistant": "You are a helpful AI assistant."
                },
                "system_prompt": {
                    "name": "helpful_assistant"
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
                    tool_registry.register(tool)

        # Create memory
        memory = None
        if "memory" in config:
            memory = await create_memory_from_config(config["memory"])

        # Create knowledge base
        knowledge_base = None
        if config.get("knowledge_base", {}).get("enabled"):
            from ..knowledge import create_knowledge_base_from_config

            knowledge_base = await create_knowledge_base_from_config(config["knowledge_base"])

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

        # Extract system prompt
        system_prompt_name = None
        if "system_prompt" in config:
            system_prompt_config = config["system_prompt"]
            if isinstance(system_prompt_config, dict):
                system_prompt_name = system_prompt_config.get("name")
            else:
                system_prompt_name = system_prompt_config

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
        **kwargs: Any,
    ) -> str:
        """Process a chat message.

        Args:
            message: User message to process
            context: Bot execution context
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            stream: Whether to stream the response
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
            ```
        """
        # Apply middleware (before)
        for mw in self.middleware:
            if hasattr(mw, "before_message"):
                await mw.before_message(message, context)

        # Build message with context from memory and knowledge
        full_message = await self._build_message_with_context(message)

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
            # Create new conversation
            metadata = {
                "client_id": context.client_id,
                "user_id": context.user_id,
                **context.session_metadata,
            }

            manager = await ConversationManager.create(
                llm=self.llm,
                prompt_builder=self.prompt_builder,
                storage=self.conversation_storage,
                system_prompt_name=self.system_prompt_name,
                metadata=metadata,
            )

        # Cache manager
        self._conversation_managers[conv_id] = manager
        return manager

    async def _build_message_with_context(self, message: str) -> str:
        """Build message with knowledge and memory context.

        Args:
            message: Original user message

        Returns:
            Message augmented with context
        """
        contexts = []

        # Add knowledge context
        if self.knowledge_base:
            kb_results = await self.knowledge_base.query(message, k=3)
            if kb_results:
                kb_context = "\n\n".join([r["text"] for r in kb_results])
                contexts.append(f"Knowledge context:\n{kb_context}")

        # Add memory context
        if self.memory:
            mem_results = await self.memory.get_context(message)
            if mem_results:
                mem_context = "\n\n".join([r["content"] for r in mem_results])
                contexts.append(f"Relevant history:\n{mem_context}")

        # Build full message
        if contexts:
            return f"{chr(10).join(contexts)}\n\nUser: {message}"
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
