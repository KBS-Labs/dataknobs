"""Conversation manager for multi-turn interactions with LLMs."""

import uuid
from typing import Optional, List, Dict, Any, AsyncIterator
from datetime import datetime

from dataknobs_structures.tree import Tree
from dataknobs_llm.llm import AsyncLLMProvider, LLMMessage, LLMResponse, LLMStreamResponse
from dataknobs_llm.prompts import AsyncPromptBuilder
from dataknobs_llm.conversations.storage import (
    ConversationNode,
    ConversationState,
    ConversationStorage,
    calculate_node_id,
    get_node_by_id,
    get_messages_for_llm,
)


class ConversationManager:
    """Manages multi-turn conversations with persistence and branching.

    This class orchestrates conversations by:
    - Tracking message history with tree-based branching
    - Managing conversation state
    - Persisting to storage backend
    - Supporting multiple conversation branches

    Example:
        >>> manager = await ConversationManager.create(
        ...     llm=llm,
        ...     prompt_builder=builder,
        ...     storage=storage_backend,
        ...     system_prompt_name="helpful_assistant"
        ... )
        >>>
        >>> # Add user message
        >>> await manager.add_message(
        ...     prompt_name="user_query",
        ...     params={"question": "What is Python?"},
        ...     role="user"
        ... )
        >>>
        >>> # Get LLM response
        >>> result = await manager.complete()
        >>>
        >>> # Continue conversation
        >>> await manager.add_message(
        ...     content="Tell me more about decorators",
        ...     role="user"
        ... )
        >>> result = await manager.complete()
        >>>
        >>> # Create alternative response branch
        >>> await manager.switch_to_node("0")  # Back to first user message
        >>> result2 = await manager.complete(branch_name="alt-response")
        >>>
        >>> # Resume after interruption
        >>> manager2 = await ConversationManager.resume(
        ...     conversation_id=manager.conversation_id,
        ...     llm=llm,
        ...     prompt_builder=builder,
        ...     storage=storage_backend
        ... )
    """

    def __init__(
        self,
        llm: AsyncLLMProvider,
        prompt_builder: AsyncPromptBuilder,
        storage: ConversationStorage,
        state: Optional[ConversationState] = None,
        metadata: Optional[Dict[str, Any]] = None,
        middleware: Optional[List["ConversationMiddleware"]] = None,
    ):
        """Initialize conversation manager.

        Note: Use ConversationManager.create() or ConversationManager.resume()
        instead of calling __init__ directly.

        Args:
            llm: LLM provider for completions
            prompt_builder: Prompt builder with library
            storage: Storage backend for persistence
            state: Optional existing conversation state
            metadata: Optional metadata for new conversations
            middleware: Optional list of middleware to execute
        """
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.storage = storage
        self.state = state
        self._initial_metadata = metadata or {}
        self.middleware = middleware or []

    @classmethod
    async def create(
        cls,
        llm: AsyncLLMProvider,
        prompt_builder: AsyncPromptBuilder,
        storage: ConversationStorage,
        system_prompt_name: Optional[str] = None,
        system_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        middleware: Optional[List["ConversationMiddleware"]] = None,
    ) -> "ConversationManager":
        """Create a new conversation.

        Args:
            llm: LLM provider
            prompt_builder: Prompt builder
            storage: Storage backend
            system_prompt_name: Optional system prompt to initialize with
            system_params: Optional params for system prompt
            metadata: Optional conversation metadata
            middleware: Optional list of middleware to execute

        Returns:
            Initialized ConversationManager

        Example:
            >>> manager = await ConversationManager.create(
            ...     llm=llm,
            ...     prompt_builder=builder,
            ...     storage=storage,
            ...     system_prompt_name="helpful_assistant"
            ... )
        """
        manager = cls(
            llm=llm,
            prompt_builder=prompt_builder,
            storage=storage,
            metadata=metadata,
            middleware=middleware,
        )

        # Initialize with system prompt if provided
        if system_prompt_name:
            await manager.add_message(
                prompt_name=system_prompt_name,
                params=system_params,
                role="system",
            )

        return manager

    @classmethod
    async def resume(
        cls,
        conversation_id: str,
        llm: AsyncLLMProvider,
        prompt_builder: AsyncPromptBuilder,
        storage: ConversationStorage,
        middleware: Optional[List["ConversationMiddleware"]] = None,
    ) -> "ConversationManager":
        """Resume an existing conversation.

        Args:
            conversation_id: Existing conversation ID
            llm: LLM provider
            prompt_builder: Prompt builder
            storage: Storage backend
            middleware: Optional list of middleware to execute

        Returns:
            ConversationManager with restored state

        Raises:
            ValueError: If conversation not found

        Example:
            >>> manager = await ConversationManager.resume(
            ...     conversation_id="conv-123",
            ...     llm=llm,
            ...     prompt_builder=builder,
            ...     storage=storage
            ... )
        """
        # Load state from storage
        state = await storage.load_conversation(conversation_id)
        if not state:
            raise ValueError(f"Conversation '{conversation_id}' not found")

        # Create manager with existing state
        manager = cls(
            llm=llm,
            prompt_builder=prompt_builder,
            storage=storage,
            state=state,
            middleware=middleware,
        )

        return manager

    async def add_message(
        self,
        role: str,
        content: Optional[str] = None,
        prompt_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        include_rag: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationNode:
        """Add a message to the current conversation node.

        Either content or prompt_name must be provided.

        Args:
            role: Message role ("system", "user", or "assistant")
            content: Direct message content (if not using prompt)
            prompt_name: Name of prompt template to render
            params: Parameters for prompt rendering
            include_rag: Whether to execute RAG searches for prompts
            metadata: Optional metadata for this message node

        Returns:
            The created ConversationNode

        Raises:
            ValueError: If neither content nor prompt_name provided

        Example:
            >>> # Add message from prompt
            >>> await manager.add_message(
            ...     role="user",
            ...     prompt_name="code_question",
            ...     params={"code": code_snippet}
            ... )
            >>>
            >>> # Add direct message
            >>> await manager.add_message(
            ...     role="user",
            ...     content="What is Python?"
            ... )
        """
        if not content and not prompt_name:
            raise ValueError("Either content or prompt_name must be provided")

        # Render prompt if needed
        if prompt_name:
            params = params or {}
            if role == "system":
                result = await self.prompt_builder.render_system_prompt(
                    prompt_name,
                    params=params,
                    include_rag=include_rag,
                )
            elif role == "user":
                # For user prompts, calculate the index based on user messages so far
                user_count = 0
                if self.state:
                    current_messages = self.state.get_current_messages()
                    user_count = sum(1 for m in current_messages if m.role == "user")

                result = await self.prompt_builder.render_user_prompt(
                    prompt_name,
                    index=user_count,
                    params=params,
                    include_rag=include_rag,
                )
            else:
                raise ValueError(f"Cannot render prompt for role '{role}'")

            content = result.content

        # Create message
        message = LLMMessage(role=role, content=content)

        # Initialize state if this is the first message
        if self.state is None:
            conversation_id = str(uuid.uuid4())
            root_node = ConversationNode(
                message=message,
                node_id="",
                prompt_name=prompt_name,
                metadata=metadata or {},
            )
            tree = Tree(root_node)
            self.state = ConversationState(
                conversation_id=conversation_id,
                message_tree=tree,
                current_node_id="",
                metadata=self._initial_metadata,
            )
        else:
            # Add as child of current node
            current_tree_node = self.state.get_current_node()
            if current_tree_node is None:
                raise ValueError(f"Current node '{self.state.current_node_id}' not found")

            # Create new tree node
            new_tree_node = Tree(
                ConversationNode(
                    message=message,
                    node_id="",  # Will be calculated after adding to tree
                    prompt_name=prompt_name,
                    metadata=metadata or {},
                )
            )

            # Add to tree
            current_tree_node.add_child(new_tree_node)

            # Calculate and set node_id
            node_id = calculate_node_id(new_tree_node)
            new_tree_node.data.node_id = node_id

            # Move current position to new node
            self.state.current_node_id = node_id

        # Update timestamp
        self.state.updated_at = datetime.now()

        # Persist
        await self._save_state()

        return self.state.get_current_node().data

    async def complete(
        self,
        branch_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **llm_kwargs,
    ) -> LLMResponse:
        """Get LLM completion and add as child of current node.

        This method:
        1. Gets conversation history from root to current node
        2. Executes middleware (pre-LLM)
        3. Calls LLM with history
        4. Executes middleware (post-LLM)
        5. Adds assistant response as child of current node
        6. Updates current position to new node
        7. Persists to storage

        Args:
            branch_name: Optional human-readable label for this branch
            metadata: Optional metadata for the assistant message node
            **llm_kwargs: Additional arguments for LLM.complete()

        Returns:
            LLM response

        Raises:
            ValueError: If conversation has no messages yet

        Example:
            >>> # Get response
            >>> result = await manager.complete()
            >>> print(result.content)
            >>>
            >>> # Create labeled branch
            >>> result = await manager.complete(branch_name="alternative-answer")
        """
        if not self.state:
            raise ValueError("Cannot complete: no messages in conversation")

        # Get messages from root to current position
        messages = self.state.get_current_messages()

        # Execute middleware (pre-LLM) in forward order
        for mw in self.middleware:
            messages = await mw.process_request(messages, self.state)

        # Call LLM
        response = await self.llm.complete(messages, **llm_kwargs)

        # Execute middleware (post-LLM) in reverse order (onion model)
        for mw in reversed(self.middleware):
            response = await mw.process_response(response, self.state)

        # Add assistant message as child
        current_tree_node = self.state.get_current_node()
        if current_tree_node is None:
            raise ValueError(f"Current node '{self.state.current_node_id}' not found")

        # Create assistant message node
        assistant_message = LLMMessage(
            role="assistant",
            content=response.content,
        )

        assistant_metadata = metadata or {}
        assistant_metadata.update({
            "usage": response.usage,
            "model": response.model,
            "finish_reason": response.finish_reason,
        })

        new_tree_node = Tree(
            ConversationNode(
                message=assistant_message,
                node_id="",  # Will be calculated
                branch_name=branch_name,
                metadata=assistant_metadata,
            )
        )

        # Add to tree
        current_tree_node.add_child(new_tree_node)

        # Calculate node_id
        node_id = calculate_node_id(new_tree_node)
        new_tree_node.data.node_id = node_id

        # Move current position
        self.state.current_node_id = node_id
        self.state.updated_at = datetime.now()

        # Persist
        await self._save_state()

        return response

    async def stream_complete(
        self,
        branch_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **llm_kwargs,
    ) -> AsyncIterator[LLMStreamResponse]:
        """Stream LLM completion and add as child of current node.

        Similar to complete() but streams the response.

        Args:
            branch_name: Optional human-readable label for this branch
            metadata: Optional metadata for the assistant message node
            **llm_kwargs: Additional arguments for LLM.stream_complete()

        Yields:
            Streaming response chunks

        Raises:
            ValueError: If conversation has no messages yet

        Example:
            >>> async for chunk in manager.stream_complete():
            ...     print(chunk.delta, end="", flush=True)
        """
        if not self.state:
            raise ValueError("Cannot complete: no messages in conversation")

        # Get messages
        messages = self.state.get_current_messages()

        # Execute middleware (pre-LLM) in forward order
        for mw in self.middleware:
            messages = await mw.process_request(messages, self.state)

        # Stream LLM response and accumulate
        full_content = ""
        final_chunk = None
        async for chunk in self.llm.stream_complete(messages, **llm_kwargs):
            full_content += chunk.delta
            final_chunk = chunk
            yield chunk

        # Create complete response for state update
        response = LLMResponse(
            content=full_content,
            model=self.llm.config.model,
            finish_reason=final_chunk.finish_reason if final_chunk else "stop",
            usage=final_chunk.usage if final_chunk else None,
        )

        # Execute middleware (post-LLM) in reverse order (onion model)
        for mw in reversed(self.middleware):
            response = await mw.process_response(response, self.state)

        # Add assistant message as child (same as complete())
        current_tree_node = self.state.get_current_node()
        if current_tree_node is None:
            raise ValueError(f"Current node '{self.state.current_node_id}' not found")

        assistant_message = LLMMessage(role="assistant", content=response.content)

        assistant_metadata = metadata or {}
        assistant_metadata.update({
            "usage": response.usage,
            "model": response.model,
            "finish_reason": response.finish_reason,
        })

        new_tree_node = Tree(
            ConversationNode(
                message=assistant_message,
                node_id="",
                branch_name=branch_name,
                metadata=assistant_metadata,
            )
        )

        current_tree_node.add_child(new_tree_node)
        node_id = calculate_node_id(new_tree_node)
        new_tree_node.data.node_id = node_id

        self.state.current_node_id = node_id
        self.state.updated_at = datetime.now()

        await self._save_state()

    async def switch_to_node(self, node_id: str) -> None:
        """Switch current position to a different node in the tree.

        This allows exploring different branches or backtracking in the conversation.

        Args:
            node_id: Target node ID (dot-delimited, e.g., "0.1" or "")

        Raises:
            ValueError: If node_id not found in tree

        Example:
            >>> # Go back to first user message
            >>> await manager.switch_to_node("0")
            >>>
            >>> # Create alternative response
            >>> result = await manager.complete(branch_name="alternative")
            >>>
            >>> # Go back to root
            >>> await manager.switch_to_node("")
        """
        if not self.state:
            raise ValueError("No conversation state")

        # Verify node exists
        target_node = get_node_by_id(self.state.message_tree, node_id)
        if target_node is None:
            raise ValueError(f"Node '{node_id}' not found in conversation tree")

        # Update current position
        self.state.current_node_id = node_id
        self.state.updated_at = datetime.now()

        # Persist
        await self._save_state()

    async def execute_flow(
        self,
        flow: "ConversationFlow",
        initial_params: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[ConversationNode]:
        """Execute a conversation flow using FSM.

        This method executes a predefined conversation flow, yielding
        conversation nodes as the flow progresses through states.

        Args:
            flow: ConversationFlow definition
            initial_params: Optional initial parameters for the flow

        Yields:
            ConversationNode for each state in the flow

        Raises:
            ValueError: If flow execution fails

        Example:
            >>> from dataknobs_llm.conversations.flow import (
            ...     ConversationFlow, FlowState,
            ...     keyword_condition
            ... )
            >>>
            >>> # Define flow
            >>> flow = ConversationFlow(
            ...     name="support",
            ...     initial_state="greeting",
            ...     states={
            ...         "greeting": FlowState(
            ...             prompt_name="support_greeting",
            ...             transitions={
            ...                 "help": "collect_issue",
            ...                 "browse": "end"
            ...             },
            ...             transition_conditions={
            ...                 "help": keyword_condition(["help", "issue"]),
            ...                 "browse": keyword_condition(["browse", "look"])
            ...             }
            ...         )
            ...     }
            ... )
            >>>
            >>> # Execute flow
            >>> async for node in manager.execute_flow(flow):
            ...     print(f"State: {node.metadata.get('state')}")
            ...     print(f"Response: {node.content}")
        """
        from dataknobs_llm.conversations.flow import ConversationFlowAdapter

        if not self.state:
            raise ValueError("No conversation state")

        # Create adapter
        adapter = ConversationFlowAdapter(
            flow=flow,
            prompt_builder=self.prompt_builder,
            llm=self.llm
        )

        # Execute flow and yield nodes
        data = initial_params or {}
        data["conversation_id"] = self.state.conversation_id

        try:
            # Execute flow (this will internally use FSM)
            result = await adapter.execute(data)

            # Convert flow history to conversation nodes
            for state_name, response in adapter.execution_state.history:
                # Create node for this state's response
                node = ConversationNode(
                    node_id=str(uuid.uuid4()),  # Temporary ID
                    role="assistant",
                    content=response,
                    timestamp=datetime.now(),
                    metadata={
                        "state": state_name,
                        "flow_name": flow.name,
                        "flow_execution": True
                    }
                )

                # Add to conversation tree
                current_tree_node = get_node_by_id(
                    self.state.message_tree,
                    self.state.current_node_id
                )

                new_tree_node = Tree(node)
                current_tree_node.add_child(new_tree_node)
                node_id = calculate_node_id(new_tree_node)
                new_tree_node.data.node_id = node_id

                self.state.current_node_id = node_id
                self.state.updated_at = datetime.now()

                await self._save_state()

                yield node

        except Exception as e:
            import logging
            logging.error(f"Flow execution failed: {e}")
            raise ValueError(f"Flow execution failed: {str(e)}") from e

    async def get_history(self) -> List[LLMMessage]:
        """Get conversation history from root to current position.

        Returns:
            List of messages in current conversation path

        Example:
            >>> messages = await manager.get_history()
            >>> for msg in messages:
            ...     print(f"{msg.role}: {msg.content}")
        """
        if not self.state:
            return []

        return self.state.get_current_messages()

    async def get_branches(self, node_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get information about branches from a given node.

        Args:
            node_id: Node to get branches from (default: current node)

        Returns:
            List of branch info dicts with keys:
                - node_id: ID of child node
                - branch_name: Optional branch name
                - role: Message role
                - preview: First 100 chars of content
                - timestamp: When created

        Example:
            >>> branches = await manager.get_branches()
            >>> for branch in branches:
            ...     print(f"{branch['branch_name']}: {branch['preview']}")
        """
        if not self.state:
            return []

        # Default to current node
        if node_id is None:
            node_id = self.state.current_node_id

        # Get node
        node = get_node_by_id(self.state.message_tree, node_id)
        if node is None or not node.children:
            return []

        # Build branch info
        branches = []
        for child in node.children:
            data = child.data
            branches.append({
                "node_id": data.node_id,
                "branch_name": data.branch_name,
                "role": data.message.role,
                "preview": data.message.content[:100],
                "timestamp": data.timestamp,
            })

        return branches

    async def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to conversation.

        Args:
            key: Metadata key
            value: Metadata value

        Example:
            >>> await manager.add_metadata("user_id", "alice")
            >>> await manager.add_metadata("session", "abc123")
        """
        if not self.state:
            raise ValueError("No conversation state")

        self.state.metadata[key] = value
        self.state.updated_at = datetime.now()
        await self._save_state()

    async def _save_state(self) -> None:
        """Persist current state to storage."""
        if self.state:
            await self.storage.save_conversation(self.state)

    @property
    def conversation_id(self) -> Optional[str]:
        """Get conversation ID."""
        return self.state.conversation_id if self.state else None

    @property
    def current_node_id(self) -> Optional[str]:
        """Get current node ID."""
        return self.state.current_node_id if self.state else None
