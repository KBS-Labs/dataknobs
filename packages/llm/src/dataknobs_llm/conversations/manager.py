"""Conversation manager for multi-turn interactions with LLMs.

This module provides ConversationManager, a comprehensive system for managing
multi-turn LLM conversations with advanced features like:

- **Tree-based History**: Conversations stored as trees, enabling branching
- **Persistence**: Automatic state saving to any storage backend
- **RAG Caching**: Reuse search results across conversation branches
- **Middleware**: Pre/post-processing pipeline for all LLM calls
- **Cost Tracking**: Automatic API cost calculation and accumulation
- **Flow Execution**: FSM-based conversation flows with state management
- **Resumability**: Save and resume conversations across sessions

Architecture:
    ConversationManager orchestrates three core components:

    1. **AsyncLLMProvider**: Handles LLM API calls (OpenAI, Anthropic, Ollama)
    2. **AsyncPromptBuilder**: Renders prompts with RAG integration
    3. **ConversationStorage**: Persists conversation state (Memory, File, S3, Postgres)

    Conversations are stored as trees where each node represents a message.
    Branching occurs when multiple responses are generated from the same point,
    enabling A/B testing, alternative explorations, and retry scenarios.

Example:
    ```python
    from dataknobs_llm import create_llm_provider
    from dataknobs_llm.prompts import AsyncPromptBuilder
    from dataknobs_llm.conversations import (
        ConversationManager,
        DataknobsConversationStorage
    )
    from dataknobs_data import database_factory

    # Setup components
    llm = create_llm_provider("openai", api_key="sk-...")
    builder = AsyncPromptBuilder.create(library_path="./prompts")
    db = database_factory.create(backend="memory")
    storage = DataknobsConversationStorage(db)

    # Create conversation
    manager = await ConversationManager.create(
        llm=llm,
        prompt_builder=builder,
        storage=storage,
        system_prompt_name="helpful_assistant"
    )

    # Add user message and get response
    await manager.add_message(
        role="user",
        content="What is Python?"
    )
    response = await manager.complete()
    print(response.content)

    # Continue conversation
    await manager.add_message(
        role="user",
        content="Show me a decorator example"
    )
    response = await manager.complete()

    # Create alternative response branch
    await manager.switch_to_node("0.0")  # Back to first assistant response
    alt_response = await manager.complete(branch_name="alternative")

    # Resume later
    conv_id = manager.conversation_id
    manager2 = await ConversationManager.resume(
        conversation_id=conv_id,
        llm=llm,
        prompt_builder=builder,
        storage=storage
    )
    ```

See Also:
    - ConversationStorage: Storage interface and implementations
    - ConversationMiddleware: Middleware system for request/response processing
    - ConversationFlow: FSM-based conversation flows
    - AsyncPromptBuilder: Prompt rendering with RAG integration
"""

import uuid
from typing import List, Dict, Any, AsyncIterator
from datetime import datetime

from dataknobs_structures.tree import Tree
from dataknobs_llm.llm import AsyncLLMProvider, LLMMessage, LLMResponse, LLMStreamResponse
from dataknobs_llm.prompts import AsyncPromptBuilder
from dataknobs_llm.conversations.flow.flow import ConversationFlow
from dataknobs_llm.conversations.middleware import ConversationMiddleware
from dataknobs_llm.conversations.storage import (
    ConversationNode,
    ConversationState,
    ConversationStorage,
    calculate_node_id,
    get_node_by_id,
)


class ConversationManager:
    """Manages multi-turn conversations with persistence and branching.

    This class orchestrates conversations by:
    - Tracking message history with tree-based branching
    - Managing conversation state
    - Persisting to storage backend
    - Supporting multiple conversation branches

    The conversation history is stored as a tree structure where:
    - Root node contains the initial system prompt (if any)
    - Each message is a tree node with a dot-delimited ID (e.g., "0.1.2")
    - Branches occur when multiple children are added to the same node
    - Current position tracks where you are in the conversation tree

    Attributes:
        llm: LLM provider for completions
        prompt_builder: Prompt builder with library
        storage: Storage backend for persistence
        state: Current conversation state (tree, metadata, position)
        middleware: List of middleware to execute on requests/responses
        cache_rag_results: Whether to store RAG metadata in nodes
        reuse_rag_on_branch: Whether to reuse cached RAG across branches
        conversation_id: Unique conversation identifier
        current_node_id: Current position in conversation tree

    Example:
        ```python
        # Create conversation
        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            storage=storage_backend,
            system_prompt_name="helpful_assistant"
        )

        # Add user message
        await manager.add_message(
            prompt_name="user_query",
            params={"question": "What is Python?"},
            role="user"
        )

        # Get LLM response
        result = await manager.complete()

        # Continue conversation
        await manager.add_message(
            content="Tell me more about decorators",
            role="user"
        )
        result = await manager.complete()

        # Create alternative response branch
        await manager.switch_to_node("0")  # Back to first user message
        result2 = await manager.complete(branch_name="alt-response")

        # Resume after interruption
        manager2 = await ConversationManager.resume(
            conversation_id=manager.conversation_id,
            llm=llm,
            prompt_builder=builder,
            storage=storage_backend
        )
        ```

    Note:
        Tree-based branching enables:

        - **A/B Testing**: Generate multiple responses from the same context
        - **Retry Logic**: Try again from a previous point after failures
        - **Alternative Explorations**: Explore different conversation paths
        - **Debugging**: Compare different middleware or RAG configurations

        Node IDs use dot notation (e.g., "0.1.2" means 3rd child of 2nd child
        of 1st child of root). The root node has ID "".

        State is automatically persisted after every operation. Use
        `resume()` to continue conversations across sessions or servers.

    See Also:
        create: Create a new conversation
        resume: Resume an existing conversation
        add_message: Add user/system message
        complete: Get LLM completion (blocking)
        stream_complete: Get LLM completion (streaming)
        switch_to_node: Navigate to different branch
        get_branches: List available branches
        get_total_cost: Calculate cumulative cost
        ConversationStorage: Storage backend implementations
        ConversationMiddleware: Request/response processing
        ConversationFlow: FSM-based conversation flows
    """

    def __init__(
        self,
        llm: AsyncLLMProvider,
        prompt_builder: AsyncPromptBuilder,
        storage: ConversationStorage,
        state: ConversationState | None = None,
        metadata: Dict[str, Any] | None = None,
        middleware: List[ConversationMiddleware] | None = None,
        cache_rag_results: bool = False,
        reuse_rag_on_branch: bool = False,
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
            cache_rag_results: If True, store RAG metadata in node metadata
                             for debugging and transparency
            reuse_rag_on_branch: If True, reuse cached RAG results when
                               possible (useful for testing/branching)
        """
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.storage = storage
        self.state = state
        self._initial_metadata = metadata or {}
        self.middleware = middleware or []
        self.cache_rag_results = cache_rag_results
        self.reuse_rag_on_branch = reuse_rag_on_branch

    @classmethod
    async def create(
        cls,
        llm: AsyncLLMProvider,
        prompt_builder: AsyncPromptBuilder,
        storage: ConversationStorage,
        system_prompt_name: str | None = None,
        system_params: Dict[str, Any] | None = None,
        metadata: Dict[str, Any] | None = None,
        middleware: List[ConversationMiddleware] | None = None,
        cache_rag_results: bool = False,
        reuse_rag_on_branch: bool = False,
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
            cache_rag_results: If True, store RAG metadata in node metadata
            reuse_rag_on_branch: If True, reuse cached RAG results when possible

        Returns:
            Initialized ConversationManager

        Example:
            >>> manager = await ConversationManager.create(
            ...     llm=llm,
            ...     prompt_builder=builder,
            ...     storage=storage,
            ...     system_prompt_name="helpful_assistant",
            ...     cache_rag_results=True
            ... )
        """
        manager = cls(
            llm=llm,
            prompt_builder=prompt_builder,
            storage=storage,
            metadata=metadata,
            middleware=middleware,
            cache_rag_results=cache_rag_results,
            reuse_rag_on_branch=reuse_rag_on_branch,
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
        middleware: List[ConversationMiddleware] | None = None,
        cache_rag_results: bool = False,
        reuse_rag_on_branch: bool = False,
    ) -> "ConversationManager":
        """Resume an existing conversation.

        Args:
            conversation_id: Existing conversation ID
            llm: LLM provider
            prompt_builder: Prompt builder
            storage: Storage backend
            middleware: Optional list of middleware to execute
            cache_rag_results: If True, store RAG metadata in node metadata
            reuse_rag_on_branch: If True, reuse cached RAG results when possible

        Returns:
            ConversationManager with restored state

        Raises:
            ValueError: If conversation not found

        Example:
            >>> manager = await ConversationManager.resume(
            ...     conversation_id="conv-123",
            ...     llm=llm,
            ...     prompt_builder=builder,
            ...     storage=storage,
            ...     cache_rag_results=True
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
            cache_rag_results=cache_rag_results,
            reuse_rag_on_branch=reuse_rag_on_branch,
        )

        return manager

    async def add_message(
        self,
        role: str,
        content: str | None = None,
        prompt_name: str | None = None,
        params: Dict[str, Any] | None = None,
        include_rag: bool = True,
        rag_configs: List[Dict[str, Any]] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> ConversationNode:
        """Add a message to the current conversation node.

        Either content or prompt_name must be provided. If using a prompt
        with RAG configuration, the RAG searches will be executed and results
        will be automatically inserted into the prompt.

        Args:
            role: Message role ("system", "user", or "assistant")
            content: Direct message content (if not using prompt)
            prompt_name: Name of prompt template to render
            params: Parameters for prompt rendering
            include_rag: Whether to execute RAG searches for prompts
            rag_configs: RAG configurations for inline content (only used when
                        content is provided without prompt_name). Allows inline
                        prompts to benefit from RAG enhancement.
            metadata: Optional metadata for this message node

        Returns:
            The created ConversationNode

        Raises:
            ValueError: If neither content nor prompt_name provided

        Example:
            ```python
            # Add message from prompt
            await manager.add_message(
                role="user",
                prompt_name="code_question",
                params={"code": code_snippet}
            )

            # Add direct message
            await manager.add_message(
                role="user",
                content="What is Python?"
            )

            # Add inline message with RAG enhancement
            await manager.add_message(
                role="system",
                content="You are a helpful assistant. Use the context below.",
                rag_configs=[{
                    "adapter_name": "docs",
                    "query": "assistant guidelines",
                    "placeholder": "CONTEXT",
                    "k": 3
                }]
            )

            # Add system prompt with custom metadata
            await manager.add_message(
                role="system",
                prompt_name="expert_coder",
                metadata={"version": "v2"}
            )
            ```

        Note:
            **RAG Caching Behavior**:

            If `cache_rag_results=True` and `reuse_rag_on_branch=True` were
            set during ConversationManager creation, this method will:

            1. Check if the same prompt+role was used elsewhere in the tree
            2. Check if the RAG query parameters match (via query hash)
            3. Reuse cached RAG results if found (no re-search!)
            4. Store new RAG results if not found

            This is particularly useful when exploring conversation branches,
            as you can avoid redundant searches for the same information.

        See Also:
            complete: Get LLM response after adding message
            get_rag_metadata: Retrieve RAG metadata from a node
        """
        if not content and not prompt_name:
            raise ValueError("Either content or prompt_name must be provided")

        # Render prompt if needed
        rag_metadata_to_store = None
        if prompt_name:
            params = params or {}

            # Check if we should try to reuse cached RAG
            cached_rag = None
            if self.reuse_rag_on_branch and include_rag:
                cached_rag = await self._find_cached_rag(prompt_name, role, params)

            if role == "system":
                result = await self.prompt_builder.render_system_prompt(
                    prompt_name,
                    params=params,
                    include_rag=include_rag,
                    return_rag_metadata=self.cache_rag_results,
                    cached_rag=cached_rag,
                )
            elif role == "user":
                result = await self.prompt_builder.render_user_prompt(
                    prompt_name,
                    params=params,
                    include_rag=include_rag,
                    return_rag_metadata=self.cache_rag_results,
                    cached_rag=cached_rag,
                )
            else:
                raise ValueError(f"Cannot render prompt for role '{role}'")

            content = result.content

            # Store RAG metadata if caching is enabled and metadata was captured
            if self.cache_rag_results and result.rag_metadata:
                rag_metadata_to_store = result.rag_metadata

        elif content and include_rag and rag_configs:
            # Render inline content with RAG enhancement
            params = params or {}
            if role == "system":
                result = await self.prompt_builder.render_inline_system_prompt(
                    content,
                    params=params,
                    rag_configs=rag_configs,
                    include_rag=True,
                    return_rag_metadata=self.cache_rag_results,
                )
            elif role == "user":
                result = await self.prompt_builder.render_inline_user_prompt(
                    content,
                    params=params,
                    rag_configs=rag_configs,
                    include_rag=True,
                    return_rag_metadata=self.cache_rag_results,
                )
            else:
                # For assistant role, just use content as-is
                result = None

            if result:
                content = result.content
                if self.cache_rag_results and result.rag_metadata:
                    rag_metadata_to_store = result.rag_metadata

        # Create message
        message = LLMMessage(role=role, content=content)

        # Prepare node metadata
        node_metadata = metadata or {}
        if rag_metadata_to_store:
            node_metadata["rag_metadata"] = rag_metadata_to_store

        # Initialize state if this is the first message
        if self.state is None:
            conversation_id = str(uuid.uuid4())
            root_node = ConversationNode(
                message=message,
                node_id="",
                prompt_name=prompt_name,
                metadata=node_metadata,
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
                    metadata=node_metadata,
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
        branch_name: str | None = None,
        metadata: Dict[str, Any] | None = None,
        **llm_kwargs: Any,
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
            LLM response with content, usage, and cost information

        Raises:
            ValueError: If conversation has no messages yet

        Example:
            ```python
            # Get response
            result = await manager.complete()
            print(result.content)
            print(f"Cost: ${result.cost_usd:.4f}")

            # Create labeled branch
            result = await manager.complete(branch_name="alternative-answer")

            # With LLM parameters
            result = await manager.complete(temperature=0.9, max_tokens=500)
            ```

        Note:
            **Middleware Execution Order** (Onion Model):

            - Pre-LLM: middleware[0] → middleware[1] → ... → middleware[N]
            - LLM call happens
            - Post-LLM: middleware[N] → ... → middleware[1] → middleware[0]

            This "onion" pattern ensures that middleware wraps around the LLM
            call symmetrically. For example, if middleware[0] starts a timer
            in `process_request()`, it will be the last to run in
            `process_response()` and can log the total elapsed time.

            **Automatic Cost Tracking**:

            The response includes `cost_usd` (this call) and `cumulative_cost_usd`
            (total conversation cost) if the LLM provider returns usage statistics.

        See Also:
            stream_complete: Streaming version for real-time output
            add_message: Add user/system message before calling complete
            switch_to_node: Navigate to different branch before completing
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

        # Calculate and track cost
        self._calculate_and_track_cost(response, assistant_metadata)

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
        branch_name: str | None = None,
        metadata: Dict[str, Any] | None = None,
        **llm_kwargs,
    ) -> AsyncIterator[LLMStreamResponse]:
        r"""Stream LLM completion and add as child of current node.

        Similar to complete() but streams the response incrementally for
        real-time display. The complete response is automatically added
        to the conversation tree after streaming finishes.

        Args:
            branch_name: Optional human-readable label for this branch
            metadata: Optional metadata for the assistant message node
            **llm_kwargs: Additional arguments for LLM.stream_complete()

        Yields:
            Streaming response chunks with delta, usage, and final metadata

        Raises:
            ValueError: If conversation has no messages yet

        Example:
            ```python
            # Real-time display
            async for chunk in manager.stream_complete():
                print(chunk.delta, end="", flush=True)
            print()  # New line after streaming

            # Accumulate full response
            full_text = ""
            async for chunk in manager.stream_complete():
                full_text += chunk.delta
                if chunk.is_final:
                    print(f"\nFinished. Total: {len(full_text)} chars")
                    print(f"Cost: ${chunk.usage.get('cost_usd', 0):.4f}")

            # With branch label
            async for chunk in manager.stream_complete(
                branch_name="creative-response",
                temperature=0.9
            ):
                print(chunk.delta, end="", flush=True)
            ```

        Note:
            The middleware execution order is the same as `complete()`:
            pre-LLM middleware runs before streaming starts, post-LLM
            middleware runs after the stream completes.

            Cost and usage information is only available in the final chunk
            (when `chunk.is_final == True`).

        See Also:
            complete: Non-streaming version for simple use cases
            add_message: Add message before streaming
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

        # Calculate and track cost
        self._calculate_and_track_cost(response, assistant_metadata)

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
        flow: ConversationFlow,
        initial_params: Dict[str, Any] | None = None
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
            await adapter.execute(data)

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
            raise ValueError(f"Flow execution failed: {e!s}") from e

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

    async def get_branches(self, node_id: str | None = None) -> List[Dict[str, Any]]:
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

    async def _find_cached_rag(
        self,
        prompt_name: str,
        role: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        """Search conversation history for cached RAG metadata.

        This method searches the entire conversation tree for cached RAG metadata
        that matches both the prompt name/role AND the resolved RAG query parameters.
        Query matching is done via query hashes.

        Args:
            prompt_name: Name of the prompt to find cached RAG for
            role: Role of the prompt ("system" or "user")
            params: Parameters for the prompt (used to match RAG queries)

        Returns:
            Cached RAG metadata dictionary if found, None otherwise

        Example:
            >>> cached = await manager._find_cached_rag("code_question", "user", {"topic": "decorators"})
            >>> if cached:
            ...     print(f"Found cached RAG with {len(cached)} placeholders")
        """
        if not self.state:
            return None

        # Get RAG configs for this prompt to determine what queries we're looking for
        rag_configs = self.prompt_builder.library.get_prompt_rag_configs(
            prompt_name=prompt_name,
            prompt_type="system" if role == "system" else "user"
        )

        if not rag_configs:
            return None

        # Compute the query hashes we're looking for
        from jinja2 import Template
        target_hashes_by_placeholder = {}
        for rag_config in rag_configs:
            placeholder = rag_config.get("placeholder", "RAG_CONTENT")
            adapter_name = rag_config.get("adapter_name", "")
            query_template = rag_config.get("query", "")

            # Render the query template with params
            try:
                template = Template(query_template)
                resolved_query = template.render(params)

                # Compute hash
                query_hash = self.prompt_builder._compute_rag_query_hash(adapter_name, resolved_query)
                target_hashes_by_placeholder[placeholder] = query_hash
            except Exception:
                # If query rendering fails, we can't match cache
                continue

        if not target_hashes_by_placeholder:
            return None

        # Search entire tree for matching cached RAG (BFS to find any match)
        from collections import deque
        queue = deque([self.state.message_tree])

        while queue:
            tree_node = queue.popleft()
            node_data = tree_node.data

            # Check if this node has the same prompt name and role
            if (node_data.prompt_name == prompt_name and
                node_data.message.role == role):

                # Check if RAG metadata exists
                rag_metadata = node_data.metadata.get("rag_metadata")
                if rag_metadata:
                    # Check if query hashes match for all placeholders
                    all_match = True
                    for placeholder, target_hash in target_hashes_by_placeholder.items():
                        if placeholder not in rag_metadata:
                            all_match = False
                            break
                        cached_hash = rag_metadata[placeholder].get("query_hash")
                        if cached_hash != target_hash:
                            all_match = False
                            break

                    if all_match:
                        return rag_metadata

            # Add children to queue (if any)
            if tree_node.children:
                queue.extend(tree_node.children)

        return None

    def get_rag_metadata(self, node_id: str | None = None) -> Dict[str, Any] | None:
        """Get RAG metadata from a conversation node.

        This method retrieves the cached RAG metadata from a specific node,
        which includes information about RAG searches executed during prompt
        rendering (queries, results, query hashes, etc.).

        Args:
            node_id: Node ID to retrieve metadata from (default: current node)

        Returns:
            RAG metadata dictionary if present, None otherwise. Structure:

            ```python
            {
                "PLACEHOLDER_NAME": {
                    "query": "resolved RAG query",
                    "query_hash": "hash of adapter+query",
                    "results": [...],  # Search results
                    "adapter_name": "name of RAG adapter used"
                },
                ...  # One entry per RAG placeholder
            }
            ```

        Raises:
            ValueError: If node_id not found in conversation tree

        Example:
            ```python
            # Get RAG metadata from current node
            metadata = manager.get_rag_metadata()
            if metadata:
                for placeholder, rag_data in metadata.items():
                    print(f"Placeholder: {placeholder}")
                    print(f"  Query: {rag_data['query']}")
                    print(f"  Adapter: {rag_data['adapter_name']}")
                    print(f"  Results: {len(rag_data['results'])} items")
                    print(f"  Hash: {rag_data['query_hash']}")

            # Get RAG metadata from specific node
            metadata = manager.get_rag_metadata(node_id="0.1")

            # Check if RAG was used for a message
            if manager.get_rag_metadata():
                print("This message used RAG-enhanced prompt")
            else:
                print("This message used direct content")
            ```

        Note:
            RAG metadata is only available if `cache_rag_results=True` was
            set during ConversationManager creation. This metadata is useful
            for debugging RAG behavior, understanding what information was
            retrieved, and implementing RAG result caching across branches.

        See Also:
            add_message: Method that executes RAG and stores metadata
            reuse_rag_on_branch: Parameter enabling RAG cache reuse
        """
        if not self.state:
            return None

        # Default to current node
        if node_id is None:
            node_id = self.state.current_node_id

        # Get node
        tree_node = get_node_by_id(self.state.message_tree, node_id)
        if tree_node is None:
            raise ValueError(f"Node '{node_id}' not found in conversation tree")

        # Return RAG metadata if present
        return tree_node.data.metadata.get("rag_metadata")

    async def _save_state(self) -> None:
        """Persist current state to storage."""
        if self.state:
            await self.storage.save_conversation(self.state)

    @property
    def conversation_id(self) -> str | None:
        """Get conversation ID."""
        return self.state.conversation_id if self.state else None

    @property
    def current_node_id(self) -> str | None:
        """Get current node ID."""
        return self.state.current_node_id if self.state else None

    def get_metadata(self, key: str | None = None, default: Any = None) -> Any:
        """Get conversation metadata.

        This provides access to the conversation-level metadata stored in
        the ConversationState. Metadata is useful for storing client_id,
        user_id, session information, and other contextual data.

        Args:
            key: Specific metadata key to retrieve. If None, returns all metadata.
            default: Default value if key not found (only used when key is specified)

        Returns:
            Metadata value, all metadata dict, or default value

        Example:
            >>> # Get all metadata
            >>> metadata = manager.get_metadata()
            >>> print(metadata)  # {'client_id': 'abc', 'user_id': '123'}
            >>>
            >>> # Get specific key
            >>> client_id = manager.get_metadata('client_id')
            >>> print(client_id)  # 'abc'
            >>>
            >>> # Get with default
            >>> tier = manager.get_metadata('user_tier', default='free')
        """
        if not self.state:
            return default if key else {}

        if key is None:
            return self.state.metadata
        else:
            return self.state.metadata.get(key, default)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set conversation metadata.

        Updates a specific key in the conversation metadata. The metadata
        is automatically persisted when save() is called.

        Args:
            key: Metadata key to set
            value: Metadata value

        Example:
            >>> manager.set_metadata('client_id', 'client-abc')
            >>> manager.set_metadata('user_tier', 'premium')
            >>> await manager.save()
        """
        if self.state:
            self.state.metadata[key] = value

    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """Update multiple metadata fields at once.

        Args:
            updates: Dictionary of metadata key-value pairs to update

        Example:
            >>> manager.update_metadata({
            ...     'client_id': 'client-abc',
            ...     'user_id': 'user-456',
            ...     'session_id': 'sess-789'
            ... })
            >>> await manager.save()
        """
        if self.state:
            self.state.metadata.update(updates)

    def remove_metadata(self, key: str) -> None:
        """Remove a metadata key.

        Args:
            key: Metadata key to remove

        Example:
            >>> manager.remove_metadata('temporary_flag')
            >>> await manager.save()
        """
        if self.state and key in self.state.metadata:
            del self.state.metadata[key]

    def get_total_cost(self) -> float:
        """Get total accumulated cost for this conversation in USD.

        Calculates the sum of all LLM API costs from the conversation history.
        Requires that cost_usd was set on LLMResponses.

        Returns:
            Total cost in USD, or 0.0 if no cost data available

        Example:
            >>> total = manager.get_total_cost()
            >>> print(f"Total cost: ${total:.4f}")
        """
        if not self.state:
            return 0.0

        total = 0.0

        # Walk the tree and sum costs from all assistant message nodes
        def walk_tree(node: Tree) -> None:
            nonlocal total
            if node.data and node.data.metadata:
                cost = node.data.metadata.get('cost_usd')
                if cost is not None:
                    total += cost

            for child in node.children:
                walk_tree(child)

        walk_tree(self.state.message_tree)
        return total

    def get_cost_by_branch(self, node_id: str | None = None) -> float:
        """Get accumulated cost for a specific conversation branch.

        Calculates the cost from root to a specific node (defaults to current).

        Args:
            node_id: Node ID to calculate cost to. If None, uses current node.

        Returns:
            Cost in USD for this branch, or 0.0 if no cost data

        Example:
            >>> # Get cost of current branch
            >>> current_cost = manager.get_cost_by_branch()
            >>>
            >>> # Get cost of specific branch
            >>> alt_cost = manager.get_cost_by_branch("0.1")
        """
        if not self.state:
            return 0.0

        target_node_id = node_id or self.state.current_node_id

        # Get messages in this branch

        # Walk from root to target node
        if not target_node_id or target_node_id == "":
            # Just root node
            return 0.0

        indexes = [int(i) for i in target_node_id.split(".")]

        total = 0.0
        current = self.state.message_tree

        for idx in indexes:
            if idx < len(current.children):
                current = current.children[idx]
                if current.data and current.data.metadata:
                    cost = current.data.metadata.get('cost_usd')
                    if cost is not None:
                        total += cost

        return total

    def _calculate_and_track_cost(
        self,
        response: LLMResponse,
        metadata: Dict[str, Any]
    ) -> None:
        """Calculate cost for a response and add to metadata.

        This is an internal helper that uses the CostCalculator utility
        to estimate costs and track them in the conversation.

        Args:
            response: LLM response to calculate cost for
            metadata: Metadata dict to add cost information to
        """
        try:
            from dataknobs_llm.llm.utils import CostCalculator

            if response.usage:
                cost = CostCalculator.calculate_cost(response, response.model)
                if cost is not None:
                    # Add to response
                    response.cost_usd = cost

                    # Calculate cumulative cost
                    cumulative = self.get_total_cost() + cost
                    response.cumulative_cost_usd = cumulative

                    # Store in metadata
                    metadata['cost_usd'] = cost
                    metadata['cumulative_cost_usd'] = cumulative
        except Exception as e:
            # Don't fail the conversation if cost calculation fails
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to calculate cost: {e}")
