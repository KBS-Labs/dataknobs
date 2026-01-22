"""Knowledge search tool for RAG integration."""

import logging
from typing import Any

from dataknobs_llm.tools import ContextAwareTool, ToolExecutionContext

logger = logging.getLogger(__name__)


class KnowledgeSearchTool(ContextAwareTool):
    """Tool for searching the knowledge base.

    This tool allows LLMs to search the bot's knowledge base
    for relevant information during conversations.

    Demonstrates the umbrella pattern for tools:
    - Static dependency: knowledge_base (via constructor injection)
    - Dynamic context: conversation_id, user_id (via ToolExecutionContext)

    Example:
        ```python
        # Create tool with knowledge base (static dependency)
        tool = KnowledgeSearchTool(knowledge_base=kb)

        # Register with bot
        bot.tool_registry.register_tool(tool)

        # LLM can now call the tool
        # Context is automatically injected by reasoning strategy
        results = await tool.execute(
            query="How do I configure the database?",
            max_results=3
        )
        ```
    """

    def __init__(self, knowledge_base: Any, name: str = "knowledge_search"):
        """Initialize knowledge search tool.

        Args:
            knowledge_base: RAGKnowledgeBase instance to search
            name: Tool name (default: knowledge_search)
        """
        super().__init__(
            name=name,
            description="Search the knowledge base for relevant information. "
            "Use this when you need to find documentation, examples, or "
            "specific information to answer user questions.",
        )
        # Static dependency - doesn't change per-request
        self.knowledge_base = knowledge_base

    @property
    def schema(self) -> dict[str, Any]:
        """Get JSON schema for tool parameters.

        Returns:
            JSON Schema for the tool parameters
        """
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or question to find information about",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        query: str,
        max_results: int = 3,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute knowledge base search with context.

        Args:
            context: Execution context with conversation/user info
            query: Search query text
            max_results: Maximum number of results (default: 3)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with search results:
                - query: Original query
                - results: List of relevant chunks
                - num_results: Number of results found
                - conversation_id: ID of conversation (if available)

        Example:
            ```python
            result = await tool.execute(
                query="How do I configure the database?",
                max_results=3
            )
            for chunk in result['results']:
                print(f"{chunk['heading_path']}: {chunk['text']}")
            ```
        """
        # Clamp max_results to valid range
        max_results = max(1, min(10, max_results))

        # Log search with context for observability
        logger.debug(
            "Knowledge search",
            extra={
                "query": query,
                "max_results": max_results,
                "conversation_id": context.conversation_id,
                "user_id": context.user_id,
            },
        )

        # Search knowledge base
        results = await self.knowledge_base.query(query, k=max_results)

        # Format response with optional context info
        response: dict[str, Any] = {
            "query": query,
            "results": [
                {
                    "text": r["text"],
                    "source": r["source"],
                    "heading": r["heading_path"],
                    "similarity": round(r["similarity"], 3),
                }
                for r in results
            ],
            "num_results": len(results),
        }

        # Include conversation_id for traceability if available
        if context.conversation_id:
            response["conversation_id"] = context.conversation_id

        return response
