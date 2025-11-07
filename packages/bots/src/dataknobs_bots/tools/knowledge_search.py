"""Knowledge search tool for RAG integration."""

from typing import Any

from dataknobs_llm.tools import Tool


class KnowledgeSearchTool(Tool):
    """Tool for searching the knowledge base.

    This tool allows LLMs to search the bot's knowledge base
    for relevant information during conversations.

    Example:
        ```python
        # Create tool with knowledge base
        tool = KnowledgeSearchTool(knowledge_base=kb)

        # Register with bot
        bot.tool_registry.register_tool(tool)

        # LLM can now call the tool
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

    async def execute(self, query: str, max_results: int = 3, **kwargs: Any) -> dict[str, Any]:
        """Execute knowledge base search.

        Args:
            query: Search query text
            max_results: Maximum number of results (default: 3)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with search results:
                - query: Original query
                - results: List of relevant chunks
                - num_results: Number of results found

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

        # Search knowledge base
        results = await self.knowledge_base.query(query, k=max_results)

        # Format response
        return {
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
