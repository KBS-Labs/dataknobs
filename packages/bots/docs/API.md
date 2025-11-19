# API Reference

Complete API documentation for DataKnobs Bots.

> **📖 See:** [Complete Auto-generated API Reference](../../../api/reference/bots.md)
>
> The complete API reference is auto-generated from source code docstrings for accuracy and maintainability.

---

## Quick Links

For convenience, here are direct links to key classes in the complete reference:

### Core Classes
- [DynaBot](../../../api/reference/bots.md#dataknobs_bots.DynaBot) - Main bot class
- [BotContext](../../../api/reference/bots.md#dataknobs_bots.BotContext) - Execution context
- [BotRegistry](../../../api/reference/bots.md#dataknobs_bots.BotRegistry) - Multi-tenant registry

### Memory
- [Memory](../../../api/reference/bots.md#dataknobs_bots.Memory) - Base memory class
- [BufferMemory](../../../api/reference/bots.md#dataknobs_bots.BufferMemory) - Sliding window memory
- [VectorMemory](../../../api/reference/bots.md#dataknobs_bots.VectorMemory) - Semantic search memory

### Knowledge Base
- [RAGKnowledgeBase](../../../api/reference/bots.md#dataknobs_bots.RAGKnowledgeBase) - RAG implementation

### Reasoning
- [ReasoningStrategy](../../../api/reference/bots.md#dataknobs_bots.ReasoningStrategy) - Base strategy
- [SimpleReasoning](../../../api/reference/bots.md#dataknobs_bots.SimpleReasoning) - Direct LLM response
- [ReActReasoning](../../../api/reference/bots.md#dataknobs_bots.ReActReasoning) - Tool-using agent

### Tools
- [KnowledgeSearchTool](../../../api/reference/bots.md#dataknobs_bots.KnowledgeSearchTool) - KB search tool

### Factory Functions
- [create_memory_from_config](../../../api/reference/bots.md#dataknobs_bots.create_memory_from_config)
- [create_knowledge_base_from_config](../../../api/reference/bots.md#dataknobs_bots.create_knowledge_base_from_config)
- [create_reasoning_from_config](../../../api/reference/bots.md#dataknobs_bots.create_reasoning_from_config)

---

## See Also

- [User Guide](../guides/user-guide.md) - Tutorials and how-to guides
- [Configuration Reference](../guides/configuration.md) - Complete configuration options
- [Tools Development](../guides/tools.md) - Creating custom tools
- [Architecture](../guides/architecture.md) - System design
- [Examples](../examples/index.md) - Working code examples
