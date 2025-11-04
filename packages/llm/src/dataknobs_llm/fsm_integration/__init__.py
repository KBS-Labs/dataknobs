"""FSM integration for LLM workflows.

This module provides integration between the dataknobs-fsm state machine
package and LLM functionality, enabling complex conversational workflows,
RAG patterns, chain-of-thought reasoning, and agent-based systems.

For LLM workflow patterns using FSM, see:
- workflows.LLMWorkflow
- workflows.LLMWorkflowConfig
- workflows.create_rag_workflow
- workflows.create_chain_workflow

For FSM resource management, see:
- resources.LLMResource

For FSM function library, see:
- functions.PromptBuilder
- functions.LLMCaller
- functions.ResponseValidator

Note: This module contains code that was previously in the dataknobs-fsm
package. It has been moved here to consolidate all LLM functionality in
the dataknobs-llm package and eliminate duplication.
"""

from .workflows import (
    WorkflowType,
    LLMStep,
    RAGConfig,
    AgentConfig,
    LLMWorkflowConfig,
    VectorRetriever,
    LLMWorkflow,
    create_simple_llm_workflow,
    create_rag_workflow,
    create_chain_workflow,
)

from .resources import (
    LLMProvider,
    LLMSession,
    LLMResource,
)

from .functions import (
    PromptBuilder,
    LLMCaller,
    ResponseValidator,
    FunctionCaller,
    ConversationManager,
    EmbeddingGenerator,
    build_prompt,
    call_llm,
    validate_response,
    call_function,
    manage_conversation,
    generate_embeddings,
)

__all__ = [
    # Workflows
    'WorkflowType',
    'LLMStep',
    'RAGConfig',
    'AgentConfig',
    'LLMWorkflowConfig',
    'VectorRetriever',
    'LLMWorkflow',
    'create_simple_llm_workflow',
    'create_rag_workflow',
    'create_chain_workflow',
    # Resources
    'LLMProvider',
    'LLMSession',
    'LLMResource',
    # Functions
    'PromptBuilder',
    'LLMCaller',
    'ResponseValidator',
    'FunctionCaller',
    'ConversationManager',
    'EmbeddingGenerator',
    'build_prompt',
    'call_llm',
    'validate_response',
    'call_function',
    'manage_conversation',
    'generate_embeddings',
]
