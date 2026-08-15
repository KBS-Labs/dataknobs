"""Tests for FSM workflow patterns.

These tests verify the LLM workflow patterns that integrate with FSM.
Migrated from dataknobs-fsm package.
"""

from dataknobs_llm.fsm_integration import (
    WorkflowType,
    LLMStep,
    RAGConfig,
    AgentConfig,
    LLMWorkflowConfig,
    LLMWorkflow,
    create_simple_llm_workflow,
    create_rag_workflow,
    create_chain_workflow,
)


def test_workflow_imports():
    """Test that workflow classes can be imported."""
    assert WorkflowType is not None
    assert LLMStep is not None
    assert RAGConfig is not None
    assert AgentConfig is not None
    assert LLMWorkflowConfig is not None
    assert LLMWorkflow is not None


def test_workflow_factory_functions():
    """Test that workflow factory functions can be imported."""
    assert create_simple_llm_workflow is not None
    assert create_rag_workflow is not None
    assert create_chain_workflow is not None


# Coverage gap, tracked: `workflows.py` is 760 lines and the assertions above
# are the whole of its test suite — they check that names import. The tests that
# covered this module were deleted by the FSM -> LLM migration (`eb1b4c2c`) and
# are recoverable from `eb1b4c2c^`. Left as a pointer rather than as a bare TODO
# so the record survives; `test_resources.py` is the worked example to follow.
