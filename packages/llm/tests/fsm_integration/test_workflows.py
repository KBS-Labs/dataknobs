"""Tests for FSM workflow patterns.

These tests verify the LLM workflow patterns that integrate with FSM.
Migrated from dataknobs-fsm package.
"""

import pytest

from dataknobs_llm.fsm_integration import (
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
from dataknobs_llm.llm.base import LLMConfig


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


# --------------------------------------------------------------------------- #
# The FSM each workflow type builds
# --------------------------------------------------------------------------- #


def _config(workflow_type: WorkflowType) -> LLMWorkflowConfig:
    return LLMWorkflowConfig(
        workflow_type=workflow_type,
        steps=[LLMStep(name="a", prompt_template="x"), LLMStep(name="b", prompt_template="y")],
        default_model_config=LLMConfig(provider="echo", model="test"),
    )


@pytest.mark.parametrize("workflow_type", list(WorkflowType), ids=lambda w: w.value)
def test_every_workflow_type_builds_an_fsm(workflow_type: WorkflowType) -> None:
    """Constructing a workflow builds its FSM, for every type.

    ``_build_fsm`` runs from ``__init__``, so this is what any use of the class
    does first. It was covered by nothing: the assertions above check that the
    name imports, which is true of a class whose constructor raises.
    """
    workflow = LLMWorkflow(_config(workflow_type))

    assert workflow._fsm.get_states()
    workflow._fsm.close()


@pytest.mark.parametrize(
    "workflow_type",
    [WorkflowType.SIMPLE, WorkflowType.CHAIN, WorkflowType.RAG, WorkflowType.COT],
    ids=lambda w: w.value,
)
def test_a_built_workflow_has_a_state_the_engine_treats_as_final(
    workflow_type: WorkflowType,
) -> None:
    """The end state has to be terminal to the engine, not just named ``end``.

    Every state here carried a ``"type"`` key --- ``initial``, ``task``,
    ``terminal`` --- and ``StateConfig`` has never declared one. The schema
    discarded what it did not declare, so ``{"name": "end", "type":
    "terminal"}`` built an ordinary state and these workflows had no terminal
    state at all. The four types below are the ones with a builder branch; the
    rest fall through to a start and an end.
    """
    workflow = LLMWorkflow(_config(workflow_type))
    fsm = workflow._fsm

    finals = [name for name in fsm.get_states() if fsm.get_state(name).is_end]

    assert finals, (
        "no state is marked final, so the engine has nothing to stop at --- a "
        "terminal state named in a key the schema discards is not a terminal state"
    )
    workflow._fsm.close()


# --------------------------------------------------------------------------- #
# The retriever the RAG workflow retrieves with
# --------------------------------------------------------------------------- #


async def test_the_retriever_indexes_and_retrieves() -> None:
    """Both public methods ran to completion for the first time here.

    Each opened with ``from dataknobs_fsm.llm.providers import get_provider``,
    a module the FSM -> LLM migration removed --- so both raised
    ``ModuleNotFoundError`` on their first line, and ``_execute_rag`` took the
    whole RAG workflow down with them. Behind that import was a branch on
    ``config.provider_config``, a field ``RAGConfig`` has never declared, whose
    ``else`` was the embedding path that actually works. The fallback was
    unreachable behind a guard that could not itself be evaluated.
    """
    docs = ["cats sit on mats", "quantum chromodynamics", "the mats were blue"]
    retriever = VectorRetriever(RAGConfig(retriever_type="vector"))

    await retriever.index_documents(docs)
    hits = await retriever.retrieve("mats", top_k=2)

    assert len(hits) == 2
    assert all(hit in docs for hit in hits)


async def test_retrieval_is_deterministic() -> None:
    """The embeddings are content-derived, so the same query ranks the same way.

    This is the property the class actually has, and asserting it rather than
    relevance is deliberate: the vectors are sha256-derived, so they carry no
    semantic structure and a test that expected ``"mats"`` to find the mats
    would be asserting a capability this retriever does not have.
    """
    docs = ["alpha", "beta", "gamma"]
    retriever = VectorRetriever(RAGConfig(retriever_type="vector"))
    await retriever.index_documents(docs)

    assert await retriever.retrieve("alpha") == await retriever.retrieve("alpha")


async def test_retrieving_before_indexing_returns_nothing() -> None:
    """The empty case returns early, ahead of the import that used to raise."""
    assert await VectorRetriever(RAGConfig(retriever_type="vector")).retrieve("q") == []


async def test_the_embedding_model_field_is_accepted_and_read_by_nothing() -> None:
    """``RAGConfig.embedding_model`` names a model nothing consults.

    It is recorded rather than removed because the field is on a public
    dataclass, and stated here rather than left implicit because a config key
    that parses and does nothing is indistinguishable from one that works ---
    which is the whole reason the branch above was able to sit broken.

    Routing this retriever through the real embedder seam
    (``create_embedding_provider`` / ``LLMProviderEmbedder``) is the fix that
    would give the field a meaning, and it belongs with the consolidation of
    the embedder shapes rather than with a type-annotation pass.
    """
    named = VectorRetriever(RAGConfig(retriever_type="vector", embedding_model="nomic-embed-text"))
    unnamed = VectorRetriever(RAGConfig(retriever_type="vector"))

    await named.index_documents(["one"])
    await unnamed.index_documents(["one"])

    assert named.embeddings == unnamed.embeddings


# Coverage gap, tracked: `workflows.py` is 760 lines, and beyond the FSM each
# workflow type builds --- covered above, because a silently-discarded config
# key had left every one of them without a terminal state --- its execution
# paths remain untested. The tests that covered this module were deleted by the
# FSM -> LLM migration (`eb1b4c2c`) and are recoverable from `eb1b4c2c^`. Left
# as a pointer rather than as a bare TODO so the record survives;
# `test_resources.py` is the worked example to follow.
