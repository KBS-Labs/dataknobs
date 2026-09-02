"""Behavioural tests for ``ConversationFlowAdapter`` execution.

The adapter had no execution coverage at all: the three sibling files in this
directory exercise ``flow.py`` and ``conditions.py``, and nothing ever called
``execute()``. Every test here fails against the pre-fix adapter — the first
one on a ``ValidationError`` raised before a single state runs.

Real constructs throughout (``EchoProvider``, ``ConfigPromptLibrary``,
``AsyncPromptBuilder``, ``AsyncMemoryDatabase``,
``DataknobsConversationStorage``); no mocks, no fakes.
"""

import pytest

from dataknobs_common.exceptions import OperationError
from dataknobs_common.testing import assert_no_blocking, requires_blockbuster
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations import ConversationManager, DataknobsConversationStorage
from dataknobs_llm.conversations.flow import (
    ConversationFlow,
    FlowState,
    LLMClassifierCondition,
    TransitionCondition,
    always,
    keyword_condition,
)
from dataknobs_llm.conversations.flow.adapter import ConversationFlowAdapter
from dataknobs_llm.llm import EchoProvider, LLMConfig
from dataknobs_llm.prompts import AsyncPromptBuilder
from dataknobs_llm.prompts.implementations.config_library import ConfigPromptLibrary

pytestmark = pytest.mark.asyncio


PROMPTS = {
    "system": {"helpful": {"template": "You are a helpful assistant"}},
    "user": {
        "greet": {"template": "Hello, how can I help?"},
        "farewell": {"template": "Goodbye for now"},
        "loop": {"template": "Still here"},
    },
}


@pytest.fixture
def builder() -> AsyncPromptBuilder:
    """A prompt builder over an in-memory library (no disk I/O)."""
    return AsyncPromptBuilder(library=ConfigPromptLibrary(PROMPTS))


@pytest.fixture
async def llm() -> EchoProvider:
    """An echo provider with no prefix, so responses are predictable."""
    provider = EchoProvider(
        LLMConfig(provider="echo", model="echo-model", options={"echo_prefix": ""})
    )
    yield provider
    await provider.close()


def two_state_flow(condition: TransitionCondition | None = None) -> ConversationFlow:
    """A greeting state that transitions to a terminal farewell state."""
    return ConversationFlow(
        name="two_state",
        initial_state="greeting",
        states={
            "greeting": FlowState(
                prompt_name="greet",
                transitions={"go": "farewell"},
                transition_conditions={"go": condition or keyword_condition(["help"])},
            ),
            "farewell": FlowState(prompt_name="farewell"),
        },
    )


class RaisingCondition(TransitionCondition):
    """A condition whose evaluation *fails* — not one that answers "no"."""

    async def evaluate(self, response: str, context: dict) -> bool:
        raise RuntimeError("classifier backend is down")

    def to_fsm_function(self) -> str:
        return "raising"


async def test_two_state_flow_runs_end_to_end(builder):
    """The whole point: both states run and both responses reach history.

    Fails pre-fix with ``ValidationError`` from the ``FSMConfig`` constructor —
    the config carries a ``functions`` key that has never been a schema field.
    """
    adapter = ConversationFlowAdapter(flow=two_state_flow(), prompt_builder=builder)

    result = await adapter.execute({"topic": "billing"})

    assert [state for state, _ in adapter.execution_state.history] == ["greeting", "farewell"]
    assert adapter.execution_state.history[0][1] == "Hello, how can I help?"
    assert adapter.execution_state.history[1][1] == "Goodbye for now"
    assert result["topic"] == "billing"
    assert result["response"] == "Goodbye for now"


async def test_to_fsm_config_is_accepted_by_the_fsm_schema(builder):
    """``to_fsm_config()`` must produce a config the FSM will actually load.

    The loader is the real validation path: it normalises the flat
    ``states``/``arcs`` shape into networks and then validates. Pre-fix it
    rejected the config outright — ``functions`` has never been a field.
    """
    from dataknobs_fsm.config.loader import ConfigLoader

    adapter = ConversationFlowAdapter(flow=two_state_flow(), prompt_builder=builder)

    config = adapter.to_fsm_config()

    loaded = ConfigLoader().load_from_dict(config)
    assert [state.name for state in loaded.networks[0].states] == ["greeting", "farewell"]


async def test_registered_transform_runs_and_renders_the_state_prompt(builder):
    """The transform reaches the prompt builder and its output reaches data."""
    adapter = ConversationFlowAdapter(flow=two_state_flow(), prompt_builder=builder)

    result = await adapter.execute()

    assert result["history"], "no state transform ever ran"
    assert result["state"] == "farewell"
    assert result["loop_count"] == 1


async def test_condition_selects_its_arc_from_the_response(builder):
    """A condition that matches drives the transition; one that does not, does not."""
    matching = ConversationFlowAdapter(
        flow=two_state_flow(keyword_condition(["help"])), prompt_builder=builder
    )
    await matching.execute()
    assert [s for s, _ in matching.execution_state.history] == ["greeting", "farewell"]

    # A condition that answers "no" leaves the flow with nowhere to go. That
    # is a dead end in a non-final state, which the engine reports as a failed
    # run rather than as a successful one that stopped early.
    non_matching = ConversationFlowAdapter(
        flow=two_state_flow(keyword_condition(["nothing-matches-this"])),
        prompt_builder=builder,
    )
    with pytest.raises(OperationError, match="greeting"):
        await non_matching.execute()
    assert [s for s, _ in non_matching.execution_state.history] == ["greeting"]


async def test_condition_error_surfaces_instead_of_silently_rejecting_the_arc(builder):
    """A *failing* condition is an error, not a "no".

    The engine draws this distinction deliberately — a raising condition must
    surface as a record error rather than de-select the arc, or an
    infrastructure outage is reported as a data-quality drop. The adapter's
    blanket ``except Exception: return False`` converted it back.
    """
    adapter = ConversationFlowAdapter(
        flow=two_state_flow(RaisingCondition()), prompt_builder=builder
    )

    with pytest.raises(OperationError, match="classifier backend is down"):
        await adapter.execute()


async def test_execute_raises_when_the_flow_fails(builder):
    """A failed run must raise, not return the input data as if it succeeded."""
    flow = ConversationFlow(
        name="missing_prompt",
        initial_state="greeting",
        states={
            "greeting": FlowState(
                prompt_name="greet",
                transitions={"go": "farewell"},
                transition_conditions={"go": keyword_condition(["help"])},
            ),
            "farewell": FlowState(prompt_name="no-such-prompt"),
        },
    )
    adapter = ConversationFlowAdapter(flow=flow, prompt_builder=builder)

    with pytest.raises(OperationError, match="farewell"):
        await adapter.execute({"topic": "billing"})


async def test_llm_classifier_condition_finds_the_adapter_provider(builder, llm):
    """``llm=`` must reach the condition, which reads ``_llm_provider``."""
    flow = two_state_flow(LLMClassifierCondition(classifier_prompt="yes", expected_value="yes"))
    adapter = ConversationFlowAdapter(flow=flow, prompt_builder=builder, llm=llm)

    await adapter.execute()

    assert [s for s, _ in adapter.execution_state.history] == ["greeting", "farewell"]


async def test_execution_summary_reports_the_state_the_flow_ended_in(builder):
    """``current_state`` tracks execution; it does not name the start forever."""
    adapter = ConversationFlowAdapter(flow=two_state_flow(), prompt_builder=builder)

    await adapter.execute()

    summary = adapter.get_execution_summary()
    assert summary["current_state"] == "farewell"
    assert summary["history_length"] == 2
    assert sorted(summary["states_visited"]) == ["farewell", "greeting"]


async def test_max_loops_stops_a_self_looping_state(builder):
    """A state that loops onto itself stops at ``max_loops`` and says why."""
    flow = ConversationFlow(
        name="looping",
        initial_state="loop",
        states={
            "loop": FlowState(
                prompt_name="loop",
                max_loops=2,
                transitions={"again": "loop"},
                transition_conditions={"again": always()},
            )
        },
    )
    adapter = ConversationFlowAdapter(flow=flow, prompt_builder=builder)

    with pytest.raises(OperationError, match="Max loops exceeded for state loop"):
        await adapter.execute()

    assert adapter.execution_state.loop_counts["loop"] == 3
    assert adapter.get_execution_summary()["stop_reason"] == "Max loops exceeded for state loop"


@requires_blockbuster
async def test_execute_does_not_block_the_event_loop(builder):
    """The adapter must drive the *async* FSM facade.

    ``SimpleFSM`` is the sync facade: its ``_run_async`` bridges to a daemon
    loop and blocks the calling thread for the whole FSM run, which on a
    shared loop freezes every other in-flight task.
    """
    adapter = ConversationFlowAdapter(flow=two_state_flow(), prompt_builder=builder)

    with assert_no_blocking():
        await adapter.execute()


async def test_execute_flow_yields_one_node_per_state(builder, llm):
    """The public surface: ``execute_flow`` yields a node per executed state.

    This is the test whose absence is the whole story — the only one that
    exercises what a consumer actually calls.
    """
    manager = await ConversationManager.create(
        llm=llm,
        prompt_builder=builder,
        storage=DataknobsConversationStorage(AsyncMemoryDatabase()),
        system_prompt_name="helpful",
    )

    nodes = [node async for node in manager.execute_flow(two_state_flow())]

    assert [node.metadata["state"] for node in nodes] == ["greeting", "farewell"]
    assert [node.message.content for node in nodes] == [
        "Hello, how can I help?",
        "Goodbye for now",
    ]
    assert all(node.message.role == "assistant" for node in nodes)
    assert [node.prompt_name for node in nodes] == ["greet", "farewell"]
