"""The contract every wizard-writing tool is held to.

This seam had no behavioural coverage at all until now, and the defect it
carried was introduced by a change that was correct about its own subject.
So the guard here is driven from **discovery** rather than a list: a KB
tool added later is covered the day it is written, without anyone
remembering to add a row.

Two halves:

* every tool in ``kb_tools`` reports an error when there is no wizard
  state, rather than writing into a dict nobody will read, and
* a shipped tool's writes accumulate across turns, which is the behaviour
  the module docstring has always claimed.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

from dataknobs_llm.tools.context import ToolExecutionContext
from dataknobs_llm.tools.context_aware import ContextAwareTool

from dataknobs_data.backends.memory import SyncMemoryDatabase

from dataknobs_bots.memory.bank import MemoryBank
from dataknobs_bots.reasoning import wizard_types
from dataknobs_bots.reasoning.wizard_types import WizardState
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder
from dataknobs_bots.tools import kb_tools
from dataknobs_bots.tools.bank_tools import AddBankRecordTool
from dataknobs_bots.tools.kb_tools import AddKBResourceTool


def _kb_tool_classes() -> list[type[ContextAwareTool]]:
    """Every context-aware tool this module defines, found not listed."""
    return [
        obj
        for _name, obj in inspect.getmembers(kb_tools, inspect.isclass)
        if issubclass(obj, ContextAwareTool)
        and obj is not ContextAwareTool
        and obj.__module__ == kb_tools.__name__
    ]


def _minimal_arguments(tool: ContextAwareTool) -> dict[str, Any]:
    """Satisfy the tool's declared required parameters, and nothing more.

    Read off the tool's own schema so a new required parameter does not
    silently turn this guard into an assertion about missing arguments.
    """
    schema = tool.schema
    required = schema.get("required", [])
    return dict.fromkeys(required, "placeholder")


def test_discovery_finds_the_kb_tools() -> None:
    """The guard below is worthless if discovery returns nothing."""
    found = {cls.__name__ for cls in _kb_tool_classes()}

    assert found >= {
        "AddKBResourceTool",
        "CheckKnowledgeSourceTool",
        "IngestKnowledgeBaseTool",
        "ListKBResourcesTool",
        "RemoveKBResourceTool",
    }


@pytest.mark.parametrize("tool_cls", _kb_tool_classes(), ids=lambda c: c.__name__)
@pytest.mark.asyncio
async def test_kb_tool_reports_missing_wizard_state(
    tool_cls: type[ContextAwareTool],
) -> None:
    """No wizard state is an error, not a write into a throwaway.

    The accessor these tools used to share returned an empty dict when
    there was no wizard state, so the tool appended to it and reported
    success -- indistinguishable, to the model and to the user, from the
    thing working.
    """
    tool = tool_cls()

    result = await tool.execute_with_context(
        ToolExecutionContext.empty(),
        **_minimal_arguments(tool),
    )

    assert isinstance(result, dict), "an error the model must act on needs a readable shape"
    assert result.get("success") is False
    assert "wizard" in result.get("error", "").lower()


def test_a_context_outside_a_wizard_has_no_wizard_data() -> None:
    """The counterfactual the error result above depends on."""
    assert ToolExecutionContext.empty().wizard_data() is None


def _kb_wizard_config() -> dict[str, Any]:
    """A wizard whose stage adds a KB resource after each extraction."""
    return (
        WizardConfigBuilder("kb-accumulation")
        .stage(
            "collect",
            is_start=True,
            prompt="Which document should I add?",
            tool_result_mapping=[
                {
                    "tool": "add_kb_resource",
                    "params": {"path": "doc_path"},
                    "mapping": {"success": "_added"},
                },
            ],
        )
        .field("doc_path", field_type="string", required=True)
        .transition("done", "has('_finished')")
        .stage("done", is_end=True, prompt="Done.")
        .build()
    )


@pytest.mark.asyncio
async def test_shipped_add_kb_resource_accumulates_across_turns() -> None:
    """A shipped tool's writes survive the turn, and add up.

    This exercises ``AddKBResourceTool`` rather than a probe, which is
    what makes it a test of the item rather than of the harness: the
    module's own docstring has always claimed resources persist across
    tool invocations, and before the strategy published its state they
    did not survive even one save.
    """
    async with await BotTestHarness.create(
        wizard_config=_kb_wizard_config(),
        main_responses=["Adding...", "Added.", "Adding...", "Added.", "Adding...", "Added."],
        extraction_results=[
            [{"doc_path": "guide.md"}],
            [{"doc_path": "faq.md"}],
            [{"doc_path": "policy.md"}],
        ],
        tools=[AddKBResourceTool()],
    ) as harness:
        await harness.chat("add guide.md")
        await harness.chat("add faq.md")
        await harness.chat("add policy.md")

        resources = harness.wizard_data.get("_kb_resources", [])

        assert [r["path"] for r in resources] == ["guide.md", "faq.md", "policy.md"]


def _bank_wizard_config() -> dict[str, Any]:
    """A wizard whose first stage records a bank record via a tool."""
    return (
        WizardConfigBuilder("bank-provenance")
        .stage(
            "collect",
            is_start=True,
            prompt="What should I record?",
            tool_result_mapping=[
                {
                    "tool": "add_bank_record",
                    "params": {"bank_name": "bank_name", "data": "record_data"},
                    "mapping": {"success": "_recorded"},
                },
            ],
        )
        .field("bank_name", field_type="string", required=True)
        .field("record_data", field_type="object", required=True)
        .transition("done", "has('_finished')")
        .stage("done", is_end=True, prompt="Done.")
        .build()
    )


@pytest.mark.asyncio
async def test_bank_record_carries_source_stage_on_the_first_turn() -> None:
    """Provenance is stamped from turn 1, not from turn 2 onwards.

    ``AddBankRecordTool`` reads ``context.wizard_state.current_stage`` to
    stamp ``source_stage``, against a comment saying tool-added records
    carry the same provenance as collection-mode adds. On a flow that does
    not open with a greeting there was no wizard state on the first turn,
    so the first record of every such conversation was stamped ``""``.

    No code in ``bank_tools`` changed for this. It starts working because
    the strategy publishes its state, which is why the assertion lives
    here rather than beside the tool.
    """
    bank = MemoryBank(name="ingredients", schema={}, db=SyncMemoryDatabase())

    async with await BotTestHarness.create(
        wizard_config=_bank_wizard_config(),
        main_responses=["Recording...", "Recorded."],
        extraction_results=[
            [{"bank_name": "ingredients", "record_data": {"name": "flour"}}],
        ],
        tools=[AddBankRecordTool(banks={"ingredients": bank})],
    ) as harness:
        await harness.chat("record flour")

        records = bank.all()
        assert len(records) == 1, f"no record was added; wizard_data={harness.wizard_data}"
        assert records[0].source_stage == "collect"


# ---------------------------------------------------------------------------
# The dict identity the live channel is built on
# ---------------------------------------------------------------------------
#
# A strategy publishes ``WizardState.data`` by reference for the duration
# of a turn, so anything that *rebinds* that attribute mid-turn strands the
# tool holding it. The behavioural half below pins the three flow changes
# that legitimately replace the data; the structural half catches a fourth
# site written later, which is the one nobody would think to add a row for.


def _state_with(data: dict[str, Any]) -> WizardState:
    state = WizardState(current_stage="gather")
    state.data.update(data)
    return state


def test_a_restart_keeps_the_data_dicts_identity() -> None:
    """``restart_cleanup`` empties the collected data without replacing it."""
    state = _state_with({"product_name": "Widget"})
    held = state.data

    state.replace_data({})

    assert state.data is held, "restart rebound the dict a published channel holds"
    assert held == {}


def test_a_subflow_push_keeps_the_data_dicts_identity() -> None:
    """A push swaps in the subflow's data in place."""
    state = _state_with({"product_name": "Widget"})
    held = state.data

    state.replace_data({"detail": "blue"})

    assert state.data is held, "push rebound the dict a published channel holds"
    assert held == {"detail": "blue"}


def test_replacing_data_with_itself_is_not_self_destruction() -> None:
    """Passing ``state.data`` itself must not clear it before copying back.

    The guard exists because the obvious implementation -- clear, then
    update -- destroys the source when the source *is* the target, and a
    caller reaching for ``replace_data(state.data)`` would be doing
    something that reads as a harmless no-op.
    """
    state = _state_with({"product_name": "Widget"})

    state.replace_data(state.data)

    assert state.data == {"product_name": "Widget"}


def test_no_reasoning_module_rebinds_wizard_state_data() -> None:
    """No module assigns to ``.data`` on a wizard state — found, not listed.

    The three known sites now call :meth:`WizardState.replace_data`. This
    reads the reasoning package's source for a *fourth*: an assignment
    like ``state.data = ...`` restores exactly the defect the method was
    written to remove, and it would not fail any behavioural test that
    does not happen to run a tool across it.
    """
    reasoning_dir = Path(wizard_types.__file__).parent
    offenders: list[str] = []

    for path in sorted(reasoning_dir.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "data"
                    and isinstance(target.value, ast.Name)
                    and target.value.id in {"state", "wizard_state"}
                ):
                    offenders.append(f"{path.name}:{node.lineno}")

    assert not offenders, (
        "wizard state data is rebound rather than replaced in place at "
        f"{offenders}; use WizardState.replace_data() so a published live "
        "channel keeps pointing at the dict the wizard is using"
    )


def test_the_rebinding_guard_can_actually_see_an_assignment() -> None:
    """Anti-vacuity: the guard's own pattern matches the shape it forbids."""
    tree = ast.parse("state.data = {}\nwizard_state.data = other\n")
    matched = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "data"
    ]
    assert len(matched) == 2
