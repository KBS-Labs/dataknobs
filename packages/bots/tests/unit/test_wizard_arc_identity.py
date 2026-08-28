"""Two arcs to one target must be told apart, not guessed between.

A wizard stage may offer several routes to the same next stage --
different conditions, different transforms, the same destination.  Until
the compiled arcs carried names, nothing downstream could say which of
them fired: the FSM reported an arc name derived from the endpoints, so
both siblings reported the same string, and every reader fell back to
scanning the stage's declared transitions for the first one whose target
matched.  That answer is right only by accident.

These tests pin the whole chain, from the name the loader derives to the
transition record that persists it.
"""

from typing import Any

import pytest

from dataknobs_bots.reasoning.observability import (
    TransitionRecord,
    execution_record_to_transition_record,
    transition_record_to_execution_record,
)
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader, arc_identity
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder


def _two_routes_config(**transition_kwargs: Any) -> dict[str, Any]:
    """A stage with two transitions declaring the same target.

    ``route`` picks between them: ``"a"`` takes the first, ``"b"`` the
    second.  Both land on ``done``, so the target alone cannot say which
    one carried the wizard there.
    """
    builder = (
        WizardConfigBuilder("two_routes")
        .stage("gather", is_start=True, prompt="Which route?")
        .field("route", field_type="string", required=True)
        .transition("done", "data.get('route') == 'a'", **transition_kwargs)
        .transition("done", "data.get('route') == 'b'")
        .stage("done", is_end=True, prompt="Arrived.")
    )
    return builder.build()


class TestArcIdentity:
    """The single derivation of an arc's target and name."""

    def test_regular_transition_keeps_its_declared_target(self) -> None:
        target, name = arc_identity("gather", {"target": "done"}, 0)
        assert target == "done"
        assert name == "gather->done#0"

    def test_subflow_transition_compiles_to_a_self_loop(self) -> None:
        """``_subflow`` is a sentinel, not a state: the arc stays put."""
        target, name = arc_identity(
            "gather",
            {"target": "_subflow", "subflow": {"network": "sub"}},
            1,
        )
        assert target == "gather"
        assert name == "gather->gather#1"

    def test_index_distinguishes_siblings(self) -> None:
        """Two transitions to one target differ only by position."""
        _, first = arc_identity("gather", {"target": "done"}, 0)
        _, second = arc_identity("gather", {"target": "done"}, 1)
        assert first != second

    def test_authored_name_wins(self) -> None:
        _, name = arc_identity(
            "gather",
            {"target": "done", "metadata": {"name": "fast_path"}},
            2,
        )
        assert name == "fast_path"

    def test_derived_name_extends_the_fsm_default(self) -> None:
        """The prefix is what ``Arc.name`` generates for an unnamed arc.

        A reader who knows the old ``"<source>-><target>"`` form still
        reads it here; only the discriminator is new.
        """
        _, name = arc_identity("gather", {"target": "done"}, 3)
        assert name.startswith("gather->done")


class TestCompiledArcCarriesItsName:
    """The name reaches the FSM, which reports it back on every step."""

    def test_stage_metadata_records_the_arc_name(self) -> None:
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(_two_routes_config())

        transitions = wizard_fsm.stages["gather"]["transitions"]
        assert [t["name"] for t in transitions] == [
            "gather->done#0",
            "gather->done#1",
        ]

    @pytest.mark.asyncio
    async def test_authored_name_reaches_both_the_arc_and_the_metadata(self) -> None:
        """One precedence rule, applied in one place.

        The arc and the metadata describing it are built by separate
        passes over the same config.  They have to agree about what the
        arc is called, or the metadata cannot be looked up by the name
        the step reports -- so this asserts on both ends of that chain:
        the recorded metadata, and the name the FSM hands back after
        actually taking the arc.
        """
        config = _two_routes_config(metadata={"name": "fast_path"})

        assert (
            WizardConfigLoader().load_from_dict(config).stages["gather"]["transitions"][0]["name"]
            == "fast_path"
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Arrived."],
            extraction_results=[[{"route": "a"}]],
        ) as harness:
            await harness.chat("route a please")

            moves = [t for t in harness.transitions if t.to_stage == "done"]
            assert moves, "no transition to 'done' was recorded"
            assert moves[-1].transition_name == "fast_path"


class TestTransitionConditionIsMatchedByName:
    """``get_transition_condition`` reports the arc that fired."""

    @staticmethod
    def _fsm() -> Any:
        return WizardConfigLoader().load_from_dict(_two_routes_config())

    def test_second_arc_reports_its_own_condition(self) -> None:
        wizard_fsm = self._fsm()
        condition = wizard_fsm.get_transition_condition("gather", "done", arc_name="gather->done#1")
        assert condition == "data.get('route') == 'b'"

    def test_first_arc_reports_its_own_condition(self) -> None:
        """Guards against a fix that simply reports the last sibling."""
        wizard_fsm = self._fsm()
        condition = wizard_fsm.get_transition_condition("gather", "done", arc_name="gather->done#0")
        assert condition == "data.get('route') == 'a'"

    def test_ambiguous_call_records_nothing(self) -> None:
        """No arc name and two candidates: the honest answer is None.

        The value is persisted as a transition record's
        ``condition_evaluated``, where a plausible-but-wrong expression
        is worse than an absent one.
        """
        wizard_fsm = self._fsm()
        assert wizard_fsm.get_transition_condition("gather", "done") is None

    def test_single_arc_target_answers_without_a_name(self) -> None:
        """Nothing is ambiguous, so the two-argument call still works."""
        config = (
            WizardConfigBuilder("one_route")
            .stage("gather", is_start=True, prompt="Name?")
            .field("name", field_type="string", required=True)
            .transition("done", "data.get('name')")
            .stage("done", is_end=True, prompt="Done.")
            .build()
        )
        wizard_fsm = WizardConfigLoader().load_from_dict(config)
        assert wizard_fsm.get_transition_condition("gather", "done") == "data.get('name')"


class TestStepLogNamesTheArcThatFired:
    """The DEBUG line the wizard emits after every step."""

    @pytest.mark.asyncio
    async def test_second_arc_is_logged_as_the_second(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async with await BotTestHarness.create(
            wizard_config=_two_routes_config(),
            main_responses=["Arrived."],
            extraction_results=[[{"route": "b"}]],
        ) as harness:
            with caplog.at_level("DEBUG", logger="dataknobs_bots.reasoning.wizard_fsm"):
                await harness.chat("route b please")

        lines = [r.getMessage() for r in caplog.records if "transition:" in r.getMessage()]
        assert lines, "no transition line was logged"
        assert any("data.get('route') == 'b'" in line for line in lines)
        assert not any("data.get('route') == 'a'" in line for line in lines)

    @pytest.mark.asyncio
    async def test_first_arc_is_logged_as_the_first(self, caplog: pytest.LogCaptureFixture) -> None:
        async with await BotTestHarness.create(
            wizard_config=_two_routes_config(),
            main_responses=["Arrived."],
            extraction_results=[[{"route": "a"}]],
        ) as harness:
            with caplog.at_level("DEBUG", logger="dataknobs_bots.reasoning.wizard_fsm"):
                await harness.chat("route a please")

        lines = [r.getMessage() for r in caplog.records if "transition:" in r.getMessage()]
        assert lines, "no transition line was logged"
        assert any("data.get('route') == 'a'" in line for line in lines)
        assert not any("data.get('route') == 'b'" in line for line in lines)


class TestTransitionRecordNamesTheArcThatFired:
    """The persisted half: what a wizard snapshot reports afterwards."""

    @pytest.mark.asyncio
    async def test_condition_evaluated_is_the_arc_that_fired(self) -> None:
        async with await BotTestHarness.create(
            wizard_config=_two_routes_config(),
            main_responses=["Arrived."],
            extraction_results=[[{"route": "b"}]],
        ) as harness:
            await harness.chat("route b please")

            moves = [t for t in harness.transitions if t.to_stage == "done"]
            assert moves, "no transition to 'done' was recorded"
            assert moves[-1].condition_evaluated == "data.get('route') == 'b'"

    @pytest.mark.asyncio
    async def test_transition_name_is_recorded(self) -> None:
        async with await BotTestHarness.create(
            wizard_config=_two_routes_config(),
            main_responses=["Arrived."],
            extraction_results=[[{"route": "b"}]],
        ) as harness:
            await harness.chat("route b please")

            moves = [t for t in harness.transitions if t.to_stage == "done"]
            assert moves, "no transition to 'done' was recorded"
            assert moves[-1].transition_name == "gather->done#1"


class TestTransitionNameSurvivesConversion:
    """The FSM's own record type has always had a field for this."""

    def test_round_trip_preserves_transition_name(self) -> None:
        original = TransitionRecord(
            from_stage="gather",
            to_stage="done",
            timestamp=1000.0,
            trigger="user_input",
            transition_name="gather->done#1",
        )

        restored = execution_record_to_transition_record(
            transition_record_to_execution_record(original)
        )

        assert restored.transition_name == "gather->done#1"

    def test_a_record_without_the_field_restores_as_none(self) -> None:
        """Records persisted before the field existed still deserialize."""
        legacy = {
            "from_stage": "gather",
            "to_stage": "done",
            "timestamp": 1000.0,
            "trigger": "user_input",
        }
        assert TransitionRecord.from_dict(legacy).transition_name is None
