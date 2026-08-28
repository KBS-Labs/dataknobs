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

            moves = [t for t in await harness.get_transitions() if t.to_stage == "done"]
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

            moves = [t for t in await harness.get_transitions() if t.to_stage == "done"]
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

            moves = [t for t in await harness.get_transitions() if t.to_stage == "done"]
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


def _duplicate_named_config() -> dict[str, Any]:
    """Two routes to one target, both carrying the *same* author name.

    The shape a copy-pasted transition block produces.  Nothing in the
    name derivation can prevent it -- ``arc_identity`` sees one
    transition at a time -- so the guarantee has to come from the load-
    time check and from the readers refusing to guess.
    """
    return (
        WizardConfigBuilder("dup_names")
        .stage("gather", is_start=True, prompt="Which route?")
        .field("route", field_type="string", required=True)
        .transition("done", "data.get('route') == 'a'", metadata={"name": "retry"})
        .transition("done", "data.get('route') == 'b'", metadata={"name": "retry"})
        .stage("done", is_end=True, prompt="Arrived.")
        .build()
    )


class TestDuplicateAuthoredNameIsAmbiguous:
    """An author-supplied name is unique only if the author made it so.

    The derived form carries the index and cannot collide.  The authored
    form can, and two arcs answering to one string are exactly as
    unidentifiable as the anonymous arcs this naming replaced -- so the
    readers have to treat the collision as the ambiguous case rather
    than as an exact match that happens to be wrong.
    """

    def test_loader_warns_when_two_transitions_compile_to_one_name(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="dataknobs_bots.reasoning.wizard_loader"):
            WizardConfigLoader().load_from_dict(_duplicate_named_config())

        warnings = [r.getMessage() for r in caplog.records if "compile to" in r.getMessage()]
        assert warnings, "duplicate arc name was not reported at load"
        assert "'retry'" in warnings[0]
        assert "transitions 0 and 1" in warnings[0]

    def test_a_unique_authored_name_is_not_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        """The check must not fire on the ordinary case."""
        with caplog.at_level("WARNING", logger="dataknobs_bots.reasoning.wizard_loader"):
            WizardConfigLoader().load_from_dict(_two_routes_config(metadata={"name": "fast_path"}))

        assert not [r for r in caplog.records if "compile to" in r.getMessage()]

    def test_condition_is_not_reported_for_a_duplicated_name(self) -> None:
        """Exactness of the match does not make the answer identifying."""
        wizard_fsm = WizardConfigLoader().load_from_dict(_duplicate_named_config())
        assert wizard_fsm.get_transition_condition("gather", "done", arc_name="retry") is None

    @pytest.mark.asyncio
    async def test_persisted_record_records_nothing_rather_than_the_wrong_arc(self) -> None:
        """The instance that lasts: what a wizard snapshot reports later.

        ``route='b'`` fires the *second* arc.  Before the readers treated
        a duplicated name as ambiguous, this persisted
        ``condition_evaluated="data.get('route') == 'a'"`` -- the first
        sibling's condition -- with ``transition_name='retry'`` beside it
        making the wrong answer look corroborated.
        """
        async with await BotTestHarness.create(
            wizard_config=_duplicate_named_config(),
            main_responses=["Arrived."],
            extraction_results=[[{"route": "b"}]],
        ) as harness:
            await harness.chat("route b please")

            moves = [t for t in await harness.get_transitions() if t.to_stage == "done"]
            assert moves, "no transition to 'done' was recorded"
            assert moves[-1].condition_evaluated is None
            # The name is still recorded: it is what the FSM reported, and
            # it is the evidence that the arc was taken. What is withheld
            # is the claim about *which* declared condition it was.
            assert moves[-1].transition_name == "retry"


class TestNameScanDoesNotCrossTargets:
    """A name identifies an arc among the routes to one target.

    Scanning every transition the stage declares let a name reused on a
    route to somewhere *else* answer for a move it did not cause.
    """

    @staticmethod
    def _config() -> dict[str, Any]:
        return (
            WizardConfigBuilder("cross")
            .stage("gather", is_start=True, prompt="Which route?")
            .field("route", field_type="string", required=True)
            .transition("elsewhere", "data.get('route') == 'x'", metadata={"name": "shared"})
            .transition("done", "data.get('route') == 'b'", metadata={"name": "shared"})
            .stage("elsewhere", is_end=True, prompt="Elsewhere.")
            .stage("done", is_end=True, prompt="Arrived.")
            .build()
        )

    def test_a_name_on_another_target_does_not_answer_here(self) -> None:
        wizard_fsm = WizardConfigLoader().load_from_dict(self._config())
        # 'done' is reached by exactly one transition, so the target scan
        # answers it -- and must answer with *that* transition, not with
        # the identically-named one leading to 'elsewhere'.
        assert (
            wizard_fsm.get_transition_condition("gather", "done", arc_name="shared")
            == "data.get('route') == 'b'"
        )

    def test_two_targets_sharing_a_name_are_not_a_collision(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The load-time check is per stage, and this stage does collide.

        Both transitions leave ``gather`` under one name, so the FSM's
        ``arc_name`` selector cannot tell them apart even though they end
        somewhere different.  The warning is about the name, not the
        destination.
        """
        with caplog.at_level("WARNING", logger="dataknobs_bots.reasoning.wizard_loader"):
            WizardConfigLoader().load_from_dict(self._config())
        assert [r for r in caplog.records if "compile to" in r.getMessage()]


class TestThreeSiblings:
    """Two is the case that motivated this; more than two must also work."""

    @staticmethod
    def _config() -> dict[str, Any]:
        return (
            WizardConfigBuilder("three")
            .stage("gather", is_start=True, prompt="Which route?")
            .field("route", field_type="string", required=True)
            .transition("done", "data.get('route') == 'a'")
            .transition("done", "data.get('route') == 'b'")
            .transition("done", "data.get('route') == 'c'")
            .stage("done", is_end=True, prompt="Arrived.")
            .build()
        )

    def test_each_sibling_reports_its_own_condition(self) -> None:
        wizard_fsm = WizardConfigLoader().load_from_dict(self._config())
        assert [
            wizard_fsm.get_transition_condition("gather", "done", arc_name=f"gather->done#{i}")
            for i in range(3)
        ] == [
            "data.get('route') == 'a'",
            "data.get('route') == 'b'",
            "data.get('route') == 'c'",
        ]

    @pytest.mark.asyncio
    async def test_the_middle_sibling_is_told_from_its_neighbours(self) -> None:
        """The end sibling passes under an off-by-one; the middle does not."""
        async with await BotTestHarness.create(
            wizard_config=self._config(),
            main_responses=["Arrived."],
            extraction_results=[[{"route": "b"}]],
        ) as harness:
            await harness.chat("route b please")

            moves = [t for t in await harness.get_transitions() if t.to_stage == "done"]
            assert moves, "no transition to 'done' was recorded"
            assert moves[-1].transition_name == "gather->done#1"
            assert moves[-1].condition_evaluated == "data.get('route') == 'b'"


class TestArcIdentityEdgeCases:
    """The guards on the authored-name and target reads."""

    def test_a_non_string_name_falls_back_to_the_derived_form(self) -> None:
        """``metadata: {name: 5}`` is not a name.

        Unguarded it reached ``Arc.name`` as an ``int``, so the arc
        answered to a value nothing downstream compares against as a
        string.
        """
        _, name = arc_identity("gather", {"target": "done", "metadata": {"name": 5}}, 0)
        assert name == "gather->done#0"

    def test_an_empty_name_falls_back_to_the_derived_form(self) -> None:
        _, name = arc_identity("gather", {"target": "done", "metadata": {"name": ""}}, 1)
        assert name == "gather->done#1"

    def test_non_mapping_metadata_is_ignored(self) -> None:
        _, name = arc_identity("gather", {"target": "done", "metadata": "oops"}, 2)
        assert name == "gather->done#2"

    def test_an_explicit_null_target_uses_the_stated_default(self) -> None:
        """``or``, not a ``get`` default, which fires only on absence."""
        target, name = arc_identity("gather", {"target": None}, 0)
        assert target == "unknown"
        assert name == "gather->unknown#0"


class TestSubflowSelfLoopEndToEnd:
    """A subflow transition's arc points at its own stage, and is named for it."""

    def test_stage_metadata_records_the_self_loop_target(self) -> None:
        config = {
            "name": "with_subflow",
            "stages": [
                {
                    "name": "gather",
                    "is_start": True,
                    "prompt": "Go?",
                    "schema": {"fields": [{"name": "go", "type": "string"}]},
                    "transitions": [
                        {
                            "target": "_subflow",
                            "condition": "data.get('go')",
                            "subflow": {"network": "sub", "return_stage": "done"},
                        }
                    ],
                },
                {"name": "done", "is_end": True, "prompt": "Done."},
            ],
            "subflows": {
                "sub": {
                    "name": "sub",
                    "stages": [
                        {
                            "name": "inner",
                            "is_start": True,
                            "is_end": True,
                            "prompt": "Inner.",
                        }
                    ],
                }
            },
        }
        wizard_fsm = WizardConfigLoader().load_from_dict(config)

        entry = wizard_fsm.stages["gather"]["transitions"][0]
        # ``target`` stays as authored; ``arc_target`` is where the arc
        # actually points, which is what a move is matched against.
        assert entry["target"] == "_subflow"
        assert entry["arc_target"] == "gather"
        assert entry["name"] == "gather->gather#0"
        assert entry["is_subflow_transition"] is True


class TestUnidentifiableArcIsReportedNotGuessed:
    """Stage metadata built by hand carries no arc names."""

    @staticmethod
    def _unnamed_fsm() -> Any:
        """A wizard FSM whose transition metadata predates arc naming."""
        wizard_fsm = WizardConfigLoader().load_from_dict(_two_routes_config())
        for entry in wizard_fsm.stages["gather"]["transitions"]:
            del entry["name"]
        return wizard_fsm

    def test_an_unmatched_name_falls_back_to_the_target_scan(self) -> None:
        """Two candidates and no usable name: nothing is recorded."""
        assert (
            self._unnamed_fsm().get_transition_condition(
                "gather", "done", arc_name="gather->done#1"
            )
            is None
        )

    def test_the_step_log_says_how_many_arcs_it_could_have_been(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The DEBUG branch for a move that cannot be attributed."""
        wizard_fsm = self._unnamed_fsm()
        with caplog.at_level("DEBUG", logger="dataknobs_bots.reasoning.wizard_fsm"):
            wizard_fsm._log_step_outcome("gather", "done", "gather->done#1")

        lines = [r.getMessage() for r in caplog.records if "via one of" in r.getMessage()]
        assert lines, "the ambiguous-move line was not logged"
        assert "via one of 2 arcs" in lines[0]
        assert "matches none of them by name" in lines[0]


class TestConditionRendering:
    """A declared-but-empty condition is not an unconditional arc."""

    def test_absent_condition_reads_as_unconditional(self) -> None:
        from dataknobs_bots.reasoning.wizard_fsm import _describe_condition

        assert _describe_condition(None) == "unconditional"

    def test_empty_condition_is_distinguished_from_an_absent_one(self) -> None:
        """``condition: ""`` compiles to an arc that can never fire.

        ``or`` reported it as one that always does.
        """
        from dataknobs_bots.reasoning.wizard_fsm import _describe_condition

        assert _describe_condition("") == "empty (never fires)"
        assert _describe_condition("   ") == "empty (never fires)"

    def test_a_real_condition_reads_as_written(self) -> None:
        from dataknobs_bots.reasoning.wizard_fsm import _describe_condition

        assert _describe_condition("data.get('x')") == "data.get('x')"


class TestTransitionRecordToleratesUnknownKeys:
    """Records outlive the build that wrote them, in both directions."""

    def test_a_record_from_a_newer_build_still_deserializes(self) -> None:
        """A field added later must not raise ``TypeError`` here.

        Forward compatibility was already covered by the defaults; this
        is the backward direction -- a downgrade, or a rolling deploy
        where two builds read one store.
        """
        future = {
            "from_stage": "gather",
            "to_stage": "done",
            "timestamp": 1000.0,
            "trigger": "user_input",
            "transition_name": "gather->done#1",
            "some_field_added_later": {"nested": True},
        }
        record = TransitionRecord.from_dict(future)
        assert record.transition_name == "gather->done#1"
        assert not hasattr(record, "some_field_added_later")
