"""Tests for WizardFSM introspection properties (enhancement 2g).

Tests the stages, stage_names, and stage_count properties added to
WizardFSM for programmatic stage introspection.
"""

from typing import Any


from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader


class TestStagesProperty:
    """Tests for WizardFSM.stages property."""

    def test_stages_returns_dict(self, wizard_fsm: Any) -> None:
        """Verify stages property returns a dict."""
        stages = wizard_fsm.stages
        assert isinstance(stages, dict)

    def test_stages_contains_all_stages(self, wizard_fsm: Any) -> None:
        """Verify stages contains all defined stages."""
        stages = wizard_fsm.stages
        assert "welcome" in stages
        assert "configure" in stages
        assert "complete" in stages

    def test_stages_contains_metadata(self, wizard_fsm: Any) -> None:
        """Verify each stage contains expected metadata."""
        stages = wizard_fsm.stages
        welcome = stages["welcome"]

        assert welcome.get("prompt") == "What would you like to do?"
        assert welcome.get("is_start") is True
        assert "suggestions" in welcome

    def test_stages_copies_the_table(self, wizard_fsm: Any) -> None:
        """Verify adding or removing a stage does not reach the FSM.

        This is the whole of what the copy buys -- the stages inside it
        are shared, which :meth:`test_stages_does_not_copy_the_stages`
        pins.
        """
        stages1 = wizard_fsm.stages
        stages2 = wizard_fsm.stages

        # Should be equal but not the same object
        assert stages1 == stages2
        assert stages1 is not stages2

        # Modifying returned dict should not affect the internal state
        stages1["new_stage"] = {"prompt": "test"}
        assert "new_stage" not in wizard_fsm.stages

        del stages1["welcome"]
        assert "welcome" in wizard_fsm.stages

    def test_stages_does_not_copy_the_stages(self, wizard_fsm: Any) -> None:
        """Verify writing *through* a returned stage does reach the FSM.

        The boundary of the shallow copy, pinned so that documenting a
        stronger guarantee than this one fails here rather than in a
        consumer's wizard. The copy is deliberate -- see the ``stages``
        docstring for the deepcopy measurement -- so a change making this
        test fail is an intentional contract change, not a bug fix.
        """
        wizard_fsm.stages["welcome"]["label"] = "MUTATED"

        assert wizard_fsm.stage_metadata_for("welcome")["label"] == "MUTATED"
        assert wizard_fsm.stages["welcome"]["label"] == "MUTATED"

    def test_current_metadata_is_the_live_stage(self, wizard_fsm: Any) -> None:
        """Verify the single-stage readers hand out the stage itself.

        ``stages`` at least copies the table; these copy nothing, so the
        same rule reaches them more directly.
        """
        assert wizard_fsm.current_metadata is wizard_fsm.current_metadata
        assert wizard_fsm.stage_metadata_for("welcome") is wizard_fsm.stages["welcome"]

    def test_stages_preserves_transitions(self, wizard_fsm: Any) -> None:
        """Verify stages includes transition definitions."""
        stages = wizard_fsm.stages
        welcome = stages["welcome"]

        transitions = welcome.get("transitions", [])
        assert len(transitions) > 0
        assert transitions[0].get("target") == "configure"


class TestStageNamesProperty:
    """Tests for WizardFSM.stage_names property."""

    def test_stage_names_returns_list(self, wizard_fsm: Any) -> None:
        """Verify stage_names returns a list."""
        names = wizard_fsm.stage_names
        assert isinstance(names, list)

    def test_stage_names_contains_all_stages(self, wizard_fsm: Any) -> None:
        """Verify stage_names contains all stage names."""
        names = wizard_fsm.stage_names
        assert "welcome" in names
        assert "configure" in names
        assert "complete" in names

    def test_stage_names_preserves_order(self) -> None:
        """Verify stage_names preserves definition order."""
        # Create a specific config with known order
        config = {
            "name": "ordered-wizard",
            "version": "1.0",
            "stages": [
                {"name": "first", "is_start": True, "prompt": "First"},
                {"name": "second", "prompt": "Second"},
                {"name": "third", "prompt": "Third"},
                {"name": "fourth", "is_end": True, "prompt": "Fourth"},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)

        names = fsm.stage_names
        assert names == ["first", "second", "third", "fourth"]

    def test_stage_names_returns_new_list(self, wizard_fsm: Any) -> None:
        """Verify stage_names returns a new list each time."""
        names1 = wizard_fsm.stage_names
        names2 = wizard_fsm.stage_names

        assert names1 == names2
        assert names1 is not names2

        # Modifying returned list should not affect internal state
        names1.append("fake_stage")
        assert "fake_stage" not in wizard_fsm.stage_names


class TestStageCountProperty:
    """Tests for WizardFSM.stage_count property."""

    def test_stage_count_returns_int(self, wizard_fsm: Any) -> None:
        """Verify stage_count returns an integer."""
        count = wizard_fsm.stage_count
        assert isinstance(count, int)

    def test_stage_count_matches_stages(self, wizard_fsm: Any) -> None:
        """Verify stage_count matches number of stages."""
        count = wizard_fsm.stage_count
        names = wizard_fsm.stage_names
        stages = wizard_fsm.stages

        assert count == len(names)
        assert count == len(stages)

    def test_stage_count_for_simple_wizard(self, wizard_fsm: Any) -> None:
        """Verify stage_count for the 3-stage simple wizard."""
        assert wizard_fsm.stage_count == 3

    def test_stage_count_for_various_sizes(self) -> None:
        """Verify stage_count works for different wizard sizes."""
        loader = WizardConfigLoader()

        # Single-stage wizard
        single_config = {
            "name": "single",
            "version": "1.0",
            "stages": [{"name": "only", "is_start": True, "is_end": True, "prompt": "Only"}],
        }
        fsm = loader.load_from_dict(single_config)
        assert fsm.stage_count == 1

        # Five-stage wizard
        multi_config = {
            "name": "multi",
            "version": "1.0",
            "stages": [{"name": f"stage_{i}", "prompt": f"Stage {i}"} for i in range(5)],
        }
        multi_config["stages"][0]["is_start"] = True
        multi_config["stages"][-1]["is_end"] = True
        fsm = loader.load_from_dict(multi_config)
        assert fsm.stage_count == 5


class TestIntrospectionConsistency:
    """Tests for consistency between introspection properties."""

    def test_all_properties_consistent(self, wizard_fsm: Any) -> None:
        """Verify all introspection properties are consistent."""
        stages = wizard_fsm.stages
        names = wizard_fsm.stage_names
        count = wizard_fsm.stage_count

        # Count should match
        assert count == len(names)
        assert count == len(stages)

        # Names should be keys of stages
        assert set(names) == set(stages.keys())

    def test_current_stage_in_stage_names(self, wizard_fsm: Any) -> None:
        """Verify current_stage is always in stage_names."""
        current = wizard_fsm.current_stage
        names = wizard_fsm.stage_names

        assert current in names

    def test_current_metadata_matches_stages(self, wizard_fsm: Any) -> None:
        """Verify current_metadata matches stages[current_stage]."""
        current = wizard_fsm.current_stage
        current_meta = wizard_fsm.current_metadata
        stages = wizard_fsm.stages

        assert stages[current] == current_meta


class TestBranchingWizardIntrospection:
    """Tests for introspection with branching wizards."""

    def test_branching_stages(self, wizard_config_with_branches: dict[str, Any]) -> None:
        """Verify introspection works with branching wizards."""
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(wizard_config_with_branches)

        stages = fsm.stages
        names = fsm.stage_names
        count = fsm.stage_count

        assert count == 4
        assert "start" in names
        assert "path_a" in names
        assert "path_b" in names
        assert "default" in names

        # All stages should have metadata
        for name in names:
            assert name in stages
            assert "prompt" in stages[name]
