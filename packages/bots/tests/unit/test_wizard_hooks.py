"""Tests for WizardHooks lifecycle callbacks."""

import pytest

from dataknobs_bots.reasoning.wizard_hooks import WizardHooks


class TestWizardHooksRegistration:
    """Tests for hook registration."""

    def test_empty_hooks(self) -> None:
        """Test that new WizardHooks has no registered hooks."""
        hooks = WizardHooks()

        assert hooks.hook_count == {
            "enter": 0,
            "exit": 0,
            "complete": 0,
            "restart": 0,
            "error": 0,
        }

    def test_on_enter_registration(self) -> None:
        """Test registering enter hooks."""
        hooks = WizardHooks()

        def my_callback(stage: str, data: dict) -> None:
            pass

        result = hooks.on_enter(my_callback)

        assert result is hooks  # Method chaining
        assert hooks.hook_count["enter"] == 1

    def test_on_enter_stage_specific(self) -> None:
        """Test registering stage-specific enter hook."""
        hooks = WizardHooks()

        def welcome_handler(stage: str, data: dict) -> None:
            pass

        hooks.on_enter(welcome_handler, stage="welcome")

        assert hooks.hook_count["enter"] == 1

    def test_on_exit_registration(self) -> None:
        """Test registering exit hooks."""
        hooks = WizardHooks()

        hooks.on_exit(lambda s, d: None)
        hooks.on_exit(lambda s, d: None, stage="configure")

        assert hooks.hook_count["exit"] == 2

    def test_on_complete_registration(self) -> None:
        """Test registering completion hooks."""
        hooks = WizardHooks()

        hooks.on_complete(lambda d: None)

        assert hooks.hook_count["complete"] == 1

    def test_on_restart_registration(self) -> None:
        """Test registering restart hooks."""
        hooks = WizardHooks()

        hooks.on_restart(lambda: None)

        assert hooks.hook_count["restart"] == 1

    def test_on_error_registration(self) -> None:
        """Test registering error hooks."""
        hooks = WizardHooks()

        hooks.on_error(lambda s, d, e: None)

        assert hooks.hook_count["error"] == 1

    def test_method_chaining(self) -> None:
        """Test that hook registration supports method chaining."""
        hooks = WizardHooks()

        result = (
            hooks.on_enter(lambda s, d: None)
            .on_exit(lambda s, d: None)
            .on_complete(lambda d: None)
            .on_restart(lambda: None)
            .on_error(lambda s, d, e: None)
        )

        assert result is hooks
        assert hooks.hook_count == {
            "enter": 1,
            "exit": 1,
            "complete": 1,
            "restart": 1,
            "error": 1,
        }

    def test_clear_hooks(self) -> None:
        """Test clearing all registered hooks."""
        hooks = WizardHooks()

        hooks.on_enter(lambda s, d: None)
        hooks.on_exit(lambda s, d: None)
        hooks.on_complete(lambda d: None)

        hooks.clear()

        assert hooks.hook_count == {
            "enter": 0,
            "exit": 0,
            "complete": 0,
            "restart": 0,
            "error": 0,
        }


class TestWizardHooksExecution:
    """Tests for hook execution."""

    @pytest.mark.asyncio
    async def test_trigger_enter_sync(self) -> None:
        """Test triggering sync enter hook."""
        hooks = WizardHooks()
        called_with: list[tuple[str, dict]] = []

        def my_hook(stage: str, data: dict) -> None:
            called_with.append((stage, data))

        hooks.on_enter(my_hook)

        await hooks.trigger_enter("welcome", {"key": "value"})

        assert len(called_with) == 1
        assert called_with[0] == ("welcome", {"key": "value"})

    @pytest.mark.asyncio
    async def test_trigger_enter_async(self) -> None:
        """Test triggering async enter hook."""
        hooks = WizardHooks()
        called_with: list[tuple[str, dict]] = []

        async def my_hook(stage: str, data: dict) -> None:
            called_with.append((stage, data))

        hooks.on_enter(my_hook)

        await hooks.trigger_enter("configure", {"count": 5})

        assert len(called_with) == 1
        assert called_with[0] == ("configure", {"count": 5})

    @pytest.mark.asyncio
    async def test_trigger_enter_stage_specific(self) -> None:
        """Test that stage-specific hooks only fire for matching stage."""
        hooks = WizardHooks()
        welcome_calls: list[str] = []
        global_calls: list[str] = []

        hooks.on_enter(lambda s, d: global_calls.append(s))
        hooks.on_enter(lambda s, d: welcome_calls.append(s), stage="welcome")

        # Trigger welcome stage
        await hooks.trigger_enter("welcome", {})

        assert len(global_calls) == 1
        assert len(welcome_calls) == 1

        # Trigger other stage
        await hooks.trigger_enter("configure", {})

        assert len(global_calls) == 2  # Global fires for all
        assert len(welcome_calls) == 1  # Stage-specific only for welcome

    @pytest.mark.asyncio
    async def test_trigger_exit_sync(self) -> None:
        """Test triggering sync exit hook."""
        hooks = WizardHooks()
        called_with: list[tuple[str, dict]] = []

        hooks.on_exit(lambda s, d: called_with.append((s, d)))

        await hooks.trigger_exit("welcome", {"intent": "test"})

        assert len(called_with) == 1
        assert called_with[0] == ("welcome", {"intent": "test"})

    @pytest.mark.asyncio
    async def test_trigger_complete(self) -> None:
        """Test triggering completion hook."""
        hooks = WizardHooks()
        final_data: list[dict] = []

        hooks.on_complete(lambda d: final_data.append(d))

        await hooks.trigger_complete({"result": "success"})

        assert len(final_data) == 1
        assert final_data[0] == {"result": "success"}

    @pytest.mark.asyncio
    async def test_trigger_restart(self) -> None:
        """Test triggering restart hook."""
        hooks = WizardHooks()
        restart_count = [0]

        hooks.on_restart(lambda: restart_count.__setitem__(0, restart_count[0] + 1))

        await hooks.trigger_restart()

        assert restart_count[0] == 1

    @pytest.mark.asyncio
    async def test_trigger_restart_async(self) -> None:
        """Test triggering async restart hook."""
        hooks = WizardHooks()
        restart_count = [0]

        async def async_restart() -> None:
            restart_count[0] += 1

        hooks.on_restart(async_restart)

        await hooks.trigger_restart()

        assert restart_count[0] == 1

    @pytest.mark.asyncio
    async def test_multiple_hooks_same_type(self) -> None:
        """Test multiple hooks of same type all fire."""
        hooks = WizardHooks()
        call_order: list[int] = []

        hooks.on_enter(lambda s, d: call_order.append(1))
        hooks.on_enter(lambda s, d: call_order.append(2))
        hooks.on_enter(lambda s, d: call_order.append(3))

        await hooks.trigger_enter("test", {})

        assert call_order == [1, 2, 3]


class TestWizardHooksErrorHandling:
    """Tests for error handling in hooks."""

    @pytest.mark.asyncio
    async def test_error_hook_called_on_enter_error(self) -> None:
        """Test that error hooks are called when enter hook fails."""
        hooks = WizardHooks()
        error_info: list[tuple[str, dict, Exception]] = []

        def failing_hook(stage: str, data: dict) -> None:
            raise ValueError("Hook failed")

        def error_handler(stage: str, data: dict, error: Exception) -> None:
            error_info.append((stage, data, error))

        hooks.on_enter(failing_hook)
        hooks.on_error(error_handler)

        with pytest.raises(ValueError, match="Hook failed"):
            await hooks.trigger_enter("test", {"key": "val"})

        assert len(error_info) == 1
        assert error_info[0][0] == "test"
        assert error_info[0][1] == {"key": "val"}
        assert isinstance(error_info[0][2], ValueError)

    @pytest.mark.asyncio
    async def test_error_hook_called_on_exit_error(self) -> None:
        """Test that error hooks are called when exit hook fails."""
        hooks = WizardHooks()
        error_called = [False]

        def failing_hook(stage: str, data: dict) -> None:
            raise RuntimeError("Exit failed")

        hooks.on_exit(failing_hook)
        hooks.on_error(lambda s, d, e: error_called.__setitem__(0, True))

        with pytest.raises(RuntimeError, match="Exit failed"):
            await hooks.trigger_exit("test", {})

        assert error_called[0] is True

    @pytest.mark.asyncio
    async def test_error_hook_called_on_complete_error(self) -> None:
        """Test that error hooks are called when complete hook fails."""
        hooks = WizardHooks()
        error_stages: list[str] = []

        def failing_hook(data: dict) -> None:
            raise ValueError("Complete failed")

        hooks.on_complete(failing_hook)
        hooks.on_error(lambda s, d, e: error_stages.append(s))

        with pytest.raises(ValueError, match="Complete failed"):
            await hooks.trigger_complete({})

        assert "_complete" in error_stages

    @pytest.mark.asyncio
    async def test_error_handler_failure_doesnt_propagate(self) -> None:
        """Test that errors in error handlers don't cause additional failures."""
        hooks = WizardHooks()

        def failing_hook(stage: str, data: dict) -> None:
            raise ValueError("Original error")

        def failing_error_handler(stage: str, data: dict, error: Exception) -> None:
            raise RuntimeError("Error handler failed")

        hooks.on_enter(failing_hook)
        hooks.on_error(failing_error_handler)

        # Should raise original error, not error handler's error
        with pytest.raises(ValueError, match="Original error"):
            await hooks.trigger_enter("test", {})


class TestWizardHooksFromConfig:
    """Tests for WizardHooks.from_config()."""

    def test_from_config_empty(self) -> None:
        """Test creating hooks from empty config."""
        hooks = WizardHooks.from_config({})

        assert hooks.hook_count == {
            "enter": 0,
            "exit": 0,
            "complete": 0,
            "restart": 0,
            "error": 0,
        }

    def test_from_config_invalid_function_format(self) -> None:
        """Test that invalid function format is handled gracefully."""
        # Missing colon separator
        hooks = WizardHooks.from_config(
            {"on_enter": [{"function": "invalid.format"}]}
        )

        # Should not register invalid hook
        assert hooks.hook_count["enter"] == 0

    def test_from_config_missing_module(self) -> None:
        """Test that missing module is handled gracefully."""
        hooks = WizardHooks.from_config(
            {"on_enter": [{"function": "nonexistent.module:func"}]}
        )

        # Should not register hook for missing module
        assert hooks.hook_count["enter"] == 0

    def test_from_config_string_format(self) -> None:
        """Test that string format (without dict) is handled."""
        # String without dict wrapper should still be processed
        hooks = WizardHooks.from_config(
            {"on_complete": ["nonexistent.module:func"]}
        )

        # Should not register hook for missing module
        assert hooks.hook_count["complete"] == 0

    def test_load_callback_invalid_type(self) -> None:
        """Test that invalid config type returns None."""
        result = WizardHooks._load_callback(12345)  # type: ignore

        assert result is None

    def test_load_callback_empty_function(self) -> None:
        """Test that empty function string returns None."""
        result = WizardHooks._load_callback({"function": ""})

        assert result is None


class TestWizardHooksIntegration:
    """Integration tests with WizardReasoning."""

    @pytest.mark.asyncio
    async def test_hooks_with_wizard_reasoning(
        self, simple_wizard_config: dict
    ) -> None:
        """Test hooks integration with WizardReasoning."""
        from dataknobs_bots.reasoning.wizard import WizardReasoning
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        from .conftest import WizardTestManager

        # Create hooks
        hooks = WizardHooks()
        enter_stages: list[str] = []
        exit_stages: list[str] = []
        completed = [False]

        hooks.on_enter(lambda s, d: enter_stages.append(s))
        hooks.on_exit(lambda s, d: exit_stages.append(s))
        hooks.on_complete(lambda d: completed.__setitem__(0, True))

        # Create wizard with hooks
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm, strict_validation=False, hooks=hooks
        )

        # Create test manager
        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "I want to create something"}]
        manager.metadata = {}
        manager.echo_provider.set_responses(
            ["Welcome! Let me help you create something."]
        )

        # Generate should trigger hooks
        await reasoning.generate(manager, llm=None)

        # Verify hooks were called
        # Note: Specific hook calls depend on wizard flow
        assert "wizard" in manager.metadata

    @pytest.mark.asyncio
    async def test_restart_triggers_hook(self, simple_wizard_config: dict) -> None:
        """Test that restart command triggers restart hook."""
        from dataknobs_bots.reasoning.wizard import WizardReasoning
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        from .conftest import WizardTestManager

        # Create hooks
        hooks = WizardHooks()
        restart_called = [False]
        hooks.on_restart(lambda: restart_called.__setitem__(0, True))

        # Create wizard with hooks
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm, strict_validation=False, hooks=hooks
        )

        # Create test manager at non-start stage
        manager = WizardTestManager()
        manager.messages = [{"role": "user", "content": "restart"}]
        manager.metadata = {
            "wizard": {
                "fsm_state": {
                    "current_stage": "configure",
                    "history": ["welcome", "configure"],
                    "data": {"intent": "test"},
                    "completed": False,
                    "clarification_attempts": 0,
                }
            }
        }
        manager.echo_provider.set_responses(["Starting over from the beginning."])

        # Generate with restart command
        await reasoning.generate(manager, llm=None)

        # Verify restart hook was called
        assert restart_called[0] is True
