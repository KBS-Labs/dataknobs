"""Tests for per-request LLM config overrides functionality."""

import pytest
from dataknobs_llm.llm.base import LLMConfig, LLMMessage
from dataknobs_llm.llm.providers.echo import EchoProvider


class TestConfigOverridesValidation:
    """Tests for config override validation."""

    @pytest.fixture
    def echo_provider(self):
        """Create an EchoProvider for testing."""
        config = LLMConfig(
            provider="echo",
            model="test-model",
            temperature=0.7,
            max_tokens=100,
        )
        return EchoProvider(config)

    def test_validate_allowed_overrides(self, echo_provider):
        """Test that allowed override fields pass validation."""
        # All allowed fields should not raise
        echo_provider._validate_config_overrides({
            "model": "new-model",
            "temperature": 0.9,
            "max_tokens": 200,
            "top_p": 0.95,
            "stop_sequences": ["END"],
            "seed": 42,
        })

    def test_validate_empty_overrides(self, echo_provider):
        """Test that empty/None overrides pass validation."""
        echo_provider._validate_config_overrides(None)
        echo_provider._validate_config_overrides({})

    def test_validate_invalid_overrides_raises(self, echo_provider):
        """Test that unsupported override fields raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            echo_provider._validate_config_overrides({
                "model": "new-model",
                "invalid_field": "value",
            })
        assert "Unsupported config overrides" in str(exc_info.value)
        assert "invalid_field" in str(exc_info.value)

    def test_validate_multiple_invalid_overrides(self, echo_provider):
        """Test that multiple unsupported fields are all reported."""
        with pytest.raises(ValueError) as exc_info:
            echo_provider._validate_config_overrides({
                "bad_field1": "value1",
                "bad_field2": "value2",
            })
        error_msg = str(exc_info.value)
        assert "bad_field1" in error_msg or "bad_field2" in error_msg


class TestGetRuntimeConfig:
    """Tests for _get_runtime_config method."""

    @pytest.fixture
    def echo_provider(self):
        """Create an EchoProvider for testing."""
        config = LLMConfig(
            provider="echo",
            model="original-model",
            temperature=0.5,
            max_tokens=100,
        )
        return EchoProvider(config)

    def test_no_overrides_returns_original_config(self, echo_provider):
        """Test that None overrides returns the original config."""
        runtime_config = echo_provider._get_runtime_config(None)
        assert runtime_config is echo_provider.config

    def test_empty_overrides_returns_original_config(self, echo_provider):
        """Test that empty dict returns original config."""
        runtime_config = echo_provider._get_runtime_config({})
        assert runtime_config is echo_provider.config

    def test_overrides_creates_new_config(self, echo_provider):
        """Test that overrides create a cloned config with new values."""
        runtime_config = echo_provider._get_runtime_config({
            "model": "overridden-model",
            "temperature": 0.9,
        })

        # Runtime config should have overridden values
        assert runtime_config.model == "overridden-model"
        assert runtime_config.temperature == 0.9

        # Original config should be unchanged
        assert echo_provider.config.model == "original-model"
        assert echo_provider.config.temperature == 0.5

        # Non-overridden values should be preserved
        assert runtime_config.max_tokens == 100

    def test_overrides_preserves_original_config(self, echo_provider):
        """Test that original config is never modified."""
        original_model = echo_provider.config.model
        original_temp = echo_provider.config.temperature

        echo_provider._get_runtime_config({
            "model": "different-model",
            "temperature": 1.0,
        })

        # Verify original config unchanged
        assert echo_provider.config.model == original_model
        assert echo_provider.config.temperature == original_temp


class TestProviderCompleteWithOverrides:
    """Tests for provider complete() with config_overrides."""

    @pytest.fixture
    def echo_provider(self):
        """Create an EchoProvider for testing."""
        config = LLMConfig(
            provider="echo",
            model="original-model",
            temperature=0.5,
        )
        return EchoProvider(config)

    @pytest.mark.asyncio
    async def test_complete_without_overrides(self, echo_provider):
        """Test complete() works without overrides."""
        response = await echo_provider.complete("Hello")
        assert response.model == "original-model"

    @pytest.mark.asyncio
    async def test_complete_with_model_override(self, echo_provider):
        """Test complete() applies model override."""
        response = await echo_provider.complete(
            "Hello",
            config_overrides={"model": "overridden-model"}
        )
        assert response.model == "overridden-model"

        # Original config should be unchanged for next call
        response2 = await echo_provider.complete("Hello again")
        assert response2.model == "original-model"

    @pytest.mark.asyncio
    async def test_complete_with_multiple_overrides(self, echo_provider):
        """Test complete() applies multiple overrides."""
        response = await echo_provider.complete(
            "Hello",
            config_overrides={
                "model": "new-model",
                "temperature": 0.9,
                "max_tokens": 500,
            }
        )
        assert response.model == "new-model"

    @pytest.mark.asyncio
    async def test_complete_invalid_override_raises(self, echo_provider):
        """Test complete() raises on invalid override."""
        with pytest.raises(ValueError) as exc_info:
            await echo_provider.complete(
                "Hello",
                config_overrides={"invalid_param": "value"}
            )
        assert "Unsupported config overrides" in str(exc_info.value)


class TestProviderStreamCompleteWithOverrides:
    """Tests for provider stream_complete() with config_overrides."""

    @pytest.fixture
    def echo_provider(self):
        """Create an EchoProvider for testing."""
        config = LLMConfig(
            provider="echo",
            model="original-model",
            temperature=0.5,
        )
        return EchoProvider(config)

    @pytest.mark.asyncio
    async def test_stream_complete_without_overrides(self, echo_provider):
        """Test stream_complete() works without overrides."""
        chunks = []
        async for chunk in echo_provider.stream_complete("Hello"):
            chunks.append(chunk)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_stream_complete_with_overrides(self, echo_provider):
        """Test stream_complete() applies overrides."""
        chunks = []
        async for chunk in echo_provider.stream_complete(
            "Hello",
            config_overrides={"model": "overridden-model", "temperature": 0.9}
        ):
            chunks.append(chunk)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_stream_complete_invalid_override_raises(self, echo_provider):
        """Test stream_complete() raises on invalid override."""
        with pytest.raises(ValueError) as exc_info:
            async for _ in echo_provider.stream_complete(
                "Hello",
                config_overrides={"bad_field": "value"}
            ):
                pass
        assert "Unsupported config overrides" in str(exc_info.value)


class TestConversationManagerWithOverrides:
    """Tests for ConversationManager with llm_config_overrides."""

    @pytest.fixture
    def echo_provider(self):
        """Create an EchoProvider for testing."""
        config = LLMConfig(
            provider="echo",
            model="original-model",
            temperature=0.5,
        )
        return EchoProvider(config)

    @pytest.fixture
    async def manager(self, echo_provider, tmp_path):
        """Create a ConversationManager with EchoProvider."""
        import yaml
        from dataknobs_llm.conversations import (
            ConversationManager,
            DataknobsConversationStorage,
        )
        from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        # Create minimal prompt files
        prompt_dir = tmp_path / "prompts"
        system_dir = prompt_dir / "system"
        system_dir.mkdir(parents=True)
        (system_dir / "test.yaml").write_text(
            yaml.dump({"template": "You are a helpful assistant"})
        )

        # Create prompt library and builder
        library = FileSystemPromptLibrary(prompt_dir)
        builder = AsyncPromptBuilder(library=library)

        storage = DataknobsConversationStorage(AsyncMemoryDatabase())
        manager = await ConversationManager.create(
            llm=echo_provider,
            storage=storage,
            prompt_builder=builder,
        )
        await manager.add_message(role="system", content="You are a helpful assistant.")
        await manager.add_message(role="user", content="Hello!")
        return manager

    @pytest.mark.asyncio
    async def test_complete_without_overrides(self, manager):
        """Test ConversationManager.complete() without overrides."""
        response = await manager.complete()
        assert response.model == "original-model"
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_complete_with_overrides(self, manager):
        """Test ConversationManager.complete() with overrides."""
        response = await manager.complete(
            llm_config_overrides={"model": "overridden-model", "temperature": 0.9}
        )
        assert response.model == "overridden-model"

    @pytest.mark.asyncio
    async def test_complete_tracks_overrides_in_metadata(self, manager):
        """Test that applied overrides are tracked in node metadata."""
        overrides = {"model": "tracked-model", "temperature": 0.8}
        await manager.complete(llm_config_overrides=overrides)

        # Get current node and check metadata
        current_node = manager.state.get_current_node()
        assert current_node is not None
        metadata = current_node.data.metadata
        assert "config_overrides_applied" in metadata
        assert metadata["config_overrides_applied"] == overrides

    @pytest.mark.asyncio
    async def test_complete_no_overrides_no_tracking(self, manager):
        """Test that no tracking when no overrides provided."""
        await manager.complete()

        current_node = manager.state.get_current_node()
        assert current_node is not None
        metadata = current_node.data.metadata
        assert "config_overrides_applied" not in metadata

    @pytest.mark.asyncio
    async def test_stream_complete_with_overrides(self, manager):
        """Test ConversationManager.stream_complete() with overrides."""
        chunks = []
        async for chunk in manager.stream_complete(
            llm_config_overrides={"model": "streamed-model", "temperature": 0.9}
        ):
            chunks.append(chunk)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_stream_complete_tracks_overrides_in_metadata(self, manager):
        """Test that stream_complete tracks overrides in metadata."""
        overrides = {"model": "stream-tracked-model"}
        async for _ in manager.stream_complete(llm_config_overrides=overrides):
            pass

        current_node = manager.state.get_current_node()
        assert current_node is not None
        metadata = current_node.data.metadata
        assert "config_overrides_applied" in metadata
        assert metadata["config_overrides_applied"] == overrides

    @pytest.mark.asyncio
    async def test_overrides_dont_persist_between_calls(self, manager):
        """Test that overrides from one call don't affect subsequent calls."""
        # First call with override
        response1 = await manager.complete(
            llm_config_overrides={"model": "temp-model"}
        )
        assert response1.model == "temp-model"

        # Add another user message
        await manager.add_message("user", "Another question")

        # Second call without override should use original
        response2 = await manager.complete()
        assert response2.model == "original-model"


class TestAllowedOverrideFields:
    """Tests to verify all allowed override fields work correctly."""

    @pytest.fixture
    def echo_provider(self):
        """Create an EchoProvider for testing."""
        config = LLMConfig(
            provider="echo",
            model="test-model",
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            seed=None,
            stop_sequences=None,
        )
        return EchoProvider(config)

    @pytest.mark.asyncio
    async def test_model_override(self, echo_provider):
        """Test model override."""
        response = await echo_provider.complete(
            "test", config_overrides={"model": "new-model"}
        )
        assert response.model == "new-model"

    @pytest.mark.asyncio
    async def test_temperature_override(self, echo_provider):
        """Test temperature override (verify config applied)."""
        runtime = echo_provider._get_runtime_config({"temperature": 0.1})
        assert runtime.temperature == 0.1
        assert echo_provider.config.temperature == 0.5

    @pytest.mark.asyncio
    async def test_max_tokens_override(self, echo_provider):
        """Test max_tokens override."""
        runtime = echo_provider._get_runtime_config({"max_tokens": 500})
        assert runtime.max_tokens == 500
        assert echo_provider.config.max_tokens == 100

    @pytest.mark.asyncio
    async def test_top_p_override(self, echo_provider):
        """Test top_p override."""
        runtime = echo_provider._get_runtime_config({"top_p": 0.5})
        assert runtime.top_p == 0.5
        assert echo_provider.config.top_p == 0.9

    @pytest.mark.asyncio
    async def test_seed_override(self, echo_provider):
        """Test seed override."""
        runtime = echo_provider._get_runtime_config({"seed": 42})
        assert runtime.seed == 42
        assert echo_provider.config.seed is None

    @pytest.mark.asyncio
    async def test_stop_sequences_override(self, echo_provider):
        """Test stop_sequences override."""
        runtime = echo_provider._get_runtime_config({"stop_sequences": ["END", "STOP"]})
        assert runtime.stop_sequences == ["END", "STOP"]
        assert echo_provider.config.stop_sequences is None


class TestExtendedOverrideFields:
    """Tests for extended override fields (presence_penalty, frequency_penalty, etc.)."""

    @pytest.fixture
    def echo_provider(self):
        """Create an EchoProvider for testing."""
        config = LLMConfig(
            provider="echo",
            model="test-model",
            temperature=0.5,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )
        return EchoProvider(config)

    def test_presence_penalty_override(self, echo_provider):
        """Test presence_penalty override."""
        runtime = echo_provider._get_runtime_config({"presence_penalty": 0.5})
        assert runtime.presence_penalty == 0.5
        assert echo_provider.config.presence_penalty == 0.0

    def test_frequency_penalty_override(self, echo_provider):
        """Test frequency_penalty override."""
        runtime = echo_provider._get_runtime_config({"frequency_penalty": 0.3})
        assert runtime.frequency_penalty == 0.3
        assert echo_provider.config.frequency_penalty == 0.0

    def test_logit_bias_override(self, echo_provider):
        """Test logit_bias override."""
        runtime = echo_provider._get_runtime_config({"logit_bias": {"50256": -100}})
        assert runtime.logit_bias == {"50256": -100}
        assert echo_provider.config.logit_bias is None

    def test_response_format_override(self, echo_provider):
        """Test response_format override."""
        runtime = echo_provider._get_runtime_config({"response_format": "json"})
        assert runtime.response_format == "json"
        assert echo_provider.config.response_format is None

    def test_functions_override(self, echo_provider):
        """Test functions override for dynamic function definitions."""
        functions = [
            {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object", "properties": {}}
            }
        ]
        runtime = echo_provider._get_runtime_config({"functions": functions})
        assert runtime.functions == functions
        assert echo_provider.config.functions is None

    def test_function_call_override(self, echo_provider):
        """Test function_call override."""
        runtime = echo_provider._get_runtime_config({"function_call": "auto"})
        assert runtime.function_call == "auto"
        assert echo_provider.config.function_call is None


class TestOptionsOverride:
    """Tests for provider-specific options dict override."""

    @pytest.fixture
    def echo_provider_with_options(self):
        """Create an EchoProvider with options."""
        config = LLMConfig(
            provider="echo",
            model="test-model",
            options={"echo_prefix": "Base: ", "custom_key": "original"},
        )
        return EchoProvider(config)

    @pytest.fixture
    def echo_provider_no_options(self):
        """Create an EchoProvider without options."""
        config = LLMConfig(
            provider="echo",
            model="test-model",
        )
        return EchoProvider(config)

    def test_options_override_merges_with_existing(self, echo_provider_with_options):
        """Test that options override merges with existing options."""
        runtime = echo_provider_with_options._get_runtime_config({
            "options": {"new_key": "new_value", "custom_key": "overridden"}
        })
        # Original keys preserved
        assert runtime.options.get("echo_prefix") == "Base: "
        # New key added
        assert runtime.options.get("new_key") == "new_value"
        # Existing key overridden
        assert runtime.options.get("custom_key") == "overridden"
        # Original config unchanged
        assert echo_provider_with_options.config.options.get("custom_key") == "original"

    def test_options_override_without_existing(self, echo_provider_no_options):
        """Test options override when no existing options."""
        runtime = echo_provider_no_options._get_runtime_config({
            "options": {"new_key": "new_value"}
        })
        assert runtime.options.get("new_key") == "new_value"


class TestOverridePresets:
    """Tests for override presets feature."""

    @pytest.fixture(autouse=True)
    def clear_presets(self):
        """Clear presets before and after each test."""
        from dataknobs_llm.llm.base import AsyncLLMProvider
        AsyncLLMProvider._override_presets.clear()
        yield
        AsyncLLMProvider._override_presets.clear()

    @pytest.fixture
    def echo_provider(self):
        """Create an EchoProvider for testing."""
        config = LLMConfig(
            provider="echo",
            model="base-model",
            temperature=0.5,
        )
        return EchoProvider(config)

    def test_register_preset(self, echo_provider):
        """Test registering a preset."""
        EchoProvider.register_preset("creative", {
            "temperature": 1.2,
            "top_p": 0.95,
        })
        assert "creative" in EchoProvider.list_presets()
        preset = EchoProvider.get_preset("creative")
        assert preset == {"temperature": 1.2, "top_p": 0.95}

    def test_preset_not_found(self, echo_provider):
        """Test error when preset not found."""
        with pytest.raises(ValueError) as exc_info:
            echo_provider._get_runtime_config({"preset": "nonexistent"})
        assert "Unknown preset" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_preset_expansion(self, echo_provider):
        """Test that preset is expanded to actual values."""
        EchoProvider.register_preset("precise", {
            "temperature": 0.1,
            "top_p": 0.5,
        })
        runtime = echo_provider._get_runtime_config({"preset": "precise"})
        assert runtime.temperature == 0.1
        assert runtime.top_p == 0.5
        assert runtime.model == "base-model"  # Preserved from original

    def test_preset_with_explicit_override(self, echo_provider):
        """Test that explicit overrides take precedence over preset."""
        EchoProvider.register_preset("fast", {
            "temperature": 0.0,
            "max_tokens": 50,
        })
        runtime = echo_provider._get_runtime_config({
            "preset": "fast",
            "temperature": 0.3,  # Override the preset value
        })
        assert runtime.temperature == 0.3  # Explicit override wins
        assert runtime.max_tokens == 50  # From preset

    def test_list_presets(self, echo_provider):
        """Test listing all presets."""
        EchoProvider.register_preset("a", {"temperature": 0.1})
        EchoProvider.register_preset("b", {"temperature": 0.2})
        EchoProvider.register_preset("c", {"temperature": 0.3})
        presets = EchoProvider.list_presets()
        assert set(presets) == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_preset_in_complete(self, echo_provider):
        """Test using preset in complete() call."""
        EchoProvider.register_preset("test_preset", {
            "model": "preset-model",
        })
        response = await echo_provider.complete(
            "Hello",
            config_overrides={"preset": "test_preset"}
        )
        assert response.model == "preset-model"


class TestOverrideCallbacks:
    """Tests for override logging/metrics callbacks."""

    @pytest.fixture(autouse=True)
    def clear_callbacks(self):
        """Clear callbacks before and after each test."""
        from dataknobs_llm.llm.base import AsyncLLMProvider
        AsyncLLMProvider.clear_override_callbacks()
        yield
        AsyncLLMProvider.clear_override_callbacks()

    @pytest.fixture
    def echo_provider(self):
        """Create an EchoProvider for testing."""
        config = LLMConfig(
            provider="echo",
            model="test-model",
            temperature=0.5,
        )
        return EchoProvider(config)

    def test_callback_is_called(self, echo_provider):
        """Test that callback is called when overrides are applied."""
        called = []

        def track_callback(provider, overrides, runtime_config):
            called.append({
                "provider": provider,
                "overrides": overrides,
                "runtime_config": runtime_config,
            })

        EchoProvider.on_override_applied(track_callback)
        echo_provider._get_runtime_config({"model": "new-model"})

        assert len(called) == 1
        assert called[0]["overrides"] == {"model": "new-model"}
        assert called[0]["runtime_config"].model == "new-model"

    def test_callback_not_called_without_overrides(self, echo_provider):
        """Test that callback is not called when no overrides."""
        called = []

        def track_callback(provider, overrides, runtime_config):
            called.append(True)

        EchoProvider.on_override_applied(track_callback)
        echo_provider._get_runtime_config(None)
        echo_provider._get_runtime_config({})

        assert len(called) == 0

    def test_multiple_callbacks(self, echo_provider):
        """Test that multiple callbacks are all called."""
        results = {"cb1": 0, "cb2": 0}

        def callback1(provider, overrides, runtime_config):
            results["cb1"] += 1

        def callback2(provider, overrides, runtime_config):
            results["cb2"] += 1

        EchoProvider.on_override_applied(callback1)
        EchoProvider.on_override_applied(callback2)
        echo_provider._get_runtime_config({"temperature": 0.9})

        assert results["cb1"] == 1
        assert results["cb2"] == 1

    def test_callback_error_does_not_break_flow(self, echo_provider):
        """Test that callback errors don't break the main flow."""
        def bad_callback(provider, overrides, runtime_config):
            raise RuntimeError("Callback error!")

        EchoProvider.on_override_applied(bad_callback)

        # Should not raise, even though callback errors
        runtime = echo_provider._get_runtime_config({"model": "new-model"})
        assert runtime.model == "new-model"

    def test_clear_callbacks(self, echo_provider):
        """Test clearing all callbacks."""
        called = []

        def track_callback(provider, overrides, runtime_config):
            called.append(True)

        EchoProvider.on_override_applied(track_callback)
        EchoProvider.clear_override_callbacks()
        echo_provider._get_runtime_config({"model": "new-model"})

        assert len(called) == 0

    @pytest.mark.asyncio
    async def test_callback_called_in_complete(self, echo_provider):
        """Test that callback is called during complete()."""
        called = []

        def track_callback(provider, overrides, runtime_config):
            called.append(overrides)

        EchoProvider.on_override_applied(track_callback)
        await echo_provider.complete("Hello", config_overrides={"model": "tracked"})

        assert len(called) == 1
        assert called[0] == {"model": "tracked"}
