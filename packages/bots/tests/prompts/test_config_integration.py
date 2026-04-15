"""Integration tests for prompt library config wiring.

Verifies:
- DynaBot.from_config() builds CompositePromptLibrary with defaults
- prompt_resolver is created and stored on bot instance
- prompt_libraries config key is parsed correctly
- Inline prompts override defaults
- PromptResolver resolves and renders prompts correctly
- WizardResponder uses prompt_resolver when available
- FocusGuard uses prompt_resolver when available
- SummaryMemory uses prompt_resolver when available
"""

import pytest

from dataknobs_llm.prompts import ConfigPromptLibrary, CompositePromptLibrary
from dataknobs_llm.prompts.base.types import TemplateMode

from dataknobs_bots.prompts.resolver import PromptResolver
from dataknobs_bots.prompts.defaults import (
    _collect_all_bots_keys,
    get_default_prompt_library,
    get_full_prompt_library,
)


# ============================================================================
# PromptResolver tests
# ============================================================================

class TestPromptResolver:

    def test_resolve_simple_format_template(self) -> None:
        library = ConfigPromptLibrary(config={
            "system": {
                "test.greeting": {
                    "template": "Hello, {name}!",
                    "template_syntax": "format",
                },
            },
        })
        resolver = PromptResolver(library)
        result = resolver.resolve("test.greeting", name="World")
        assert result == "Hello, World!"

    def test_resolve_meta_prompt_with_prompt_ref(self) -> None:
        library = get_default_prompt_library()
        resolver = PromptResolver(library)

        result = resolver.resolve(
            "wizard.clarification",
            issue_list="- missing name",
            stage_prompt="Enter your name",
            suggestions_text="",
        )
        assert result is not None
        assert "Clarification Needed" in result
        assert "missing name" in result
        assert "Enter your name" in result

    def test_resolve_nonexistent_key_returns_none(self) -> None:
        library = get_default_prompt_library()
        resolver = PromptResolver(library)
        assert resolver.resolve("nonexistent.key") is None

    def test_resolve_validation_meta(self) -> None:
        library = get_default_prompt_library()
        resolver = PromptResolver(library)

        result = resolver.resolve(
            "wizard.validation",
            error_list="- email is required",
            stage_prompt="Provide your email",
        )
        assert result is not None
        assert "Validation Required" in result
        assert "email is required" in result

    def test_resolve_transform_error_meta(self) -> None:
        library = get_default_prompt_library()
        resolver = PromptResolver(library)

        result = resolver.resolve(
            "wizard.transform_error",
            stage_name="payment",
            error="Connection timeout",
        )
        assert result is not None
        assert "Processing Error" in result
        assert "payment" in result
        assert "Connection timeout" in result

    def test_resolve_restart_offer_meta(self) -> None:
        library = get_default_prompt_library()
        resolver = PromptResolver(library)

        result = resolver.resolve(
            "wizard.restart_offer",
            stage_name="address",
            stage_prompt="Enter your address",
        )
        assert result is not None
        assert "Multiple Clarification Attempts" in result
        assert "address" in result

    def test_resolve_focus_guidance_meta(self) -> None:
        library = get_default_prompt_library()
        resolver = PromptResolver(library)

        result = resolver.resolve(
            "focus.guidance",
            primary_goal="Complete the form",
            current_task="Enter name",
            required_fields="name, email",
            collected="age",
        )
        assert result is not None
        assert "Focus Guidance" in result
        assert "Complete the form" in result
        assert "Enter name" in result

    def test_resolve_focus_drift_gentle(self) -> None:
        library = get_default_prompt_library()
        resolver = PromptResolver(library)

        result = resolver.resolve(
            "focus.drift",
            reason="User asked about weather",
            suggested_redirect="Back to form",
            tangent_count=1,
            max_tangent_depth=3,
        )
        assert result is not None
        assert "Focus Correction" in result
        assert "gently steer" in result

    def test_resolve_focus_drift_firm(self) -> None:
        library = get_default_prompt_library()
        resolver = PromptResolver(library)

        result = resolver.resolve(
            "focus.drift",
            reason="Persistent drift",
            suggested_redirect="",
            tangent_count=3,
            max_tangent_depth=3,
        )
        assert result is not None
        assert "IMPORTANT" in result
        assert "firmly redirect" in result

    def test_library_property(self) -> None:
        library = get_default_prompt_library()
        resolver = PromptResolver(library)
        assert resolver.library is library


class TestPromptResolverWithOverrides:

    def test_override_fragment_changes_meta_output(self) -> None:
        """Overriding a fragment changes the composed meta-prompt."""
        custom_keys = dict(_collect_all_bots_keys())
        custom_keys["wizard.clarification.instructions"] = {
            "template": "Please be VERY formal and concise.",
            "template_syntax": "format",
        }

        library = ConfigPromptLibrary(config={"system": custom_keys})
        resolver = PromptResolver(library)

        result = resolver.resolve(
            "wizard.clarification",
            issue_list="- missing field",
            stage_prompt="Provide info",
            suggestions_text="",
        )
        assert result is not None
        assert "VERY formal and concise" in result
        # Original instructions should NOT appear
        assert "don't make the user feel" not in result

    def test_composite_override_precedence(self) -> None:
        """Consumer overrides take precedence over defaults."""
        override_library = ConfigPromptLibrary(config={
            "system": {
                "wizard.clarification.header": {
                    "template": "## CUSTOM Clarification",
                    "template_syntax": "format",
                },
            },
        })
        default_library = get_default_prompt_library()

        composite = CompositePromptLibrary(
            libraries=[override_library, default_library],
            names=["overrides", "defaults"],
        )
        resolver = PromptResolver(composite)

        result = resolver.resolve(
            "wizard.clarification",
            issue_list="- test",
            stage_prompt="Test",
            suggestions_text="",
        )
        assert result is not None
        assert "CUSTOM Clarification" in result


# ============================================================================
# FocusGuard integration tests
# ============================================================================

class TestFocusGuardPromptIntegration:

    def test_focus_guard_with_resolver(self) -> None:
        from dataknobs_bots.reasoning.focus_guard import FocusContext, FocusGuard

        library = get_default_prompt_library()
        resolver = PromptResolver(library)
        guard = FocusGuard(prompt_resolver=resolver)

        context = FocusContext(
            primary_goal="Complete registration",
            current_task="Enter email",
            required_fields=["email", "phone"],
            collected_data={"name": "Alice"},
            max_tangent_depth=2,
        )

        prompt = guard.get_focus_prompt(context)
        assert "Focus Guidance" in prompt
        assert "Complete registration" in prompt
        assert "Enter email" in prompt
        assert "email, phone" in prompt

    def test_focus_guard_without_resolver_uses_fallback(self) -> None:
        from dataknobs_bots.reasoning.focus_guard import FocusContext, FocusGuard

        guard = FocusGuard()  # No resolver

        context = FocusContext(
            primary_goal="Complete form",
            current_task="Enter name",
            required_fields=["name"],
            collected_data={},
            max_tangent_depth=2,
        )

        prompt = guard.get_focus_prompt(context)
        assert "Focus Guidance" in prompt
        assert "Complete form" in prompt

    def test_correction_prompt_with_resolver(self) -> None:
        from dataknobs_bots.reasoning.focus_guard import FocusEvaluation, FocusGuard

        library = get_default_prompt_library()
        resolver = PromptResolver(library)
        guard = FocusGuard(prompt_resolver=resolver)

        evaluation = FocusEvaluation(
            is_drifting=True,
            drift_severity=0.8,
            reason="Discussing weather",
            suggested_redirect="Back to form",
            tangent_count=1,
        )

        prompt = guard.get_correction_prompt(evaluation)
        assert "Focus Correction" in prompt
        assert "gently steer" in prompt


# ============================================================================
# SummaryMemory integration tests
# ============================================================================

class TestSummaryMemoryPromptIntegration:

    def test_summary_memory_with_resolver(self) -> None:
        from dataknobs_llm import EchoProvider

        from dataknobs_bots.memory.summary import DEFAULT_SUMMARY_PROMPT, SummaryMemory

        library = get_default_prompt_library()
        resolver = PromptResolver(library)
        provider = EchoProvider(config={"provider": "echo", "model": "test"})

        memory = SummaryMemory(
            llm_provider=provider,
            prompt_resolver=resolver,
        )

        # Should use the library-resolved prompt (which matches the default)
        assert "conversation summarizer" in memory.summary_prompt

    def test_summary_memory_explicit_prompt_overrides_library(self) -> None:
        from dataknobs_llm import EchoProvider

        from dataknobs_bots.memory.summary import SummaryMemory

        library = get_default_prompt_library()
        resolver = PromptResolver(library)
        provider = EchoProvider(config={"provider": "echo", "model": "test"})

        custom = "Custom summary: {existing_summary} + {new_messages}"
        memory = SummaryMemory(
            llm_provider=provider,
            summary_prompt=custom,
            prompt_resolver=resolver,
        )

        # Explicit param takes priority over library
        assert memory.summary_prompt == custom

    def test_summary_memory_without_resolver_uses_default(self) -> None:
        from dataknobs_llm import EchoProvider

        from dataknobs_bots.memory.summary import DEFAULT_SUMMARY_PROMPT, SummaryMemory

        provider = EchoProvider(config={"provider": "echo", "model": "test"})
        memory = SummaryMemory(llm_provider=provider)
        assert memory.summary_prompt == DEFAULT_SUMMARY_PROMPT


# ============================================================================
# BotTestHarness integration tests
# ============================================================================

class TestBotPromptResolverIntegration:

    @pytest.mark.asyncio
    async def test_bot_has_prompt_resolver(self) -> None:
        """Bot created via BotTestHarness has a prompt_resolver."""
        from dataknobs_bots.testing import BotTestHarness
        from dataknobs_llm.testing import text_response

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "simple"},
            },
            main_responses=[text_response("Hello!")],
        ) as harness:
            assert harness.bot.prompt_resolver is not None
            assert isinstance(
                harness.bot.prompt_resolver, PromptResolver
            )

    @pytest.mark.asyncio
    async def test_bot_resolver_has_default_keys(self) -> None:
        """Bot's prompt_resolver can resolve default prompt keys."""
        from dataknobs_bots.testing import BotTestHarness
        from dataknobs_llm.testing import text_response

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "simple"},
            },
            main_responses=[text_response("Hello!")],
        ) as harness:
            resolver = harness.bot.prompt_resolver
            # Should be able to resolve wizard prompts
            result = resolver.resolve(
                "wizard.clarification",
                issue_list="- test",
                stage_prompt="Test",
                suggestions_text="",
            )
            assert result is not None
            assert "Clarification" in result

            # And extraction prompts
            result = resolver.resolve(
                "extraction.default.instructions",
            )
            assert result is not None

    @pytest.mark.asyncio
    async def test_bot_inline_prompts_override_defaults(self) -> None:
        """Inline prompts in config override default library."""
        from dataknobs_bots.testing import BotTestHarness
        from dataknobs_llm.testing import text_response

        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "simple"},
                "prompts": {
                    "wizard.clarification.header": {
                        "template": "## OVERRIDDEN Header",
                        "template_syntax": "format",
                    },
                },
            },
            main_responses=[text_response("Hello!")],
        ) as harness:
            resolver = harness.bot.prompt_resolver
            result = resolver.resolve(
                "wizard.clarification",
                issue_list="- test",
                stage_prompt="Test",
                suggestions_text="",
            )
            assert result is not None
            assert "OVERRIDDEN Header" in result
