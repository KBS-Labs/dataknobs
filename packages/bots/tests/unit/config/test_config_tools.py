"""Tests for tools/config_tools.py."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml
from dataknobs_common.testing import assert_no_blocking
from dataknobs_llm.tools.context import ToolExecutionContext, ToolWizardState

from dataknobs_bots.config.builder import DynaBotConfigBuilder
from dataknobs_bots.config.drafts import ConfigDraftManager
from dataknobs_bots.config.schema import DynaBotConfigSchema
from dataknobs_bots.config.templates import (
    ConfigTemplate,
    ConfigTemplateRegistry,
    TemplateVariable,
)
from dataknobs_bots.config.validation import ConfigValidator, ValidationResult
from dataknobs_bots.tools.config_tools import (
    GetTemplateDetailsTool,
    ListAvailableToolsTool,
    ListTemplatesTool,
    PreviewConfigTool,
    SaveConfigTool,
    ValidateConfigTool,
)


def _make_context(
    wizard_data: dict[str, Any] | None = None,
) -> ToolExecutionContext:
    """Create a ToolExecutionContext with wizard state."""
    if wizard_data is not None:
        wizard_state = ToolWizardState(
            current_stage="test",
            collected_data=wizard_data,
            history=["test"],
            completed=False,
        )
    else:
        wizard_state = None
    return ToolExecutionContext(
        conversation_id="test-conv",
        user_id="test-user",
        wizard_state=wizard_state,
    )


def _make_registry() -> ConfigTemplateRegistry:
    """Create a registry with test templates."""
    registry = ConfigTemplateRegistry()
    registry.register(
        ConfigTemplate(
            name="basic",
            description="Basic bot",
            version="1.0.0",
            tags=["simple"],
            variables=[
                TemplateVariable(name="bot_name", required=True),
            ],
            structure={
                "bot": {
                    "llm": {"provider": "ollama"},
                    "conversation_storage": {"backend": "memory"},
                    "system_prompt": "I am {{bot_name}}",
                }
            },
        )
    )
    registry.register(
        ConfigTemplate(
            name="advanced",
            description="Advanced bot",
            tags=["advanced", "rag"],
            variables=[
                TemplateVariable(name="bot_name", required=True),
                TemplateVariable(name="subject", required=True),
            ],
            structure={
                "bot": {
                    "llm": {"provider": "ollama"},
                    "conversation_storage": {"backend": "memory"},
                    "knowledge_base": {"enabled": True},
                }
            },
        )
    )
    return registry


def _typo_builder_factory(wizard_data: dict[str, Any]) -> DynaBotConfigBuilder:
    """Builder factory whose config carries a misspelled $resource marker.

    ``$requred`` reads as *not required* to a resolver, so it survives to
    whichever deployment lacks the resource. Only a schema-aware validator
    catches it -- which is exactly the validator the builder carries and
    ``ValidateConfigTool`` used to substitute away.
    """
    builder = (
        DynaBotConfigBuilder()
        .set_llm_resource(wizard_data.get("llm_resource", "default"))
        .set_conversation_storage(wizard_data.get("storage_backend", "memory"))
    )
    builder.merge_overrides({"llm": {"$requred": True}})
    return builder


def _basic_builder_factory(wizard_data: dict[str, Any]) -> DynaBotConfigBuilder:
    """Simple builder factory for testing."""
    builder = (
        DynaBotConfigBuilder()
        .set_llm(
            wizard_data.get("llm_provider", "ollama"),
            model=wizard_data.get("llm_model", "llama3.2"),
        )
        .set_conversation_storage(wizard_data.get("storage_backend", "memory"))
    )
    if "system_prompt" in wizard_data:
        builder.set_system_prompt(content=wizard_data["system_prompt"])
    return builder


def _portable_builder_factory(wizard_data: dict[str, Any]) -> DynaBotConfigBuilder:
    """Builder factory that adds custom sections (for portable test)."""
    builder = (
        DynaBotConfigBuilder()
        .set_llm(
            wizard_data.get("llm_provider", "ollama"),
            model=wizard_data.get("llm_model", "llama3.2"),
        )
        .set_conversation_storage(wizard_data.get("storage_backend", "memory"))
    )
    if wizard_data.get("domain_id"):
        builder.set_custom_section(
            "domain",
            {
                "id": wizard_data["domain_id"],
            },
        )
    return builder


class TestListTemplatesTool:
    """Tests for ListTemplatesTool."""

    @pytest.mark.asyncio
    async def test_list_all(self) -> None:
        tool = ListTemplatesTool(template_registry=_make_registry())
        result = await tool.execute_with_context(_make_context())
        assert result["count"] == 2
        names = [t["name"] for t in result["templates"]]
        assert "basic" in names
        assert "advanced" in names

    @pytest.mark.asyncio
    async def test_list_with_tags(self) -> None:
        tool = ListTemplatesTool(template_registry=_make_registry())
        result = await tool.execute_with_context(_make_context(), tags=["rag"])
        assert result["count"] == 1
        assert result["templates"][0]["name"] == "advanced"

    @pytest.mark.asyncio
    async def test_list_empty_tag_match(self) -> None:
        tool = ListTemplatesTool(template_registry=_make_registry())
        result = await tool.execute_with_context(_make_context(), tags=["nonexistent"])
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_schema(self) -> None:
        tool = ListTemplatesTool(template_registry=_make_registry())
        assert tool.schema["type"] == "object"
        assert "tags" in tool.schema["properties"]


class TestGetTemplateDetailsTool:
    """Tests for GetTemplateDetailsTool."""

    @pytest.mark.asyncio
    async def test_get_existing(self) -> None:
        tool = GetTemplateDetailsTool(template_registry=_make_registry())
        result = await tool.execute_with_context(_make_context(), template_name="basic")
        assert result["name"] == "basic"
        assert result["description"] == "Basic bot"
        assert len(result["variables"]) == 1
        assert len(result["required_variables"]) == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent(self) -> None:
        tool = GetTemplateDetailsTool(template_registry=_make_registry())
        result = await tool.execute_with_context(_make_context(), template_name="missing")
        assert "error" in result
        assert "available" in result

    @pytest.mark.asyncio
    async def test_schema(self) -> None:
        tool = GetTemplateDetailsTool(template_registry=_make_registry())
        assert "template_name" in tool.schema["properties"]
        assert "template_name" in tool.schema["required"]

    @pytest.mark.asyncio
    async def test_omitted_template_name_reports_rather_than_raises(self) -> None:
        """An LLM omitting a required argument gets an error, not a TypeError.

        The base class checks the declared-required set before
        forwarding, and this tool enriches the result with the names
        that would have worked -- so the model can close the gap on its
        next turn rather than reading a Python binding message.
        """
        tool = GetTemplateDetailsTool(template_registry=_make_registry())
        result = await tool.execute()
        assert "error" in result
        assert "template_name" in result["error"]
        assert result["available"] == ["basic", "advanced"]

    @pytest.mark.asyncio
    async def test_empty_template_name_reports_not_found_not_missing(self) -> None:
        """An empty name was supplied, so "missing" names the wrong problem.

        The guard tests presence, not truthiness. ``""`` is a value the
        model sent and the registry can answer for, and the answer it
        deserves is the one every other unmatched name gets.
        """
        tool = GetTemplateDetailsTool(template_registry=_make_registry())

        result = await tool.execute(template_name="")

        assert result["error"] == "Template not found: "
        assert result["available"] == ["basic", "advanced"]


class TestPreviewConfigTool:
    """Tests for PreviewConfigTool."""

    @pytest.mark.asyncio
    async def test_preview_summary(self) -> None:
        tool = PreviewConfigTool(builder_factory=_basic_builder_factory)
        context = _make_context({"llm_provider": "ollama", "llm_model": "llama3.2"})
        result = await tool.execute_with_context(context, format="summary")
        assert "sections" in result

    @pytest.mark.asyncio
    async def test_preview_full(self) -> None:
        tool = PreviewConfigTool(builder_factory=_basic_builder_factory)
        context = _make_context({"llm_provider": "ollama", "storage_backend": "memory"})
        result = await tool.execute_with_context(context, format="full")
        assert "config" in result
        assert result["config"]["llm"]["provider"] == "ollama"

    @pytest.mark.asyncio
    async def test_preview_yaml(self) -> None:
        tool = PreviewConfigTool(builder_factory=_basic_builder_factory)
        context = _make_context({"llm_provider": "openai", "storage_backend": "sqlite"})
        result = await tool.execute_with_context(context, format="yaml")
        assert "yaml" in result
        parsed = yaml.safe_load(result["yaml"])
        assert parsed["llm"]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_preview_no_wizard_data(self) -> None:
        tool = PreviewConfigTool(builder_factory=_basic_builder_factory)
        result = await tool.execute_with_context(_make_context())
        assert "error" in result

    @pytest.mark.asyncio
    async def test_preview_builder_error(self) -> None:
        def bad_factory(data: dict[str, Any]) -> DynaBotConfigBuilder:
            raise ValueError("oops")

        tool = PreviewConfigTool(builder_factory=bad_factory)
        context = _make_context({"some": "data"})
        result = await tool.execute_with_context(context)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_preview_reports_invalid_config(self) -> None:
        """The preview must carry the verdict validate_config reaches.

        One builder factory, one process, one marker typo: the preview
        rendered the config with no verdict of any kind while
        ``validate_config`` returned ``valid=False`` on the same wizard
        data in the same turn -- and ``build()`` raised, so there was no
        final config for the preview to be showing. The two tools are
        wired to the same factory; they must not answer differently.
        """
        wizard_data = {"llm_resource": "default", "storage_backend": "memory"}
        context = _make_context(wizard_data)
        preview_tool = PreviewConfigTool(builder_factory=_typo_builder_factory)
        validate_tool = ValidateConfigTool(builder_factory=_typo_builder_factory)

        preview = await preview_tool.execute_with_context(context, format="summary")
        verdict = await validate_tool.execute_with_context(context)

        assert verdict["valid"] is False, "the fixture must be an invalid config"
        assert preview["valid"] is False
        assert preview["errors"] == verdict["errors"]

    @pytest.mark.asyncio
    async def test_preview_reports_valid_config(self) -> None:
        """The anti-vacuity half: a clean config is reported clean.

        Without this, a preview that hard-coded ``valid=False`` would
        pass the test above.
        """
        context = _make_context({"llm_provider": "ollama", "storage_backend": "memory"})
        tool = PreviewConfigTool(builder_factory=_basic_builder_factory)

        preview = await tool.execute_with_context(context, format="summary")

        assert preview["valid"] is True
        assert preview["errors"] == []

    @pytest.mark.asyncio
    async def test_preview_verdict_in_every_format(self) -> None:
        """All three formats carry it -- a per-branch omission is the half-fix.

        ``summary``, ``full`` and ``yaml`` are three separate returns, so
        carrying the verdict in one is the obvious way to leave the other
        two lying.
        """
        wizard_data = {"llm_resource": "default", "storage_backend": "memory"}
        context = _make_context(wizard_data)
        tool = PreviewConfigTool(builder_factory=_typo_builder_factory)

        for fmt, rendered_key in (
            ("summary", "sections"),
            ("full", "config"),
            ("yaml", "yaml"),
        ):
            preview = await tool.execute_with_context(context, format=fmt)
            assert preview["valid"] is False, f"format={fmt} carries no verdict"
            assert preview["errors"], f"format={fmt} carries no errors"
            assert rendered_key in preview, f"format={fmt} stopped rendering"

    @pytest.mark.asyncio
    async def test_preview_still_renders_when_invalid(self) -> None:
        """Report *and* render -- not a second validate tool.

        The ruling is that the preview carries the verdict and still shows
        the config. A fix that returned only the verdict would satisfy the
        three tests above and destroy the tool.
        """
        wizard_data = {"llm_resource": "default", "storage_backend": "memory"}
        context = _make_context(wizard_data)
        tool = PreviewConfigTool(builder_factory=_typo_builder_factory)

        preview = await tool.execute_with_context(context, format="full")

        assert preview["valid"] is False
        assert preview["config"]["llm"]["$resource"] == "default"
        parsed = yaml.safe_load(
            (await tool.execute_with_context(context, format="yaml"))["yaml"]
        )
        assert parsed["llm"]["$resource"] == "default"

    @pytest.mark.asyncio
    async def test_preview_renders_when_the_validator_itself_raises(self) -> None:
        """A validator that raises must not cost the render.

        ``ConfigValidator.validate`` guards its registered validators but
        calls ``self._schema.validate(config)`` unguarded, so a schema a
        consumer supplies is a reachable raise path -- and subclassing
        that schema is exactly what a consumer does. ``build_unvalidated``
        has already succeeded by then, so there is a config to show;
        reporting the failure as the verdict keeps the preview honest
        without turning a schema bug into a dead tool.
        """

        class _RaisingSchema(DynaBotConfigSchema):
            def validate(self, config: dict[str, Any]) -> ValidationResult:
                raise RuntimeError("schema exploded")

        def raising_schema_factory(data: dict[str, Any]) -> DynaBotConfigBuilder:
            return (
                DynaBotConfigBuilder(schema=_RaisingSchema())
                .set_llm("ollama", model="llama3.2")
                .set_conversation_storage("memory")
            )

        context = _make_context({"llm_provider": "ollama", "storage_backend": "memory"})
        tool = PreviewConfigTool(builder_factory=raising_schema_factory)

        preview = await tool.execute_with_context(context, format="full")

        assert preview["valid"] is False
        assert any("schema exploded" in e for e in preview["errors"])
        assert preview["config"]["llm"]["provider"] == "ollama"


class TestValidateConfigTool:
    """Tests for ValidateConfigTool."""

    @pytest.mark.asyncio
    async def test_valid_config(self) -> None:
        validator = ConfigValidator()
        tool = ValidateConfigTool(validator=validator, builder_factory=_basic_builder_factory)
        context = _make_context({"llm_provider": "ollama", "storage_backend": "memory"})
        result = await tool.execute_with_context(context)
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_no_wizard_data(self) -> None:
        validator = ConfigValidator()
        tool = ValidateConfigTool(validator=validator)
        result = await tool.execute_with_context(_make_context())
        assert result["valid"] is False

    @pytest.mark.asyncio
    async def test_without_builder_factory(self) -> None:
        validator = ConfigValidator()
        tool = ValidateConfigTool(validator=validator)
        context = _make_context(
            {
                "llm": {"provider": "ollama"},
                "conversation_storage": {"backend": "memory"},
            }
        )
        result = await tool.execute_with_context(context)
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_agrees_with_save_on_a_schema_error(self, tmp_path: Path) -> None:
        """validate_config must not say yes to what save_config refuses.

        The tool built its own schema-less ``ConfigValidator`` and ran it
        over ``builder._build_internal()``, while save ran the builder's
        own schema-aware validator. An SME was told the config was valid
        and then told it could not be saved, with no way to reconcile the
        two. The builder already knows the answer: it exposes ``validate()``.
        """
        wizard_data = {"domain_id": "test-bot"}
        validate_tool = ValidateConfigTool(
            validator=ConfigValidator(),
            builder_factory=_typo_builder_factory,
        )
        save_tool = SaveConfigTool(
            draft_manager=ConfigDraftManager(output_dir=tmp_path),
            builder_factory=_typo_builder_factory,
            portable=True,
        )
        context = _make_context(wizard_data)

        validated = await validate_tool.execute_with_context(context)
        saved = await save_tool.execute_with_context(context)

        assert saved["success"] is False
        assert validated["valid"] is False, (
            "validate_config reported valid while save_config refused the same config"
        )
        assert any("$requred" in e for e in validated["errors"])

    @pytest.mark.asyncio
    async def test_agrees_with_save_when_save_is_not_portable(self, tmp_path: Path) -> None:
        """``portable=False`` is the default, and it validated nothing.

        The disagreement this pair was fixed for is symmetrical: with
        the flag left at its default, save built through the unvalidated
        path and wrote to disk exactly the config the validate tool had
        just refused, reporting success. An author reading only the save
        result has no way to learn the config is broken, and the broken
        config is now a file.
        """
        wizard_data = {"domain_id": "test-bot"}
        validate_tool = ValidateConfigTool(builder_factory=_typo_builder_factory)
        save_tool = SaveConfigTool(
            draft_manager=ConfigDraftManager(output_dir=tmp_path),
            builder_factory=_typo_builder_factory,
        )
        context = _make_context(wizard_data)

        validated = await validate_tool.execute_with_context(context)
        saved = await save_tool.execute_with_context(context)

        assert validated["valid"] is False
        assert saved["success"] is False, "save_config wrote a config that validate_config refused"
        written = await asyncio.to_thread(lambda: sorted(tmp_path.iterdir()))
        assert written == [], "an invalid config reached disk"

    @pytest.mark.asyncio
    async def test_agrees_with_save_on_a_clean_config(self, tmp_path: Path) -> None:
        """The false-positive guard for the test above.

        Restoring agreement must not be achieved by making the tool
        pessimistic -- a clean config still passes both.
        """
        wizard_data = {
            "domain_id": "test-bot",
            "llm_provider": "ollama",
            "storage_backend": "memory",
        }
        validate_tool = ValidateConfigTool(
            validator=ConfigValidator(),
            builder_factory=_basic_builder_factory,
        )
        save_tool = SaveConfigTool(
            draft_manager=ConfigDraftManager(output_dir=tmp_path),
            builder_factory=_basic_builder_factory,
            portable=True,
        )
        context = _make_context(wizard_data)

        assert (await validate_tool.execute_with_context(context))["valid"] is True
        assert (await save_tool.execute_with_context(context))["success"] is True

    @pytest.mark.asyncio
    async def test_agrees_with_save_when_both_are_built_from_config(self, tmp_path: Path) -> None:
        """The agreement has to hold on the YAML path, which is the real one.

        Both tools were only ever pinned as directly-constructed
        objects. Production wires them through ``from_config``, which
        resolves its own factory and used to mint its own validator --
        so the path that carries the guarantee was the one path no test
        exercised.
        """
        factory_ref = f"{__name__}:_typo_builder_factory"
        validate_tool = ValidateConfigTool.from_config({"builder_factory": factory_ref})
        save_tool = SaveConfigTool.from_config(
            {"config_dir": str(tmp_path), "builder_factory": factory_ref}
        )
        context = _make_context({"domain_id": "test-bot"})

        validated = await validate_tool.execute_with_context(context)
        saved = await save_tool.execute_with_context(context)

        assert validated["valid"] is False
        assert any("$requred" in e for e in validated["errors"])
        assert saved["success"] is False

    @pytest.mark.asyncio
    async def test_explicit_validator_runs_in_addition_to_the_builders(self) -> None:
        """An explicitly supplied validator adds to the builder's verdict.

        It must not replace it. Replacing is what lets validate and save
        disagree, and the failure directions are not symmetric: an extra
        error blocks an SME, a missing one misleads them.
        """

        def reject_test_bots(config: dict[str, Any]) -> ValidationResult:
            return ValidationResult.error("bots named in a test are not allowed")

        explicit = ConfigValidator()
        explicit.register_validator("no_test_bots", reject_test_bots)
        tool = ValidateConfigTool(
            validator=explicit,
            builder_factory=_typo_builder_factory,
        )

        result = await tool.execute_with_context(_make_context({"domain_id": "test-bot"}))

        assert result["valid"] is False
        assert any("$requred" in e for e in result["errors"]), (
            "the builder's schema-aware verdict was discarded"
        )
        assert "bots named in a test are not allowed" in result["errors"], (
            "the explicitly supplied validator was discarded"
        )
        assert len(result["errors"]) == len(set(result["errors"])), (
            f"duplicate errors from merging two overlapping validators: {result['errors']}"
        )


class TestSaveConfigTool:
    """Tests for SaveConfigTool."""

    @pytest.mark.asyncio
    async def test_save_basic(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        tool = SaveConfigTool(
            draft_manager=manager,
            builder_factory=_basic_builder_factory,
        )
        context = _make_context(
            {
                "domain_id": "test-bot",
                "llm_provider": "ollama",
                "storage_backend": "memory",
            }
        )
        result = await tool.execute_with_context(context)
        assert result["success"] is True
        assert result["config_name"] == "test-bot"
        assert (tmp_path / "test-bot.yaml").exists()

    @pytest.mark.asyncio
    async def test_save_does_not_block_event_loop(self, tmp_path: Path) -> None:
        """The config-save write/finalize must run off the event loop.

        ``execute_with_context`` mkdir's the output dir and writes the YAML
        config to disk; doing that synchronously on the running loop stalls
        every other concurrent conversation on a shared event loop. The
        persist work is offloaded via ``asyncio.to_thread``.
        """
        manager = ConfigDraftManager(output_dir=tmp_path)
        tool = SaveConfigTool(
            draft_manager=manager,
            builder_factory=_basic_builder_factory,
        )
        context = _make_context(
            {
                "domain_id": "test-bot",
                "llm_provider": "ollama",
                "storage_backend": "memory",
            }
        )
        with assert_no_blocking():
            result = await tool.execute_with_context(context)
        assert result["success"] is True
        assert (tmp_path / "test-bot.yaml").exists()

    @pytest.mark.asyncio
    async def test_save_with_draft_does_not_block_event_loop(self, tmp_path: Path) -> None:
        """The draft-finalize path must also run off the event loop.

        When a draft exists, ``execute_with_context`` finalizes it (read /
        unlink / rewrite — all blocking disk I/O) before writing the final
        config; the whole persist tail is offloaded together.
        """
        manager = ConfigDraftManager(output_dir=tmp_path)
        draft_id = manager.create_draft(
            {"llm": {"provider": "ollama"}, "conversation_storage": {"backend": "memory"}}
        )
        tool = SaveConfigTool(
            draft_manager=manager,
            builder_factory=_basic_builder_factory,
        )
        context = _make_context(
            {
                "_draft_id": draft_id,
                "domain_id": "test-bot",
                "llm_provider": "ollama",
                "storage_backend": "memory",
            }
        )
        with assert_no_blocking():
            result = await tool.execute_with_context(context)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_save_with_draft(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        draft_id = manager.create_draft(
            {"llm": {"provider": "ollama"}, "conversation_storage": {"backend": "memory"}}
        )

        tool = SaveConfigTool(
            draft_manager=manager,
            builder_factory=_basic_builder_factory,
        )
        context = _make_context(
            {
                "_draft_id": draft_id,
                "domain_id": "test-bot",
                "llm_provider": "ollama",
                "storage_backend": "memory",
            }
        )
        result = await tool.execute_with_context(context)
        assert result["success"] is True
        # Draft file should be cleaned up
        assert not (tmp_path / f"_draft-{draft_id}.yaml").exists()

    @pytest.mark.asyncio
    async def test_save_with_callback(self, tmp_path: Path) -> None:
        saved_args: list[tuple[str, dict[str, Any]]] = []

        def on_save(name: str, config: dict[str, Any]) -> None:
            saved_args.append((name, config))

        manager = ConfigDraftManager(output_dir=tmp_path)
        tool = SaveConfigTool(
            draft_manager=manager,
            on_save=on_save,
            builder_factory=_basic_builder_factory,
        )
        context = _make_context(
            {
                "domain_id": "cb-bot",
                "llm_provider": "ollama",
                "storage_backend": "memory",
            }
        )
        result = await tool.execute_with_context(context)
        assert result["success"] is True
        assert len(saved_args) == 1
        assert saved_args[0][0] == "cb-bot"

    @pytest.mark.asyncio
    async def test_save_rejects_path_traversal_name(self, tmp_path: Path) -> None:
        """A config name with a path separator is rejected, not written.

        ``config_name`` flows from an LLM tool argument and ``domain_id`` from
        user-driven wizard data; a value like ``../escape`` would otherwise
        compose ``output_dir/../escape.yaml`` and write outside the output
        directory (path traversal). The name is validated before the path is
        composed.
        """
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        manager = ConfigDraftManager(output_dir=out_dir)
        tool = SaveConfigTool(
            draft_manager=manager,
            builder_factory=_basic_builder_factory,
        )
        context = _make_context({"llm_provider": "ollama", "storage_backend": "memory"})
        result = await tool.execute_with_context(context, config_name="../escape")
        assert result["success"] is False
        assert "Invalid config name" in result["error"]
        # Nothing was written outside the output directory.
        assert not (tmp_path / "escape.yaml").exists()

    @pytest.mark.asyncio
    async def test_save_no_wizard_data(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        tool = SaveConfigTool(draft_manager=manager)
        result = await tool.execute_with_context(_make_context())
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_save_no_name(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        tool = SaveConfigTool(
            draft_manager=manager,
            builder_factory=_basic_builder_factory,
        )
        context = _make_context({"llm_provider": "ollama", "storage_backend": "memory"})
        result = await tool.execute_with_context(context)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_save_with_explicit_name(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        tool = SaveConfigTool(
            draft_manager=manager,
            builder_factory=_basic_builder_factory,
        )
        context = _make_context({"llm_provider": "ollama", "storage_backend": "memory"})
        result = await tool.execute_with_context(context, config_name="explicit-name")
        assert result["success"] is True
        assert result["config_name"] == "explicit-name"
        assert (tmp_path / "explicit-name.yaml").exists()

    @pytest.mark.asyncio
    async def test_save_portable(self, tmp_path: Path) -> None:
        """Verify portable=True uses build_portable() (bot wrapper)."""
        manager = ConfigDraftManager(output_dir=tmp_path)
        tool = SaveConfigTool(
            draft_manager=manager,
            builder_factory=_portable_builder_factory,
            portable=True,
        )
        context = _make_context(
            {
                "domain_id": "portable-bot",
                "llm_provider": "ollama",
                "storage_backend": "memory",
            }
        )
        result = await tool.execute_with_context(context)
        assert result["success"] is True

        # Read back and verify portable format (bot wrapper + custom section)
        saved = yaml.safe_load((tmp_path / "portable-bot.yaml").read_text())
        assert "bot" in saved, "Portable format should have 'bot' wrapper key"
        assert saved["bot"]["llm"]["provider"] == "ollama"
        assert saved["domain"]["id"] == "portable-bot"

    @pytest.mark.asyncio
    async def test_save_non_portable(self, tmp_path: Path) -> None:
        """Verify portable=False (default) produces the flat format."""
        manager = ConfigDraftManager(output_dir=tmp_path)
        tool = SaveConfigTool(
            draft_manager=manager,
            builder_factory=_portable_builder_factory,
        )
        context = _make_context(
            {
                "domain_id": "flat-bot",
                "llm_provider": "ollama",
                "storage_backend": "memory",
            }
        )
        result = await tool.execute_with_context(context)
        assert result["success"] is True

        saved = yaml.safe_load((tmp_path / "flat-bot.yaml").read_text())
        assert "bot" not in saved, "Flat format should NOT have 'bot' wrapper"
        assert saved["llm"]["provider"] == "ollama"


class TestListAvailableToolsTool:
    """Tests for ListAvailableToolsTool."""

    @pytest.mark.asyncio
    async def test_list_all(self) -> None:
        catalog = [
            {"name": "search", "description": "Search", "category": "info"},
            {"name": "calc", "description": "Calculator", "category": "math"},
            {"name": "weather", "description": "Weather", "category": "info"},
        ]
        tool = ListAvailableToolsTool(available_tools=catalog)
        result = await tool.execute_with_context(_make_context())
        assert result["count"] == 3
        assert len(result["tools"]) == 3
        assert set(result["categories"]) == {"info", "math"}

    @pytest.mark.asyncio
    async def test_filter_by_category(self) -> None:
        catalog = [
            {"name": "search", "description": "Search", "category": "info"},
            {"name": "calc", "description": "Calculator", "category": "math"},
            {"name": "weather", "description": "Weather", "category": "info"},
        ]
        tool = ListAvailableToolsTool(available_tools=catalog)
        result = await tool.execute_with_context(_make_context(), category="info")
        assert result["count"] == 2
        names = [t["name"] for t in result["tools"]]
        assert "search" in names
        assert "weather" in names
        # Categories still lists all available categories
        assert set(result["categories"]) == {"info", "math"}

    @pytest.mark.asyncio
    async def test_filter_case_insensitive(self) -> None:
        catalog = [
            {"name": "search", "description": "Search", "category": "Info"},
        ]
        tool = ListAvailableToolsTool(available_tools=catalog)
        result = await tool.execute_with_context(_make_context(), category="info")
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_empty_catalog(self) -> None:
        tool = ListAvailableToolsTool(available_tools=[])
        result = await tool.execute_with_context(_make_context())
        assert result["count"] == 0
        assert result["tools"] == []
        assert result["categories"] == []

    @pytest.mark.asyncio
    async def test_schema(self) -> None:
        tool = ListAvailableToolsTool(available_tools=[])
        assert tool.schema["type"] == "object"
        assert "category" in tool.schema["properties"]


class TestDeclaredDependencyInjection:
    """Each tool that declares ``requires`` must use what is injected.

    ``DynaBot._resolve_tool`` reads ``catalog_metadata()['requires']``,
    copies matching entries out of its ``dependencies`` map into the same
    ``params`` dict that carries YAML scalars, and hands the dict to
    ``from_config``. A ``from_config`` that reads only the YAML keys
    throws the live object away — silently for the three tools that
    ignore the key, and with a ``DottedPathError`` for the two that run
    every value through ``resolve_callable``.

    Every assertion here is behavioural: the injected object has to be
    the one the tool *uses*, not merely the one it stored.
    """

    @pytest.mark.asyncio
    async def test_list_templates_uses_the_injected_registry(self) -> None:
        registry = _make_registry()
        tool = ListTemplatesTool.from_config({"template_registry": registry})
        result = await tool.execute_with_context(_make_context())
        assert {t["name"] for t in result["templates"]} == {"basic", "advanced"}

    @pytest.mark.asyncio
    async def test_get_template_details_uses_the_injected_registry(self) -> None:
        registry = _make_registry()
        tool = GetTemplateDetailsTool.from_config({"template_registry": registry})
        result = await tool.execute_with_context(_make_context(), template_name="basic")
        assert result["name"] == "basic"

    @pytest.mark.asyncio
    async def test_preview_config_uses_the_injected_factory(self) -> None:
        tool = PreviewConfigTool.from_config({"builder_factory": _basic_builder_factory})
        context = _make_context({"llm_provider": "openai", "storage_backend": "memory"})
        result = await tool.execute_with_context(context, format="full")
        assert result["config"]["llm"]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_validate_config_uses_the_injected_factory(self) -> None:
        # The typo factory is the one whose verdict differs from a
        # factory-less validation, so a passing assertion here can only
        # mean the injected factory ran.
        tool = ValidateConfigTool.from_config({"builder_factory": _typo_builder_factory})
        context = _make_context({"llm_resource": "main", "storage_backend": "memory"})
        result = await tool.execute_with_context(context)
        assert result["valid"] is False
        assert any("requred" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_save_config_uses_the_injected_draft_manager(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # chdir so that a tool falling back to its relative default
        # ("configs") writes under tmp_path rather than into the repo.
        monkeypatch.chdir(tmp_path)
        injected = ConfigDraftManager(output_dir=tmp_path / "injected")
        # No ``builder_factory``: the draft manager is the only injected
        # dependency, so the assertion below can only be about it.
        tool = SaveConfigTool.from_config({"draft_manager": injected})
        context = _make_context({"llm_provider": "ollama", "storage_backend": "memory"})
        result = await tool.execute_with_context(context, config_name="mybot")
        assert result["success"] is True
        assert Path(result["file_path"]).parent == tmp_path / "injected"
