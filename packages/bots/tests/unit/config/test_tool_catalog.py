"""Tests for config/tool_catalog.py."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_common.exceptions import NotFoundError, OperationError

from dataknobs_bots.config.tool_catalog import (
    CatalogDescribable,
    ToolCatalog,
    ToolEntry,
    create_default_catalog,
    default_catalog,
)


# -- Helpers --


class _FakeTool:
    """Minimal tool class with catalog_metadata() for testing."""

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        return {
            "name": "fake_tool",
            "description": "A fake tool for testing.",
            "tags": ("test", "fake"),
            "requires": ("fake_dep",),
            "default_params": {"param1": "value1"},
        }


class _NoMetadataTool:
    """Tool class without catalog_metadata() for testing."""

    pass


# -- ToolEntry Tests --


class TestToolEntry:
    """Tests for ToolEntry dataclass."""

    def test_create_entry(self) -> None:
        entry = ToolEntry(
            name="my_tool",
            class_path="my.module.MyTool",
            description="A test tool.",
            default_params={"k": 5},
            tags=frozenset({"general", "rag"}),
            requires=frozenset({"knowledge_base"}),
        )
        assert entry.name == "my_tool"
        assert entry.class_path == "my.module.MyTool"
        assert entry.description == "A test tool."
        assert entry.default_params == {"k": 5}
        assert entry.tags == frozenset({"general", "rag"})
        assert entry.requires == frozenset({"knowledge_base"})

    def test_frozen(self) -> None:
        entry = ToolEntry(name="t", class_path="m.T")
        with pytest.raises(AttributeError):
            entry.name = "changed"  # type: ignore[misc]

    def test_default_params_none_becomes_empty_dict(self) -> None:
        entry = ToolEntry(name="t", class_path="m.T")
        assert entry.default_params == {}

    def test_to_dict(self) -> None:
        entry = ToolEntry(
            name="tool",
            class_path="m.Tool",
            description="desc",
            default_params={"k": 5},
            tags=frozenset({"b", "a"}),
            requires=frozenset({"req"}),
        )
        d = entry.to_dict()
        assert d["name"] == "tool"
        assert d["class_path"] == "m.Tool"
        assert d["description"] == "desc"
        assert d["default_params"] == {"k": 5}
        assert d["tags"] == ["a", "b"]  # sorted
        assert d["requires"] == ["req"]

    def test_to_dict_omits_empty_fields(self) -> None:
        entry = ToolEntry(name="t", class_path="m.T")
        d = entry.to_dict()
        assert "description" not in d
        assert "default_params" not in d
        assert "tags" not in d
        assert "requires" not in d

    def test_from_dict(self) -> None:
        data = {
            "name": "tool",
            "class_path": "m.Tool",
            "description": "desc",
            "default_params": {"k": 5},
            "tags": ["a", "b"],
            "requires": ["req"],
        }
        entry = ToolEntry.from_dict(data)
        assert entry.name == "tool"
        assert entry.class_path == "m.Tool"
        assert entry.description == "desc"
        assert entry.default_params == {"k": 5}
        assert entry.tags == frozenset({"a", "b"})
        assert entry.requires == frozenset({"req"})

    def test_from_dict_handles_missing_optionals(self) -> None:
        data = {"name": "t", "class_path": "m.T"}
        entry = ToolEntry.from_dict(data)
        assert entry.description == ""
        assert entry.default_params == {}
        assert entry.tags == frozenset()
        assert entry.requires == frozenset()

    def test_roundtrip(self) -> None:
        entry = ToolEntry(
            name="tool",
            class_path="m.Tool",
            description="desc",
            default_params={"k": 5},
            tags=frozenset({"a", "b"}),
            requires=frozenset({"req"}),
        )
        restored = ToolEntry.from_dict(entry.to_dict())
        assert restored.name == entry.name
        assert restored.class_path == entry.class_path
        assert restored.description == entry.description
        assert restored.default_params == entry.default_params
        assert restored.tags == entry.tags
        assert restored.requires == entry.requires

    def test_to_bot_config(self) -> None:
        entry = ToolEntry(
            name="tool",
            class_path="m.Tool",
            default_params={"k": 5},
        )
        config = entry.to_bot_config()
        assert config == {"class": "m.Tool", "params": {"k": 5}}

    def test_to_bot_config_with_overrides(self) -> None:
        entry = ToolEntry(
            name="tool",
            class_path="m.Tool",
            default_params={"k": 5, "mode": "fast"},
        )
        config = entry.to_bot_config(k=10, extra="val")
        assert config["class"] == "m.Tool"
        assert config["params"]["k"] == 10
        assert config["params"]["mode"] == "fast"
        assert config["params"]["extra"] == "val"

    def test_to_bot_config_no_params(self) -> None:
        entry = ToolEntry(name="tool", class_path="m.Tool")
        config = entry.to_bot_config()
        assert config == {"class": "m.Tool"}
        assert "params" not in config


# -- ToolCatalog Tests --


class TestToolCatalog:
    """Tests for ToolCatalog core functionality."""

    def test_register_and_get(self) -> None:
        catalog = ToolCatalog()
        entry = ToolEntry(name="t", class_path="m.T")
        catalog.register_entry(entry)
        assert catalog.get("t") is entry

    def test_register_tool_convenience(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(
            name="t",
            class_path="m.T",
            description="desc",
            tags=("a",),
            requires=("r",),
        )
        entry = catalog.get("t")
        assert entry.class_path == "m.T"
        assert entry.description == "desc"
        assert entry.tags == frozenset({"a"})
        assert entry.requires == frozenset({"r"})

    def test_register_from_dict(self) -> None:
        catalog = ToolCatalog()
        catalog.register_from_dict({
            "name": "t",
            "class_path": "m.T",
            "tags": ["a"],
        })
        entry = catalog.get("t")
        assert entry.tags == frozenset({"a"})

    def test_register_many_from_dicts(self) -> None:
        catalog = ToolCatalog()
        catalog.register_many_from_dicts([
            {"name": "t1", "class_path": "m.T1"},
            {"name": "t2", "class_path": "m.T2"},
        ])
        assert catalog.count() == 2

    def test_get_unknown_raises(self) -> None:
        catalog = ToolCatalog()
        with pytest.raises(NotFoundError):
            catalog.get("nonexistent")

    def test_has(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(name="t", class_path="m.T")
        assert catalog.has("t") is True
        assert catalog.has("missing") is False

    def test_list_tools(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(name="t1", class_path="m.T1")
        catalog.register_tool(name="t2", class_path="m.T2")
        tools = catalog.list_tools()
        assert len(tools) == 2

    def test_list_tools_with_tag_filter(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(name="t1", class_path="m.T1", tags=("a", "b"))
        catalog.register_tool(name="t2", class_path="m.T2", tags=("b", "c"))
        catalog.register_tool(name="t3", class_path="m.T3", tags=("d",))

        result = catalog.list_tools(tags=["a"])
        assert len(result) == 1
        assert result[0].name == "t1"

        # ANY semantics: a OR c
        result = catalog.list_tools(tags=["a", "c"])
        assert len(result) == 2

    def test_list_tools_no_matching_tags(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(name="t1", class_path="m.T1", tags=("a",))
        result = catalog.list_tools(tags=["z"])
        assert len(result) == 0

    def test_get_names(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(name="t1", class_path="m.T1")
        catalog.register_tool(name="t2", class_path="m.T2")
        names = catalog.get_names()
        assert set(names) == {"t1", "t2"}

    def test_to_bot_config(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(
            name="t",
            class_path="m.T",
            default_params={"k": 5},
        )
        config = catalog.to_bot_config("t")
        assert config == {"class": "m.T", "params": {"k": 5}}

    def test_to_bot_configs(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(name="t1", class_path="m.T1")
        catalog.register_tool(name="t2", class_path="m.T2")
        configs = catalog.to_bot_configs(["t1", "t2"])
        assert len(configs) == 2
        assert configs[0]["class"] == "m.T1"
        assert configs[1]["class"] == "m.T2"

    def test_to_bot_configs_with_overrides(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(name="t1", class_path="m.T1")
        catalog.register_tool(name="t2", class_path="m.T2")
        configs = catalog.to_bot_configs(
            ["t1", "t2"],
            overrides={"t1": {"k": 10}},
        )
        assert configs[0]["params"]["k"] == 10
        assert "params" not in configs[1]

    def test_to_dict(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(name="t1", class_path="m.T1")
        catalog.register_tool(name="t2", class_path="m.T2")
        d = catalog.to_dict()
        assert "tools" in d
        assert len(d["tools"]) == 2

    def test_from_dict(self) -> None:
        data = {
            "tools": [
                {"name": "t1", "class_path": "m.T1"},
                {"name": "t2", "class_path": "m.T2"},
            ]
        }
        catalog = ToolCatalog.from_dict(data)
        assert catalog.count() == 2
        assert catalog.has("t1")
        assert catalog.has("t2")

    def test_duplicate_registration_raises(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(name="t", class_path="m.T")
        with pytest.raises(OperationError):
            catalog.register_tool(name="t", class_path="m.T2")

    def test_contains(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(name="t", class_path="m.T")
        assert "t" in catalog
        assert "missing" not in catalog

    def test_len(self) -> None:
        catalog = ToolCatalog()
        assert len(catalog) == 0
        catalog.register_tool(name="t", class_path="m.T")
        assert len(catalog) == 1


# -- Self-Describing Tool Tests --


class TestSelfDescribingTools:
    """Tests for register_from_class() and CatalogDescribable."""

    def test_register_from_class(self) -> None:
        catalog = ToolCatalog()
        catalog.register_from_class(_FakeTool)
        assert catalog.has("fake_tool")

    def test_register_from_class_metadata_fields(self) -> None:
        catalog = ToolCatalog()
        catalog.register_from_class(_FakeTool)
        entry = catalog.get("fake_tool")
        assert entry.description == "A fake tool for testing."
        assert entry.tags == frozenset({"test", "fake"})
        assert entry.requires == frozenset({"fake_dep"})
        assert entry.default_params == {"param1": "value1"}

    def test_register_from_class_no_metadata_raises(self) -> None:
        catalog = ToolCatalog()
        with pytest.raises(ValueError, match="does not implement"):
            catalog.register_from_class(_NoMetadataTool)

    def test_class_path_auto_computed(self) -> None:
        catalog = ToolCatalog()
        catalog.register_from_class(_FakeTool)
        entry = catalog.get("fake_tool")
        expected = f"{_FakeTool.__module__}.{_FakeTool.__qualname__}"
        assert entry.class_path == expected

    def test_catalog_describable_protocol(self) -> None:
        assert isinstance(_FakeTool, CatalogDescribable)
        assert not isinstance(_NoMetadataTool, CatalogDescribable)

    def test_all_builtin_tools_describable(self) -> None:
        from dataknobs_bots.tools import (
            AddKBResourceTool,
            CheckKnowledgeSourceTool,
            GetTemplateDetailsTool,
            IngestKnowledgeBaseTool,
            KnowledgeSearchTool,
            ListAvailableToolsTool,
            ListKBResourcesTool,
            ListTemplatesTool,
            PreviewConfigTool,
            RemoveKBResourceTool,
            SaveConfigTool,
            ValidateConfigTool,
        )

        for tool_class in [
            KnowledgeSearchTool,
            ListTemplatesTool,
            GetTemplateDetailsTool,
            PreviewConfigTool,
            ValidateConfigTool,
            SaveConfigTool,
            ListAvailableToolsTool,
            CheckKnowledgeSourceTool,
            ListKBResourcesTool,
            AddKBResourceTool,
            RemoveKBResourceTool,
            IngestKnowledgeBaseTool,
        ]:
            assert hasattr(tool_class, "catalog_metadata"), (
                f"{tool_class.__name__} missing catalog_metadata()"
            )
            meta = tool_class.catalog_metadata()
            assert "name" in meta, f"{tool_class.__name__} metadata missing 'name'"
            assert "description" in meta, (
                f"{tool_class.__name__} metadata missing 'description'"
            )


# -- Dependency Validation Tests --


class TestDependencyValidation:
    """Tests for get_requirements() and check_requirements()."""

    def test_get_requirements(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(
            name="t1", class_path="m.T1", requires=("a", "b")
        )
        catalog.register_tool(
            name="t2", class_path="m.T2", requires=("b", "c")
        )
        reqs = catalog.get_requirements(["t1", "t2"])
        assert reqs == frozenset({"a", "b", "c"})

    def test_get_requirements_no_requires(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(name="t1", class_path="m.T1")
        reqs = catalog.get_requirements(["t1"])
        assert reqs == frozenset()

    def test_check_requirements_satisfied(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(
            name="t1", class_path="m.T1", requires=("knowledge_base",)
        )
        warnings = catalog.check_requirements(
            ["t1"], {"knowledge_base": {"enabled": True}}
        )
        assert warnings == []

    def test_check_requirements_missing(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(
            name="t1", class_path="m.T1", requires=("knowledge_base",)
        )
        warnings = catalog.check_requirements(["t1"], {})
        assert len(warnings) == 1
        assert "knowledge_base" in warnings[0]
        assert "t1" in warnings[0]

    def test_check_requirements_partial(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(
            name="t1",
            class_path="m.T1",
            requires=("knowledge_base", "vector_store"),
        )
        warnings = catalog.check_requirements(
            ["t1"], {"knowledge_base": True}
        )
        assert len(warnings) == 1
        assert "vector_store" in warnings[0]


# -- Instantiation Tests --


class TestToolInstantiation:
    """Tests for instantiate_tool() and create_tool_registry()."""

    def test_instantiate_simple_tool(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(
            name="check_knowledge_source",
            class_path=(
                "dataknobs_bots.tools.kb_tools.CheckKnowledgeSourceTool"
            ),
        )
        tool = catalog.instantiate_tool("check_knowledge_source")
        assert tool.name == "check_knowledge_source"

    def test_instantiate_unknown_raises(self) -> None:
        catalog = ToolCatalog()
        with pytest.raises(NotFoundError):
            catalog.instantiate_tool("nonexistent")

    def test_create_tool_registry(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(
            name="check_knowledge_source",
            class_path=(
                "dataknobs_bots.tools.kb_tools.CheckKnowledgeSourceTool"
            ),
        )
        catalog.register_tool(
            name="list_kb_resources",
            class_path="dataknobs_bots.tools.kb_tools.ListKBResourcesTool",
        )
        registry = catalog.create_tool_registry(
            names=["check_knowledge_source", "list_kb_resources"],
        )
        assert registry.has("check_knowledge_source")
        assert registry.has("list_kb_resources")

    def test_create_tool_registry_non_strict_skips_failures(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(
            name="bad_tool",
            class_path="nonexistent.module.Tool",
        )
        catalog.register_tool(
            name="check_knowledge_source",
            class_path=(
                "dataknobs_bots.tools.kb_tools.CheckKnowledgeSourceTool"
            ),
        )
        registry = catalog.create_tool_registry(strict=False)
        # bad_tool skipped, check_knowledge_source succeeded
        assert registry.has("check_knowledge_source")
        assert not registry.has("bad_tool")

    def test_create_tool_registry_strict_raises(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(
            name="bad_tool",
            class_path="nonexistent.module.Tool",
        )
        with pytest.raises(Exception):
            catalog.create_tool_registry(strict=True)


# -- Default Catalog Tests --


class TestDefaultCatalog:
    """Tests for default_catalog and create_default_catalog()."""

    def test_default_catalog_populated(self) -> None:
        assert default_catalog.count() == 12

    def test_default_catalog_has_knowledge_search(self) -> None:
        assert default_catalog.has("knowledge_search")

    def test_default_catalog_tool_entries_valid(self) -> None:
        for entry in default_catalog.list_items():
            assert entry.name
            assert entry.class_path
            assert "." in entry.class_path

    def test_create_default_catalog_independent(self) -> None:
        catalog = create_default_catalog()
        assert catalog.count() == default_catalog.count()
        # Adding to copy doesn't affect original
        catalog.register_tool(name="extra", class_path="m.Extra")
        assert catalog.count() == default_catalog.count() + 1

    def test_create_default_catalog_extensible(self) -> None:
        catalog = create_default_catalog()
        catalog.register_tool(name="custom", class_path="m.Custom")
        assert catalog.has("custom")
        assert catalog.has("knowledge_search")

    def test_default_catalog_class_paths_importable(self) -> None:
        from dataknobs_bots.tools.resolve import resolve_callable

        # All default catalog class paths should be importable
        for entry in default_catalog.list_items():
            resolved = resolve_callable(entry.class_path)
            assert callable(resolved), (
                f"Class path {entry.class_path} did not resolve to callable"
            )

    def test_default_catalog_has_all_expected_tools(self) -> None:
        expected_names = {
            "knowledge_search",
            "list_templates",
            "get_template_details",
            "preview_config",
            "validate_config",
            "save_config",
            "list_available_tools",
            "check_knowledge_source",
            "list_kb_resources",
            "add_kb_resource",
            "remove_kb_resource",
            "ingest_knowledge_base",
        }
        actual_names = set(default_catalog.get_names())
        assert actual_names == expected_names


# -- Builder Integration Tests --


class TestBuilderIntegration:
    """Tests for DynaBotConfigBuilder.add_tool_by_name()."""

    def test_add_tool_by_name(self) -> None:
        from dataknobs_bots.config.builder import DynaBotConfigBuilder

        catalog = ToolCatalog()
        catalog.register_tool(
            name="t",
            class_path="m.T",
            default_params={"k": 5},
        )
        builder = (
            DynaBotConfigBuilder()
            .set_llm("ollama")
            .set_conversation_storage("memory")
            .add_tool_by_name(catalog, "t")
        )
        config = builder.build()
        assert len(config["tools"]) == 1
        assert config["tools"][0]["class"] == "m.T"
        assert config["tools"][0]["params"]["k"] == 5

    def test_add_tool_by_name_with_overrides(self) -> None:
        from dataknobs_bots.config.builder import DynaBotConfigBuilder

        catalog = ToolCatalog()
        catalog.register_tool(
            name="t",
            class_path="m.T",
            default_params={"k": 5},
        )
        builder = (
            DynaBotConfigBuilder()
            .set_llm("ollama")
            .set_conversation_storage("memory")
            .add_tool_by_name(catalog, "t", k=10, extra="val")
        )
        config = builder.build()
        assert config["tools"][0]["params"]["k"] == 10
        assert config["tools"][0]["params"]["extra"] == "val"

    def test_add_tools_by_name(self) -> None:
        from dataknobs_bots.config.builder import DynaBotConfigBuilder

        catalog = ToolCatalog()
        catalog.register_tool(name="t1", class_path="m.T1")
        catalog.register_tool(name="t2", class_path="m.T2")
        builder = (
            DynaBotConfigBuilder()
            .set_llm("ollama")
            .set_conversation_storage("memory")
            .add_tools_by_name(catalog, ["t1", "t2"])
        )
        config = builder.build()
        assert len(config["tools"]) == 2

    def test_add_tool_by_name_unknown_raises(self) -> None:
        from dataknobs_bots.config.builder import DynaBotConfigBuilder

        catalog = ToolCatalog()
        builder = DynaBotConfigBuilder()
        with pytest.raises(NotFoundError):
            builder.add_tool_by_name(catalog, "nonexistent")

    def test_add_tool_by_name_chains(self) -> None:
        from dataknobs_bots.config.builder import DynaBotConfigBuilder

        catalog = ToolCatalog()
        catalog.register_tool(name="t1", class_path="m.T1")
        catalog.register_tool(name="t2", class_path="m.T2")
        builder = (
            DynaBotConfigBuilder()
            .set_llm("ollama")
            .set_conversation_storage("memory")
            .add_tool_by_name(catalog, "t1")
            .add_tool_by_name(catalog, "t2")
        )
        config = builder.build()
        assert len(config["tools"]) == 2


# -- Wizard Builder Integration Tests --


class TestWizardBuilderIntegration:
    """Tests for WizardConfigBuilder.set_tool_catalog()."""

    def test_validate_with_catalog_valid(self) -> None:
        from dataknobs_bots.config.wizard_builder import WizardConfigBuilder

        catalog = ToolCatalog()
        catalog.register_tool(name="t1", class_path="m.T1")

        wizard = (
            WizardConfigBuilder("test")
            .set_tool_catalog(catalog)
            .add_conversation_stage(
                name="chat",
                prompt="Hello",
                tools=["t1"],
                is_start=True,
            )
            .build()
        )
        assert wizard.name == "test"

    def test_validate_with_catalog_invalid(self) -> None:
        from dataknobs_bots.config.wizard_builder import WizardConfigBuilder

        catalog = ToolCatalog()
        catalog.register_tool(name="t1", class_path="m.T1")

        with pytest.raises(ValueError, match="unknown tool"):
            (
                WizardConfigBuilder("test")
                .set_tool_catalog(catalog)
                .add_conversation_stage(
                    name="chat",
                    prompt="Hello",
                    tools=["nonexistent_tool"],
                    is_start=True,
                )
                .build()
            )

    def test_validate_without_catalog(self) -> None:
        from dataknobs_bots.config.wizard_builder import WizardConfigBuilder

        # Without catalog, any tool names are accepted (backward compat)
        wizard = (
            WizardConfigBuilder("test")
            .add_conversation_stage(
                name="chat",
                prompt="Hello",
                tools=["any_tool_name"],
                is_start=True,
            )
            .build()
        )
        assert wizard.name == "test"

    def test_set_tool_catalog_chains(self) -> None:
        from dataknobs_bots.config.wizard_builder import WizardConfigBuilder

        catalog = ToolCatalog()
        builder = WizardConfigBuilder("test").set_tool_catalog(catalog)
        assert builder._tool_catalog is catalog


# -- Serialization Roundtrip Tests --


class TestCatalogSerialization:
    """Tests for full catalog serialization roundtrip."""

    def test_roundtrip(self) -> None:
        catalog = ToolCatalog()
        catalog.register_tool(
            name="t1",
            class_path="m.T1",
            description="First tool",
            default_params={"k": 5},
            tags=("a", "b"),
            requires=("r1",),
        )
        catalog.register_tool(
            name="t2",
            class_path="m.T2",
            tags=("c",),
        )

        data = catalog.to_dict()
        restored = ToolCatalog.from_dict(data)

        assert restored.count() == 2
        e1 = restored.get("t1")
        assert e1.class_path == "m.T1"
        assert e1.description == "First tool"
        assert e1.default_params == {"k": 5}
        assert e1.tags == frozenset({"a", "b"})
        assert e1.requires == frozenset({"r1"})

        e2 = restored.get("t2")
        assert e2.class_path == "m.T2"
        assert e2.tags == frozenset({"c"})
