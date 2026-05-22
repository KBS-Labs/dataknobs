"""StructuredConfig migration coverage for the ``config/`` dataclasses.

Each migrated dataclass is a
:class:`~dataknobs_common.structured_config.StructuredConfig` subclass:
``from_dict`` is inherited (recursing into nested sub-configs), while the
bespoke serialization contracts (omit-empty output, the ``id`` key rename,
frozenset/tuple coercion) are preserved via overridden ``to_dict`` /
``__post_init`` / ``_normalize_dict``. These tests pin both halves:
field-projection parity and the round-trip property, including the nested
shapes that the inherited ``from_dict`` now rebuilds for free.
"""

from __future__ import annotations

from dataknobs_common.structured_config import StructuredConfig
from dataknobs_common.testing import assert_structured_config_roundtrip

from dataknobs_bots.config.drafts import DraftMetadata
from dataknobs_bots.config.schema import ComponentSchema
from dataknobs_bots.config.templates import ConfigTemplate, TemplateVariable
from dataknobs_bots.config.tool_catalog import ToolEntry
from dataknobs_bots.config.versioning import ConfigVersion
from dataknobs_bots.config.wizard_builder import (
    ContextGenerationConfig,
    IntentDetectionConfig,
    StageConfig,
    TransitionConfig,
    WizardConfig,
)

_MIGRATED = [
    ToolEntry,
    TemplateVariable,
    ConfigTemplate,
    ConfigVersion,
    DraftMetadata,
    ComponentSchema,
    TransitionConfig,
    IntentDetectionConfig,
    ContextGenerationConfig,
    StageConfig,
    WizardConfig,
]


def test_all_migrated_classes_are_structured_configs() -> None:
    for cls in _MIGRATED:
        assert issubclass(cls, StructuredConfig)


# ── ToolEntry: frozenset coercion + omit-empty/sorted output ──────────
class TestToolEntry:
    def test_from_dict_coerces_sets_and_fills_params(self) -> None:
        entry = ToolEntry.from_dict(
            {
                "name": "search",
                "class_path": "pkg.Search",
                "tags": ["b", "a"],
                "requires": ["knowledge_base"],
            }
        )
        assert entry.tags == frozenset({"a", "b"})
        assert entry.requires == frozenset({"knowledge_base"})
        assert entry.default_params == {}

    def test_to_dict_omits_defaults_and_sorts_sets(self) -> None:
        entry = ToolEntry(name="search", class_path="pkg.Search")
        # Empty description/params/tags/requires are omitted.
        assert entry.to_dict() == {
            "name": "search",
            "class_path": "pkg.Search",
        }
        full = ToolEntry(
            name="search",
            class_path="pkg.Search",
            description="Find things",
            default_params={"k": 5},
            tags=frozenset({"rag", "general"}),
            requires=frozenset({"knowledge_base"}),
        )
        d = full.to_dict()
        assert d["tags"] == ["general", "rag"]  # sorted list, not a set
        assert d["requires"] == ["knowledge_base"]

    def test_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            ToolEntry(
                name="search",
                class_path="pkg.Search",
                description="d",
                default_params={"k": 5},
                tags=frozenset({"a", "b"}),
                requires=frozenset({"kb"}),
            )
        )


# ── TemplateVariable / ConfigTemplate: nested variables rebuilt ───────
class TestTemplates:
    def test_template_variable_from_dict_parity(self) -> None:
        tv = TemplateVariable.from_dict(
            {"name": "domain_id", "type": "string", "required": True}
        )
        assert tv == TemplateVariable(
            name="domain_id", type="string", required=True
        )

    def test_template_variable_to_dict_omits_empty(self) -> None:
        tv = TemplateVariable(name="x")
        assert tv.to_dict() == {"name": "x", "type": "string", "required": False}

    def test_config_template_rebuilds_nested_variables(self) -> None:
        tmpl = ConfigTemplate.from_dict(
            {
                "name": "tutor",
                "variables": [
                    {"name": "domain_id", "required": True},
                    {"name": "level", "default": "beginner"},
                ],
                "structure": {"llm": {"provider": "ollama"}},
            }
        )
        assert all(isinstance(v, TemplateVariable) for v in tmpl.variables)
        assert tmpl.variables[0].name == "domain_id"
        assert tmpl.variables[1].default == "beginner"

    def test_config_template_to_dict_uses_variable_to_dict(self) -> None:
        tmpl = ConfigTemplate(
            name="tutor", variables=[TemplateVariable(name="x")]
        )
        # Nested variable serialized via its own omit-empty to_dict.
        assert tmpl.to_dict()["variables"] == [
            {"name": "x", "type": "string", "required": False}
        ]

    def test_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            ConfigTemplate(
                name="tutor",
                description="A tutor template",
                version="2.0.0",
                tags=["educational"],
                variables=[
                    TemplateVariable(name="domain_id", required=True),
                    TemplateVariable(
                        name="level", default="beginner", choices=["a", "b"]
                    ),
                ],
                structure={"llm": {"provider": "ollama"}},
            )
        )


# ── ConfigVersion: version-only identity preserved ────────────────────
class TestConfigVersion:
    def test_from_dict_parity(self) -> None:
        v = ConfigVersion.from_dict(
            {"version": 2, "config": {"a": 1}, "reason": "upgrade"}
        )
        assert v.version == 2
        assert v.config == {"a": 1}
        assert v.reason == "upgrade"

    def test_identity_is_version_only(self) -> None:
        a = ConfigVersion(version=1, config={"x": 1})
        b = ConfigVersion(version=1, config={"x": 999})
        assert a == b  # same version number, different config
        assert hash(a) == hash(b)

    def test_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            ConfigVersion(
                version=3,
                config={"name": "Bot"},
                reason="r",
                previous_version=2,
                created_by="tester",
                metadata={"k": "v"},
            )
        )


# ── DraftMetadata: id<->draft_id key rename + omit-empty ──────────────
class TestDraftMetadata:
    def test_id_key_maps_to_draft_id(self) -> None:
        meta = DraftMetadata.from_dict(
            {"id": "abc123", "created_at": "t0", "last_updated": "t1"}
        )
        assert meta.draft_id == "abc123"

    def test_to_dict_emits_id_and_omits_unset(self) -> None:
        meta = DraftMetadata(
            draft_id="abc123", created_at="t0", last_updated="t1"
        )
        assert meta.to_dict() == {
            "id": "abc123",
            "created_at": "t0",
            "last_updated": "t1",
            "complete": False,
        }

    def test_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            DraftMetadata(
                draft_id="abc123",
                created_at="t0",
                last_updated="t1",
                stage="configure_llm",
                complete=True,
                config_name="my-bot",
            )
        )


# ── ComponentSchema: gains from_dict/to_dict ──────────────────────────
class TestComponentSchema:
    def test_roundtrip_and_methods(self) -> None:
        comp = ComponentSchema(
            name="llm",
            description="LLM provider configuration",
            schema={"properties": {"provider": {"enum": ["ollama", "openai"]}}},
            required=True,
        )
        assert comp.get_valid_options("provider") == ["ollama", "openai"]
        assert_structured_config_roundtrip(comp)
        rebuilt = ComponentSchema.from_dict(comp.to_dict())
        assert rebuilt == comp


# ── Wizard configs: the nested-rebuild win ────────────────────────────
class TestWizardConfigs:
    def test_transition_to_dict_omits_none(self) -> None:
        t = TransitionConfig(target="done", condition="data.get('x')")
        assert t.to_dict() == {
            "target": "done",
            "condition": "data.get('x')",
        }

    def test_intent_detection_coerces_intents_to_tuple(self) -> None:
        intent = IntentDetectionConfig.from_dict(
            {"method": "keyword", "intents": [{"name": "quiz"}]}
        )
        assert isinstance(intent.intents, tuple)
        assert intent.intents == ({"name": "quiz"},)

    def test_intent_detection_method_defaults_to_keyword(self) -> None:
        assert IntentDetectionConfig.from_dict({}).method == "keyword"

    def test_stage_config_rebuilds_nested_and_coerces_tuples(self) -> None:
        stage = StageConfig.from_dict(
            {
                "name": "gather",
                "prompt": "Tell me.",
                "suggestions": ["A", "B"],
                "tools": ["knowledge_search"],
                "transitions": [
                    {"target": "done", "condition": "data.get('name')"}
                ],
                "intent_detection": {
                    "method": "keyword",
                    "intents": [{"name": "quiz"}],
                },
                "context_generation": {"variables": {"summary": "..."}},
            }
        )
        # Primitive tuple fields coerced from list.
        assert stage.suggestions == ("A", "B")
        assert stage.tools == ("knowledge_search",)
        # Nested dataclasses rebuilt from raw dicts.
        assert isinstance(stage.transitions[0], TransitionConfig)
        assert stage.transitions[0].target == "done"
        assert isinstance(stage.intent_detection, IntentDetectionConfig)
        assert isinstance(stage.context_generation, ContextGenerationConfig)
        assert stage.context_generation.variables == {"summary": "..."}

    def test_stage_config_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            StageConfig(
                name="gather",
                prompt="Tell me.",
                is_start=True,
                suggestions=("A", "B"),
                tools=("knowledge_search",),
                transitions=(
                    TransitionConfig(target="done", condition="data.get('n')"),
                ),
                intent_detection=IntentDetectionConfig(
                    method="keyword", intents=({"name": "quiz"},)
                ),
                context_generation=ContextGenerationConfig(
                    variables={"summary": "..."}
                ),
            )
        )

    def test_wizard_config_rebuilds_nested_stages(self) -> None:
        wiz = WizardConfig.from_dict(
            {
                "name": "onboarding",
                "version": "1.0",
                "stages": [
                    {"name": "welcome", "prompt": "Hi", "is_start": True},
                    {"name": "done", "prompt": "Bye", "is_end": True},
                ],
            }
        )
        assert isinstance(wiz.stages, tuple)
        assert all(isinstance(s, StageConfig) for s in wiz.stages)
        assert wiz.stages[0].name == "welcome"

    def test_wizard_config_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            WizardConfig(
                name="onboarding",
                version="1.0",
                description="Onboard a user",
                settings={"tool_reasoning": "react"},
                stages=(
                    StageConfig(name="welcome", prompt="Hi", is_start=True),
                    StageConfig(name="done", prompt="Bye", is_end=True),
                ),
                global_tasks=({"type": "log"},),
            )
        )
