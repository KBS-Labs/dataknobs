"""Tests for template composition and inheritance functionality."""

import pytest
from dataknobs_llm.prompts.utils import TemplateComposer
from dataknobs_llm.prompts.implementations import ConfigPromptLibrary


class TestBasicComposition:
    """Test basic section substitution without inheritance."""

    def test_simple_section_substitution(self):
        """Test replacing a single section placeholder."""
        composer = TemplateComposer()

        template = "Code: {{CODE_SECTION}}"
        sections = {"CODE_SECTION": "```python\nprint('hello')\n```"}

        result = composer.compose_template(template, sections)
        assert result == "Code: ```python\nprint('hello')\n```"

    def test_multiple_section_substitution(self):
        """Test replacing multiple section placeholders."""
        composer = TemplateComposer()

        template = "{{HEADER}}\n\n{{BODY}}\n\n{{FOOTER}}"
        sections = {
            "HEADER": "# Title",
            "BODY": "Content here",
            "FOOTER": "---\nEnd"
        }

        result = composer.compose_template(template, sections)
        expected = "# Title\n\nContent here\n\n---\nEnd"
        assert result == expected

    def test_section_with_spaces(self):
        """Test that {{ SECTION }} with spaces is also replaced."""
        composer = TemplateComposer()

        template = "{{ CODE_SECTION }}"
        sections = {"CODE_SECTION": "code content"}

        result = composer.compose_template(template, sections)
        assert result == "code content"

    def test_no_sections_returns_original(self):
        """Test that template without sections is returned unchanged."""
        composer = TemplateComposer()

        template = "No sections here"
        result = composer.compose_template(template, None)
        assert result == template

        result = composer.compose_template(template, {})
        assert result == template

    def test_missing_section_leaves_placeholder(self):
        """Test that undefined section placeholders are left as-is."""
        composer = TemplateComposer()

        template = "{{DEFINED}} and {{UNDEFINED}}"
        sections = {"DEFINED": "value"}

        result = composer.compose_template(template, sections)
        assert result == "value and {{UNDEFINED}}"

    def test_section_with_variables(self):
        """Test that sections can contain template variables."""
        composer = TemplateComposer()

        template = "{{CODE_SECTION}}"
        sections = {"CODE_SECTION": "```{{language}}\n{{code}}\n```"}

        result = composer.compose_template(template, sections)
        # Variables in sections should be preserved for later rendering
        assert result == "```{{language}}\n{{code}}\n```"

    def test_nested_section_placeholders(self):
        """Test that section content can reference other sections."""
        composer = TemplateComposer()

        template = "{{OUTER}}"
        sections = {
            "OUTER": "Start {{INNER}} End",
            "INNER": "middle"
        }

        # Our implementation replaces all sections in one pass
        # This is efficient and matches template variable resolution behavior
        result = composer.compose_template(template, sections)
        assert result == "Start middle End"


class TestConfigMerging:
    """Test configuration merging for inheritance."""

    def test_merge_sections(self):
        """Test merging section definitions."""
        composer = TemplateComposer()

        base = {"sections": {"A": "base A", "B": "base B"}}
        derived = {"sections": {"B": "derived B", "C": "derived C"}}

        merged = composer.merge_prompt_configs(base, derived)

        assert merged["sections"] == {
            "A": "base A",
            "B": "derived B",  # Overridden
            "C": "derived C"
        }

    def test_merge_defaults(self):
        """Test merging default parameter values."""
        composer = TemplateComposer()

        base = {"defaults": {"lang": "python", "style": "pep8"}}
        derived = {"defaults": {"style": "black", "format": "strict"}}

        merged = composer.merge_prompt_configs(base, derived)

        assert merged["defaults"] == {
            "lang": "python",
            "style": "black",  # Overridden
            "format": "strict"
        }

    def test_merge_validation(self):
        """Test that derived validation completely overrides base."""
        composer = TemplateComposer()

        base = {
            "validation": {
                "level": "error",
                "required_params": ["x", "y"]
            }
        }
        derived = {
            "validation": {
                "level": "warn",
                "required_params": ["z"]
            }
        }

        merged = composer.merge_prompt_configs(base, derived)

        # Derived validation completely replaces base
        assert merged["validation"] == {
            "level": "warn",
            "required_params": ["z"]
        }

    def test_merge_rag_config_refs(self):
        """Test merging RAG config references."""
        composer = TemplateComposer()

        base = {"rag_config_refs": ["base_search", "common"]}
        derived = {"rag_config_refs": ["derived_search", "common"]}

        merged = composer.merge_prompt_configs(base, derived)

        # Should combine and deduplicate
        assert merged["rag_config_refs"] == ["base_search", "common", "derived_search"]

    def test_merge_rag_configs(self):
        """Test merging inline RAG configurations."""
        composer = TemplateComposer()

        base = {
            "rag_configs": [
                {"adapter_name": "base", "query": "base query"}
            ]
        }
        derived = {
            "rag_configs": [
                {"adapter_name": "derived", "query": "derived query"}
            ]
        }

        merged = composer.merge_prompt_configs(base, derived)

        # Should append (both configs present)
        assert len(merged["rag_configs"]) == 2
        assert merged["rag_configs"][0]["adapter_name"] == "base"
        assert merged["rag_configs"][1]["adapter_name"] == "derived"

    def test_merge_template(self):
        """Test that derived template replaces base template."""
        composer = TemplateComposer()

        base = {"template": "Base template"}
        derived = {"template": "Derived template"}

        merged = composer.merge_prompt_configs(base, derived)

        assert merged["template"] == "Derived template"

    def test_merge_metadata(self):
        """Test merging metadata."""
        composer = TemplateComposer()

        base = {"metadata": {"author": "Alice", "version": "1.0"}}
        derived = {"metadata": {"version": "2.0", "tags": ["new"]}}

        merged = composer.merge_prompt_configs(base, derived)

        assert merged["metadata"] == {
            "author": "Alice",
            "version": "2.0",  # Overridden
            "tags": ["new"]
        }

    def test_merge_empty_configs(self):
        """Test merging when one or both configs are empty."""
        composer = TemplateComposer()

        # Empty base
        merged = composer.merge_prompt_configs({}, {"defaults": {"x": "y"}})
        assert merged["defaults"] == {"x": "y"}

        # Empty derived
        merged = composer.merge_prompt_configs({"defaults": {"x": "y"}}, {})
        assert merged["defaults"] == {"x": "y"}

        # Both empty
        merged = composer.merge_prompt_configs({}, {})
        assert merged == {}


class TestInheritance:
    """Test template inheritance with extends field."""

    def test_simple_inheritance(self):
        """Test basic template inheritance."""
        # Create library with base and derived templates
        config = {
            "system": {
                "base": {
                    "template": "{{CODE}}",
                    "sections": {"CODE": "```{{lang}}\n{{code}}\n```"},
                    "defaults": {"lang": "python"}
                },
                "derived": {
                    "extends": "base",
                    "sections": {"CODE": "```{{language}}\n{{code}}\n```"},
                    "defaults": {"language": "python"}
                }
            }
        }

        library = ConfigPromptLibrary(config)
        composer = TemplateComposer(library)

        # Get derived prompt
        derived = library.get_system_prompt("derived")

        # Get sections with inheritance
        sections = composer.get_sections_for_prompt("derived", derived)

        # Derived CODE section should override base
        assert "CODE" in sections
        assert "{{language}}" in sections["CODE"]  # Uses 'language' not 'lang'

    def test_multi_level_inheritance(self):
        """Test inheritance chain: grandparent -> parent -> child."""
        config = {
            "system": {
                "grandparent": {
                    "template": "{{A}} {{B}} {{C}}",
                    "sections": {
                        "A": "grandparent A",
                        "B": "grandparent B",
                        "C": "grandparent C"
                    }
                },
                "parent": {
                    "extends": "grandparent",
                    "sections": {
                        "B": "parent B"  # Override B
                    }
                },
                "child": {
                    "extends": "parent",
                    "sections": {
                        "C": "child C"  # Override C
                    }
                }
            }
        }

        library = ConfigPromptLibrary(config)
        composer = TemplateComposer(library)

        child = library.get_system_prompt("child")
        sections = composer.get_sections_for_prompt("child", child)

        # Child should have all sections with proper overrides
        assert sections["A"] == "grandparent A"  # Inherited from grandparent
        assert sections["B"] == "parent B"        # Inherited from parent
        assert sections["C"] == "child C"         # Child's own

    def test_resolve_full_inheritance(self):
        """Test resolve_inheritance method for full config resolution."""
        config = {
            "system": {
                "base": {
                    "template": "Base",
                    "defaults": {"x": "base_x", "y": "base_y"},
                    "sections": {"A": "base A"}
                },
                "derived": {
                    "extends": "base",
                    "template": "Derived",
                    "defaults": {"y": "derived_y", "z": "derived_z"},
                    "sections": {"B": "derived B"}
                }
            }
        }

        library = ConfigPromptLibrary(config)
        composer = TemplateComposer(library)

        derived = library.get_system_prompt("derived")
        resolved = composer.resolve_inheritance("derived", derived)

        # Should have merged configs
        assert resolved["template"] == "Derived"  # Overridden
        assert resolved["defaults"] == {
            "x": "base_x",      # From base
            "y": "derived_y",   # Overridden
            "z": "derived_z"    # From derived
        }
        assert resolved["sections"] == {
            "A": "base A",      # From base
            "B": "derived B"    # From derived
        }

    def test_circular_inheritance_detection(self):
        """Test that circular inheritance is detected and raises error."""
        config = {
            "system": {
                "a": {"extends": "b"},
                "b": {"extends": "c"},
                "c": {"extends": "a"}  # Circular!
            }
        }

        library = ConfigPromptLibrary(config)
        composer = TemplateComposer(library)

        a = library.get_system_prompt("a")

        with pytest.raises(ValueError, match="Circular inheritance"):
            composer.resolve_inheritance("a", a)

    def test_self_inheritance_detection(self):
        """Test that self-inheritance is detected."""
        config = {
            "system": {
                "self_ref": {"extends": "self_ref"}
            }
        }

        library = ConfigPromptLibrary(config)
        composer = TemplateComposer(library)

        prompt = library.get_system_prompt("self_ref")

        with pytest.raises(ValueError, match="Circular inheritance"):
            composer.get_sections_for_prompt("self_ref", prompt)

    def test_missing_base_template(self):
        """Test handling of missing base template."""
        config = {
            "system": {
                "derived": {"extends": "nonexistent"}
            }
        }

        library = ConfigPromptLibrary(config)
        composer = TemplateComposer(library)

        derived = library.get_system_prompt("derived")

        # Should log warning but not crash
        sections = composer.get_sections_for_prompt("derived", derived)
        assert sections == {}  # No sections since base doesn't exist

    def test_inheritance_without_library(self):
        """Test that inheritance fails gracefully without library."""
        composer = TemplateComposer()  # No library

        config = {"extends": "base", "sections": {"A": "derived A"}}

        # Should return only derived sections
        sections = composer.get_sections_for_prompt("test", config)
        assert sections == {"A": "derived A"}


class TestCaching:
    """Test caching functionality."""

    def test_template_composition_cache(self):
        """Test that composed templates are cached."""
        composer = TemplateComposer()

        template = "{{SECTION}}"
        sections = {"SECTION": "content"}

        # First call
        result1 = composer.compose_template(template, sections, prompt_name="test")

        # Second call should use cache (modify sections to verify)
        sections["SECTION"] = "modified"
        result2 = composer.compose_template(template, sections, prompt_name="test")

        # Should still have original cached result
        assert result1 == "content"
        assert result2 == "content"

    def test_cache_without_prompt_name(self):
        """Test that composition without prompt_name doesn't cache."""
        composer = TemplateComposer()

        template = "{{SECTION}}"
        sections = {"SECTION": "content"}

        # First call without name
        result1 = composer.compose_template(template, sections)

        # Modify sections
        sections["SECTION"] = "modified"

        # Second call should use new sections
        result2 = composer.compose_template(template, sections)

        assert result1 == "content"
        assert result2 == "modified"

    def test_config_resolution_cache(self):
        """Test that resolved configs are cached."""
        config = {
            "system": {
                "base": {"template": "Base {{x}}", "defaults": {"x": "base"}},
                "derived": {"extends": "base", "defaults": {"y": "derived"}}
            }
        }

        library = ConfigPromptLibrary(config)
        composer = TemplateComposer(library)

        derived = library.get_system_prompt("derived")

        # First resolution
        resolved1 = composer.resolve_inheritance("derived", derived)

        # Second resolution should return the same cached object
        resolved2 = composer.resolve_inheritance("derived", derived)

        # Verify caching - both should have same config
        assert resolved1["defaults"]["x"] == "base"
        assert resolved2["defaults"]["x"] == "base"
        assert resolved1["defaults"]["y"] == "derived"
        assert resolved2["defaults"]["y"] == "derived"

        # Verify they're using the cache (same computation result)
        assert id(resolved1) == id(resolved2)  # Same cached object

    def test_clear_cache(self):
        """Test that clear_cache clears all caches."""
        config = {
            "system": {
                "base": {"defaults": {"x": "base"}},
                "derived": {"extends": "base"}
            }
        }

        library = ConfigPromptLibrary(config)
        composer = TemplateComposer(library)

        # Populate caches
        template = "{{S}}"
        sections = {"S": "value"}
        composer.compose_template(template, sections, prompt_name="test")

        derived = library.get_system_prompt("derived")
        composer.resolve_inheritance("derived", derived)

        # Verify caches are populated
        assert len(composer._composition_cache) > 0
        assert len(composer._config_cache) > 0

        # Clear caches
        composer.clear_cache()

        # Verify caches are empty
        assert len(composer._composition_cache) == 0
        assert len(composer._config_cache) == 0


class TestEndToEnd:
    """Test complete composition workflows."""

    def test_compose_and_render(self):
        """Test composing a template then rendering it."""
        config = {
            "system": {
                "code_analysis": {
                    "template": "{{HEADER}}\n\n{{CODE}}\n\n{{INSTRUCTIONS}}",
                    "sections": {
                        "HEADER": "# Code Analysis",
                        "CODE": "```{{language}}\n{{code}}\n```",
                        "INSTRUCTIONS": "Analyze the code above."
                    },
                    "defaults": {"language": "python"}
                }
            }
        }

        library = ConfigPromptLibrary(config)
        composer = TemplateComposer(library)

        # Get prompt
        prompt = library.get_system_prompt("code_analysis")

        # Get sections
        sections = composer.get_sections_for_prompt("code_analysis", prompt)

        # Compose template with sections
        template = prompt["template"]
        composed = composer.compose_template(template, sections)

        # The composed template should have sections expanded
        assert "# Code Analysis" in composed
        assert "```{{language}}" in composed  # Variables still there
        assert "Analyze the code above" in composed

    def test_inherited_composition(self):
        """Test composition with inheritance."""
        config = {
            "system": {
                "base_analysis": {
                    "template": "{{HEADER}}\n{{BODY}}",
                    "sections": {
                        "HEADER": "# Analysis",
                        "BODY": "Generic analysis"
                    }
                },
                "security_analysis": {
                    "extends": "base_analysis",
                    "sections": {
                        "HEADER": "# Security Analysis",
                        "BODY": "Security-focused analysis"
                    }
                }
            }
        }

        library = ConfigPromptLibrary(config)
        composer = TemplateComposer(library)

        # Get derived prompt
        security = library.get_system_prompt("security_analysis")

        # Resolve inheritance
        resolved = composer.resolve_inheritance("security_analysis", security)

        # Get sections (should have overrides)
        sections = composer.get_sections_for_prompt("security_analysis", security)

        # Compose
        composed = composer.compose_template(resolved["template"], sections)

        # Should have security-specific content
        assert "# Security Analysis" in composed
        assert "Security-focused analysis" in composed

    def test_complex_inheritance_chain(self):
        """Test complex inheritance with multiple levels and overrides."""
        config = {
            "system": {
                "base": {
                    "template": "{{A}}\n{{B}}\n{{C}}",
                    "sections": {
                        "A": "Base A",
                        "B": "Base B",
                        "C": "Base C"
                    },
                    "defaults": {"x": "1", "y": "2", "z": "3"}
                },
                "middle": {
                    "extends": "base",
                    "sections": {"B": "Middle B"},
                    "defaults": {"y": "20"}
                },
                "final": {
                    "extends": "middle",
                    "sections": {"C": "Final C"},
                    "defaults": {"z": "300"}
                }
            }
        }

        library = ConfigPromptLibrary(config)
        composer = TemplateComposer(library)

        # Get final prompt
        final = library.get_system_prompt("final")

        # Resolve full inheritance
        resolved = composer.resolve_inheritance("final", final)

        # Check merged defaults
        assert resolved["defaults"] == {
            "x": "1",    # From base
            "y": "20",   # From middle
            "z": "300"   # From final
        }

        # Check merged sections
        assert resolved["sections"]["A"] == "Base A"    # From base
        assert resolved["sections"]["B"] == "Middle B"  # From middle
        assert resolved["sections"]["C"] == "Final C"   # From final

        # Compose
        composed = composer.compose_template(
            resolved["template"],
            resolved["sections"]
        )

        assert "Base A" in composed
        assert "Middle B" in composed
        assert "Final C" in composed
