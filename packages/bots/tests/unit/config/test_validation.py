"""Tests for config/validation.py."""

from __future__ import annotations

from typing import Any


from dataknobs_bots.config.schema import DynaBotConfigSchema
from dataknobs_bots.config.validation import ConfigValidator, ValidationResult


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_ok(self) -> None:
        result = ValidationResult.ok()
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_error(self) -> None:
        result = ValidationResult.error("something broke")
        assert result.valid is False
        assert result.errors == ["something broke"]
        assert result.warnings == []

    def test_warning(self) -> None:
        result = ValidationResult.warning("heads up")
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == ["heads up"]

    def test_merge_both_valid(self) -> None:
        a = ValidationResult.ok()
        b = ValidationResult.warning("note")
        merged = a.merge(b)
        assert merged.valid is True
        assert merged.warnings == ["note"]

    def test_merge_one_invalid(self) -> None:
        a = ValidationResult.error("bad")
        b = ValidationResult.ok()
        merged = a.merge(b)
        assert merged.valid is False
        assert merged.errors == ["bad"]

    def test_merge_keeps_every_message(self) -> None:
        """``merge`` concatenates; it does not decide what is a duplicate.

        Whether a repeated string is one finding or two is a property of
        the composition that produced it, and this primitive cannot see
        that. Around twenty call sites merge results, most of them
        accumulating findings from one validator, where dropping a
        repeat would drop a real finding.
        """
        a = ValidationResult(valid=False, errors=["missing llm"], warnings=["w"])
        b = ValidationResult(valid=False, errors=["missing llm"], warnings=["w"])

        merged = a.merge(b)

        assert merged.errors == ["missing llm", "missing llm"]
        assert merged.warnings == ["w", "w"]

    def test_merge_unique_deduplicates_identical_messages(self) -> None:
        """``merge_unique`` is for composing validators that overlap.

        ``ConfigValidator.validate`` and ``DynaBotConfigBuilder.validate``
        both run ``validate_completeness`` over the same config, so a
        caller running both gets every completeness failure twice. The
        second copy is an artefact of running two validators, not a
        second defect.
        """
        a = ValidationResult(valid=False, errors=["missing llm"], warnings=["w"])
        b = ValidationResult(valid=False, errors=["missing llm"], warnings=["w"])

        merged = a.merge_unique(b)

        assert merged.errors == ["missing llm"]
        assert merged.warnings == ["w"]

    def test_merge_unique_preserves_order_of_distinct_messages(self) -> None:
        """De-duplication must not reorder or drop distinct messages."""
        a = ValidationResult(valid=False, errors=["one", "two"])
        b = ValidationResult(valid=False, errors=["two", "three"])

        assert a.merge_unique(b).errors == ["one", "two", "three"]

    def test_merge_unique_keeps_validity_semantics(self) -> None:
        """Both operations agree on validity: invalid wins."""
        ok = ValidationResult.ok()
        bad = ValidationResult.error("nope")

        assert ok.merge_unique(bad).valid is False
        assert ok.merge_unique(ValidationResult.ok()).valid is True

    def test_merge_both_invalid(self) -> None:
        a = ValidationResult.error("err1")
        b = ValidationResult.error("err2")
        merged = a.merge(b)
        assert merged.valid is False
        assert merged.errors == ["err1", "err2"]

    def test_merge_accumulates(self) -> None:
        a = ValidationResult(valid=True, warnings=["w1"])
        b = ValidationResult(valid=False, errors=["e1"], warnings=["w2"])
        merged = a.merge(b)
        assert merged.valid is False
        assert merged.errors == ["e1"]
        assert merged.warnings == ["w1", "w2"]

    def test_to_dict(self) -> None:
        result = ValidationResult(valid=False, errors=["err"], warnings=["warn"])
        d = result.to_dict()
        assert d == {"valid": False, "errors": ["err"], "warnings": ["warn"]}


class TestConfigValidator:
    """Tests for ConfigValidator."""

    def test_validate_completeness_valid(self) -> None:
        validator = ConfigValidator()
        config: dict[str, Any] = {
            "llm": {"provider": "ollama"},
            "conversation_storage": {"backend": "memory"},
        }
        result = validator.validate_completeness(config)
        assert result.valid is True

    def test_validate_completeness_missing_llm(self) -> None:
        validator = ConfigValidator()
        config: dict[str, Any] = {
            "conversation_storage": {"backend": "memory"},
        }
        result = validator.validate_completeness(config)
        assert result.valid is False
        assert any("llm" in e for e in result.errors)

    def test_validate_completeness_missing_storage(self) -> None:
        validator = ConfigValidator()
        config: dict[str, Any] = {
            "llm": {"provider": "ollama"},
        }
        result = validator.validate_completeness(config)
        assert result.valid is False
        assert any("conversation_storage" in e for e in result.errors)

    def test_validate_completeness_missing_both(self) -> None:
        validator = ConfigValidator()
        result = validator.validate_completeness({})
        assert result.valid is False
        assert len(result.errors) == 2

    def test_validate_completeness_portable_format(self) -> None:
        validator = ConfigValidator()
        config: dict[str, Any] = {
            "bot": {
                "llm": {"$resource": "default"},
                "conversation_storage": {"$resource": "db"},
            },
        }
        result = validator.validate_completeness(config)
        assert result.valid is True

    def test_validate_portability_clean(self) -> None:
        validator = ConfigValidator()
        config: dict[str, Any] = {
            "llm": {"provider": "ollama", "model": "llama3.2"},
        }
        result = validator.validate_portability(config)
        assert result.valid is True

    def test_validate_portability_with_local_path(self) -> None:
        validator = ConfigValidator()
        config: dict[str, Any] = {
            "storage": {"path": "/Users/dev/data"},
        }
        result = validator.validate_portability(config)
        assert result.valid is True  # warnings, not errors
        assert len(result.warnings) > 0

    def test_register_custom_validator(self) -> None:
        validator = ConfigValidator()

        def check_name(config: dict[str, Any]) -> ValidationResult:
            if "name" not in config:
                return ValidationResult.warning("Config has no name")
            return ValidationResult.ok()

        validator.register_validator("name_check", check_name)
        config: dict[str, Any] = {
            "llm": {"provider": "ollama"},
            "conversation_storage": {"backend": "memory"},
        }
        result = validator.validate(config)
        assert result.valid is True
        assert any("name" in w for w in result.warnings)

    def test_validate_with_schema(self) -> None:
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        config: dict[str, Any] = {
            "llm": {"provider": "invalid_provider"},
            "conversation_storage": {"backend": "memory"},
        }
        result = validator.validate(config)
        assert result.valid is False
        assert any("invalid_provider" in e for e in result.errors)

    def test_validate_component(self) -> None:
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        result = validator.validate_component("llm", {"provider": "ollama", "model": "llama3.2"})
        assert result.valid is True

    def test_validate_component_invalid(self) -> None:
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        result = validator.validate_component("llm", {"provider": "not_a_provider"})
        assert result.valid is False

    def test_validator_exception_handling(self) -> None:
        validator = ConfigValidator()

        def bad_validator(config: dict[str, Any]) -> ValidationResult:
            raise RuntimeError("boom")

        validator.register_validator("bad", bad_validator)
        config: dict[str, Any] = {
            "llm": {"provider": "ollama"},
            "conversation_storage": {"backend": "memory"},
        }
        result = validator.validate(config)
        assert result.valid is False
        assert any("bad" in e for e in result.errors)


class TestResourceReferenceMarkers:
    """A ``$resource`` reference reaches schema validation before it resolves.

    A component section may be a reference rather than a literal config, so
    the validator skipped every ``$``-prefixed key wholesale. That let a
    misspelled marker past the one check that runs at config-lint time --
    ``$requred: true`` reads as *not required*, and it was left to fail at
    resolution, in whichever deployment happened to lack the resource.

    The marker set is exported by ``dataknobs-config`` precisely so a second
    reader can ask rather than copy the literals.
    """

    def test_a_reference_section_still_validates(self) -> None:
        """The skip exists for this: the keys are markers, not schema fields."""
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        result = validator.validate_component(
            "llm", {"$resource": "default", "type": "llm_providers"}
        )
        assert result.valid is True

    def test_every_marker_is_accepted(self) -> None:
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        result = validator.validate_component(
            "llm",
            {
                "$resource": "default",
                "type": "llm_providers",
                "$required": True,
                "$requires": ["streaming"],
            },
        )
        assert result.valid is True

    def test_a_misspelled_marker_is_reported(self) -> None:
        """It was skipped as a ``$``-key and deferred to resolution."""
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        result = validator.validate_component(
            "llm", {"$resource": "default", "type": "llm_providers", "$requred": True}
        )
        assert result.valid is False
        assert any("$requred" in e for e in result.errors)

    def test_a_dollar_key_outside_a_reference_is_left_alone(self) -> None:
        """Only a reference block has a closed vocabulary.

        An ordinary section is not a reference, and this validator is not the
        place to decide what ``$``-prefixed keys mean elsewhere in a config.
        """
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        result = validator.validate_component(
            "llm", {"provider": "ollama", "$custom": "passed through"}
        )
        assert result.valid is True


class TestMarkerRuleDepth:
    """The marker rule applies at every depth of a config, and to both halves.

    The validator used to carry a transcription of one clause of that rule,
    applied to a component's own top level. It agreed with the resolver about a
    reference section handed to it directly and disagreed about everything
    else: a reference nested inside one, a misspelled ``$resource`` selector,
    and a section no schema is registered for. A config the validator called
    valid then raised at resolution.
    """

    def test_a_nested_reference_marker_is_caught(self) -> None:
        """The gap the transcription had: it looked one level down, once."""
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        config: dict[str, Any] = {
            "bot": {
                "llm": {"provider": "ollama"},
                "conversation_storage": {"backend": "memory"},
                "knowledge_base": {
                    "vector_store": {
                        "$resource": "vectors",
                        "type": "vector_stores",
                        "$requred": True,
                    }
                },
            }
        }

        result = validator.validate(config)

        assert result.valid is False
        assert any("$requred" in e for e in result.errors)

    def test_a_misspelled_selector_is_caught(self) -> None:
        """``$resorce`` produces an ordinary dict that reaches a factory."""
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        config: dict[str, Any] = {
            "bot": {
                "llm": {"$resorce": "default", "type": "llm_providers", "$required": True},
                "conversation_storage": {"backend": "memory"},
            }
        }

        result = validator.validate(config)

        assert result.valid is False
        assert any("$required" in e for e in result.errors)

    def test_an_orphaned_policy_marker_is_caught(self) -> None:
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        config: dict[str, Any] = {
            "bot": {
                "llm": {"provider": "ollama"},
                "conversation_storage": {"backend": "memory", "$requires": ["persistence"]},
            }
        }

        result = validator.validate(config)

        assert result.valid is False
        assert any("$requires" in e for e in result.errors)

    def test_an_unregistered_section_is_checked(self) -> None:
        """Nothing looked at a section no schema is registered for."""
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        config: dict[str, Any] = {
            "bot": {
                "llm": {"provider": "ollama"},
                "conversation_storage": {"backend": "memory"},
                "custom_thing": {"$resource": "whatever", "$requred": True},
            }
        }

        result = validator.validate(config)

        assert result.valid is False
        assert any("$requred" in e for e in result.errors)

    def test_a_clean_config_is_still_valid(self) -> None:
        """The anti-vacuity half: a check that always fires checks nothing."""
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        config: dict[str, Any] = {
            "bot": {
                "llm": {"$resource": "default", "type": "llm_providers", "$required": True},
                "conversation_storage": {"backend": "memory"},
                "knowledge_base": {
                    "vector_store": {"$resource": "vectors", "type": "vector_stores"}
                },
            }
        }

        result = validator.validate(config)

        assert result.valid is True

    def test_a_violation_is_reported_once(self) -> None:
        """Two entry points cover this ground; only one may report it."""
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        config: dict[str, Any] = {
            "bot": {
                "llm": {"$resource": "default", "type": "llm_providers", "$requred": True},
                "conversation_storage": {"backend": "memory"},
            }
        }

        result = validator.validate(config)

        assert result.valid is False
        assert len([e for e in result.errors if "$requred" in e]) == 1


class TestMarkerRuleBreadth:
    """The rule applies to the whole config file, not to its ``bot`` section.

    Its sibling above covers depth. This covers breadth, and the two are the
    same argument: a ``$resource`` block under a key no schema is registered
    for is a block nothing else looks at. That was true of a key beside
    ``bot`` for as long as it was true of a key beneath one.

    The path a finding names is the same question asked from the other end.
    Walking the file rather than one section of it is what makes a finding
    say ``bot.knowledge_base.vector_store`` -- the path built by the walk is
    the path the reader has open, with no prefix to supply by hand.
    """

    def test_a_section_beside_bot_is_checked(self) -> None:
        """The narrowing's blind spot: a sibling of ``bot``, at any depth."""
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        config: dict[str, Any] = {
            "bot": {
                "llm": {"provider": "ollama"},
                "conversation_storage": {"backend": "memory"},
            },
            "domain": {
                "some_ref": {"$resource": "whatever", "$requred": True},
            },
        }

        result = validator.validate(config)

        assert result.valid is False
        assert any("$requred" in e for e in result.errors)
        assert any("domain.some_ref" in e for e in result.errors)

    def test_a_finding_names_the_path_the_reader_has_open(self) -> None:
        """``knowledge_base.vector_store`` locates nothing in a wrapped file."""
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        config: dict[str, Any] = {
            "bot": {
                "llm": {"provider": "ollama"},
                "conversation_storage": {"backend": "memory"},
                "knowledge_base": {
                    "vector_store": {
                        "$resource": "vectors",
                        "type": "vector_stores",
                        "$requred": True,
                    }
                },
            }
        }

        result = validator.validate(config)

        assert result.valid is False
        assert any("'bot.knowledge_base.vector_store'" in e for e in result.errors)

    def test_an_unwrapped_config_is_not_given_a_bot_prefix(self) -> None:
        """Green before this change -- it is the guard on a hardcoded prefix.

        ``path="bot"`` on the narrowed call would have fixed the wrapped
        shape by asserting a key this config does not have.
        """
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        config: dict[str, Any] = {
            "llm": {"provider": "ollama"},
            "conversation_storage": {"backend": "memory"},
            "knowledge_base": {
                "vector_store": {
                    "$resource": "vectors",
                    "type": "vector_stores",
                    "$requred": True,
                }
            },
        }

        result = validator.validate(config)

        assert result.valid is False
        assert any("'knowledge_base.vector_store'" in e for e in result.errors)
        assert not any("bot." in e for e in result.errors)

    def test_a_clean_sibling_section_is_left_alone(self) -> None:
        """Widening a check asks what it now reports, not only what it catches.

        Every shape below sits in a section the walk reaches for the first
        time, and none of them is a violation. The reference half needs a
        ``$resource`` in the block to fire at all, and the orphan half is
        closed to two literal keys -- so a JSON Schema's ``$ref`` is
        invisible to both, and a correctly spelled reference stays correct
        wherever in the file it is written.
        """
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)
        config: dict[str, Any] = {
            "bot": {
                "llm": {"provider": "ollama"},
                "conversation_storage": {"backend": "memory"},
            },
            "domain": {
                "id": "acme",
                "store": {
                    "$resource": "vectors",
                    "type": "vector_stores",
                    "$required": True,
                },
                "answer_schema": {"$schema": "...", "$ref": "#/$defs/answer"},
            },
        }

        result = validator.validate(config)

        assert result.valid is True
        assert not any("$" in m for m in result.errors + result.warnings)


class TestMarkerRuleOnASubtree:
    """``validate_component`` reports the same rule, rooted at the component.

    Deleting the transcription and routing the check only through the
    whole-config entry point would leave a consumer validating one component at
    a time with less than it had: the transcribed branch really did catch a
    misspelled marker on a component's own top level.
    """

    def test_validate_component_still_catches_a_top_level_marker(self) -> None:
        """Green before this change -- it is the guard on the deletion."""
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)

        result = validator.validate_component(
            "llm", {"$resource": "default", "type": "llm_providers", "$requred": True}
        )

        assert result.valid is False
        assert any("$requred" in e for e in result.errors)

    def test_validate_component_catches_a_nested_marker(self) -> None:
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)

        result = validator.validate_component(
            "knowledge_base",
            {"vector_store": {"$resource": "vectors", "type": "vector_stores", "$requred": True}},
        )

        assert result.valid is False
        assert any("$requred" in e for e in result.errors)

    def test_validate_component_names_the_component_in_the_path(self) -> None:
        """A message about ``vector_store`` alone would not locate anything."""
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)

        result = validator.validate_component(
            "knowledge_base",
            {"vector_store": {"$resource": "vectors", "type": "vector_stores", "$requred": True}},
        )

        assert any("knowledge_base.vector_store" in e for e in result.errors)

    def test_validate_component_catches_an_orphaned_marker(self) -> None:
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)

        result = validator.validate_component("llm", {"$resorce": "default", "$required": True})

        assert result.valid is False
