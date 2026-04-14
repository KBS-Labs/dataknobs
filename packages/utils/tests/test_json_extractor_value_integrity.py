"""Tests for JSONExtractor value integrity — string-aware brace matching.

Covers scenarios where the old non-string-aware brace matching in
``_find_json_objects`` would corrupt or lose JSON values:

1. Baseline: valid JSON with multi-word values
2. Truncated JSON at various points within values
3. Surrounding text with stray braces (fallback scan)
4. String values containing brace/bracket characters
5. Multiple JSON fragments from LLM-style responses
6. _find_json_objects string-awareness regression tests
"""

from dataknobs_utils.json_extractor import JSONExtractor


# ──────────────────────────────────────────────────────────────────
# Category 1: Baseline — valid JSON with multi-word values
# ──────────────────────────────────────────────────────────────────


class TestBaselineMultiWordValues:
    """Confirm that well-formed JSON with multi-word values works."""

    def test_simple_multi_word_value(self):
        """Value 'formal and academic' in clean JSON."""
        extractor = JSONExtractor()
        text = '{"tone": "formal and academic"}'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal and academic"
        assert len(extractor.complete_jsons) == 1

    def test_multi_word_value_with_surrounding_text(self):
        """Value preserved when JSON is embedded in prose."""
        extractor = JSONExtractor()
        text = 'Here is the extraction: {"tone": "formal and academic"} based on the input.'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal and academic"

    def test_multiple_multi_word_values(self):
        """Multiple multi-word values in one object."""
        extractor = JSONExtractor()
        text = '{"tone": "formal and academic", "style": "clear and concise"}'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal and academic"
        assert result[0]["style"] == "clear and concise"

    def test_multi_word_value_with_commas(self):
        """Value with commas and conjunctions."""
        extractor = JSONExtractor()
        text = '{"description": "formal, academic, and professional"}'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["description"] == "formal, academic, and professional"

    def test_multi_word_value_in_markdown_code_block(self):
        """JSON in a markdown code block (common LLM output)."""
        extractor = JSONExtractor()
        text = '```json\n{"tone": "formal and academic"}\n```'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal and academic"


# ──────────────────────────────────────────────────────────────────
# Category 2: Truncated JSON at various points
# ──────────────────────────────────────────────────────────────────


class TestTruncatedJSON:
    """Test _fix_json behavior when JSON is truncated mid-value.

    These test the scenario where an LLM hits token limits and the
    response is cut off. The JSONExtractor attempts to repair these.
    """

    def test_truncated_after_first_word(self):
        """JSON cut after 'formal' — loses 'and academic'.

        This is a key scenario: the LLM intended 'formal and academic'
        but the response was truncated. _fix_json closes the string and
        brace, yielding {"tone": "formal"}.
        """
        extractor = JSONExtractor()
        text = '{"tone": "formal'
        result = extractor.extract_jsons(text)

        # Document actual behavior — _fix_json produces {"tone": "formal"}
        assert len(result) == 1
        # The value is truncated — this is the documented behavior
        assert result[0]["tone"] == "formal"
        assert len(extractor.fixed_jsons) == 1

    def test_truncated_mid_value_with_and(self):
        """JSON cut after 'formal and' — loses 'academic'."""
        extractor = JSONExtractor()
        text = '{"tone": "formal and'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal and"
        assert len(extractor.fixed_jsons) == 1

    def test_truncated_with_complete_value_no_closing_brace(self):
        """Full value present but missing closing brace."""
        extractor = JSONExtractor()
        text = '{"tone": "formal and academic"'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal and academic"
        assert len(extractor.fixed_jsons) == 1

    def test_truncated_with_complete_value_no_closing_quote(self):
        """Full value present but missing closing quote and brace."""
        extractor = JSONExtractor()
        text = '{"tone": "formal and academic'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal and academic"
        assert len(extractor.fixed_jsons) == 1

    def test_truncated_multi_field_after_first_value(self):
        """Truncated after first field — second field lost entirely."""
        extractor = JSONExtractor()
        text = '{"tone": "formal", "style": "academic and clear'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal"
        assert result[0]["style"] == "academic and clear"

    def test_truncated_multi_field_mid_second_value(self):
        """Truncated in the middle of the second field's value."""
        extractor = JSONExtractor()
        text = '{"tone": "formal and academic", "confidence": '
        result = extractor.extract_jsons(text)

        # Value after colon is empty — _fix_json closes braces but
        # json.loads rejects the missing value, so nothing parses.
        assert len(result) == 0


# ──────────────────────────────────────────────────────────────────
# Category 3: Surrounding text with stray braces
# ──────────────────────────────────────────────────────────────────


class TestSurroundingBraces:
    """Test behavior when surrounding text contains unmatched braces.

    Stray braces in surrounding prose can absorb the real JSON during
    the primary scan.  The fallback scan recovers by trying later '{'
    positions that look like JSON starts.
    """

    def test_preceding_text_with_matched_braces(self):
        """Matched braces in text before JSON — should find both."""
        extractor = JSONExtractor()
        text = 'The user wants {a formal tone}. Extraction: {"tone": "formal and academic"}'
        result = extractor.extract_jsons(text)

        # {a formal tone} fails json.loads, {"tone": "formal and academic"} succeeds
        valid_results = [r for r in result if "tone" in r]
        assert len(valid_results) == 1
        assert valid_results[0]["tone"] == "formal and academic"

    def test_preceding_text_with_unmatched_open_brace(self):
        """Unmatched { in text before JSON — fallback scan recovers it.

        The primary scan absorbs the stray { into the brace stack, so
        the real JSON never closes at depth 0.  The fallback scan tries
        later { positions and finds the real JSON.
        """
        extractor = JSONExtractor()
        text = 'The user wants {a formal tone. Extraction: {"tone": "formal and academic"}'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal and academic"
        assert len(extractor.complete_jsons) == 1

    def test_preceding_text_with_unmatched_close_brace(self):
        """Unmatched } in text before JSON."""
        extractor = JSONExtractor()
        text = 'The user wants a formal tone}. Extraction: {"tone": "formal and academic"}'
        result = extractor.extract_jsons(text)

        valid_results = [r for r in result if isinstance(r, dict) and "tone" in r]
        assert len(valid_results) == 1
        assert valid_results[0]["tone"] == "formal and academic"

    def test_prior_json_then_stray_brace_absorbing_target(self):
        """Valid JSON before a stray { that absorbs the target — fallback recovers both.

        The primary scan finds {"id": 1} and {stray {"tone": ...}} (braces
        balance). The second candidate fails json.loads. The fallback then
        scans the remaining non_json_text and recovers the inner object.
        """
        extractor = JSONExtractor()
        text = '{"id": 1} then {stray absorbed {"tone": "formal and academic"}}'
        result = extractor.extract_jsons(text)

        assert len(result) == 2
        assert result[0] == {"id": 1}
        assert result[1]["tone"] == "formal and academic"

    def test_text_with_braces_around_json(self):
        """Wrapping braces around the actual JSON (LLM sometimes does this).

        The outer {{...}} fails json.loads, so the fallback scan finds
        the inner {"tone": ...} object.
        """
        extractor = JSONExtractor()
        text = '{{"tone": "formal and academic"}}'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal and academic"

    def test_triple_braces_around_json(self):
        """Triple-wrapped braces — fallback peels through to find the object."""
        extractor = JSONExtractor()
        text = '{{{"tone": "formal and academic"}}}'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal and academic"


# ──────────────────────────────────────────────────────────────────
# Category 4: String values containing brace characters
# ──────────────────────────────────────────────────────────────────


class TestBracesInStringValues:
    """Test the core suspected bug: _find_json_objects doesn't track
    string context, so braces inside string values confuse brace matching.
    """

    def test_closing_brace_in_earlier_value(self):
        """A } inside a string value prematurely closes the JSON match.

        This is the most likely generalized bug: if ANY string value
        contains }, the brace matching terminates early, potentially
        truncating later values.
        """
        extractor = JSONExtractor()
        text = '{"note": "use a {formal} tone", "tone": "formal and academic"}'
        result = extractor.extract_jsons(text)

        # Without string awareness, _find_json_objects sees:
        # { at pos 0 → stack=["{"]
        # { at "formal}" → stack=["{", "{"]
        # } at "formal}" → stack=["{"]
        # } at "tone"," → stack=[] → extracts {"note": "use a {formal} tone"}
        #   But wait — the } after "tone" is at `, "tone"...` which isn't a }
        # Let me reconsider...
        # Actually "use a {formal} tone" contains { and }
        # { at pos 0 → stack=["{"]
        # { inside the string → stack=["{", "{"]
        # } inside the string "formal}" → stack=["{"]
        # next } would be at `tone", "tone"...` — no, the next } is the end
        # The structure: {"note": "use a {formal} tone", "tone": "formal and academic"}
        # Real } is at the very end
        # With the inner {} balanced, the outer {} would also balance correctly!
        # So this might actually work due to the inner braces being balanced.

        valid_results = [r for r in result if isinstance(r, dict) and "tone" in r]
        assert len(valid_results) == 1
        assert valid_results[0]["tone"] == "formal and academic"

    def test_unbalanced_brace_in_value_close_only(self):
        """A lone } in a string value must not cause premature close.

        Without string-awareness, the } inside "formal}" would close
        the brace stack, losing the "tone" field entirely.
        """
        extractor = JSONExtractor()
        text = '{"note": "formal}", "tone": "formal and academic"}'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal and academic"
        assert result[0]["note"] == "formal}"

    def test_unbalanced_brace_in_value_open_only(self):
        """A lone { in a string value must not deepen the brace stack."""
        extractor = JSONExtractor()
        text = '{"note": "see {details", "tone": "formal and academic"}'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal and academic"
        assert result[0]["note"] == "see {details"

    def test_closing_bracket_in_value(self):
        """A ] in a string value (less likely but similar class of bug)."""
        extractor = JSONExtractor()
        text = '{"items": ["formal]", "academic"], "tone": "formal and academic"}'
        result = extractor.extract_jsons(text)

        # Brackets don't affect _find_json_objects brace matching,
        # but they could affect _fix_json bracket counting
        valid_results = [r for r in result if isinstance(r, dict) and "tone" in r]
        assert len(valid_results) == 1
        assert valid_results[0]["tone"] == "formal and academic"

    def test_value_with_template_braces(self):
        """Template-style {name} braces in a string value."""
        extractor = JSONExtractor()
        text = '{"template": "Hello {name}", "tone": "formal and academic"}'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal and academic"
        assert result[0]["template"] == "Hello {name}"


# ──────────────────────────────────────────────────────────────────
# Category 5: LLM response patterns
# ──────────────────────────────────────────────────────────────────


class TestLLMResponsePatterns:
    """Test realistic LLM output formats that could trigger bugs."""

    def test_llm_explanation_then_json(self):
        """LLM explains before giving JSON (common pattern)."""
        extractor = JSONExtractor()
        text = (
            'Based on the user\'s message "Let\'s set the tone to formal and academic", '
            'I extracted the following:\n\n'
            '{"tone": "formal and academic"}'
        )
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal and academic"

    def test_llm_json_then_explanation(self):
        """LLM gives JSON then explains (also common)."""
        extractor = JSONExtractor()
        text = (
            '{"tone": "formal and academic"}\n\n'
            'I set the tone to "formal and academic" based on the user\'s request.'
        )
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal and academic"

    def test_llm_multiple_json_objects(self):
        """LLM returns multiple JSON objects (extraction + metadata)."""
        extractor = JSONExtractor()
        text = (
            '{"tone": "formal"}\n'
            '{"style": "academic"}'
        )
        result = extractor.extract_jsons(text)

        # Both objects extracted — but tone is "formal", not "formal and academic"
        # This is how an LLM might split the value across objects
        assert len(result) == 2
        assert result[0]["tone"] == "formal"
        assert result[1]["style"] == "academic"

    def test_llm_nested_extraction_result(self):
        """LLM wraps extraction in a result structure."""
        extractor = JSONExtractor()
        text = '{"result": {"tone": "formal and academic"}, "confidence": 0.95}'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["result"]["tone"] == "formal and academic"

    def test_llm_with_curly_brace_in_explanation(self):
        """LLM uses curly braces in explanatory text (e.g., template syntax)."""
        extractor = JSONExtractor()
        text = (
            'The user wants to set {tone} to a formal value.\n'
            '{"tone": "formal and academic"}'
        )
        result = extractor.extract_jsons(text)

        # {tone} has matched braces so it should be found and fail json.loads
        # Then {"tone": "formal and academic"} should be found separately
        valid_results = [r for r in result if isinstance(r, dict) and r.get("tone") == "formal and academic"]
        assert len(valid_results) == 1

    def test_llm_with_unmatched_brace_in_explanation(self):
        """LLM has stray { in explanation — fallback scan recovers JSON."""
        extractor = JSONExtractor()
        text = (
            'I see the user wants to configure {the writing tone.\n'
            '{"tone": "formal and academic"}'
        )
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal and academic"

    def test_llm_response_with_function_call_syntax(self):
        """LLM output with function-call-like braces in surrounding text."""
        extractor = JSONExtractor()
        text = (
            'set_tone({"tone": "formal and academic"})'
        )
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0]["tone"] == "formal and academic"


# ──────────────────────────────────────────────────────────────────
# Category 6: _find_json_objects string-awareness audit
# ──────────────────────────────────────────────────────────────────


class TestFindJsonObjectsStringAwareness:
    """Directly test _find_json_objects string-aware brace matching.

    Verifies that braces inside JSON string values do not confuse
    the brace depth tracker (the fix this branch introduces).
    """

    def test_find_objects_simple(self):
        """Baseline: simple object found correctly."""
        extractor = JSONExtractor()
        objects = extractor._find_json_objects('{"key": "value"}')
        assert len(objects) == 1
        assert objects[0] == ('{"key": "value"}', True)

    def test_find_objects_with_close_brace_in_string(self):
        """} in a string value should not close the outer object."""
        extractor = JSONExtractor()
        text = '{"a": "x}", "b": "y"}'
        objects = extractor._find_json_objects(text)

        # Correct behavior: find the complete object
        # Bug behavior: find {"a": "x"} (premature close)
        assert len(objects) >= 1
        # The full text should be the found object
        found_text = objects[0][0]
        assert found_text == text, (
            f"Expected full object but got premature close: {found_text!r}"
        )

    def test_find_objects_with_open_brace_in_string(self):
        """{ in a string value should not increase nesting depth."""
        extractor = JSONExtractor()
        text = '{"a": "x{", "b": "y"}'
        objects = extractor._find_json_objects(text)

        assert len(objects) >= 1
        found_text = objects[0][0]
        assert found_text == text, (
            f"Expected full object but got: {found_text!r}"
        )

    def test_find_objects_with_both_braces_in_string(self):
        """Both { and } in a string value — balanced but inside string."""
        extractor = JSONExtractor()
        text = '{"template": "Hello {name}!", "tone": "formal and academic"}'
        objects = extractor._find_json_objects(text)

        assert len(objects) >= 1
        found_text = objects[0][0]
        # If inner braces are balanced, the outer should still close correctly
        # even without string awareness (lucky coincidence)
        assert found_text == text, (
            f"Expected full object: {found_text!r}"
        )

    def test_find_objects_with_nested_json_in_string(self):
        """JSON-like content inside a string value."""
        extractor = JSONExtractor()
        text = '{"raw": "{\\"inner\\": \\"val\\"}", "tone": "formal"}'
        objects = extractor._find_json_objects(text)

        # The inner { and } should be ignored (they're in a string)
        # But without string awareness, they're counted
        assert len(objects) >= 1

    def test_find_objects_quotes_in_string_dont_toggle(self):
        """Escaped quotes in string values should not toggle string state."""
        extractor = JSONExtractor()
        text = '{"msg": "He said \\"hello\\"", "tone": "formal and academic"}'
        objects = extractor._find_json_objects(text)

        assert len(objects) >= 1
        found_text = objects[0][0]
        assert found_text == text, (
            f"Escaped quotes broke object detection: {found_text!r}"
        )
