"""Tests for the read-time history-redaction primitive.

Covers the canonical ``HistoryRedaction`` dataclass, the
``_compile_history_redactions`` harvester, the generic
``apply_history_redactions`` helper (driven by the accessor trio), and the
dict-shape convenience wrapper ``apply_history_redactions_to_dicts`` that the
``dataknobs-bots`` memory backends call.

The structural-symmetry test guards the core design claim: dict-shaped
(memory) and ``LLMMessage``-shaped (middleware) callers drive ONE
implementation and produce identical content output.
"""

import re

import pytest

from dataknobs_llm.conversations.history_redaction import (
    HistoryRedaction,
    _compile_history_redactions,
    apply_history_redactions,
    apply_history_redactions_to_dicts,
)
from dataknobs_llm.llm import LLMMessage


class TestHistoryRedactionConfig:
    """Tests for the ``HistoryRedaction`` config dataclass itself.

    These guard against footguns that surface as content corruption rather
    than failures, which is why they live next to the read-time tests.
    """

    def test_history_redaction_empty_pattern_raises_valueerror(self) -> None:
        """An empty regex matches every position — combined with any non-empty
        replacement it shreds message content. Reject at config-load.

        Also covers the no-arg ``HistoryRedaction()`` case (empty pattern
        default), which is unsafe for the same reason.
        """
        with pytest.raises(ValueError, match="pattern"):
            HistoryRedaction(pattern="", replacement="X")
        with pytest.raises(ValueError, match="pattern"):
            HistoryRedaction()

    def test_history_redaction_invalid_regex_raises_at_construction(self) -> None:
        """Invalid regex surfaces at config-load time, not consumer build time."""
        with pytest.raises(re.error):
            HistoryRedaction(pattern="(unclosed", replacement="")

    def test_history_redaction_valid_construction_round_trips(self) -> None:
        """A well-formed pattern + replacement constructs and round-trips."""
        r = HistoryRedaction(pattern=r"\bbib:\d+\b", replacement="[prior citation]")
        assert r.pattern == r"\bbib:\d+\b"
        assert r.replacement == "[prior citation]"
        assert HistoryRedaction.from_dict(r.to_dict()) == r

    def test_compile_history_redactions_reuses_cached_pattern(self) -> None:
        """``_compile_history_redactions`` harvests the cached compiled pattern
        rather than recompiling — the eager ``__post_init__`` compile is reused.
        """
        r = HistoryRedaction(pattern=r"\bbib:\d+\b", replacement="[x]")
        out1 = _compile_history_redactions([r])
        out2 = _compile_history_redactions([r])
        assert out1[0][0] is out2[0][0]
        # And it is the same object stashed on the dataclass.
        assert out1[0][0] is r._compiled_pattern


class TestApplyHistoryRedactionsToDicts:
    """Tests for the dict-shape wrapper that powers the memory-backend path."""

    def test_dict_accessor_path_coerces_none_content(self) -> None:
        """``{"content": None}`` (and a missing ``content`` key) must not crash —
        tool-call assistant messages legitimately carry no content. Coerce to
        empty string before regex sub.
        """
        patterns = [(re.compile(r"\bbib:\d+\b"), "[x]")]

        none_content = apply_history_redactions_to_dicts(
            [{"role": "assistant", "content": None}], patterns
        )
        assert len(none_content) == 1
        assert none_content[0]["role"] == "assistant"

        missing_key = apply_history_redactions_to_dicts(
            [{"role": "assistant"}], patterns
        )
        assert len(missing_key) == 1
        assert missing_key[0]["role"] == "assistant"

    def test_dict_accessor_path_redacts_assistant_only(self) -> None:
        """Assistant content is rewritten; system + user pass through.

        Redacted elements are new dict objects; original input dicts are not
        mutated.
        """
        patterns = _compile_history_redactions(
            [
                HistoryRedaction(
                    pattern=r"\[bib:\d+[^\]]*\]", replacement="[prior citation]"
                ),
                HistoryRedaction(
                    pattern=r"\bbib:\d+\b", replacement="[prior citation]"
                ),
            ]
        )
        system = {"role": "system", "content": "bib:5 stays"}
        user = {"role": "user", "content": "What about bib:5?"}
        assistant = {
            "role": "assistant",
            "content": "Cited [bib:5 · vendor] and bib:3.",
        }

        out = apply_history_redactions_to_dicts([system, user, assistant], patterns)

        assert out[0]["content"] == "bib:5 stays"
        assert out[1]["content"] == "What about bib:5?"
        assert out[2]["content"] == "Cited [prior citation] and [prior citation]."
        # Redacted element is a fresh dict; original unmutated.
        assert out[2] is not assistant
        assert assistant["content"] == "Cited [bib:5 · vendor] and bib:3."

    def test_dict_accessor_path_passes_through_by_identity_for_non_redacted_roles(
        self,
    ) -> None:
        """Non-redacted-role elements pass through by IDENTITY.

        Regression guard for the helper's contract — a behavior change from
        the original dict-only helper, which shallow-copied every message.
        """
        patterns = [(re.compile(r"\bbib:\d+\b"), "[x]")]
        user = {"role": "user", "content": "hi"}
        assistant = {"role": "assistant", "content": "cited bib:5"}

        out = apply_history_redactions_to_dicts([user, assistant], patterns)

        assert out[0] is user
        assert out[1] is not assistant

    def test_dict_accessor_path_empty_redactions_is_passthrough(self) -> None:
        """Empty redactions ⇒ output equals ``list(input)``; no rewriting."""
        user = {"role": "user", "content": "hi"}
        assistant = {"role": "assistant", "content": "cited bib:5"}

        out = apply_history_redactions_to_dicts([user, assistant], [])

        assert out == [user, assistant]
        assert out[1]["content"] == "cited bib:5"

    def test_dict_accessor_path_preserves_pattern_order(self) -> None:
        """Bracketed-header pattern MUST run before the bare-token pattern.

        If the bare-token rule ran first it would consume ``bib:N`` inside the
        bracket and leave a malformed ``[ · vendor · …]`` header.
        """
        patterns = _compile_history_redactions(
            [
                HistoryRedaction(
                    pattern=r"\[bib:\d+[^\]]*\]", replacement="[prior citation]"
                ),
                HistoryRedaction(
                    pattern=r"\bbib:\d+\b", replacement="[prior citation]"
                ),
            ]
        )
        messages = [
            {"role": "assistant", "content": "See [bib:5 · vendor] and also bib:3."},
        ]

        out = apply_history_redactions_to_dicts(messages, patterns)

        assert out[0]["content"] == "See [prior citation] and also [prior citation]."

    def test_redact_roles_override_extends_to_tool_role(self) -> None:
        """``redact_roles`` override rewrites the named roles; others pass through."""
        patterns = [(re.compile(r"\bbib:\d+\b"), "[x]")]
        messages = [
            {"role": "system", "content": "bib:1"},
            {"role": "user", "content": "bib:2"},
            {"role": "assistant", "content": "bib:3"},
            {"role": "tool", "content": "bib:4"},
        ]

        out = apply_history_redactions_to_dicts(
            messages, patterns, redact_roles=frozenset({"assistant", "tool"})
        )

        assert out[0]["content"] == "bib:1"  # system passthrough
        assert out[1]["content"] == "bib:2"  # user passthrough
        assert out[2]["content"] == "[x]"  # assistant redacted
        assert out[3]["content"] == "[x]"  # tool redacted


class TestGenericHelperSymmetry:
    """Tests that the generic helper drives both element shapes identically."""

    def test_generic_helper_drives_both_shapes_to_same_output(self) -> None:
        """Same source content as both an ``LLMMessage`` list and a dict list,
        run through the generic helper with each shape's accessor trio, produces
        identical redacted content (the structural-symmetry guard).
        """
        patterns = _compile_history_redactions(
            [
                HistoryRedaction(
                    pattern=r"\[bib:\d+[^\]]*\]", replacement="[prior citation]"
                ),
                HistoryRedaction(
                    pattern=r"\bbib:\d+\b", replacement="[prior citation]"
                ),
            ]
        )
        roles_and_content = [
            ("system", "bib:5 stays"),
            ("user", "What about bib:5?"),
            ("assistant", "Cited [bib:5 · vendor] and bib:3."),
        ]

        llm_messages = [LLMMessage(role=r, content=c) for r, c in roles_and_content]
        dict_messages = [{"role": r, "content": c} for r, c in roles_and_content]

        llm_out = apply_history_redactions(
            llm_messages,
            patterns,
            role_of=lambda m: m.role,
            content_of=lambda m: m.content,
            replace_content=lambda m, c: m.__class__(role=m.role, content=c),
        )
        dict_out = apply_history_redactions_to_dicts(dict_messages, patterns)

        llm_contents = [m.content for m in llm_out]
        dict_contents = [m["content"] for m in dict_out]
        assert llm_contents == dict_contents

    def test_generic_helper_does_not_mutate_input_elements(self) -> None:
        """Input elements (both shapes) are untouched after the call."""
        patterns = [(re.compile(r"\bbib:\d+\b"), "[x]")]

        llm_msg = LLMMessage(role="assistant", content="cited bib:5")
        dict_msg = {"role": "assistant", "content": "cited bib:5"}

        apply_history_redactions(
            [llm_msg],
            patterns,
            role_of=lambda m: m.role,
            content_of=lambda m: m.content,
            replace_content=lambda m, c: m.__class__(role=m.role, content=c),
        )
        apply_history_redactions_to_dicts([dict_msg], patterns)

        assert llm_msg.content == "cited bib:5"
        assert dict_msg["content"] == "cited bib:5"
