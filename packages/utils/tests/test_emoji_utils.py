"""Behavioural tests for the emoji-test.txt parser.

This module had none. It is ~330 lines that parse a Unicode data file into a
lookup table, and two `dataknobs_xization` modules read that table, so every
consumer of them inherited whatever the parser got wrong -- with nothing
asserting what it gets right.

The first test below is the reproduce-first case for a line the parser used to
swallow; the rest are the happy-path coverage whose absence is why nothing
noticed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dataknobs_utils.emoji_utils import (
    EmojiData,
    build_emoji_dataclass,
    get_emoji_seq,
)

#: A data line in emoji-test.txt format: codepoints, status, then a comment
#: carrying the rendered emoji, the version it arrived in, and its name.
GRINNING = "1F600 ; fully-qualified     # \U0001f600 E1.0 grinning face"
BIG_EYES = "1F603 ; fully-qualified     # \U0001f603 E0.6 grinning face with big eyes"
SMILING_EYES = "1F604 ; fully-qualified     # \U0001f604 E0.6 grinning face with smiling eyes"


def _write(tmp_path: Path, body: str) -> str:
    path = tmp_path / "emoji-test.txt"
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_a_data_line_directly_after_the_status_counts_is_still_loaded(tmp_path: Path) -> None:
    """Reproduce-first: the line that ends the status-count block was dropped.

    The block used to be read by a second ``for line in f`` nested inside the
    first. Both iterate the same handle, so the line that fails the ``# ``
    test -- the one that breaks the inner loop -- has already been consumed
    when the outer loop resumes. It is never classified, so if it is an emoji,
    that emoji is silently missing from the table.

    The shipped 15.0 file happens to put a blank line there, which is why this
    never showed up as a missing emoji. That is not a defence: what decided
    whether an emoji survived was a property of the input file that nothing
    stated and nothing checked, and the next revision of that file was under
    no obligation to keep it.
    """
    path = _write(
        tmp_path,
        "# group: Smileys & Emotion\n"
        "\n"
        "# subgroup: face-smiling\n"
        f"{GRINNING}\n"
        f"{BIG_EYES}\n"
        "\n"
        "# Status Counts\n"
        "# fully-qualified : 2\n"
        # No blank line here -- the terminator of the count block is real data.
        f"{SMILING_EYES}\n",
    )

    data = EmojiData(path)

    assert "\U0001f604" in data.emojis, (
        "the emoji on the line directly after the status-count block is "
        f"missing; the parser loaded {sorted(data.emojis)}"
    )


def test_status_counts_that_disagree_with_the_file_are_rejected(tmp_path: Path) -> None:
    """The count block is a self-check, and it has to still be one.

    Restructuring the block from a nested loop into a mode of the outer one
    keeps the same lines under the same comparison. If that comparison stopped
    happening the parser would go quiet rather than wrong, so it is asserted
    from the outside here rather than trusted.
    """
    path = _write(
        tmp_path,
        "# group: Smileys & Emotion\n"
        "# subgroup: face-smiling\n"
        f"{GRINNING}\n"
        "\n"
        "# Status Counts\n"
        "# fully-qualified : 99\n",
    )

    with pytest.raises(AssertionError):
        EmojiData(path)


def test_group_and_subgroup_are_carried_onto_each_emoji(tmp_path: Path) -> None:
    """The two header kinds are state, and they apply to what follows them."""
    path = _write(
        tmp_path,
        "# group: Smileys & Emotion\n"
        "# subgroup: face-smiling\n"
        f"{GRINNING}\n"
        "# group: People & Body\n"
        "# subgroup: hand-fingers-open\n"
        f"{BIG_EYES}\n",
    )

    data = EmojiData(path)

    assert data.emojis["\U0001f600"].group == "Smileys & Emotion"
    assert data.emojis["\U0001f600"].subgroup == "face-smiling"
    assert data.emojis["\U0001f603"].group == "People & Body"
    assert data.emojis["\U0001f603"].subgroup == "hand-fingers-open"


def test_emoji_bio_and_get_emojis_find_emojis_in_text(tmp_path: Path) -> None:
    """The two lookups every consumer of this module actually calls."""
    path = _write(
        tmp_path,
        f"# group: Smileys & Emotion\n# subgroup: face-smiling\n{GRINNING}\n{BIG_EYES}\n",
    )

    data = EmojiData(path)
    text = "hi \U0001f600 there \U0001f603"

    bio = data.emoji_bio(text)
    assert len(bio) == len(text), "the BIO tagging must be positionally aligned with the text"
    assert bio[text.index("\U0001f600")] == "B"

    found = [e.short_name for e in data.get_emojis(text)]
    assert found == ["grinning face", "grinning face with big eyes"]


def test_build_emoji_dataclass_parses_a_line_and_rejects_a_non_line() -> None:
    """The line parser, which decides what a data line even is."""
    parsed = build_emoji_dataclass(GRINNING)

    assert parsed is not None
    assert parsed.emoji == "\U0001f600"
    assert parsed.status == "fully-qualified"
    assert parsed.since_version == "E1.0"
    assert parsed.short_name == "grinning face"

    assert build_emoji_dataclass("# group: Smileys & Emotion") is None


def test_the_shipped_unicode_resource_parses_and_self_checks() -> None:
    """The real input, which is the only one production ever sees.

    Every case above is a hand-written file a few lines long, chosen to isolate
    one decision. None of them proves the parser survives the actual 5,000-line
    Unicode data file -- and that file carries its own status-count block, so
    loading it exercises the tallies against the counts its publisher wrote.
    """
    resource = Path(__file__).resolve().parents[3] / "resources" / "emoji-test.15.0.txt"
    if not resource.exists():  # pragma: no cover - the resource is committed
        pytest.skip(f"{resource} is not present")

    data = EmojiData(str(resource))

    # The file's own trailing block claims these four totals; loading it
    # asserts them internally, and repeating them here says what "it parsed"
    # is worth. A silent drop shows up as a shortfall in one of the four.
    tallies: dict[str, int] = {}
    for emoji in data.emojis.values():
        tallies[emoji.status] = tallies.get(emoji.status, 0) + 1

    assert tallies == {
        "fully-qualified": 3655,
        "minimally-qualified": 827,
        "unqualified": 242,
        "component": 9,
    }


def test_get_emoji_seq_reports_code_points_both_ways() -> None:
    """Used by callers deciding whether two renderings are the same sequence."""
    assert get_emoji_seq("\U0001f600") == [0x1F600]
    assert get_emoji_seq("\U0001f600", as_hex=True) == ["0x1f600"]
