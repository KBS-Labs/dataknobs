"""Utilities for working with unicode emojis.

References:
* https://home.unicode.org/emoji/about-emoji/
   * https://unicode.org/emoji/techindex.html
      * https://www.unicode.org/Public/emoji/15.0/
         * https://www.unicode.org/Public/emoji/15.0/emoji-test.txt


Emoji basics:

* Emojis are represented by one ore more characters
* Compound emojis are built using a "zero width joiner" of U+200D
   * This distinguishes a complex compound emoji from an adjacent emoji
   * That can be displayed as a single emoji
   * Such sequences are called "ZWJ" sequences
* An optional "variation selector 16", U+FE0F, can follow an emoji's chars
   * This indicates to render the emoji with its variation(s)

Version and Updates:

* Current unicode version is 15.0
   * Data collection is from emoji-test.txt
      * Watch out for format changes in that file if/when updating the version

Usage:

* Download the desired version's emoji-test.txt resource.
* Create an EmojiData instance with the path to the resource
* Use EmojiData to
   * Mark emoji locations (BIO) in text
   * Extract emojis from text
   * Lookup, browse, investigate emojis with their metadata (Emoji dataclass)
"""

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Union

LATEST_EMOJI_DATA = "resources/emoji-test.15.0.txt"


ETESTLINE_RE = re.compile(r"^([0-9A-F ]+); ([\w\-]+)\s+#.+(E\d+\.\d+) (.*)$")
STATUS_COUNT_RE = re.compile(r"^# ([\w\-]+)\s+:\s+(\d+).*$")


ZERO_WIDTH_JOINER = chr(int("0x200D", 16))
VARIATION_SELECTOR_16 = chr(int("0xFE0F", 16))

SPECIAL_CHARS = {
    ZERO_WIDTH_JOINER,
    VARIATION_SELECTOR_16,
}


@dataclass
class Emoji:
    """Metadata for a Unicode emoji from the emoji-test.txt file.

    Attributes:
        emoji: The emoji character(s) as a string.
        status: Qualification status (e.g., 'fully-qualified', 'minimally-qualified',
            'unqualified', 'component').
        since_version: Unicode version when emoji was introduced (e.g., 'E13.0').
        short_name: Short English name describing the emoji.
        group: Optional emoji group category from test file. Defaults to None.
        subgroup: Optional emoji subgroup category from test file. Defaults to None.
    """

    emoji: str
    status: str
    since_version: str
    short_name: str
    group: str | None = None
    subgroup: str | None = None


def build_emoji_dataclass(emoji_test_line: str) -> Emoji | None:
    """Parse an emoji-test.txt file line into an Emoji dataclass.

    Parses lines matching the emoji-test.txt format to extract emoji metadata.
    Lines not matching the expected format are ignored.

    Args:
        emoji_test_line: Single line from emoji-test.txt file.

    Returns:
        Emoji | None: Parsed emoji metadata, or None if line doesn't match
            expected format.
    """
    result = None
    m = ETESTLINE_RE.match(emoji_test_line)
    if m:
        result = Emoji(
            "".join(chr(int(x, 16)) for x in m.group(1).split()),
            m.group(2),
            m.group(3),
            m.group(4).strip(),
        )
    return result


def get_emoji_seq(emoji_str: str, as_hex: bool = False) -> List[Union[int, str]]:
    """Convert emoji string to sequence of code points.

    Args:
        emoji_str: Emoji string to convert.
        as_hex: If True, returns hex strings; if False, returns integers.
            Defaults to False.

    Returns:
        List[Union[int, str]]: List of code points as integers or hex strings.
    """
    return [hex(ord(x)) for x in emoji_str] if as_hex else [ord(x) for x in emoji_str]


class EmojiData:
    """Parser and analyzer for Unicode emoji-test.txt files.

    Loads emoji metadata from Unicode's emoji-test.txt file and provides
    utilities for identifying emojis in text, extracting emoji sequences,
    and querying emoji properties.

    Attributes:
        emojis: Dictionary mapping emoji characters to their Emoji metadata.
    """

    def __init__(self, emoji_test_path: str):
        self.emojis: Dict[str, Emoji] = {}  # emojichars -> EmojiData
        self._echars: List[int] | None = None
        self._ldepechars: Dict[int, Set[int]] | None = None
        self._rdepechars: Dict[int, Set[int]] | None = None
        self._load_emoji_test(emoji_test_path)

    @property
    def echars(self) -> List[int]:
        """Get code points that standalone represent complete emojis.

        Returns:
            List[int]: List of code points that by themselves form valid emojis.
        """
        if self._echars is None:
            self._compute_echars()
        return self._echars if self._echars is not None else []

    @property
    def ldepechars(self) -> Dict[int, Set[int]]:
        """Get code points that precede other code points in emoji sequences.

        Maps code points that are not standalone emojis but can precede emoji
        code points in compound sequences.

        Returns:
            Dict[int, Set[int]]: Dictionary where keys are left-dependent code
                points and values are sets of emoji code points that can follow.
        """
        if self._ldepechars is None:
            self._compute_echars()
        return self._ldepechars if self._ldepechars is not None else {}

    @property
    def rdepechars(self) -> Dict[int, Set[int]]:
        """Get code points that follow other code points in emoji sequences.

        Maps code points that are not standalone emojis but can follow emoji
        code points in compound sequences.

        Returns:
            Dict[int, Set[int]]: Dictionary where keys are right-dependent code
                points and values are sets of emoji code points that can precede.
        """
        if self._rdepechars is None:
            self._compute_echars()
        return self._rdepechars if self._rdepechars is not None else {}

    def _compute_echars(self) -> None:
        loneechars = [ord(emoji[0]) for emoji in self.emojis if len(emoji) == 1]
        ldepechars: Dict[int, Set[int]] = defaultdict(set)
        rdepechars: Dict[int, Set[int]] = defaultdict(set)
        for seq in self.emojis:
            for idx, echar in enumerate(seq):
                echar_ord = ord(echar)
                if echar not in [chr(c) for c in loneechars] and echar not in SPECIAL_CHARS:
                    if idx + 1 < len(seq):
                        ldepechars[echar_ord].add(ord(seq[idx + 1]))
                    else:
                        rdepechars[echar_ord].add(ord(seq[idx - 1]))
        self._echars = loneechars
        self._ldepechars = ldepechars
        self._rdepechars = rdepechars

    def emojis_with_cp(self, cp: int) -> List[Emoji]:
        """Find all emojis containing a specific code point.

        Args:
            cp: Unicode code point to search for.

        Returns:
            List[Emoji]: List of emoji metadata objects containing the code point.
        """
        return [e for emoji, e in self.emojis.items() if cp in get_emoji_seq(emoji)]

    def emoji_bio(self, emoji_text: str) -> str:
        """Create BIO tags identifying emoji character positions in text.

        Generates a string of the same length as the input where each character
        is tagged as 'B' (Begin - first char of emoji), 'I' (Internal - subsequent
        chars in emoji), or 'O' (Outer - not part of emoji).

        Args:
            emoji_text: Input text to analyze.

        Returns:
            str: BIO-tagged string of same length as input.
        """
        result = []
        start_pos = -1
        textlen = len(emoji_text)
        prevc: str | None = None
        for idx, c in enumerate(emoji_text):
            c_ord = ord(c)
            isechar = c_ord in self.echars
            isrechar = (
                c_ord in self.rdepechars
                and prevc is not None
                and ord(prevc) in self.rdepechars[c_ord]
            )
            islechar = (
                idx + 1 < textlen
                and c_ord in self.ldepechars
                and ord(emoji_text[idx + 1]) in self.ldepechars[c_ord]
            )
            issp = c in SPECIAL_CHARS

            if start_pos < 0:
                if isechar or islechar:
                    start_pos = idx
                    result.append("B")
                else:
                    result.append("O")
            elif not isrechar and not issp:
                if isechar or islechar:
                    if prevc != ZERO_WIDTH_JOINER:
                        start_pos = idx
                        result.append("B")
                    else:
                        result.append("I")
                else:
                    start_pos = -1
                    result.append("O")
            else:
                result.append("I")
            prevc = c
        return "".join(result)

    def get_emojis(self, text: str) -> List[Emoji]:
        """Extract all emojis from text with their metadata.

        Args:
            text: Arbitrary text to search for emojis.

        Returns:
            List[Emoji]: List of emoji metadata objects found in the text
                (empty if no emojis found).
        """
        result = []
        bio = self.emoji_bio(text)
        biolen = len(bio)
        start_pos = 0
        while "B" in bio[start_pos:]:
            start_pos = bio.index("B", start_pos)
            end_pos = start_pos + 1
            while end_pos < biolen and bio[end_pos] == "I":
                end_pos += 1
            result.append(self.emojis[text[start_pos:end_pos]])
            start_pos = end_pos
        return result

    def _load_emoji_test(self, emoji_test_path: str) -> None:
        curgroup = ""
        cursubgroup = ""
        with open(emoji_test_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("# group:"):
                    curgroup = line[9:].strip()
                elif line.startswith("# subgroup:"):
                    cursubgroup = line[12:].strip()
                elif line.startswith("# Status Counts"):
                    # Verify actual counts
                    c: Counter = Counter()
                    for e in self.emojis.values():
                        c[e.status] += 1
                    # with expectations
                    for line in f:
                        if line.startswith("# "):
                            m = STATUS_COUNT_RE.match(line)
                            if m:
                                assert c[m.group(1)] == int(m.group(2))
                        else:
                            break
                elif not line.startswith("#"):
                    line = line.strip()
                    if line:
                        emoji_data: Emoji | None = build_emoji_dataclass(line)
                        if emoji_data is not None:
                            emoji_data.group = curgroup
                            emoji_data.subgroup = cursubgroup
                            self.emojis[emoji_data.emoji] = emoji_data


def load_emoji_data() -> EmojiData | None:
    """Load emoji data from emoji-test.txt file.

    Attempts to load from the EMOJI_TEST_DATA environment variable if set,
    otherwise uses the default latest emoji data file.

    Returns:
        EmojiData | None: Loaded emoji data, or None if the data file doesn't exist.
    """
    result = None
    datapath = os.environ.get("EMOJI_TEST_DATA", LATEST_EMOJI_DATA)
    if os.path.exists(datapath):
        result = EmojiData(datapath)
    return result
