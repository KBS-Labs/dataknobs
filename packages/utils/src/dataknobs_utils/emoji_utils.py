'''
Utilities for working with unicode emojis.


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
'''
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set


LATEST_EMOJI_DATA = 'resources/emoji-test.15.0.txt'


ETESTLINE_RE = re.compile(r'^([0-9A-F ]+); ([\w\-]+)\s+#.+(E\d+\.\d+) (.*)$')
STATUS_COUNT_RE = re.compile(r'^# ([\w\-]+)\s+:\s+(\d+).*$')


ZERO_WIDTH_JOINER = chr(int('0x200D', 16))
VARIATION_SELECTOR_16 = chr(int('0xFE0F', 16))

SPECIAL_CHARS = {
    ZERO_WIDTH_JOINER,
    VARIATION_SELECTOR_16,
}


@dataclass
class Emoji:
    emoji: str            # The emoji characters
    status: str           # component  (fully-  minimally-  un) + qualified
    since_version: str    # "Ex.x"
    short_name: str       # (english) short name
    group: str = None     # test file "group" name
    subgroup: str = None  # test file "subgroup" name


def build_emoji_dataclass(emoji_test_line: str) -> Emoji:
    '''
    Build an Emoji dataclass from the emoji-test file line.
    '''
    result = None
    m = ETESTLINE_RE.match(emoji_test_line)
    if m:
        result = Emoji(
            ''.join(chr(int(x, 16)) for x in m.group(1).split()),
            m.group(2),
            m.group(3),
            m.group(4).strip()
        )
    return result


def get_emoji_seq(emoji_str: str, as_hex: bool = False) -> List[int]:
    return [
        hex(ord(x)) for x in emoji_str
    ] if as_hex else [
        ord(x) for x in emoji_str
    ]


class EmojiData:
    '''
    Class for interpreting the unicode emoji_test.txt file.
    '''

    def __init__(self, emoji_test_path: str):
        self.emojis = dict()  # emojichars -> EmojiData
        self._echars = None
        self._ldepechars = None
        self._rdepechars = None
        self._load_emoji_test(emoji_test_path)

    @property
    def echars(self) -> List[int]:
        '''
        Code points that alone are an emoji code point.
        '''
        if self._echars is None:
            self._compute_echars()
        return self._echars

    @property
    def ldepechars(self) -> Dict[int, Set[int]]:
        '''
        Code points are not an emoji alone, but precede emoji code points,
        where ldepechars[cp] is the set of all emoji code points that follow cp.
        '''
        if self._ldepechars is None:
            self._compute_echars()
        return self._ldepechars

    @property
    def rdepechars(self) -> Dict[int, Set[int]]:
        '''
        Code points are not an emoji alone, but follow emoji code points,
        where rdepechars[cp] is the set of all emoji code points that precede cp.
        '''
        if self._rdepechars is None:
            self._compute_echars()
        return self._rdepechars

    def _compute_echars(self):
        loneechars = [
            emoji[0]
            for emoji in self.emojis
            if len(emoji) == 1
        ]
        ldepechars = defaultdict(set)
        rdepechars = defaultdict(set)
        for seq in self.emojis:
            for idx, echar in enumerate(seq):
                if (echar not in loneechars and echar not in SPECIAL_CHARS):
                    if idx + 1 < len(seq):
                        ldepechars[echar].add(seq[idx+1])
                    else:
                        rdepechars[echar].add(seq[idx-1])
        self._echars = loneechars
        self._ldepechars = ldepechars
        self._rdepechars = rdepechars

    def emojis_with_cp(self, cp: int) -> List[Emoji]:
        '''
        Find emojis containing the given code point.
        :param cp: The code point
        :return: The list of emoji dataclasses
        '''
        return [
            e for emoji, e in self.emojis.items()
            if cp in get_emoji_seq(emoji)
        ]

    def emoji_bio(self, emoji_text: str) -> str:
        '''
        Given a string of text, create a "BIO" string to identify Begin,
        Internal, and Outer emoji characters in the text.
        :param text: The input text
        :return: a BIO string
        '''
        result = list()
        start_pos = -1
        textlen = len(emoji_text)
        prevc = None
        for idx, c in enumerate(emoji_text):
            isechar = c in self.echars
            isrechar = c in self.rdepechars and prevc in self.rdepechars[c]
            islechar = (
                idx+1 < textlen and
                c in self.ldepechars and
                emoji_text[idx+1] in self.ldepechars[c]
            )
            issp = c in SPECIAL_CHARS

            if start_pos < 0:
                if isechar or islechar:
                    start_pos = idx
                    result.append('B')
                else:
                    result.append('O')
            else:
                if not isrechar and not issp:
                    if isechar or islechar:
                        if prevc != ZERO_WIDTH_JOINER:
                            start_pos = idx
                            result.append('B')
                        else:
                            result.append('I')
                    else:
                        start_pos = -1
                        result.append('O')
                else:
                    result.append('I')
            prevc = c
        return ''.join(result)

    def get_emojis(self, text: str) -> List['EmojiData']:
        '''
        Get emoji data for all emojis in the given text.
        :param text: Arbitrary text
        :return: The (possibly empty) list of emojis found
        '''
        result = list()
        bio = self.emoji_bio(text)
        biolen = len(bio)
        start_pos = 0
        while 'B' in bio[start_pos:]:
            start_pos = bio.index('B', start_pos)
            end_pos = start_pos + 1
            while end_pos < biolen and bio[end_pos] == 'I':
                end_pos += 1
            result.append(self.emojis[text[start_pos:end_pos]])
            start_pos = end_pos
        return result

    def _load_emoji_test(self, emoji_test_path: str):
        curgroup = ''
        cursubgroup = ''
        with open(emoji_test_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('# group:'):
                    curgroup = line[9:].strip()
                elif line.startswith('# subgroup:'):
                    cursubgroup = line[12:].strip()
                elif line.startswith('# Status Counts'):
                    # Verify actual counts
                    c = Counter()
                    for e in self.emojis.values():
                        c[e.status] += 1
                    # with expectations
                    for l in f:
                        if l.startswith('# '):
                            m = STATUS_COUNT_RE.match(l)
                            if m:
                                assert c[m.group(1)] == int(m.group(2))
                        else:
                            break
                elif not line.startswith('#'):
                    line = line.strip()
                    if line:
                        e = build_emoji_dataclass(line)
                        if e:
                            e.group = curgroup
                            e.subgroup = cursubgroup
                            self.emojis[e.emoji] = e


def load_emoji_data() -> EmojiData:
    '''
    Load latest emoji-test.txt or reference EMOJI_TEST_DATA env var.
    '''
    result = None
    datapath = os.environ.get('EMOJI_TEST_DATA', LATEST_EMOJI_DATA)
    if os.path.exists(datapath):
        result = EmojiData(datapath)
    return result
