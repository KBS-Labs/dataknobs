import bs4
import itertools
import mmap
import pandas as pd
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Callable, List, TextIO, Union


class XmlStream(ABC):
    '''
    Abstract base class for streaming XML content.

    NOTE: Extending classes need to implement __iter__
    '''
    def __init__(self, source: Union[str, TextIO]):
        '''
        :param source: An XML filename or file object
        '''
        self.iter = ET.iterparse(source, events=['start', 'end'])
        # context is the current "stack" of elements from the root
        self.context = list()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.iter.close()

    def _add_to_context(self, elem: ET.Element):
        ''' Add the element to the context '''
        self.context.append(elem)

    def _find_context_idx(self, elem: ET.Element):
        '''
        Find the latest index of the element in the context (or -1).
        '''
        idx = len(self.context) - 1
        while idx >= 0:
            if self.context[idx] == elem:
                break
        return idx

    def _pop_closed_from_context(
            self,
            closed_elem: ET.Element,
            idx: int = None
    ):
        '''
        Pop the closed element from the context
        :param closed_elem: The closed element
        :param idx: The element's index within the context (if known)
        '''
        if idx is None:
            idx = self._find_context_idx(closed_elem)
        if idx >= 0:
            self.context = self.context[:idx]

    def take(self, n: int) -> List[ET.Element]:
        '''
        Take the next N items from this iterator.
        '''
        return itertools.islice(self, n)

    @staticmethod
    def to_string(elt_path: List[ET.Element], with_text_or_atts: Union[bool, str, List[str]] = True) -> str:
        '''
        Show the element path  as a dot-delimited string of tags, optionally
        adding the leaf node's text or value from an attribute.
    
        :param elt_path: The list of elements from root to leaf
        :param with_text_or_atts: If True or a non-empty string or list of
            strings, then append the first non-empty value from the element's
            text or attribute.
        '''
        rv = '.'.join(e.tag for e in elt_path)
        if with_text_or_atts:
            elem = elt_path[-1]
            text = None
            if elem.text:
                text = f'|text="{elem.text}"'
            elif isinstance(with_text_or_atts, str):
                val = elem.get(with_text_or_atts)
                if val:
                    text = f'|{with_text_or_atts}="{text}"'
                    text = val
            elif isinstance(with_text_or_atts, list):
                for att in with_text_or_atts:
                    val = elem.get(att)
                    if val:
                        text = f'|{att}="{val}"'
                        break
            if text:
                rv += text
        return rv


class XmlLeafStream(XmlStream):
    '''
    Class to get each XML leaf node with its parents to the root from an xml
    stream.

    Usage example to show the first 10 xpaths to leaf nodes, including node text
    taken from actual text or attributes:

    >>> import dataknobs.utils.xml_utils as xml_utils
    >>> s = xml_utils.XmlLeafStream(xml_fpath)
    >>> for idx, elts in enumerate(s):
            print(f'{idx} ', s.to_string(elts, ["value", "extension", "code", "ID"]))
    >>>     if idx >= 9:
                break
    '''

    def __init__(self, source: Union[str, TextIO]):
        '''
        :param source: An XML filename or file object
        '''
        super().__init__(source)
        self._last_elt = None  # The last new element
        self.count = 0  # The number of terminal nodes seen
        self.elts = None  # The latest yielded sequence

    def __iter__(self) -> List[ET.Element]:
        '''
        Generate the next sequence of ET.Element instances from the root
        (at index 0) to the terminal element.
        '''
        for event, elem in self.iter:
            if event == 'start':
                self._last_elt = elem
                self._add_to_context(elem)
            elif event == 'end':
                last_idx = len(self.context) - 1
                idx = self._find_context_idx(elem)
                if idx == last_idx and elem == self._last_elt:
                    # Is terminal if its the last elt that was added that ended
                    self.elts = self.context.copy()
                    self.count += 1
                    yield self.elts
                # Pop off the closed element(s)
                self._pop_closed_from_context(elem, idx=idx)
                # Reset to record the next added element
                self._last_elt = None


class XmlElementGrabber(XmlStream):
    '''
    Class to grab each highest matching DOM element from streamed XML.

    Usage example to get the first 10 xpaths to element nodes with the tag "foo"
    >>> import dataknobs.utils.xml_utils as xml_utils
    >>> g = xml_utils.XmlElementGrabber(xml_fpath, "foo")
    >>> first_10_foos = g.take(10)
    '''

    def __init__(
            self,
            source: Union[str, TextIO],
            match: Union[str, Callable[[ET.Element], bool]]
    ):
        '''
        :param source: An XML filename or file object
        :param match: A tag name to match or a function returning True when a
           an element is encountered.
        '''
        super().__init__(source)
        self.match = match
        self.count = 0  # The number of match nodes seen

    def _is_match(self, elem: ET.Element):
        matches = False
        if isinstance(self.match, str):
            if ':' in self.match:
                matches = (elem.tag == self.match)
            else:
                # match ns0:<match> or {...:...}<match> (NOTE: 0x7d == '}')
                matches = re.match(r'^.*[:\x7d]{match}$'.format(match=self.match), elem.tag)
        else:
            matches = self.match(elem)
        return matches

    def __iter__(self) -> List[ET.Element]:
        '''
        Generate the next match ET.Element, returning in context from the root
        node to the element.
        '''
        grabbing = None
        for event, elem in self.iter:
            if event == 'start':
                self.context.append(elem)
                if grabbing is None and self._is_match(elem):
                    grabbing = elem
            elif event == 'end':
                if grabbing is not None and elem == grabbing:
                    # Finished collecting match element
                    grabbing = None
                    self.count += 1
                    yield self.context.copy()
                # Pop the closed element from the context
                self._pop_closed_from_context(elem)


class XMLTagStream:
    '''
    Class for chunking a large XML file into separate bs4 "soup" objects based
    on a tag.
    '''
    def __init__(self, path, tag_name, with_attrs=False, encoding='utf-8'):
        self.file = open(path, encoding='utf-8')
        self.stream = mmap.mmap(
            self.file.fileno(), 0, access=mmap.ACCESS_READ
        )
        self.tag_name = tag_name
        self.encoding = encoding
        tag_end = ' ' if with_attrs else '>'
        self.start_tag = f'<{tag_name}{tag_end}'.encode(encoding)
        self.end_tag = f'</{tag_name}>'.encode(encoding)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.stream.close()
        self.file.close()

    def __iter__(self):
        end = 0
        while (begin := self.stream.find(self.start_tag, end)) != -1:
            end = self.stream.find(self.end_tag, begin)
            yield self.parse(self.stream[begin: end + len(self.end_tag)])

    def parse(self, chunk):
        return bs4.BeautifulSoup(chunk.decode(self.encoding), features='lxml-xml')


def soup_generator(xmlfilepath, tag_name, with_attrs=False, encoding='utf-8'):
    '''
    A generator for bs4 soup objects for each tag_name in the xml file.
    :param xmlfilepath: The path to the xml file
    :param tag_name: The tag name to extract
    :param with_attrs: True to find the tag with attributes
    :param encoding: File's encoding
    :yield: A soup object for each xml tag in the file
    '''
    with XMLTagStream(
            xmlfilepath, tag_name, with_attrs=with_attrs, encoding=encoding,
    ) as stream:
        for soup in stream:
            yield soup


def html_table_scraper(
        soup_table: bs4.element.Tag,
        add_header_as_row: bool = False,
) -> pd.DataFrame:
    '''
    Scrape html table information into a dataframe.
    :param soup_table: The soup table element
    :param add_header_as_row: If True, the header row will also be added as
        a table row.
    :return: A dataframe with the scraped table data
    '''
    columns = None
    rows = list()

    def html_text(elt):
        return elt.text.replace('\xa0', ' ').strip()

    def td_text(td):
        return '\n'.join([  # add a \n between text of td elements
            x for x in [
                html_text(c)
                for c in td.children
            ]
            if x  # drop empty elements under a td
        ])

    for tr in soup_table.find_all('tr'):
        if columns is None:
            if tr.find('th') is not None:
                columns = [
                    td_text(th)
                    for th in tr.find_all('th')
                ]
                if add_header_as_row:
                    rows.append(columns)
                continue
            else:
                columns = []
        row = [
            td_text(td)
            for td in tr.find_all('td')
        ]
        if len(row) > 0:
            rows.append(row)

    return pd.DataFrame(
        rows,
        columns=columns if len(columns) > 0 else None
    )
