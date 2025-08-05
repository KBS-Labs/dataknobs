import bs4
import mmap
import pandas as pd
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Callable, List, TextIO, Tuple, Union


class XmlStream(ABC):
    '''
    Abstract base class for streaming XML content.

    NOTE: Extending classes need to implement __iter__
    '''
    def __init__(self, source: Union[str, TextIO], auto_clear_elts: bool = True):
        '''
        :param source: An XML filename or file object
        '''
        self._xml_iter = None
        self.source = source
        self.auto_clear_elts = auto_clear_elts
        self._context = list()
        self._closed_elt = None

    @property
    def context(self) -> List[ET.Element]:
        '''
        Get the current "stack" of elements from the root.
        '''
        return self._context.copy()

    @property
    def context_length(self) -> int:
        '''
        Get the length of the current context.
        '''
        return len(self._context)

    def next_xml_iter(self) -> Tuple[str, ET.Element]:
        '''
        Get the next (event, elem) from the underlying xml iterator or raise
        a StopIteration exception if exhausted.
        '''
        if self._xml_iter is None:
            self.__iter__()
        try:
            event, elem = next(self._xml_iter)
            return event, elem
        except StopIteration as exc:
            raise StopIteration from exc

    @abstractmethod
    def loop_through_elements(self) -> List[ET.Element]:
        '''
        Loop through elements of self.xml_iter, adding elements to the context,
        until the next desired element has been collected.
        '''
        raise NotImplementedError

    def close(self):
        self.__exit__()

    def __iter__(self):
        self._xml_iter = ET.iterparse(self.source, events=['start', 'end'])
        self._context = list()
        if self._closed_elt is not None:
            self._closed_elt.clear()
            self._closed_elt = None
        return self

    def __next__(self):
        return self.loop_through_elements()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if self._closed_elt is not None:
            self._closed_elt.clear()
            self._closed_elt = None

    def add_to_context(self, elem: ET.Element):
        ''' Add the element to the context '''
        self._context.append(elem)

    def find_context_idx(self, elem: ET.Element):
        '''
        Find the latest index of the element in the context (or -1).
        '''
        idx = len(self._context) - 1
        while idx >= 0:
            if self._context[idx] == elem:
                break
        return idx

    def pop_closed_from_context(
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
            idx = self.find_context_idx(closed_elem)
        if idx >= 0:
            self._context = self._context[:idx]
        if self.auto_clear_elts:
            if self._closed_elt is not None:
                self._closed_elt.clear()  # Clear memory
            self._closed_elt = closed_elem

    def take(self, n: int) -> List[List[ET.Element]]:
        '''
        Take the next N items from this iterator.
        '''
        # items = list()
        idx = 0
        while idx < n:
            try:
                elts = next(self)
            except StopIteration:
                return
            yield elts
            # items.append(elts)
            idx += 1
        # return items

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

    def __init__(self, source: Union[str, TextIO], auto_clear_elts: bool = True):
        '''
        :param source: An XML filename or file object
        '''
        super().__init__(source, auto_clear_elts=auto_clear_elts)
        self._last_elt = None  # The last new element
        self.count = 0  # The number of terminal nodes seen
        self.elts = None  # The latest yielded sequence

    def loop_through_elements(self) -> Tuple[str, ET.Element]:
        '''
        Loop through elements of self.xml_iter, adding elements to the context,
        until the next terminal element has been collected.
        '''
        gotit = None
        while True:
            event, elem = self.next_xml_iter()
            if event == 'start':
                self._last_elt = elem
                self.add_to_context(elem)
            elif event == 'end':
                last_idx = self.context_length - 1
                idx = self.find_context_idx(elem)
                if idx == last_idx and elem == self._last_elt:
                    # Is terminal if its the last elt that was added that ended
                    self.elts = self.context
                    self.count += 1
                    gotit = self.elts
                # Pop off the closed element(s)
                self.pop_closed_from_context(elem, idx=idx)
                # Reset to record the next added element
                self._last_elt = None
                if gotit:
                    break
        return gotit or None


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
            match: Union[str, Callable[[ET.Element], bool]],
            auto_clear_elts: bool = True,
    ):
        '''
        :param source: An XML filename or file object
        :param match: A tag name to match or a function returning True when a
           an element is encountered.
        '''
        super().__init__(source, auto_clear_elts=auto_clear_elts)
        self.match = match
        self.count = 0  # The number of match nodes seen

    def _is_match(self, elem: ET.Element):
        matches = False
        if isinstance(self.match, str):
            if ':' in self.match:
                matches = (elem.tag == self.match)
            elif ':' in elem.tag:
                # match ns0:<match> or {...:...}<match> (NOTE: 0x7d == '}')
                matches = re.match(r'^.*[:\x7d]{match}$'.format(match=self.match), elem.tag)
            else:
                matches = (self.match == elem.tag)
        else:
            matches = self.match(elem)
        return matches

    def loop_through_elements(self) -> List[ET.Element]:
        '''
        Find the next match ET.Element, returning in context from the root
        node to the element.
        '''
        gotit = None
        grabbing = None
        while True:
            event, elem = self.next_xml_iter()
            if event == 'start':
                self.add_to_context(elem)
                if grabbing is None and self._is_match(elem):
                    grabbing = elem
            elif event == 'end':
                if grabbing is not None and elem == grabbing:
                    # Finished collecting match element
                    grabbing = None
                    self.count += 1
                    gotit = self.context
                # Pop the closed element from the context
                self.pop_closed_from_context(elem)
                if gotit:
                    break
        return gotit or None


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
