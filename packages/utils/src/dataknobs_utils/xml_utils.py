"""XML processing utilities with streaming and memory-efficient parsing.

Provides classes and functions for parsing and streaming XML documents,
including the XmlStream abstract base class for memory-efficient processing.
"""

import mmap
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from typing import Any, List, TextIO, Tuple, Union

import bs4
import pandas as pd


class XmlStream(ABC):
    """Abstract base class for streaming XML content with memory management.

    Provides a framework for iterating through XML documents using element-based
    streaming. Subclasses must implement loop_through_elements() to define specific
    element collection behavior. Automatically manages element cleanup to prevent
    memory buildup when processing large XML files.

    Note:
        Extending classes must implement the loop_through_elements() abstract method.

    Attributes:
        source: XML filename or file object.
        auto_clear_elts: If True, automatically clears closed elements to save memory.
    """

    def __init__(self, source: Union[str, TextIO], auto_clear_elts: bool = True):
        """Initialize the XML stream.

        Args:
            source: Path to XML file or file object.
            auto_clear_elts: If True, automatically clears closed elements to prevent
                memory buildup. Defaults to True.
        """
        self._xml_iter: Any | None = None
        self.source = source
        self.auto_clear_elts = auto_clear_elts
        self._context: List[ET.Element] = []
        self._closed_elt: ET.Element | None = None

    @property
    def context(self) -> List[ET.Element]:
        """Get the current element context stack from root to current position.

        Returns:
            List[ET.Element]: Copy of the element stack from root to current element.
        """
        return self._context.copy()

    @property
    def context_length(self) -> int:
        """Get the depth of the current element in the XML tree.

        Returns:
            int: Number of elements in the context stack.
        """
        return len(self._context)

    def next_xml_iter(self) -> Tuple[str, ET.Element]:
        """Get the next event and element from the XML iterator.

        Returns:
            Tuple[str, ET.Element]: Tuple of (event, element) where event is
                'start' or 'end'.

        Raises:
            StopIteration: When the XML iterator is exhausted.
        """
        if self._xml_iter is None:
            self.__iter__()
        if self._xml_iter is not None:
            try:
                event, elem = next(self._xml_iter)
                return event, elem
            except StopIteration as exc:
                raise StopIteration from exc
        raise StopIteration

    @abstractmethod
    def loop_through_elements(self) -> List[ET.Element]:
        """Process XML elements until collecting the next desired element(s).

        Subclasses must implement this method to define specific element
        collection behavior, updating the context as elements are encountered.

        Returns:
            List[ET.Element]: The collected element(s) in context from root.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def close(self) -> None:
        self.__exit__()

    def __iter__(self) -> "XmlStream":
        self._xml_iter = ET.iterparse(self.source, events=["start", "end"])
        self._context = []
        if self._closed_elt is not None:
            self._closed_elt.clear()
            self._closed_elt = None
        return self

    def __next__(self) -> List[ET.Element]:
        return self.loop_through_elements()

    def __enter__(self) -> "XmlStream":
        return self

    def __exit__(self, *args: object, **kwargs: Any) -> None:
        if self._closed_elt is not None:
            self._closed_elt.clear()
            self._closed_elt = None

    def add_to_context(self, elem: ET.Element) -> None:
        """Add an element to the context stack.

        Args:
            elem: Element to add to the context.
        """
        self._context.append(elem)

    def find_context_idx(self, elem: ET.Element) -> int:
        """Find the most recent index of an element in the context stack.

        Searches backwards through the context to find the element's index.

        Args:
            elem: Element to find in the context.

        Returns:
            int: Index of the element in context, or -1 if not found.
        """
        idx = len(self._context) - 1
        while idx >= 0:
            if self._context[idx] == elem:
                break
            idx -= 1
        return idx

    def pop_closed_from_context(self, closed_elem: ET.Element, idx: int | None = None) -> None:
        """Remove a closed element and its descendants from the context.

        Truncates the context at the element's position and optionally clears
        the element's memory if auto_clear_elts is enabled.

        Args:
            closed_elem: The element that has closed.
            idx: Index of the element in context (computed if None). Defaults to None.
        """
        if idx is None:
            idx = self.find_context_idx(closed_elem)
        if idx >= 0:
            self._context = self._context[:idx]
        if self.auto_clear_elts:
            if self._closed_elt is not None:
                self._closed_elt.clear()  # Clear memory
            self._closed_elt = closed_elem

    def take(self, n: int) -> Generator[List[ET.Element], None, None]:
        """Generate the next N items from this iterator.

        Args:
            n: Number of items to take.

        Yields:
            List[ET.Element]: Each collected element list, stopping early if
                iterator is exhausted.
        """
        idx = 0
        while idx < n:
            try:
                elts = next(self)
            except StopIteration:
                return
            yield elts
            idx += 1

    @staticmethod
    def to_string(
        elt_path: List[ET.Element], with_text_or_atts: Union[bool, str, List[str]] = True
    ) -> str:
        r"""Convert element path to dot-delimited string with optional leaf value.

        Creates an XPath-like string representation of the element hierarchy,
        optionally appending text or attribute values from the leaf element.

        Args:
            elt_path: List of elements from root to leaf.
            with_text_or_atts: Controls value appending behavior:
                - True: Append leaf element's text if present
                - str: Append value of specified attribute if present
                - List[str]: Append first non-empty attribute value from list
                - False: Don't append any values

        Returns:
            str: Dot-delimited path (e.g., "root.child.leaf|text=\"value\"").
        """
        rv = ".".join(e.tag for e in elt_path)
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
    """Stream XML leaf nodes with their full path from root.

    Iterates through an XML document yielding each leaf (terminal) element
    along with its ancestor path from the root. Useful for extracting data
    from deeply nested XML structures.

    Examples:
        Show paths to first 10 leaf nodes with text or attribute values:

        ```python
        import dataknobs_utils.xml_utils as xml_utils
        xml_fpath = "path/to/file.xml"  # Replace with actual XML file path
        s = xml_utils.XmlLeafStream(xml_fpath)
        for idx, elts in enumerate(s):
            print(f'{idx} ', s.to_string(elts, ["value", "extension", "code", "ID"]))
            if idx >= 9:
                break
        ```

    Attributes:
        count: Number of leaf nodes processed.
        elts: Most recently yielded element path.
    """

    def __init__(self, source: Union[str, TextIO], auto_clear_elts: bool = True):
        """Initialize the XML leaf stream.

        Args:
            source: Path to XML file or file object.
            auto_clear_elts: If True, automatically clears closed elements.
                Defaults to True.
        """
        super().__init__(source, auto_clear_elts=auto_clear_elts)
        self._last_elt: ET.Element | None = None  # The last new element
        self.count = 0  # The number of terminal nodes seen
        self.elts: List[ET.Element] | None = None  # The latest yielded sequence

    def loop_through_elements(self) -> List[ET.Element]:
        """Collect the next leaf element with its full path from root.

        Returns:
            List[ET.Element]: Element path from root to the next leaf element.
        """
        gotit: List[ET.Element] | None = None
        while True:
            event, elem = self.next_xml_iter()
            if event == "start":
                self._last_elt = elem
                self.add_to_context(elem)
            elif event == "end":
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
        return gotit or []


class XmlElementGrabber(XmlStream):
    """Stream matching XML elements with their full path from root.

    Finds and yields elements matching a tag name or custom condition,
    returning the highest matching element (not searching within matched elements).

    Examples:
        Get first 10 elements with tag "foo":

        >>> import dataknobs_utils.xml_utils as xml_utils
        >>> xml_fpath = "path/to/file.xml"  # Replace with actual XML file path
        >>> g = xml_utils.XmlElementGrabber(xml_fpath, "foo")
        >>> first_10_foos = list(g.take(10))

    Attributes:
        match: Tag name or callable for matching elements.
        count: Number of matching elements found.
    """

    def __init__(
        self,
        source: Union[str, TextIO],
        match: Union[str, Callable[[ET.Element], bool]],
        auto_clear_elts: bool = True,
    ):
        """Initialize the XML element grabber.

        Args:
            source: Path to XML file or file object.
            match: Tag name to match or callable that returns True when an
                element should be matched.
            auto_clear_elts: If True, automatically clears closed elements.
                Defaults to True.
        """
        super().__init__(source, auto_clear_elts=auto_clear_elts)
        self.match = match
        self.count = 0  # The number of match nodes seen

    def _is_match(self, elem: ET.Element) -> bool:
        matches = False
        if isinstance(self.match, str):
            if ":" in self.match:
                matches = elem.tag == self.match
            elif ":" in elem.tag:
                # match ns0:<match> or {...:...}<match> (NOTE: 0x7d == '}')
                matches = re.match(rf"^.*[:\x7d]{self.match}$", elem.tag) is not None
            else:
                matches = self.match == elem.tag
        else:
            matches = self.match(elem)
        return matches

    def loop_through_elements(self) -> List[ET.Element]:
        """Find the next matching element with its full path from root.

        Returns:
            List[ET.Element]: Element path from root to the next matching element.
        """
        gotit: List[ET.Element] | None = None
        grabbing: ET.Element | None = None
        while True:
            event, elem = self.next_xml_iter()
            if event == "start":
                self.add_to_context(elem)
                if grabbing is None and self._is_match(elem):
                    grabbing = elem
            elif event == "end":
                if grabbing is not None and elem == grabbing:
                    # Finished collecting match element
                    grabbing = None
                    self.count += 1
                    gotit = self.context
                # Pop the closed element from the context
                self.pop_closed_from_context(elem)
                if gotit:
                    break
        return gotit or []


class XMLTagStream:
    """Memory-efficient XML tag chunking using memory-mapped file access.

    Processes large XML files by extracting individual tag instances as
    BeautifulSoup objects using memory-mapped I/O for efficient processing
    without loading the entire file.

    Attributes:
        tag_name: XML tag name to extract.
        encoding: Character encoding of the XML file.
    """

    def __init__(
        self, path: str, tag_name: str, with_attrs: bool = False, encoding: str = "utf-8"
    ) -> None:
        self.file = open(path, encoding="utf-8")
        self.stream = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.tag_name = tag_name
        self.encoding = encoding
        tag_end = " " if with_attrs else ">"
        self.start_tag = f"<{tag_name}{tag_end}".encode(encoding)
        self.end_tag = f"</{tag_name}>".encode(encoding)

    def __enter__(self) -> "XMLTagStream":
        return self

    def __exit__(self, *args: object, **kwargs: Any) -> None:
        self.stream.close()
        self.file.close()

    def __iter__(self) -> Generator[bs4.BeautifulSoup, None, None]:
        end = 0
        while (begin := self.stream.find(self.start_tag, end)) != -1:
            end = self.stream.find(self.end_tag, begin)
            yield self.parse(self.stream[begin : end + len(self.end_tag)])

    def parse(self, chunk: bytes) -> bs4.BeautifulSoup:
        return bs4.BeautifulSoup(chunk.decode(self.encoding), features="lxml-xml")


def soup_generator(
    xmlfilepath: str, tag_name: str, with_attrs: bool = False, encoding: str = "utf-8"
) -> Generator[bs4.BeautifulSoup, None, None]:
    """Generate BeautifulSoup objects for each occurrence of a tag in XML.

    Efficiently processes large XML files by yielding BeautifulSoup objects
    for each instance of the specified tag using memory-mapped I/O.

    Args:
        xmlfilepath: Path to the XML file.
        tag_name: XML tag name to extract.
        with_attrs: If True, matches tags with attributes. Defaults to False.
        encoding: File character encoding. Defaults to "utf-8".

    Yields:
        bs4.BeautifulSoup: Soup object for each tag instance in the file.
    """
    with XMLTagStream(
        xmlfilepath,
        tag_name,
        with_attrs=with_attrs,
        encoding=encoding,
    ) as stream:
        for soup in stream:
            yield soup


def html_table_scraper(
    soup_table: bs4.element.Tag,
    add_header_as_row: bool = False,
) -> pd.DataFrame:
    """Extract HTML table data into a pandas DataFrame.

    Scrapes table data from a BeautifulSoup table element, handling headers,
    nested elements, and non-breaking spaces. Concatenates text from nested
    elements within cells.

    Args:
        soup_table: BeautifulSoup table element to scrape.
        add_header_as_row: If True, includes header row in the data rows
            in addition to using it as column names. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with scraped table data. Column names are taken
            from <th> elements if present, otherwise columns are unnamed.
    """
    columns: List[str] | None = None
    rows: List[List[str]] = []

    def html_text(elt: bs4.element.Tag) -> str:
        return elt.text.replace("\xa0", " ").strip()

    def td_text(td: bs4.element.Tag) -> str:
        return "\n".join(
            [  # add a \n between text of td elements
                x
                for x in [html_text(c) for c in td.children if isinstance(c, bs4.element.Tag)]
                if x  # drop empty elements under a td
            ]
        )

    for tr in soup_table.find_all("tr"):
        if columns is None:
            if tr.find("th") is not None:
                columns = [td_text(th) for th in tr.find_all("th")]
                if add_header_as_row:
                    rows.append(columns)
                continue
            else:
                columns = []
        row = [td_text(td) for td in tr.find_all("td")]
        if len(row) > 0:
            rows.append(row)

    return pd.DataFrame(rows, columns=columns if columns and len(columns) > 0 else None)
