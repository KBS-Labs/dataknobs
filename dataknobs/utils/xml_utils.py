import bs4
import mmap
import pandas as pd


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
