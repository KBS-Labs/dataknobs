import io

from dataknobs_utils import xml_utils


def test_xml_leaf_stream():
    xml = """<a>
               <b0><foo><bar><baz value="0"/></bar></foo></b0>
               <b1><foo><bar><baz code="1"/></bar></foo></b1>
               <b2><foo><bar><baz>2</baz></bar></foo></b2>
             </a>
          """
    xml_file = io.StringIO(xml)
    xls = xml_utils.XmlLeafStream(xml_file)
    leaf_paths = [
        'a.b0.foo.bar.baz|value="0"',
        'a.b1.foo.bar.baz|code="1"',
        'a.b2.foo.bar.baz|text="2"',
    ]
    for idx, elts in enumerate(xls):
        assert leaf_paths[idx] == xls.to_string(elts, ["value", "code"])


def test_xml_element_grabber():
    xml = """<a>
               <b0><foo><bar><baz value="0"/></bar></foo></b0>
               <b1><foo><bar><baz code="1"/></bar></foo></b1>
               <b2><foo><bar><baz>2</baz></bar></foo></b2>
             </a>
          """
    xml_file = io.StringIO(xml)
    xls = xml_utils.XmlLeafStream(xml_file)
    xeg = xml_utils.XmlElementGrabber(xml_file, "foo")
    first2 = list(xeg.take(2))
    assert len(first2) == 2
    assert xeg.to_string(first2[0]) == "a.b0.foo"
    assert xeg.to_string(first2[1]) == "a.b1.foo"
    last1 = list(xeg.take(100))
    assert xeg.to_string(last1[0]) == "a.b2.foo"


# def test_soup_generator():
#     ...


# def test_html_table_scraper():
#     ...
