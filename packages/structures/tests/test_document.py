import dataknobs_structures.document as dk_doc


def test_simple_text():
    text = dk_doc.Text("This is a test.", None)
    assert text.text == "This is a test."
    assert text.text_id == 0
    assert text.text_label == "text"
    assert text.metadata.get_value("random", missing="value") == "value"
