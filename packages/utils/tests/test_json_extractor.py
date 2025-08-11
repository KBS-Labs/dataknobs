from dataknobs_utils import json_extractor


class TestJSONExtractor:
    def test_complete_json(self):
        extractor = json_extractor.JSONExtractor()
        text = 'Here is a JSON: {"name": "John", "age": 30} and some text after.'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0] == {"name": "John", "age": 30}
        assert extractor.get_complete_jsons() == [{"name": "John", "age": 30}]
        assert extractor.get_fixed_jsons() == []
        assert extractor.get_non_json_text() == "Here is a JSON:  and some text after."
        assert extractor.get_values("age") == [30]
        assert extractor.get_values("old_age") == []
        assert extractor.get_value("age", 100) == 30
        assert extractor.get_value("old_age", 100) == 100

    def test_multiple_json(self):
        extractor = json_extractor.JSONExtractor()
        text = 'First JSON: {"id": 1, "value": true} and second JSON: {"id": 2, "value": false}'
        result = extractor.extract_jsons(text)

        assert len(result) == 2
        assert result[0] == {"id": 1, "value": True}
        assert result[1] == {"id": 2, "value": False}
        assert len(extractor.get_complete_jsons()) == 2
        assert extractor.get_fixed_jsons() == []
        assert extractor.get_non_json_text() == "First JSON:  and second JSON:"
        assert extractor.get_values("id") == [1, 2]
        assert extractor.get_values("old_id") == []
        assert extractor.get_value("id", 100) == 1
        assert extractor.get_value("old_id", 100) == 100

    def test_nested_json(self):
        extractor = json_extractor.JSONExtractor()
        text = 'Nested JSON: {"user": {"name": "Alice", "roles": ["admin", "user"]}}'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0] == {"user": {"name": "Alice", "roles": ["admin", "user"]}}
        assert extractor.get_complete_jsons() == [
            {"user": {"name": "Alice", "roles": ["admin", "user"]}}
        ]
        assert extractor.get_fixed_jsons() == []
        assert extractor.get_non_json_text() == "Nested JSON:"
        assert extractor.get_value("user.roles") == ["admin", "user"]
        assert extractor.get_value("user.roles[1]") == "user"
        assert extractor.get_value("user.roles[0]") == "admin"
        assert extractor.get_value("user.roles[?]") == "admin"
        assert extractor.get_value("user.roles[*]") == ["admin", "user"]

    def test_malformed_json_unclosed_brace(self):
        extractor = json_extractor.JSONExtractor()
        text = 'Malformed JSON: {"name": "Bob", "data": [1, 2, 3'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0] == {"name": "Bob", "data": [1, 2, 3]}
        assert extractor.get_complete_jsons() == []
        assert len(extractor.get_fixed_jsons()) == 1
        assert extractor.get_non_json_text() == "Malformed JSON:"

    def test_malformed_json_unclosed_string(self):
        extractor = json_extractor.JSONExtractor()
        text = 'Bad string: {"name": "John'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0] == {"name": "John"}
        assert extractor.get_complete_jsons() == []
        assert len(extractor.get_fixed_jsons()) == 1
        assert extractor.get_non_json_text() == "Bad string:"

    def test_malformed_json_trailing_comma(self):
        extractor = json_extractor.JSONExtractor()
        text = 'Trailing comma: {"items": [1, 2, 3,], "status": "pending",}'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0] == {"items": [1, 2, 3], "status": "pending"}
        assert extractor.get_complete_jsons() == []
        assert len(extractor.get_fixed_jsons()) == 1
        assert extractor.get_non_json_text() == "Trailing comma:"

    def test_no_json(self):
        extractor = json_extractor.JSONExtractor()
        text = "This text contains no JSON objects at all."
        result = extractor.extract_jsons(text)

        assert len(result) == 0
        assert extractor.get_complete_jsons() == []
        assert extractor.get_fixed_jsons() == []
        assert extractor.get_non_json_text() == "This text contains no JSON objects at all."

    def test_multiple_malformed_json(self):
        extractor = json_extractor.JSONExtractor()
        text = 'First: {"id": 1} Second: {"name": "test", "values": [1, 2,'
        result = extractor.extract_jsons(text)

        assert len(result) == 2
        assert result[0] == {"id": 1}
        assert result[1] == {"name": "test", "values": [1, 2]}
        assert extractor.get_complete_jsons() == [{"id": 1}]
        assert len(extractor.get_fixed_jsons()) == 1
        assert extractor.get_non_json_text() == "First:  Second:"

    def test_severely_malformed_json(self):
        extractor = json_extractor.JSONExtractor()
        text = 'Very bad: {"key": "'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0] == {"key": ""}
        assert extractor.get_complete_jsons() == []
        assert len(extractor.get_fixed_jsons()) == 1
        assert extractor.get_non_json_text() == "Very bad:"

    def test_json_with_escaped_quotes(self):
        extractor = json_extractor.JSONExtractor()
        text = 'Escaped quotes: {"message": "He said \\"hello\\""}'
        result = extractor.extract_jsons(text)

        assert len(result) == 1
        assert result[0] == {"message": 'He said "hello"'}
        assert extractor.get_complete_jsons() == [{"message": 'He said "hello"'}]
        assert extractor.get_fixed_jsons() == []
        assert extractor.get_non_json_text() == "Escaped quotes:"
