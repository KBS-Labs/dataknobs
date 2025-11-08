"""Extract and repair JSON objects from text strings.

Provides the JSONExtractor class for finding, extracting, and repairing
JSON objects embedded in text, with categorization of complete vs fixed objects.
"""

import json
import re
from typing import Any, Dict, List, Tuple

from dataknobs_utils.json_utils import get_value


class JSONExtractor:
    """Extract and repair JSON objects from text strings.

    Provides functionality to extract well-formed JSON objects, attempt to repair
    malformed JSON, and separate non-JSON text. Extracted objects are categorized
    as complete (well-formed) or fixed (repaired from malformed state).

    Attributes:
        complete_jsons: List of well-formed JSON objects extracted from text.
        fixed_jsons: List of JSON objects that were malformed but successfully repaired.
        non_json_text: Text content that doesn't contain JSON objects.
    """

    def __init__(self) -> None:
        # List of complete, well-formed JSON objects
        self.complete_jsons: List[Dict[str, Any]] = []
        # List of JSON objects that were malformed but fixed
        self.fixed_jsons: List[Dict[str, Any]] = []
        # Text that doesn't contain JSON objects
        self.non_json_text: str = ""

    def get_values(self, key_path: str) -> List[Any]:
        """Get values at a specific path from all complete JSON objects.

        Uses dot notation to navigate nested JSON structures and retrieves values
        from all complete JSON objects that contain the specified path.

        Args:
            key_path: Dot-separated path to the value (e.g., 's3.bucket' for
                nested structures).

        Returns:
            List[Any]: List of values found at the path (empty if none found).
        """
        values = []
        for json_obj in self.complete_jsons:
            value = get_value(json_obj, key_path)
            if value is not None:
                values.append(value)
        return values

    def get_value(self, key_path: str, default: Any | None = None) -> Any:
        """Get a value at a specific path from the first matching JSON object.

        Uses dot notation to navigate nested JSON structures and retrieves the value
        from the first complete JSON object that contains the specified path.

        Args:
            key_path: Dot-separated path to the value (e.g., 's3.bucket' for
                nested structures).
            default: Value to return if the path doesn't exist in any object.
                Defaults to None.

        Returns:
            Any: The value found at the path, or the default value if not found.
        """
        value = default
        for json_obj in self.complete_jsons:
            cur_value = get_value(json_obj, key_path)
            if cur_value is not None:
                value = cur_value
                break
        return value

    def extract_jsons(self, text: str) -> List[Dict[str, Any]]:
        """Extract all JSON objects from text, repairing malformed JSON when possible.

        Searches for JSON objects in the text, attempts to parse them, and repairs
        malformed JSON by closing unclosed brackets, quotes, and fixing trailing commas.
        Updates instance attributes with categorized results and remaining non-JSON text.

        Args:
            text: Text string potentially containing JSON objects.

        Returns:
            List[Dict[str, Any]]: All successfully extracted and parsed JSON objects
                (both complete and fixed).
        """
        self.complete_jsons = []
        self.fixed_jsons = []
        self.non_json_text = text

        # Find all potential JSON objects using regex
        # Look for patterns that start with { and end with }
        potential_jsons = self._find_json_objects(text)

        extracted_jsons = []

        for json_text, is_complete in potential_jsons:
            try:
                # Try to parse the JSON text
                json_obj = json.loads(json_text)
                if is_complete:
                    self.complete_jsons.append(json_obj)
                else:
                    self.fixed_jsons.append(json_obj)
                extracted_jsons.append(json_obj)

                # Remove the JSON text from non_json_text
                self.non_json_text = self.non_json_text.replace(json_text, "", 1)
            except json.JSONDecodeError:
                # If it's malformed, try to fix it
                fixed_json = self._fix_json(json_text)
                if fixed_json:
                    try:
                        json_obj = json.loads(fixed_json)
                        self.fixed_jsons.append(json_obj)
                        extracted_jsons.append(json_obj)

                        # Remove the original JSON text from non_json_text
                        self.non_json_text = self.non_json_text.replace(json_text, "", 1)
                    except json.JSONDecodeError:
                        # If we still can't parse it, leave it in non_json_text
                        pass

        # Clean up any remaining JSON brackets in non_json_text
        self.non_json_text = self.non_json_text.strip()

        return extracted_jsons

    def _find_json_objects(self, text: str) -> List[Tuple[str, bool]]:
        """Find potential JSON objects using bracket matching.

        Scans text for brace-delimited objects using a stack-based approach to
        identify complete JSON objects (with matching braces) and incomplete ones.

        Args:
            text: Text to search for JSON-like patterns.

        Returns:
            List[Tuple[str, bool]]: List of tuples where each tuple contains
                (json_text, is_complete). is_complete is True for objects with
                matching braces, False for incomplete objects.
        """
        result = []

        # First try to find complete JSON objects
        stack = []
        start_index = None

        for i, char in enumerate(text):
            if char == "{" and start_index is None:
                start_index = i
                stack.append(char)
            elif char == "{" and stack:
                stack.append(char)
            elif char == "}" and stack:
                stack.pop()
                if not stack and start_index is not None:
                    # Complete JSON object found
                    json_text = text[start_index : i + 1]
                    result.append((json_text, True))
                    start_index = None

        # Now look for incomplete JSON objects that started but didn't finish
        if start_index is not None:
            incomplete_json = text[start_index:]
            result.append((incomplete_json, False))

        return result

    def _fix_json(self, json_text: str) -> str | None:
        """Repair malformed JSON by closing unclosed elements.

        Attempts to fix common JSON formatting issues including unclosed strings,
        brackets, braces, and trailing commas. Uses character-by-character parsing
        to track string context and nesting depth.

        Args:
            json_text: Potentially malformed JSON text starting with '{'.

        Returns:
            str | None: Repaired JSON text if fixable, None if the text doesn't
                start with '{' or is empty.
        """
        if not json_text:
            return None

        if not json_text.startswith("{"):
            return None

        # Count unclosed braces
        open_braces = 0
        open_brackets = 0
        in_string = False
        escape_next = False

        for char in json_text:
            if escape_next:
                escape_next = False
                continue

            if char == "\\" and in_string:
                escape_next = True
            elif char == '"' and not escape_next:
                in_string = not in_string
            elif char == "{" and not in_string:
                open_braces += 1
            elif char == "}" and not in_string:
                open_braces -= 1
            elif char == "[" and not in_string:
                open_brackets += 1
            elif char == "]" and not in_string:
                open_brackets -= 1

        # Fix unclosed strings
        fixed_json = json_text
        if in_string:
            fixed_json += '"'

        # Fix unclosed braces and brackets
        fixed_json += "]" * max(0, open_brackets)
        fixed_json += "}" * max(0, open_braces)

        # Handle trailing commas before closing braces/brackets
        fixed_json = re.sub(r",\s*\]", "]", fixed_json)
        fixed_json = re.sub(r",\s*}", "}", fixed_json)

        return fixed_json

    def get_complete_jsons(self) -> List[Dict[str, Any]]:
        """Get all well-formed JSON objects extracted from text.

        Returns:
            List[Dict[str, Any]]: List of JSON objects that were successfully
                parsed without requiring repairs.
        """
        return self.complete_jsons

    def get_fixed_jsons(self) -> List[Dict[str, Any]]:
        """Get all JSON objects that were repaired from malformed state.

        Returns:
            List[Dict[str, Any]]: List of JSON objects that required repair
                (closing brackets, quotes, etc.) before successful parsing.
        """
        return self.fixed_jsons

    def get_non_json_text(self) -> str:
        """Get text content after removing all extracted JSON objects.

        Returns:
            str: Remaining text with all JSON objects (both complete and fixed)
                removed.
        """
        return self.non_json_text
