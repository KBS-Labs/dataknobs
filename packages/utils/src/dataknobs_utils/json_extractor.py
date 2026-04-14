"""Extract and repair JSON objects from text strings.

Provides the JSONExtractor class for finding, extracting, and repairing
JSON objects embedded in text, with categorization of complete vs fixed objects.
"""

import json
import re
from typing import Any

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
        self.complete_jsons: list[dict[str, Any]] = []
        # List of JSON objects that were malformed but fixed
        self.fixed_jsons: list[dict[str, Any]] = []
        # Text that doesn't contain JSON objects
        self.non_json_text: str = ""

    def get_values(self, key_path: str) -> list[Any]:
        """Get values at a specific path from all complete JSON objects.

        Uses dot notation to navigate nested JSON structures and retrieves values
        from all complete JSON objects that contain the specified path.

        Args:
            key_path: Dot-separated path to the value (e.g., 's3.bucket' for
                nested structures).

        Returns:
            list[Any]: List of values found at the path (empty if none found).
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

    def extract_jsons(self, text: str) -> list[dict[str, Any]]:
        """Extract all JSON objects from text, repairing malformed JSON when possible.

        Searches for JSON objects in the text, attempts to parse them, and repairs
        malformed JSON by closing unclosed brackets, quotes, and fixing trailing commas.
        Updates instance attributes with categorized results and remaining non-JSON text.

        Args:
            text: Text string potentially containing JSON objects.

        Returns:
            list[dict[str, Any]]: All successfully extracted and parsed JSON objects
                (both complete and fixed).
        """
        self.complete_jsons = []
        self.fixed_jsons = []
        self.non_json_text = text

        # Find all potential JSON objects using string-aware brace matching
        potential_jsons = self._find_json_objects(text)

        extracted_jsons = self._try_parse_candidates(potential_jsons)

        # Fallback: if non_json_text still contains potential JSON starts,
        # a stray '{' in surrounding text may have absorbed them during the
        # primary scan.  Try scanning from later '{' positions.  This
        # covers both "nothing parsed" and "some parsed but more remain".
        if '{"' in self.non_json_text:
            fallback_jsons = self._fallback_scan(self.non_json_text)
            if fallback_jsons:
                extracted_jsons.extend(self._try_parse_candidates(fallback_jsons))

        # Clean up any remaining JSON brackets in non_json_text
        self.non_json_text = self.non_json_text.strip()

        return extracted_jsons

    def _try_parse_candidates(
        self, candidates: list[tuple[str, bool]]
    ) -> list[dict[str, Any]]:
        """Attempt to parse a list of candidate JSON text fragments.

        Tries ``json.loads`` on each candidate, falling back to
        :meth:`_fix_json` for malformed ones.  Successfully parsed
        objects are appended to :attr:`complete_jsons` or
        :attr:`fixed_jsons` and removed from :attr:`non_json_text`.

        Returns:
            List of successfully parsed JSON objects.
        """
        extracted: list[dict[str, Any]] = []

        for json_text, is_complete in candidates:
            try:
                json_obj = json.loads(json_text)
                if is_complete:
                    self.complete_jsons.append(json_obj)
                else:
                    self.fixed_jsons.append(json_obj)
                extracted.append(json_obj)
                self.non_json_text = self.non_json_text.replace(json_text, "", 1)
            except json.JSONDecodeError:
                fixed_json = self._fix_json(json_text)
                if fixed_json:
                    try:
                        json_obj = json.loads(fixed_json)
                        self.fixed_jsons.append(json_obj)
                        extracted.append(json_obj)
                        self.non_json_text = self.non_json_text.replace(
                            json_text, "", 1
                        )
                    except json.JSONDecodeError:
                        pass

        return extracted

    def _find_json_objects(self, text: str) -> list[tuple[str, bool]]:
        """Primary scan: string-aware brace matching over the full text.

        Args:
            text: Text to search for JSON-like ``{...}`` patterns.

        Returns:
            List of ``(json_text, is_complete)`` tuples.  *is_complete* is
            ``True`` when the braces balanced; ``False`` for a trailing
            fragment that started with ``{`` but never closed.
        """
        result: list[tuple[str, bool]] = []
        stack: list[str] = []
        start_index: int | None = None
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue

            if in_string:
                if char == "\\":
                    escape_next = True
                elif char == '"':
                    in_string = False
                continue

            # Outside of a string
            if char == '"':
                if stack:  # only enter string state inside an object
                    in_string = True
            elif char == "{":
                if start_index is None:
                    start_index = i
                stack.append(char)
            elif char == "}" and stack:
                stack.pop()
                if not stack and start_index is not None:
                    json_text = text[start_index : i + 1]
                    result.append((json_text, True))
                    start_index = None

        # Incomplete JSON that started but didn't close
        if start_index is not None:
            incomplete_json = text[start_index:]
            result.append((incomplete_json, False))

        return result

    def _fallback_scan(self, text: str) -> list[tuple[str, bool]]:
        """Fallback scan: try later ``{`` positions in residual text.

        Called on :attr:`non_json_text` (the residual after the primary
        scan removed any successfully-parsed objects).  A stray ``{`` in
        surrounding prose can swallow the real JSON during the primary
        scan; this method recovers it by trying each ``{`` that looks
        like a JSON object start (``{"`` pattern).

        The first ``{`` in *text* is skipped because it is typically the
        stray brace that caused the primary scan to fail.  Candidates
        are collected from all matching positions so that multiple
        layers of brace-wrapping (``{{{...}}}``) are handled — a
        double-wrapped candidate will fail ``json.loads``, but the
        deeper single-wrapped candidate will succeed.

        After finding candidates at a given position, the search
        advances past the end of the first found object to avoid
        extracting nested inner objects as separate duplicates.

        Note:
            Residual wrapper fragments (e.g. ``{stray }``) may remain
            in :attr:`non_json_text` after the inner JSON is extracted.
            These are original prose characters, not lost JSON.
        """
        first_brace = text.find("{")
        if first_brace == -1:
            return []
        search_start = first_brace + 1

        all_candidates: list[tuple[str, bool]] = []
        while True:
            idx = text.find("{", search_start)
            if idx == -1:
                break
            # Quick heuristic: a JSON object starts with {"
            # lstrip('{') peels any depth of brace-wrapping (e.g. {{{" )
            # so the check works for both direct {"key" and wrapped {{"key".
            remaining = text[idx:]
            stripped = remaining.lstrip("{")
            if stripped and stripped[0] == '"':
                candidates = self._find_json_objects(remaining)
                if candidates:
                    all_candidates.extend(candidates)
                    first_text = candidates[0][0]
                    # Skip past the found object only if it starts with {"
                    # (likely real JSON).  Brace-wrapped candidates (starting
                    # with "{{") will fail json.loads, so we must NOT skip —
                    # the next position may hold the real inner object.
                    if first_text.startswith('{"'):
                        search_start = idx + len(first_text)
                        continue
            search_start = idx + 1

        return all_candidates

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

    def get_complete_jsons(self) -> list[dict[str, Any]]:
        """Get all well-formed JSON objects extracted from text.

        Returns:
            list[dict[str, Any]]: List of JSON objects that were successfully
                parsed without requiring repairs.
        """
        return self.complete_jsons

    def get_fixed_jsons(self) -> list[dict[str, Any]]:
        """Get all JSON objects that were repaired from malformed state.

        Returns:
            list[dict[str, Any]]: List of JSON objects that required repair
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
