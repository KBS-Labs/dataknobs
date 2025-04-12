import json
import re
from dataknobs.utils.json_utils import get_value
from typing import Dict, List, Tuple, Any, Optional

class JSONExtractor:
    """
    A class that extracts JSON objects from text strings, handling malformed JSON 
    and providing access to extracted objects and non-JSON text.
    """
    
    def __init__(self):
        self.complete_jsons = []  # List of complete, well-formed JSON objects
        self.fixed_jsons = []     # List of JSON objects that were malformed but fixed
        self.non_json_text = ""   # Text that doesn't contain JSON objects
    
    def get_values(self, key_path: str) -> List[Any]:
        """
        Get the specific json values using dot notation from the all json
        objects that have the key_path.
        
        Args:
            key_path: Path to the json value (e.g., 's3.bucket')
            
        Returns:
            The possibly empty list of json values
        """
        values = []
        for json_obj in self.complete_jsons:
            value = get_value(json_obj, key_path)
            if value is not None:
                values.append(value)
        return values
    
    def get_value(self, key_path: str, default: Optional[Any] = None) -> Any:
        """
        Get a specific json value using dot notation from the first json
        object that has the key_path.
        
        Args:
            key_path: Path to the json value (e.g., 's3.bucket')
            default: Default value to return if the key doesn't exist
            
        Returns:
            The json value, or the default if not found
        """
        value = default
        for json_obj in self.complete_jsons:
            cur_value = get_value(json_obj, key_path)
            if cur_value is not None:
                value = cur_value
                break
        return value

    def extract_jsons(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract JSON objects from the given text string.
        
        Args:
            text: The text string to extract JSON objects from
            
        Returns:
            A list of dictionaries representing the extracted JSON objects
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
                self.non_json_text = self.non_json_text.replace(json_text, '', 1)
            except json.JSONDecodeError:
                # If it's malformed, try to fix it
                fixed_json = self._fix_json(json_text)
                if fixed_json:
                    try:
                        json_obj = json.loads(fixed_json)
                        self.fixed_jsons.append(json_obj)
                        extracted_jsons.append(json_obj)
                        
                        # Remove the original JSON text from non_json_text
                        self.non_json_text = self.non_json_text.replace(json_text, '', 1)
                    except json.JSONDecodeError:
                        # If we still can't parse it, leave it in non_json_text
                        pass
        
        # Clean up any remaining JSON brackets in non_json_text
        self.non_json_text = self.non_json_text.strip()
        
        return extracted_jsons
    
    def _find_json_objects(self, text: str) -> List[Tuple[str, bool]]:
        """
        Find potential JSON objects in the text.
        
        Args:
            text: The text to search for JSON objects
            
        Returns:
            A list of tuples containing (json_text, is_complete)
        """
        result = []
        
        # First try to find complete JSON objects
        stack = []
        start_index = None
        
        for i, char in enumerate(text):
            if char == '{' and start_index is None:
                start_index = i
                stack.append(char)
            elif char == '{' and stack:
                stack.append(char)
            elif char == '}' and stack:
                stack.pop()
                if not stack and start_index is not None:
                    # Complete JSON object found
                    json_text = text[start_index:i+1]
                    result.append((json_text, True))
                    start_index = None
        
        # Now look for incomplete JSON objects that started but didn't finish
        if start_index is not None:
            incomplete_json = text[start_index:]
            result.append((incomplete_json, False))
        
        return result
    
    def _fix_json(self, json_text: str) -> Optional[str]:
        """
        Attempt to fix malformed JSON by closing any unclosed brackets, quotes, etc.
        
        Args:
            json_text: The malformed JSON text to fix
            
        Returns:
            Fixed JSON text if possible, None otherwise
        """
        if not json_text:
            return None
        
        if not json_text.startswith('{'):
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
                
            if char == '\\' and in_string:
                escape_next = True
            elif char == '"' and not escape_next:
                in_string = not in_string
            elif char == '{' and not in_string:
                open_braces += 1
            elif char == '}' and not in_string:
                open_braces -= 1
            elif char == '[' and not in_string:
                open_brackets += 1
            elif char == ']' and not in_string:
                open_brackets -= 1
        
        # Fix unclosed strings
        fixed_json = json_text
        if in_string:
            fixed_json += '"'
        
        # Fix unclosed braces and brackets
        fixed_json += ']' * max(0, open_brackets)
        fixed_json += '}' * max(0, open_braces)
        
        # Handle trailing commas before closing braces/brackets
        fixed_json = re.sub(r',\s*\]', ']', fixed_json)
        fixed_json = re.sub(r',\s*}', '}', fixed_json)
        
        return fixed_json
    
    def get_complete_jsons(self) -> List[Dict[str, Any]]:
        """Get a list of complete JSON objects that were extracted."""
        return self.complete_jsons
    
    def get_fixed_jsons(self) -> List[Dict[str, Any]]:
        """Get a list of JSON objects that were malformed but fixed."""
        return self.fixed_jsons
    
    def get_non_json_text(self) -> str:
        """Get the text that doesn't contain JSON objects."""
        return self.non_json_text
