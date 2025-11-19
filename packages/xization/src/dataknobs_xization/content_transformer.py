"""Content transformation utilities for converting various formats to markdown.

This module provides tools for converting structured data formats (JSON, YAML, CSV)
into markdown format suitable for RAG ingestion and chunking.

The ContentTransformer supports:
- Generic conversion that preserves structure through heading hierarchy
- Custom schemas for specialized formatting of known data structures
- Nested object and array handling
- Configurable heading levels and formatting options

Example:
    >>> transformer = ContentTransformer()
    >>>
    >>> # Generic conversion
    >>> data = {"name": "Chain of Thought", "description": "Step by step reasoning"}
    >>> markdown = transformer.transform_json(data)
    >>>
    >>> # With custom schema
    >>> transformer.register_schema("pattern", {
    ...     "title_field": "name",
    ...     "sections": [
    ...         {"field": "description", "heading": "Description"},
    ...         {"field": "example", "heading": "Example", "format": "code"}
    ...     ]
    ... })
    >>> markdown = transformer.transform_json(data, schema="pattern")
"""

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ContentTransformer:
    """Transform structured content into markdown for RAG ingestion.

    This class converts various data formats (JSON, YAML, CSV) into well-structured
    markdown that can be parsed by MarkdownParser and chunked by MarkdownChunker.

    The transformer creates markdown with appropriate heading hierarchy so that
    the chunker can create semantic boundaries around logical content units.

    Attributes:
        schemas: Dictionary of registered custom schemas
        config: Transformer configuration options
    """

    def __init__(
        self,
        base_heading_level: int = 2,
        include_field_labels: bool = True,
        code_block_fields: list[str] | None = None,
        list_fields: list[str] | None = None,
    ):
        """Initialize the content transformer.

        Args:
            base_heading_level: Starting heading level for top-level items (default: 2)
            include_field_labels: Whether to bold field names in output (default: True)
            code_block_fields: Field names that should be rendered as code blocks
            list_fields: Field names that should be rendered as bullet lists
        """
        self.base_heading_level = base_heading_level
        self.include_field_labels = include_field_labels
        self.code_block_fields = set(code_block_fields or ["example", "code", "snippet"])
        self.list_fields = set(list_fields or ["items", "steps", "objectives", "symptoms", "solutions"])
        self.schemas: dict[str, dict[str, Any]] = {}

    def register_schema(self, name: str, schema: dict[str, Any]) -> None:
        """Register a custom schema for specialized content conversion.

        Schemas define how to map JSON fields to markdown structure.

        Args:
            name: Schema identifier
            schema: Schema definition with the following structure:
                - title_field: Field to use as the main heading (required)
                - description_field: Field for intro text (optional)
                - sections: List of section definitions, each with:
                    - field: Source field name
                    - heading: Section heading text
                    - format: "text", "code", "list", or "subsections" (default: "text")
                    - language: Code block language (for format="code")
                - metadata_fields: Fields to render as key-value metadata

        Example:
            >>> transformer.register_schema("pattern", {
            ...     "title_field": "name",
            ...     "description_field": "description",
            ...     "sections": [
            ...         {"field": "use_case", "heading": "When to Use"},
            ...         {"field": "example", "heading": "Example", "format": "code"}
            ...     ],
            ...     "metadata_fields": ["category", "difficulty"]
            ... })
        """
        self.schemas[name] = schema
        logger.debug(f"Registered schema: {name}")

    def transform(
        self,
        content: Any,
        format: str = "json",
        schema: str | None = None,
        title: str | None = None,
    ) -> str:
        """Transform content to markdown.

        Args:
            content: Content to transform (dict, list, string, or file path)
            format: Content format - "json", "yaml", or "csv"
            schema: Optional schema name for custom conversion
            title: Optional document title

        Returns:
            Markdown formatted string

        Raises:
            ValueError: If format is not supported
        """
        if format == "json":
            if isinstance(content, (str, Path)):
                with open(content, encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = content
            return self.transform_json(data, schema=schema, title=title)
        elif format == "yaml":
            return self.transform_yaml(content, schema=schema, title=title)
        elif format == "csv":
            return self.transform_csv(content, title=title)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json', 'yaml', or 'csv'.")

    def transform_json(
        self,
        data: dict[str, Any] | list[Any],
        schema: str | None = None,
        title: str | None = None,
    ) -> str:
        """Transform JSON data to markdown.

        Args:
            data: JSON data (dict or list)
            schema: Optional schema name for custom conversion
            title: Optional document title

        Returns:
            Markdown formatted string
        """
        lines: list[str] = []

        # Add document title if provided
        if title:
            lines.extend([f"# {title}", ""])

        # Use custom schema if specified
        if schema and schema in self.schemas:
            return self._transform_with_schema(data, self.schemas[schema], title)

        # Generic transformation
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    lines.extend(self._transform_dict_generic(item, self.base_heading_level))
                    lines.extend(["---", ""])
                else:
                    lines.append(f"- {item}")
                    lines.append("")
        elif isinstance(data, dict):
            lines.extend(self._transform_dict_generic(data, self.base_heading_level))
        else:
            lines.append(str(data))

        return "\n".join(lines)

    def transform_yaml(
        self,
        content: str | Path,
        schema: str | None = None,
        title: str | None = None,
    ) -> str:
        """Transform YAML content to markdown.

        Args:
            content: YAML string or file path
            schema: Optional schema name for custom conversion
            title: Optional document title

        Returns:
            Markdown formatted string

        Raises:
            ImportError: If PyYAML is not installed
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML transformation. Install with: pip install pyyaml") from None

        if isinstance(content, (str, Path)) and Path(content).exists():
            with open(content, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            data = yaml.safe_load(content)

        return self.transform_json(data, schema=schema, title=title)

    def transform_csv(
        self,
        content: str | Path,
        title: str | None = None,
        title_field: str | None = None,
    ) -> str:
        """Transform CSV content to markdown.

        Each row becomes a section with the first column (or title_field) as heading.

        Args:
            content: CSV string or file path
            title: Optional document title
            title_field: Column to use as section title (default: first column)

        Returns:
            Markdown formatted string
        """
        lines: list[str] = []

        if title:
            lines.extend([f"# {title}", ""])

        # Read CSV
        if isinstance(content, Path) or (isinstance(content, str) and Path(content).exists()):
            with open(content, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        else:
            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)

        if not rows:
            return "\n".join(lines)

        # Determine title field
        fieldnames = list(rows[0].keys())
        if title_field and title_field in fieldnames:
            title_col = title_field
        else:
            title_col = fieldnames[0]

        # Transform each row
        for row in rows:
            row_title = row.get(title_col, "Untitled")
            lines.append(f"{'#' * self.base_heading_level} {row_title}")
            lines.append("")

            for field, value in row.items():
                if field == title_col or not value:
                    continue

                if self.include_field_labels:
                    lines.append(f"**{self._format_field_name(field)}**: {value}")
                else:
                    lines.append(value)
                lines.append("")

            lines.extend(["---", ""])

        return "\n".join(lines)

    def _transform_with_schema(
        self,
        data: dict[str, Any] | list[Any],
        schema: dict[str, Any],
        title: str | None = None,
    ) -> str:
        """Transform data using a custom schema.

        Args:
            data: Data to transform (list or dict)
                - List format: [{"name": "Item", ...}, ...]
                - Dict format: {"Item": {...}, ...} (keys become title_field values)
            schema: Schema definition
            title: Optional document title

        Returns:
            Markdown formatted string
        """
        lines: list[str] = []

        if title:
            lines.extend([f"# {title}", ""])

        # Normalize dict-keyed format to list format
        # Dict format: {"Item Name": {"field": "value"}} -> [{"name": "Item Name", "field": "value"}]
        if isinstance(data, dict):
            # Check if this looks like a keyed dict (values are dicts)
            # vs a single item dict (values are primitive)
            if all(isinstance(v, dict) for v in data.values()):
                title_field = schema.get("title_field", "name")
                data = [
                    {title_field: key, **value}
                    for key, value in data.items()
                ]
                logger.debug(f"Normalized dict-keyed data to list format with {len(data)} items")

        items = data if isinstance(data, list) else [data]

        for item in items:
            if not isinstance(item, dict):
                continue

            # Title
            title_field = schema.get("title_field", "name")
            item_title = item.get(title_field, "Untitled")
            lines.append(f"{'#' * self.base_heading_level} {item_title}")
            lines.append("")

            # Metadata fields (rendered as bold key-value pairs)
            metadata_fields = schema.get("metadata_fields", [])
            for field in metadata_fields:
                if item.get(field):
                    formatted_name = self._format_field_name(field)
                    lines.append(f"**{formatted_name}**: {item[field]}")
            if metadata_fields:
                lines.append("")

            # Description field (intro text without heading)
            desc_field = schema.get("description_field")
            if desc_field and desc_field in item and item[desc_field]:
                lines.extend([item[desc_field], ""])

            # Sections
            for section in schema.get("sections", []):
                field = section.get("field")
                if field not in item or not item[field]:
                    continue

                heading = section.get("heading", self._format_field_name(field))
                format_type = section.get("format", "text")

                lines.append(f"{'#' * (self.base_heading_level + 1)} {heading}")
                lines.append("")

                value = item[field]

                if format_type == "code":
                    language = section.get("language", "")
                    lines.append(f"```{language}")
                    lines.append(str(value))
                    lines.append("```")
                elif format_type == "list":
                    if isinstance(value, list):
                        for v in value:
                            lines.append(f"- {v}")
                    else:
                        lines.append(f"- {value}")
                elif format_type == "subsections":
                    # For nested objects
                    if isinstance(value, dict):
                        for k, v in value.items():
                            lines.append(f"**{self._format_field_name(k)}**: {v}")
                    elif isinstance(value, list):
                        for v in value:
                            if isinstance(v, dict):
                                name = v.get("name", v.get("title", "Item"))
                                desc = v.get("description", "")
                                lines.append(f"- **{name}**: {desc}")
                            else:
                                lines.append(f"- {v}")
                else:  # text
                    lines.append(str(value))

                lines.append("")

            lines.extend(["---", ""])

        return "\n".join(lines)

    def _transform_dict_generic(
        self,
        data: dict[str, Any],
        heading_level: int,
    ) -> list[str]:
        """Transform a dictionary to markdown using generic rules.

        Args:
            data: Dictionary to transform
            heading_level: Current heading level

        Returns:
            List of markdown lines
        """
        lines: list[str] = []

        # Try to find a title field
        title = None
        title_candidates = ["name", "title", "id", "key"]
        for candidate in title_candidates:
            if candidate in data and isinstance(data[candidate], str):
                title = data[candidate]
                break

        if title:
            lines.append(f"{'#' * heading_level} {title}")
            lines.append("")

        # Process fields
        for key, value in data.items():
            # Skip title field if we already used it
            if key in title_candidates and key == title:
                continue

            if value is None or value == "":
                continue

            formatted_key = self._format_field_name(key)

            # Handle different value types
            if isinstance(value, dict):
                # Nested object becomes a subsection
                lines.append(f"{'#' * (heading_level + 1)} {formatted_key}")
                lines.append("")
                lines.extend(self._transform_dict_generic(value, heading_level + 2))

            elif isinstance(value, list):
                if key in self.list_fields or all(isinstance(v, str) for v in value):
                    # Render as bullet list
                    lines.append(f"{'#' * (heading_level + 1)} {formatted_key}")
                    lines.append("")
                    for item in value:
                        if isinstance(item, dict):
                            # Complex list item
                            name = item.get("name", item.get("title", str(item)))
                            desc = item.get("description", "")
                            if desc:
                                lines.append(f"- **{name}**: {desc}")
                            else:
                                lines.append(f"- {name}")
                        else:
                            lines.append(f"- {item}")
                    lines.append("")
                else:
                    # List of complex objects
                    lines.append(f"{'#' * (heading_level + 1)} {formatted_key}")
                    lines.append("")
                    for item in value:
                        if isinstance(item, dict):
                            lines.extend(self._transform_dict_generic(item, heading_level + 2))
                        else:
                            lines.append(f"- {item}")
                    lines.append("")

            elif key in self.code_block_fields:
                # Render as code block
                lines.append(f"{'#' * (heading_level + 1)} {formatted_key}")
                lines.append("")
                lines.append("```")
                lines.append(str(value))
                lines.append("```")
                lines.append("")

            else:
                # Simple value
                if self.include_field_labels:
                    lines.append(f"**{formatted_key}**: {value}")
                else:
                    lines.append(str(value))
                lines.append("")

        return lines

    def _format_field_name(self, field: str) -> str:
        """Format a field name for display.

        Converts snake_case and camelCase to Title Case.

        Args:
            field: Field name to format

        Returns:
            Formatted field name
        """
        # Handle snake_case
        words = field.replace("_", " ").replace("-", " ")

        # Handle camelCase
        result = []
        for i, char in enumerate(words):
            if char.isupper() and i > 0 and words[i-1].islower():
                result.append(" ")
            result.append(char)

        return "".join(result).title()


# Convenience function for quick transformations
def json_to_markdown(
    data: dict[str, Any] | list[Any],
    title: str | None = None,
    base_heading_level: int = 2,
) -> str:
    """Convert JSON data to markdown.

    This is a convenience function that creates a ContentTransformer
    and transforms the data in one call.

    Args:
        data: JSON data to transform
        title: Optional document title
        base_heading_level: Starting heading level (default: 2)

    Returns:
        Markdown formatted string

    Example:
        >>> patterns = [
        ...     {"name": "Chain of Thought", "description": "Step by step"},
        ...     {"name": "Few-Shot", "description": "Learning from examples"}
        ... ]
        >>> markdown = json_to_markdown(patterns, title="Prompt Patterns")
    """
    transformer = ContentTransformer(base_heading_level=base_heading_level)
    return transformer.transform_json(data, title=title)


def yaml_to_markdown(
    content: str | Path,
    title: str | None = None,
    base_heading_level: int = 2,
) -> str:
    """Convert YAML content to markdown.

    Args:
        content: YAML string or file path
        title: Optional document title
        base_heading_level: Starting heading level (default: 2)

    Returns:
        Markdown formatted string
    """
    transformer = ContentTransformer(base_heading_level=base_heading_level)
    return transformer.transform_yaml(content, title=title)


def csv_to_markdown(
    content: str | Path,
    title: str | None = None,
    title_field: str | None = None,
    base_heading_level: int = 2,
) -> str:
    """Convert CSV content to markdown.

    Args:
        content: CSV string or file path
        title: Optional document title
        title_field: Column to use as section title
        base_heading_level: Starting heading level (default: 2)

    Returns:
        Markdown formatted string
    """
    transformer = ContentTransformer(base_heading_level=base_heading_level)
    return transformer.transform_csv(content, title=title, title_field=title_field)
