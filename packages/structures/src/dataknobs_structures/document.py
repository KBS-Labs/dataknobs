"""Text and metadata containers for document processing.

This module provides classes for managing text documents with associated metadata,
useful for text analysis, NLP pipelines, and document processing workflows.

The main classes are:
- MetaData: Generic key-value metadata container
- TextMetaData: Specialized metadata for text documents with IDs and labels
- Text: Wrapper combining text content with its metadata

Typical usage example:

    ```python
    from dataknobs_structures import Text, TextMetaData

    # Create text with metadata
    metadata = TextMetaData(text_id="doc_001", text_label="article")
    doc = Text("This is the document content.", metadata)

    print(doc.text_id)     # "doc_001"
    print(doc.text_label)  # "article"
    print(doc.text)        # "This is the document content."
    ```
"""

from typing import Any

# Key text metadata attributes
TEXT_ID_ATTR = "text_id"
TEXT_LABEL_ATTR = "text_label"
TEXT_LABEL = "text"


class MetaData:
    """Generic container for managing key-value metadata.

    Stores metadata as a dictionary and provides convenient access methods.
    The metadata is divided into required "key" data and optional additional values.

    Attributes:
        data: Dictionary containing all metadata key-value pairs.

    Example:
        ```python
        # Create metadata with required and optional fields
        meta = MetaData(
            key_data={"id": "123", "type": "document"},
            author="John Doe",
            created_at="2025-01-01"
        )

        print(meta.data)  # {"id": "123", "type": "document",
                         #  "author": "John Doe", "created_at": "2025-01-01"}
        print(meta.get_value("author"))  # "John Doe"
        print(meta.get_value("missing", "default"))  # "default"
        ```
    """

    def __init__(self, key_data: dict[str, Any], **kwargs: Any) -> None:
        """Initialize with mandatory key data and optional additional values.

        Args:
            key_data: Required metadata dictionary. Will be copied to avoid
                external mutations.
            **kwargs: Additional optional metadata key-value pairs to include.

        Example:
            ```python
            meta = MetaData(
                {"id": "123"},
                category="news",
                language="en"
            )
            ```
        """
        self._data = key_data.copy() if key_data is not None else {}
        if kwargs is not None:
            self._data.update(kwargs)

    @property
    def data(self) -> dict[str, Any]:
        """The metadata dictionary.

        Returns:
            Dictionary containing all metadata key-value pairs.
        """
        return self._data

    def get_value(self, attribute: str, missing: str | None = None) -> Any:
        """Get the value for a metadata attribute.

        Args:
            attribute: The metadata key to retrieve.
            missing: Default value to return if the attribute doesn't exist.
                Defaults to None.

        Returns:
            The attribute's value if it exists, otherwise the missing value.

        Example:
            ```python
            meta = MetaData({"id": "123"}, status="active")
            print(meta.get_value("status"))          # "active"
            print(meta.get_value("missing"))         # None
            print(meta.get_value("missing", "N/A"))  # "N/A"
            ```
        """
        return self.data.get(attribute, missing)


class TextMetaData(MetaData):
    """Specialized metadata container for text documents.

    Extends MetaData to provide standard text document metadata fields including
    a unique identifier (text_id) and a label/category (text_label).

    Attributes:
        text_id: Unique identifier for the text document.
        text_label: Label or category for the text (e.g., "article", "email").
        data: Full metadata dictionary including text_id, text_label, and any kwargs.

    Example:
        ```python
        # Create text metadata
        meta = TextMetaData(
            text_id="doc_001",
            text_label="article",
            author="Jane Smith",
            word_count=1500
        )

        print(meta.text_id)      # "doc_001"
        print(meta.text_label)   # "article"
        print(meta.get_value("author"))  # "Jane Smith"
        ```
    """

    def __init__(self, text_id: Any, text_label: str = TEXT_LABEL, **kwargs: Any) -> None:
        """Initialize text metadata with ID and label.

        Args:
            text_id: Unique identifier for the text. Can be any type (string, int, etc.).
            text_label: Label or category for the text. Defaults to "text".
            **kwargs: Additional optional metadata key-value pairs.

        Example:
            ```python
            # Minimal metadata
            meta = TextMetaData(text_id="123")

            # With custom label and extra fields
            meta = TextMetaData(
                text_id="doc_001",
                text_label="news_article",
                published="2025-01-01",
                section="technology"
            )
            ```
        """
        super().__init__(
            {
                TEXT_ID_ATTR: text_id,
                TEXT_LABEL_ATTR: text_label,
            },
            **kwargs,
        )

    @property
    def text_id(self) -> Any:
        """The unique identifier for this text.

        Returns:
            The text_id value from metadata.
        """
        return self.data[TEXT_ID_ATTR]

    @property
    def text_label(self) -> str | Any:
        """The label/category for this text.

        Returns:
            The text_label value from metadata.
        """
        return self.data[TEXT_LABEL_ATTR]


class Text:
    """Container combining text content with metadata.

    Wraps a text string with associated metadata for document processing,
    text analysis, and NLP pipelines. Provides convenient access to both
    the text content and its metadata attributes.

    Attributes:
        text: The text content string.
        text_id: Unique identifier from metadata.
        text_label: Label/category from metadata.
        metadata: Full TextMetaData object.

    Example:
        ```python
        # Create text with metadata
        metadata = TextMetaData(
            text_id="email_001",
            text_label="customer_inquiry",
            sender="customer@example.com"
        )
        doc = Text("Hello, I have a question about...", metadata)

        # Access content and metadata
        print(doc.text)        # "Hello, I have a question about..."
        print(doc.text_id)     # "email_001"
        print(doc.text_label)  # "customer_inquiry"
        print(doc.metadata.get_value("sender"))  # "customer@example.com"

        # Minimal usage (auto-creates metadata)
        simple_doc = Text("Some text", None)
        print(simple_doc.text_id)  # 0 (default)
        ```
    """

    def __init__(
        self,
        text: str,
        metadata: TextMetaData | None,
    ) -> None:
        """Initialize text document with content and metadata.

        Args:
            text: The text content string.
            metadata: TextMetaData object with document metadata. If None,
                creates default metadata with text_id=0 and text_label="text".

        Example:
            ```python
            # With metadata
            meta = TextMetaData(text_id="doc1", text_label="article")
            doc = Text("Article content...", meta)

            # Without metadata (uses defaults)
            doc = Text("Content without metadata", None)
            ```
        """
        self._text = text
        self._metadata = metadata if metadata is not None else TextMetaData(0, TEXT_LABEL)

    @property
    def text(self) -> str:
        """The text content.

        Returns:
            The text string.
        """
        return self._text

    @property
    def text_id(self) -> Any:
        """The unique identifier from metadata.

        Returns:
            The text_id value.
        """
        return self.metadata.text_id

    @property
    def text_label(self) -> str:
        """The label/category from metadata.

        Returns:
            The text_label value.
        """
        return self.metadata.text_label

    @property
    def metadata(self) -> TextMetaData:
        """The metadata object.

        Returns:
            The TextMetaData instance containing all metadata.
        """
        return self._metadata
