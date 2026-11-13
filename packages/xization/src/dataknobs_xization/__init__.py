# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Text normalization, annotation, conversion, chunking and ingestion tools.

**Deprecated, and removed in the next major release:** the
:mod:`~dataknobs_xization.authorities`, :mod:`~dataknobs_xization.lexicon` and
:mod:`~dataknobs_xization.masking_tokenizer` modules, and with the last of them
``CharacterFeatures`` and ``TextFeatures``. They are not imported with the
package. Reaching one through it -- ``dataknobs_xization.lexicon``,
``from dataknobs_xization import TextFeatures`` -- imports it then, and warns.
"""

import inspect
import sys
from typing import Any

from dataknobs_xization import (
    _deprecation,
    annotations,
    content_transformer,
    html,
    ingestion,
    json,
    markdown,
    normalize,
)
from dataknobs_xization.content_transformer import (
    ContentTransformer,
    csv_to_markdown,
    json_to_markdown,
    yaml_to_markdown,
)
from dataknobs_xization.html import (
    HTMLConverter,
    HTMLConverterConfig,
    html_to_markdown,
)
from dataknobs_xization.markdown import (
    AdaptiveStreamingProcessor,
    Chunk,
    ChunkFormat,
    ChunkMetadata,
    ChunkQualityConfig,
    ChunkQualityFilter,
    EnrichedChunkData,
    HeadingInclusion,
    MarkdownChunker,
    MarkdownNode,
    MarkdownParser,
    StreamingMarkdownProcessor,
    build_enriched_text,
    chunk_markdown_tree,
    format_heading_display,
    get_dynamic_heading_display,
    is_multiword,
    parse_markdown,
    stream_markdown_file,
    stream_markdown_string,
)
from dataknobs_xization.chunking import (
    ChunkTransform,
    Chunker,
    CompositeChunker,
    DocumentInfo,
    MarkdownTreeChunker,
    MergeSmallChunks,
    QualityFilterTransform,
    SplitLargeChunks,
    chunker_registry,
    create_chunker,
    register_chunker,
    register_transform,
    split_text,
    transform_registry,
)
from dataknobs_xization.json import (
    JSONChunk,
    JSONChunkConfig,
    JSONChunker,
)
from dataknobs_xization.ingestion import (
    DirectoryProcessor,
    FilePatternConfig,
    IngestionConfigError,
    KnowledgeBaseConfig,
    ProcessedDocument,
    process_directory,
)

__version__ = "2.2.1"

__all__ = [
    # Existing exports
    "annotations",
    "content_transformer",
    "normalize",
    # Content transformation
    "ContentTransformer",
    "csv_to_markdown",
    "json_to_markdown",
    "yaml_to_markdown",
    # HTML conversion
    "html",
    "HTMLConverter",
    "HTMLConverterConfig",
    "html_to_markdown",
    # JSON module
    "json",
    "JSONChunk",
    "JSONChunkConfig",
    "JSONChunker",
    # Markdown module
    "markdown",
    # Markdown chunking classes and functions
    "AdaptiveStreamingProcessor",
    "Chunk",
    "ChunkFormat",
    "ChunkMetadata",
    "ChunkQualityConfig",
    "ChunkQualityFilter",
    "EnrichedChunkData",
    "HeadingInclusion",
    "MarkdownChunker",
    "MarkdownNode",
    "MarkdownParser",
    "StreamingMarkdownProcessor",
    "build_enriched_text",
    "chunk_markdown_tree",
    "format_heading_display",
    "get_dynamic_heading_display",
    "is_multiword",
    "parse_markdown",
    "stream_markdown_file",
    "stream_markdown_string",
    # Chunking abstraction
    "ChunkTransform",
    "Chunker",
    "CompositeChunker",
    "DocumentInfo",
    "MarkdownTreeChunker",
    "MergeSmallChunks",
    "QualityFilterTransform",
    "SplitLargeChunks",
    "chunker_registry",
    "create_chunker",
    "register_chunker",
    "register_transform",
    "split_text",
    "transform_registry",
    # Ingestion module
    "ingestion",
    "DirectoryProcessor",
    "FilePatternConfig",
    "IngestionConfigError",
    "KnowledgeBaseConfig",
    "ProcessedDocument",
    "process_directory",
]


def __getattr__(name: str) -> Any:
    """Reach a deprecated module or name lazily, warning that it is deprecated.

    Importing the package does not import the deprecated stack, so it does not
    warn. This is the door to it: the warning is raised here, attributed to the
    caller's line, and the module's own import-time warning is withheld so a
    first access says it once. Any deprecated module *that* module imports
    still warns. A module reached this way becomes an attribute of the
    package, so later accesses of a module skip this function; the two class
    names are never bound, so each access of one warns.

    ``from dataknobs_xization import TextFeatures`` reaches this function
    twice: the import system asks ``hasattr`` first, then the statement reads
    the attribute. Only the second warns, so the statement warns once.
    """
    module_name = _deprecation.PACKAGE_DOOR.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name == module_name or not _asked_by_the_import_system():
        _deprecation.warn_door(name)
    with _deprecation.door_import(module_name):
        _import_deprecated(module_name)
    module = sys.modules[f"{__name__}.{module_name}"]
    return module if name == module_name else getattr(module, name)


def _asked_by_the_import_system() -> bool:
    """Whether ``__getattr__`` was called by the import system's fromlist check.

    For ``from <package> import <name>``, ``importlib._bootstrap._handle_fromlist``
    calls ``hasattr`` on the package before the statement reads the name. A
    module name is bound by that first call and never asked again, but a class
    name is asked both times.
    """
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_back if frame and frame.f_back else None
        return (
            caller is not None
            and caller.f_code.co_name == "_handle_fromlist"
            and "_bootstrap" in caller.f_code.co_filename
        )
    finally:
        del frame


def _import_deprecated(module_name: str) -> None:
    """Import one deprecated module, by a statement naming it.

    A statement, not ``from dataknobs_xization import <module>``: that form
    asks the package for the attribute first, which is this ``__getattr__``,
    and would recurse.
    """
    if module_name == "authorities":
        import dataknobs_xization.authorities
    elif module_name == "lexicon":
        import dataknobs_xization.lexicon
    elif module_name == "masking_tokenizer":
        import dataknobs_xization.masking_tokenizer
    else:
        raise AssertionError(f"no deprecated module {module_name!r}")
