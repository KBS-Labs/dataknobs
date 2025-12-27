"""Knowledge base ingestion module.

This module provides configuration and processing for ingesting
documents from a directory into a knowledge base.
"""

from dataknobs_xization.ingestion.config import (
    FilePatternConfig,
    IngestionConfigError,
    KnowledgeBaseConfig,
)
from dataknobs_xization.ingestion.processor import (
    DirectoryProcessor,
    ProcessedDocument,
    process_directory,
)

__all__ = [
    # Config
    "FilePatternConfig",
    "IngestionConfigError",
    "KnowledgeBaseConfig",
    # Processor
    "DirectoryProcessor",
    "ProcessedDocument",
    "process_directory",
]
