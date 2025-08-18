"""Pandas integration for DataKnobs data package.

This module provides seamless conversion between DataKnobs Records/Fields
and Pandas DataFrames for efficient data analysis and manipulation.
"""

from .converter import DataFrameConverter, ConversionOptions
from .type_mapper import TypeMapper, PandasTypeMapping
from .batch_ops import BatchOperations, ChunkedProcessor, BatchConfig
from .metadata import MetadataHandler, MetadataStrategy

__all__ = [
    "DataFrameConverter",
    "ConversionOptions",
    "TypeMapper",
    "PandasTypeMapping",
    "BatchOperations",
    "BatchConfig",
    "ChunkedProcessor",
    "MetadataHandler",
    "MetadataStrategy",
]