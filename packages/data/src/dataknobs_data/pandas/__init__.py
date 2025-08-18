"""Pandas integration for DataKnobs data package.

This module provides seamless conversion between DataKnobs Records/Fields
and Pandas DataFrames for efficient data analysis and manipulation.
"""

from .batch_ops import BatchConfig, BatchOperations, ChunkedProcessor
from .converter import ConversionOptions, DataFrameConverter
from .metadata import MetadataHandler, MetadataStrategy
from .type_mapper import PandasTypeMapping, TypeMapper

__all__ = [
    "BatchConfig",
    "BatchOperations",
    "ChunkedProcessor",
    "ConversionOptions",
    "DataFrameConverter",
    "MetadataHandler",
    "MetadataStrategy",
    "PandasTypeMapping",
    "TypeMapper",
]
