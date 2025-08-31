"""Metadata preservation for DataKnobs-Pandas conversions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from dataknobs_data.records import Record


class MetadataStrategy(Enum):
    """Strategy for handling metadata during conversion."""
    NONE = "none"  # Don't preserve metadata
    ATTRS = "attrs"  # Store in DataFrame.attrs
    COLUMNS = "columns"  # Store as additional columns
    MULTI_INDEX = "multi_index"  # Use multi-level column index


@dataclass
class MetadataConfig:
    """Configuration for metadata handling."""
    strategy: MetadataStrategy = MetadataStrategy.ATTRS
    include_record_metadata: bool = True
    include_field_metadata: bool = True
    metadata_prefix: str = "_meta_"
    preserve_record_ids: bool = True


class MetadataHandler:
    """Handles metadata preservation during conversions."""

    def __init__(self, config: MetadataConfig | None = None):
        """Initialize metadata handler.
        
        Args:
            config: Metadata configuration
        """
        self.config = config or MetadataConfig()

    def extract_metadata_from_records(self, records: list[Record]) -> dict[str, Any]:
        """Extract metadata from records.
        
        Args:
            records: List of records
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "record_count": len(records),
            "has_record_ids": all(r.id for r in records),
            "field_names": self._get_all_field_names(records),
            "field_types": self._get_field_types(records),
        }

        if self.config.include_record_metadata:
            metadata["record_metadata"] = self._extract_record_metadata(records)

        if self.config.include_field_metadata:
            metadata["field_metadata"] = self._extract_field_metadata(records)

        return metadata

    def apply_metadata_to_dataframe(
        self,
        df: pd.DataFrame,
        metadata: dict[str, Any],
        records: list[Record] | None = None
    ) -> pd.DataFrame:
        """Apply metadata to DataFrame based on strategy.
        
        Args:
            df: Target DataFrame
            metadata: Metadata to apply
            records: Original records (for additional metadata)
            
        Returns:
            DataFrame with metadata
        """
        if self.config.strategy == MetadataStrategy.NONE:
            return df

        elif self.config.strategy == MetadataStrategy.ATTRS:
            df.attrs.update(metadata)  # type: ignore[arg-type]
            if records and self.config.preserve_record_ids:
                record_ids = [r.id for r in records]
                df.attrs["record_ids"] = record_ids

        elif self.config.strategy == MetadataStrategy.COLUMNS:
            # Add metadata as columns
            if self.config.include_record_metadata and records:
                for key, values in self._get_record_metadata_columns(records).items():
                    col_name = f"{self.config.metadata_prefix}{key}"
                    df[col_name] = values

        elif self.config.strategy == MetadataStrategy.MULTI_INDEX:
            # Create multi-level column index with metadata
            if "field_types" in metadata:
                arrays = [
                    df.columns.tolist(),
                    [metadata["field_types"].get(col, "unknown") for col in df.columns]
                ]
                df.columns = pd.MultiIndex.from_arrays(
                    arrays,
                    names=["field_name", "field_type"]
                )

        return df

    def extract_metadata_from_dataframe(self, df: pd.DataFrame) -> dict[str, Any]:
        """Extract metadata from DataFrame.
        
        Args:
            df: Source DataFrame
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}

        if self.config.strategy == MetadataStrategy.ATTRS:
            # Convert attrs keys to strings for consistency
            for key, value in df.attrs.items():
                if key is not None:
                    metadata[str(key)] = value

        elif self.config.strategy == MetadataStrategy.COLUMNS:
            # Extract from metadata columns
            meta_cols = [col for col in df.columns if col.startswith(self.config.metadata_prefix)]
            for col in meta_cols:
                key = col.replace(self.config.metadata_prefix, "")
                metadata[key] = df[col].tolist()

        elif self.config.strategy == MetadataStrategy.MULTI_INDEX:
            # Extract from multi-level index
            if isinstance(df.columns, pd.MultiIndex):
                metadata["field_names"] = df.columns.get_level_values(0).tolist()
                if df.columns.nlevels > 1:
                    metadata["field_types"] = df.columns.get_level_values(1).tolist()

        return metadata

    def create_records_with_metadata(
        self,
        df: pd.DataFrame,
        base_records: list[Record],
        metadata: dict[str, Any] | None = None
    ) -> list[Record]:
        """Create records with preserved metadata.
        
        Args:
            df: Source DataFrame
            base_records: Base records from conversion
            metadata: Additional metadata
            
        Returns:
            Records with metadata
        """
        if not metadata:
            metadata = self.extract_metadata_from_dataframe(df)

        # Apply record IDs if preserved
        if "record_ids" in metadata and len(metadata["record_ids"]) == len(base_records):
            for record, record_id in zip(base_records, metadata["record_ids"], strict=False):
                if record_id:
                    record.id = record_id

        # Apply record metadata if present
        if "record_metadata" in metadata:
            record_meta = metadata["record_metadata"]
            for i, record in enumerate(base_records):
                if i < len(record_meta) and record_meta[i]:
                    record.metadata = record_meta[i]

        # Apply field metadata if present
        if "field_metadata" in metadata:
            field_meta = metadata["field_metadata"]
            for record in base_records:
                for field_name, field in record.fields.items():
                    if field_name in field_meta:
                        field.metadata = field_meta[field_name]

        return base_records

    def _get_all_field_names(self, records: list[Record]) -> list[str]:
        """Get all unique field names from records."""
        field_names = set()
        for record in records:
            field_names.update(record.fields.keys())
        return sorted(field_names)

    def _get_field_types(self, records: list[Record]) -> dict[str, str]:
        """Get field types from records."""
        field_types = {}
        for record in records:
            for field_name, field in record.fields.items():
                if field_name not in field_types and field.type:
                    field_types[field_name] = field.type.value
        return field_types

    def _extract_record_metadata(self, records: list[Record]) -> list[dict[str, Any]]:
        """Extract metadata from each record."""
        return [r.metadata if r.metadata else {} for r in records]

    def _extract_field_metadata(self, records: list[Record]) -> dict[str, dict[str, Any]]:
        """Extract metadata from fields."""
        field_metadata = {}
        for record in records:
            for field_name, field in record.fields.items():
                if field.metadata and field_name not in field_metadata:
                    field_metadata[field_name] = field.metadata
        return field_metadata

    def _get_record_metadata_columns(self, records: list[Record]) -> dict[str, list]:
        """Get record metadata as column data."""
        columns = {}

        # Collect all metadata keys
        all_keys = set()
        for record in records:
            if record.metadata:
                all_keys.update(record.metadata.keys())

        # Create column for each metadata key
        for key in all_keys:
            values = []
            for record in records:
                value = record.metadata.get(key) if record.metadata else None
                values.append(value)
            columns[key] = values

        return columns

    def clean_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove metadata columns from DataFrame.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            DataFrame without metadata columns
        """
        if self.config.strategy == MetadataStrategy.COLUMNS:
            # Remove metadata columns
            meta_cols = [col for col in df.columns if col.startswith(self.config.metadata_prefix)]
            return df.drop(columns=meta_cols)

        elif self.config.strategy == MetadataStrategy.MULTI_INDEX:
            # Flatten multi-index to single level
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        return df
