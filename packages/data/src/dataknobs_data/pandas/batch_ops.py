"""Batch operations for DataKnobs-Pandas integration."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Union

import pandas as pd
import numpy as np

from dataknobs_data.records import Record
from dataknobs_data.query import Query
from dataknobs_data.database import Database, SyncDatabase
from .converter import DataFrameConverter, ConversionOptions

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch operations."""
    chunk_size: int = 1000
    parallel: bool = False
    max_workers: int = 4
    progress_callback: Optional[Callable[[int, int], None]] = None
    error_handling: str = "raise"  # "raise", "skip", "log"
    memory_efficient: bool = True


class ChunkedProcessor:
    """Process DataFrames in chunks for memory efficiency."""
    
    def __init__(self, chunk_size: int = 1000):
        """Initialize chunked processor.
        
        Args:
            chunk_size: Size of each chunk
        """
        self.chunk_size = chunk_size
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        processor: Callable[[pd.DataFrame], Any],
        combine: Optional[Callable[[List[Any]], Any]] = None
    ) -> Any:
        """Process DataFrame in chunks.
        
        Args:
            df: DataFrame to process
            processor: Function to process each chunk
            combine: Function to combine results
            
        Returns:
            Combined results or list of chunk results
        """
        results = []
        
        for chunk in self.iter_chunks(df):
            result = processor(chunk)
            results.append(result)
        
        if combine:
            return combine(results)
        return results
    
    def iter_chunks(self, df: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
        """Iterate over DataFrame in chunks.
        
        Args:
            df: DataFrame to chunk
            
        Yields:
            DataFrame chunks
        """
        for start_idx in range(0, len(df), self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, len(df))
            yield df.iloc[start_idx:end_idx]
    
    def read_csv_chunked(
        self,
        filepath: str,
        processor: Callable[[pd.DataFrame], Any],
        **read_kwargs
    ) -> List[Any]:
        """Read CSV file in chunks and process.
        
        Args:
            filepath: Path to CSV file
            processor: Function to process each chunk
            **read_kwargs: Additional arguments for pd.read_csv
            
        Returns:
            List of processed results
        """
        results = []
        
        for chunk in pd.read_csv(filepath, chunksize=self.chunk_size, **read_kwargs):
            result = processor(chunk)
            results.append(result)
        
        return results


class BatchOperations:
    """Batch operations for DataKnobs databases using DataFrames."""
    
    def __init__(
        self,
        database: Union[Database, SyncDatabase],
        converter: Optional[DataFrameConverter] = None
    ):
        """Initialize batch operations.
        
        Args:
            database: Target database
            converter: DataFrame converter
        """
        self.database = database
        self.converter = converter or DataFrameConverter()
        self.is_async = hasattr(database, 'create') and asyncio.iscoroutinefunction(database.create)
    
    def bulk_insert_dataframe(
        self,
        df: pd.DataFrame,
        config: Optional[BatchConfig] = None,
        conversion_options: Optional[ConversionOptions] = None
    ) -> Dict[str, Any]:
        """Bulk insert DataFrame rows into database.
        
        Args:
            df: DataFrame to insert
            config: Batch configuration
            conversion_options: Options for DataFrame conversion
            
        Returns:
            Insert statistics
        """
        config = config or BatchConfig()
        conversion_options = conversion_options or ConversionOptions()
        
        stats = {
            "total_rows": len(df),
            "inserted": 0,
            "failed": 0,
            "errors": []
        }
        
        # Process in chunks if memory efficient mode
        if config.memory_efficient and len(df) > config.chunk_size:
            processor = ChunkedProcessor(config.chunk_size)
            
            def process_chunk(chunk_df: pd.DataFrame) -> Dict[str, int]:
                return self._insert_chunk(chunk_df, config, conversion_options)
            
            chunk_results = processor.process_dataframe(df, process_chunk)
            
            # Aggregate results
            for result in chunk_results:
                stats["inserted"] += result["inserted"]
                stats["failed"] += result["failed"]
                if "errors" in result:
                    stats["errors"].extend(result["errors"])
        else:
            # Process entire DataFrame at once
            stats = self._insert_chunk(df, config, conversion_options)
        
        return stats
    
    def query_as_dataframe(
        self,
        query: Query,
        conversion_options: Optional[ConversionOptions] = None
    ) -> pd.DataFrame:
        """Execute query and return results as DataFrame.
        
        Args:
            query: Query to execute
            conversion_options: Options for conversion
            
        Returns:
            Query results as DataFrame
        """
        conversion_options = conversion_options or ConversionOptions()
        
        # Execute query
        if self.is_async:
            import asyncio
            records = asyncio.run(self.database.search(query))
        else:
            records = self.database.search(query)
        
        # Convert to DataFrame
        return self.converter.records_to_dataframe(records, conversion_options)
    
    def update_from_dataframe(
        self,
        df: pd.DataFrame,
        id_column: str,
        config: Optional[BatchConfig] = None,
        conversion_options: Optional[ConversionOptions] = None
    ) -> Dict[str, Any]:
        """Update records from DataFrame using ID column.
        
        Args:
            df: DataFrame with updates
            id_column: Column containing record IDs
            config: Batch configuration
            conversion_options: Conversion options
            
        Returns:
            Update statistics
        """
        config = config or BatchConfig()
        conversion_options = conversion_options or ConversionOptions()
        
        stats = {
            "total_rows": len(df),
            "updated": 0,
            "failed": 0,
            "not_found": 0,
            "errors": []
        }
        
        # Ensure ID column exists
        if id_column not in df.columns:
            raise ValueError(f"ID column '{id_column}' not found in DataFrame")
        
        # Convert DataFrame to records
        conversion_options.index_column = id_column
        records = self.converter.dataframe_to_records(df, conversion_options)
        
        # Update each record
        for record in records:
            try:
                if self.is_async:
                    import asyncio
                    success = asyncio.run(self.database.update(record))
                else:
                    success = self.database.update(record)
                
                if success:
                    stats["updated"] += 1
                else:
                    stats["not_found"] += 1
                    
            except Exception as e:
                stats["failed"] += 1
                if config.error_handling == "raise":
                    raise
                elif config.error_handling == "log":
                    logger.error(f"Failed to update record {record.id}: {e}")
                    stats["errors"].append(str(e))
                # else "skip"
                
            # Progress callback
            if config.progress_callback:
                processed = stats["updated"] + stats["failed"] + stats["not_found"]
                config.progress_callback(processed, len(records))
        
        return stats
    
    def aggregate(
        self,
        query: Query,
        aggregations: Dict[str, Union[str, Callable]],
        group_by: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Perform aggregations on query results.
        
        Args:
            query: Query to execute
            aggregations: Dictionary of column: aggregation function
            group_by: Columns to group by
            
        Returns:
            Aggregated DataFrame
        """
        # Get data as DataFrame
        df = self.query_as_dataframe(query)
        
        if df.empty:
            return pd.DataFrame()
        
        # Perform aggregations
        if group_by:
            grouped = df.groupby(group_by)
            return grouped.agg(aggregations)
        else:
            # Single row with aggregations
            result = {}
            for col, agg_func in aggregations.items():
                if col in df.columns:
                    if isinstance(agg_func, str):
                        result[f"{col}_{agg_func}"] = df[col].agg(agg_func)
                    else:
                        result[f"{col}_agg"] = agg_func(df[col])
            return pd.DataFrame([result])
    
    def transform_and_save(
        self,
        query: Query,
        transformer: Callable[[pd.DataFrame], pd.DataFrame],
        config: Optional[BatchConfig] = None
    ) -> Dict[str, Any]:
        """Query, transform with pandas, and save back.
        
        Args:
            query: Query to get records
            transformer: Function to transform DataFrame
            config: Batch configuration
            
        Returns:
            Operation statistics
        """
        config = config or BatchConfig()
        
        # Get data
        df = self.query_as_dataframe(query)
        
        if df.empty:
            return {"total_rows": 0, "transformed": 0}
        
        # Apply transformation
        transformed_df = transformer(df)
        
        # Save back if index preserved (has record IDs)
        if df.index.name == "record_id" and transformed_df.index.name == "record_id":
            return self.update_from_dataframe(
                transformed_df,
                id_column=None,  # Use index
                config=config
            )
        else:
            # Insert as new records
            return self.bulk_insert_dataframe(transformed_df, config)
    
    def _insert_chunk(
        self,
        df: pd.DataFrame,
        config: BatchConfig,
        conversion_options: ConversionOptions
    ) -> Dict[str, Any]:
        """Insert a chunk of DataFrame rows.
        
        Args:
            df: DataFrame chunk
            config: Batch configuration
            conversion_options: Conversion options
            
        Returns:
            Insert statistics for chunk
        """
        stats = {
            "total_rows": len(df),
            "inserted": 0,
            "failed": 0,
            "errors": []
        }
        
        # Convert to records
        records = self.converter.dataframe_to_records(df, conversion_options)
        
        # Insert records
        for i, record in enumerate(records):
            try:
                if self.is_async:
                    import asyncio
                    asyncio.run(self.database.create(record))
                else:
                    self.database.create(record)
                stats["inserted"] += 1
                
            except Exception as e:
                stats["failed"] += 1
                if config.error_handling == "raise":
                    raise
                elif config.error_handling == "log":
                    logger.error(f"Failed to insert row {i}: {e}")
                    stats["errors"].append(str(e))
                # else "skip"
            
            # Progress callback
            if config.progress_callback:
                config.progress_callback(i + 1, len(records))
        
        return stats
    
    def export_to_csv(
        self,
        query: Query,
        filepath: str,
        conversion_options: Optional[ConversionOptions] = None,
        **to_csv_kwargs
    ) -> None:
        """Export query results to CSV file.
        
        Args:
            query: Query to execute
            filepath: Output file path
            conversion_options: Conversion options
            **to_csv_kwargs: Additional arguments for DataFrame.to_csv
        """
        df = self.query_as_dataframe(query, conversion_options)
        df.to_csv(filepath, **to_csv_kwargs)
    
    def export_to_parquet(
        self,
        query: Query,
        filepath: str,
        conversion_options: Optional[ConversionOptions] = None,
        **to_parquet_kwargs
    ) -> None:
        """Export query results to Parquet file.
        
        Args:
            query: Query to execute
            filepath: Output file path
            conversion_options: Conversion options
            **to_parquet_kwargs: Additional arguments for DataFrame.to_parquet
        """
        df = self.query_as_dataframe(query, conversion_options)
        df.to_parquet(filepath, **to_parquet_kwargs)


# Import asyncio only if needed
try:
    import asyncio
except ImportError:
    asyncio = None