"""Tests for pandas integration functionality."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional

from dataknobs_data.pandas.converter import DataFrameConverter, ConversionOptions
from dataknobs_data.pandas.batch_ops import BatchOperations, BatchConfig
from dataknobs_data.pandas.type_mapper import TypeMapper
from dataknobs_data.records import Record, Field


class Status(Enum):
    """Test enum for field types."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


class TestDataFrameConverter:
    """Test DataFrameConverter functionality."""
    
    def test_dataframe_to_records_basic(self):
        """Test basic DataFrame to records conversion."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "active": [True, False, True]
        })
        
        converter = DataFrameConverter()
        records = converter.dataframe_to_records(df)
        
        assert len(records) == 3
        assert records[0].get_value("id") == 1
        assert records[0].get_value("name") == "Alice"
        assert records[0].get_value("age") == 25
        assert records[0].get_value("active") is True
    
    def test_dataframe_to_records_with_nulls(self):
        """Test conversion with null values."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", None, "Charlie"],
            "value": [10.5, np.nan, 20.0]
        })
        
        converter = DataFrameConverter()
        records = converter.dataframe_to_records(df)
        
        assert len(records) == 3
        assert records[1].get_value("name") is None
        assert pd.isna(records[1].get_value("value"))
    
    def test_dataframe_to_records_with_metadata(self):
        """Test conversion with metadata column."""
        df = pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "_metadata": [{"created": "2024-01-01"}, {"created": "2024-01-02"}]
        })
        
        converter = DataFrameConverter()
        options = ConversionOptions(metadata_columns=["_metadata"])
        records = converter.dataframe_to_records(df, options)
        
        assert len(records) == 2
        assert records[0].metadata["created"] == "2024-01-01"
        assert records[1].metadata["created"] == "2024-01-02"
    
    def test_dataframe_to_records_with_index(self):
        """Test conversion with DataFrame index as ID."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "age": [25, 30]
        })
        df.index = ["user1", "user2"]
        df.index.name = "user_id"
        
        converter = DataFrameConverter()
        options = ConversionOptions(use_index_as_id=True)
        records = converter.dataframe_to_records(df, options)
        
        assert len(records) == 2
        assert records[0].id == "user1"
        assert records[1].id == "user2"
    
    def test_records_to_dataframe_basic(self):
        """Test basic records to DataFrame conversion."""
        records = [
            Record({"id": 1, "name": "Alice", "age": 25}),
            Record({"id": 2, "name": "Bob", "age": 30}),
            Record({"id": 3, "name": "Charlie", "age": 35})
        ]
        
        converter = DataFrameConverter()
        df = converter.records_to_dataframe(records)
        
        assert len(df) == 3
        assert list(df.columns) == ["id", "name", "age"]
        assert df.iloc[0]["name"] == "Alice"
    
    def test_records_to_dataframe_with_nested(self):
        """Test conversion with nested fields."""
        records = [
            Record({"id": 1, "data": {"name": "Alice", "age": 25}}),
            Record({"id": 2, "data": {"name": "Bob", "age": 30}})
        ]
        
        converter = DataFrameConverter()
        options = ConversionOptions(flatten_nested=True)
        df = converter.records_to_dataframe(records, options)
        
        assert "data.name" in df.columns
        assert "data.age" in df.columns
        assert df.iloc[0]["data.name"] == "Alice"
    
    def test_records_to_dataframe_with_metadata(self):
        """Test conversion including metadata."""
        records = [
            Record({"id": 1, "name": "Alice"}, metadata={"created": "2024-01-01"}),
            Record({"id": 2, "name": "Bob"}, metadata={"created": "2024-01-02"})
        ]
        
        converter = DataFrameConverter()
        options = ConversionOptions(include_metadata=True)
        df = converter.records_to_dataframe(records, options)
        
        assert "_metadata" in df.columns
        assert df.iloc[0]["_metadata"]["created"] == "2024-01-01"
    
    def test_bidirectional_conversion(self):
        """Test converting DataFrame to records and back."""
        original_df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [95.5, 87.3, 92.1]
        })
        
        converter = DataFrameConverter()
        records = converter.dataframe_to_records(original_df)
        recovered_df = converter.records_to_dataframe(records)
        
        # Check that we recover the same data
        pd.testing.assert_frame_equal(original_df, recovered_df)
    
    def test_complex_types_conversion(self):
        """Test conversion of complex data types."""
        df = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-01 12:00:00")],
            "date": [date(2024, 1, 1)],
            "timedelta": [timedelta(days=1)],
            "category": pd.Categorical(["A"]),
            "object": [{"key": "value"}]
        })
        
        converter = DataFrameConverter()
        records = converter.dataframe_to_records(df)
        
        assert len(records) == 1
        record = records[0]
        
        # Check type conversions
        assert isinstance(record.get_value("datetime"), pd.Timestamp)
        assert isinstance(record.get_value("date"), date)
        assert isinstance(record.get_value("timedelta"), timedelta)
        assert record.get_value("category") == "A"
        assert record.get_value("object") == {"key": "value"}


class TestTypeMapper:
    """Test TypeMapper functionality."""
    
    def test_infer_field_type_basic(self):
        """Test basic field type inference."""
        mapper = TypeMapper()
        
        # Integer
        field_type = mapper.infer_field_type(pd.Series([1, 2, 3]))
        assert field_type == "integer"
        
        # Float
        field_type = mapper.infer_field_type(pd.Series([1.5, 2.5, 3.5]))
        assert field_type == "number"
        
        # String
        field_type = mapper.infer_field_type(pd.Series(["a", "b", "c"]))
        assert field_type == "string"
        
        # Boolean
        field_type = mapper.infer_field_type(pd.Series([True, False, True]))
        assert field_type == "boolean"
    
    def test_infer_field_type_datetime(self):
        """Test datetime field type inference."""
        mapper = TypeMapper()
        
        # Datetime
        series = pd.Series([
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-02")
        ])
        field_type = mapper.infer_field_type(series)
        assert field_type == "datetime"
        
        # Date
        series = pd.Series([date(2024, 1, 1), date(2024, 1, 2)])
        field_type = mapper.infer_field_type(series)
        assert field_type == "date"
        
        # Time
        series = pd.Series([time(12, 0), time(13, 0)])
        field_type = mapper.infer_field_type(series)
        assert field_type == "time"
    
    def test_infer_field_type_mixed(self):
        """Test inference with mixed types."""
        mapper = TypeMapper()
        
        # Mixed numeric becomes float
        series = pd.Series([1, 2.5, 3])
        field_type = mapper.infer_field_type(series)
        assert field_type == "number"
        
        # Mixed with strings becomes string
        series = pd.Series([1, "two", 3])
        field_type = mapper.infer_field_type(series)
        assert field_type == "string"
    
    def test_get_pandas_dtype(self):
        """Test pandas dtype mapping."""
        mapper = TypeMapper()
        
        assert mapper.get_pandas_dtype("integer") == "int64"
        assert mapper.get_pandas_dtype("number") == "float64"
        assert mapper.get_pandas_dtype("string") == "object"
        assert mapper.get_pandas_dtype("boolean") == "bool"
        assert mapper.get_pandas_dtype("datetime") == "datetime64[ns]"
    
    def test_convert_value_basic(self):
        """Test basic value conversion."""
        mapper = TypeMapper()
        
        # Integer conversion
        assert mapper.convert_value(1.5, "integer") == 1
        assert mapper.convert_value("42", "integer") == 42
        
        # Float conversion
        assert mapper.convert_value(1, "number") == 1.0
        assert mapper.convert_value("3.14", "number") == 3.14
        
        # String conversion
        assert mapper.convert_value(42, "string") == "42"
        assert mapper.convert_value(True, "string") == "True"
        
        # Boolean conversion
        assert mapper.convert_value(1, "boolean") is True
        assert mapper.convert_value(0, "boolean") is False
    
    def test_convert_value_datetime(self):
        """Test datetime value conversion."""
        mapper = TypeMapper()
        
        # String to datetime
        result = mapper.convert_value("2024-01-01", "datetime")
        assert isinstance(result, pd.Timestamp)
        
        # String to date
        result = mapper.convert_value("2024-01-01", "date")
        assert isinstance(result, date)
        
        # String to time
        result = mapper.convert_value("12:30:00", "time")
        assert isinstance(result, time)
    
    def test_get_optimal_dtype(self):
        """Test optimal dtype detection."""
        mapper = TypeMapper()
        
        # Integer series
        series = pd.Series([1, 2, 3, 4, 5])
        dtype = mapper.get_optimal_dtype(series)
        assert dtype == "int8"  # Small integers fit in int8
        
        # Large integers
        series = pd.Series([1000000, 2000000, 3000000])
        dtype = mapper.get_optimal_dtype(series)
        assert dtype == "int32"
        
        # Float series
        series = pd.Series([1.5, 2.5, 3.5])
        dtype = mapper.get_optimal_dtype(series)
        assert dtype == "float32" or dtype == "float64"
        
        # String series
        series = pd.Series(["a", "b", "c"])
        dtype = mapper.get_optimal_dtype(series)
        assert dtype == "object"
        
        # Boolean series
        series = pd.Series([True, False, True])
        dtype = mapper.get_optimal_dtype(series)
        assert dtype == "bool"
    
    def test_cast_dataframe_dtypes(self):
        """Test DataFrame dtype casting."""
        mapper = TypeMapper()
        
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True]
        })
        
        dtype_map = {
            "int_col": "int32",
            "float_col": "float32",
            "str_col": "string",
            "bool_col": "bool"
        }
        
        result_df = mapper.cast_dataframe_dtypes(df, dtype_map)
        
        assert result_df["int_col"].dtype == "int32"
        assert result_df["float_col"].dtype == "float32"
        assert result_df["bool_col"].dtype == "bool"
    
    def test_normalize_timezone(self):
        """Test timezone normalization."""
        mapper = TypeMapper()
        
        # Create timezone-aware series
        series = pd.Series([
            pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),
            pd.Timestamp("2024-01-02 12:00:00", tz="UTC")
        ])
        
        # Normalize to different timezone
        normalized = mapper.normalize_timezone(series, "US/Eastern")
        # Check timezone - handle both pytz and standard library timezones
        tz = normalized.dt.tz
        if hasattr(tz, 'zone'):
            assert tz.zone == "US/Eastern"
        else:
            assert str(tz) == "US/Eastern" or "Eastern" in str(tz)
        
        # Normalize naive datetime
        naive_series = pd.Series([
            pd.Timestamp("2024-01-01 12:00:00"),
            pd.Timestamp("2024-01-02 12:00:00")
        ])
        normalized = mapper.normalize_timezone(naive_series, "UTC")
        # Check timezone - handle both pytz and standard library timezones
        tz = normalized.dt.tz
        if hasattr(tz, 'zone'):
            assert tz.zone == "UTC"
        else:
            assert str(tz) == "UTC" or "UTC" in str(tz)


class TestBatchConfig:
    """Test BatchConfig validation and defaults."""
    
    def test_default_config(self):
        """Test default BatchConfig values."""
        config = BatchConfig()
        
        assert config.chunk_size == 1000
        assert config.parallel is False
        assert config.max_workers == 4
        assert config.error_handling == "raise"
        assert config.memory_efficient is True
        assert config.progress_callback is None
    
    def test_custom_config(self):
        """Test custom BatchConfig values."""
        def progress_cb(current, total):
            pass
        
        config = BatchConfig(
            chunk_size=500,
            parallel=True,
            max_workers=8,
            error_handling="log",
            memory_efficient=False,  # Override default
            progress_callback=progress_cb
        )
        
        assert config.chunk_size == 500
        assert config.parallel is True
        assert config.max_workers == 8
        assert config.error_handling == "log"
        assert config.memory_efficient is False
        assert config.progress_callback == progress_cb
    
    def test_invalid_chunk_size(self):
        """Test invalid chunk_size raises error."""
        with pytest.raises(ValueError):
            BatchConfig(chunk_size=0)
        
        with pytest.raises(ValueError):
            BatchConfig(chunk_size=-1)
    
    def test_invalid_error_handling(self):
        """Test invalid error_handling raises error."""
        with pytest.raises(ValueError):
            BatchConfig(error_handling="invalid")


class TestConversionOptions:
    """Test ConversionOptions configuration."""
    
    def test_default_options(self):
        """Test default ConversionOptions values."""
        options = ConversionOptions()
        
        assert options.include_metadata is False
        assert options.metadata_columns == []
        assert options.flatten_nested is False
        assert options.preserve_index is True
        assert options.use_index_as_id is False
        assert options.type_mapping == {}
        assert options.null_handling == "preserve"
        assert options.datetime_format is None
        assert options.timezone is None
    
    def test_custom_options(self):
        """Test custom ConversionOptions values."""
        options = ConversionOptions(
            include_metadata=True,
            metadata_columns=["_meta"],
            flatten_nested=True,
            preserve_index=False,
            use_index_as_id=True,
            type_mapping={"col1": "string"},
            null_handling="drop",
            datetime_format="%Y-%m-%d",
            timezone="UTC"
        )
        
        assert options.include_metadata is True
        assert options.metadata_columns == ["_meta"]
        assert options.flatten_nested is True
        assert options.preserve_index is False
        assert options.use_index_as_id is True
        assert options.type_mapping == {"col1": "string"}
        assert options.null_handling == "drop"
        assert options.datetime_format == "%Y-%m-%d"
        assert options.timezone == "UTC"
    
    def test_merge_metadata(self):
        """Test metadata merging logic."""
        options = ConversionOptions()
        
        # Simple merge
        meta1 = {"a": 1, "b": 2}
        meta2 = {"b": 3, "c": 4}
        merged = options.merge_metadata(meta1, meta2)
        
        assert merged["a"] == 1
        assert merged["b"] == 3  # meta2 overwrites
        assert merged["c"] == 4
        
        # Nested merge
        meta1 = {"nested": {"x": 1}}
        meta2 = {"nested": {"y": 2}}
        merged = options.merge_metadata(meta1, meta2)
        
        assert merged["nested"]["x"] == 1
        assert merged["nested"]["y"] == 2


class TestBatchOperations:
    """Test BatchOperations class."""
    
    def setup_method(self):
        """Set up test batch operations."""
        # Use a real MemoryDatabase instead of mocking
        from dataknobs_data.backends.memory import SyncMemoryDatabase
        self.db = SyncMemoryDatabase()
        self.batch_ops = BatchOperations(self.db)
        
        # Create sample DataFrame
        self.df = pd.DataFrame({
            "id": range(100),
            "value": range(100, 200),
            "category": ["A", "B", "C", "D"] * 25,
        })
    
    def test_bulk_insert_dataframe(self):
        """Test bulk insert of DataFrame."""
        config = BatchConfig(chunk_size=20)
        
        result = self.batch_ops.bulk_insert_dataframe(
            self.df,
            config=config
        )
        
        # Verify results
        assert result["total_rows"] == 100
        assert result["inserted"] == 100
        assert result["failed"] == 0
        
        # Verify records were actually inserted
        from dataknobs_data.query import Query
        all_records = self.db.search(Query())
        assert len(all_records) == 100
    
    def test_query_as_dataframe(self):
        """Test querying database as DataFrame."""
        from dataknobs_data.query import Query
        
        # Insert test records
        for i in range(10):
            record = Record({"id": i, "value": i * 10})
            self.db.create(record)
        
        query = Query().filter("value", ">=", 50)
        df = self.batch_ops.query_as_dataframe(query)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # Values 50, 60, 70, 80, 90
        assert all(df["value"] >= 50)
    
    def test_update_from_dataframe(self):
        """Test updating records from DataFrame."""
        # First insert some records
        for i in range(5):
            record = Record({"id": i, "value": i * 10, "category": "OLD"})
            self.db.create(record)
        
        # Get all records to get their IDs
        from dataknobs_data.query import Query
        records = self.db.search(Query())
        record_ids = [r.id for r in records[:3]]  # Get first 3 IDs
        
        # Create a DataFrame with updates
        update_df = pd.DataFrame({
            "record_id": record_ids,
            "value": [100, 200, 300],
            "category": ["NEW", "NEW", "NEW"]
        })
        
        config = BatchConfig(chunk_size=2)
        result = self.batch_ops.update_from_dataframe(
            update_df,
            id_column="record_id",
            config=config
        )
        
        # Verify updates
        assert result["total_rows"] == 3
        assert result["updated"] == 3
        
        # Check that values were actually updated
        updated_records = self.db.search(Query().filter("category", "==", "NEW"))
        assert len(updated_records) == 3
        values = sorted([r.get_value("value") for r in updated_records])
        assert values == [100, 200, 300]
    
    def test_bulk_insert_with_error(self):
        """Test bulk insert with error handling."""
        # Create a custom database that fails on certain records
        from dataknobs_data.backends.memory import SyncMemoryDatabase
        
        class TestDB(SyncMemoryDatabase):
            def create(self, record):
                # Fail on specific values
                if record.get_value("value") == 150:
                    raise ValueError("Test error")
                return super().create(record)
            
            def create_batch(self, records):
                # Force individual processing by always failing batch
                raise ValueError("Batch creation disabled for testing")
        
        test_db = TestDB()
        batch_ops = BatchOperations(test_db)
        
        # Small DataFrame that includes problematic value
        df = pd.DataFrame({
            "value": [140, 150, 160],  # 150 will fail
            "category": ["A", "B", "C"]
        })
        
        config = BatchConfig(
            chunk_size=5,
            error_handling="log"  # Continue on error
        )
        
        result = batch_ops.bulk_insert_dataframe(df, config=config)
        
        # Should have 2 successful, 1 failed
        assert result["total_rows"] == 3
        assert result["inserted"] == 2
        assert result["failed"] == 1
        assert len(result["errors"]) == 1
    
    def test_parallel_bulk_insert(self):
        """Test parallel bulk insert."""
        config = BatchConfig(
            chunk_size=25,
            parallel=True,
            max_workers=2
        )
        
        result = self.batch_ops.bulk_insert_dataframe(
            self.df,
            config=config
        )
        
        # Should have processed all records
        assert result["total_rows"] == 100
        assert result["inserted"] == 100
        
        # Verify in database
        from dataknobs_data.query import Query
        all_records = self.db.search(Query())
        assert len(all_records) == 100
    
    def test_aggregate(self):
        """Test aggregation functionality."""
        # Insert test data
        for i in range(10):
            record = Record({
                "category": "A" if i < 5 else "B",
                "value": i * 10,
                "count": i
            })
            self.db.create(record)
        
        from dataknobs_data.query import Query
        query = Query()  # Get all records
        
        # Test aggregation without grouping
        agg_result = self.batch_ops.aggregate(
            query,
            aggregations={
                "value": "sum",
                "count": "mean"
            }
        )
        
        assert len(agg_result) == 1
        assert agg_result.iloc[0]["value_sum"] == 450  # Sum of 0+10+20+...+90
        assert agg_result.iloc[0]["count_mean"] == 4.5  # Mean of 0-9
        
        # Test aggregation with grouping
        grouped_result = self.batch_ops.aggregate(
            query,
            aggregations={"value": "sum", "count": "count"},
            group_by=["category"]
        )
        
        assert len(grouped_result) == 2
        # Results should be grouped by category
        assert grouped_result.loc["A"]["value"] == 100  # 0+10+20+30+40
        assert grouped_result.loc["B"]["value"] == 350  # 50+60+70+80+90
    
    def test_transform_and_save(self):
        """Test transform and save functionality."""
        # Insert initial data
        for i in range(5):
            record = Record({"value": i, "doubled": 0})
            self.db.create(record)
        
        from dataknobs_data.query import Query
        
        # Define transformation function
        def double_values(df):
            df["doubled"] = df["value"] * 2
            return df
        
        # Transform and save
        result = self.batch_ops.transform_and_save(
            Query(),
            transformer=double_values,
            config=BatchConfig(chunk_size=2)
        )
        
        assert result["total_rows"] == 5
        
        # Verify transformation was applied
        all_records = self.db.search(Query())
        # Since we're inserting new records (not updating), we'll have 10 total
        assert len(all_records) >= 5
        
        # Check that doubled values exist
        for record in all_records[-5:]:  # Check last 5 records (newly inserted)
            if record.get_value("value") is not None:
                expected_doubled = record.get_value("value") * 2
                assert record.get_value("doubled") == expected_doubled
    
    def test_export_to_csv(self, tmp_path):
        """Test CSV export functionality."""
        # Insert test data
        for i in range(5):
            record = Record({"id": i, "name": f"item_{i}"})
            self.db.create(record)
        
        from dataknobs_data.query import Query
        
        # Export to CSV
        csv_path = tmp_path / "export.csv"
        self.batch_ops.export_to_csv(
            Query(),
            str(csv_path),
            index=False
        )
        
        # Verify CSV was created and contains data
        import_df = pd.read_csv(csv_path)
        assert len(import_df) == 5
        assert "id" in import_df.columns
        assert "name" in import_df.columns
    
    def test_export_to_parquet(self, tmp_path):
        """Test Parquet export functionality."""
        # Skip test if pyarrow is not installed
        pytest.importorskip("pyarrow", reason="pyarrow not installed")
        
        # Insert test data
        for i in range(5):
            record = Record({"id": i, "value": i * 100})
            self.db.create(record)
        
        from dataknobs_data.query import Query
        
        # Export to Parquet
        parquet_path = tmp_path / "export.parquet"
        self.batch_ops.export_to_parquet(
            Query(),
            str(parquet_path)
        )
        
        # Verify Parquet was created and contains data
        import_df = pd.read_parquet(parquet_path)
        assert len(import_df) == 5
        assert "id" in import_df.columns
        assert "value" in import_df.columns