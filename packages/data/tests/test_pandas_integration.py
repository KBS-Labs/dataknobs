"""Tests for Pandas integration."""

import json
from datetime import datetime
from typing import List

import pytest
import pandas as pd
import numpy as np

from dataknobs_data.records import Record
from dataknobs_data.fields import Field, FieldType
from dataknobs_data.query import Query
from dataknobs_data.backends.memory import SyncMemoryDatabase
from dataknobs_data.pandas import (
    DataFrameConverter,
    ConversionOptions,
    TypeMapper,
    BatchOperations,
    ChunkedProcessor,
    MetadataHandler,
    MetadataStrategy,
)


class TestTypeMapper:
    """Test TypeMapper functionality."""
    
    def test_field_type_to_pandas(self):
        """Test converting FieldType to pandas dtype."""
        mapper = TypeMapper()
        
        assert mapper.field_type_to_pandas(FieldType.STRING) == "string"
        assert mapper.field_type_to_pandas(FieldType.INTEGER) == "Int64"
        assert mapper.field_type_to_pandas(FieldType.FLOAT) == "Float64"
        assert mapper.field_type_to_pandas(FieldType.BOOLEAN) == "boolean"
        assert mapper.field_type_to_pandas(FieldType.DATETIME) == "datetime64[ns]"
        assert mapper.field_type_to_pandas(FieldType.JSON) == "object"
    
    def test_pandas_to_field_type(self):
        """Test inferring FieldType from pandas dtype."""
        mapper = TypeMapper()
        
        assert mapper.pandas_to_field_type("int64") == FieldType.INTEGER
        assert mapper.pandas_to_field_type("float64") == FieldType.FLOAT
        assert mapper.pandas_to_field_type("bool") == FieldType.BOOLEAN
        assert mapper.pandas_to_field_type("datetime64[ns]") == FieldType.DATETIME
        assert mapper.pandas_to_field_type("string") == FieldType.STRING
        assert mapper.pandas_to_field_type("object") == FieldType.STRING
    
    def test_value_conversion_to_pandas(self):
        """Test converting values to pandas format."""
        mapper = TypeMapper()
        
        # Test None handling
        assert pd.isna(mapper.convert_value_to_pandas(None, FieldType.STRING))
        
        # Test datetime conversion
        dt = datetime(2023, 1, 15, 10, 30)
        result = mapper.convert_value_to_pandas(dt, FieldType.DATETIME)
        assert isinstance(result, pd.Timestamp)
        
        # Test JSON handling
        json_data = {"key": "value"}
        result = mapper.convert_value_to_pandas(json_data, FieldType.JSON)
        assert result == json_data
    
    def test_value_conversion_from_pandas(self):
        """Test converting pandas values to field format."""
        mapper = TypeMapper()
        
        # Test pandas NA
        assert mapper.convert_value_from_pandas(pd.NA, FieldType.STRING) is None
        assert mapper.convert_value_from_pandas(np.nan, FieldType.FLOAT) is None
        
        # Test numpy type conversion
        assert mapper.convert_value_from_pandas(np.int64(42), FieldType.INTEGER) == 42
        assert mapper.convert_value_from_pandas(np.float64(3.14), FieldType.FLOAT) == 3.14
        assert mapper.convert_value_from_pandas(np.bool_(True), FieldType.BOOLEAN) is True
    
    def test_infer_field_type(self):
        """Test field type inference from values."""
        mapper = TypeMapper()
        
        assert mapper.infer_field_type_from_value(None) == FieldType.STRING
        assert mapper.infer_field_type_from_value(True) == FieldType.BOOLEAN
        assert mapper.infer_field_type_from_value(42) == FieldType.INTEGER
        assert mapper.infer_field_type_from_value(3.14) == FieldType.FLOAT
        assert mapper.infer_field_type_from_value(datetime.now()) == FieldType.DATETIME
        assert mapper.infer_field_type_from_value(b"bytes") == FieldType.BINARY
        assert mapper.infer_field_type_from_value({"key": "value"}) == FieldType.JSON
        assert mapper.infer_field_type_from_value([1, 2, 3]) == FieldType.JSON
        assert mapper.infer_field_type_from_value("short") == FieldType.STRING
        assert mapper.infer_field_type_from_value("x" * 1001) == FieldType.TEXT


class TestDataFrameConverter:
    """Test DataFrameConverter functionality."""
    
    def create_test_records(self) -> List[Record]:
        """Create test records."""
        records = []
        
        for i in range(5):
            record = Record(id=f"rec_{i}")
            record.fields["name"] = Field(name="name", value=f"Item {i}", type=FieldType.STRING)
            record.fields["count"] = Field(name="count", value=i * 10, type=FieldType.INTEGER)
            record.fields["price"] = Field(name="price", value=i * 9.99, type=FieldType.FLOAT)
            record.fields["active"] = Field(name="active", value=i % 2 == 0, type=FieldType.BOOLEAN)
            record.fields["created"] = Field(
                name="created",
                value=datetime(2023, 1, i + 1),
                type=FieldType.DATETIME
            )
            record.fields["tags"] = Field(
                name="tags",
                value=["tag1", f"tag{i}"],
                type=FieldType.JSON
            )
            record.metadata = {"source": "test", "version": i}
            records.append(record)
        
        return records
    
    def test_records_to_dataframe_basic(self):
        """Test basic conversion from Records to DataFrame."""
        converter = DataFrameConverter()
        records = self.create_test_records()
        
        df = converter.records_to_dataframe(records)
        
        assert len(df) == 5
        assert "name" in df.columns
        assert "count" in df.columns
        assert "price" in df.columns
        assert "active" in df.columns
        assert "created" in df.columns
        assert "tags" in df.columns
        
        # Check values
        assert df.iloc[0]["name"] == "Item 0"
        assert df.iloc[1]["count"] == 10
        assert df.iloc[2]["price"] == 19.98
        assert df.iloc[3]["active"] == False
        assert isinstance(df.iloc[4]["created"], pd.Timestamp)
    
    def test_records_to_dataframe_with_index(self):
        """Test conversion with record IDs as index."""
        converter = DataFrameConverter()
        records = self.create_test_records()
        
        options = ConversionOptions(preserve_index=True)
        df = converter.records_to_dataframe(records, options)
        
        assert df.index.name == "record_id"
        assert list(df.index) == ["rec_0", "rec_1", "rec_2", "rec_3", "rec_4"]
    
    def test_records_to_dataframe_with_metadata(self):
        """Test conversion with metadata preservation."""
        converter = DataFrameConverter()
        records = self.create_test_records()
        
        options = ConversionOptions(
            include_metadata=True,
            metadata_strategy=MetadataStrategy.ATTRS
        )
        df = converter.records_to_dataframe(records, options)
        
        assert "record_count" in df.attrs
        assert df.attrs["record_count"] == 5
        assert "field_types" in df.attrs
        assert "record_ids" in df.attrs
    
    def test_dataframe_to_records_basic(self):
        """Test basic conversion from DataFrame to Records."""
        converter = DataFrameConverter()
        
        # Create test DataFrame
        data = {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [95.5, 87.3, 92.1],
            "active": [True, False, True]
        }
        df = pd.DataFrame(data)
        df.index = ["id1", "id2", "id3"]
        
        records = converter.dataframe_to_records(df)
        
        assert len(records) == 3
        assert records[0].id == "id1"
        assert records[0].fields["name"].value == "Alice"
        assert records[0].fields["age"].value == 25
        assert records[0].fields["score"].value == 95.5
        assert records[0].fields["active"].value is True
    
    def test_round_trip_conversion(self):
        """Test round-trip conversion accuracy."""
        converter = DataFrameConverter()
        original_records = self.create_test_records()
        
        # Convert to DataFrame and back
        options = ConversionOptions(
            preserve_types=True,
            preserve_index=True,
            include_metadata=True
        )
        df = converter.records_to_dataframe(original_records, options)
        recovered_records = converter.dataframe_to_records(df, options)
        
        assert len(recovered_records) == len(original_records)
        
        for orig, recovered in zip(original_records, recovered_records):
            assert orig.id == recovered.id
            assert set(orig.fields.keys()) == set(recovered.fields.keys())
            
            for field_name in orig.fields:
                orig_value = orig.fields[field_name].value
                recovered_value = recovered.fields[field_name].value
                
                # Handle datetime comparison
                if isinstance(orig_value, datetime):
                    assert isinstance(recovered_value, datetime)
                    assert orig_value.date() == recovered_value.date()
                elif isinstance(orig_value, (list, np.ndarray)):
                    # Handle list/array comparison
                    if isinstance(recovered_value, np.ndarray):
                        np.testing.assert_array_equal(orig_value, recovered_value)
                    else:
                        assert list(orig_value) == list(recovered_value)
                else:
                    assert orig_value == recovered_value
    
    def test_empty_records(self):
        """Test handling of empty records."""
        converter = DataFrameConverter()
        
        df = converter.records_to_dataframe([])
        assert df.empty
        assert isinstance(df, pd.DataFrame)
        
        records = converter.dataframe_to_records(pd.DataFrame())
        assert records == []
    
    def test_missing_values(self):
        """Test handling of missing values."""
        converter = DataFrameConverter()
        
        # Create records with missing fields
        records = []
        r1 = Record(id="r1")
        r1.fields["a"] = Field(name="a", value="value1")
        r1.fields["b"] = Field(name="b", value=10)
        records.append(r1)
        
        r2 = Record(id="r2")
        r2.fields["a"] = Field(name="a", value="value2")
        # Missing field "b"
        r2.fields["c"] = Field(name="c", value=True)
        records.append(r2)
        
        df = converter.records_to_dataframe(records)
        
        assert len(df) == 2
        assert "a" in df.columns
        assert "b" in df.columns
        assert "c" in df.columns
        assert pd.isna(df.loc[df.index[1], "b"])
        assert pd.isna(df.loc[df.index[0], "c"])
    
    def test_type_preservation(self):
        """Test that types are preserved during conversion."""
        converter = DataFrameConverter()
        
        record = Record(id="test")
        record.fields["int_field"] = Field(name="int_field", value=42, type=FieldType.INTEGER)
        record.fields["float_field"] = Field(name="float_field", value=3.14, type=FieldType.FLOAT)
        record.fields["bool_field"] = Field(name="bool_field", value=True, type=FieldType.BOOLEAN)
        record.fields["str_field"] = Field(name="str_field", value="test", type=FieldType.STRING)
        
        options = ConversionOptions(preserve_types=True)
        df = converter.records_to_dataframe([record], options)
        
        assert str(df["int_field"].dtype) == "Int64"
        assert str(df["float_field"].dtype) == "Float64"
        assert str(df["bool_field"].dtype) == "boolean"
        assert str(df["str_field"].dtype) == "string"


class TestBatchOperations:
    """Test batch operations."""
    
    def test_bulk_insert_dataframe(self):
        """Test bulk inserting from DataFrame."""
        db = SyncMemoryDatabase()
        batch_ops = BatchOperations(db)
        
        # Create test DataFrame
        data = {
            "name": [f"Item {i}" for i in range(100)],
            "value": list(range(100)),
            "active": [i % 2 == 0 for i in range(100)]
        }
        df = pd.DataFrame(data)
        
        stats = batch_ops.bulk_insert_dataframe(df)
        
        assert stats["total_rows"] == 100
        assert stats["inserted"] == 100
        assert stats["failed"] == 0
        
        # Verify records in database
        records = db.search(Query())
        assert len(records) == 100
    
    def test_query_as_dataframe(self):
        """Test querying and returning results as DataFrame."""
        db = SyncMemoryDatabase()
        batch_ops = BatchOperations(db)
        
        # Add test records
        for i in range(10):
            record = Record(id=f"rec_{i}")
            record.fields["value"] = Field(name="value", value=i)
            record.fields["category"] = Field(name="category", value="A" if i < 5 else "B")
            db.create(record)
        
        # Query as DataFrame
        query = Query().filter("category", "==", "A")
        df = batch_ops.query_as_dataframe(query)
        
        assert len(df) == 5
        assert "value" in df.columns
        assert "category" in df.columns
        assert all(df["category"] == "A")
    
    def test_aggregation(self):
        """Test aggregation operations."""
        db = SyncMemoryDatabase()
        batch_ops = BatchOperations(db)
        
        # Add test records
        for i in range(20):
            record = Record()
            record.fields["category"] = Field(name="category", value="A" if i < 10 else "B")
            record.fields["value"] = Field(name="value", value=i)
            db.create(record)
        
        # Perform aggregation
        result = batch_ops.aggregate(
            Query(),
            aggregations={"value": ["sum", "mean", "max"]},
            group_by=["category"]
        )
        
        assert len(result) == 2
        assert "sum" in result.columns[0] or "value_sum" in str(result.columns)
        
        # Check aggregated values
        category_a = result.loc[result.index == "A"]
        category_b = result.loc[result.index == "B"]
        
        if not category_a.empty:
            # Sum of 0-9 = 45
            assert category_a["value"]["sum"].iloc[0] == 45
        if not category_b.empty:
            # Sum of 10-19 = 145
            assert category_b["value"]["sum"].iloc[0] == 145
    
    def test_chunked_processing(self):
        """Test chunked processing of large DataFrames."""
        processor = ChunkedProcessor(chunk_size=10)
        
        # Create large DataFrame
        df = pd.DataFrame({
            "value": range(100)
        })
        
        # Process in chunks
        def sum_chunk(chunk):
            return chunk["value"].sum()
        
        results = processor.process_dataframe(df, sum_chunk)
        
        assert len(results) == 10  # 100 rows / 10 per chunk
        assert sum(results) == sum(range(100))
    
    def test_export_to_csv(self, tmp_path):
        """Test exporting query results to CSV."""
        db = SyncMemoryDatabase()
        batch_ops = BatchOperations(db)
        
        # Add test records
        for i in range(5):
            record = Record()
            record.fields["name"] = Field(name="name", value=f"Item {i}")
            record.fields["value"] = Field(name="value", value=i * 10)
            db.create(record)
        
        # Export to CSV
        csv_path = tmp_path / "export.csv"
        batch_ops.export_to_csv(Query(), str(csv_path))
        
        # Verify CSV content
        df = pd.read_csv(csv_path, index_col=0)
        assert len(df) == 5
        assert "name" in df.columns
        assert "value" in df.columns


class TestMetadataHandling:
    """Test metadata preservation."""
    
    def test_metadata_in_attrs(self):
        """Test storing metadata in DataFrame attrs."""
        converter = DataFrameConverter()
        
        records = []
        for i in range(3):
            record = Record(id=f"id_{i}")
            record.fields["value"] = Field(name="value", value=i)
            record.metadata = {"source": "test", "index": i}
            records.append(record)
        
        options = ConversionOptions(
            include_metadata=True,
            metadata_strategy=MetadataStrategy.ATTRS
        )
        df = converter.records_to_dataframe(records, options)
        
        assert "record_metadata" in df.attrs
        assert len(df.attrs["record_metadata"]) == 3
        assert df.attrs["record_ids"] == ["id_0", "id_1", "id_2"]
    
    def test_metadata_round_trip(self):
        """Test metadata preservation in round-trip conversion."""
        converter = DataFrameConverter()
        
        # Create record with metadata
        record = Record(id="test_id")
        record.fields["value"] = Field(
            name="value",
            value=42,
            metadata={"unit": "meters", "precision": 2}
        )
        record.metadata = {"created_by": "test", "version": 1}
        
        options = ConversionOptions(
            include_metadata=True,
            preserve_index=True,
            metadata_strategy=MetadataStrategy.ATTRS
        )
        
        # Round trip
        df = converter.records_to_dataframe([record], options)
        recovered = converter.dataframe_to_records(df, options)
        
        assert len(recovered) == 1
        assert recovered[0].id == "test_id"
        assert recovered[0].metadata == {"created_by": "test", "version": 1}