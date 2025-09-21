"""Comprehensive unit tests for file_utils module."""

import asyncio
import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from dataknobs_fsm.utils.file_utils import (
    create_csv_writer,
    create_file_reader,
    create_file_writer,
    create_json_writer,
    create_jsonl_writer,
    detect_format,
    get_csv_delimiter,
    read_csv_file,
    read_json_file,
    read_jsonl_file,
    read_text_file,
)


class TestFormatDetection:
    """Test format detection functions."""

    def test_detect_format_jsonl(self):
        """Test JSONL format detection."""
        assert detect_format("data.jsonl") == "jsonl"
        assert detect_format("data.ndjson") == "jsonl"
        assert detect_format("/path/to/file.JSONL") == "jsonl"

    def test_detect_format_json(self):
        """Test JSON format detection."""
        assert detect_format("data.json") == "json"
        assert detect_format("/path/to/file.JSON") == "json"

    def test_detect_format_csv(self):
        """Test CSV format detection."""
        assert detect_format("data.csv") == "csv"
        assert detect_format("data.tsv") == "csv"
        assert detect_format("/path/to/file.CSV") == "csv"

    def test_detect_format_text(self):
        """Test text format detection."""
        assert detect_format("data.txt") == "text"
        assert detect_format("data.text") == "text"
        assert detect_format("data.log") == "text"
        assert detect_format("/path/to/file.TXT") == "text"

    def test_detect_format_unknown(self):
        """Test unknown format detection."""
        # For input, unknown defaults to text
        assert detect_format("data.xyz") == "text"
        assert detect_format("data") == "text"

        # For output, unknown defaults to jsonl
        assert detect_format("data.xyz", for_output=True) == "jsonl"
        assert detect_format("data", for_output=True) == "jsonl"

    def test_get_csv_delimiter(self):
        """Test CSV delimiter detection."""
        assert get_csv_delimiter("data.csv") == ","
        assert get_csv_delimiter("data.tsv") == "\t"
        assert get_csv_delimiter("data.txt") == ","  # Default
        assert get_csv_delimiter("/path/to/file.TSV") == "\t"


class TestFileReaders:
    """Test file reading functions."""

    @pytest.mark.asyncio
    async def test_read_jsonl_file(self, tmp_path):
        """Test reading JSONL file."""
        # Create test JSONL file
        jsonl_file = tmp_path / "test.jsonl"
        test_data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Charlie"}
        ]

        with open(jsonl_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            # Add empty line and malformed JSON to test error handling
            f.write('\n')
            f.write('invalid json\n')

        # Read the file
        results = []
        async for record in read_jsonl_file(jsonl_file):
            results.append(record)

        assert len(results) == 3
        assert results == test_data

    @pytest.mark.asyncio
    async def test_read_json_file_array(self, tmp_path):
        """Test reading JSON file with array."""
        json_file = tmp_path / "test.json"
        test_data = [
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"}
        ]

        with open(json_file, 'w') as f:
            json.dump(test_data, f)

        results = []
        async for record in read_json_file(json_file):
            results.append(record)

        assert len(results) == 2
        assert results == test_data

    @pytest.mark.asyncio
    async def test_read_json_file_object(self, tmp_path):
        """Test reading JSON file with single object."""
        json_file = tmp_path / "test.json"
        test_data = {"status": "success", "count": 42}

        with open(json_file, 'w') as f:
            json.dump(test_data, f)

        results = []
        async for record in read_json_file(json_file):
            results.append(record)

        assert len(results) == 1
        assert results[0] == test_data

    @pytest.mark.asyncio
    async def test_read_csv_file_with_header(self, tmp_path):
        """Test reading CSV file with header."""
        csv_file = tmp_path / "test.csv"

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'age', 'city'])
            writer.writerow(['Alice', '30', 'New York'])
            writer.writerow(['Bob', '25', 'San Francisco'])

        results = []
        async for record in read_csv_file(csv_file, delimiter=',', has_header=True):
            results.append(record)

        assert len(results) == 2
        assert results[0] == {'name': 'Alice', 'age': '30', 'city': 'New York'}
        assert results[1] == {'name': 'Bob', 'age': '25', 'city': 'San Francisco'}

    @pytest.mark.asyncio
    async def test_read_csv_file_without_header(self, tmp_path):
        """Test reading CSV file without header."""
        csv_file = tmp_path / "test.csv"

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Alice', '30', 'New York'])
            writer.writerow(['Bob', '25', 'San Francisco'])

        results = []
        async for record in read_csv_file(csv_file, delimiter=',', has_header=False):
            results.append(record)

        assert len(results) == 2
        assert results[0] == {'col_0': 'Alice', 'col_1': '30', 'col_2': 'New York'}
        assert results[1] == {'col_0': 'Bob', 'col_1': '25', 'col_2': 'San Francisco'}

    @pytest.mark.asyncio
    async def test_read_tsv_file(self, tmp_path):
        """Test reading TSV file."""
        tsv_file = tmp_path / "test.tsv"

        with open(tsv_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['name', 'score'])
            writer.writerow(['Alice', '95'])
            writer.writerow(['Bob', '87'])

        results = []
        async for record in read_csv_file(tsv_file, delimiter='\t', has_header=True):
            results.append(record)

        assert len(results) == 2
        assert results[0] == {'name': 'Alice', 'score': '95'}

    @pytest.mark.asyncio
    async def test_read_text_file(self, tmp_path):
        """Test reading text file."""
        text_file = tmp_path / "test.txt"

        with open(text_file, 'w') as f:
            f.write("Line 1\n")
            f.write("Line 2\n")
            f.write("\n")  # Empty line
            f.write("Line 3\n")

        # Test with skipping empty lines
        results = []
        async for record in read_text_file(text_file, field_name='content', skip_empty=True):
            results.append(record)

        assert len(results) == 3
        assert results[0] == {'content': 'Line 1'}
        assert results[1] == {'content': 'Line 2'}
        assert results[2] == {'content': 'Line 3'}

        # Test without skipping empty lines
        results = []
        async for record in read_text_file(text_file, field_name='text', skip_empty=False):
            results.append(record)

        assert len(results) == 4
        assert results[2] == {'text': ''}

    @pytest.mark.asyncio
    async def test_create_file_reader_auto_detect(self, tmp_path):
        """Test create_file_reader with auto format detection."""
        # Test JSONL auto-detection
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, 'w') as f:
            f.write('{"test": "data"}\n')

        results = []
        async for record in create_file_reader(jsonl_file):
            results.append(record)
        assert len(results) == 1
        assert results[0] == {"test": "data"}

        # Test CSV auto-detection
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['col1', 'col2'])
            writer.writerow(['val1', 'val2'])

        results = []
        async for record in create_file_reader(csv_file):
            results.append(record)
        assert len(results) == 1
        assert results[0] == {'col1': 'val1', 'col2': 'val2'}

        # Test text auto-detection
        txt_file = tmp_path / "test.txt"
        with open(txt_file, 'w') as f:
            f.write('Hello world\n')

        results = []
        async for record in create_file_reader(txt_file):
            results.append(record)
        assert len(results) == 1
        assert results[0] == {'text': 'Hello world'}

    @pytest.mark.asyncio
    async def test_create_file_reader_explicit_format(self, tmp_path):
        """Test create_file_reader with explicit format."""
        # Force reading a .txt file as JSON
        json_file = tmp_path / "data.txt"
        with open(json_file, 'w') as f:
            json.dump([{"id": 1}, {"id": 2}], f)

        results = []
        async for record in create_file_reader(json_file, input_format='json'):
            results.append(record)
        assert len(results) == 2
        assert results[0] == {"id": 1}

    @pytest.mark.asyncio
    async def test_create_file_reader_invalid_format(self):
        """Test create_file_reader with invalid format."""
        with pytest.raises(ValueError, match="Unsupported input format"):
            async for _ in create_file_reader("dummy.txt", input_format="invalid"):
                pass


class TestFileWriters:
    """Test file writing functions."""

    def test_create_jsonl_writer(self, tmp_path):
        """Test JSONL writer."""
        output_file = tmp_path / "output.jsonl"

        writer = create_jsonl_writer(output_file)

        # Write first batch
        writer([{"id": 1, "name": "Alice"}])

        # Write second batch (should append)
        writer([{"id": 2, "name": "Bob"}])

        # Verify output
        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": 1, "name": "Alice"}
        assert json.loads(lines[1]) == {"id": 2, "name": "Bob"}

    def test_create_csv_writer(self, tmp_path):
        """Test CSV writer with cleanup."""
        output_file = tmp_path / "output.csv"

        writer, cleanup = create_csv_writer(output_file, delimiter=',')

        try:
            # Write first batch
            writer([{"name": "Alice", "age": 30}])

            # Write second batch
            writer([{"name": "Bob", "age": 25}])
        finally:
            # Cleanup closes the file
            cleanup()

        # Verify output
        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0] == {"name": "Alice", "age": "30"}
        assert rows[1] == {"name": "Bob", "age": "25"}

    def test_create_tsv_writer(self, tmp_path):
        """Test TSV writer."""
        output_file = tmp_path / "output.tsv"

        writer, cleanup = create_csv_writer(output_file, delimiter='\t')

        try:
            writer([{"col1": "a", "col2": "b"}])
            writer([{"col1": "c", "col2": "d"}])
        finally:
            cleanup()

        # Verify output
        with open(output_file) as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0] == {"col1": "a", "col2": "b"}

    def test_create_json_writer(self, tmp_path):
        """Test JSON writer with accumulation."""
        output_file = tmp_path / "output.json"

        writer, cleanup = create_json_writer(output_file)

        # Write multiple batches
        writer([{"id": 1}])
        writer([{"id": 2}, {"id": 3}])

        # Cleanup writes the accumulated data
        cleanup()

        # Verify output
        with open(output_file) as f:
            data = json.load(f)

        assert len(data) == 3
        assert data == [{"id": 1}, {"id": 2}, {"id": 3}]

    def test_create_file_writer_auto_detect(self, tmp_path):
        """Test file writer with auto format detection."""
        # Test JSONL detection
        jsonl_file = tmp_path / "output.jsonl"
        writer, cleanup = create_file_writer(jsonl_file)
        assert cleanup is None  # JSONL doesn't need cleanup
        writer([{"test": "data"}])

        with open(jsonl_file) as f:
            assert json.loads(f.read()) == {"test": "data"}

        # Test CSV detection
        csv_file = tmp_path / "output.csv"
        writer, cleanup = create_file_writer(csv_file)
        assert cleanup is not None  # CSV needs cleanup
        writer([{"col": "val"}])
        cleanup()

        with open(csv_file) as f:
            assert "col\n" in f.read()  # Header should be present

        # Test JSON detection
        json_file = tmp_path / "output.json"
        writer, cleanup = create_file_writer(json_file)
        assert cleanup is not None  # JSON needs cleanup
        writer([{"data": "test"}])
        cleanup()

        with open(json_file) as f:
            data = json.load(f)
            assert data == [{"data": "test"}]


class TestIntegration:
    """Integration tests for reading and writing."""

    @pytest.mark.asyncio
    async def test_roundtrip_jsonl(self, tmp_path):
        """Test reading and writing JSONL files."""
        # Original data
        original_data = [
            {"id": i, "value": f"item_{i}"}
            for i in range(5)
        ]

        # Write data
        output_file = tmp_path / "test.jsonl"
        writer = create_jsonl_writer(output_file)
        writer(original_data)

        # Read data back
        read_data = []
        async for record in read_jsonl_file(output_file):
            read_data.append(record)

        assert read_data == original_data

    @pytest.mark.asyncio
    async def test_roundtrip_csv(self, tmp_path):
        """Test reading and writing CSV files."""
        # Original data
        original_data = [
            {"name": "Alice", "score": "95"},
            {"name": "Bob", "score": "87"}
        ]

        # Write data
        output_file = tmp_path / "test.csv"
        writer, cleanup = create_csv_writer(output_file)
        writer(original_data)
        cleanup()

        # Read data back
        read_data = []
        async for record in read_csv_file(output_file):
            read_data.append(record)

        assert read_data == original_data

    @pytest.mark.asyncio
    async def test_convert_text_to_jsonl(self, tmp_path):
        """Test converting text file to JSONL."""
        # Create text file
        text_file = tmp_path / "input.txt"
        with open(text_file, 'w') as f:
            f.write("Line 1\n")
            f.write("Line 2\n")
            f.write("Line 3\n")

        # Read text and write as JSONL
        output_file = tmp_path / "output.jsonl"
        writer = create_jsonl_writer(output_file)

        data = []
        async for record in read_text_file(text_file):
            data.append(record)

        writer(data)

        # Verify JSONL output
        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) == 3
        assert json.loads(lines[0]) == {"text": "Line 1"}
        assert json.loads(lines[2]) == {"text": "Line 3"}

    @pytest.mark.asyncio
    async def test_convert_csv_to_json(self, tmp_path):
        """Test converting CSV file to JSON."""
        # Create CSV file
        csv_file = tmp_path / "input.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name'])
            writer.writerow(['1', 'Alice'])
            writer.writerow(['2', 'Bob'])

        # Read CSV and write as JSON
        output_file = tmp_path / "output.json"
        json_writer, cleanup = create_json_writer(output_file)

        data = []
        async for record in read_csv_file(csv_file):
            data.append(record)

        json_writer(data)
        cleanup()

        # Verify JSON output
        with open(output_file) as f:
            json_data = json.load(f)

        assert len(json_data) == 2
        assert json_data[0] == {"id": "1", "name": "Alice"}
        assert json_data[1] == {"id": "2", "name": "Bob"}