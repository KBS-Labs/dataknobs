"""Tests for streaming file utilities."""

import asyncio
import csv
import json
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

from dataknobs_fsm.streaming.core import StreamChunk, StreamConfig
from dataknobs_fsm.utils.streaming_file_utils import (
    StreamingFileReader,
    StreamingFileWriter,
    StreamingFileProcessor,
    create_streaming_file_reader,
    create_streaming_file_writer,
)


class TestStreamingFileReader:
    """Test streaming file reader."""

    @pytest.mark.asyncio
    async def test_read_jsonl_chunks(self, tmp_path):
        """Test reading JSONL file in chunks."""
        # Create test JSONL file
        jsonl_file = tmp_path / "test.jsonl"
        test_data = [{"id": i, "value": f"item_{i}"} for i in range(25)]

        with open(jsonl_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        # Read with chunk size of 10
        reader = StreamingFileReader(jsonl_file, chunk_size=10)
        chunks = []

        async for chunk in reader.read_chunks():
            chunks.append(chunk)

        # Should have 3 chunks: 10, 10, 5
        assert len(chunks) == 3
        assert len(chunks[0].data) == 10
        assert len(chunks[1].data) == 10
        assert len(chunks[2].data) == 5
        assert chunks[2].is_last is True

        # Verify metrics
        assert reader.metrics.chunks_processed == 3
        assert reader.metrics.items_processed == 25

    @pytest.mark.asyncio
    async def test_read_csv_chunks(self, tmp_path):
        """Test reading CSV file in chunks."""
        csv_file = tmp_path / "test.csv"

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'value'])
            for i in range(15):
                writer.writerow([i, f'name_{i}', i * 10])

        reader = StreamingFileReader(csv_file, chunk_size=5, input_format='csv')
        chunks = []

        async for chunk in reader.read_chunks():
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].sequence_number == 0
        assert chunks[1].sequence_number == 1
        assert chunks[2].sequence_number == 2
        assert chunks[2].is_last is True

        # Check data
        first_record = chunks[0].data[0]
        assert 'id' in first_record
        assert 'name' in first_record
        assert 'value' in first_record

    @pytest.mark.asyncio
    async def test_read_text_chunks(self, tmp_path):
        """Test reading text file in chunks."""
        text_file = tmp_path / "test.txt"

        with open(text_file, 'w') as f:
            for i in range(20):
                f.write(f"Line {i}\n")
            f.write("\n")  # Empty line
            f.write("Last line\n")

        reader = StreamingFileReader(
            text_file,
            chunk_size=8,
            input_format='text',
            text_field_name='content',
            skip_empty_lines=True
        )

        chunks = []
        async for chunk in reader.read_chunks():
            chunks.append(chunk)

        # 21 non-empty lines / 8 per chunk = 3 chunks
        assert len(chunks) == 3
        assert reader.metrics.items_processed == 21  # Skipped empty line

        # Check content
        assert chunks[0].data[0] == {'content': 'Line 0'}
        assert chunks[-1].data[-1] == {'content': 'Last line'}

    @pytest.mark.asyncio
    async def test_read_json_array_chunks(self, tmp_path):
        """Test reading JSON array file in chunks."""
        json_file = tmp_path / "test.json"
        test_data = [{"id": i} for i in range(12)]

        with open(json_file, 'w') as f:
            json.dump(test_data, f)

        reader = StreamingFileReader(json_file, chunk_size=5, input_format='json')
        chunks = []

        async for chunk in reader.read_chunks():
            chunks.append(chunk)

        # 12 items / 5 per chunk = 3 chunks (5, 5, 2)
        assert len(chunks) == 3
        assert len(chunks[0].data) == 5
        assert len(chunks[2].data) == 2
        assert chunks[2].is_last is True

    @pytest.mark.asyncio
    async def test_error_handling(self, tmp_path):
        """Test error handling for malformed data."""
        jsonl_file = tmp_path / "bad.jsonl"

        with open(jsonl_file, 'w') as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json\n')
            f.write('{"another": "valid"}\n')

        reader = StreamingFileReader(jsonl_file, chunk_size=10)
        chunks = []

        async for chunk in reader.read_chunks():
            chunks.append(chunk)

        assert len(chunks) == 1
        assert len(chunks[0].data) == 2  # Only valid JSON lines
        assert reader.metrics.errors_count == 1  # One error for invalid JSON


class TestStreamingFileWriter:
    """Test streaming file writer."""

    @pytest.mark.asyncio
    async def test_write_jsonl_chunks(self, tmp_path):
        """Test writing JSONL file with chunks."""
        output_file = tmp_path / "output.jsonl"

        async with StreamingFileWriter(output_file, buffer_size=5) as writer:
            # Write first chunk
            chunk1 = StreamChunk(data=[{"id": i} for i in range(3)])
            await writer.write_chunk(chunk1)

            # Write second chunk (should trigger flush due to buffer size)
            chunk2 = StreamChunk(data=[{"id": i} for i in range(3, 8)])
            await writer.write_chunk(chunk2)

            # Write last chunk
            chunk3 = StreamChunk(data=[{"id": 8}], is_last=True)
            await writer.write_chunk(chunk3)

        # Verify output
        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) == 9
        for i, line in enumerate(lines):
            data = json.loads(line)
            assert data == {"id": i}

    @pytest.mark.asyncio
    async def test_write_csv_chunks(self, tmp_path):
        """Test writing CSV file with chunks."""
        output_file = tmp_path / "output.csv"

        writer = StreamingFileWriter(output_file, output_format='csv', buffer_size=3)
        writer.open()

        try:
            # Write chunks
            chunk1 = StreamChunk(data=[
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ])
            await writer.write_chunk(chunk1)

            chunk2 = StreamChunk(data=[
                {"name": "Charlie", "age": 35}
            ], is_last=True)
            await writer.write_chunk(chunk2)

        finally:
            await writer.close()

        # Verify CSV output
        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0] == {"name": "Alice", "age": "30"}
        assert rows[2] == {"name": "Charlie", "age": "35"}

    @pytest.mark.asyncio
    async def test_write_json_chunks(self, tmp_path):
        """Test writing JSON file with accumulation."""
        output_file = tmp_path / "output.json"

        writer = StreamingFileWriter(output_file, output_format='json')
        writer.open()

        try:
            for i in range(3):
                chunk = StreamChunk(
                    data=[{"batch": i, "item": j} for j in range(2)],
                    is_last=(i == 2)
                )
                await writer.write_chunk(chunk)
        finally:
            await writer.close()

        # Verify JSON output
        with open(output_file) as f:
            data = json.load(f)

        assert len(data) == 6
        assert data[0] == {"batch": 0, "item": 0}
        assert data[-1] == {"batch": 2, "item": 1}

    @pytest.mark.asyncio
    async def test_write_text_chunks(self, tmp_path):
        """Test writing text file with chunks."""
        output_file = tmp_path / "output.txt"

        writer = StreamingFileWriter(output_file, output_format='text')
        writer.open()

        try:
            chunk1 = StreamChunk(data=[{"text": f"Line {i}"} for i in range(5)])
            await writer.write_chunk(chunk1)

            chunk2 = StreamChunk(data=[{"text": "Last line"}], is_last=True)
            await writer.write_chunk(chunk2)
        finally:
            await writer.close()

        # Verify text output
        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) == 6
        assert lines[0].strip() == "Line 0"
        assert lines[-1].strip() == "Last line"

    @pytest.mark.asyncio
    async def test_buffering_and_flush(self, tmp_path):
        """Test that buffering and flushing work correctly."""
        output_file = tmp_path / "output.jsonl"

        writer = StreamingFileWriter(
            output_file,
            buffer_size=3,
            flush_interval=0.1  # Short interval for testing
        )
        writer.open()

        try:
            # Write less than buffer size
            chunk1 = StreamChunk(data=[{"id": 0}, {"id": 1}])
            await writer.write_chunk(chunk1)

            # File should be empty (buffered)
            with open(output_file) as f:
                content = f.read()
            assert content == ""

            # Write to exceed buffer size
            chunk2 = StreamChunk(data=[{"id": 2}, {"id": 3}])
            await writer.write_chunk(chunk2)

            # Now file should have content
            with open(output_file) as f:
                lines = f.readlines()
            assert len(lines) >= 3

        finally:
            await writer.close()

        # All data should be written after close
        with open(output_file) as f:
            lines = f.readlines()
        assert len(lines) == 4


class TestStreamingFileProcessor:
    """Test the high-level streaming file processor."""

    @pytest.mark.asyncio
    async def test_process_with_transform(self, tmp_path):
        """Test processing file with transformation."""
        # Create input file
        input_file = tmp_path / "input.jsonl"
        with open(input_file, 'w') as f:
            for i in range(10):
                f.write(json.dumps({"value": i}) + '\n')

        output_file = tmp_path / "output.jsonl"

        # Create processor with transformation
        def transform(record: Dict) -> Dict:
            return {
                "original": record["value"],
                "doubled": record["value"] * 2
            }

        processor = StreamingFileProcessor(
            input_file,
            output_file,
            transform_fn=transform,
            chunk_size=3
        )

        metrics = await processor.process()

        # Check metrics
        assert metrics.items_processed == 10
        assert metrics.chunks_processed > 0

        # Verify output
        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) == 10
        first = json.loads(lines[0])
        assert first == {"original": 0, "doubled": 0}
        last = json.loads(lines[-1])
        assert last == {"original": 9, "doubled": 18}

    @pytest.mark.asyncio
    async def test_process_with_progress_callback(self, tmp_path):
        """Test processing with progress callback."""
        # Create CSV input
        input_file = tmp_path / "input.csv"
        with open(input_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'value'])
            for i in range(15):
                writer.writerow([i, i * 10])

        output_file = tmp_path / "output.jsonl"

        # Track progress
        progress_updates = []

        def progress_callback(items, chunks):
            progress_updates.append((items, chunks))

        processor = StreamingFileProcessor(
            input_file,
            output_file,
            chunk_size=5,
            input_format='csv'
        )

        await processor.process(progress_callback=progress_callback)

        # Should have received progress updates
        assert len(progress_updates) > 0
        # Final update should show all items
        assert progress_updates[-1][0] == 15

    @pytest.mark.asyncio
    async def test_format_conversion(self, tmp_path):
        """Test converting between file formats."""
        # Create CSV input
        csv_file = tmp_path / "input.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'score'])
            writer.writerow(['Alice', '95'])
            writer.writerow(['Bob', '87'])

        # Convert to JSON
        json_file = tmp_path / "output.json"

        processor = StreamingFileProcessor(
            csv_file,
            json_file,
            chunk_size=10,
            input_format='csv',
            output_format='json'
        )

        await processor.process()

        # Verify JSON output
        with open(json_file) as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0] == {"name": "Alice", "score": "95"}
        assert data[1] == {"name": "Bob", "score": "87"}


class TestIntegrationWithSimpleFSM:
    """Test integration with SimpleFSM's streaming mode."""

    @pytest.mark.asyncio
    async def test_create_streaming_reader(self, tmp_path):
        """Test creating a streaming reader for SimpleFSM."""
        # Create test file
        input_file = tmp_path / "input.txt"
        with open(input_file, 'w') as f:
            for i in range(25):
                f.write(f"Line {i}\n")

        config = StreamConfig(chunk_size=10)

        # Create streaming reader
        reader = create_streaming_file_reader(
            input_file,
            config,
            input_format='text'
        )

        # Read chunks
        chunks_received = []
        async for chunk in reader:
            chunks_received.append(chunk)

        # Should receive chunks of data
        assert len(chunks_received) == 3  # 25 lines / 10 per chunk
        assert len(chunks_received[0]) == 10
        assert len(chunks_received[-1]) == 5

    @pytest.mark.asyncio
    async def test_create_streaming_writer(self, tmp_path):
        """Test creating a streaming writer for SimpleFSM."""
        output_file = tmp_path / "output.jsonl"
        config = StreamConfig(buffer_size=5)

        write_fn, cleanup_fn = await create_streaming_file_writer(
            output_file,
            config
        )

        try:
            # Write data
            await write_fn([{"id": i} for i in range(3)])
            await write_fn([{"id": i} for i in range(3, 6)])
        finally:
            await cleanup_fn()

        # Verify output
        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) == 6
        for i, line in enumerate(lines):
            assert json.loads(line) == {"id": i}