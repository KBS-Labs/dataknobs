#!/usr/bin/env python3
"""Unit tests for normalize_file_example.py"""

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from dataknobs_fsm.api.simple import SimpleFSM
from examples.normalize_file_example import (
    NORMALIZE_FILE_WORKFLOW_YAML,
    WORKFLOW_CONFIG,
    normalize_file_streaming,
    normalize_file_simple,
    normalize_lines,
    normalize_batch,
)


class TestWorkflowConfiguration:
    """Test the workflow configuration."""

    def test_workflow_yaml_is_valid(self):
        """Test that the YAML workflow string is valid."""
        config = yaml.safe_load(NORMALIZE_FILE_WORKFLOW_YAML)

        assert config is not None
        assert config['name'] == 'text_normalization_workflow'
        assert 'states' in config
        assert 'arcs' in config

        # Check states
        states = config['states']
        assert len(states) == 3
        state_names = [s['name'] for s in states]
        assert 'start' in state_names
        assert 'normalize' in state_names
        assert 'complete' in state_names

        # Check start and end states
        start_states = [s for s in states if s.get('is_start')]
        end_states = [s for s in states if s.get('is_end')]
        assert len(start_states) == 1
        assert len(end_states) == 1

        # Check arcs
        arcs = config['arcs']
        assert len(arcs) == 2
        assert arcs[1]['transform']['type'] == 'inline'
        assert 'lambda' in arcs[1]['transform']['code']

    def test_workflow_config_loads_correctly(self):
        """Test that WORKFLOW_CONFIG is properly loaded."""
        assert WORKFLOW_CONFIG is not None
        assert isinstance(WORKFLOW_CONFIG, dict)
        assert WORKFLOW_CONFIG['name'] == 'text_normalization_workflow'

    def test_fsm_can_be_created_from_config(self):
        """Test that SimpleFSM can be created from the config."""
        fsm = SimpleFSM(WORKFLOW_CONFIG)
        assert fsm is not None

        # Check that the FSM has the expected states
        states = fsm.get_states()
        assert 'start' in states
        assert 'normalize' in states
        assert 'complete' in states

        fsm.close()


class TestNormalizeFileStreaming:
    """Test the streaming file normalization function."""

    def test_normalize_file_streaming_basic(self, tmp_path):
        """Test basic streaming file normalization."""
        # Create input file
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.jsonl"

        test_lines = [
            "  HELLO WORLD  ",
            "this is a TEST",
            "   Mixed   CASE   text   ",
            "email@EXAMPLE.COM",
            "  whitespace    issues  "
        ]

        with open(input_file, 'w') as f:
            for line in test_lines:
                f.write(line + '\n')

        # Process the file - no mocking needed, lambda function handles it
        normalize_file_streaming(str(input_file), str(output_file))

        # Check output
        assert output_file.exists()

        with open(output_file) as f:
            output_lines = [line.strip() for line in f.readlines()]

        # Each line should be JSON with normalized text
        assert len(output_lines) == len(test_lines)

        expected = [
            "hello world",
            "this is a test",
            "mixed   case   text",
            "email@example.com",
            "whitespace    issues"
        ]

        for i, line in enumerate(output_lines):
            result = json.loads(line)
            assert 'text' in result
            # Check that text was normalized (lowercase and stripped)
            assert result['text'] == expected[i]

    def test_normalize_file_streaming_empty_file(self, tmp_path):
        """Test streaming normalization with empty file."""
        input_file = tmp_path / "empty.txt"
        output_file = tmp_path / "output.jsonl"

        # Create empty file
        input_file.touch()

        normalize_file_streaming(str(input_file), str(output_file))

        # Output should exist but be empty or have no content
        assert output_file.exists()
        content = output_file.read_text()
        assert content == "" or content == "[]"


class TestNormalizeFileSimple:
    """Test the simple file normalization function."""

    def test_normalize_file_simple_basic(self, tmp_path):
        """Test simple file normalization."""
        # Create input file
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.jsonl"

        test_lines = [
            "  UPPER CASE  ",
            "lower case",
            "   MiXeD CaSe   "
        ]

        with open(input_file, 'w') as f:
            for line in test_lines:
                f.write(line + '\n')

        # Process the file - no mocking needed
        results = normalize_file_simple(str(input_file), str(output_file))

        assert results is not None
        assert 'total_processed' in results
        assert 'successful' in results

        # Check output file
        assert output_file.exists()

        with open(output_file) as f:
            output_lines = f.readlines()

        assert len(output_lines) == len(test_lines)

        expected = [
            "upper case",
            "lower case",
            "mixed case"
        ]

        for i, line in enumerate(output_lines):
            result = json.loads(line)
            assert result['text'] == expected[i]


class TestNormalizeLines:
    """Test the individual line normalization function."""

    def test_normalize_lines_basic(self):
        """Test normalizing individual lines."""
        test_lines = [
            "  UPPERCASE TEXT  ",
            "MiXeD cAsE",
            "   extra   spaces   ",
            "normal text"
        ]

        # Use the function directly - it now uses the lambda
        normalized = normalize_lines(test_lines)

        assert len(normalized) == len(test_lines)

        expected = [
            "uppercase text",
            "mixed case",
            "extra   spaces",  # Note: lambda only strips, doesn't compress spaces
            "normal text"
        ]

        for result, expected_text in zip(normalized, expected):
            assert result == expected_text

    def test_normalize_lines_empty_list(self):
        """Test normalizing an empty list."""
        normalized = normalize_lines([])
        assert normalized == []

    def test_normalize_lines_with_empty_strings(self):
        """Test normalizing lines including empty strings."""
        test_lines = ["text", "", "  ", "more text"]

        normalized = normalize_lines(test_lines)

        assert len(normalized) == len(test_lines)
        assert normalized[0] == "text"
        assert normalized[1] == ""
        assert normalized[2] == ""
        assert normalized[3] == "more text"


class TestNormalizeBatch:
    """Test the batch processing normalization function."""

    def test_normalize_batch_basic(self):
        """Test batch normalization."""
        test_lines = [
            "  BATCH LINE 1  ",
            "batch line 2",
            "BATCH LINE 3",
            "  Batch Line 4  "
        ]

        normalized = normalize_batch(test_lines)

        assert len(normalized) == len(test_lines)

        expected = [
            "batch line 1",
            "batch line 2",
            "batch line 3",
            "batch line 4"
        ]

        for result, expected_text in zip(normalized, expected):
            assert result == expected_text

    def test_normalize_batch_large_dataset(self):
        """Test batch processing with a larger dataset."""
        # Create 500 test lines
        test_lines = [f"  Line Number {i}  " for i in range(500)]

        normalized = normalize_batch(test_lines)

        assert len(normalized) == 500

        # Check a few samples
        assert normalized[0] == "line number 0"
        assert normalized[100] == "line number 100"
        assert normalized[499] == "line number 499"

    def test_normalize_batch_empty(self):
        """Test batch processing with empty input."""
        normalized = normalize_batch([])
        assert normalized == []


class TestIntegration:
    """Integration tests using multiple methods."""

    def test_all_methods_produce_consistent_results(self, tmp_path):
        """Test that all methods produce consistent results."""
        test_lines = [
            "  CONSISTENT TEST  ",
            "Another Line",
            "   FINAL LINE   "
        ]

        # Method 1: Streaming
        input_file = tmp_path / "input.txt"
        output_stream = tmp_path / "output_stream.jsonl"

        with open(input_file, 'w') as f:
            for line in test_lines:
                f.write(line + '\n')

        normalize_file_streaming(str(input_file), str(output_stream))

        # Method 2: Simple
        output_simple = tmp_path / "output_simple.jsonl"
        normalize_file_simple(str(input_file), str(output_simple))

        # Method 3: Individual lines
        normalized_lines = normalize_lines(test_lines)

        # Method 4: Batch
        normalized_batch = normalize_batch(test_lines)

        # All methods should produce the same normalized text
        expected = [
            "consistent test",
            "another line",
            "final line"
        ]

        # Check streaming output
        with open(output_stream) as f:
            for i, line in enumerate(f):
                result = json.loads(line)
                assert result['text'] == expected[i]

        # Check simple output
        with open(output_simple) as f:
            for i, line in enumerate(f):
                result = json.loads(line)
                assert result['text'] == expected[i]

        # Check individual lines
        assert normalized_lines == expected

        # Check batch
        assert normalized_batch == expected


class TestErrorHandling:
    """Test error handling in normalization functions."""

    def test_normalize_lines_with_error(self, capsys):
        """Test that errors in normalization are handled gracefully."""
        test_lines = ["good line", "problematic line"]

        # The lambda function will handle all inputs without errors
        normalized = normalize_lines(test_lines)

        # The function should handle all lines
        assert len(normalized) == 2
        assert normalized[0] == "good line"
        assert normalized[1] == "problematic line"

    def test_streaming_with_malformed_input(self, tmp_path):
        """Test streaming with malformed input."""
        input_file = tmp_path / "malformed.txt"
        output_file = tmp_path / "output.jsonl"

        # Create file with various edge cases
        with open(input_file, 'w') as f:
            f.write("\n")  # Empty line
            f.write("   \n")  # Whitespace only
            f.write("normal line\n")
            f.write("\t\t\n")  # Tabs only

        normalize_file_streaming(str(input_file), str(output_file))

        assert output_file.exists()

        with open(output_file) as f:
            lines = f.readlines()

        # Should process all non-empty lines (empty lines are skipped by default)
        assert len(lines) == 3

        # Check the normalized output
        results = [json.loads(line) for line in lines]
        assert results[0]['text'] == ""  # Whitespace only becomes empty string
        assert results[1]['text'] == "normal line"
        assert results[2]['text'] == ""  # Tabs only becomes empty string

    def test_streaming_with_empty_lines_included(self, tmp_path):
        """Test streaming with skip_empty_lines=False."""
        input_file = tmp_path / "with_empty.txt"
        output_file = tmp_path / "output.jsonl"

        # Create file with empty lines
        with open(input_file, 'w') as f:
            f.write("\n")  # Empty line
            f.write("   \n")  # Whitespace only
            f.write("normal line\n")
            f.write("\t\t\n")  # Tabs only

        # Process with skip_empty_lines=False
        from dataknobs_fsm.api.simple import SimpleFSM
        fsm = SimpleFSM(WORKFLOW_CONFIG)

        try:
            results = fsm.process_stream(
                source=str(input_file),
                sink=str(output_file),
                input_format='text',
                text_field_name='text',
                chunk_size=1000,
                use_streaming=True,
                skip_empty_lines=False  # Include empty lines
            )

            assert results['total_processed'] == 4  # All 4 lines should be processed

        finally:
            fsm.close()

        assert output_file.exists()

        with open(output_file) as f:
            lines = f.readlines()

        # Should process all lines including empty ones
        assert len(lines) == 4

        # Check the normalized output
        results = [json.loads(line) for line in lines]
        assert results[0]['text'] == ""  # Empty line becomes empty string
        assert results[1]['text'] == ""  # Whitespace only becomes empty string
        assert results[2]['text'] == "normal line"
        assert results[3]['text'] == ""  # Tabs only becomes empty string
