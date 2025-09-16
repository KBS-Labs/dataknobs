#!/usr/bin/env python3
"""Example of using SimpleFSM to normalize text files."""

import yaml
from pathlib import Path
from dataknobs_fsm.api.simple import SimpleFSM, process_file


# FSM workflow for normalizing text lines from a file
NORMALIZE_FILE_WORKFLOW_YAML = '''
name: text_normalization_workflow
description: Process text lines through normalization pipeline

# Define states for the processing pipeline
states:
  - name: start
    is_start: true
    metadata:
      description: Initial state to receive text line

  - name: normalize
    metadata:
      description: Apply text normalization

  - name: complete
    is_end: true
    metadata:
      description: Final state with normalized text

# Define transitions between states
arcs:
  - from: start
    to: normalize
    metadata:
      description: Move text to normalization stage

  - from: normalize
    to: complete
    # Use a lambda function to normalize text
    transform:
      type: inline
      code: "lambda data, ctx: {**data, 'text': data.get('text', '').lower().strip()}"
    metadata:
      description: Apply normalization and complete processing
'''


# Parse the YAML workflow once
WORKFLOW_CONFIG = yaml.safe_load(NORMALIZE_FILE_WORKFLOW_YAML)


# Method 1: Using SimpleFSM directly with process_stream (synchronous)
def normalize_file_streaming(input_file: str, output_file: str):
    """Normalize a text file using SimpleFSM with streaming."""

    # Initialize FSM with the workflow config dictionary
    fsm = SimpleFSM(WORKFLOW_CONFIG)

    try:
        # Process the file with streaming for large files
        results = fsm.process_stream(
            source=input_file,
            sink=output_file,
            input_format='text',      # Read as text lines
            text_field_name='text',   # Each line becomes {'text': 'line content'}
            chunk_size=1000,          # Process 1000 lines at a time
            use_streaming=True        # Enable memory-efficient streaming
        )

        print(f"Processing completed:")
        print(f"  Total processed: {results['total_processed']}")
        print(f"  Successful: {results['successful']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Duration: {results['duration']:.2f} seconds")
        print(f"  Throughput: {results['throughput']:.2f} items/second")

    finally:
        fsm.close()


# Method 2: Using the convenience function
def normalize_file_simple(input_file: str, output_file: str):
    """Normalize a text file using the convenience function."""

    # Use the convenience function that handles FSM lifecycle
    results = process_file(
        fsm_config=WORKFLOW_CONFIG,
        input_file=input_file,
        output_file=output_file,
        input_format='text',       # Auto-detect or specify format
        text_field_name='text',    # Field name for text lines
        chunk_size=5000,           # Larger chunks for better performance
        use_streaming=True         # Enable streaming for large files
    )

    print(f"File processed: {results}")
    return results


# Method 3: Process individual lines (for small files or testing)
def normalize_lines(lines: list[str]) -> list[str]:
    """Normalize a list of text lines."""

    fsm = SimpleFSM(WORKFLOW_CONFIG)
    normalized = []

    try:
        for line in lines:
            # Process each line through the FSM
            result = fsm.process({'text': line})

            if result['success']:
                # Extract the normalized text from the result
                normalized_text = result['data'].get('text', line)
                normalized.append(normalized_text)
            else:
                print(f"Failed to normalize: {line}")
                print(f"Error: {result.get('error')}")
                normalized.append(line)  # Keep original if failed

    finally:
        fsm.close()

    return normalized


# Method 4: Batch processing for medium-sized datasets
def normalize_batch(input_lines: list[str]) -> list[str]:
    """Process multiple lines in batches."""

    fsm = SimpleFSM(WORKFLOW_CONFIG)

    try:
        # Convert lines to the expected format
        data = [{'text': line} for line in input_lines]

        # Process in batches - works in both sync and async contexts now
        results = fsm.process_batch(
            data=data,
            batch_size=100,
            max_workers=4
        )

        # Extract normalized text from results
        normalized = []
        for i, result in enumerate(results):
            if result['success']:
                normalized.append(result['data'].get('text', input_lines[i]))
            else:
                print(f"Failed item {i}: {result.get('error')}")
                normalized.append(input_lines[i])

        return normalized

    finally:
        fsm.close()


# Example usage
if __name__ == "__main__":
    import os

    # Create tmp directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)

    # Create a sample input file
    input_path = "tmp/sample_input.txt"
    output_path = "tmp/normalized_output.txt"

    # Create sample data
    with open(input_path, 'w') as f:
        f.write("  HELLO WORLD  \n")
        f.write("this is a TEST\n")
        f.write("   Mixed   CASE   text   \n")
        f.write("email@EXAMPLE.COM\n")
        f.write("  whitespace    issues  \n")

    print("=" * 60)
    print("Example 1: Streaming file processing")
    print("=" * 60)
    # Run streaming example
    normalize_file_streaming(input_path, output_path)

    # Show results
    print("\nNormalized output:")
    with open(output_path) as f:
        for line in f:
            print(f"  '{line.rstrip()}'")

    print("\n" + "=" * 60)
    print("Example 2: Simple file processing")
    print("=" * 60)
    output_path2 = "tmp/normalized_output2.jsonl"
    normalize_file_simple(input_path, output_path2)

    print("\n" + "=" * 60)
    print("Example 3: Process individual lines")
    print("=" * 60)
    test_lines = [
        "  UPPERCASE TEXT  ",
        "MiXeD cAsE",
        "   extra   spaces   "
    ]
    normalized = normalize_lines(test_lines)
    for original, norm in zip(test_lines, normalized):
        print(f"'{original}' -> '{norm}'")

    print("\n" + "=" * 60)
    print("Example 4: Batch processing")
    print("=" * 60)
    batch_normalized = normalize_batch(test_lines)
    for original, norm in zip(test_lines, batch_normalized):
        print(f"'{original}' -> '{norm}'")