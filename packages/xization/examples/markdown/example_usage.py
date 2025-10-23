#!/usr/bin/env python3
"""Example script demonstrating markdown chunking functionality."""

from dataknobs_xization import (
    ChunkFormat,
    HeadingInclusion,
    chunk_markdown_tree,
    parse_markdown,
    stream_markdown_file,
)


def example_basic_chunking():
    """Demonstrate basic markdown parsing and chunking."""
    print("=" * 70)
    print("Example 1: Basic Chunking")
    print("=" * 70)

    markdown = """# Introduction
This is a sample document about RAG systems.

## What is RAG?
RAG stands for Retrieval-Augmented Generation. It combines retrieval with generation.

### Key Benefits
RAG systems provide more accurate and up-to-date responses.

## Implementation
Here's how to implement a RAG system.
"""

    # Parse markdown into tree
    tree = parse_markdown(markdown)

    # Generate chunks with default settings
    chunks = chunk_markdown_tree(tree, max_chunk_size=500)

    print(f"\nGenerated {len(chunks)} chunks:\n")

    for chunk in chunks:
        print(f"Chunk {chunk.metadata.chunk_index}:")
        print(f"  Headings: {chunk.metadata.get_heading_path()}")
        print(f"  Size: {chunk.metadata.chunk_size} chars")
        print(f"  Text preview: {chunk.text[:100]}...")
        print()


def example_heading_inclusion_options():
    """Demonstrate different heading inclusion options."""
    print("=" * 70)
    print("Example 2: Heading Inclusion Options")
    print("=" * 70)

    markdown = """# Chapter 1
This is the first chapter.

## Section 1.1
Content for section 1.1.
"""

    tree = parse_markdown(markdown)

    # Option 1: Headings in both text and metadata
    print("\n1. HeadingInclusion.BOTH:")
    chunks = chunk_markdown_tree(
        tree,
        heading_inclusion=HeadingInclusion.BOTH,
    )
    for chunk in chunks[:1]:  # Show first chunk
        print(f"   Text: {chunk.text[:80]}...")
        print(f"   Metadata headings: {chunk.metadata.headings}")

    # Option 2: Headings only in metadata
    print("\n2. HeadingInclusion.IN_METADATA:")
    chunks = chunk_markdown_tree(
        tree,
        heading_inclusion=HeadingInclusion.IN_METADATA,
    )
    for chunk in chunks[:1]:
        print(f"   Text: {chunk.text[:80]}...")
        print(f"   Metadata headings: {chunk.metadata.headings}")

    # Option 3: No headings
    print("\n3. HeadingInclusion.NONE:")
    chunks = chunk_markdown_tree(
        tree,
        heading_inclusion=HeadingInclusion.NONE,
    )
    for chunk in chunks[:1]:
        print(f"   Text: {chunk.text[:80]}...")
        print(f"   Metadata headings: {chunk.metadata.headings}")


def example_chunk_formats():
    """Demonstrate different output formats."""
    print("\n" + "=" * 70)
    print("Example 3: Chunk Formats")
    print("=" * 70)

    markdown = """# Title
Body text here.
"""

    tree = parse_markdown(markdown)

    # Markdown format
    print("\n1. ChunkFormat.MARKDOWN:")
    chunks = chunk_markdown_tree(
        tree,
        chunk_format=ChunkFormat.MARKDOWN,
        heading_inclusion=HeadingInclusion.IN_TEXT,
    )
    for chunk in chunks:
        print(f"   {chunk.text}")

    # Plain format
    print("\n2. ChunkFormat.PLAIN:")
    chunks = chunk_markdown_tree(
        tree,
        chunk_format=ChunkFormat.PLAIN,
        heading_inclusion=HeadingInclusion.IN_TEXT,
    )
    for chunk in chunks:
        print(f"   {chunk.text}")


def example_chunk_sizing():
    """Demonstrate chunk size control."""
    print("\n" + "=" * 70)
    print("Example 4: Chunk Size Control")
    print("=" * 70)

    # Create long text
    long_text = " ".join([f"Sentence {i}." for i in range(50)])
    markdown = f"# Title\n{long_text}"

    tree = parse_markdown(markdown)

    # Small chunks
    print("\nSmall chunks (max_size=100, overlap=20):")
    chunks = chunk_markdown_tree(
        tree,
        max_chunk_size=100,
        chunk_overlap=20,
        heading_inclusion=HeadingInclusion.NONE,
    )
    print(f"   Generated {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"   Chunk {i}: {len(chunk.text)} chars - {chunk.text[:50]}...")

    # Large chunks
    print("\nLarge chunks (max_size=500, overlap=50):")
    chunks = chunk_markdown_tree(
        tree,
        max_chunk_size=500,
        chunk_overlap=50,
        heading_inclusion=HeadingInclusion.NONE,
    )
    print(f"   Generated {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i}: {len(chunk.text)} chars")


def example_streaming():
    """Demonstrate streaming from file."""
    print("\n" + "=" * 70)
    print("Example 5: Streaming from File")
    print("=" * 70)

    # Use the example document
    from pathlib import Path
    example_file = Path(__file__).parent / "example_document.md"
    print(f"\nStreaming {example_file.name}:")

    chunk_count = 0
    total_chars = 0

    for chunk in stream_markdown_file(
        str(example_file),
        max_chunk_size=500,
        heading_inclusion=HeadingInclusion.IN_METADATA,
    ):
        chunk_count += 1
        total_chars += chunk.metadata.chunk_size

        if chunk_count <= 3:  # Show first 3 chunks
            print(f"\nChunk {chunk.metadata.chunk_index}:")
            print(f"  Headings: {chunk.metadata.get_heading_path()}")
            print(f"  Size: {chunk.metadata.chunk_size} chars")
            print(f"  Preview: {chunk.text[:80]}...")

    print(f"\nTotal: {chunk_count} chunks, {total_chars} total characters")


def example_json_export():
    """Demonstrate exporting chunks as JSON."""
    print("\n" + "=" * 70)
    print("Example 6: JSON Export")
    print("=" * 70)

    markdown = """# Section 1
Content for section 1.

## Subsection 1.1
More content here.
"""

    tree = parse_markdown(markdown)
    chunks = chunk_markdown_tree(
        tree,
        heading_inclusion=HeadingInclusion.IN_METADATA,
    )

    print("\nChunks as JSON dictionaries:")
    import json

    for chunk in chunks[:2]:  # Show first 2
        print(json.dumps(chunk.to_dict(), indent=2))
        print()


def example_rag_use_case():
    """Demonstrate typical RAG use case."""
    print("\n" + "=" * 70)
    print("Example 7: RAG Use Case - Vector Store Loading")
    print("=" * 70)

    markdown = """# User Guide

## Installation
Install the package using pip.

## Configuration
Configure your settings in config.yaml.

## Usage
Use the CLI or Python API.
"""

    tree = parse_markdown(markdown)

    # Simulate loading into a vector store
    print("\nPreparing chunks for vector store:")

    chunks = chunk_markdown_tree(
        tree,
        max_chunk_size=200,
        heading_inclusion=HeadingInclusion.IN_METADATA,
    )

    for chunk in chunks:
        # In a real scenario, you would:
        # 1. Generate embeddings for chunk.text
        # 2. Store in vector database with chunk.metadata
        print(f"\nChunk {chunk.metadata.chunk_index}:")
        print(f"  Context: {chunk.metadata.get_heading_path(' → ')}")
        print(f"  Text to embed: {chunk.text}")
        print(f"  Metadata to store: {chunk.metadata.to_dict()}")


def main():
    """Run all examples."""
    examples = [
        example_basic_chunking,
        example_heading_inclusion_options,
        example_chunk_formats,
        example_chunk_sizing,
        example_streaming,
        example_json_export,
        example_rag_use_case,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {example_func.__name__}: {e}")

        print("\n")


if __name__ == "__main__":
    main()
