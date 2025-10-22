#!/usr/bin/env python3
"""Command-line interface for markdown chunking utilities.

This CLI tool provides commands for parsing markdown documents and generating
chunks suitable for RAG applications.
"""

import argparse
import json
import sys
from typing import TextIO

from dataknobs_xization.markdown.md_chunker import ChunkFormat, HeadingInclusion
from dataknobs_xization.markdown.md_parser import parse_markdown
from dataknobs_xization.markdown.md_streaming import stream_markdown_string


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Markdown chunking utilities for RAG applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse and chunk a markdown file
  %(prog)s chunk document.md

  # Chunk with custom size and overlap
  %(prog)s chunk document.md --max-size 500 --overlap 50

  # Output as JSON
  %(prog)s chunk document.md --output-format json

  # Include headings only in metadata
  %(prog)s chunk document.md --headings metadata

  # Parse and display tree structure
  %(prog)s parse document.md --show-tree
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Chunk command
    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Chunk a markdown document for RAG",
    )
    chunk_parser.add_argument(
        "input",
        type=str,
        help="Input markdown file (use '-' for stdin)",
    )
    chunk_parser.add_argument(
        "--max-size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters (default: 1000)",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Chunk overlap in characters (default: 100)",
    )
    chunk_parser.add_argument(
        "--headings",
        type=str,
        choices=["both", "text", "metadata", "none"],
        default="both",
        help="How to include headings (default: both)",
    )
    chunk_parser.add_argument(
        "--output-format",
        type=str,
        choices=["markdown", "plain", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    chunk_parser.add_argument(
        "--output",
        type=str,
        help="Output file (default: stdout)",
    )
    chunk_parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Show chunk metadata",
    )
    chunk_parser.add_argument(
        "--separator",
        type=str,
        default="\n---\n",
        help="Separator between chunks (default: \\n---\\n)",
    )

    # Parse command
    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse markdown and show tree structure",
    )
    parse_parser.add_argument(
        "input",
        type=str,
        help="Input markdown file (use '-' for stdin)",
    )
    parse_parser.add_argument(
        "--show-tree",
        action="store_true",
        help="Display tree structure",
    )
    parse_parser.add_argument(
        "--output",
        type=str,
        help="Output file (default: stdout)",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about a markdown document",
    )
    info_parser.add_argument(
        "input",
        type=str,
        help="Input markdown file (use '-' for stdin)",
    )

    return parser.parse_args()


def get_input_content(input_path: str) -> str:
    """Read input content from file or stdin.

    Args:
        input_path: Path to file or '-' for stdin

    Returns:
        Content string
    """
    if input_path == "-":
        return sys.stdin.read()
    else:
        with open(input_path, encoding="utf-8") as f:
            return f.read()


def get_output_file(output_path: str | None) -> TextIO:
    """Get output file handle.

    Args:
        output_path: Path to output file or None for stdout

    Returns:
        File handle
    """
    if output_path:
        return open(output_path, "w", encoding="utf-8")
    else:
        return sys.stdout


def heading_inclusion_from_str(value: str) -> HeadingInclusion:
    """Convert string to HeadingInclusion enum.

    Args:
        value: String value

    Returns:
        HeadingInclusion enum value
    """
    mapping = {
        "both": HeadingInclusion.BOTH,
        "text": HeadingInclusion.IN_TEXT,
        "metadata": HeadingInclusion.IN_METADATA,
        "none": HeadingInclusion.NONE,
    }
    return mapping.get(value.lower(), HeadingInclusion.BOTH)


def chunk_format_from_str(value: str) -> ChunkFormat:
    """Convert string to ChunkFormat enum.

    Args:
        value: String value

    Returns:
        ChunkFormat enum value
    """
    mapping = {
        "markdown": ChunkFormat.MARKDOWN,
        "plain": ChunkFormat.PLAIN,
        "json": ChunkFormat.DICT,
    }
    return mapping.get(value.lower(), ChunkFormat.MARKDOWN)


def cmd_chunk(args: argparse.Namespace) -> None:
    """Execute chunk command.

    Args:
        args: Parsed command-line arguments
    """
    # Get input
    content = get_input_content(args.input)

    # Configure chunking
    heading_inclusion = heading_inclusion_from_str(args.headings)
    chunk_format = chunk_format_from_str(args.output_format)

    # Generate chunks
    chunks = list(stream_markdown_string(
        content,
        max_chunk_size=args.max_size,
        chunk_overlap=args.overlap,
        heading_inclusion=heading_inclusion,
        chunk_format=chunk_format,
    ))

    # Output chunks
    with get_output_file(args.output) as out:
        if args.output_format == "json":
            # Output as JSON array
            chunk_dicts = [chunk.to_dict() for chunk in chunks]
            json.dump(chunk_dicts, out, indent=2, ensure_ascii=False)
            out.write("\n")
        else:
            # Output as text with separators
            for i, chunk in enumerate(chunks):
                if i > 0:
                    out.write(args.separator)

                if args.show_metadata:
                    out.write(f"Chunk {chunk.metadata.chunk_index}\n")
                    out.write(f"Line: {chunk.metadata.line_number}\n")
                    out.write(f"Size: {chunk.metadata.chunk_size} chars\n")
                    if chunk.metadata.headings:
                        out.write(f"Headings: {chunk.metadata.get_heading_path()}\n")
                    out.write("\n")

                out.write(chunk.text)
                out.write("\n")

    if args.output:
        print(f"Generated {len(chunks)} chunks -> {args.output}", file=sys.stderr)
    else:
        print(f"Generated {len(chunks)} chunks", file=sys.stderr)


def cmd_parse(args: argparse.Namespace) -> None:
    """Execute parse command.

    Args:
        args: Parsed command-line arguments
    """
    # Get input
    content = get_input_content(args.input)

    # Parse markdown
    tree = parse_markdown(content)

    # Output
    with get_output_file(args.output) as out:
        if args.show_tree:
            out.write("Tree structure:\n")
            out.write(str(tree))
            out.write("\n\n")

        # Show statistics
        all_nodes = tree.find_nodes(lambda _: True)
        heading_nodes = tree.find_nodes(lambda n: n.data.is_heading())
        body_nodes = tree.find_nodes(lambda n: n.data.is_body())

        out.write(f"Total nodes: {len(all_nodes)}\n")
        out.write(f"Heading nodes: {len(heading_nodes)}\n")
        out.write(f"Body nodes: {len(body_nodes)}\n")
        out.write(f"Tree depth: {tree.get_deepest_left().depth}\n")


def cmd_info(args: argparse.Namespace) -> None:
    """Execute info command.

    Args:
        args: Parsed command-line arguments
    """
    # Get input
    content = get_input_content(args.input)

    # Parse markdown
    tree = parse_markdown(content)

    # Collect statistics
    all_nodes = tree.find_nodes(lambda _: True)
    heading_nodes = tree.find_nodes(lambda n: n.data.is_heading())
    body_nodes = tree.find_nodes(lambda n: n.data.is_body())

    # Count by heading level
    level_counts = {}
    for node in heading_nodes:
        level = node.data.level
        level_counts[level] = level_counts.get(level, 0) + 1

    # Calculate text statistics
    total_chars = sum(len(n.data.text) for n in body_nodes)
    total_lines = len(body_nodes)

    # Output info
    print("Document Information")
    print("=" * 50)
    print(f"Total nodes: {len(all_nodes)}")
    print(f"Heading nodes: {len(heading_nodes)}")
    print(f"Body text nodes: {len(body_nodes)}")
    print(f"Tree depth: {tree.get_deepest_left().depth}")
    print()
    print("Heading levels:")
    for level in sorted(level_counts.keys()):
        print(f"  Level {level}: {level_counts[level]}")
    print()
    print(f"Total characters: {total_chars:,}")
    print(f"Total lines: {total_lines}")
    if total_lines > 0:
        print(f"Average line length: {total_chars / total_lines:.1f} chars")


def main() -> None:
    """Main entry point for CLI."""
    args = parse_args()

    if not args.command:
        print("Error: Please specify a command (chunk, parse, or info)", file=sys.stderr)
        print("Use --help for usage information", file=sys.stderr)
        sys.exit(1)

    try:
        if args.command == "chunk":
            cmd_chunk(args)
        elif args.command == "parse":
            cmd_parse(args)
        elif args.command == "info":
            cmd_info(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
