#!/usr/bin/env python3
"""
Find print statements in Python code using AST parsing.

This tool parses Python source files and identifies print() function calls
in executable code, ignoring comments, docstrings, and string literals.

Print statements are ignored in the following cases (considered proper usage):
- Print statements that explicitly specify a 'file' argument
  (e.g., print(..., file=sys.stderr))
- Print statements inside 'if __name__ == "__main__":' blocks
  (CLI/script usage where stdout printing is expected)
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple


class PrintStatementFinder(ast.NodeVisitor):
    """AST visitor that finds print() function calls."""

    def __init__(self, filename: str):
        self.filename = filename
        self.print_statements: List[Tuple[int, int, str]] = []
        self._in_docstring = False
        self._in_main_block = False

    def visit_Expr(self, node: ast.Expr) -> None:
        """Visit expression nodes to detect docstrings."""
        # Check if this is a docstring (string literal as statement)
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            # This is a docstring, skip it
            return
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        """Visit If nodes to detect if __name__ == "__main__" blocks."""
        is_main_block = self._is_main_guard(node.test)

        if is_main_block:
            # Set flag before visiting the block
            old_in_main = self._in_main_block
            self._in_main_block = True

            # Visit the body of the if block
            for child in node.body:
                self.visit(child)

            # Visit the else block (orelse) if it exists, without the main flag
            self._in_main_block = old_in_main
            for child in node.orelse:
                self.visit(child)
        else:
            # Regular if statement, visit normally
            self.generic_visit(node)

    def _is_main_guard(self, test_node: ast.expr) -> bool:
        """Check if a test expression is 'if __name__ == "__main__"'."""
        if isinstance(test_node, ast.Compare):
            # Check for __name__ == "__main__" or "__main__" == __name__
            if len(test_node.ops) == 1 and isinstance(test_node.ops[0], ast.Eq):
                left = test_node.left
                comparators = test_node.comparators

                if len(comparators) == 1:
                    right = comparators[0]

                    # Check both orders: __name__ == "__main__" and "__main__" == __name__
                    name_is_left = isinstance(left, ast.Name) and left.id == "__name__"
                    main_is_right = isinstance(right, ast.Constant) and right.value == "__main__"

                    name_is_right = isinstance(right, ast.Name) and right.id == "__name__"
                    main_is_left = isinstance(left, ast.Constant) and left.value == "__main__"

                    return (name_is_left and main_is_right) or (name_is_right and main_is_left)

        return False

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call nodes to find print() calls."""
        # Check if this is a print() call
        is_print = False

        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            is_print = True
        elif isinstance(node.func, ast.Attribute) and node.func.attr == 'print':
            # Could be something like console.print, but we'll report it
            is_print = True

        if is_print:
            # Ignore prints in if __name__ == "__main__" blocks
            if self._in_main_block:
                self.generic_visit(node)
                return

            # Check if the print call uses the 'file' argument
            # If it does, it's considered proper usage and should be ignored
            has_file_arg = any(
                keyword.arg == 'file'
                for keyword in node.keywords
            )

            if not has_file_arg:
                # Only flag prints that don't specify a file argument
                line_num = node.lineno
                col_offset = node.col_offset
                self.print_statements.append((line_num, col_offset, self.filename))

        # Continue visiting child nodes
        self.generic_visit(node)


def find_prints_in_file(filepath: Path) -> List[Tuple[int, int, str]]:
    """
    Find all print statements in a Python file.

    Args:
        filepath: Path to the Python file

    Returns:
        List of tuples containing (line_number, column_offset, filepath)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        # Parse the source code into an AST
        tree = ast.parse(source, filename=str(filepath))

        # Find print statements
        finder = PrintStatementFinder(str(filepath))
        finder.visit(tree)

        return finder.print_statements

    except SyntaxError as e:
        # If the file has syntax errors, skip it (will be caught by other checks)
        return []
    except Exception as e:
        # For any other errors, print to stderr and continue
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return []


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Usage: find_print_statements.py <file_or_directory> [file_or_directory...]", file=sys.stderr)
        sys.exit(1)

    all_prints = []

    for arg in sys.argv[1:]:
        path = Path(arg)

        if not path.exists():
            print(f"Warning: {path} does not exist", file=sys.stderr)
            continue

        # Collect Python files to check
        if path.is_file():
            if path.suffix == '.py':
                files = [path]
            else:
                continue
        else:
            # Find all Python files in the directory
            files = list(path.rglob('*.py'))

        # Find prints in each file
        for file in files:
            prints = find_prints_in_file(file)
            all_prints.extend(prints)

    # Output results
    if all_prints:
        # Sort by filename and line number
        all_prints.sort(key=lambda x: (x[2], x[0]))

        for line_num, col_offset, filepath in all_prints:
            # Read the actual line content
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if line_num <= len(lines):
                        line_content = lines[line_num - 1].strip()
                        print(f"{filepath}:{line_num}:{col_offset}:{line_content}")
                    else:
                        print(f"{filepath}:{line_num}:{col_offset}:")
            except Exception:
                print(f"{filepath}:{line_num}:{col_offset}:")

        sys.exit(1)  # Exit with error if prints found
    else:
        sys.exit(0)  # Exit successfully if no prints found


if __name__ == '__main__':
    main()
