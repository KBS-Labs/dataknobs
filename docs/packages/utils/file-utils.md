# File Utilities API Documentation

The `file_utils` module provides utilities for file operations, path handling, and file format detection.

## Overview

This module includes functions for:

- Generating file paths recursively
- Reading and writing files with compression support
- Line-by-line file processing
- File format detection (gzip)

## Functions

### filepath_generator()
```python
def filepath_generator(
    rootpath: str,
    descend: bool = True,
    seen: Optional[Set[str]] = None,
    files_only: bool = True,
) -> Generator[str, None, None]
```

Generate all filepaths under the root path.

**Parameters:**
- `rootpath` (str): The root path under which to find files
- `descend` (bool, default=True): True to descend into subdirectories
- `seen` (Optional[Set[str]], default=None): Set of filepaths and/or directories to ignore
- `files_only` (bool, default=True): True to generate only paths to files; False to include directories

**Yields:** Each file path (str)

**Example:**
```python
from dataknobs_utils import file_utils

# Generate all Python files in a directory
for filepath in file_utils.filepath_generator("/path/to/project"):
    if filepath.endswith(".py"):
        print(filepath)

# Include directories in output
for path in file_utils.filepath_generator("/path/to/project", files_only=False):
    print(f"Found: {path}")

# Don't descend into subdirectories
for filepath in file_utils.filepath_generator("/path/to/project", descend=False):
    print(filepath)
```

### fileline_generator()
```python
def fileline_generator(
    filename: str, 
    rootdir: Optional[str] = None
) -> Generator[str, None, None]
```

Generate lines from the file with automatic gzip decompression support.

**Parameters:**
- `filename` (str): The name of the file
- `rootdir` (Optional[str], default=None): The directory of the file

**Yields:** Each stripped file line (str)

**Example:**
```python
from dataknobs_utils import file_utils

# Read lines from a regular text file
for line in file_utils.fileline_generator("data.txt", "/path/to/data"):
    print(f"Line: {line}")

# Read lines from a compressed file (automatic detection)
for line in file_utils.fileline_generator("data.txt.gz", "/path/to/data"):
    print(f"Line: {line}")

# Process log files
log_lines = list(file_utils.fileline_generator("app.log"))
print(f"Total log entries: {len(log_lines)}")
```

### write_lines()
```python
def write_lines(
    outfile: str, 
    lines: List[str], 
    rootdir: Optional[str] = None
) -> None
```

Write the lines to the file with automatic gzip compression support.

**Parameters:**
- `outfile` (str): The name of the output file
- `lines` (List[str]): The lines to write
- `rootdir` (Optional[str], default=None): The directory of the file

**Note:** Lines are automatically sorted before writing.

**Example:**
```python
from dataknobs_utils import file_utils

# Write lines to a regular text file
lines = ["line 3", "line 1", "line 2"]
file_utils.write_lines("output.txt", lines, "/path/to/output")
# File will contain sorted lines: "line 1", "line 2", "line 3"

# Write to a compressed file (automatic compression)
file_utils.write_lines("output.txt.gz", lines, "/path/to/output")

# Process and save filtered data
data_lines = []
for line in file_utils.fileline_generator("input.txt"):
    if "error" in line.lower():
        data_lines.append(line)
        
file_utils.write_lines("errors.txt", data_lines)
```

### is_gzip_file()
```python
def is_gzip_file(filepath: str) -> bool
```

Determine whether the file at filepath is gzipped by checking the magic bytes.

**Parameters:**
- `filepath` (str): The path to the file

**Returns:** True if the file is gzipped, False otherwise

**Example:**
```python
from dataknobs_utils import file_utils

# Check if file is compressed
if file_utils.is_gzip_file("/path/to/data.txt.gz"):
    print("File is compressed")
else:
    print("File is not compressed")

# Process files based on compression
filepath = "/path/to/unknown.txt"
if file_utils.is_gzip_file(filepath):
    # Handle compressed file
    for line in file_utils.fileline_generator(filepath):
        process_line(line)
else:
    # Handle uncompressed file
    with open(filepath, 'r') as f:
        for line in f:
            process_line(line.strip())
```

## Usage Patterns

### Batch File Processing
```python
from dataknobs_utils import file_utils
import os

# Process all text files in a directory
for filepath in file_utils.filepath_generator("/data/input"):
    if filepath.endswith(".txt"):
        # Read and process file
        processed_lines = []
        for line in file_utils.fileline_generator(filepath):
            processed_lines.append(line.upper())
        
        # Write processed output
        basename = os.path.basename(filepath)
        output_path = f"processed_{basename}"
        file_utils.write_lines(output_path, processed_lines, "/data/output")
```

### Log File Analysis
```python
from dataknobs_utils import file_utils
from collections import Counter

# Analyze log files for error patterns
error_counts = Counter()

for filepath in file_utils.filepath_generator("/var/log"):
    if "error" in filepath:
        for line in file_utils.fileline_generator(filepath):
            if "ERROR" in line:
                # Extract error type
                parts = line.split()
                if len(parts) > 3:
                    error_type = parts[3]
                    error_counts[error_type] += 1

print("Top errors:", error_counts.most_common(10))
```

### Data Pipeline Integration
```python
from dataknobs_utils import file_utils
from dataknobs_structures import Tree

# Build tree structure from file hierarchy
def build_file_tree(root_path: str) -> Tree:
    root = Tree(root_path)
    
    for filepath in file_utils.filepath_generator(root_path, files_only=False):
        # Add file/directory to tree structure
        path_parts = filepath.replace(root_path, "").strip("/").split("/")
        current = root
        
        for part in path_parts:
            if part:
                # Find or create child node
                child = None
                if current.has_children():
                    for child_node in current.children:
                        if child_node.data == part:
                            child = child_node
                            break
                
                if child is None:
                    child = current.add_child(part)
                current = child
    
    return root
```

## Error Handling

```python
from dataknobs_utils import file_utils
import os

try:
    # Safe file processing
    if os.path.exists("/path/to/file.txt"):
        lines = list(file_utils.fileline_generator("/path/to/file.txt"))
        processed = [line.strip().upper() for line in lines if line.strip()]
        file_utils.write_lines("/path/to/output.txt", processed)
except IOError as e:
    print(f"File operation failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

- Use generators for memory-efficient processing of large files
- The `fileline_generator` automatically handles gzip files without loading entire file into memory
- Files are processed line-by-line, suitable for large datasets
- The `filepath_generator` uses `os.walk()` internally for efficient directory traversal

## Integration with Other Modules

### With JSON Processing
```python
from dataknobs_utils import file_utils, json_utils
import json

# Process JSON files in a directory
for filepath in file_utils.filepath_generator("/data"):
    if filepath.endswith(".json"):
        for line in file_utils.fileline_generator(filepath):
            try:
                data = json.loads(line)
                # Process JSON data
                processed_data = json_utils.process_data(data)
            except json.JSONDecodeError:
                continue
```

### With Text Processing
```python
from dataknobs_utils import file_utils
from dataknobs_xization import normalize

# Normalize text files
for filepath in file_utils.filepath_generator("/text/data"):
    if filepath.endswith(".txt"):
        normalized_lines = []
        for line in file_utils.fileline_generator(filepath):
            normalized = normalize.basic_normalization_fn(line)
            if normalized.strip():
                normalized_lines.append(normalized)
        
        # Save normalized text
        output_name = f"normalized_{os.path.basename(filepath)}"
        file_utils.write_lines(output_name, normalized_lines, "/text/output")
```