"""Re-export utils from dataknobs-utils package."""

# Import the submodules explicitly to make them available
from dataknobs_utils import (
    elasticsearch_utils,
    emoji_utils,
    file_utils,
    json_extractor,
    json_utils,
    llm_utils,
    pandas_utils,
    requests_utils,
    resource_utils,
    sql_utils,
    stats_utils,
    subprocess_utils,
    sys_utils,
    xml_utils,
)
from dataknobs_utils.file_utils import (
    fileline_generator,
    filepath_generator,
    is_gzip_file,
    write_lines,
)

# Also import commonly used functions for backward compatibility
from dataknobs_utils.json_utils import build_tree_from_string, get_value

# Make submodules available as attributes
__all__ = [
    "build_tree_from_string",
    "elasticsearch_utils",
    "emoji_utils",
    "file_utils",
    "fileline_generator",
    "filepath_generator",
    "get_value",
    "is_gzip_file",
    "json_extractor",
    "json_utils",
    "llm_utils",
    "pandas_utils",
    "requests_utils",
    "resource_utils",
    "sql_utils",
    "stats_utils",
    "subprocess_utils",
    "sys_utils",
    "write_lines",
    "xml_utils",
]
