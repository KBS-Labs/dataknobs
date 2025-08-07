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

# Also import commonly used functions for backward compatibility
from dataknobs_utils.json_utils import get_value, build_tree_from_string
from dataknobs_utils.file_utils import (
    filepath_generator,
    fileline_generator,
    write_lines,
    is_gzip_file,
)

# Make submodules available as attributes
__all__ = [
    'elasticsearch_utils',
    'emoji_utils',
    'file_utils',
    'json_extractor',
    'json_utils',
    'llm_utils',
    'pandas_utils',
    'requests_utils',
    'resource_utils',
    'sql_utils',
    'stats_utils',
    'subprocess_utils',
    'sys_utils',
    'xml_utils',
    'get_value',
    'build_tree_from_string',
    'filepath_generator',
    'fileline_generator',
    'write_lines',
    'is_gzip_file',
]