"""Re-export utils from dataknobs-utils package."""

from dataknobs._aliasing import alias_submodules

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

# Attribute binding alone leaves ``dataknobs.utils.<name>`` unresolvable as a
# dotted module path, which is the form pre-split code uses.
alias_submodules(
    __name__,
    (
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
    ),
)

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
