"""
Dataknobs - Legacy compatibility package.

This package maintains backward compatibility for existing users.
Please consider using the modular packages instead:
- dataknobs-common
- dataknobs-structures
- dataknobs-utils
- dataknobs-xization
"""

import warnings

warnings.warn(
    "The 'dataknobs' package is deprecated. "
    "Please use the modular packages (dataknobs-common, dataknobs-structures, "
    "dataknobs-utils, dataknobs-xization) instead.",
    DeprecationWarning,
    stacklevel=2,
)

__version__ = "0.1.1"
