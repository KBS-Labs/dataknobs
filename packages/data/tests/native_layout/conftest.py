"""Shared set-up for the native column layout's tests."""

from __future__ import annotations

from dataknobs_common.testing import declare_import_root

# This directory, so its shared helper (``_native_tables``) imports by bare
# name: the tests run under importlib mode with no ``__init__.py``. Declared
# here rather than for the whole test tree, whose ``integration`` directory
# would then claim a top-level name another package's tests also claim.
declare_import_root(__file__)
