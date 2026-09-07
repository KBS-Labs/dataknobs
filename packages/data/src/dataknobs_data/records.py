"""Structured data records, re-exported.

``Record`` is defined in :mod:`dataknobs_common.records`. It is re-exported
here so that ``from dataknobs_data.records import Record`` keeps resolving.
"""

from __future__ import annotations

from dataknobs_common.records import Record

__all__ = ["Record"]
