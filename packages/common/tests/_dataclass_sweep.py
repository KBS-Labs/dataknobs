"""This package's dataclass sweep, bound once so both suites share the walk.

The sweep itself is :class:`~dataknobs_common.testing.DataclassSweep`, which
ships beside the other testing constructs because it is about *a* package's
dataclasses rather than this one's --- ``dataknobs-data`` asks the same
question of its own tree. What stays here is the binding: two suites walk one
instance, so the tree is imported and its ``TYPE_CHECKING`` blocks replayed
once per session rather than once per suite.
"""

from __future__ import annotations

import dataknobs_common
from dataknobs_common.testing import DataclassSweep

#: The walk both the hashable-contract census and the relation-spelling
#: population read. One instance, because :meth:`DataclassSweep.every_dataclass`
#: caches per instance and two would pay for the walk twice.
SWEEP = DataclassSweep(dataknobs_common)
