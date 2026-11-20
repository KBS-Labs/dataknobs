"""Prompt versions are stamped by the common clock.

Under :func:`dataknobs_common.clock.use_clock` with a
:class:`~dataknobs_common.testing.FixedClock`, a version's ``created_at``
**equals** the fixed instant.
"""

from __future__ import annotations

from datetime import UTC, datetime

from dataknobs_common.clock import use_clock
from dataknobs_common.testing import FixedClock

from dataknobs_llm.prompts.versioning import VersionManager

AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


async def test_a_prompt_version_is_stamped_by_the_clock() -> None:
    manager = VersionManager()
    with use_clock(FixedClock(AT)):
        version = await manager.create_version(
            name="greeting", prompt_type="system", template="Hello {{name}}"
        )
    assert version.created_at == AT
