"""In-memory storage backend for execution history.

A thin registration over ``UnifiedDatabaseStorage`` that selects
dataknobs_data's memory backend.
"""

from __future__ import annotations

from dataknobs_fsm.storage.base import StorageBackend, StorageFactory
from dataknobs_fsm.storage.database import UnifiedDatabaseStorage


class InMemoryStorage(UnifiedDatabaseStorage):
    """In-memory storage implementation using dataknobs_data's memory backend.

    Backend selection is driven by ``StorageConfig.backend``, which this
    class is registered against, so the base class needs nothing from here
    to reach ``AsyncMemoryDatabase``. Record-type isolation when sharing a
    single database is handled by the base via ``_history_query()`` /
    ``_steps_query()`` EXISTS filters.

    There is deliberately no ``__init__`` override. One used to inject
    ``max_size=1000`` and ``enable_indexing=True`` into
    ``connection_params``; ``AsyncMemoryDatabase`` accepts neither, so both
    were dropped by the config projection and the store was never bounded
    or indexed by them. That is the same defect the commit which fixed
    ``compress`` -> ``compression`` in the sibling ``FileStorage`` was
    written to remove (902d6eb5) — it repaired the instance in front of it
    and left these two behind. The backend configs now reject an
    unrecognised key rather than dropping it, so the class of defect
    reports instead of accumulating.
    """


# Register memory backend
StorageFactory.register(StorageBackend.MEMORY, InMemoryStorage)
