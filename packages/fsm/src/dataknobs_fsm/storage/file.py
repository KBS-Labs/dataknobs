"""File storage backend for execution history.

This is a thin wrapper around UnifiedDatabaseStorage that uses
dataknobs_data's file backend.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from dataknobs_fsm.storage.base import StorageBackend, StorageConfig, StorageFactory
from dataknobs_fsm.storage.database import UnifiedDatabaseStorage

if TYPE_CHECKING:
    from dataknobs_data.database import AsyncDatabase


class FileStorage(UnifiedDatabaseStorage):
    """File storage implementation using dataknobs_data's file backend.

    This storage backend uses dataknobs_data's ``AsyncFileDatabase``,
    which persists *all* records to a single JSON or YAML file at the
    configured ``path``.  Configuration knobs:

    - ``path``: the on-disk file (not a directory).  ``AsyncFileDatabase``
      treats it as a single document; if a ``.gz`` suffix is present
      or ``compression='gzip'`` is set, the file is gzip-compressed
      transparently.
    - ``format``: ``'json'`` (default) or ``'yaml'``.
    - ``compression``: ``'gzip'`` or ``None``.  Forwarded from
      ``StorageConfig.compression`` (a bool).

    Note: there is no directory layout, no file rotation, and no
    separate metadata index — everything lives in the one file.
    """

    def __init__(
        self,
        config: StorageConfig,
        *,
        database: AsyncDatabase | None = None,
        steps_database: AsyncDatabase | None = None,
    ):
        """Initialize file storage.

        Args:
            config: Storage configuration.
            database: Optional pre-built AsyncDatabase instance.
            steps_database: Optional separate AsyncDatabase for step records.
        """
        # Build a local working copy of the connection params and apply
        # file-backend defaults, leaving the (immutable) caller config
        # untouched.
        #
        # Backend selection is now driven by ``StorageConfig.backend``
        # (the canonical enum), so no ``'type'`` injection is needed.
        # This class is registered for
        # ``StorageBackend.FILE`` and the parent's ``_setup_backend``
        # reads from the enum, so the file backend is selected
        # automatically.
        params = dict(config.connection_params)

        # Set default file path if not provided
        params.setdefault('path', './fsm_history')

        # Set file format (json or yaml)
        params.setdefault('format', 'json')

        # Forward FSM ``StorageConfig.compression`` to the data
        # backend's ``compression`` config key (the established
        # ``AsyncFileDatabase`` API at backends/file.py — read at
        # ``self.config.get("compression", None)``).  An earlier
        # implementation injected ``'compress'`` here, which the data
        # backend silently ignored, so file storage was effectively
        # uncompressed regardless of ``StorageConfig.compression``.
        params.setdefault('compression', 'gzip' if config.compression else None)

        config = replace(config, connection_params=params)

        super().__init__(config, database=database, steps_database=steps_database)


# Register file backend
StorageFactory.register(StorageBackend.FILE, FileStorage)
