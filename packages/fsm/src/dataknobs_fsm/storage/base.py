"""Base interfaces and classes for history storage."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dataknobs_common.structured_config import StructuredConfig

from dataknobs_data import SortSpec
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.execution.history import ExecutionHistory, ExecutionStep


class StorageBackend(Enum):
    """Available storage backends."""

    MEMORY = "memory"
    FILE = "file"
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MONGODB = "mongodb"
    S3 = "s3"
    ELASTICSEARCH = "elasticsearch"


@dataclass(frozen=True)
class StorageConfig(StructuredConfig):
    """Configuration for history storage.

    Attributes:
        backend: Storage backend to use.
        connection_params: Backend-specific connection parameters.
        retention_policy: Policy for data retention.
        compression: Whether to compress stored data.
        batch_size: Batch size for bulk operations.
        mode_specific_config: Configuration per data mode.
    """

    backend: StorageBackend = StorageBackend.MEMORY
    connection_params: dict[str, Any] = field(default_factory=dict)
    retention_policy: dict[str, Any] = field(default_factory=dict)
    compression: bool = False
    batch_size: int = 100
    mode_specific_config: dict[DataHandlingMode, dict[str, Any]] = field(
        default_factory=dict
    )

    def get_mode_config(self, mode: DataHandlingMode) -> dict[str, Any]:
        """Get configuration for a specific data mode.

        Args:
            mode: Data mode.

        Returns:
            Configuration for that mode.
        """
        return self.mode_specific_config.get(mode, {})


class IHistoryStorage(ABC):
    """Interface for history storage backends."""

    @abstractmethod
    async def save_history(
        self, history: ExecutionHistory, metadata: dict[str, Any] | None = None
    ) -> str:
        """Save execution history.

        Args:
            history: Execution history to save.
            metadata: Optional metadata.

        Returns:
            Storage ID for the saved history.
        """
        pass

    @abstractmethod
    async def load_history(self, history_id: str) -> ExecutionHistory | None:
        """Load execution history by ID.

        Args:
            history_id: ID of the history to load.

        Returns:
            ExecutionHistory if found, None otherwise.
        """
        pass

    @abstractmethod
    async def save_step(
        self,
        execution_id: str,
        step: ExecutionStep,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a single execution step.

        Args:
            execution_id: Execution ID this step belongs to.
            step: Execution step to save.
            parent_id: Parent step ID if branching.
            metadata: Cross-cutting context (``tenant_id``, ``correlation_id``,
                audit info) routed to the underlying record's ``metadata``
                column for indexable filtering via ``metadata.X``
                dot-notation on SQL/JSONB backends.

        Returns:
            Storage ID for the saved step.
        """
        pass

    @abstractmethod
    async def load_steps(
        self,
        execution_id: str,
        filters: dict[str, Any] | None = None,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[ExecutionStep]:
        """Load execution steps.

        Args:
            execution_id: Execution ID to load steps for.
            filters: Optional equality filters on data columns
                (e.g., ``state_name``, ``status``).  Mirrors the legacy
                positional dict for back-compat.
            filter_metadata: Equality filter over the ``metadata`` column.
                Entries are routed via the ``metadata.X`` field-path
                convention so SQL/JSONB backends push the filter into
                the indexable column.  Symmetry kwarg with the registry
                surfaces (``ArtifactRegistry.list(...)``,
                ``GeneratorRegistry.list_definitions(...)``).
            sort: Optional multi-key sort specification, pushed down to
                the database query.
            limit: Optional row limit, pushed down to the database.
                ``limit=0`` honors Python-slice semantics (empty result).
            offset: Optional row offset, pushed down to the database.

        Returns:
            List of execution steps.
        """
        pass

    @abstractmethod
    async def query_histories(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
    ) -> list[dict[str, Any]]:
        """Query execution histories.

        Args:
            filters: Query filters. Supported keys:

                - ``fsm_name`` — exact match on FSM name
                - ``data_mode`` — exact match on data handling mode
                - ``status`` — exact match on status string
                - ``start_time_after`` — histories started at or after this time
                - ``start_time_before`` — histories started at or before this time
                - ``failed`` — if ``True``, only histories with failed steps;
                  if ``False``, only histories with no failures
                - ``metadata.<key>`` — exact match on a metadata field stored
                  in the ``Record.metadata`` dict, e.g.
                  ``{"metadata.work_order_id": "WO-001"}``.  Dot-notation
                  paths are supported (e.g. ``metadata.tenant.region``).
                  All backends support this: SQL backends use native JSON
                  path extraction, memory/file backends use
                  ``Record.get_value()``.  Equivalent to passing the same
                  keys (without the ``metadata.`` prefix) via the
                  ``filter_metadata=`` kwarg — both routes AND-combine
                  when supplied together.

                Unknown keys are logged as warnings and ignored.
            limit: Maximum results to return.
            offset: Result offset for pagination.
            filter_metadata: Equality filter over the ``metadata`` column.
                Symmetry kwarg with the registry surfaces; entries are
                routed via the ``metadata.X`` field-path convention.
                AND-combines with any ``metadata.<key>`` entries supplied
                via ``filters``.
            sort: Optional multi-key sort specification.  When ``None``,
                the default is ``start_time DESC`` (most-recent-first).

        Returns:
            List of history summary dicts, each containing: ``id``,
            ``fsm_name``, ``data_mode``, ``status``, ``start_time``,
            ``end_time``, ``total_steps``, ``failed_steps``, ``metadata``.
        """
        pass

    @abstractmethod
    async def delete_history(self, history_id: str) -> bool:
        """Delete execution history.

        Args:
            history_id: ID of history to delete.

        Returns:
            True if deleted successfully.
        """
        pass

    @abstractmethod
    async def get_statistics(self, execution_id: str | None = None) -> dict[str, Any]:
        """Get storage statistics.

        Args:
            execution_id: Optional execution ID for specific stats.

        Returns:
            Storage statistics.
        """
        pass

    @abstractmethod
    async def cleanup(self, before_timestamp: float | None = None, keep_failed: bool = True) -> int:
        """Clean up old histories.

        Args:
            before_timestamp: Delete histories before this timestamp.
            keep_failed: Whether to keep failed executions.

        Returns:
            Number of histories deleted.
        """
        pass

    async def close(self) -> None:  # noqa: B027 — intentionally non-abstract; most backends are no-op
        """Close the storage backend and release resources.

        Implementations that own their database connections should close
        them here.  Injected (externally owned) connections must NOT be
        closed — the originator is responsible for their lifecycle.

        The default implementation is a no-op so that subclasses only
        need to override when they manage connections.
        """


class BaseHistoryStorage(IHistoryStorage):
    """Base class for history storage implementations."""

    def __init__(self, config: StorageConfig):
        """Initialize storage with configuration.

        Args:
            config: Storage configuration.
        """
        self.config = config
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the storage backend."""
        if not self._initialized:
            await self._setup_backend()
            self._initialized = True

    @abstractmethod
    async def _setup_backend(self) -> None:
        """Set up the backend storage."""
        pass

    def _serialize_history(self, history: ExecutionHistory) -> dict[str, Any]:
        """Serialize execution history for storage.

        Args:
            history: Execution history to serialize.

        Returns:
            Serialized history.
        """
        data = history.to_dict()

        # Apply compression if configured
        if self.config.compression:
            import base64
            import json
            import zlib

            json_str = json.dumps(data)
            compressed = zlib.compress(json_str.encode("utf-8"))
            data = {"compressed": True, "data": base64.b64encode(compressed).decode("utf-8")}

        return data

    def _deserialize_history(
        self, data: dict[str, Any], fsm_name: str, execution_id: str
    ) -> ExecutionHistory:
        """Deserialize execution history from storage.

        Args:
            data: Serialized history data.
            fsm_name: FSM name.
            execution_id: Execution ID.

        Returns:
            ExecutionHistory instance.
        """
        # Decompress if needed
        if data.get("compressed"):
            import base64
            import json
            import zlib

            compressed = base64.b64decode(data["data"])
            json_str = zlib.decompress(compressed).decode("utf-8")
            data = json.loads(json_str)

        # Use ExecutionHistory.from_dict which properly reconstructs the tree
        return ExecutionHistory.from_dict(data)

    def _apply_retention_policy(self, histories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply retention policy to histories.

        Args:
            histories: List of history records.

        Returns:
            Filtered list based on retention policy.
        """
        if not self.config.retention_policy:
            return histories

        import time

        max_age = self.config.retention_policy.get("max_age_days")
        max_count = self.config.retention_policy.get("max_count")

        if max_age:
            cutoff = time.time() - (max_age * 86400)
            histories = [h for h in histories if h.get("timestamp", 0) > cutoff]

        if max_count and len(histories) > max_count:
            # Keep most recent
            histories = sorted(histories, key=lambda x: x.get("timestamp", 0), reverse=True)
            histories = histories[:max_count]

        return histories


class StorageFactory:
    """Factory for creating history storage instances."""

    _registry: dict[StorageBackend, type[IHistoryStorage]] = {}

    @classmethod
    def register(cls, backend: StorageBackend, storage_class: type[IHistoryStorage]) -> None:
        """Register a storage backend.

        Args:
            backend: Backend type.
            storage_class: Storage class to register.
        """
        cls._registry[backend] = storage_class

    @classmethod
    def create(
        cls,
        config: StorageConfig,
        **kwargs: Any,
    ) -> IHistoryStorage:
        """Create a storage instance.

        Args:
            config: Storage configuration.
            **kwargs: Additional keyword arguments forwarded to the storage
                constructor (e.g. ``database``, ``steps_database``).

        Returns:
            Storage instance.

        Raises:
            ValueError: If backend not registered.
        """
        storage_class = cls._registry.get(config.backend)
        if not storage_class:
            raise ValueError(f"Unknown storage backend: {config.backend}")

        return storage_class(config, **kwargs)  # type: ignore

    @classmethod
    def get_available_backends(cls) -> list[StorageBackend]:
        """Get list of available backends.

        Returns:
            List of registered backends.
        """
        return list(cls._registry.keys())
