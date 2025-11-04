"""Versioned prompt library implementation.

This module provides a prompt library with full versioning support,
combining version management, A/B testing, and metrics tracking.
"""

from typing import Any, Dict, List

from ..base import AbstractPromptLibrary, PromptTemplateDict, MessageIndex, RAGConfig
from ..versioning import (
    VersionManager,
    ABTestManager,
    MetricsCollector,
    PromptVersion,
    PromptExperiment,
    PromptVariant,
    PromptMetrics,
    VersionStatus,
)


class VersionedPromptLibrary(AbstractPromptLibrary):
    """Prompt library with versioning, A/B testing, and metrics tracking.

    This library extends the base prompt library interface with:
    - Version management with semantic versioning
    - A/B testing experiments with traffic splitting
    - Performance metrics tracking
    - Rollback capabilities

    Example:
        ```python
        from dataknobs_llm.prompts import VersionedPromptLibrary

        # Create library (with optional backend storage)
        library = VersionedPromptLibrary(storage=backend)

        # Create a version
        v1 = await library.create_version(
            name="greeting",
            prompt_type="system",
            template="Hello {{name}}!",
            version="1.0.0"
        )

        # Get latest version (returns PromptTemplateDict for compatibility)
        template = library.get_system_prompt("greeting")

        # Create A/B test
        experiment = await library.create_experiment(
            name="greeting",
            prompt_type="system",
            variants=[
                PromptVariant("1.0.0", 0.5, "Original"),
                PromptVariant("1.1.0", 0.5, "Improved")
            ]
        )

        # Track metrics
        await library.record_usage(
            version_id=v1.version_id,
            success=True,
            response_time=0.5,
            tokens=100
        )
        ```
    """

    def __init__(
        self,
        storage: Any | None = None,
        base_library: AbstractPromptLibrary | None = None,
    ):
        """Initialize versioned prompt library.

        Args:
            storage: Backend storage for persistence (None for in-memory)
            base_library: Optional base library to wrap (for migration)
        """
        self.storage = storage
        self.base_library = base_library

        # Initialize managers
        self.version_manager = VersionManager(storage)
        self.ab_test_manager = ABTestManager(storage)
        self.metrics_collector = MetricsCollector(storage)

        # Cache for converting versions to templates
        self._template_cache: Dict[str, PromptTemplateDict] = {}

    # ===== Version Management API =====

    async def create_version(
        self,
        name: str,
        prompt_type: str,
        template: str,
        version: str | None = None,
        defaults: Dict[str, Any] | None = None,
        validation: Dict[str, Any] | None = None,
        metadata: Dict[str, Any] | None = None,
        created_by: str | None = None,
        tags: List[str] | None = None,
        status: VersionStatus = VersionStatus.ACTIVE,
    ) -> PromptVersion:
        """Create a new prompt version.

        Args:
            name: Prompt name
            prompt_type: Prompt type ("system", "user", "message")
            template: Template content
            version: Semantic version (auto-increments if None)
            defaults: Default parameter values
            validation: Validation configuration
            metadata: Additional metadata
            created_by: Creator username/ID
            tags: List of tags
            status: Initial status

        Returns:
            Created PromptVersion
        """
        # Find parent version (latest version)
        latest = await self.version_manager.get_version(name, prompt_type)
        parent_version = latest.version_id if latest else None

        return await self.version_manager.create_version(
            name=name,
            prompt_type=prompt_type,
            template=template,
            version=version,
            defaults=defaults,
            validation=validation,
            metadata=metadata,
            created_by=created_by,
            parent_version=parent_version,
            tags=tags,
            status=status,
        )

    async def get_version(
        self,
        name: str,
        prompt_type: str,
        version: str = "latest",
    ) -> PromptVersion | None:
        """Get a specific prompt version.

        Args:
            name: Prompt name
            prompt_type: Prompt type
            version: Version string or "latest"

        Returns:
            PromptVersion if found, None otherwise
        """
        return await self.version_manager.get_version(name, prompt_type, version)

    async def list_versions(
        self,
        name: str,
        prompt_type: str,
        tags: List[str] | None = None,
        status: VersionStatus | None = None,
    ) -> List[PromptVersion]:
        """List all versions of a prompt.

        Args:
            name: Prompt name
            prompt_type: Prompt type
            tags: Filter by tags
            status: Filter by status

        Returns:
            List of PromptVersion objects
        """
        return await self.version_manager.list_versions(name, prompt_type, tags, status)

    async def tag_version(
        self,
        version_id: str,
        tag: str,
    ) -> PromptVersion:
        """Add a tag to a version.

        Args:
            version_id: Version ID
            tag: Tag to add

        Returns:
            Updated PromptVersion
        """
        return await self.version_manager.tag_version(version_id, tag)

    # ===== A/B Testing API =====

    async def create_experiment(
        self,
        name: str,
        prompt_type: str,
        variants: List[PromptVariant],
        traffic_split: Dict[str, float] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> PromptExperiment:
        """Create an A/B test experiment.

        Args:
            name: Prompt name
            prompt_type: Prompt type
            variants: List of variants to test
            traffic_split: Optional custom traffic split
            metadata: Additional metadata

        Returns:
            Created PromptExperiment
        """
        return await self.ab_test_manager.create_experiment(
            name=name,
            prompt_type=prompt_type,
            variants=variants,
            traffic_split=traffic_split,
            metadata=metadata,
        )

    async def get_variant_for_user(
        self,
        experiment_id: str,
        user_id: str,
    ) -> str:
        """Get variant for a user (sticky assignment).

        Args:
            experiment_id: Experiment ID
            user_id: User identifier

        Returns:
            Version string of assigned variant
        """
        return await self.ab_test_manager.get_variant_for_user(experiment_id, user_id)

    async def get_random_variant(
        self,
        experiment_id: str,
    ) -> str:
        """Get a random variant.

        Args:
            experiment_id: Experiment ID

        Returns:
            Version string of selected variant
        """
        return await self.ab_test_manager.get_random_variant(experiment_id)

    async def get_experiment(
        self,
        experiment_id: str,
    ) -> PromptExperiment | None:
        """Get an experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            PromptExperiment if found, None otherwise
        """
        return await self.ab_test_manager.get_experiment(experiment_id)

    async def list_experiments(
        self,
        name: str | None = None,
        prompt_type: str | None = None,
        status: str | None = None,
    ) -> List[PromptExperiment]:
        """List experiments.

        Args:
            name: Filter by prompt name
            prompt_type: Filter by prompt type
            status: Filter by status

        Returns:
            List of experiments
        """
        return await self.ab_test_manager.list_experiments(name, prompt_type, status)

    # ===== Metrics API =====

    async def record_usage(
        self,
        version_id: str,
        success: bool = True,
        response_time: float | None = None,
        tokens: int | None = None,
        user_rating: float | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        """Record a usage event for metrics tracking.

        Args:
            version_id: Version ID
            success: Whether the use was successful
            response_time: Response time in seconds
            tokens: Number of tokens used
            user_rating: User rating 1-5
            metadata: Additional event metadata
        """
        await self.metrics_collector.record_event(
            version_id=version_id,
            success=success,
            response_time=response_time,
            tokens=tokens,
            user_rating=user_rating,
            metadata=metadata,
        )

    async def get_metrics(
        self,
        version_id: str,
    ) -> PromptMetrics:
        """Get metrics for a version.

        Args:
            version_id: Version ID

        Returns:
            PromptMetrics with aggregated statistics
        """
        return await self.metrics_collector.get_metrics(version_id)

    async def compare_variants(
        self,
        version_ids: List[str],
    ) -> Dict[str, PromptMetrics]:
        """Compare metrics across versions.

        Args:
            version_ids: List of version IDs

        Returns:
            Dictionary mapping version_id to PromptMetrics
        """
        return await self.metrics_collector.compare_variants(version_ids)

    # ===== AbstractPromptLibrary Implementation =====

    def get_system_prompt(
        self,
        name: str,
        version: str = "latest",
        **kwargs: Any
    ) -> PromptTemplateDict | None:
        """Get a system prompt template.

        This method is synchronous for compatibility with AbstractPromptLibrary.
        For async version access, use get_version() directly.

        Args:
            name: Prompt name
            version: Version string or "latest"
            **kwargs: Additional parameters

        Returns:
            PromptTemplateDict if found, None otherwise
        """
        import asyncio

        # Run async version retrieval
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        prompt_version = loop.run_until_complete(
            self.version_manager.get_version(name, "system", version)
        )

        if not prompt_version:
            # Fall back to base library if available
            if self.base_library:
                return self.base_library.get_system_prompt(name, **kwargs)
            return None

        return self._version_to_template(prompt_version)

    def get_user_prompt(
        self,
        name: str,
        version: str = "latest",
        **kwargs: Any
    ) -> PromptTemplateDict | None:
        """Get a user prompt template.

        Args:
            name: Prompt name
            version: Version string or "latest"
            **kwargs: Additional parameters

        Returns:
            PromptTemplateDict if found, None otherwise
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        prompt_version = loop.run_until_complete(
            self.version_manager.get_version(name, "user", version)
        )

        if not prompt_version:
            if self.base_library:
                return self.base_library.get_user_prompt(name, **kwargs)
            return None

        return self._version_to_template(prompt_version)

    def list_system_prompts(self) -> List[str]:
        """List all system prompt names.

        Returns:
            List of prompt names
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Get unique prompt names from version index
        names = set()
        for key in self.version_manager._version_index.keys():
            name, ptype = key.split(":", 1)
            if ptype == "system":
                names.add(name)

        # Add from base library if available
        if self.base_library:
            names.update(self.base_library.list_system_prompts())

        return sorted(names)

    def list_user_prompts(self) -> List[str]:
        """List all user prompt names.

        Returns:
            List of prompt names
        """
        names = set()
        for key in self.version_manager._version_index.keys():
            name, ptype = key.split(":", 1)
            if ptype == "user":
                names.add(name)

        if self.base_library:
            names.update(self.base_library.list_user_prompts())

        return sorted(names)

    def get_message_index(
        self,
        name: str,
        **kwargs: Any
    ) -> MessageIndex | None:
        """Get a message index.

        Note: Message indexes are not versioned in this implementation.
        Falls back to base library if available.

        Args:
            name: Message index name
            **kwargs: Additional parameters

        Returns:
            MessageIndex if found, None otherwise
        """
        if self.base_library:
            return self.base_library.get_message_index(name, **kwargs)
        return None

    def list_message_indexes(self) -> List[str]:
        """List all message index names.

        Returns:
            List of message index names
        """
        if self.base_library:
            return self.base_library.list_message_indexes()
        return []

    def get_rag_config(
        self,
        name: str,
        **kwargs: Any
    ) -> RAGConfig | None:
        """Get a RAG configuration.

        Note: RAG configs are not versioned in this implementation.
        Falls back to base library if available.

        Args:
            name: RAG config name
            **kwargs: Additional parameters

        Returns:
            RAGConfig if found, None otherwise
        """
        if self.base_library:
            return self.base_library.get_rag_config(name, **kwargs)
        return None

    def get_prompt_rag_configs(
        self,
        prompt_name: str,
        prompt_type: str = "user",
        **kwargs: Any
    ) -> List[RAGConfig]:
        """Get RAG configurations for a prompt.

        Args:
            prompt_name: Prompt name
            prompt_type: Prompt type
            **kwargs: Additional parameters

        Returns:
            List of RAG configurations
        """
        if self.base_library:
            return self.base_library.get_prompt_rag_configs(prompt_name, prompt_type, **kwargs)
        return []

    def get_metadata(self) -> Dict[str, Any]:
        """Get library metadata.

        Returns:
            Metadata dictionary
        """
        return {
            "type": "VersionedPromptLibrary",
            "storage": str(type(self.storage).__name__) if self.storage else "in-memory",
            "has_base_library": self.base_library is not None,
            "version_count": len(self.version_manager._versions),
            "experiment_count": len(self.ab_test_manager._experiments),
        }

    # ===== Helper Methods =====

    def _version_to_template(self, version: PromptVersion) -> PromptTemplateDict:
        """Convert PromptVersion to PromptTemplateDict for compatibility."""
        # Check cache
        cache_key = version.version_id
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]

        template: PromptTemplateDict = {
            "template": version.template,
            "defaults": version.defaults,
            "metadata": {
                **version.metadata,
                "version_id": version.version_id,
                "version": version.version,
                "created_at": version.created_at.isoformat(),
                "tags": version.tags,
                "status": version.status.value,
            }
        }

        if version.validation:
            template["validation"] = version.validation  # type: ignore[typeddict-item]

        # Cache it
        self._template_cache[cache_key] = template

        return template
