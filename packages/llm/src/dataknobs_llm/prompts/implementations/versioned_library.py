# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Versioned prompt library implementation.

This module provides a prompt library with full versioning support,
combining version management, A/B testing, and metrics tracking.
"""

from typing import Any, Dict, List

from ..base import (
    AbstractPromptLibrary,
    AsyncPromptLibrary,
    MessageIndex,
    PromptTemplateDict,
    RAGConfig,
    as_async,
)
from ..versioning import (
    VersionManager,
    ABTestManager,
    MetricsCollector,
    InMemoryVersionStore,
    PromptVersion,
    PromptExperiment,
    PromptVariant,
    PromptMetrics,
    VersioningStore,
    VersionStatus,
    require_store,
)


def _normalized_base(
    library: AbstractPromptLibrary | AsyncPromptLibrary | None,
) -> AsyncPromptLibrary | None:
    """The base library in the one flavour this library can await.

    Total over its argument, which is the point. It was a two-way branch ---
    ``as_async`` for an :class:`AbstractPromptLibrary`, store as-is otherwise
    --- over a three-way question, so anything it did not recognise was kept as
    though it were already asynchronous. That constructed cleanly and raised
    ``TypeError: ... can't be used in 'await' expression`` from a fallback
    lookup, which only runs for a name carrying no version and so need not
    happen anywhere near construction.

    Args:
        library: A library of either flavour, or ``None`` for no base.

    Returns:
        An :class:`AsyncPromptLibrary`, or ``None``.

    Raises:
        TypeError: If ``library`` answers to neither protocol.
    """
    if library is None:
        return None
    if isinstance(library, AsyncPromptLibrary):
        # Already the flavour this library awaits; wrapping would buy a thread
        # hop and nothing else. Checked first so a class declaring both is
        # taken at its asynchronous word rather than offloaded.
        return library
    if isinstance(library, AbstractPromptLibrary):
        return as_async(library)
    raise TypeError(
        f"base_library must be an AbstractPromptLibrary or an AsyncPromptLibrary, "
        f"got {type(library).__name__}. A library extending BasePromptLibrary answers "
        f"to neither until it names a flavour, because that mixin declares no interface"
    )


class VersionedPromptLibrary(AsyncPromptLibrary):
    """Prompt library with versioning, A/B testing, and metrics tracking.

    An :class:`AsyncPromptLibrary`, because every answer it gives comes from
    an asynchronous version manager. A synchronous consumer reaches it through
    :func:`~dataknobs_llm.prompts.base.views.as_sync`, paying a bridge thread
    and a blocked calling thread for the privilege; an async consumer awaits it
    and pays neither.

    This library extends the base prompt library interface with:
    - Version management with semantic versioning
    - A/B testing experiments with traffic splitting
    - Performance metrics tracking
    - Rollback capabilities

    Example:
        ```python
        from dataknobs_llm.prompts import VersionedPromptLibrary

        # In memory by default; DatabaseVersionStore(db) for any of the
        # seven dataknobs backends.
        library = VersionedPromptLibrary()

        # Create a version
        v1 = await library.create_version(
            name="greeting",
            prompt_type="system",
            template="Hello {{name}}!",
            version="1.0.0"
        )

        # Get latest version (returns PromptTemplateDict for compatibility)
        template = await library.get_system_prompt("greeting")

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
        store: VersioningStore | None = None,
        base_library: AbstractPromptLibrary | AsyncPromptLibrary | None = None,
    ):
        """Initialize versioned prompt library.

        Args:
            store: Where versions, experiments and metrics live. Defaults to
                :class:`~dataknobs_llm.prompts.versioning.InMemoryVersionStore`;
                pass
                :class:`~dataknobs_llm.prompts.versioning.DatabaseVersionStore`
                for persistence across processes. One object serves all three
                managers, which is why the parameter is singular and why the
                protocol it names is the three of theirs together.
            base_library: Optional base library to wrap (for migration). Either
                flavour: a synchronous one is reached through
                :func:`~dataknobs_llm.prompts.base.views.as_async`, so a
                fallback lookup cannot stall this library's caller on a
                filesystem read. ``base_library`` keeps exactly the object that
                was passed; the view is private.

        Raises:
            TypeError: If ``store`` is not a
                :class:`~dataknobs_llm.prompts.versioning.VersioningStore`, or
                if ``base_library`` answers to neither library protocol. For
                the latter a class extending :class:`BasePromptLibrary` alone
                is the likely case: that mixin declares no interface, so such a
                library has to name a flavour before anything can tell which
                one it is.
        """
        if store is None:
            store = InMemoryVersionStore()
        require_store(store, VersioningStore, holder="VersionedPromptLibrary")
        self.store = store
        self.base_library = base_library

        # One store, three managers -- each declaring only the part of it that
        # it uses.
        self.version_manager = VersionManager(store)
        self.ab_test_manager = ABTestManager(store)
        self.metrics_collector = MetricsCollector(store)

        # Cache for converting versions to templates
        self._template_cache: Dict[str, PromptTemplateDict] = {}

    @property
    def base_library(self) -> AbstractPromptLibrary | AsyncPromptLibrary | None:
        """The base library exactly as it was handed over, either flavour.

        A property rather than a plain attribute because the flavour-normalised
        view every lookup actually reaches is derived from it. As a plain pair
        the two drifted on assignment: ``get_metadata`` reported the new library
        while every fallback kept consulting the old one.
        """
        return self._base_library

    @base_library.setter
    def base_library(self, library: AbstractPromptLibrary | AsyncPromptLibrary | None) -> None:
        self._base_library = library
        self._async_base = _normalized_base(library)

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
    ) -> None:
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

    # ===== AsyncPromptLibrary Implementation =====

    async def get_system_prompt(
        self, name: str, version: str = "latest", **kwargs: Any
    ) -> PromptTemplateDict | None:
        """Get a system prompt template.

        Args:
            name: Prompt name
            version: Version string or "latest"
            **kwargs: Additional parameters

        Returns:
            PromptTemplateDict if found, None otherwise
        """
        prompt_version = await self.version_manager.get_version(name, "system", version)

        if not prompt_version:
            # Fall back to base library if available
            if self._async_base:
                return await self._async_base.get_system_prompt(name, **kwargs)
            return None

        return self._version_to_template(prompt_version)

    async def get_user_prompt(
        self, name: str, version: str = "latest", **kwargs: Any
    ) -> PromptTemplateDict | None:
        """Get a user prompt template.

        Args:
            name: Prompt name
            version: Version string or "latest"
            **kwargs: Additional parameters

        Returns:
            PromptTemplateDict if found, None otherwise
        """
        prompt_version = await self.version_manager.get_version(name, "user", version)

        if not prompt_version:
            if self._async_base:
                return await self._async_base.get_user_prompt(name, **kwargs)
            return None

        return self._version_to_template(prompt_version)

    async def list_system_prompts(self) -> List[str]:
        """List all system prompt names.

        The walk belongs to the version manager, which owns the index and its
        key format. Doing it here meant reading that manager's private
        dictionary and parsing its keys back --- a second implementation of a
        private detail, which drifted from it in two ways at once. A listing
        that asks the manager can follow it to a store; one that reads its
        in-memory dict cannot.

        Returns:
            List of prompt names
        """
        names = await self.version_manager.list_names("system")

        # Add from base library if available
        if self._async_base:
            names.update(await self._async_base.list_system_prompts())

        return sorted(names)

    async def list_user_prompts(self) -> List[str]:
        """List all user prompt names.

        Returns:
            List of prompt names
        """
        names = await self.version_manager.list_names("user")

        if self._async_base:
            names.update(await self._async_base.list_user_prompts())

        return sorted(names)

    async def get_message_index(self, name: str, **kwargs: Any) -> MessageIndex | None:
        """Get a message index.

        Note: Message indexes are not versioned in this implementation.
        Falls back to base library if available.

        Args:
            name: Message index name
            **kwargs: Additional parameters

        Returns:
            MessageIndex if found, None otherwise
        """
        if self._async_base:
            return await self._async_base.get_message_index(name, **kwargs)
        return None

    async def list_message_indexes(self) -> List[str]:
        """List all message index names.

        Returns:
            List of message index names
        """
        if self._async_base:
            return await self._async_base.list_message_indexes()
        return []

    async def get_rag_config(self, name: str, **kwargs: Any) -> RAGConfig | None:
        """Get a RAG configuration.

        Note: RAG configs are not versioned in this implementation.
        Falls back to base library if available.

        Args:
            name: RAG config name
            **kwargs: Additional parameters

        Returns:
            RAGConfig if found, None otherwise
        """
        if self._async_base:
            return await self._async_base.get_rag_config(name, **kwargs)
        return None

    async def get_prompt_rag_configs(
        self, prompt_name: str, prompt_type: str = "user", **kwargs: Any
    ) -> List[RAGConfig]:
        """Get RAG configurations for a prompt.

        Args:
            prompt_name: Prompt name
            prompt_type: Prompt type
            **kwargs: Additional parameters

        Returns:
            List of RAG configurations
        """
        if self._async_base:
            return await self._async_base.get_prompt_rag_configs(prompt_name, prompt_type, **kwargs)
        return []

    def get_metadata(self) -> Dict[str, Any]:
        """Get library metadata.

        Synchronous, like its twin: every key here answers from this object's
        own construction rather than from its content. It reported a
        ``version_count`` and an ``experiment_count`` too, which did neither ---
        they counted rows by reaching into two managers' private dictionaries,
        and counting rows in a store is a query, not metadata about a library.
        Nothing read either key.

        Returns:
            Metadata dictionary
        """
        return {
            "type": "VersionedPromptLibrary",
            "store": type(self.store).__name__,
            "has_base_library": self.base_library is not None,
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
            },
        }

        if version.validation:
            template["validation"] = version.validation  # type: ignore[typeddict-item]

        # Cache it
        self._template_cache[cache_key] = template

        return template
