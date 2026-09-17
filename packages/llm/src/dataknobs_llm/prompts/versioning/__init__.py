# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Prompt versioning and A/B testing support.

This module provides:
- **Version Management**: Track prompt versions over time with semantic versioning
- **A/B Testing**: Create experiments with multiple variants and traffic splits
- **Metrics Tracking**: Monitor performance metrics for each version
- **Rollback Support**: Easy rollback to previous versions

Quick Start:

    from dataknobs_data import async_database_factory
    from dataknobs_llm.prompts import VersionedPromptLibrary
    from dataknobs_llm.prompts.versioning import DatabaseVersionStore, PromptVariant

    # Create a versioned library over any of the seven dataknobs backends.
    # Omit ``store`` for the in-memory default. The factory does not connect,
    # so a backend that needs a connection wants one before the first call.
    db = async_database_factory.create(backend="memory")
    await db.connect()
    library = VersionedPromptLibrary(store=DatabaseVersionStore(db))

    # Create a version
    v1 = await library.create_version(
        name="greeting",
        prompt_type="system",
        template="Hello {{name}}!",
        version="1.0.0",
        metadata={"author": "alice"}
    )

    # Create A/B test experiment
    experiment = await library.create_experiment(
        name="greeting",
        prompt_type="system",
        variants=[
            PromptVariant("1.0.0", 0.5, "Original"),
            PromptVariant("1.1.0", 0.5, "More friendly")
        ]
    )

    # Get variant for user
    variant = await library.get_variant_for_user(
        experiment.experiment_id,
        user_id="user123"
    )
"""

from dataknobs_llm.exceptions import VersioningError

from .types import (
    PromptVersion,
    PromptExperiment,
    PromptVariant,
    PromptMetrics,
    VersionStatus,
    MetricEvent,
)

from .store import (
    DatabaseVersionStore,
    ExperimentStore,
    InMemoryVersionStore,
    MetricsStore,
    VersionStore,
    VersioningStore,
    require_store,
)

from .version_manager import VersionManager

from .ab_testing import ABTestManager

from .metrics import MetricsCollector

__all__ = [
    # Types
    "PromptVersion",
    "PromptExperiment",
    "PromptVariant",
    "PromptMetrics",
    "VersioningError",
    "VersionStatus",
    "MetricEvent",
    # Managers
    "VersionManager",
    "ABTestManager",
    "MetricsCollector",
    # Storage
    "VersionStore",
    "ExperimentStore",
    "MetricsStore",
    "VersioningStore",
    "InMemoryVersionStore",
    "DatabaseVersionStore",
    "require_store",
]
