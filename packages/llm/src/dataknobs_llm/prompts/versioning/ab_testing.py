# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A/B testing management for prompt experiments.

This module provides:
- Experiment creation and management
- Random variant selection
- User-sticky variant assignment
- Traffic split management
"""

import uuid
import hashlib
import random
from typing import Any, Dict, List
from datetime import UTC, datetime

from dataknobs_llm.exceptions import VersioningError

from .store import ExperimentStore, InMemoryVersionStore, require_store
from .types import (
    PromptExperiment,
    PromptVariant,
)


class ABTestManager:
    """Manages A/B test experiments for prompts.

    Supports multiple selection strategies:
    - Random: Each request gets a random variant based on traffic split
    - User-sticky: Same user always gets same variant (consistent experience)

    Example:
        ```python
        # In memory when nothing is passed; DatabaseVersionStore(db)
        # keeps experiments in any of the seven dataknobs backends.
        manager = ABTestManager()

        # Create experiment
        experiment = await manager.create_experiment(
            name="greeting",
            prompt_type="system",
            variants=[
                PromptVariant("1.0.0", 0.5, "Control"),
                PromptVariant("1.1.0", 0.5, "Treatment")
            ]
        )

        # Get variant for user (sticky assignment)
        variant_version = await manager.get_variant_for_user(
            experiment.experiment_id,
            user_id="user123"
        )

        # Get random variant
        variant_version = await manager.get_random_variant(
            experiment.experiment_id
        )
        ```
    """

    def __init__(self, store: ExperimentStore | None = None):
        """Initialize A/B test manager.

        Args:
            store: Where experiments and their user assignments live. Defaults
                to :class:`~.store.InMemoryVersionStore`, which is what this
                manager used to hold in two instance dictionaries. Pass
                :class:`~.store.DatabaseVersionStore` for any of the seven
                ``dataknobs_data`` backends.

        Raises:
            TypeError: If ``store`` is not an :class:`~.store.ExperimentStore`.
                An assignment used to be persisted as a bare string under a
                composed key, which is not a record and which no backend could
                hold, so an object written for the old parameter is caught here.
        """
        if store is None:
            store = InMemoryVersionStore()
        require_store(store, ExperimentStore, holder="ABTestManager")
        self.store = store

    async def create_experiment(
        self,
        name: str,
        prompt_type: str,
        variants: List[PromptVariant],
        traffic_split: Dict[str, float] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> PromptExperiment:
        """Create a new A/B test experiment.

        Args:
            name: Prompt name
            prompt_type: Prompt type
            variants: List of variants to test
            traffic_split: Optional custom traffic split (if None, derives from variant weights)
            metadata: Additional metadata

        Returns:
            Created PromptExperiment

        Raises:
            VersioningError: If variants are invalid or traffic split doesn't sum to 1.0
        """
        if len(variants) < 2:
            raise VersioningError("Experiment must have at least 2 variants")

        # Generate experiment ID
        experiment_id = str(uuid.uuid4())

        # Derive traffic split from variant weights if not provided
        if traffic_split is None:
            # Normalize weights to ensure they sum to 1.0
            total_weight = sum(v.weight for v in variants)
            traffic_split = {v.version: v.weight / total_weight for v in variants}

        # Create experiment
        experiment = PromptExperiment(
            experiment_id=experiment_id,
            name=name,
            prompt_type=prompt_type,
            variants=variants,
            traffic_split=traffic_split,
            start_date=datetime.now(UTC),
            status="running",
            metadata=metadata or {},
        )

        await self.store.save_experiment(experiment)

        return experiment

    async def get_experiment(
        self,
        experiment_id: str,
    ) -> PromptExperiment | None:
        """Retrieve an experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            PromptExperiment if found, None otherwise
        """
        return await self.store.load_experiment(experiment_id)

    async def list_experiments(
        self,
        name: str | None = None,
        prompt_type: str | None = None,
        status: str | None = None,
    ) -> List[PromptExperiment]:
        """List experiments with optional filters.

        Args:
            name: Filter by prompt name
            prompt_type: Filter by prompt type
            status: Filter by status ("running", "paused", "completed")

        Returns:
            List of matching experiments
        """
        return await self.store.load_experiments(name, prompt_type, status)

    async def get_random_variant(
        self,
        experiment_id: str,
    ) -> str:
        """Get a random variant based on traffic split.

        Each call returns a potentially different variant.

        Args:
            experiment_id: Experiment ID

        Returns:
            Version string of selected variant

        Raises:
            VersioningError: If experiment not found
        """
        experiment = await self._running_experiment(experiment_id)

        # Weighted random selection
        versions = list(experiment.traffic_split.keys())
        weights = list(experiment.traffic_split.values())

        return random.choices(versions, weights=weights)[0]

    async def get_variant_for_user(
        self,
        experiment_id: str,
        user_id: str,
    ) -> str:
        """Get variant for a specific user (sticky assignment).

        The same user always gets the same variant for consistent experience.
        Uses hash-based assignment to ensure deterministic selection.

        Args:
            experiment_id: Experiment ID
            user_id: User identifier

        Returns:
            Version string of assigned variant

        Raises:
            VersioningError: If experiment not found
        """
        experiment = await self._running_experiment(experiment_id)

        # Check if user already has assignment
        existing = await self.store.load_assignment(experiment_id, user_id)
        if existing:
            return existing

        # Assign user to variant using hash-based selection
        assigned_version = self._hash_based_assignment(user_id, experiment.traffic_split)

        await self.store.save_assignment(experiment_id, user_id, assigned_version)

        return assigned_version

    async def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        end_date: datetime | None = None,
    ) -> PromptExperiment:
        """Update experiment status.

        Args:
            experiment_id: Experiment ID
            status: New status ("running", "paused", "completed")
            end_date: Optional end date (auto-set to now if status is "completed")

        Returns:
            Updated experiment

        Raises:
            VersioningError: If experiment not found
        """
        experiment = await self.store.load_experiment(experiment_id)
        if not experiment:
            raise VersioningError(f"Experiment not found: {experiment_id}")

        experiment.status = status

        if status == "completed" and end_date is None:
            experiment.end_date = datetime.now(UTC)
        elif end_date:
            experiment.end_date = end_date

        await self.store.save_experiment(experiment)

        return experiment

    async def get_user_assignment(
        self,
        experiment_id: str,
        user_id: str,
    ) -> str | None:
        """Get existing user assignment without creating a new one.

        Args:
            experiment_id: Experiment ID
            user_id: User ID

        Returns:
            Assigned version if exists, None otherwise
        """
        return await self.store.load_assignment(experiment_id, user_id)

    async def get_experiment_assignments(
        self,
        experiment_id: str,
    ) -> Dict[str, str]:
        """Get all user assignments for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Dictionary mapping user_id to assigned version
        """
        return await self.store.load_assignments(experiment_id)

    async def delete_experiment(
        self,
        experiment_id: str,
    ) -> bool:
        """Delete an experiment.

        Note: This also removes all user assignments.

        Args:
            experiment_id: Experiment ID

        Returns:
            True if deleted, False if not found. The store removes the
            assignments with it, so neither outlives the other.
        """
        return await self.store.delete_experiment(experiment_id)

    # ===== Helper Methods =====

    async def _running_experiment(self, experiment_id: str) -> PromptExperiment:
        """Load an experiment that is accepting traffic, or explain why not.

        Args:
            experiment_id: Experiment ID

        Returns:
            The experiment, with status ``"running"``

        Raises:
            VersioningError: If the experiment is absent or is not running
        """
        experiment = await self.store.load_experiment(experiment_id)
        if not experiment:
            raise VersioningError(f"Experiment not found: {experiment_id}")

        if experiment.status != "running":
            raise VersioningError(
                f"Experiment {experiment_id} is not running (status: {experiment.status})"
            )

        return experiment

    def _hash_based_assignment(
        self,
        user_id: str,
        traffic_split: Dict[str, float],
    ) -> str:
        """Assign user to variant using consistent hash-based selection.

        This ensures the same user always gets the same variant.

        Args:
            user_id: User identifier
            traffic_split: Version to percentage mapping

        Returns:
            Selected version string
        """
        # Hash user_id to get a deterministic number
        hash_hex = hashlib.md5(user_id.encode()).hexdigest()
        hash_val = int(hash_hex, 16)

        # Map to [0, 1) range
        normalized = (hash_val % 1000) / 1000.0

        # Select variant based on cumulative traffic split
        cumulative = 0.0
        versions = sorted(traffic_split.keys())  # Sort for consistency

        for version in versions:
            cumulative += traffic_split[version]
            if normalized < cumulative:
                return version

        # Fallback to last version (handles floating point errors)
        return versions[-1]

    async def get_variant_distribution(
        self,
        experiment_id: str,
    ) -> Dict[str, int]:
        """Get actual distribution of users across variants.

        Args:
            experiment_id: Experiment ID

        Returns:
            Dictionary mapping version to user count

        Raises:
            VersioningError: If experiment not found
        """
        experiment = await self.store.load_experiment(experiment_id)
        if not experiment:
            raise VersioningError(f"Experiment not found: {experiment_id}")

        assignments = await self.store.load_assignments(experiment_id)
        distribution: Dict[str, int] = {v.version: 0 for v in experiment.variants}

        for version in assignments.values():
            if version in distribution:
                distribution[version] += 1

        return distribution
