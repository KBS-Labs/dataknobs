"""Tests for A/B testing functionality."""

import pytest
from collections import Counter

from dataknobs_llm.prompts.versioning import (
    ABTestManager,
    PromptExperiment,
    PromptVariant,
    VersioningError,
)


class TestABTestManager:
    """Test ABTestManager functionality."""

    @pytest.fixture
    def manager(self):
        """Create an ABTestManager instance for testing."""
        return ABTestManager()

    @pytest.fixture
    def variants(self):
        """Create test variants."""
        return [
            PromptVariant("1.0.0", 0.5, "Control"),
            PromptVariant("1.1.0", 0.5, "Treatment"),
        ]

    @pytest.mark.asyncio
    async def test_create_experiment(self, manager, variants):
        """Test creating an A/B test experiment."""
        experiment = await manager.create_experiment(
            name="greeting",
            prompt_type="system",
            variants=variants,
            metadata={"description": "Test greeting variants"}
        )

        assert experiment.name == "greeting"
        assert experiment.prompt_type == "system"
        assert len(experiment.variants) == 2
        assert experiment.status == "running"
        assert experiment.traffic_split == {"1.0.0": 0.5, "1.1.0": 0.5}

    @pytest.mark.asyncio
    async def test_create_experiment_with_custom_split(self, manager):
        """Test creating experiment with custom traffic split."""
        variants = [
            PromptVariant("1.0.0", 0.3, "Control"),
            PromptVariant("1.1.0", 0.7, "Treatment"),
        ]

        custom_split = {"1.0.0": 0.3, "1.1.0": 0.7}

        experiment = await manager.create_experiment(
            name="test",
            prompt_type="system",
            variants=variants,
            traffic_split=custom_split
        )

        assert experiment.traffic_split == custom_split

    @pytest.mark.asyncio
    async def test_create_experiment_normalizes_weights(self, manager):
        """Test that variant weights are normalized."""
        # Weights don't sum to 1.0
        variants = [
            PromptVariant("1.0.0", 1.0, "Control"),
            PromptVariant("1.1.0", 2.0, "Treatment"),
        ]

        experiment = await manager.create_experiment(
            name="test",
            prompt_type="system",
            variants=variants
        )

        # Should be normalized to sum to 1.0
        total = sum(experiment.traffic_split.values())
        assert abs(total - 1.0) < 0.01

        # Check proportions are correct
        assert abs(experiment.traffic_split["1.0.0"] - 1/3) < 0.01
        assert abs(experiment.traffic_split["1.1.0"] - 2/3) < 0.01

    @pytest.mark.asyncio
    async def test_create_experiment_requires_two_variants(self, manager):
        """Test that experiment requires at least 2 variants."""
        variants = [PromptVariant("1.0.0", 1.0, "Only one")]

        with pytest.raises(VersioningError, match="at least 2 variants"):
            await manager.create_experiment(
                name="test",
                prompt_type="system",
                variants=variants
            )

    @pytest.mark.asyncio
    async def test_get_experiment(self, manager, variants):
        """Test retrieving an experiment."""
        created = await manager.create_experiment(
            name="test",
            prompt_type="system",
            variants=variants
        )

        retrieved = await manager.get_experiment(created.experiment_id)

        assert retrieved is not None
        assert retrieved.experiment_id == created.experiment_id
        assert retrieved.name == created.name

    @pytest.mark.asyncio
    async def test_get_nonexistent_experiment(self, manager):
        """Test getting nonexistent experiment returns None."""
        result = await manager.get_experiment("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_experiments(self, manager, variants):
        """Test listing all experiments."""
        exp1 = await manager.create_experiment(
            name="exp1",
            prompt_type="system",
            variants=variants
        )

        exp2 = await manager.create_experiment(
            name="exp2",
            prompt_type="user",
            variants=variants
        )

        experiments = await manager.list_experiments()

        assert len(experiments) == 2
        exp_ids = [e.experiment_id for e in experiments]
        assert exp1.experiment_id in exp_ids
        assert exp2.experiment_id in exp_ids

    @pytest.mark.asyncio
    async def test_list_experiments_filtered_by_name(self, manager, variants):
        """Test filtering experiments by name."""
        await manager.create_experiment(
            name="exp1",
            prompt_type="system",
            variants=variants
        )

        exp2 = await manager.create_experiment(
            name="exp2",
            prompt_type="system",
            variants=variants
        )

        experiments = await manager.list_experiments(name="exp2")

        assert len(experiments) == 1
        assert experiments[0].experiment_id == exp2.experiment_id

    @pytest.mark.asyncio
    async def test_list_experiments_filtered_by_status(self, manager, variants):
        """Test filtering experiments by status."""
        exp1 = await manager.create_experiment(
            name="exp1",
            prompt_type="system",
            variants=variants
        )

        exp2 = await manager.create_experiment(
            name="exp2",
            prompt_type="system",
            variants=variants
        )

        # Pause one experiment
        await manager.update_experiment_status(exp2.experiment_id, "paused")

        # Get running experiments
        running = await manager.list_experiments(status="running")
        assert len(running) == 1
        assert running[0].experiment_id == exp1.experiment_id

        # Get paused experiments
        paused = await manager.list_experiments(status="paused")
        assert len(paused) == 1
        assert paused[0].experiment_id == exp2.experiment_id

    @pytest.mark.asyncio
    async def test_get_random_variant(self, manager, variants):
        """Test getting random variant."""
        experiment = await manager.create_experiment(
            name="test",
            prompt_type="system",
            variants=variants
        )

        # Get 100 random variants
        selections = []
        for _ in range(100):
            variant = await manager.get_random_variant(experiment.experiment_id)
            selections.append(variant)

        # Both variants should be selected
        assert "1.0.0" in selections
        assert "1.1.0" in selections

        # Distribution should be roughly 50/50 (allow variance)
        count_v1 = selections.count("1.0.0")
        count_v2 = selections.count("1.1.0")
        assert 30 <= count_v1 <= 70  # Allow 20% variance from 50%
        assert 30 <= count_v2 <= 70

    @pytest.mark.asyncio
    async def test_get_random_variant_respects_traffic_split(self, manager):
        """Test that random selection respects traffic split."""
        variants = [
            PromptVariant("1.0.0", 0.1, "Control"),
            PromptVariant("1.1.0", 0.9, "Treatment"),
        ]

        experiment = await manager.create_experiment(
            name="test",
            prompt_type="system",
            variants=variants
        )

        # Get 1000 selections
        selections = []
        for _ in range(1000):
            variant = await manager.get_random_variant(experiment.experiment_id)
            selections.append(variant)

        # Count distribution
        count_v1 = selections.count("1.0.0")
        count_v2 = selections.count("1.1.0")

        # Should be roughly 100/900 (allow variance)
        assert 50 <= count_v1 <= 150    # 10% ± 5%
        assert 850 <= count_v2 <= 950   # 90% ± 5%

    @pytest.mark.asyncio
    async def test_get_variant_from_non_running_experiment(self, manager, variants):
        """Test that getting variant from non-running experiment fails."""
        experiment = await manager.create_experiment(
            name="test",
            prompt_type="system",
            variants=variants
        )

        # Pause the experiment
        await manager.update_experiment_status(experiment.experiment_id, "paused")

        # Should raise error
        with pytest.raises(VersioningError, match="not running"):
            await manager.get_random_variant(experiment.experiment_id)

    @pytest.mark.asyncio
    async def test_get_variant_for_user(self, manager, variants):
        """Test user-sticky variant assignment."""
        experiment = await manager.create_experiment(
            name="test",
            prompt_type="system",
            variants=variants
        )

        # Get variant for user1 multiple times
        user1_variant = await manager.get_variant_for_user(
            experiment.experiment_id,
            user_id="user1"
        )

        # Should get same variant on subsequent calls
        for _ in range(10):
            variant = await manager.get_variant_for_user(
                experiment.experiment_id,
                user_id="user1"
            )
            assert variant == user1_variant

    @pytest.mark.asyncio
    async def test_get_variant_for_different_users(self, manager, variants):
        """Test that different users can get different variants."""
        experiment = await manager.create_experiment(
            name="test",
            prompt_type="system",
            variants=variants
        )

        # Get variants for 100 different users
        user_assignments = {}
        for i in range(100):
            user_id = f"user{i}"
            variant = await manager.get_variant_for_user(
                experiment.experiment_id,
                user_id=user_id
            )
            user_assignments[user_id] = variant

        # Both variants should be assigned
        variants_used = set(user_assignments.values())
        assert "1.0.0" in variants_used
        assert "1.1.0" in variants_used

        # Distribution should be roughly even
        count_v1 = sum(1 for v in user_assignments.values() if v == "1.0.0")
        count_v2 = sum(1 for v in user_assignments.values() if v == "1.1.0")
        assert 30 <= count_v1 <= 70
        assert 30 <= count_v2 <= 70

    @pytest.mark.asyncio
    async def test_hash_based_assignment_is_deterministic(self, manager, variants):
        """Test that hash-based assignment is deterministic."""
        experiment = await manager.create_experiment(
            name="test",
            prompt_type="system",
            variants=variants
        )

        # Get assignment for user multiple times
        assignments = []
        for _ in range(10):
            variant = await manager.get_variant_for_user(
                experiment.experiment_id,
                user_id="test_user"
            )
            assignments.append(variant)

        # All assignments should be the same
        assert len(set(assignments)) == 1

    @pytest.mark.asyncio
    async def test_update_experiment_status(self, manager, variants):
        """Test updating experiment status."""
        experiment = await manager.create_experiment(
            name="test",
            prompt_type="system",
            variants=variants
        )

        assert experiment.status == "running"

        # Pause it
        updated = await manager.update_experiment_status(
            experiment.experiment_id,
            "paused"
        )

        assert updated.status == "paused"

    @pytest.mark.asyncio
    async def test_update_experiment_status_to_completed(self, manager, variants):
        """Test that completing experiment sets end_date."""
        experiment = await manager.create_experiment(
            name="test",
            prompt_type="system",
            variants=variants
        )

        assert experiment.end_date is None

        updated = await manager.update_experiment_status(
            experiment.experiment_id,
            "completed"
        )

        assert updated.status == "completed"
        assert updated.end_date is not None

    @pytest.mark.asyncio
    async def test_get_user_assignment(self, manager, variants):
        """Test getting existing user assignment."""
        experiment = await manager.create_experiment(
            name="test",
            prompt_type="system",
            variants=variants
        )

        # No assignment yet
        assignment = await manager.get_user_assignment(
            experiment.experiment_id,
            "user1"
        )
        assert assignment is None

        # Create assignment
        variant = await manager.get_variant_for_user(
            experiment.experiment_id,
            "user1"
        )

        # Should now exist
        assignment = await manager.get_user_assignment(
            experiment.experiment_id,
            "user1"
        )
        assert assignment == variant

    @pytest.mark.asyncio
    async def test_get_experiment_assignments(self, manager, variants):
        """Test getting all assignments for an experiment."""
        experiment = await manager.create_experiment(
            name="test",
            prompt_type="system",
            variants=variants
        )

        # Assign users
        await manager.get_variant_for_user(experiment.experiment_id, "user1")
        await manager.get_variant_for_user(experiment.experiment_id, "user2")
        await manager.get_variant_for_user(experiment.experiment_id, "user3")

        assignments = await manager.get_experiment_assignments(
            experiment.experiment_id
        )

        assert len(assignments) == 3
        assert "user1" in assignments
        assert "user2" in assignments
        assert "user3" in assignments

    @pytest.mark.asyncio
    async def test_delete_experiment(self, manager, variants):
        """Test deleting an experiment."""
        experiment = await manager.create_experiment(
            name="test",
            prompt_type="system",
            variants=variants
        )

        # Verify it exists
        retrieved = await manager.get_experiment(experiment.experiment_id)
        assert retrieved is not None

        # Delete it
        deleted = await manager.delete_experiment(experiment.experiment_id)
        assert deleted is True

        # Verify it's gone
        retrieved = await manager.get_experiment(experiment.experiment_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_experiment_removes_assignments(self, manager, variants):
        """Test that deleting experiment removes user assignments."""
        experiment = await manager.create_experiment(
            name="test",
            prompt_type="system",
            variants=variants
        )

        # Create assignments
        await manager.get_variant_for_user(experiment.experiment_id, "user1")
        await manager.get_variant_for_user(experiment.experiment_id, "user2")

        # Verify assignments exist
        assignments = await manager.get_experiment_assignments(experiment.experiment_id)
        assert len(assignments) == 2

        # Delete experiment
        await manager.delete_experiment(experiment.experiment_id)

        # Assignments should be gone
        assignments = await manager.get_experiment_assignments(experiment.experiment_id)
        assert len(assignments) == 0

    @pytest.mark.asyncio
    async def test_get_variant_distribution(self, manager, variants):
        """Test getting actual user distribution across variants."""
        experiment = await manager.create_experiment(
            name="test",
            prompt_type="system",
            variants=variants
        )

        # Assign 10 users
        for i in range(10):
            await manager.get_variant_for_user(
                experiment.experiment_id,
                user_id=f"user{i}"
            )

        distribution = await manager.get_variant_distribution(
            experiment.experiment_id
        )

        assert "1.0.0" in distribution
        assert "1.1.0" in distribution
        assert distribution["1.0.0"] + distribution["1.1.0"] == 10

    @pytest.mark.asyncio
    async def test_experiment_to_dict_and_from_dict(self, manager, variants):
        """Test serialization/deserialization of PromptExperiment."""
        original = await manager.create_experiment(
            name="test",
            prompt_type="system",
            variants=variants,
            metadata={"description": "Test experiment"}
        )

        # Convert to dict
        exp_dict = original.to_dict()

        # Convert back
        restored = PromptExperiment.from_dict(exp_dict)

        assert restored.experiment_id == original.experiment_id
        assert restored.name == original.name
        assert restored.prompt_type == original.prompt_type
        assert len(restored.variants) == len(original.variants)
        assert restored.traffic_split == original.traffic_split
        assert restored.status == original.status

    @pytest.mark.asyncio
    async def test_variant_weight_validation(self):
        """Test that variant weight must be positive."""
        # Valid: weights > 1.0 are allowed (they're relative weights)
        variant = PromptVariant("1.0.0", 2.0, "Valid")
        assert variant.weight == 2.0

        # Invalid: negative or zero weights
        with pytest.raises(ValueError, match="weight must be positive"):
            PromptVariant("1.0.0", -0.1, "Invalid")

        with pytest.raises(ValueError, match="weight must be positive"):
            PromptVariant("1.0.0", 0.0, "Invalid")
