"""Tests for config/drafts.py."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dataknobs_bots.config.drafts import ConfigDraftManager, DraftMetadata


class TestDraftMetadata:
    """Tests for DraftMetadata dataclass."""

    def test_to_dict(self) -> None:
        meta = DraftMetadata(
            draft_id="abc123",
            created_at="2025-01-01T00:00:00+00:00",
            last_updated="2025-01-01T01:00:00+00:00",
            stage="configure_llm",
            complete=False,
            config_name="my-bot",
        )
        d = meta.to_dict()
        assert d["id"] == "abc123"
        assert d["created_at"] == "2025-01-01T00:00:00+00:00"
        assert d["stage"] == "configure_llm"
        assert d["config_name"] == "my-bot"
        assert d["complete"] is False

    def test_from_dict_round_trip(self) -> None:
        original = DraftMetadata(
            draft_id="xyz",
            created_at="2025-01-01T00:00:00+00:00",
            last_updated="2025-01-01T00:30:00+00:00",
            stage="review",
            complete=True,
            config_name="test-bot",
        )
        restored = DraftMetadata.from_dict(original.to_dict())
        assert restored.draft_id == original.draft_id
        assert restored.created_at == original.created_at
        assert restored.last_updated == original.last_updated
        assert restored.stage == original.stage
        assert restored.complete == original.complete
        assert restored.config_name == original.config_name


class TestConfigDraftManager:
    """Tests for ConfigDraftManager with real file system (tmp_path)."""

    def _sample_config(self) -> dict[str, Any]:
        return {
            "llm": {"provider": "ollama", "model": "llama3.2"},
            "conversation_storage": {"backend": "memory"},
        }

    def test_create_draft(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        draft_id = manager.create_draft(self._sample_config(), stage="welcome")

        assert len(draft_id) == 8
        draft_path = tmp_path / f"_draft-{draft_id}.yaml"
        assert draft_path.exists()

        with open(draft_path) as f:
            data = yaml.safe_load(f)
        assert data["llm"]["provider"] == "ollama"
        assert data["_draft"]["id"] == draft_id
        assert data["_draft"]["stage"] == "welcome"

    def test_get_draft(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        draft_id = manager.create_draft(self._sample_config(), stage="welcome")

        result = manager.get_draft(draft_id)
        assert result is not None
        config, metadata = result
        assert config["llm"]["provider"] == "ollama"
        assert metadata.draft_id == draft_id
        assert metadata.stage == "welcome"
        # Config should not contain _draft key
        assert "_draft" not in config

    def test_get_draft_not_found(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        assert manager.get_draft("nonexistent") is None

    def test_update_draft(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        draft_id = manager.create_draft(self._sample_config(), stage="welcome")

        updated = self._sample_config()
        updated["system_prompt"] = "Hello"
        manager.update_draft(draft_id, updated, stage="configure_llm")

        result = manager.get_draft(draft_id)
        assert result is not None
        config, metadata = result
        assert config["system_prompt"] == "Hello"
        assert metadata.stage == "configure_llm"

    def test_update_draft_with_config_name(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        draft_id = manager.create_draft(self._sample_config())
        manager.update_draft(
            draft_id,
            self._sample_config(),
            config_name="my-bot",
        )

        # Should also write named file
        named_path = tmp_path / "my-bot.yaml"
        assert named_path.exists()

    def test_update_draft_not_found(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        try:
            manager.update_draft("nonexistent", {})
            assert False, "Should have raised"
        except FileNotFoundError:
            pass

    def test_finalize(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        draft_id = manager.create_draft(self._sample_config(), stage="review")

        final = manager.finalize(draft_id, final_name="my-bot")

        assert final["llm"]["provider"] == "ollama"
        assert "_draft" not in final

        final_path = tmp_path / "my-bot.yaml"
        assert final_path.exists()

        # Draft file should be cleaned up
        draft_path = tmp_path / f"_draft-{draft_id}.yaml"
        assert not draft_path.exists()

        # Final file should not have _draft metadata
        with open(final_path) as f:
            saved = yaml.safe_load(f)
        assert "_draft" not in saved

    def test_finalize_uses_metadata_name(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        draft_id = manager.create_draft(self._sample_config())
        manager.update_draft(draft_id, self._sample_config(), config_name="auto-name")

        final = manager.finalize(draft_id)
        assert (tmp_path / "auto-name.yaml").exists()
        assert final["llm"]["provider"] == "ollama"

    def test_finalize_not_found(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        try:
            manager.finalize("nonexistent", final_name="test")
            assert False, "Should have raised"
        except FileNotFoundError:
            pass

    def test_finalize_no_name(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        draft_id = manager.create_draft(self._sample_config())
        try:
            manager.finalize(draft_id)
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_discard(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        draft_id = manager.create_draft(self._sample_config())

        assert manager.discard(draft_id) is True
        assert not (tmp_path / f"_draft-{draft_id}.yaml").exists()

    def test_discard_not_found(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        assert manager.discard("nonexistent") is False

    def test_list_drafts(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        id1 = manager.create_draft(self._sample_config(), stage="s1")
        id2 = manager.create_draft(self._sample_config(), stage="s2")

        drafts = manager.list_drafts()
        assert len(drafts) == 2
        draft_ids = {d.draft_id for d in drafts}
        assert id1 in draft_ids
        assert id2 in draft_ids

    def test_list_drafts_empty(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        assert manager.list_drafts() == []

    def test_list_drafts_nonexistent_dir(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path / "does-not-exist")
        assert manager.list_drafts() == []

    def test_cleanup_stale(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path, max_age_hours=0.001)

        draft_id = manager.create_draft(self._sample_config())

        # Manually backdate the draft to make it stale
        draft_path = tmp_path / f"_draft-{draft_id}.yaml"
        with open(draft_path) as f:
            data = yaml.safe_load(f)
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
        data["_draft"]["last_updated"] = old_time
        with open(draft_path, "w") as f:
            yaml.dump(data, f)

        cleaned = manager.cleanup_stale()
        assert cleaned == 1
        assert not draft_path.exists()

    def test_cleanup_stale_strips_named_metadata(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        draft_id = manager.create_draft(self._sample_config())
        manager.update_draft(draft_id, self._sample_config(), config_name="named-bot")

        # Backdate the named file
        named_path = tmp_path / "named-bot.yaml"
        with open(named_path) as f:
            data = yaml.safe_load(f)
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
        data["_draft"]["last_updated"] = old_time
        with open(named_path, "w") as f:
            yaml.dump(data, f)

        # Also backdate draft file
        draft_path = tmp_path / f"_draft-{draft_id}.yaml"
        with open(draft_path) as f:
            data = yaml.safe_load(f)
        data["_draft"]["last_updated"] = old_time
        with open(draft_path, "w") as f:
            yaml.dump(data, f)

        cleaned = manager.cleanup_stale(max_age_hours=0.001)
        assert cleaned >= 1

        # Named file should still exist but without _draft
        with open(named_path) as f:
            data = yaml.safe_load(f)
        assert "_draft" not in data

    def test_cleanup_stale_keeps_fresh(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        draft_id = manager.create_draft(self._sample_config())

        cleaned = manager.cleanup_stale(max_age_hours=24.0)
        assert cleaned == 0
        assert (tmp_path / f"_draft-{draft_id}.yaml").exists()

    def test_custom_prefix(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(
            output_dir=tmp_path, draft_prefix="wip-"
        )
        draft_id = manager.create_draft(self._sample_config())
        assert (tmp_path / f"wip-{draft_id}.yaml").exists()

    def test_output_dir_property(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        assert manager.output_dir == tmp_path
