"""Tests for the data mode system."""

import pytest
from threading import Thread
from typing import Dict, Any

from dataknobs_fsm.core.data_modes import (
    DataHandlingMode,
    DataHandler,
    CopyModeHandler,
    ReferenceModeHandler,
    DirectModeHandler,
    DataModeManager,
)


class TestCopyModeHandler:
    """Tests for COPY mode handler."""
    
    def test_deep_copy_on_entry(self):
        """Test that data is deep copied on entry."""
        handler = CopyModeHandler()
        original = {"key": "value", "nested": {"inner": "data"}}
        
        copied = handler.on_entry(original)
        
        assert copied == original
        assert copied is not original
        assert copied["nested"] is not original["nested"]
    
    def test_isolated_modifications(self):
        """Test that modifications don't affect original."""
        handler = CopyModeHandler()
        original = {"key": "value"}
        
        copied = handler.on_entry(original)
        copied["key"] = "modified"
        copied["new_key"] = "new_value"
        
        assert original["key"] == "value"
        assert "new_key" not in original
    
    def test_commit_on_exit(self):
        """Test commit behavior on exit."""
        handler = CopyModeHandler()
        original = {"key": "value"}
        
        copied = handler.on_entry(original)
        copied["key"] = "modified"
        
        # With commit
        result = handler.on_exit(copied, commit=True)
        assert result["key"] == "modified"
        
        # Without commit
        copied2 = handler.on_entry(original)
        copied2["key"] = "modified"
        result2 = handler.on_exit(copied2, commit=False)
        assert result2["key"] == "value"
    
    def test_concurrent_access_support(self):
        """Test that COPY mode supports concurrent access."""
        handler = CopyModeHandler()
        assert handler.supports_concurrent_access() is True


class TestReferenceModeHandler:
    """Tests for REFERENCE mode handler."""
    
    def test_reference_on_entry(self):
        """Test that data is referenced, not copied."""
        handler = ReferenceModeHandler()
        original = {"key": "value"}
        
        referenced = handler.on_entry(original)
        
        assert referenced is original
    
    def test_in_place_modifications(self):
        """Test that modifications affect original."""
        handler = ReferenceModeHandler()
        original = {"key": "value"}
        
        referenced = handler.on_entry(original)
        referenced["key"] = "modified"
        
        assert original["key"] == "modified"
    
    def test_version_tracking(self):
        """Test that modifications update version."""
        handler = ReferenceModeHandler()
        data = {"key": "value"}
        
        referenced = handler.on_entry(data)
        initial_version = handler._versions.get(id(data), 0)
        
        handler.on_modification(referenced)
        updated_version = handler._versions.get(id(data), 0)
        
        assert updated_version > initial_version
    
    def test_concurrent_access_support(self):
        """Test that REFERENCE mode supports concurrent access."""
        handler = ReferenceModeHandler()
        assert handler.supports_concurrent_access() is True
    
    def test_multiple_references(self):
        """Test handling multiple references to same data."""
        handler = ReferenceModeHandler()
        data = {"key": "value"}
        
        ref1 = handler.on_entry(data)
        ref2 = handler.on_entry(data)
        
        assert ref1 is ref2 is data


class TestDirectModeHandler:
    """Tests for DIRECT mode handler."""
    
    def test_direct_access_on_entry(self):
        """Test that data is used directly."""
        handler = DirectModeHandler()
        original = {"key": "value"}
        
        direct = handler.on_entry(original)
        
        assert direct is original
    
    def test_in_place_modifications(self):
        """Test that modifications are in-place."""
        handler = DirectModeHandler()
        original = {"key": "value"}
        
        direct = handler.on_entry(original)
        direct["key"] = "modified"
        
        assert original["key"] == "modified"
    
    def test_no_concurrent_access(self):
        """Test that DIRECT mode prevents concurrent access."""
        handler = DirectModeHandler()
        data1 = {"key": "value1"}
        data2 = {"key": "value2"}
        
        handler.on_entry(data1)
        
        with pytest.raises(RuntimeError, match="does not support concurrent access"):
            handler.on_entry(data2)
        
        handler.on_exit(data1)
        # Should work after exit
        handler.on_entry(data2)
    
    def test_concurrent_access_support(self):
        """Test that DIRECT mode does not support concurrent access."""
        handler = DirectModeHandler()
        assert handler.supports_concurrent_access() is False


class TestDataModeManager:
    """Tests for the DataModeManager."""
    
    def test_default_mode(self):
        """Test default mode handling."""
        manager = DataModeManager()
        assert manager.default_mode == DataHandlingMode.COPY
        
        handler = manager.get_handler()
        assert isinstance(handler, CopyModeHandler)
    
    def test_get_specific_handler(self):
        """Test getting specific mode handlers."""
        manager = DataModeManager()
        
        copy_handler = manager.get_handler(DataHandlingMode.COPY)
        assert isinstance(copy_handler, CopyModeHandler)
        
        ref_handler = manager.get_handler(DataHandlingMode.REFERENCE)
        assert isinstance(ref_handler, ReferenceModeHandler)
        
        direct_handler = manager.get_handler(DataHandlingMode.DIRECT)
        assert isinstance(direct_handler, DirectModeHandler)
    
    def test_set_default_mode(self):
        """Test changing the default mode."""
        manager = DataModeManager()
        
        manager.set_default_mode(DataHandlingMode.DIRECT)
        assert manager.default_mode == DataHandlingMode.DIRECT
        
        handler = manager.get_handler()
        assert isinstance(handler, DirectModeHandler)
    
    def test_handler_reuse(self):
        """Test that handlers are reused."""
        manager = DataModeManager()
        
        handler1 = manager.get_handler(DataHandlingMode.COPY)
        handler2 = manager.get_handler(DataHandlingMode.COPY)
        
        assert handler1 is handler2


@pytest.mark.integration
class TestDataModesIntegration:
    """Integration tests for data modes."""
    
    def test_mode_switching(self):
        """Test switching between modes."""
        manager = DataModeManager()
        data = {"counter": 0}
        
        # COPY mode
        copy_handler = manager.get_handler(DataHandlingMode.COPY)
        copied = copy_handler.on_entry(data)
        copied["counter"] += 1
        copy_handler.on_exit(copied, commit=True)
        assert data["counter"] == 0  # Original unchanged
        
        # REFERENCE mode
        ref_handler = manager.get_handler(DataHandlingMode.REFERENCE)
        referenced = ref_handler.on_entry(data)
        referenced["counter"] += 1
        ref_handler.on_exit(referenced, commit=True)
        assert data["counter"] == 1  # Original changed
        
        # DIRECT mode
        direct_handler = manager.get_handler(DataHandlingMode.DIRECT)
        direct = direct_handler.on_entry(data)
        direct["counter"] += 1
        direct_handler.on_exit(direct, commit=True)
        assert data["counter"] == 2  # Original changed
    
    def test_concurrent_copy_mode(self):
        """Test concurrent access in COPY mode."""
        handler = CopyModeHandler()
        original = {"counter": 0}
        results = []
        
        def worker(data: Dict[str, Any]):
            copied = handler.on_entry(data)
            copied["counter"] += 1
            result = handler.on_exit(copied, commit=True)
            results.append(result["counter"])
        
        threads = [Thread(target=worker, args=(original,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert original["counter"] == 0  # Original unchanged
        assert all(r == 1 for r in results)  # Each thread got its own copy
