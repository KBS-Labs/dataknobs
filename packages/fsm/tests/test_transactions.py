"""Tests for transaction management."""

import pytest
from typing import List

from dataknobs_fsm.core.transactions import (
    TransactionStrategy,
    TransactionState,
    Transaction,
    SingleTransactionManager,
    BatchTransactionManager,
    ManualTransactionManager,
    create_transaction_manager,
)


class TestTransaction:
    """Tests for the Transaction class."""
    
    def test_transaction_creation(self):
        """Test creating a transaction."""
        txn = Transaction("test_txn", TransactionStrategy.SINGLE)
        
        assert txn.id == "test_txn"
        assert txn.strategy == TransactionStrategy.SINGLE
        assert txn.state == TransactionState.PENDING
        assert txn.operations == []
        assert txn.metadata == {}
    
    def test_add_operation(self):
        """Test adding operations to a transaction."""
        txn = Transaction("test_txn", TransactionStrategy.SINGLE)
        txn.state = TransactionState.ACTIVE
        
        operation = {"type": "update", "data": "test"}
        txn.add_operation(operation)
        
        assert len(txn.operations) == 1
        assert txn.operations[0] == operation
    
    def test_cannot_add_to_inactive_transaction(self):
        """Test that operations cannot be added to inactive transactions."""
        txn = Transaction("test_txn", TransactionStrategy.SINGLE)
        
        with pytest.raises(RuntimeError, match="Cannot add operation"):
            txn.add_operation({"type": "test"})
    
    def test_rollback_handlers(self):
        """Test rollback handler execution."""
        txn = Transaction("test_txn", TransactionStrategy.SINGLE)
        txn.state = TransactionState.ACTIVE
        
        rollback_calls = []
        
        def handler1():
            rollback_calls.append("handler1")
        
        def handler2():
            rollback_calls.append("handler2")
        
        txn.add_rollback_handler(handler1)
        txn.add_rollback_handler(handler2)
        
        txn.rollback()
        
        # Handlers called in reverse order
        assert rollback_calls == ["handler2", "handler1"]
        assert txn.state == TransactionState.ROLLED_BACK
    
    def test_commit(self):
        """Test committing a transaction."""
        txn = Transaction("test_txn", TransactionStrategy.SINGLE)
        txn.state = TransactionState.ACTIVE
        
        txn.commit()
        
        assert txn.state == TransactionState.COMMITTED


class TestSingleTransactionManager:
    """Tests for SingleTransactionManager."""
    
    def test_begin_transaction(self):
        """Test beginning a single transaction."""
        manager = SingleTransactionManager()
        
        txn = manager.begin_transaction()
        
        assert txn is not None
        assert txn.strategy == TransactionStrategy.SINGLE
        assert txn.state == TransactionState.ACTIVE
        assert manager.get_active_transaction() == txn
    
    def test_auto_commit_previous(self):
        """Test that previous transaction is auto-committed."""
        manager = SingleTransactionManager()
        
        txn1 = manager.begin_transaction()
        txn2 = manager.begin_transaction()
        
        assert txn1.state == TransactionState.COMMITTED
        assert txn2.state == TransactionState.ACTIVE
        assert manager.get_active_transaction() == txn2
    
    def test_commit_transaction(self):
        """Test committing a transaction."""
        manager = SingleTransactionManager()
        
        txn = manager.begin_transaction()
        manager.commit_transaction()
        
        assert txn.state == TransactionState.COMMITTED
        assert manager.get_active_transaction() is None
    
    def test_rollback_transaction(self):
        """Test rolling back a transaction."""
        manager = SingleTransactionManager()
        
        txn = manager.begin_transaction()
        manager.rollback_transaction()
        
        assert txn.state == TransactionState.ROLLED_BACK
        assert manager.get_active_transaction() is None
    
    def test_should_commit_always_true(self):
        """Test that single transactions always commit."""
        manager = SingleTransactionManager()
        assert manager.should_commit() is True
    
    def test_context_manager(self):
        """Test using transaction as context manager."""
        manager = SingleTransactionManager()
        
        with manager.transaction() as txn:
            assert txn.state == TransactionState.ACTIVE
        
        assert txn.state == TransactionState.COMMITTED
    
    def test_context_manager_rollback_on_error(self):
        """Test rollback on error in context manager."""
        manager = SingleTransactionManager()
        
        with pytest.raises(ValueError):
            with manager.transaction() as txn:
                raise ValueError("Test error")
        
        assert txn.state == TransactionState.ROLLED_BACK


class TestBatchTransactionManager:
    """Tests for BatchTransactionManager."""
    
    def test_begin_transaction(self):
        """Test beginning a batch transaction."""
        manager = BatchTransactionManager(batch_size=3)
        
        txn = manager.begin_transaction()
        
        assert txn is not None
        assert txn.strategy == TransactionStrategy.BATCH
        assert txn.state == TransactionState.ACTIVE
    
    def test_reuse_active_transaction(self):
        """Test that active transaction is reused."""
        manager = BatchTransactionManager(batch_size=3)
        
        txn1 = manager.begin_transaction()
        txn2 = manager.begin_transaction()
        
        assert txn1 is txn2
    
    def test_add_to_batch(self):
        """Test adding operations to batch."""
        manager = BatchTransactionManager(batch_size=3)
        
        manager.add_to_batch({"op": 1})
        manager.add_to_batch({"op": 2})
        
        txn = manager.get_active_transaction()
        assert len(txn.operations) == 2
        assert txn.state == TransactionState.ACTIVE
    
    def test_auto_commit_on_batch_size(self):
        """Test auto-commit when batch is full."""
        manager = BatchTransactionManager(batch_size=2, auto_commit=True)
        
        manager.add_to_batch({"op": 1})
        txn1 = manager.get_active_transaction()
        
        manager.add_to_batch({"op": 2})  # Should trigger commit
        
        assert txn1.state == TransactionState.COMMITTED
        assert manager.get_active_transaction() is None
    
    def test_no_auto_commit(self):
        """Test disabling auto-commit."""
        manager = BatchTransactionManager(batch_size=2, auto_commit=False)
        
        manager.add_to_batch({"op": 1})
        manager.add_to_batch({"op": 2})
        manager.add_to_batch({"op": 3})
        
        txn = manager.get_active_transaction()
        assert txn.state == TransactionState.ACTIVE
        assert len(txn.operations) == 3
    
    def test_flush(self):
        """Test flushing the batch."""
        manager = BatchTransactionManager(batch_size=10)
        
        manager.add_to_batch({"op": 1})
        txn = manager.get_active_transaction()
        
        manager.flush()
        
        assert txn.state == TransactionState.COMMITTED
        assert manager.get_active_transaction() is None
    
    def test_should_commit(self):
        """Test should_commit logic."""
        # Test with auto_commit disabled to check batch size logic
        manager = BatchTransactionManager(batch_size=2, auto_commit=False)
        
        assert manager.should_commit() is False
        
        manager.add_to_batch({"op": 1})
        assert manager.should_commit() is False
        
        manager.add_to_batch({"op": 2})
        # With auto_commit=False, should_commit will still be False
        assert manager.should_commit() is False
        
        # Test with auto_commit enabled
        manager2 = BatchTransactionManager(batch_size=2, auto_commit=True)
        manager2.add_to_batch({"op": 1})
        assert manager2.should_commit() is False
        
        # After this, auto-commit will trigger and reset the count
        manager2.add_to_batch({"op": 2})
        # After auto-commit, count is 0 again
        assert manager2._operation_count == 0


class TestManualTransactionManager:
    """Tests for ManualTransactionManager."""
    
    def test_begin_transaction(self):
        """Test beginning a manual transaction."""
        manager = ManualTransactionManager()
        
        txn = manager.begin_transaction("my_txn")
        
        assert txn.id == "my_txn"
        assert txn.strategy == TransactionStrategy.MANUAL
        assert txn.state == TransactionState.ACTIVE
    
    def test_cannot_duplicate_transaction_id(self):
        """Test that duplicate transaction IDs are rejected."""
        manager = ManualTransactionManager()
        
        manager.begin_transaction("txn1")
        
        with pytest.raises(ValueError, match="already exists"):
            manager.begin_transaction("txn1")
    
    def test_manual_commit(self):
        """Test manual commit."""
        manager = ManualTransactionManager()
        
        txn = manager.begin_transaction()
        manager.commit_transaction()
        
        assert txn.state == TransactionState.COMMITTED
        assert manager.get_active_transaction() is None
    
    def test_manual_rollback(self):
        """Test manual rollback."""
        manager = ManualTransactionManager()
        
        txn = manager.begin_transaction()
        manager.rollback_transaction()
        
        assert txn.state == TransactionState.ROLLED_BACK
        assert manager.get_active_transaction() is None
    
    def test_commit_requires_active_transaction(self):
        """Test that commit requires an active transaction."""
        manager = ManualTransactionManager()
        
        with pytest.raises(ValueError, match="No active transaction"):
            manager.commit_transaction()
    
    def test_rollback_requires_active_transaction(self):
        """Test that rollback requires an active transaction."""
        manager = ManualTransactionManager()
        
        with pytest.raises(ValueError, match="No active transaction"):
            manager.rollback_transaction()
    
    def test_should_commit_always_false(self):
        """Test that manual transactions never auto-commit."""
        manager = ManualTransactionManager()
        assert manager.should_commit() is False
    
    def test_context_manager_no_auto_commit(self):
        """Test that context manager doesn't auto-commit."""
        manager = ManualTransactionManager()
        
        with manager.transaction() as txn:
            assert txn.state == TransactionState.ACTIVE
        
        # Manual transactions don't auto-commit
        assert txn.state == TransactionState.ACTIVE
        
        # Must manually commit
        manager.commit_transaction()
        assert txn.state == TransactionState.COMMITTED


class TestTransactionFactory:
    """Tests for create_transaction_manager factory."""
    
    def test_create_single_manager(self):
        """Test creating a single transaction manager."""
        manager = create_transaction_manager(TransactionStrategy.SINGLE)
        assert isinstance(manager, SingleTransactionManager)
    
    def test_create_batch_manager(self):
        """Test creating a batch transaction manager."""
        manager = create_transaction_manager(
            TransactionStrategy.BATCH,
            batch_size=50,
            auto_commit=False
        )
        assert isinstance(manager, BatchTransactionManager)
        assert manager.batch_size == 50
        assert manager.auto_commit is False
    
    def test_create_manual_manager(self):
        """Test creating a manual transaction manager."""
        manager = create_transaction_manager(TransactionStrategy.MANUAL)
        assert isinstance(manager, ManualTransactionManager)