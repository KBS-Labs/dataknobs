"""Transaction management for FSM data processing.

This module provides different transaction strategies:
- SINGLE: Each state transition is a transaction
- BATCH: Multiple transitions batched into one transaction
- MANUAL: Explicit transaction control
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Dict, List, TypeVar
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TransactionStrategy(Enum):
    """Transaction management strategies."""
    
    SINGLE = "single"  # One transaction per state transition
    BATCH = "batch"  # Batch multiple transitions
    MANUAL = "manual"  # Explicit transaction control


class TransactionState(Enum):
    """State of a transaction."""
    
    PENDING = "pending"
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class Transaction:
    """Represents a single transaction."""
    
    def __init__(self, transaction_id: str, strategy: TransactionStrategy):
        """Initialize a transaction.
        
        Args:
            transaction_id: Unique identifier for the transaction.
            strategy: The transaction strategy being used.
        """
        self.id = transaction_id
        self.strategy = strategy
        self.state = TransactionState.PENDING
        self.operations: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self._rollback_handlers: List[Callable[[], None]] = []
    
    def add_operation(self, operation: Dict[str, Any]) -> None:
        """Add an operation to the transaction.
        
        Args:
            operation: Operation details to add.
        """
        if self.state != TransactionState.ACTIVE:
            raise RuntimeError(f"Cannot add operation to transaction in state {self.state}")
        self.operations.append(operation)
    
    def add_rollback_handler(self, handler: Callable[[], None]) -> None:
        """Add a rollback handler.
        
        Args:
            handler: Function to call on rollback.
        """
        self._rollback_handlers.append(handler)
    
    def rollback(self) -> None:
        """Execute rollback handlers in reverse order."""
        for handler in reversed(self._rollback_handlers):
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in rollback handler: {e}")
        self.state = TransactionState.ROLLED_BACK
    
    def commit(self) -> None:
        """Mark transaction as committed."""
        self.state = TransactionState.COMMITTED
    
    def __repr__(self) -> str:
        """String representation of the transaction."""
        return f"Transaction(id={self.id}, strategy={self.strategy}, state={self.state})"


class TransactionManager(ABC):
    """Abstract base class for transaction managers."""
    
    def __init__(self, strategy: TransactionStrategy):
        """Initialize the transaction manager.
        
        Args:
            strategy: The transaction strategy to use.
        """
        self.strategy = strategy
        self._transactions: Dict[str, Transaction] = {}
        self._active_transaction: Transaction | None = None
    
    @classmethod
    def create(cls, strategy: TransactionStrategy, **config) -> "TransactionManager":  # noqa: ARG003
        """Factory method to create appropriate transaction manager.
        
        Args:
            strategy: Transaction strategy to use
            **config: Strategy-specific configuration (currently unused)
            
        Returns:
            Appropriate TransactionManager subclass instance
            
        Raises:
            ValueError: If strategy is unknown
        """
        # Import here to avoid circular dependencies
        if strategy == TransactionStrategy.SINGLE:
            return SingleTransactionManager()
        elif strategy == TransactionStrategy.BATCH:
            return BatchTransactionManager()
        elif strategy == TransactionStrategy.MANUAL:
            return ManualTransactionManager()
        else:
            raise ValueError(f"Unknown transaction strategy: {strategy}")
    
    @abstractmethod
    def begin_transaction(self, transaction_id: str | None = None) -> Transaction:
        """Begin a new transaction.
        
        Args:
            transaction_id: Optional ID for the transaction.
            
        Returns:
            The created transaction.
        """
        pass
    
    @abstractmethod
    def commit_transaction(self, transaction_id: str | None = None) -> None:
        """Commit a transaction.
        
        Args:
            transaction_id: ID of transaction to commit, or None for active.
        """
        pass
    
    @abstractmethod
    def rollback_transaction(self, transaction_id: str | None = None) -> None:
        """Rollback a transaction.
        
        Args:
            transaction_id: ID of transaction to rollback, or None for active.
        """
        pass
    
    @abstractmethod
    def should_commit(self) -> bool:
        """Determine if a commit should happen now.
        
        Returns:
            True if transaction should be committed.
        """
        pass
    
    @contextmanager
    def transaction(self, transaction_id: str | None = None):
        """Context manager for transactions.
        
        Args:
            transaction_id: Optional ID for the transaction.
            
        Yields:
            The active transaction.
        """
        txn = self.begin_transaction(transaction_id)
        try:
            yield txn
            if self.should_commit():
                self.commit_transaction(txn.id)
        except Exception as e:
            self.rollback_transaction(txn.id)
            raise e
    
    def get_transaction(self, transaction_id: str) -> Transaction | None:
        """Get a transaction by ID.
        
        Args:
            transaction_id: ID of the transaction.
            
        Returns:
            The transaction if found, None otherwise.
        """
        return self._transactions.get(transaction_id)
    
    def get_active_transaction(self) -> Transaction | None:
        """Get the currently active transaction.
        
        Returns:
            The active transaction if any.
        """
        return self._active_transaction


class SingleTransactionManager(TransactionManager):
    """Manager for SINGLE transaction strategy - one per transition."""
    
    def __init__(self):
        """Initialize the single transaction manager."""
        super().__init__(TransactionStrategy.SINGLE)
        self._transaction_counter = 0
    
    def begin_transaction(self, transaction_id: str | None = None) -> Transaction:
        """Begin a new single transaction.
        
        Args:
            transaction_id: Optional ID for the transaction.
            
        Returns:
            The created transaction.
        """
        if self._active_transaction is not None:
            self.commit_transaction(self._active_transaction.id)
        
        if transaction_id is None:
            self._transaction_counter += 1
            transaction_id = f"single_txn_{self._transaction_counter}"
        
        txn = Transaction(transaction_id, self.strategy)
        txn.state = TransactionState.ACTIVE
        self._transactions[transaction_id] = txn
        self._active_transaction = txn
        return txn
    
    def commit_transaction(self, transaction_id: str | None = None) -> None:
        """Commit a single transaction.
        
        Args:
            transaction_id: ID of transaction to commit.
        """
        if transaction_id is None:
            if self._active_transaction is None:
                return
            transaction_id = self._active_transaction.id
        
        txn = self._transactions.get(transaction_id)
        if txn:
            txn.commit()
            if self._active_transaction == txn:
                self._active_transaction = None
    
    def rollback_transaction(self, transaction_id: str | None = None) -> None:
        """Rollback a single transaction.
        
        Args:
            transaction_id: ID of transaction to rollback.
        """
        if transaction_id is None:
            if self._active_transaction is None:
                return
            transaction_id = self._active_transaction.id
        
        txn = self._transactions.get(transaction_id)
        if txn:
            txn.rollback()
            if self._active_transaction == txn:
                self._active_transaction = None
    
    def should_commit(self) -> bool:
        """Single transactions always commit immediately.
        
        Returns:
            True, always commit after each operation.
        """
        return True


class BatchTransactionManager(TransactionManager):
    """Manager for BATCH transaction strategy."""
    
    def __init__(self, batch_size: int = 100, auto_commit: bool = True):
        """Initialize the batch transaction manager.
        
        Args:
            batch_size: Number of operations before auto-commit.
            auto_commit: Whether to auto-commit when batch is full.
        """
        super().__init__(TransactionStrategy.BATCH)
        self.batch_size = batch_size
        self.auto_commit = auto_commit
        self._batch_counter = 0
        self._operation_count = 0
    
    def begin_transaction(self, transaction_id: str | None = None) -> Transaction:
        """Begin or get the current batch transaction.
        
        Args:
            transaction_id: Optional ID for the transaction.
            
        Returns:
            The active batch transaction.
        """
        if self._active_transaction is None:
            if transaction_id is None:
                self._batch_counter += 1
                transaction_id = f"batch_txn_{self._batch_counter}"
            
            txn = Transaction(transaction_id, self.strategy)
            txn.state = TransactionState.ACTIVE
            self._transactions[transaction_id] = txn
            self._active_transaction = txn
            self._operation_count = 0
        
        return self._active_transaction
    
    def add_to_batch(self, operation: Dict[str, Any]) -> None:
        """Add an operation to the current batch.
        
        Args:
            operation: Operation to add to the batch.
        """
        if self._active_transaction is None:
            self.begin_transaction()
        
        self._active_transaction.add_operation(operation)
        self._operation_count += 1
        
        if self.auto_commit and self._operation_count >= self.batch_size:
            self.commit_transaction()
    
    def commit_transaction(self, transaction_id: str | None = None) -> None:
        """Commit the batch transaction.
        
        Args:
            transaction_id: ID of transaction to commit.
        """
        if transaction_id is None:
            if self._active_transaction is None:
                return
            transaction_id = self._active_transaction.id
        
        txn = self._transactions.get(transaction_id)
        if txn:
            txn.commit()
            if self._active_transaction == txn:
                self._active_transaction = None
                self._operation_count = 0
    
    def rollback_transaction(self, transaction_id: str | None = None) -> None:
        """Rollback the batch transaction.
        
        Args:
            transaction_id: ID of transaction to rollback.
        """
        if transaction_id is None:
            if self._active_transaction is None:
                return
            transaction_id = self._active_transaction.id
        
        txn = self._transactions.get(transaction_id)
        if txn:
            txn.rollback()
            if self._active_transaction == txn:
                self._active_transaction = None
                self._operation_count = 0
    
    def should_commit(self) -> bool:
        """Check if batch should be committed.
        
        Returns:
            True if batch is full or explicitly requested.
        """
        return self.auto_commit and self._operation_count >= self.batch_size
    
    def flush(self) -> None:
        """Force commit of the current batch."""
        if self._active_transaction is not None:
            self.commit_transaction()


class ManualTransactionManager(TransactionManager):
    """Manager for MANUAL transaction strategy - explicit control."""
    
    def __init__(self):
        """Initialize the manual transaction manager."""
        super().__init__(TransactionStrategy.MANUAL)
        self._transaction_counter = 0
    
    def begin_transaction(self, transaction_id: str | None = None) -> Transaction:
        """Begin a new manual transaction.
        
        Args:
            transaction_id: Optional ID for the transaction.
            
        Returns:
            The created transaction.
        """
        if transaction_id is None:
            self._transaction_counter += 1
            transaction_id = f"manual_txn_{self._transaction_counter}"
        
        if transaction_id in self._transactions:
            raise ValueError(f"Transaction {transaction_id} already exists")
        
        txn = Transaction(transaction_id, self.strategy)
        txn.state = TransactionState.ACTIVE
        self._transactions[transaction_id] = txn
        self._active_transaction = txn
        return txn
    
    def commit_transaction(self, transaction_id: str | None = None) -> None:
        """Manually commit a transaction.
        
        Args:
            transaction_id: ID of transaction to commit.
        """
        if transaction_id is None:
            if self._active_transaction is None:
                raise ValueError("No active transaction to commit")
            transaction_id = self._active_transaction.id
        
        txn = self._transactions.get(transaction_id)
        if not txn:
            raise ValueError(f"Transaction {transaction_id} not found")
        
        if txn.state != TransactionState.ACTIVE:
            raise ValueError(f"Transaction {transaction_id} is not active")
        
        txn.commit()
        if self._active_transaction == txn:
            self._active_transaction = None
    
    def rollback_transaction(self, transaction_id: str | None = None) -> None:
        """Manually rollback a transaction.
        
        Args:
            transaction_id: ID of transaction to rollback.
        """
        if transaction_id is None:
            if self._active_transaction is None:
                raise ValueError("No active transaction to rollback")
            transaction_id = self._active_transaction.id
        
        txn = self._transactions.get(transaction_id)
        if not txn:
            raise ValueError(f"Transaction {transaction_id} not found")
        
        if txn.state != TransactionState.ACTIVE:
            raise ValueError(f"Transaction {transaction_id} is not active")
        
        txn.rollback()
        if self._active_transaction == txn:
            self._active_transaction = None
    
    def should_commit(self) -> bool:
        """Manual transactions never auto-commit.
        
        Returns:
            False, manual control required.
        """
        return False


def create_transaction_manager(
    strategy: TransactionStrategy,
    **kwargs
) -> TransactionManager:
    """Factory function to create transaction managers.
    
    Args:
        strategy: The transaction strategy to use.
        **kwargs: Additional arguments for specific managers.
        
    Returns:
        The appropriate transaction manager.
    """
    if strategy == TransactionStrategy.SINGLE:
        return SingleTransactionManager()
    elif strategy == TransactionStrategy.BATCH:
        batch_size = kwargs.get("batch_size", 100)
        auto_commit = kwargs.get("auto_commit", True)
        return BatchTransactionManager(batch_size, auto_commit)
    elif strategy == TransactionStrategy.MANUAL:
        return ManualTransactionManager()
    else:
        raise ValueError(f"Unknown transaction strategy: {strategy}")
