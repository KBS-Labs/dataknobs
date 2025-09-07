"""Data and transaction modes for FSM."""

from enum import Enum


class DataMode(Enum):
    """Data processing mode for FSM execution."""
    
    SINGLE = "single"
    """Process a single record at a time."""
    
    BATCH = "batch"
    """Process multiple records in a batch."""
    
    STREAM = "stream"
    """Process continuous stream of data."""


class TransactionMode(Enum):
    """Transaction handling mode for FSM execution."""
    
    NONE = "none"
    """No transaction support."""
    
    PER_RECORD = "per_record"
    """One transaction per record."""
    
    PER_BATCH = "per_batch"
    """One transaction per batch."""
    
    PER_SESSION = "per_session"
    """One transaction for entire session."""
    
    DISTRIBUTED = "distributed"
    """Distributed transaction support."""