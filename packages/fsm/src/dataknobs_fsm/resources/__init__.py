"""Resource management for FSM.

This module provides resource management capabilities for states,
including connection pooling, lifecycle management, and health checks.
"""

from dataknobs_fsm.resources.base import (
    IResourceProvider,
    IResourcePool,
    ResourceStatus,
    ResourceHealth,
    ResourceMetrics,
)
from dataknobs_fsm.resources.manager import ResourceManager
from dataknobs_fsm.resources.pool import ResourcePool, PoolConfig

__all__ = [
    "IResourceProvider",
    "IResourcePool", 
    "ResourceStatus",
    "ResourceHealth",
    "ResourceMetrics",
    "ResourceManager",
    "ResourcePool",
    "PoolConfig",
]
