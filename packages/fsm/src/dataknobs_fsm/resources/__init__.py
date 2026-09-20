# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Resource management for FSM.

This module provides resource management capabilities for states,
including connection pooling, lifecycle management, and health checks.
"""

from dataknobs_fsm.resources.base import (
    AsyncCleanable,
    AsyncClosable,
    IResourceProvider,
    IResourcePool,
    ResourceStatus,
    ResourceHealth,
    ResourceMetrics,
)
from dataknobs_fsm.resources.manager import ResourceManager
from dataknobs_fsm.resources.pool import ResourcePool, PoolConfig

__all__ = [
    "AsyncCleanable",
    "AsyncClosable",
    "IResourceProvider",
    "IResourcePool",
    "ResourceStatus",
    "ResourceHealth",
    "ResourceMetrics",
    "ResourceManager",
    "ResourcePool",
    "PoolConfig",
]
