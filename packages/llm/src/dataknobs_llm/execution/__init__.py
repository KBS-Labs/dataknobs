# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Execution utilities for parallel and sequential LLM task processing."""

from .parallel import (
    DeterministicTask,
    LLMTask,
    ParallelLLMExecutor,
    TaskResult,
)

__all__ = [
    "DeterministicTask",
    "LLMTask",
    "ParallelLLMExecutor",
    "TaskResult",
]
