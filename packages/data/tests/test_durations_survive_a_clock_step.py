# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""How long something took does not depend on what the wall clock did meanwhile.

The system clock is adjusted while a process runs, and a duration taken as the
difference of two readings of it comes out short, long or negative.
"""

from __future__ import annotations

import time
from itertools import count
from typing import TYPE_CHECKING

import pytest

from dataknobs_data import Record
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.migration.progress import MigrationProgress

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture
def clock_stepping_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wall clock an hour earlier at every reading."""
    readings = count()
    start = time.time()
    monkeypatch.setattr(time, "time", lambda: start - 3600 * next(readings))


@pytest.mark.usefixtures("clock_stepping_back")
class TestMigrationProgress:
    def test_while_it_runs(self) -> None:
        assert 0 <= MigrationProgress().start().duration < 60

    def test_once_finished(self) -> None:
        progress = MigrationProgress().start().finish()
        assert 0 <= progress.duration < 60

    def test_started_again(self) -> None:
        progress = MigrationProgress().start().finish().start()
        assert progress.end_time is None
        assert 0 <= progress.duration < 60
        assert 0 <= progress.finish().duration < 60


@pytest.mark.usefixtures("clock_stepping_back")
class TestAStreamedWrite:
    def test_sync(self) -> None:
        assert 0 <= SyncMemoryDatabase().stream_write(iter([Record({"n": 1})])).duration < 60

    async def test_async(self) -> None:
        async def records() -> AsyncIterator[Record]:
            yield Record({"n": 1})

        assert 0 <= (await AsyncMemoryDatabase().stream_write(records())).duration < 60


class TestMigrationProgressKeepsItsTimes:
    def test_the_times_are_the_wall_clocks(self) -> None:
        before = time.time()
        progress = MigrationProgress().start().finish()
        assert progress.start_time is not None
        assert progress.end_time is not None
        assert before <= progress.start_time <= progress.end_time <= time.time()

    def test_times_given_are_the_ones_compared(self) -> None:
        assert MigrationProgress(start_time=100.0, end_time=130.0).duration == 30.0

    def test_times_set_after_a_run_are_the_ones_compared(self) -> None:
        progress = MigrationProgress().start().finish()
        progress.start_time, progress.end_time = 100.0, 130.0
        assert progress.duration == 30.0

    def test_not_started(self) -> None:
        assert MigrationProgress().duration == 0.0
