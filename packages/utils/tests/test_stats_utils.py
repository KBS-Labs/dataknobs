import math
import random
import time
from datetime import datetime

import dataknobs_utils.stats_utils as dk_stats


def test_empty():
    s = dk_stats.StatsAccumulator("empty")
    assert s.as_dict() == {"label": "empty", "max": 0.0, "mean": 0, "min": 0.0, "n": 0, "std": 0}


def test_stats_accumulator_basics1():
    s = dk_stats.StatsAccumulator("basics1")
    s.add(1, 2, 3)

    assert s.label == "basics1"
    assert s.n == 3
    assert s.min == 1
    assert s.max == 3
    assert s.mean == 2.0
    assert s.std == 1.0
    assert s.var == 1.0

    s.clear("empty")
    assert s.as_dict() == {"label": "empty", "max": 0.0, "mean": 0, "min": 0.0, "n": 0, "std": 0}


def test_stats_accumulator_basics2():
    s = dk_stats.StatsAccumulator("basics2", values=[215.0, 215.0, 215.0])

    assert s.label == "basics2"
    assert s.n == 3
    assert s.min == 215.0
    assert s.max == 215.0
    assert s.mean == 215.0
    assert s.std == 0.0
    assert s.var == 0.0


def test_combine():
    s1 = dk_stats.StatsAccumulator("1")
    s1.add(1, 2, 3)
    s2 = dk_stats.StatsAccumulator("2")
    s2.add(4, 5, 6)
    s3 = dk_stats.StatsAccumulator("3")
    s3.add(1, 2, 3, 4, 5, 6)

    s = dk_stats.StatsAccumulator.combine("3", s1, s2)
    assert s3.as_dict() == s.as_dict()


def test_copy_constructor():
    s1 = dk_stats.StatsAccumulator("1")
    s1.add(1, 2, 3)
    s_copy = dk_stats.StatsAccumulator("1", s1)
    assert s1.as_dict() == s_copy.as_dict()


def test_initialize_from_dict1():
    s1 = dk_stats.StatsAccumulator("1")
    s1.add(1, 2, 3)
    as_dict = s1.as_dict()

    s_other = dk_stats.StatsAccumulator()
    s_other.initialize(**as_dict)

    assert s_other.label == "1"
    assert s_other.n == 3
    assert s_other.min == 1
    assert s_other.max == 3
    assert s_other.mean == 2.0
    assert s_other.std == 1.0
    assert s_other.var == 1.0

    s_other = dk_stats.StatsAccumulator(as_dict=as_dict)
    assert s_other.label == "1"
    assert s_other.n == 3
    assert s_other.min == 1
    assert s_other.max == 3
    assert s_other.mean == 2.0
    assert s_other.std == 1.0
    assert s_other.var == 1.0

    as_dict = s1.as_dict(with_sums=True)
    assert as_dict["sum"] == 6
    assert as_dict["sos"] == 14
    s_other = dk_stats.StatsAccumulator(as_dict=as_dict)
    assert s_other.label == "1"
    assert s_other.n == 3
    assert s_other.min == 1
    assert s_other.max == 3
    assert s_other.mean == 2.0
    assert s_other.std == 1.0
    assert s_other.var == 1.0


def test_floating_point():
    s = dk_stats.StatsAccumulator("basics")
    s.add(1.0, 2.0, 3.0)

    assert s.label == "basics"
    assert s.n == 3.0
    assert s.min == 1.0
    assert s.max == 3.0
    assert s.mean == 2.0
    assert s.std == 1.0
    assert s.var == 1.0


def test_simple1():
    rolling_stats = dk_stats.RollingStats(100, 50)

    stats1 = dk_stats.StatsAccumulator()
    seg2value = -1

    # add stats during segment 0
    while rolling_stats.last_segment == 0:
        value = random.uniform(1, 1000)
        seg_num = rolling_stats.add(value)
        stats1.add(value)
        if seg_num != 0:
            seg2value = value
            break

    window0_stats = rolling_stats.window_stats

    # don't add anything during segment1
    while rolling_stats.cur_segment != 0:
        time.sleep(0.001)

    # get window stats after rolling beyond segment1. (should match seg2value)
    window1_stats = rolling_stats.window_stats

    # don't add anything during segment0
    while rolling_stats.cur_segment == 0:
        time.sleep(0.001)

    # window stats should now be empty
    emptyStats = rolling_stats.window_stats

    # do checks
    assert rolling_stats.num_segments == 2

    # window0_stats should match stats1
    assert stats1.n == window0_stats.n
    assert math.isclose(stats1.mean, window0_stats.mean, rel_tol=0.005)

    # window1_stats should match seg2value
    if seg2value < 0:
        # window1_stats should be empty
        assert window1_stats.n == 0
    else:
        # window1_stats should have 1 value: seg2value
        assert window1_stats.n == 1
        assert math.isclose(seg2value, window1_stats.mean, rel_tol=0.005)

    # emptyStats should be empty
    assert emptyStats.n == 0

    # cumulativeStats should match stats1
    cumulative_stats = rolling_stats.cumulative_stats
    assert stats1.n == cumulative_stats.n
    assert math.isclose(stats1.mean, cumulative_stats.mean, rel_tol=0.005)

    # exercise building summary info dictionary, value doesn't matter
    as_dict = rolling_stats.as_dict()


def test_monitor_basics_with_processing():
    monitor = dk_stats.Monitor("testBasics", default_window_width=100, default_segment_width=50)
    for i in range(5):
        starttime = datetime.now()
        waittime = dk_stats.wait_for_random_millis(55)
        endtime = datetime.now()
        monitor.mark(starttime, endtime)

    # Default stats are processing_cumulative_stats
    assert monitor.processing_cumulative_stats == monitor.get_stats()

    # Make sure get_stats behaves as advertised
    assert monitor.processing_cumulative_stats == monitor.get_stats(access=False, window=False)
    assert monitor.processing_cumulative_stats != monitor.get_stats(access=False, window=True)
    assert monitor.access_cumulative_stats == monitor.get_stats(access=True, window=False)
    assert monitor.access_cumulative_stats != monitor.get_stats(access=True, window=True)

    cumulative_processing_stats = monitor.get_stats()
    assert cumulative_processing_stats.n == 5

    # exercise building summary info dictionary, value doesn't matter
    time.sleep(0.051)  # ensure window stats is different from cumulative
    as_dict = monitor.as_dict()


def test_monitor_basics_with_access_only():
    monitor = dk_stats.Monitor("test_access")
    for i in range(5):
        starttime = datetime.now()
        waittime = dk_stats.wait_for_random_millis(55)
        monitor.mark(starttime)

    assert monitor.processing_cumulative_stats is None
    assert monitor.processing_window_stats is None
    assert monitor.access_cumulative_stats is not None
    assert monitor.access_window_stats is not None

    cumulative_access_stats = monitor.get_stats(True, False)
    assert cumulative_access_stats.n == 4

    # exercise building summary info dictionary, value doesn't matter
    time.sleep(0.051)  # ensure window stats is different from cumulative
    as_dict = monitor.as_dict()


def simulate_monitor_process(monitor_manager, monitor_label):
    monitor = monitor_manager.get_monitor(monitor_label, create_if_missing=True, description=None)
    starttime = datetime.now()
    waittime = dk_stats.wait_for_random_millis(55)
    endtime = datetime.now()
    monitor.mark(starttime, endtime)


def simulate_monitor_event(monitor_manager, event_type, event_description):
    monitor = monitor_manager.get_or_create_monitor_by_key_type(event_type, event_description)
    starttime = datetime.now()
    monitor.mark(starttime)


def test_monitor_manager_basics():
    monitor_manager = dk_stats.MonitorManager(default_window_width=100, default_segment_width=50)

    simulate_monitor_event(monitor_manager, "error", "Intermittent error 1")
    simulate_monitor_event(monitor_manager, "error", "Recurring error")
    simulate_monitor_event(monitor_manager, "error", "Intermittent error 2")
    simulate_monitor_event(monitor_manager, "error", "Recurring error")
    simulate_monitor_process(monitor_manager, "simulated_process_1")
    simulate_monitor_event(monitor_manager, "error", "Recurring error")
    simulate_monitor_event(monitor_manager, "error", "Intermittent error 3")
    simulate_monitor_event(monitor_manager, "error", "Recurring error")

    simulate_monitor_process(monitor_manager, "simulated_process_2")

    simulate_monitor_event(monitor_manager, "error", "Recurring error")
    simulate_monitor_process(monitor_manager, "simulated_process_1")
    simulate_monitor_event(monitor_manager, "error", "Recurring error")
    simulate_monitor_process(monitor_manager, "simulated_process_1")
    simulate_monitor_event(monitor_manager, "error", "Recurring error")

    # exercise building summary info dictionary, value doesn't matter
    time.sleep(0.051)  # ensure window stats is different from cumulative
    as_dict = monitor_manager.as_dict()


def test_key_manager():
    key_manager = dk_stats.KeyManager()
    info0 = key_manager.get_key("info", "First info description")
    warn0 = key_manager.get_key("warn", "First warn description")
    error0 = key_manager.get_key("error", "First error description")

    warn1 = key_manager.get_key("warn", "Second warn description")
    error1 = key_manager.get_key("error", "Second error description")

    error2 = key_manager.get_key("error", "Third error description")

    assert info0 == "info-0"
    assert key_manager.get_key("info", "First info description") == "info-0"

    assert warn0 == "warn-0"
    assert key_manager.get_key("warn", "First warn description") == "warn-0"
    assert warn1 == "warn-1"
    assert key_manager.get_key("warn", "Second warn description") == "warn-1"

    assert error0 == "error-0"
    assert key_manager.get_key("error", "First error description") == "error-0"
    assert error1 == "error-1"
    assert key_manager.get_key("error", "Second error description") == "error-1"
    assert error2 == "error-2"
    assert key_manager.get_key("error", "Third error description") == "error-2"
