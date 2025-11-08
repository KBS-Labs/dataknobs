"""Statistical utility functions and timing helpers.

Provides utilities for timing operations, random waits, rate limiting,
and basic statistical calculations.
"""

import json
import math
import random
import time
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union


def wait_for_random_millis(uptomillis: float) -> float:
    """Wait for a random number of milliseconds.

    Args:
        uptomillis: Maximum number of milliseconds to wait.

    Returns:
        float: The actual wait time in milliseconds.
    """
    waittime = random.uniform(1, uptomillis)
    time.sleep(waittime / 1000)
    return waittime


class StatsAccumulator:
    """A low-memory helper class to collect statistical samples and provide summary statistics.

    Accumulates statistical values in a thread-safe manner, computing mean, variance,
    standard deviation, min, max, and other summary statistics incrementally without
    storing individual values. Supports combining multiple accumulators and
    initialization from dictionaries.

    Attributes:
        label: A label or name for this instance.
        n: The number of values added.
        min: The minimum value added.
        max: The maximum value added.
        sum: The sum of all values added.
        sum_of_squares: The sum of all squared values.
        mean: The mean of the values.
        std: The standard deviation of the values.
        var: The variance of the values.
    """

    def __init__(
        self,
        label: str = "",
        other: Optional["StatsAccumulator"] = None,
        as_dict: Dict[str, Any] | None = None,
        values: Union[float, List[float]] | None = None,
    ) -> None:
        """Initialize a StatsAccumulator.

        Args:
            label: A label or name for this instance. Defaults to "".
            other: Another StatsAccumulator instance to copy data from.
                Defaults to None.
            as_dict: Dictionary representation of stats to initialize with.
                Defaults to None.
            values: A list of initial values or a single value to add.
                Defaults to None.
        """
        self._label: str = label
        self._n = 0
        self._min = 0.0
        self._max = 0.0
        self._sum = 0.0
        self._sos = 0.0
        self._modlock = Lock()

        if other is not None:
            if label != "":
                self._label = other._label
            self._n = other._n
            self._min = other._min
            self._max = other._max
            self._sum = other._sum
            self._sos = other._sos
        if as_dict is not None:
            self.initialize(label=label, as_dict=as_dict)

        if values is not None:
            if isinstance(values, list):
                self.add(*values)
            else:
                self.add(values)

    @property
    def label(self) -> str:
        """Get the label or name"""
        return self._label

    @label.setter
    def label(self, val: str) -> None:
        """Set the label or name"""
        self._label = val

    @property
    def n(self) -> int:
        """Get the number of values added"""
        return self._n

    @property
    def min(self) -> float:
        """Get the minimum value added"""
        return self._min

    @property
    def max(self) -> float:
        """Get the maximum value added"""
        return self._max

    @property
    def sum(self) -> float:
        """Get the sum of all values added"""
        return self._sum

    @property
    def sum_of_squares(self) -> float:
        """Get the sum of all squared values"""
        return self._sos

    @property
    def mean(self) -> float:
        """Get the mean of the values"""
        return 0 if self._n == 0 else self._sum / self._n

    @property
    def std(self) -> float:
        """Get the standard deviation of the values"""
        return 0 if self._n < 2 else math.sqrt(self.var)

    @property
    def var(self) -> float:
        """Get the variance of the values"""
        var = 0.0
        if self._n > 1:
            var = abs(
                (1.0 / (self._n - 1.0)) * (self._sos - (1.0 / self._n) * self._sum * self._sum)
            )
        return var

    def clear(self, label: str = "") -> None:
        """Clear all values and reset statistics.

        Args:
            label: Optional new label to assign. Defaults to "".
        """
        self._modlock.acquire()
        try:
            self._label = label
            self._n = 0
            self._min = 0.0
            self._max = 0.0
            self._sum = 0.0
            self._sos = 0.0
        finally:
            self._modlock.release()

    def initialize(
        self,
        label: str = "",
        n: int = 0,
        min: float = 0,
        max: float = 0,
        mean: float = 0,
        std: float = 0,
        as_dict: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize with the given values.

        When as_dict is provided, its values override the individual parameters.
        Computes internal sum and sum_of_squares from mean and std if not provided
        in the dictionary.

        Args:
            label: A label or name for this instance. Defaults to "".
            n: Number of values. Defaults to 0.
            min: Minimum value. Defaults to 0.
            max: Maximum value. Defaults to 0.
            mean: Mean value. Defaults to 0.
            std: Standard deviation. Defaults to 0.
            as_dict: Dictionary with keys: label, n, min, max, mean, std, sum, sos.
                Values from this dict override individual parameters. Defaults to None.
        """
        if as_dict is not None:
            if "label" in as_dict:
                label = as_dict["label"]
            if "n" in as_dict:
                n = as_dict["n"]
            if "min" in as_dict:
                min = as_dict["min"]
            if "max" in as_dict:
                max = as_dict["max"]
            if "mean" in as_dict:
                mean = as_dict["mean"]
            if "std" in as_dict:
                std = as_dict["std"]

        self._modlock.acquire()
        try:
            self._label = label
            self._n = n
            self._min = min
            self._max = max
            if as_dict is not None and "sum" in as_dict:
                self._sum = as_dict["sum"]
            else:
                self._sum = mean * n
            if as_dict is not None and "sos" in as_dict:
                self._sos = as_dict["sos"]
            else:
                self._sos = 0 if n == 0 else std * std * (n - 1.0) + self._sum * self._sum / n
        finally:
            self._modlock.release()

    def as_dict(self, with_sums: bool = False) -> Dict[str, Any]:
        """Get a dictionary containing a summary of this instance's information.

        Args:
            with_sums: If True, include sum and sos (sum of squares) in the output.
                Defaults to False.

        Returns:
            Dict[str, Any]: Dictionary with keys: label, n, min, max, mean, std,
                and optionally sum and sos.
        """
        d = {
            "label": self.label,
            "n": self.n,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
        }
        if with_sums:
            d.update({"sum": self._sum, "sos": self._sos})
        return d

    def __str__(self) -> str:
        """Get the statistics as a JSON string.

        Returns:
            str: JSON representation of the statistics dictionary.
        """
        return json.dumps(self.as_dict(), sort_keys=True)

    def add(self, *values: float) -> "StatsAccumulator":
        """Add one or more values to the accumulator (thread-safe).

        Args:
            *values: One or more numeric values to add.

        Returns:
            StatsAccumulator: This instance for method chaining.
        """
        self._modlock.acquire()
        try:
            for value in values:
                self._do_add(value)
        finally:
            self._modlock.release()
        return self

    def _do_add(self, value: float) -> None:
        """Internal method to add a single value.

        Args:
            value: The numeric value to add.
        """
        if self._n == 0:
            self._min = value
            self._max = value
        else:
            self._min = min(self._min, value)
            self._max = max(self._max, value)

        self._n += 1
        self._sos += value * value
        self._sum += value

    @staticmethod
    def combine(label: str, *stats_accumulators: "StatsAccumulator") -> "StatsAccumulator":
        """Create a new StatsAccumulator combining data from multiple accumulators.

        Args:
            label: Label for the new combined accumulator.
            *stats_accumulators: One or more StatsAccumulator instances to combine.

        Returns:
            StatsAccumulator: New accumulator as if it had accumulated all data
                from the provided accumulators.
        """
        result = StatsAccumulator(label)
        for stats in stats_accumulators:
            result.incorporate(stats)
        return result

    def incorporate(self, other: Optional["StatsAccumulator"]) -> None:
        """Incorporate another accumulator's data into this one.

        Merges the statistics as if this accumulator had accumulated all values
        from both accumulators.

        Args:
            other: Another StatsAccumulator to incorporate. If None, no action taken.
        """
        if other is None:
            return

        self._modlock.acquire()
        try:
            if self._n == 0:
                self._min = other.min
                self._max = other.max
            else:
                self._min = min(self._min, other.min)
                self._max = max(self._max, other.max)

            self._n += other.n
            self._sos += other.sum_of_squares
            self._sum += other.sum
        finally:
            self._modlock.release()


class LinearRegression:
    """A low-memory helper class to collect (x,y) samples and compute linear regression.

    Incrementally computes the line of best fit (y = mx + b) from (x,y) samples
    without storing individual points. Uses the least squares method to calculate
    the slope (m) and intercept (b).

    Attributes:
        n: Number of (x,y) samples added.
        x_sum: Sum of all x values.
        y_sum: Sum of all y values.
        xy_sum: Sum of all x*y products.
        xx_sum: Sum of all x*x products.
        yy_sum: Sum of all y*y products.
        m: Slope of the regression line (computed lazily).
        b: Y-intercept of the regression line (computed lazily).
    """

    def __init__(self) -> None:
        """Initialize a LinearRegression accumulator."""
        self._n = 0
        self._x_sum = 0.0
        self._y_sum = 0.0
        self._xy_sum = 0.0
        self._xx_sum = 0.0
        self._yy_sum = 0.0

        self._m: float | None = None
        self._b: float | None = None

    @property
    def n(self) -> int:
        """Get the number of (x,y) samples added."""
        return self._n

    @property
    def x_sum(self) -> float:
        """Get the sum of all x values."""
        return self._x_sum

    @property
    def y_sum(self) -> float:
        """Get the sum of all y values."""
        return self._y_sum

    @property
    def xy_sum(self) -> float:
        """Get the sum of all x*y products."""
        return self._xy_sum

    @property
    def xx_sum(self) -> float:
        """Get the sum of all x*x products."""
        return self._xx_sum

    @property
    def yy_sum(self) -> float:
        """Get the sum of all y*y products."""
        return self._yy_sum

    @property
    def m(self) -> float:
        """Get the slope of the regression line."""
        if self._m is None:
            self._compute_regression()
        return self._m if self._m is not None else 0.0

    @property
    def b(self) -> float:
        """Get the y-intercept of the regression line."""
        if self._b is None:
            self._compute_regression()
        return self._b if self._b is not None else 0.0

    def get_y(self, x: float) -> float:
        """Compute the predicted y value for a given x.

        Args:
            x: The x value.

        Returns:
            float: The predicted y value (y = mx + b).
        """
        return self.m * x + self.b

    def add(self, x: float, y: float) -> None:
        """Add an (x,y) sample point.

        Args:
            x: The x value.
            y: The y value.
        """
        self._m = None
        self._b = None
        self._n += 1
        self._x_sum += x
        self._y_sum += y
        self._xy_sum += x * y
        self._xx_sum += x * x
        self._yy_sum += y * y

    def __str__(self) -> str:
        """Get the regression equation as a string.

        Returns:
            str: The equation in the form "y = m x + b".
        """
        return f"y = {self.m:.4f} x + {self.b:.4f}"

    def _compute_regression(self) -> None:
        """Internal method to compute the slope and intercept."""
        self._m = 0.0
        self._b = 0.0

        denominator = self._n * self._xx_sum - self._x_sum * self._x_sum
        if denominator != 0:
            m_numerator = self._n * self._xy_sum - self._x_sum * self._y_sum
            b_numerator = self._y_sum * self._xx_sum - self._x_sum * self._xy_sum

            self._m = m_numerator / denominator
            self._b = b_numerator / denominator


class RollingStats:
    """A collection of statistics through a rolling window of time.

    Maintains both cumulative statistics (since initialization) and rolling window
    statistics (over a recent time period). The rolling window is divided into
    segments that are automatically cleared as time advances, providing an efficient
    way to track recent activity without storing all historical data.

    Attributes:
        window_width: Width of the rolling window in milliseconds.
        num_segments: Number of segments dividing the window.
        cur_segment: Current segment index.
        last_segment: Last segment index (same as cur_segment).
        start_time: Time when this RollingStats was created or reset.
        ref_time: Most recent reference time (updates on each access).
        cumulative_stats: Statistics accumulated since start_time.
        window_stats: Statistics for the current rolling window.
        current_label: Label for the current window.
    """

    def __init__(self, window_width: int = 300000, segment_width: int = 5000) -> None:
        """Initialize a RollingStats instance.

        Args:
            window_width: Width of the rolling window in milliseconds.
                Defaults to 300000 (5 minutes).
            segment_width: Width of each segment in milliseconds.
                Defaults to 5000 (5 seconds).
        """
        self._modlock = Lock()
        self._window_width = window_width
        self._segment_width = segment_width
        self._window_delta = timedelta(milliseconds=window_width)
        self._cumulative_stats = StatsAccumulator("cumulative")

        self._num_segments = round(window_width / segment_width)
        self._segment_stats = [
            StatsAccumulator("segment-" + str(i)) for i in range(self._num_segments)
        ]
        self._starttime = datetime.now()
        self._reftime = self._starttime
        self._cur_segment = 0

    @property
    def window_width(self) -> int:
        """Get the width of the rolling window in milliseconds."""
        return self._window_width

    @property
    def num_segments(self) -> int:
        """Get the number of segments in the rolling window."""
        return self._num_segments

    @property
    def cur_segment(self) -> int:
        """Get the current segment index (updates to current time)."""
        result = -1
        self._modlock.acquire()
        try:
            self._inc_to_cur_segment()
            result = self._cur_segment
        finally:
            self._modlock.release()
        return result

    @property
    def last_segment(self) -> int:
        """Get the last segment index (same as cur_segment without update)."""
        return self._cur_segment

    @property
    def start_time(self) -> datetime:
        """Get the time when this RollingStats was created or last reset."""
        return self._starttime

    @property
    def ref_time(self) -> datetime:
        """Get the most recent reference time."""
        return self._reftime

    @property
    def cumulative_stats(self) -> StatsAccumulator:
        """Get cumulative statistics since start_time."""
        return self._cumulative_stats

    @property
    def window_stats(self) -> StatsAccumulator:
        """Get statistics for the current rolling window."""
        self._modlock.acquire()
        try:
            self._inc_to_cur_segment()
            result = StatsAccumulator.combine(self.current_label, *self._segment_stats)
        finally:
            self._modlock.release()
        return result

    @property
    def current_label(self) -> str:
        """Get the label for the current window."""
        return f"Window-{self._reftime}-{self._window_delta}"

    def as_dict(self) -> Dict[str, Any]:
        """Get a dictionary containing a summary of this instance's information.

        Returns:
            Dict[str, Any]: Dictionary with keys: now, cumulative, window.
        """
        result: Dict[str, Any] = {"now": str(datetime.now())}

        cumulative_info: Dict[str, Any] = {"since": str(self.start_time)}
        self._add_stats_info(self.cumulative_stats, cumulative_info)
        result["cumulative"] = cumulative_info

        window_info: Dict[str, Any] = {"width_millis": self.window_width}
        current_window_stats = self.window_stats
        self._add_stats_info(current_window_stats, window_info)
        result["window"] = window_info

        return result

    def __str__(self) -> str:
        """Get the statistics as a JSON string.

        Returns:
            str: JSON representation of the statistics dictionary.
        """
        return json.dumps(self.as_dict(), sort_keys=True)

    def reset(self) -> None:
        """Reset all statistics and restart from current time."""
        self._modlock.acquire()
        try:
            self._starttime = datetime.now()
            self._reftime = self._starttime
            self._cur_segment = 0
            for segment in self._segment_stats:
                segment.clear()
            self._cumulative_stats.clear()
        finally:
            self._modlock.release()

    def add(self, *values: float) -> int:
        """Add one or more values to the current segment.

        Args:
            *values: One or more numeric values to add.

        Returns:
            int: The segment number to which the values were added (useful for testing).
        """
        self._modlock.acquire()
        try:
            self._inc_to_cur_segment()
            self._segment_stats[self._cur_segment].add(*values)
            result = self._cur_segment
            self._cumulative_stats.add(*values)
        finally:
            self._modlock.release()

        return result

    def has_window_activity(self) -> Tuple[bool, StatsAccumulator]:
        """Determine whether the current window has activity.

        Returns:
            Tuple[bool, StatsAccumulator]: A tuple of (has_activity, current_window_stats)
                where has_activity is True if the window has any values.
        """
        window_stats = self.window_stats
        return (window_stats.n > 0, window_stats)

    def _inc_to_cur_segment(self) -> None:
        """Internal method to advance to the current segment, clearing old segments."""
        result = datetime.now()
        seg_num = int(
            (self._get_millis(result, self._starttime) % self._window_width) / self._segment_width
        )

        diff = self._get_millis(result, self._reftime)

        # if advanced to new segment or wrapped around current
        if seg_num != self._cur_segment or diff > self._segment_width:
            if self._num_segments == 1:
                # special case: wrapped around one and only segment in window
                self._segment_stats[0].clear()
            elif diff > self._window_width:
                # wrapped around the entire window, need to clear all
                for stats in self._segment_stats:
                    stats.clear()
                self._cur_segment = seg_num
            else:
                # walk up to and including new current segment, clearing each
                next_seg_num = (seg_num + 1) % self._num_segments
                i = (self._cur_segment + 1) % self._num_segments
                while i != next_seg_num:
                    self._segment_stats[i].clear()
                    i = (i + 1) % self._num_segments
                self._cur_segment = seg_num

        self._reftime = result

    def _get_millis(self, laterdatetime: datetime, earlierdatetime: datetime) -> int:
        """Internal method to compute milliseconds between two datetimes.

        Args:
            laterdatetime: The later datetime.
            earlierdatetime: The earlier datetime.

        Returns:
            int: Milliseconds between the two datetimes.
        """
        return int((laterdatetime - earlierdatetime).total_seconds() * 1000)

    def _add_stats_info(self, stats: StatsAccumulator, info: Dict[str, Any]) -> None:
        """Add stats summary information to the info dictionary.

        Args:
            stats: The StatsAccumulator to summarize.
            info: Dictionary to update with summary information.
        """
        if stats.n > 0:
            info["status"] = "active"
            info["stats"] = stats.as_dict()
            info["millis_per_item"] = self.get_millis_per_item(stats)
            info["items_per_milli"] = self.get_items_per_milli(stats)
        else:
            info["status"] = "inactive"

    @staticmethod
    def get_millis_per_item(stats: StatsAccumulator | None) -> float | None:
        """Extract the average number of milliseconds per item from the stats.

        Args:
            stats: StatsAccumulator instance or None.

        Returns:
            float | None: Average milliseconds per item, or None if stats is None
                or has no values.
        """
        result = None

        if stats is not None:
            if stats.n > 0:
                result = stats.mean

        return result

    @staticmethod
    def get_items_per_milli(stats: StatsAccumulator | None) -> float | None:
        """Extract the average number of items per millisecond from the stats.

        Args:
            stats: StatsAccumulator instance or None.

        Returns:
            float | None: Average items per millisecond, or None if stats is None
                or mean is zero.
        """
        result = None

        if stats is not None:
            if stats.mean > 0:
                result = 1.0 / stats.mean

        return result


class Monitor:
    """A monitor tracks processing and/or access times and rates for some function.

    Monitors measure both access frequency (how often something is called) and
    processing time (how long it takes). Uses RollingStats to provide both
    cumulative and rolling window statistics.

    Attributes:
        alive_since: Time when this monitor was created.
        description: Optional description of what is being monitored.
        last_access_time: Time when access was last recorded.
        access_times: RollingStats for tracking time between accesses.
        processing_times: RollingStats for tracking processing duration.
        access_cumulative_stats: Cumulative access time statistics.
        access_window_stats: Rolling window access time statistics.
        access_window_width: Width of the access time rolling window.
        processing_cumulative_stats: Cumulative processing time statistics.
        processing_window_stats: Rolling window processing time statistics.
        processing_window_width: Width of the processing time rolling window.
        default_window_width: Default window width for auto-created RollingStats.
        default_segment_width: Default segment width for auto-created RollingStats.
    """

    def __init__(
        self,
        description: str | None = None,
        access_times: RollingStats | None = None,
        processing_times: RollingStats | None = None,
        default_window_width: int = 300000,
        default_segment_width: int = 5000,
    ) -> None:
        """Initialize a Monitor.

        Args:
            description: Optional description of what is being monitored.
                Defaults to None.
            access_times: RollingStats instance for tracking access times.
                Defaults to None (created on first mark).
            processing_times: RollingStats instance for tracking processing times.
                Defaults to None (created on first mark with endtime).
            default_window_width: Default window width in milliseconds for
                auto-created RollingStats. Defaults to 300000 (5 minutes).
            default_segment_width: Default segment width in milliseconds for
                auto-created RollingStats. Defaults to 5000 (5 seconds).
        """
        self._modlock = Lock()
        self._alive_since = datetime.now()
        self._description = description
        self._last_start_time: datetime | None = None
        self._access_times = access_times
        self._processing_times = processing_times

        # Defaults for when creating RollingStats from within this Monitor
        self._default_window_width = default_window_width
        self._default_segment_width = default_segment_width

    @property
    def alive_since(self) -> datetime:
        """Get the time when this monitor was created."""
        return self._alive_since

    @property
    def description(self) -> str | None:
        """Get the description of what is being monitored."""
        return self._description

    @description.setter
    def description(self, val: str | None) -> None:
        """Set the description of what is being monitored."""
        self._description = val

    @property
    def last_access_time(self) -> datetime | None:
        """Get the time at which access was last recorded."""
        return self._last_start_time

    @property
    def access_times(self) -> RollingStats | None:
        """Get the access_times (RollingStats)."""
        return self._access_times

    @property
    def processing_times(self) -> RollingStats | None:
        """Get the processing_times (RollingStats)."""
        return self._processing_times

    @property
    def access_cumulative_stats(self) -> StatsAccumulator | None:
        """Get cumulative access time statistics."""
        return None if self._access_times is None else self._access_times.cumulative_stats

    @property
    def access_window_stats(self) -> StatsAccumulator | None:
        """Get rolling window access time statistics."""
        return None if self._access_times is None else self._access_times.window_stats

    @property
    def access_window_width(self) -> int | None:
        """Get the access time rolling window width in milliseconds."""
        return None if self._access_times is None else self._access_times.window_width

    @property
    def processing_cumulative_stats(self) -> StatsAccumulator | None:
        """Get cumulative processing time statistics."""
        return None if self._processing_times is None else self._processing_times.cumulative_stats

    @property
    def processing_window_stats(self) -> StatsAccumulator | None:
        """Get rolling window processing time statistics."""
        return None if self._processing_times is None else self._processing_times.window_stats

    @property
    def processing_window_width(self) -> int | None:
        """Get the processing time rolling window width in milliseconds."""
        return None if self._processing_times is None else self._processing_times.window_width

    @property
    def default_window_width(self) -> int:
        """Get the default window width for auto-created RollingStats."""
        return self._default_window_width

    @default_window_width.setter
    def default_window_width(self, val: int) -> None:
        """Set the default window width for auto-created RollingStats."""
        self._default_window_width = val

    @property
    def default_segment_width(self) -> int:
        """Get the default segment width for auto-created RollingStats."""
        return self._default_segment_width

    @default_segment_width.setter
    def default_segment_width(self, val: int) -> None:
        """Set the default segment width for auto-created RollingStats."""
        self._default_segment_width = val

    def as_dict(self) -> Dict[str, Any]:
        """Get a dictionary containing a summary of this instance's information.

        Returns:
            Dict[str, Any]: Dictionary with keys: alive_since, description (optional),
                last_mark (optional), access_stats (optional), processing_stats (optional).
        """
        result: Dict[str, Any] = {}

        result["alive_since"] = str(self.alive_since)
        if self.description is not None:
            result["description"] = self.description
        if self.last_access_time is not None:
            result["last_mark"] = str(self.last_access_time)
        if self.access_times is not None:
            result["access_stats"] = self.access_times.as_dict()
        if self.processing_times is not None:
            result["processing_stats"] = self.processing_times.as_dict()

        return result

    def __str__(self) -> str:
        """Get the monitor statistics as a JSON string.

        Returns:
            str: JSON representation of the statistics dictionary.
        """
        return json.dumps(self.as_dict(), sort_keys=True)

    def get_stats(self, access: bool = False, window: bool = False) -> StatsAccumulator | None:
        """Get window or cumulative access or processing stats.

        Args:
            access: If True, get access stats; if False, get processing stats.
                Defaults to False.
            window: If True, get window stats; if False, get cumulative stats.
                Defaults to False.

        Returns:
            StatsAccumulator | None: The requested statistics, or None if not available.
        """
        result = None

        rolling_stats = self._access_times if access else self._processing_times
        if rolling_stats is not None:
            result = rolling_stats.window_stats if window else rolling_stats.cumulative_stats

        return result

    def mark(self, starttime: datetime, endtime: datetime | None = None) -> None:
        """Mark another access and optionally record processing time.

        Records the time between this call and the previous call as an access time.
        If endtime is provided, also records the time between starttime and endtime
        as processing time.

        Args:
            starttime: The start time of this access.
            endtime: Optional end time for recording processing duration.
                Defaults to None.
        """
        self._modlock.acquire()
        try:
            if self._last_start_time is not None:
                if self._access_times is None:
                    # initialize if needed
                    self._access_times = RollingStats(
                        self.default_window_width, self.default_segment_width
                    )

                self._access_times.add(self._get_millis(starttime, self._last_start_time))

            if endtime is not None:
                if self._processing_times is None:
                    # initialize if needed
                    self._processing_times = RollingStats(
                        self.default_window_width, self.default_segment_width
                    )

                self._processing_times.add(self._get_millis(endtime, starttime))

            self._last_start_time = starttime

        finally:
            self._modlock.release()

    def _get_millis(self, laterdatetime: datetime, earlierdatetime: datetime) -> int:
        """Internal method to compute milliseconds between two datetimes.

        Args:
            laterdatetime: The later datetime.
            earlierdatetime: The earlier datetime.

        Returns:
            int: Milliseconds between the two datetimes.
        """
        return int((laterdatetime - earlierdatetime).total_seconds() * 1000)


class MonitorManager:
    """Manage a set of monitors by label, providing rollup views across all monitors.

    Maintains a collection of Monitor instances identified by labels, allowing
    aggregation of statistics across multiple monitors. Uses a KeyManager to
    generate unique keys from type and description pairs.

    Attributes:
        default_window_width: Default window width for auto-created Monitors.
        default_segment_width: Default segment width for auto-created Monitors.
    """

    def __init__(
        self, default_window_width: int = 300000, default_segment_width: int = 5000
    ) -> None:
        """Initialize a MonitorManager.

        Args:
            default_window_width: Default window width in milliseconds for
                auto-created Monitors. Defaults to 300000 (5 minutes).
            default_segment_width: Default segment width in milliseconds for
                auto-created Monitors. Defaults to 5000 (5 seconds).
        """
        self._monitors: Dict[str, Monitor] = {}
        self._key_manager = KeyManager()

        # Defaults for when creating a Monitor from within this MonitorManager
        self._default_window_width = default_window_width
        self._default_segment_width = default_segment_width

    @property
    def default_window_width(self) -> int:
        """Get the default window width for auto-created Monitors."""
        return self._default_window_width

    @default_window_width.setter
    def default_window_width(self, val: int) -> None:
        """Set the default window width for auto-created Monitors."""
        self._default_window_width = val

    @property
    def default_segment_width(self) -> int:
        """Get the default segment width for auto-created Monitors."""
        return self._default_segment_width

    @default_segment_width.setter
    def default_segment_width(self, val: int) -> None:
        """Set the default segment width for auto-created Monitors."""
        self._default_segment_width = val

    def get_monitors(self) -> Dict[str, Monitor]:
        """Get all monitors.

        Returns:
            Dict[str, Monitor]: Dictionary mapping labels to Monitor instances.
        """
        return self._monitors

    def get_monitor(
        self, label: str, create_if_missing: bool = False, description: str | None = None
    ) -> Monitor | None:
        """Get a monitor by label, optionally creating it if missing.

        Args:
            label: The monitor label.
            create_if_missing: If True, create a new Monitor if the label doesn't exist.
                Defaults to False.
            description: Optional description for a newly created Monitor.
                Defaults to None.

        Returns:
            Monitor | None: The Monitor instance, or None if not found and
                create_if_missing is False.
        """
        result = None

        if label in self._monitors:
            result = self._monitors[label]

        elif create_if_missing:
            result = Monitor(
                description=description,
                default_window_width=self.default_window_width,
                default_segment_width=self.default_segment_width,
            )
            self._monitors[label] = result

        return result

    def get_or_create_monitor_by_key_type(self, keytype: str, description: str) -> Monitor:
        """Get or create a monitor using a key type and description.

        Generates a unique key from the keytype and description using the internal
        KeyManager, then gets or creates a monitor with that key.

        Args:
            keytype: The key type (e.g., "endpoint", "function").
            description: The description to convert to a key.

        Returns:
            Monitor: The Monitor instance (always created if missing).
        """
        key = self._key_manager.get_key(keytype, description)
        result = self.get_monitor(key, create_if_missing=True, description=description)
        assert result is not None  # Always created when create_if_missing=True
        return result

    def set_monitor(self, label: str, monitor: Monitor) -> None:
        """Set or replace a monitor for a given label.

        Args:
            label: The monitor label.
            monitor: The Monitor instance to set.
        """
        self._monitors[label] = monitor

    def get_stats(
        self, label: str | None = None, access: bool = False, window: bool = False
    ) -> StatsAccumulator | None:
        """Get window or cumulative access or processing stats for a label or all monitors.

        Args:
            label: The monitor label, or None to aggregate across all monitors.
                Defaults to None.
            access: If True, get access stats; if False, get processing stats.
                Defaults to False.
            window: If True, get window stats; if False, get cumulative stats.
                Defaults to False.

        Returns:
            StatsAccumulator | None: The requested statistics, or None if not available.
        """
        result = None

        if label is None:
            result = StatsAccumulator("rollup")

            # Combine access stats across all monitors
            for monitor in self._monitors.values():
                result.incorporate(monitor.get_stats(access, window))

        elif label in self._monitors:
            result = self._monitors[label].get_stats(access, window)

        return result

    def as_dict(self) -> Dict[str, Any]:
        """Get a dictionary containing a summary of this instance's information.

        Returns:
            Dict[str, Any]: Dictionary with overall_stats and individual monitor
                statistics keyed by label.
        """
        result = {}

        # add overall processing/access, cumulative/window stats
        result["overall_stats"] = self.get_overall_stats()

        # add as_dict for each individual monitor
        for key, monitor in self._monitors.items():
            result[key] = monitor.as_dict()

        return result

    def __str__(self) -> str:
        """Get the monitor manager statistics as a JSON string.

        Returns:
            str: JSON representation of the statistics dictionary.
        """
        return json.dumps(self.as_dict(), sort_keys=True)

    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall statistics aggregated across all monitors.

        Returns:
            Dict[str, Any]: Dictionary with keys: cumulative_processing,
                cumulative_access, window_processing, window_access (each optional).
        """
        result = {}

        cumulative_processing = self.get_stats(access=False, window=False)
        if cumulative_processing is not None:
            result["cumulative_processing"] = cumulative_processing.as_dict()

        cumulative_access = self.get_stats(access=True, window=False)
        if cumulative_access is not None:
            result["cumulative_access"] = cumulative_access.as_dict()

        window_processing = self.get_stats(access=False, window=True)
        if window_processing is not None:
            result["window_processing"] = window_processing.as_dict()

        window_access = self.get_stats(access=True, window=True)
        if window_access is not None:
            result["window_access"] = window_access.as_dict()

        return result


class KeyManager:
    """Turn descriptions of key types into keys of the form type-N.

    Maps (keytype, description) pairs to unique keys by tracking descriptions
    for each keytype and assigning sequential numbers. The same description
    always maps to the same key for a given keytype.

    Example:
        >>> from dataknobs_utils.stats_utils import KeyManager
        >>> km = KeyManager()
        >>> km.get_key("endpoint", "/api/users")
        "endpoint-0"
        >>> km.get_key("endpoint", "/api/posts")
        "endpoint-1"
        >>> km.get_key("endpoint", "/api/users")
        "endpoint-0"
    """

    def __init__(self) -> None:
        """Initialize a KeyManager."""
        self._keytype2descriptions: Dict[str, List[str]] = {}

    def get_key(self, keytype: str, description: str) -> str:
        """Get a unique key for a (keytype, description) pair.

        Args:
            keytype: The key type (e.g., "endpoint", "function").
            description: The description to convert to a key.

        Returns:
            str: A unique key in the form "keytype-N" where N is the index.
        """
        if keytype in self._keytype2descriptions:
            descriptions = self._keytype2descriptions[keytype]
        else:
            descriptions = []
            self._keytype2descriptions[keytype] = descriptions

        if description in descriptions:
            index = descriptions.index(description)
        else:
            index = len(descriptions)
            descriptions.append(description)

        result = f"{keytype}-{index}"
        return result
