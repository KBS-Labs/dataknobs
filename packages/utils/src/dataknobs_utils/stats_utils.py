import json
import math
import random
import time
from datetime import datetime, timedelta
from threading import Lock


def wait_for_random_millis(uptomillis):
    waittime = random.uniform(1, uptomillis)
    time.sleep(waittime / 1000)
    return waittime


class StatsAccumulator:
    '''
    A low-memory helper class to collect statistical samples and provide
    summary statistics.
    '''

    def __init__(self, label='', other=None, as_dict=None, values=None):
        '''
        :param label: A label or name for this instance.
        :param other: An other instance for copying into this
        :param as_dict: An "as_dict" form of stats to start with
        :param values: A list of initial values or a single value to add
        '''
        self.label = label
        self._n = 0
        self._min = 0.0
        self._max = 0.0
        self._sum = 0.0
        self._sos = 0.0
        self._modlock = Lock()

        if other is not None:
            if not label == '':
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
    def label(self):
        ''' Get the label or name '''
        return self._label

    @label.setter
    def label(self, val):
        ''' Set the label or name '''
        self._label = val

    @property
    def n(self):
        ''' Get the number of values added '''
        return self._n

    @property
    def min(self):
        ''' Get the minimum value added '''
        return self._min

    @property
    def max(self):
        ''' Get the maximum value added '''
        return self._max

    @property
    def sum(self):
        ''' Get the sum of all values added '''
        return self._sum

    @property
    def sum_of_squares(self):
        ''' Get the sum of all squared values '''
        return self._sos

    @property
    def mean(self):
        ''' Get the mean of the values '''
        return 0 if self._n == 0 else self._sum / self._n

    @property
    def std(self):
        ''' Get the standard deviation of the values '''
        return 0 if self._n < 2 else math.sqrt(self.var)

    @property
    def var(self):
        ''' Get the variance of the values '''
        var = 0
        if self._n > 1:
            var = abs(
                (1.0 / (self._n - 1.0)) *
                (self._sos - (1.0 / self._n) * self._sum * self._sum)
            )
        return var

    def clear(self, label=''):
        ''' Clear all values (reset) '''
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

    def initialize(self, label='', n=0, min=0, max=0, mean=0, std=0, as_dict=None):
        '''
        Initialize with the given values, preferring existing values from the dictionary.
        '''
        if as_dict is not None:
            if 'label' in as_dict:
                label = as_dict['label']
            if 'n' in as_dict:
                n = as_dict['n']
            if 'min' in as_dict:
                min = as_dict['min']
            if 'max' in as_dict:
                max = as_dict['max']
            if 'mean' in as_dict:
                mean = as_dict['mean']
            if 'std' in as_dict:
                std = as_dict['std']
                

        self._modlock.acquire()
        try:
            self._label = label
            self._n = n
            self._min = min
            self._max = max
            if as_dict is not None and 'sum' in as_dict:
                self._sum = as_dict['sum']
            else:
                self._sum = mean * n
            if as_dict is not None and 'sos' in as_dict:
                self._sos = as_dict['sos']
            else:
                self._sos = 0 if n == 0 else std * std * (n - 1.0) + self._sum * self._sum / n
        finally:
            self._modlock.release()

    def as_dict(self, with_sums=False):
        '''
        Get a dictionary containing a summary of this instance's information.
        '''
        d = {
            'label': self.label,
            'n': self.n,
            'min': self.min,
            'max': self.max,
            'mean': self.mean,
            'std': self.std
        }
        if with_sums:
            d.update({
                'sum': self._sum,
                'sos': self._sos
            })
        return d

    def __str__(self):
        ''' Get the info as a json string '''
        return json.dumps(self.as_dict(), sort_keys=True)

    def add(self, *values):
        ''' Add the value(s) (thread-safe) '''
        self._modlock.acquire()
        try:
            for value in values:
                self._do_add(value)
        finally:
            self._modlock.release()
        return self

    def _do_add(self, value):
        ''' Do the work of adding a value '''
        if self._n == 0:
            self._min = value
            self._max = value
        else:
            if value < self._min:
                self._min = value
            if value > self._max:
                self._max = value

        self._n += 1
        self._sos += (value * value)
        self._sum += value

    @staticmethod
    def combine(label, *stats_accumulators):
        '''
        Create a new statsAccumulator as if it had accumulated all data from
        the given list of stats accumulators.
        '''
        result = StatsAccumulator(label)
        for stats in stats_accumulators:
            result.incorporate(stats)
        return result

    def incorporate(self, other):
        '''
        Incorporate the other stats_accumulator's data into this as if this had
        accumulated the other's along with its own.
        '''
        if other is None:
            return
        
        self._modlock.acquire()
        try:
            if self._n == 0:
                self._min = other.min
                self._max = other.max
            else:
                if other.min < self._min:
                    self._min = other.min
                if other.max > self._max:
                    self._max = other.max
                        
            self._n += other.n
            self._sos += other.sum_of_squares
            self._sum += other.sum
        finally:
            self._modlock.release()


class LinearRegression:
    '''
    A low-memory helper class to collect (x,y) samples and compute the
    linear regression.
    '''

    def __init__(self):
        self._n = 0
        self._x_sum = 0.0
        self._y_sum = 0.0
        self._xy_sum = 0.0
        self._xx_sum = 0.0
        self._yy_sum = 0.0

        self._m = None
        self._b = None

    @property
    def n(self):
        return self._n

    @property
    def x_sum(self):
        return self._x_sum

    @property
    def y_sum(self):
        return self._y_sum

    @property
    def xy_sum(self):
        return self._xy_sum

    @property
    def xx_sum(self):
        return self._xx_sum

    @property
    def yy_sum(self):
        return self._yy_sum

    @property
    def m(self):
        if self._m is None:
            self._compute_regression()
        return self._m

    @property
    def b(self):
        if (self._b is None):
            self._compute_regression()
        return self._b

    def get_y(self, x):
        return self.m * x + self.b

    def add(self, x, y):
        self._m = None
        self._b = None
        self._n += 1
        self._x_sum += x
        self._y_sum += y
        self._xy_sum += (x * y)
        self._xx_sum += (x * x)
        self._yy_sum += (y * y)

    def __str__(self):
        return 'y = %.4f x + %.4f' % (self.m, self.b)

    def _compute_regression(self):
        self._m = 0.0
        self._b = 0.0

        denominator = self._n * self._xx_sum - self._x_sum * self._x_sum
        if (denominator != 0):
            m_numerator = self._n * self._xy_sum - self._x_sum * self._y_sum
            b_numerator = self._y_sum * self._xx_sum - self._x_sum * self._xy_sum

            self._m = m_numerator / denominator
            self._b = b_numerator / denominator


class RollingStats:
    '''
    Implementation for a collection of stats through a rolling window of time.
    '''

    def __init__(self, window_width=300000, segment_width=5000):
        self._modlock = Lock()
        self._window_width = window_width
        self._segment_width = segment_width
        self._window_delta = timedelta(milliseconds=window_width)
        self._cumulative_stats = StatsAccumulator("cumulative")

        self._num_segments = int(round(window_width / segment_width))
        self._segment_stats = [StatsAccumulator("segment-" + str(i)) for i in range(self._num_segments)]
        self._starttime = datetime.now()
        self._reftime = self._starttime
        self._cur_segment = 0

    @property
    def window_width(self):
        return self._window_width

    @property
    def num_segments(self):
        return self._num_segments

    @property
    def cur_segment(self):
        result = -1
        self._modlock.acquire()
        try:
            self._inc_to_cur_segment()
            result = self._cur_segment
        finally:
            self._modlock.release()
        return result

    @property
    def last_segment(self):
        return self._cur_segment

    @property
    def start_time(self):
        return self._starttime

    @property
    def ref_time(self):
        return self._reftime

    @property
    def cumulative_stats(self):
        return self._cumulative_stats

    @property
    def window_stats(self):
        '''
        Get the stats for the current window.
        '''
        self._modlock.acquire()
        try:
            self._inc_to_cur_segment()
            result = StatsAccumulator.combine(self.current_label, *self._segment_stats)
        finally:
            self._modlock.release()
        return result

    @property
    def current_label(self):
        return 'Window-%s-%s' % (str(self._reftime), str(self._window_delta))

    def as_dict(self):
        '''
        Get a dictionary containing a summary of this instance's information.
        '''
        result = {'now': str(datetime.now())}

        cumulative_info = {'since': str(self.start_time)}
        self._add_stats_info(self.cumulative_stats, cumulative_info)
        result['cumulative'] = cumulative_info

        window_info = {'width_millis': self.window_width}
        current_window_stats = self.window_stats
        self._add_stats_info(current_window_stats, window_info)
        result['window'] = window_info

        return result

    def __str__(self):
        return json.dumps(self.as_dict(), sort_keys=True)

    def reset(self):
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

    def add(self, *values):
        '''
        Add the current value(s), returning the segment number to which the
        value as added (useful for testing).
        '''
        self._modlock.acquire()
        try:
            self._inc_to_cur_segment()
            self._segment_stats[self._cur_segment].add(*values)
            result = self._cur_segment
            self._cumulative_stats.add(*values)
        finally:
            self._modlock.release()

        return result

    def has_window_activity(self):
        '''
        Determines whether the current window has activity, returning
            (has_activity, current_window_stats)
        '''
        window_stats = self.window_stats
        return (window_stats.n > 0, window_stats)

    def _inc_to_cur_segment(self):
        result = datetime.now()
        seg_num = int((self._get_millis(result, self._starttime) % self._window_width) / self._segment_width)
        
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
    
    def _get_millis(self, laterdatetime, earlierdatetime):
        return int((laterdatetime - earlierdatetime).total_seconds() * 1000)

    def _add_stats_info(self, stats, info):
        '''
        Add stats summary information to the 'info' dict.
        '''
        if stats.n > 0:
            info['status'] = 'active'
            info['stats'] = stats.as_dict()
            info['millis_per_item'] = self.get_millis_per_item(stats)
            info['items_per_milli'] = self.get_items_per_milli(stats)
        else:
            info['status'] = 'inactive'

    @staticmethod
    def get_millis_per_item(stats):
        '''
        Extract the (average) number of milliseconds per item from the stats.
        '''
        result = None

        if stats is not None:
            if stats.n > 0:
                result = stats.mean

        return result

    @staticmethod
    def get_items_per_milli(stats):
        '''
        Extract the (average) number of items per millisecond from the stats.
        '''
        result = None

        if stats is not None:
            if stats.mean > 0:
                result = 1.0 / stats.mean

        return result


class Monitor:
    '''
    A monitor tracks processing and/or access times and rates for some function.
    '''

    def __init__(
            self,
            description=None,
            access_times=None,
            processing_times=None,
            default_window_width=300000,
            default_segment_width=5000
    ):
        '''
        NOTE: access_times, processing_times should be RollingStats instances.
        '''
        self._modlock = Lock()
        self._alive_since = datetime.now()
        self._description = description
        self._last_start_time = None
        self._access_times = access_times
        self._processing_times = processing_times

        # Defaults for when creating RollingStats from within this Monitor
        self._default_window_width = default_window_width
        self._default_segment_width = default_segment_width


    @property
    def alive_since(self):
        return self._alive_since

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, val):
        self._description = val

    @property
    def last_access_time(self):
        '''
        Get the time at which access was last recorded.
        '''
        return self._last_start_time

    @property
    def access_times(self):
        '''
        Get the access_times (RollingStats).
        '''
        return self._access_times

    @property
    def processing_times(self):
        '''
        Get the processing_times (RollingStats).
        '''
        return self._processing_times

    @property
    def access_cumulative_stats(self):
        return None if self._access_times is None else self._access_times.cumulative_stats

    @property
    def access_window_stats(self):
        return None if self._access_times is None else self._access_times.window_stats

    @property
    def access_window_width(self):
        return None if self._access_times is None else self._access_times.window_width

    @property
    def processing_cumulative_stats(self):
        return None if self._processing_times is None else self._processing_times.cumulative_stats

    @property
    def processing_window_stats(self):
        return None if self._processing_times is None else self._processing_times.window_stats

    @property
    def processing_window_width(self):
        return None if self._processing_times is None else self._processing_times.window_width

    @property
    def default_window_width(self):
        return self._default_window_width

    @default_window_width.setter
    def default_window_width(self, val):
        self._default_window_width = val

    @property
    def default_segment_width(self):
        return self._default_segment_width

    @default_segment_width.setter
    def default_segment_width(self, val):
        self._default_segment_width = val


    def as_dict(self):
        '''
        Get a dictionary containing a summary of this instance's information.
        '''
        result = {}

        result['alive_since'] = str(self.alive_since)
        if self.description is not None:
            result['description'] = self.description
        if self.last_access_time is not None:
            result['last_mark'] = str(self.last_access_time)
        if self.access_times is not None:
            result['access_stats'] = self.access_times.as_dict()
        if self.processing_times is not None:
            result['processing_stats'] = self.processing_times.as_dict()

        return result

    def __str__(self):
        return json.dumps(self.as_dict(), sort_keys=True)

    def get_stats(self, access=False, window=False):
        '''
        Get window/cumulative access/processing stats.
        '''
        result = None

        rolling_stats = self._access_times if access else self._processing_times
        if rolling_stats is not None:
            result = rolling_stats.window_stats if window else rolling_stats.cumulative_stats

        return result

    def mark(self, starttime, endtime=None):
        '''
        Mark another access time, and processing time if endtime is not None.
        '''
        self._modlock.acquire()
        try:
            if self._last_start_time is not None:
                if self._access_times is None:
                    # initialize if needed
                    self._access_times = RollingStats(self.default_window_width, self.default_segment_width)

                self._access_times.add(self._get_millis(starttime, self._last_start_time))

            if endtime is not None:
                if self._processing_times is None:
                    # initialize if needed
                    self._processing_times = RollingStats(self.default_window_width, self.default_segment_width)

                self._processing_times.add(self._get_millis(endtime, starttime))
                    
            self._last_start_time = starttime

        finally:
            self._modlock.release()

    def _get_millis(self, laterdatetime, earlierdatetime):
        return int((laterdatetime - earlierdatetime).total_seconds() * 1000)


class MonitorManager:
    '''
    Class to manage a set of monitors by label, providing rollup views across.
    '''

    def __init__(self, default_window_width=300000, default_segment_width=5000):
        self._monitors = {}
        self._key_manager = KeyManager()

        # Defaults for when creating a Monitor from within this MonitorManager
        self._default_window_width = default_window_width
        self._default_segment_width = default_segment_width


    @property
    def default_window_width(self):
        return self._default_window_width

    @default_window_width.setter
    def default_window_width(self, val):
        self._default_window_width = val

    @property
    def default_segment_width(self):
        return self._default_segment_width

    @default_segment_width.setter
    def default_segment_width(self, val):
        self._default_segment_width = val

    def get_monitors(self):
        return self._monitors

    def get_monitor(self, label, create_if_missing=False, description=None):
        result = None

        if label in self._monitors:
            result = self._monitors[label]

        elif create_if_missing:
            result = Monitor(description=description, default_window_width=self.default_window_width, default_segment_width=self.default_segment_width)
            self._monitors[label] = result

        return result

    def get_or_create_monitor_by_key_type(self, keytype, description):
        key = self._key_manager.get_key(keytype, description)
        return self.get_monitor(key, create_if_missing=True, description=description)

    def set_monitor(self, label, monitor):
        self._monitors[label] = monitor

    def get_stats(self, label=None, access=False, window=False):
        '''
        Get window/cumulative access/processing stats for label or all.
        '''
        result = None

        if label is None:
            result = StatsAccumulator('rollup')

            # Combine access stats across all monitors
            for monitor in self._monitors.values():
                result.incorporate(monitor.get_stats(access, window))

        elif label in self._monitors:
            result = self._monitors[label].get_stats(access, window)

        return result

    def as_dict(self):
        '''
        Get a dictionary containing a summary of this instance's information.
        '''
        result = {}

        # add overall processing/access, cumulative/window stats
        result['overall_stats'] = self.get_overall_stats()

        # add as_dict for each individual monitor
        for key, monitor in self._monitors.items():
            result[key] = monitor.as_dict()

        return result

    def __str__(self):
        return json.dumps(self.as_dict(), sort_keys=True)

    def get_overall_stats(self):
        result = {}

        cumulative_processing = self.get_stats(access=False, window=False)
        if cumulative_processing is not None:
            result['cumulative_processing'] = cumulative_processing.as_dict()

        cumulative_access = self.get_stats(access=True, window=False)
        if cumulative_access is not None:
            result['cumulative_access'] = cumulative_access.as_dict()

        window_processing = self.get_stats(access=False, window=True)
        if window_processing is not None:
            result['window_processing'] = window_processing.as_dict()

        window_access = self.get_stats(access=True, window=True)
        if window_access is not None:
            result['window_access'] = window_access.as_dict()

        return result


class KeyManager:
    '''
    Class to turn descriptions of key types into keys of the form type-N.
    '''

    def __init__(self):
        self._keytype2descriptions = {}

    def get_key(self, keytype, description):

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

        result = '%s-%d' % (keytype, index)
        return result
