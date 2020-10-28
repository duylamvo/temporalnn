"""Utils for supporting climate data."""
import pandas as pd
import numpy as np
import tqdm
import tempfile
import os
import csv
import itertools
import multiprocessing as mp

# set to suppress scientific number
np.set_printoptions(suppress=True)


def df_to_ts_numpy(data, independents, dependent, group_col=None,
                   n_in_steps=7, n_out_steps=1, stride=1,
                   split_test=True, **kwargs):
    """Convert a pandas data frame to time series numpy for training.

    :param data: (pandas.DataFrame) data frame which already has index as time-based.
        If it is not yet set, please use df.set_index(time_col) to make sure that is time series
        data frame.
    :param independents: (list of str) independent Dimensions y ~ AX + b
    :param dependent: (str) dependent variable.
    :param group_col: (str) a group is a categorical variable in multivariate time series. That breaks a
        a data frame into multi multivariate time series.
    :param n_in_steps: (int) input steps t for sliding.
    :param n_out_steps: (int) out put steps t for estimation. For example, if we want to estimate the output
        of next 2 days weather based on 7-days previous data, then x_steps = 7, and y_steps = 2
    :param stride: (int) stride is to how sliding windows jump to next window. For example after 7 days,
        in stead of slide to next window, it will jump to position of 7 + stride to capture input and output.
    :param split_test: (boolean) if true, then the return will be a tuple of x_train, x_test, y_train, y_test,
        if false, then it will return only x_train, y_train data.
    :param kwargs: other argument of train_test_split function in sklearn.model_selection
    """
    pool = mp.Pool(mp.cpu_count())
    queues = []
    if isinstance(independents, str):
        independents = [independents]

    if group_col is None:
        group_col = "group"
        data[group_col] = 1
        for group, _ts in data.groupby(group_col):
            in_dep_data = _ts[independents]
            dep_data = _ts[dependent]

            args = [in_dep_data, dep_data, n_in_steps, n_out_steps, stride]
            queues.append(pool.apply_async(_generate_time_series_train_set, args))

    _X = []
    _Y = []
    for q in tqdm.tqdm(queues):
        x_train, y_train = q.get()
        if len(x_train) and len(y_train):
            _X.extend(x_train)
            _Y.extend(y_train)

    if not len(_X):
        raise ValueError("No sample are generated")
    _shape = _X[0].shape
    _expected_shape = (n_in_steps, len(independents))
    assert _shape == _expected_shape, \
        "Shape {} does not match {}".format(_shape, _expected_shape)
    assert len(_X) == len(_Y)

    x_data = np.stack(_X)
    y_data = np.stack(_Y)

    if split_test:
        from sklearn.model_selection import train_test_split
        return train_test_split(x_data, y_data, **kwargs)
    else:
        return x_data, y_data


def _generate_time_series_train_set(x_data, y_data, x_steps=7, y_steps=1, stride=1):
    x_size = len(x_data.index)
    y_size = len(y_data.index)

    assert x_size == y_size

    _inputs = []
    _outputs = []
    if x_size > x_steps + y_steps + 1:
        i = 0
        while True:
            _x_start = i * stride
            _x_end = _x_start + x_steps
            _y_start = _x_end
            _y_end = _x_end + y_steps

            if _y_end > x_size:
                _y_end = x_size
                _y_start = x_size - y_steps
                _x_end = _y_start
                _x_start = _x_end - x_steps

            _inputs.append(x_data[_x_start:_x_end].to_numpy())
            _outputs.append(y_data[_y_start:_y_end].to_numpy())

            if _y_end == x_size:
                break
            i = i + 1
    return _inputs, _outputs


def save_numpy_ts(data, file_path=None):
    """Save numpy ts to file, which could be loaded again by using numpy.load function.

    """
    file_path = file_path or tempfile.mktemp()
    base_dir = os.path.dirname(file_path)

    os.makedirs(base_dir, exist_ok=True)
    np.save(file_path, data)
    return file_path


def read_numpy_ts(file_path):
    return np.load(file_path)


def interval(ts, freq="D"):
    # to date time
    ts = pd.to_datetime(ts)

    # create full date_range, daily
    min_date = ts.min()
    max_date = ts.max()

    date_range = pd.date_range(start=min_date, end=max_date, freq=freq)

    return date_range


def is_na(_df):
    col_stats = _df.isna().sum()
    _cols_with_nan = col_stats[col_stats > 0]
    _is_na = len(_cols_with_nan) != 0
    if _is_na:
        Warning("There is NaN data in columns: {}".format(_cols_with_nan))
    return _is_na


def cleaning_ts(ts, time_col, na_values):
    # convert measure_date to real_date
    df = ts.sort_values(by=[time_col])

    # each of group has different date range, but should be smoothly
    date_range = interval(ts[time_col])

    df[time_col] = pd.to_datetime(df[time_col])

    df = df.set_index(time_col)
    df = df.reindex(date_range)
    df = df.interpolate(limit_direction="both")
    df = df.fillna(na_values)

    if is_na(df):
        raise ValueError("Exist NAN values in")
    return df


def preprocess_data(data, time_col, group_col, na_values, workers=1):
    print("Preprocessing data")
    with mp.Pool(workers) as pool:
        _queues = []
        for group, _df in data.groupby(group_col):
            args = [_df, time_col, na_values]
            _queues.append(pool.apply_async(cleaning_ts, args))

        _processed = [_q.get() for _q in tqdm.tqdm(_queues)]
        return pd.concat(_processed)


def shift_ts(ts, cols, step, shift_up=True):
    # Using negative i will shift data for each col up
    if shift_up:
        _sign = -1
    else:
        _sign = 1

    # drop na will help to remove nan when shift ts
    return ts[cols].shift(_sign * step).dropna()


def ts_steps_gen(ts, cols, steps=7,
                 start_step=0, shift_up=True,
                 use_multiprocessing=False, workers=1):
    _processed = []
    if use_multiprocessing:
        with mp.Pool(workers) as pool:
            _queues = []
            for i in range(start_step, start_step + steps):
                args = [ts, cols, i, shift_up]
                _queues.append(pool.apply_async(shift_ts, args))

            # set time out to avoid process locked
            _processed = [_q.get() for _q in _queues]
    else:
        for i in range(start_step, start_step + steps):
            _processed.append(shift_ts(ts, cols, i, shift_up))
    return pd.concat(_processed, axis=1).dropna()


def _combine_to_array(_ts, _c):
    temp = _ts[[_c]].apply(np.array, axis=1)
    temp.name = _c
    return temp


def combine_ts(ts, use_multiprocessing=False, workers=1):
    """Combine all columns which has same name into array.
    Here we try to generate each row is an example.
        sample x1             x2              y
        1      [v1, v2, v3]   [v1, v2, v3]    [y1, y2]
        2      [v1, v2, v3]   [v1, v2, v3]    [y1, y2]
    """
    _cols = ts.keys().unique()
    _processed = []
    if use_multiprocessing:
        with mp.Pool(workers) as pool:
            _queues = []
            for c in _cols:
                args = [ts, c]
                _queues.append(pool.apply_async(_combine_to_array, args))
            _processed = [_q.get() for _q in _queues]
    else:
        for c in _cols:
            _processed.append(_combine_to_array(ts, c))
    return pd.concat(_processed, axis=1)


def yield_samples(train_df, group_col, indep_cols, dep_col,
                  x_steps, y_steps, batch_size=32, stride=1,
                  use_multiprocessing=False, workers=1
                  ):
    workers = 1 if not use_multiprocessing else workers
    walk_forward = WalkForward(x_steps, y_steps, stride=stride)
    args = [train_df, group_col, indep_cols, dep_col]
    while True:
        with mp.Pool(workers) as pool:
            _queues = []
            _walks = walk_forward.walk_through(train_df, group_col)
            for walk in _walks:
                _args = [walk] + args
                _queues.append(pool.apply_async(to_array, _args))

            for _q in _queues:
                batch_x = []
                batch_y = []
                for i in range(batch_size):
                    x, y = _q.get()
                    batch_x.append(x)
                    batch_y.append(y)

                if len(batch_x) > 0:
                    # 1 item = 1 batch_size of samples of
                    # (batch_size, n_steps, n_features)
                    yield np.stack(batch_x), np.stack(batch_y)


def write_samples(filename, train_df, group_col, indep_cols, dep_col,
                  x_steps, y_steps, stride=1,
                  batch_size=32, use_multiprocessing=False, workers=1
                  ):
    """Write numpy arrays to text.
    One 2D rows is a sample [[...],[...]]
    Todo: How to write numpy to correct format that is easy to read back
    """
    fx = open(filename + "_x", "a")
    fy = open(filename + "_y", "a")
    writer_x = csv.writer(fx, delimiter=";")
    writer_y = csv.writer(fy, delimiter=";")

    workers = 1 if not use_multiprocessing else workers
    walk_forward = WalkForward(x_steps, y_steps, stride=stride)

    with mp.Pool(workers) as pool:
        _queues = []
        walks = walk_forward.walk_through(train_df, group_col)

        while True:
            _batch_walks = list(itertools.islice(walks, batch_size))
            if len(_batch_walks) == 0:
                break
            args = [_batch_walks, train_df, group_col, indep_cols, dep_col]
            _queues.append(pool.apply_async(to_batch, args))

        # for _q in _queues:
        x, y = _queues[0].get()
        writer_x.writerows(x)
        writer_y.writerows(y)

        x, y = _queues[1].get()
        writer_x.writerows(x)
        writer_y.writerows(y)


def get_steps_per_epoch(df, group_col, x_steps, y_steps, batch_size=0, stride=1):
    """Approximation of steps.

     ~ it is approximated if df has different lengths
     """

    # assume that all group has same date_range
    n_groups = len(df[group_col].unique())
    _range = interval(df.index, "D")

    # estimate number possible of N_samples per a ts (by group)
    n_samples = len(_range) - (x_steps + y_steps) + 1
    if stride > 1:
        n_samples = (n_samples // stride) + 1
    total_samples = (n_samples * n_groups)
    if batch_size == 0:
        raise ValueError("Batch size must be larger than 0")

    steps_per_epoch = (total_samples // batch_size // stride) + 1

    return steps_per_epoch


def to_batch(walks, train_df, group_col, indep_cols, dep_col):
    # A walk is a slice of index that including group, index_x, index_y
    batch_x = []
    batch_y = []

    for w in walks:
        arr_x, arr_y = to_array(w, train_df, group_col, indep_cols, dep_col)
        batch_x.append(arr_x)
        batch_y.append(arr_y)

    if len(batch_x) > 0:
        # 1 item = 1 batch_size of samples of (batch_size, n_steps, n_features)
        return np.stack(batch_x), np.stack(batch_y)


def to_array(walk, train_df, group_col, indep_cols, dep_col):
    group, _idx, _idy = walk

    indices_g = train_df[group_col] == group

    # Get all matching index - (true/false series)
    indices_x = train_df.index.isin(_idx)
    indices_y = train_df.index.isin(_idy)

    # Get all data matching both indices group and train | output
    df_x = train_df[indices_g & indices_x][indep_cols]
    df_y = train_df[indices_g & indices_y][dep_col]

    # Convert to n_d_array -> (n_steps, n_features) shape
    arr_x = df_x.to_numpy().round(3)
    arr_y = df_y.to_numpy().round(3)
    return arr_x, arr_y


class WalkForward:
    def __init__(self, train_size, test_size=1, stride=1, keep_start_point=False):
        self.train_size = train_size
        self.test_size = test_size
        self.keep_start_point = keep_start_point
        self.stride = stride

    def split(self, ts, stride=1, group=None):
        """

        :param ts: Index Series
        :param stride: stride to shift a window
        :param group: group or id of the time series
        :return: generator of series index.
         (group, train_indices, test_indices) is yielded
        """
        _indices = pd.Index(pd.Series(ts)).sort_values().copy()
        if self.train_size + self.test_size > len(_indices):
            Warning("Train size + Test size smaller than length of data")
            return
        if stride < 1:
            raise ValueError("Stride cannot smaller than 1")

        i = 0
        while True:
            _x_start = i * stride
            _x_end = _x_start + self.train_size
            _y_start = _x_end
            _y_end = _x_end + self.test_size

            if _y_end > len(ts):
                _y_end = len(ts)
                _y_start = len(ts) - self.test_size
                _x_end = _y_start
                _x_start = _x_end - self.train_size

            if self.keep_start_point:
                _x_start = 0

            yield group, _indices[_x_start:_x_end], _indices[_y_start: _y_end]

            if _y_end == len(ts):
                break
            i = i + 1

    def walk_through(self, train_df, group_col):
        """Slicing through all data set, and generate slices.
        Slicing through index without generating new data frames will help
        reducing time to make and copy time series.

        return a generator of all tuples (group, indices_of_x, indices_of_y)
        """
        # Todo: Support when group_col = None ~ when single mts
        for group, _df in train_df.groupby(group_col):
            # walk forward and capture indices of a group
            yield from self.split(_df.index, self.stride, group=group)

    def walk_to_batches(self, train_df, group_col, batch_size=1):

        # number of batches or steps
        print("Setup Walker to go through and capture batches")
        _batches = []
        _walks = self.walk_through(train_df, group_col)

        while True:
            b = list(itertools.islice(_walks, batch_size))
            if len(b) == 0:
                break
            _batches.append(b)
        return _batches
