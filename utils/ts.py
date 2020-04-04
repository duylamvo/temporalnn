"""Utils for supporting climate data."""
import pandas as pd
import numpy as np
import tqdm
import os
import csv
import itertools
import multiprocessing as mp
import threading
import pickle
import keras

from keras import backend as K
from keras.callbacks import (EarlyStopping,
                             TensorBoard,
                             ReduceLROnPlateau)

from ..models import temporal as ts_model

# set to suppress scientific number
np.set_printoptions(suppress=True)

climate_schema = {
    "rsk": np.float64,
    "rskf": np.float64,
    "shk_tag": np.float64,
    "nm": np.float64,
    "vpm": np.float64,
    "tmk": np.float64,
    "upm": np.float64,
    "txk": np.float64,
    "tnk": np.float64,
    "tgk": np.float64,
    "rs": np.float64,
    "sdk": np.float64,
    "pm": np.float64,
    "fx": np.float64,
    "fm": np.float64,
    "rsf": np.float64,
    "sh_tag": np.float64,
    "nsh_tag": np.float64,
    "v_te002m": np.float64,
    "v_te005m": np.float64,
    "v_te010m": np.float64,
    "v_te020m": np.float64,
    "v_te050m": np.float64,
    "atmo_strahl": np.float64,
    "fd_strahl": np.float64,
    "fg_strahl": np.float64,
    "sd_strahl": np.float64,
    "ash_6": np.float64,
    "wash_6": np.float64,
    "waas_6": np.float64,
    "stations_height": np.float64,
    "geo_width": np.float64,
    "geo_length": np.float64,
    "stations_id": np.uint32,
    "year": np.uint32,
    "measure_date": np.datetime64
}


def load_csv(f, ignore, schema=None, mode="sample", sep=";"):
    print("Loading csv file")
    chunks = pd.read_csv(f, sep=sep, header=0, chunksize=5000)
    if mode == "sample":
        df = next(chunks)
    else:
        df = pd.concat(tqdm.tqdm(chunks))

    # drop unnecessary columns
    ignore = [k for k in ignore if k in df.columns]
    df = df.drop(ignore, axis=1)

    # drop rows with headers because error when generate file csv (if yes)
    keys = df.keys()
    key = keys[0]
    df = df[df[key] != key]

    # infer objects
    df = df.infer_objects()

    if isinstance(schema, dict):
        ig_keys = [k for k in schema.keys()
                   if k in ignore or k not in df.columns]
        for k in ig_keys:
            schema.pop(k)

    # set non-inferred objects by hand
    df = df.astype(schema)

    # round two decimals
    return df


def get_date_range(ts, freq="D"):
    # to date time
    ts = pd.to_datetime(ts)

    # create full date_range, daily
    min_date = ts.min()
    max_date = ts.max()

    date_range = pd.date_range(start=min_date, end=max_date, freq=freq)

    return date_range


def is_na(_df):
    _col_stats = _df.isna().sum()
    _cols_with_nan = _col_stats[_col_stats > 0]
    _is_na = len(_cols_with_nan) != 0
    if _is_na:
        Warning("There is NaN data in columns: {}".format(_cols_with_nan))
    return _is_na


def cleaning_ts(ts, time_col, na_values):
    # convert measure_date to real_date
    _df = ts.sort_values(by=[time_col])

    # each of group has different date range, but should be smoothly
    date_range = get_date_range(ts[time_col])

    _df[time_col] = pd.to_datetime(_df[time_col])

    _df = _df.set_index(time_col)
    _df = _df.reindex(date_range)
    _df = _df.interpolate(limit_direction="both")
    _df = _df.fillna(na_values)

    if is_na(_df):
        raise ValueError("Exist NAN values in")
    return _df


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


def train_on_generator(batches, input_shape, x_steps, y_steps,
                       weight_file=None, history_file=None,
                       log_dir='./logs', **kwargs
                       ):
    wn = ts_model.WaveNet()
    model = wn.build_model(input_shape=input_shape,
                           n_in_steps=x_steps,
                           n_out_steps=y_steps,
                           gated_activations=['relu', 'sigmoid'],
                           n_conv_filters=32)
    if weight_file:
        if os.path.isfile(weight_file):
            model.load_weights(weight_file)

    # Train
    early_stopper = EarlyStopping(monitor='loss',
                                  min_delta=0.01,
                                  patience=10,
                                  verbose=1,
                                  mode='auto')
    tensor_brd = TensorBoard(log_dir=log_dir,
                             histogram_freq=0)
    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=0.2,
                                  patience=5,
                                  min_lr=0.0001)

    # Root mean square error - which not defined in keras
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    model.compile(optimizer='nadam',
                  loss=rmse,
                  metrics=['mse', 'mae', 'mape', 'cosine'],
                  )

    history = model.fit_generator(batches,
                                  callbacks=[early_stopper, tensor_brd, reduce_lr],
                                  **kwargs)

    if weight_file:
        model.save_weights(weight_file)
    if history_file:
        with open(history_file, 'wb') as f:
            pickle.dump(history.history, f)
    return model


def train_on_memory(x_train, y_train, x_test, y_test,
                    input_shape, x_steps, y_steps,
                    weight_file=None, history_file=None,
                    log_dir='./outputs/logs', **kwargs
                    ):
    wn = ts_model.WaveNet()
    model = wn.build_model(input_shape=input_shape,
                           n_in_steps=x_steps,
                           n_out_steps=y_steps,
                           gated_activations=['relu', 'sigmoid'],
                           n_conv_filters=32)
    if weight_file:
        if os.path.isfile(weight_file):
            model.load_weights(weight_file)

    # Train
    early_stopper = EarlyStopping(monitor='loss',
                                  min_delta=0.01,
                                  patience=10,
                                  verbose=1,
                                  mode='auto')
    tensor_brd = TensorBoard(log_dir=log_dir,
                             histogram_freq=0)
    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=0.2,
                                  patience=5,
                                  min_lr=0.0001)

    # Root mean square error - which not defined in keras
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    model.compile(optimizer='nadam',
                  loss=rmse,
                  metrics=['mse', 'mae', 'mape', 'cosine'],
                  )
    validation_data = (x_test, y_test)
    history = model.fit(x_train, y_train,
                        validation_data=validation_data,
                        callbacks=[early_stopper, tensor_brd, reduce_lr],
                        **kwargs)

    if weight_file:
        model.save_weights(weight_file)
    if history_file:
        with open(history_file, 'wb') as f:
            pickle.dump(history.history, f)
    return model


def plot_history(history_file):
    from matplotlib import pyplot

    if not os.path.isfile(history_file):
        return
    with open(history_file, 'rb') as f:
        history = pickle.load(f)

    measures = ['loss',
                'mean_squared_error', 'mse',
                'mean_absolute_error', 'mae',
                'mean_absolute_percentage_error', 'mape',
                'cosine_proximity', 'cosine'
                ]
    for m in measures:
        if m in history:
            pyplot.plot(history[m])
    pyplot.show()


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
                _queues.append(pool.apply_async(_to_array, _args))

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
                    yield (np.stack(batch_x), np.stack(batch_y))


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


def _to_array(walk, train_df, group_col, indep_cols, dep_col):
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


def to_batch(walks, train_df, group_col, indep_cols, dep_col,
             use_multiprocessing=False):
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


def get_steps_per_epoch(df, group_col, x_steps, y_steps, batch_size=0, stride=1):
    """Approximation of steps.

     ~ it is approximated if df has different lengths
     """

    # assume that all group has same date_range
    n_groups = len(df[group_col].unique())
    _range = get_date_range(df.index, "D")

    # estimate number possible of N_samples per a ts (by group)
    n_samples = len(_range) - (x_steps + y_steps) + 1
    if stride > 1:
        n_samples = (n_samples // stride) + 1
    total_samples = (n_samples * n_groups)
    if batch_size == 0:
        raise ValueError("Batch size must be larger than 0")

    steps_per_epoch = (total_samples // batch_size // stride) + 1

    return steps_per_epoch


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


def read_numpy(save_dir):
    x_train = np.load("{}/x_train.npy".format(save_dir))
    y_train = np.load("{}/y_train.npy".format(save_dir))
    x_test = np.load("{}/x_test.npy".format(save_dir))
    y_test = np.load("{}/y_test.npy".format(save_dir))
    return x_train, y_train, x_test, y_test


class MTSSequence(keras.utils.Sequence):
    """Sequence class for Multivariate Series.

    Basically group_col is as an categorical which define each category is an mts.
    If we have more than 2 categorical -> combine all into one categorical as group-mts
    Currently support only one category in the data set

    A walk is a slice of data frame ~ 1 sample to feed into NN and walks are all generated
    slices through whole data set per group, which is len(walks) = len(samples in epoch)
    """

    # Todo: support sequence to use multi processing
    def __init__(self, train_df, indep_cols, dep_col,
                 x_steps, y_steps, batch_size=32,
                 group_col=None, stride=1, wait_per_batch=0.5,
                 batches=[]
                 ):
        # Todo: support when group_col is None ~ one mts
        self.train_df = train_df
        self.group_col = group_col
        self.indep_cols = indep_cols
        self.dep_col = dep_col
        self.wait_per_batch = wait_per_batch
        self.x_steps = x_steps
        self.y_steps = y_steps
        self.stride = stride
        self.batch_size = batch_size

        self.batches = batches

        # self._slicer = WalkForward(x_steps, y_steps, stride=self.stride)
        # self._walks = self._slicer.walk_through(self.train_df, self.group_col)

    def __getitem__(self, index):
        """Gets batch at position `index`.

        :param index: position of the batch in the Sequence.
        :return: a batch of numpy array shape ~ (n_steps, n_features)
        """
        batch = to_batch(self.batches[index], self.train_df,
                         self.group_col, self.indep_cols, self.dep_col,
                         )
        return batch

    def __len__(self):
        """Number of batch in the Sequence."""
        return len(self.batches)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            if item is not None:
                yield item


class MTSLoader(keras.utils.Sequence):
    def __init__(self, data, x_cols, y_cols,
                 x_steps, y_steps, stride=1,
                 group_col=None, batch_size=1,
                 ):
        self.data = data
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.group_col = group_col

        self.length = self._get_len(x_steps, y_steps, stride, batch_size)
        if self.length is None:
            raise ValueError("Time steps x {} and y {} "
                             "over each time series length".format(x_steps, y_steps))

        self.batch_size = batch_size
        self.x_input_shape = (x_steps, len(x_cols))
        self.x_batch_shape = (batch_size, x_steps, len(x_cols))

        self.items = self._get_samples(x_steps, y_steps, stride)
        self.lock = threading.Lock()

    def __getitem__(self, index):
        """Gets batch at position `index`.

        :param index: position of the batch in the Sequence.
        :return: a batch of numpy array shape ~ (n_steps, n_features)
        """
        with self.lock:
            batch_x = []
            batch_y = []

            batch = list(itertools.islice(self.items, self.batch_size))
            for x, y in batch:
                if len(x) > 0:
                    batch_x.append(x)
                    batch_y.append(y)

            if len(batch_x) > 0:
                return np.stack(batch_x), np.stack(batch_y)

    def __len__(self):
        """Number of batch in the Sequence."""
        return int(self.length)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            if item is not None:
                yield item

    def _get_len(self, x_steps, y_steps, stride=1, batch_size=1):
        # Go through data set
        total_samples = 0
        for group, _ts in self.data.groupby(self.group_col):
            size = len(_ts.index)
            if size >= x_steps + y_steps + 1:
                samples = 1 + (size - x_steps - y_steps) // stride
                total_samples += samples if samples > 0 else 0

        total_samples = total_samples / batch_size // 1  # floor
        if total_samples > 0:
            return total_samples

    def _get_samples(self, x_steps, y_steps, stride=1):

        for group, _ts in self.data.groupby(self.group_col):
            indep_data = _ts[self.x_cols]
            dep_data = _ts[self.y_cols]
            size = len(_ts.index)

            if size < x_steps + y_steps + 1:
                break
            i = 0
            while True:
                _x_start = i * stride
                _x_end = _x_start + x_steps
                _y_start = _x_end
                _y_end = _x_end + y_steps

                # try to get last item
                if _y_end > size:
                    _y_end = size
                    _y_start = size - y_steps
                    _x_end = _y_start
                    _x_start = _x_end - x_steps

                if _x_start < 0 or _y_end > size:
                    break
                _sample_x = indep_data[_x_start:_x_end].to_numpy().round(3)
                _sample_y = dep_data[_y_start:_y_end].to_numpy().round(3)

                yield _sample_x, _sample_y

                if _y_end == size:
                    break
                i = i + 1
