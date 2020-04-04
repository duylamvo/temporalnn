"""Datasets Tools"""
import os
import tqdm
import numpy as np
import multiprocessing as mp


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


def gen_train(data, x_cols, y_col, group_col,
              x_steps=7, y_steps=1, stride=1,
              save=False, save_dir=None):
    pool = mp.Pool(mp.cpu_count())
    queues = []
    if isinstance(x_cols, str):
        x_cols = [x_cols]
    for group, _ts in data.groupby(group_col):
        in_dep_data = _ts[x_cols]
        dep_data = _ts[y_col]

        args = [in_dep_data, dep_data, x_steps, y_steps, stride]
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
    _expected_shape = (x_steps, len(x_cols))
    assert _shape == _expected_shape, \
        "Shape {} does not match {}".format(_shape, _expected_shape)
    assert len(_X) == len(_Y)

    x_data = np.stack(_X)
    y_data = np.stack(_Y)

    print("final data", x_data.shape, y_data.shape)
    if save:
        if not save_dir:
            save_dir = "data/train"
        save_dir += "/train_numpy_{}_{}".format(x_steps, y_steps)
        save_dir = save_dir.replace("//", "/")
        os.makedirs(save_dir, exist_ok=True)

        np.save("{}/x_data".format(save_dir), x_data)
        np.save("{}/y_data".format(save_dir), y_data)
    return x_data, y_data


def test_gen_data():
    import pandas as pd
    ds = pd.read_csv("data/climate/preprocessed_climate.csv")
    ds.measure_date = pd.to_datetime(ds.measure_date)
    ds = ds.set_index("measure_date")
    # sub_ds = ds[ds.index.year > 2016]
    sub_ds = ds
    x_train, y_train = gen_train(sub_ds, "tmk", "tmk", "stations_id",
                                 x_steps=32,
                                 save=True, save_dir="data/train/uts")
