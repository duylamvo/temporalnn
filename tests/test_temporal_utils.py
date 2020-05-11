"""Test module for temporalnn"""

import pandas as pd
import json
import numpy as np
import os

from temporalnn.utils import ts as ts_util
from temporalnn.utils.ts_trainer import train

DATA_DIR = "tests/data" if os.path.isdir("tests") else "data"

with open(f"{DATA_DIR}/climate_small_schema.json") as f:
    schema = json.load(f)

climate_data = pd.read_csv(f"{DATA_DIR}/climate_small.csv")
climate_data = climate_data.astype(schema)


def test_schema_climate_small():
    assert ("tmk" in schema
            and "fx" in schema
            and "measure_date" in schema
            and "stations_id" in schema
            )
    assert (climate_data.measure_date.dtype == '<M8[ns]'
            and climate_data.stations_id.dtype == np.dtype('int'))


def test_generate_ts_numpy_data():
    df = climate_data.copy()
    df.set_index("measure_date")

    dependent_columns = ["fx"]
    independent_column = "tmk"
    group_col = "stations_id"

    x_steps = 7
    y_steps = 2
    stride = 1
    test_set = ts_util.df_to_ts_numpy(df,
                                      dependent_columns,
                                      independent_column,
                                      group_col,
                                      x_steps, y_steps,
                                      stride, split_test=False)
    x_train, y_train = test_set
    # x should have 3 dimensions (samples, ts_steps, dimensions)
    # y should have 2 dimensions (samples, ts_steps)
    assert x_train.ndim == 3 and y_train.ndim == 2
    n_samples, n_steps, n_dim = x_train.shape
    assert (n_samples > 0 and
            n_dim == len(dependent_columns) and
            n_steps == x_steps
            )

    n_samples, n_steps = y_train.shape
    assert (n_samples > 0 and
            n_steps == y_steps
            )


def test_split_train_test_ts():
    df = climate_data.copy()
    df.set_index("measure_date")

    dependent_columns = ["fx"]
    independent_column = "tmk"
    group_col = "stations_id"

    x_steps = 7
    y_steps = 2
    stride = 1
    x_train, x_test, y_train, y_test = ts_util.df_to_ts_numpy(
        df, dependent_columns,
        independent_column, group_col,
        x_steps, y_steps, stride,
        split_test=True, test_size=0.3, random_state=42)

    assert x_train.ndim == 3 and y_train.ndim == 2
    assert x_test.ndim == 3 and y_test.ndim == 2

    n_samples, n_steps, n_dim = x_train.shape
    assert (n_samples > 0 and
            n_dim == len(dependent_columns) and
            n_steps == x_steps
            )

    n_samples, n_steps, n_dim = x_test.shape
    assert (n_samples > 0 and
            n_dim == len(dependent_columns) and
            n_steps == x_steps
            )


def test_train_on_wavenet():
    df = climate_data.copy()
    df.set_index("measure_date")

    dependent_columns = ["fx"]
    independent_column = "tmk"
    group_col = "stations_id"

    x_steps = 7
    y_steps = 2
    stride = 1
    x_train, x_test, y_train, y_test = ts_util.df_to_ts_numpy(
        df, dependent_columns, independent_column, group_col,
        x_steps, y_steps, stride,
        split_test=True, test_size=0.3, random_state=42)

    train(x_train, y_train, x_test, y_test)


def test_train_and_save_model_on_wavenet(tmpdir):
    df = climate_data.copy()
    df.set_index("measure_date")

    dependent_columns = ["fx"]
    independent_column = "tmk"
    group_col = "stations_id"

    x_steps = 7
    y_steps = 2
    stride = 1
    x_train, x_test, y_train, y_test = ts_util.df_to_ts_numpy(
        df, dependent_columns, independent_column, group_col,
        x_steps, y_steps, stride,
        split_test=True, test_size=0.3, random_state=42)

    import os
    weight_file = tmpdir + "/" + "test.h5"
    train(x_train, y_train, x_test, y_test, weight_file=weight_file)
    assert os.path.isfile(weight_file)
