"""Test module for temporalnn"""
import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf

from temporalnn.utils import ts as ts_util
from temporalnn.utils.ts_trainer import train

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

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


def _bejin_dataset():
    from datetime import datetime
    data_link = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv"
    df = pd.read_csv(data_link,
                     parse_dates=[['year', 'month', 'day', 'hour']],
                     index_col=0,
                     date_parser=lambda x: datetime.strptime(x, '%Y %m %d %H'),
                     )
    df.index.name = 'date'
    df = df[24:]  # drop first 24 hours (NA data)
    df = df.drop('No', axis=1)  # drop index col
    df = df.rename({'pm2.5': 'pollution',
                    'DEWP': 'dew',
                    'TEMP': 'temp',
                    'PRES': 'press',
                    'cbwd': 'wind_direction',
                    'Iws': 'wind_speed',
                    'Is': 'snow',
                    'Ir': 'rain'
                    }, axis=1)
    df['pollution'] = df['pollution'].interpolate()

    df.to_csv(f"{DATA_DIR}/pollution.csv")


def _generate_train_test_on_bejin_np_dataset():
    from sklearn.preprocessing import LabelEncoder

    data_link = f"{DATA_DIR}/pollution.csv"
    df = pd.read_csv(data_link)

    # Set date-time as index
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date")

    # Normalization
    numerical_df = df[["dew", "temp", "press", "wind_speed", "snow", "rain", "pollution"]]
    normalized_df = (numerical_df - numerical_df.min()) / (numerical_df.max() - numerical_df.min())

    # Encoding wind_direction to integer
    encoder = LabelEncoder()
    normalized_df["wind_direction"] = encoder.fit_transform(df["wind_direction"])
    normalized_df["wind_direction"] = normalized_df["wind_direction"].astype("float")

    # To decode
    # list(encoder.inverse_transform([2, 2, 1]))

    # Create train/test data
    x_train, x_test, y_train, y_test = ts_util.df_to_ts_numpy(
        data=normalized_df,
        independents=["dew", "temp", "press", "wind_direction",
                      "wind_speed", "snow", "rain"],
        dependent="pollution",
        n_in_steps=128,
        n_out_steps=1,
        stride=1,
        split_test=True,
        test_size=0.3,
        random_state=42
    )

    assert x_train.ndim == 3 and x_train.shape[1] == 128
    assert x_test.shape[1] == 128

    return x_train, x_test, y_train, y_test


def test_train_wavenet_on_bejin_dataset():
    x_train, x_test, y_train, y_test = _generate_train_test_on_bejin_np_dataset()
    model_file = f"{DATA_DIR}/wavenet_mts_128_1.h5"
    log_dir = f"{DATA_DIR}/log-wavenet"
    history_file = f"{DATA_DIR}/log-wavenet/history/wavenet_train_history.h5"
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    train(x_train, y_train, x_test, y_test,
          model_file=model_file,
          history_file=history_file,
          log_dir=log_dir,
          epochs=50,
          batch_size=128
          )


def test_train_lstm_on_bejin_dataset():
    import pickle
    from temporalnn.models.temporal import SimpleLSTM
    from keras.callbacks import (EarlyStopping,
                                 TensorBoard,
                                 ReduceLROnPlateau)

    x_train, x_test, y_train, y_test = _generate_train_test_on_bejin_np_dataset()
    model_file = f"{DATA_DIR}/lstm_mts_128_1.h5"
    log_dir = f"{DATA_DIR}/log-lstm"
    history_file = f"{DATA_DIR}/log-lstm/history/lstm_train_history.h5"
    os.makedirs(os.path.dirname(history_file), exist_ok=True)

    # Train
    early_stopper = EarlyStopping(monitor='loss',
                                  min_delta=0.001,
                                  patience=10,
                                  verbose=1,
                                  mode='auto')
    tensor_brd = TensorBoard(log_dir=log_dir,
                             histogram_freq=0)
    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=0.2,
                                  patience=5,
                                  min_lr=0.0001)
    model = SimpleLSTM.build_model(input_shape=x_train.shape[1:])
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        callbacks=[early_stopper, tensor_brd, reduce_lr],
                        epochs=50,
                        batch_size=64
                        )

    if model_file:
        model.save(model_file)
    if history_file:
        with open(history_file, 'wb') as wf:
            pickle.dump(history.history, wf)
