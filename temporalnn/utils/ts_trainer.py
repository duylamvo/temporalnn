import keras
import pickle
import itertools
import os
import threading
import numpy as np
import tempfile

from keras import backend as K
from keras.callbacks import (EarlyStopping,
                             TensorBoard,
                             ReduceLROnPlateau)

from ..models import temporal as ts_model
from .ts import to_batch


def train_on_generator(batches, input_shape, x_steps, y_steps,
                       weight_file=None, history_file=None,
                       log_dir='./logs', **kwargs
                       ):
    wn = ts_model.WaveNet()
    model = wn.build_model(input_shape=input_shape,
                           x_steps=x_steps,
                           y_steps=y_steps,
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
                                  callbacks=[early_stopper,
                                             tensor_brd, reduce_lr],
                                  **kwargs)

    if weight_file:
        model.save_weights(weight_file)
    if history_file:
        with open(history_file, 'wb') as f:
            pickle.dump(history.history, f)
    return model


def train(x_train, y_train, x_test, y_test,
          # input_shape=None, x_steps=None, y_steps=None,
          weight_file=None, history_file=None,
          log_dir=None, **kwargs
          ):
    wn = ts_model.WaveNet()

    # Extract input shape

    assert isinstance(x_train, np.ndarray) and x_train.ndim == 3
    assert isinstance(y_train, np.ndarray) and y_train.ndim == 2    # Todo support multi-dim output?
    _, x_steps, x_dims = x_train.shape
    _, y_steps = y_train.shape
    input_shape = (x_steps, x_dims)

    model = wn.build_model(input_shape=input_shape,
                           x_steps=x_steps,
                           y_steps=y_steps,
                           gated_activations=['relu', 'sigmoid'],
                           n_conv_filters=32)
    if weight_file:
        if os.path.isfile(weight_file):
            model.load_weights(weight_file)
    log_dir = log_dir or tempfile.mkdtemp()
    os.makedirs(log_dir, exist_ok=True)

    # Train
    early_stopper = EarlyStopping(monitor='loss',
                                  min_delta=0.01 or kwargs.get("min_delta"),
                                  patience=10 or kwargs.get("patience"),
                                  verbose=1 or kwargs.get("verbose"),
                                  mode='auto' or kwargs.get("mode"))

    tensor_brd = TensorBoard(log_dir=log_dir,
                             histogram_freq=0)
    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=0.2 or kwargs.get("factor"),
                                  patience=5 or kwargs.get("patience"),
                                  min_lr=0.0001 or kwargs.get("min_lr"))

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
