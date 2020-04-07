"""Module to support distributed using dask"""
import keras
from dask.distributed import Client


class MtsDaskSequence(keras.utils.Sequence):
    """ A modification of Sequence class for Multivariate Series.
    """

    def __init__(self, scheduler, port):
        self.client = Client(scheduler + ":" + port)
        self.futures = []

    def prepare_futures(self, data, batch_indices, fn, *args, **kwargs):
        for b in batch_indices:
            [df_future] = self.client.scatter([data], broadcast=True)

            fu = self.client.submit(fn, b, df_future, *args, **kwargs)
            self.futures.append(fu)

    def __getitem__(self, index):
        """Gets batch at position `index`.

        :param index: position of the batch in the Sequence.
        :return: a batch of numpy array shape ~ (n_steps, n_features)
        """
        return self.futures[index].result()

    def __len__(self):
        """Number of batch in the Sequence."""
        return len(self.futures)
