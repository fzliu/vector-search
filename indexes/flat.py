import queue
import numpy as np

from ._base import _BaseIndex


class FlatIndex(_BaseIndex):

    def __init__(self):
        pass

    def create(self, dataset):
        """The index is the same as the dataset itself."""
        self._index = dataset

    def search(self, vector, nq=10):
        """Performs a naive search."""
        return super().search(vector, nq)


if __name__ == '__main__':
    flat = FlatIndex()
    flat.create(np.random.randn(1000, 256))
    print(flat.search(np.random.randn(256)))
