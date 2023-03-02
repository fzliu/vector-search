import numpy as np

from ._base import _BaseIndex


class ScalarQuantizer(_BaseIndex):

    def __init__(self):
        self._index = None
        self._starts = None
        self._steps = None

    def create(self, dataset):
        """Calculates and stores SQ parameters based on the input dataset."""
        self._starts = np.min(dataset)
        self._steps = (np.max(dataset) - self._starts) / 255
        self._index = np.uint8((dataset - self._starts) / self._steps)

    def search(self, vector, nq=10):
        """Performs quantization + naive search."""
        quantized = self.quantize(vector)
        return super().search(quantized, nq)

    def quantize(self, vector):
        """Quantizes the input vector based on SQ parameters"""
        return np.uint8((vector - self._starts) / self._steps)

    def restore(self, vector):
        """Restores the original vector using SQ parameters."""
        return (vector * self._steps) + self._starts


if __name__ == "__main__":
    sq = ScalarQuantizer()
    sq.create(np.random.randn(1000, 256))
    print(sq.search(np.random.randn(256)))