import numpy as np


class ScalarQuantizer:

    def __init__(self):
        self._dataset = None

    def create(self):
        """Calculates and stores SQ parameters based on the input dataset."""
        self._dtype = dataset.dtype  # original dtype
        self._starts = np.min(dataset, axis=1)
        self._steps = (np.max(dataset, axis=1) - self._starts) / 255

        # the internal dataset uses `uint8_t` quantization
        self._dataset = np.uint8((dataset - self._starts) / self._steps)

    def quantize(self, vector):
        """Quantizes the input vector based on SQ parameters"""
        return np.uint8((vector - self._starts) / self._steps)

    def restore(self, vector):
        """Restores the original vector using SQ parameters."""
        return (vector * self._steps) + self._starts

    @property
    def dataset(self):
        if self._dataset:
            return self._dataset
        raise ValueError("Call ScalarQuantizer.create() first")

