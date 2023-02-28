import numpy as np
from scipy.cluster.vq import kmeans2


class ProductQuantizer(object):

    def __init__(self, M=16, K=256):
        self.M = 16
        self.K = 256
        self._dataset = None

    def create(self, dataset):
        """Fits PQ model based on the input dataset."""
        sublen = dataset.shape[1] // self.M
        self._centroids = np.empty((self.M, self.K, sublen), dtype=np.float64)
        self._dataset = np.empty((dataset.shape[0], self.M), dtype=np.uint8)
        for m in range(self.M):
            subspace = dataset[:,m*sublen:(m+1)*sublen]
            (centroids, assignments) = kmeans2(subspace, self.K, iter=32)
            self._centroids[m,:,:] = centroids
            self._dataset[:,m] = np.uint8(assignments)

    def quantize(self, vector):
        """Quantizes the input vector based on PQ parameters"""
        quantized = np.empty((self.M,), dtype=np.uint8)
        for m in range(self.M):
            centroids = self._centroids[m,:,:]
            distances = np.linalg.norm(vector - centroids, axis=1)
            quantized[m] = np.argmin(distances)
        return quantized

    def restore(self, vector):
        """Restores the original vector using PQ parameters."""
        return np.hstack([self._centroids[m,vector[m],:] for m in range(M)])

    @property
    def dataset(self):
        if self._dataset:
            return self._dataset
        raise ValueError("Call ProductQuantizer.create() first")
