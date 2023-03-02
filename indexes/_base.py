import queue

import numpy as np


class _BaseIndex:

    def __init__(self):
        self._index = None

    def create(self, dataset):
        """Index builder."""
        raise NotImplementedError()

    def insert(self, vector):
        """Insert a single vector."""
        raise NotImplementedError()

    def search(self, vector, nq=10):
        """Naive (flat) search."""
        nns = queue.PriorityQueue()  # should probably use heapq
        for (n, v) in enumerate(self._index):
            d = -np.linalg.norm(v - vector)
            if nns.qsize() == 0 or d > nns.queue[0][0]:
                nns.put((d, n))
                if nns.qsize() > nq:
                    nns.get()
        out = []
        for n in range(nq):
            if nns.empty():
                break
            out.insert(0, nns.get())
        return out

    @property
    def index(self):
        if self._index:
            return self._index
        raise ValueError("Call create() first")
