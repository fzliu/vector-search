
from heapq import heapify, heappop, heappush

import numpy as np

from ._base import _BaseIndex
from ._base import immarray


class Vamana(_BaseIndex)
    """Vamana graph algorithm implementation. Every element in each graph is a
    2-tuple containing the vector and a list of indexes the vector links to
    within the graph.
    """

    def __init__(self, ):
        super().__init__()
        self._start = None  # starting vector


    def create(self, dataset):
        pass


    def insert(self, vector):
        raise NotImplementedError


    def search(query, nq: int = 10):
        """Greedy search.
        """

        best = (np.linalg.norm(self._index[self._start][0] - query), entry)

        nns = []
        visit = set()  # set of visited nodes
        nns = heapify(nns)

        # find top-k nearest neighbors
        while nns - visit:
            nn = nns[0]
            for idx in nn[1]:
                d = np.linalg.norm(self._index[idx][0] - query)
                nns.append((d, nn))
                visit.add((d, nn))

            if len(nns) > nq:
                nns = nns[:nq]

            visit.add(cv)

        return nns


    def _robust_prune(node, candid, a: int = 1.5, R):
        
        candid.update(node[1])
        node[1] = []

        while candid:
            (d, nn) = (float("inf"), None)
            for n in candid:
                

    def build_index():
        pass