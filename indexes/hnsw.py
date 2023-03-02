from bisect import insort
from heapq import heapify, heappop, heappush

import numpy as np

from ._base import _BaseIndex


class HNSW(_BaseIndex):

    def __init__(self, L=5, mL=0.62, efc=10):
        self._L = L
        self._mL = mL
        self._efc = efc
        self._index = [[] for _ in range(L)]

    @staticmethod
    def _search_layer(graph, entry, query, ef=1):

        best = (np.linalg.norm(graph[entry][0] - query), entry)

        nns = [best]
        visit = set(best)  # set of visited nodes
        candid = [best]  # candidate nodes to insert into nearest neighbors
        heapify(candid)

        # find top-k nearest neighbors
        while candid:
            cv = heappop(candid)

            if nns[-1][0] > cv[0]:
                break

            # loop through all nearest neighbors to the candidate vector
            for e in graph[cv[1]][1]:
                d = np.linalg.norm(graph[e][0] - query)
                if (d, e) not in visit:
                    visit.add((d, e))

                    # push only "better" vectors into candidate heap
                    if d < nns[-1][0] or len(nns) < ef:
                        heappush(candid, (d, e))
                        insort(nns, (d, e))
                        if len(nns) > ef:
                            nns.pop()

        return nns

    def create(self, dataset):
        for v in dataset:
            self.insert(v)

    def search(self, query, ef=1):

        # if the index is empty, return an empty list
        if not self._index[0]:
            return []

        best_v = 0  # set the initial best vertex to the entry point
        for graph in self._index:
            best_d, best_v = HNSW._search_layer(graph, best_v, query, ef=1)[0]
            if graph[best_v][2]:
                best_v = graph[best_v][2]
            else:
                return HNSW._search_layer(graph, best_v, query, ef=ef)

    def _get_insert_layer(self):
        # ml is a multiplicative factor used to normalize the distribution
        l = -int(np.log(np.random.random()) * self._mL)
        return min(l, self._L-1)

    def insert(self, vec, efc=10):

        # if the index is empty, insert the vector into all layers and return
        if not self._index[0]:
            i = None
            for graph in self._index[::-1]:
                graph.append((vec, [], i))
                i = 0
            return

        l = self._get_insert_layer()

        start_v = 0
        for n, graph in enumerate(self._index):

            # perform insertion for layers [l, L) only
            if n < l:
                _, start_v = self._search_layer(graph, start_v, vec, ef=1)[0]
            else:
                node = (vec, [], len(self._index[n+1]) if n < self._L-1 else None)
                nns = self._search_layer(graph, start_v, vec, ef=efc)
                for nn in nns:
                    node[1].append(nn[1])  # outbound connections to NNs
                    graph[nn[1]][1].append(len(graph))  # inbound connections to node
                graph.append(node)

            # set the starting vertex to the nearest neighbor in the next layer
            start_v = graph[start_v][2]


if __name__ == "__main__":
    hnsw = HNSW()
    hnsw.create(np.random.randn(1000, 256))
    print(hnsw.search(np.random.randn(256)))



