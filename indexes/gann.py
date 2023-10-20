"""
gann.py: Good (Great/Gangster/Godlike) Approximate Nearest Neighbors.
"""

from timeit import default_timer
from typing import List, Optional, Tuple
import random

import numpy as np


def perfect_split(vecs: List[np.ndarray]) ->  Tuple[np.ndarray]:
    """Returns reference points which split the input vectors perfectly.
    """
    pass


class Node(object):
    """Initialize with a set of vectors, then call `split()`.
    """

    def __init__(self, ref: np.ndarray, vecs: List[np.ndarray]):
        self._ref = ref
        self._vecs = vecs
        self._left = None
        self._right = None

    @property
    def ref(self) -> Optional[np.ndarray]:
        """Reference point in n-d hyperspace. Evaluates to `False` if root node.
        """
        return self._ref

    @property
    def vecs(self) -> List[np.ndarray]:
        """Vectors for this leaf node. Evaluates to `False` if not a leaf.
        """
        return self._vecs

    @property
    def left(self) -> Optional[object]:
        """Left node.
        """
        return self._left

    @property
    def right(self) -> Optional[object]:
        """Right node.
        """
        return self._right

    def split(self, K: int, imb: float) -> bool:

        # stopping condition: maximum # of vectors for a leaf node
        if len(self._vecs) <= K:
            return False

        # continue for a maximum of 5 iterations
        for n in range(5):
            left_vecs = []
            right_vecs = []

            # take two random indexes and set as left and right halves
            left_ref = self._vecs.pop(np.random.randint(len(self._vecs)))
            right_ref = self._vecs.pop(np.random.randint(len(self._vecs)))

            # split vectors into halves
            for vec in self._vecs:
                dist_l = np.linalg.norm(vec - left_ref)
                dist_r = np.linalg.norm(vec - right_ref)
                if dist_l < dist_r:
                    left_vecs.append(vec)
                else:
                    right_vecs.append(vec)

            # check to make sure that the tree is mostly balanced
            r = len(left_vecs) / len(right_vecs)
            r = len(left_vecs) / len(self._vecs)
            #print(r)
            if r < imb and r > (1 - imb):
                self._left = Node(left_ref, left_vecs)
                self._right = Node(right_ref, right_vecs)
                return True

            # redo tree build process if imbalance is high

        print("fuck")
        return False


def _select_nearby(node: Node, q: np.ndarray, thresh: int = 0):
    """Functions identically to _is_query_in_left_half, but can return both.
    """
    if not node.left or not node.right:
        return ()
    dist_l = np.linalg.norm(q - node.left.ref)
    dist_r = np.linalg.norm(q - node.right.ref)
    if np.abs(dist_l - dist_r) < thresh:
        return (node.left, node.right)
    if dist_l < dist_r:
        return (node.left,)
    return (node.right,)


def _build_tree(node, K: int, imb: float):
    """Recurses on left and right halves to build a tree.
    """
    node.split(K=K, imb=imb)
    if node.left and node.right:
        _build_tree(node.left, K=K, imb=imb)
        _build_tree(node.right, K=K, imb=imb)


def build_forest(vecs: List[np.ndarray], N: int = 8, K: int = 64, imb: float = 0.95) -> List[Node]:
    """Builds a forest of `N` trees.
    """
    forest = []
    for _ in range(N):
        root = Node(None, vecs)
        _build_tree(root, K, imb)
        forest.append(root)
    return forest


def _query_linear(vecs: List[np.ndarray], q: np.ndarray, k: int) -> List[np.ndarray]:
    vecs = np.array(vecs)
    idxs = np.argsort(np.linalg.norm(vecs - q, axis=1))
    vecs = vecs[idxs][:k]
    return list(vecs)


def _query_tree(root: Node, q: np.ndarray, k: int) -> List[np.ndarray]:
    """Queries a single tree.
    """

    pq = [root]
    nns = []
    while pq:
        # iteratively determine whether right or left node is closer
        node = pq.pop(0)
        nearby = _select_nearby(node, q, thresh=1e-2)
        if nearby:
            pq.extend(nearby)
        else:
            nns.extend(node.vecs)

    # brute-force search the nearest neighbors
    return _query_linear(nns, q, k)


def query_forest(forest: List[Node], q, k: int = 10):
    nns = []
    for root in forest:
        nns.extend(_query_tree(root, q, k))
    return _query_linear(nns, q, k)


if __name__ == "__main__":

    # create dataset
    N = 100000
    d = 128
    k = 10
    dataset = np.random.random((N, d))
    dataset = [np.random.random(d) for _ in range(N)]

    # create query vector
    query = np.random.random(128)

    # create index
    index = build_forest(dataset)

    # perform query
    start = default_timer()
    result = query_forest(index, query, k=k)
    print(default_timer() - start)

    # brute-force ground truth
    start = default_timer()
    actual = _query_linear(dataset, query, k=k)
    print(default_timer() - start)

    # determine top-k recall
    r = 0
    for vec in actual:
        for vec2 in result:
            if np.all(vec == vec2):
                r += 1
                break

    # print the top-k for each 

    print(f"recall: {r}/{k}")


