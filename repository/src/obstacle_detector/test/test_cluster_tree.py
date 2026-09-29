"""KD-tree components must match an exact distance check, including borders and duplicates."""
import unittest

import numpy as np

from obstacle_detector.detector import cluster_labels_by_distance


def _partition(labels):
    groups = {}
    for index, label in enumerate(np.asarray(labels).tolist()):
        groups.setdefault(int(label), set()).add(index)
    return frozenset(frozenset(members) for members in groups.values())


def _brute(points, limit):
    n = int(points.shape[0])
    parent = list(range(n))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    limit2 = float(limit) * float(limit)
    for i in range(n):
        for j in range(i + 1, n):
            delta = points[i] - points[j]
            if float(delta @ delta) <= limit2:
                parent[find(j)] = find(i)
    roots = [find(i) for i in range(n)]
    return _partition(roots)


class ClusterTreeTests(unittest.TestCase):
    def test_border_duplicate_and_dense_cell_match_exact_distance(self):
        rng = np.random.default_rng(4)
        cloud = rng.normal(size=(36, 3))
        cloud[1] = cloud[0]
        cloud[2] = cloud[0] + np.array([0.15, 0.0, 0.0])
        cloud[3] = cloud[0] + np.array([0.1500000000001, 0.0, 0.0])
        dense = rng.normal(scale=0.04, size=(12, 3))
        points = np.vstack([cloud, dense])
        labels = cluster_labels_by_distance(points[:, 0], points[:, 1], points[:, 2], 0.15, 0.15)
        self.assertEqual(_partition(labels), _brute(points, 0.15))


if __name__ == "__main__":
    unittest.main()
