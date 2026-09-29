import unittest

import numpy as np

from obstacle_detector.display_cloud import highlight_rows, sample_display
from obstacle_detector.playback import pair_arrival


class PlaybackPairTests(unittest.TestCase):
    def test_cloud_then_token_keeps_sequence_when_stamps_match(self):
        cloud_a = {"stamp": 5, "name": "a"}
        cloud_b = {"stamp": 5, "name": "b"}
        pair, token, cloud = pair_arrival(None, None, cloud=cloud_a)
        self.assertIsNone(pair)
        pair, token, cloud = pair_arrival(token, cloud, token={"sequence": 0, "stamp": 5})
        self.assertEqual(pair[0]["name"], "a")
        self.assertEqual(pair[1]["sequence"], 0)
        pair, token, cloud = pair_arrival(token, cloud, cloud=cloud_b)
        self.assertIsNone(pair)
        pair, token, cloud = pair_arrival(token, cloud, token={"sequence": 1, "stamp": 5})
        self.assertEqual(pair[0]["name"], "b")
        self.assertEqual(pair[1]["sequence"], 1)

    def test_token_then_cloud(self):
        pair, token, cloud = pair_arrival(None, None, token={"sequence": 4})
        pair, token, cloud = pair_arrival(token, cloud, cloud={"name": "c"})
        self.assertEqual(pair[1]["sequence"], 4)
        self.assertEqual(pair[0]["name"], "c")
        self.assertIsNone(token)
        self.assertIsNone(cloud)


class DisplaySampleTests(unittest.TestCase):
    def test_highlight_points_survive_the_stride(self):
        xyz = np.zeros((10000, 3), dtype=np.float32)
        xyz[:, 0] = np.arange(10000)
        highlight = np.array([3, 17, 5000, 9999], dtype=np.int64)
        chosen, labels = sample_display(xyz, highlight, background_limit=100)
        marked = set(chosen[labels > 0.5, 0].astype(int).tolist())
        self.assertEqual(marked, {3, 17, 5000, 9999})
        self.assertLess(chosen.shape[0], 200)

    def test_timeout_record_has_no_highlight(self):
        rows = highlight_rows({"conditional_intrusions": [], "candidates": None, "status": "processing_timeout"})
        self.assertEqual(rows.size, 0)
        chosen, labels = sample_display(np.zeros((10, 3), dtype=np.float32), rows, 4)
        self.assertTrue(np.all(labels == 0))

    def test_highlight_uses_candidate_rows_when_the_card_has_none(self):
        record = {
            "conditional_intrusions": [
                {"candidate_id": "L1"},
                {"candidate_id": "L2"},
            ],
            "candidates": [
                {"candidate_id": "L1", "source_indices": [4, 5]},
                {"candidate_id": "L9", "source_indices": [100]},
                {"candidate_id": "L2", "source_indices": [7]},
            ],
        }
        self.assertEqual(set(highlight_rows(record).tolist()), {4, 5, 7})
        xyz = np.zeros((10000, 3), dtype=np.float32)
        xyz[:, 0] = np.arange(10000)
        chosen, labels = sample_display(xyz, highlight_rows(record), background_limit=100)
        marked = set(chosen[labels > 0.5, 0].astype(int).tolist())
        self.assertEqual(marked, {4, 5, 7})

    def test_every_accepted_cluster_is_kept(self):
        record = {
            "conditional_intrusions": [
                {"source_indices": [1, 2]},
                {"source_indices": [8]},
            ]
        }
        rows = highlight_rows(record)
        self.assertEqual(set(rows.tolist()), {1, 2, 8})


if __name__ == "__main__":
    unittest.main()
