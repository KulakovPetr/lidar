import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

from flag_score import score_part


def _candidate(candidate_id, rows, range_m):
    return {"candidate_id": candidate_id, "kind": "large", "branch": "volume", "rows": np.asarray(rows), "range_from_lidar_m": range_m}


def _detected(candidates, inside):
    return {
        "masks": {"inside": np.asarray(inside, dtype=bool)},
        "candidates_large": candidates,
        "candidates_rare": [],
        "candidates_protrusion_large": [],
        "candidates_protrusion_rare": [],
    }


class FlagScoreTests(unittest.TestCase):
    def test_cloth_and_pole_may_be_different_candidates(self):
        detected = _detected(
            [_candidate("L1", [0, 1], 10.0), _candidate("L2", [2, 3], 10.2)],
            [True, True, True, True],
        )
        cloth = score_part(detected, [0, 1], np.array([10.0, 10.1, 10.2, 10.3]), [0, 1, 2, 3])
        pole = score_part(detected, [2, 3], np.array([10.0, 10.1, 10.2, 10.3]), [0, 1, 2, 3])
        self.assertTrue(cloth["localized"])
        self.assertTrue(pole["localized"])
        self.assertEqual(cloth["localizing_ids"], ["L1"])
        self.assertEqual(pole["localizing_ids"], ["L2"])
        self.assertEqual(cloth["duplicate_candidates"], 0)

    def test_foreign_background_is_not_localization(self):
        detected = _detected([_candidate("L1", [0, 4, 5, 6], 8.0)], [True, False, False, False, False, False, False])
        scored = score_part(detected, [0], np.full(7, 10.0), [0, 1])
        self.assertFalse(scored["localized"])
        self.assertTrue(scored["merged_with_background"])

    def test_duplicate_of_one_part_is_counted(self):
        detected = _detected(
            [_candidate("L1", [0, 1], 10.0), _candidate("L2", [0, 1], 10.1)],
            [True, True],
        )
        scored = score_part(detected, [0, 1], np.array([10.0, 10.2]))
        self.assertEqual(scored["duplicate_candidates"], 1)
        self.assertTrue(scored["localized"])


if __name__ == "__main__":
    unittest.main()
