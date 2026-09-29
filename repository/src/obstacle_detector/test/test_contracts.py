import unittest

from obstacle_detector.contracts import (
    MAX_QUEUE_DEPTH,
    Candidate,
    Corridor,
    Frame,
    PathHypothesis,
    RouteSelection,
    check_queue_depth,
    unevaluated_diagnostics,
)


class ContractTests(unittest.TestCase):
    def test_frame_allows_missing_ring_time_imu(self) -> None:
        frame = Frame(
            sequence=0,
            frame_id="lidar",
            stamp_ns=None,
            point_count=10,
            source_bag="for_hackathon/doubleT_platform",
            source_topic="/lidar_points",
            has_intensity=True,
        )
        self.assertFalse(frame.has_ring)
        self.assertFalse(frame.has_point_time)
        self.assertFalse(frame.has_imu)
        self.assertIsNone(frame.stamp_ns)

    def test_tunnel_center_is_not_a_path(self) -> None:
        for origin in ("tunnel_center", "centerline", "tunnel_axis"):
            with self.assertRaises(ValueError):
                PathHypothesis(hypothesis_id="p0", frame_sequence=0, origin=origin)

    def test_supported_length_has_no_default_horizon(self) -> None:
        path = PathHypothesis(hypothesis_id="p0", frame_sequence=0, origin="unknown")
        self.assertIsNone(path.supported_length_m)
        self.assertIsNone(path.quality)

    def test_corridor_gauge_is_unknown(self) -> None:
        corridor = Corridor(path_id="p0")
        self.assertIsNone(corridor.half_width_m)
        self.assertIsNone(corridor.s_far_m)

    def test_score_without_calibration_is_not_a_probability(self) -> None:
        candidate = Candidate(
            candidate_id="c0",
            frame_sequence=0,
            point_count=12,
            geometric_score=0.8,
        )
        self.assertFalse(candidate.score_is_probability)
        self.assertEqual(candidate.score_label, "uncalibrated_geometric_score")

    def test_calibrated_score_needs_an_id(self) -> None:
        candidate = Candidate(
            candidate_id="c0",
            frame_sequence=0,
            point_count=12,
            geometric_score=0.8,
            calibration_id="dev-set-v1",
        )
        self.assertTrue(candidate.score_is_probability)
        self.assertEqual(candidate.score_label, "calibrated_probability")

    def test_unevaluated_diagnostics_claim_nothing(self) -> None:
        diagnostics = unevaluated_diagnostics()
        self.assertIsNone(diagnostics.obstacle.reported)
        self.assertIsNone(diagnostics.obstacle.nearest_distance_m)
        self.assertIsNone(diagnostics.path_quality.supported_length_m)
        self.assertIsNone(diagnostics.route.path_id)
        self.assertIsNone(diagnostics.coverage.path_supported_range_m)
        self.assertEqual(diagnostics.path_quality.reason, "not_estimated")

    def test_queue_is_bounded(self) -> None:
        self.assertEqual(check_queue_depth(MAX_QUEUE_DEPTH), MAX_QUEUE_DEPTH)
        with self.assertRaises(ValueError):
            check_queue_depth(MAX_QUEUE_DEPTH + 1)
        with self.assertRaises(ValueError):
            RouteSelection(path_id="p0", origin="tunnel_center")


if __name__ == "__main__":
    unittest.main()
