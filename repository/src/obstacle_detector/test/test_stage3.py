import time
import unittest

from obstacle_detector.axis_compatibility import assess_frame, assess_hypothesis
from obstacle_detector.frame_budget import call_bounded, sleep_for_budget_test


SPEC = {
    "width_m": 2.1,
    "center_l_m": 0.0,
    "heading_tolerance_dl_ds": 0.15,
    "heading_tolerance_source": "config/session.yaml match.heading_tolerance",
    "min_support_span_m": 4.0,
    "min_support_span_source": "corridor.yaml section gap",
}


def _section(s_m, lateral):
    return {"s_m": s_m, "left_l_m": lateral - 0.7, "right_l_m": lateral + 0.7}


class AxisTests(unittest.TestCase):
    def test_weak_combination_does_not_forbid(self):
        row = assess_hypothesis(
            {
                "hypothesis_id": "H1",
                "kind": "insufficient_rail_features",
                "observed_sections": [_section(6, 0.0), _section(10, 0.2), _section(16, 1.4), _section(20, 2.2)],
            },
            SPEC,
        )
        self.assertEqual(row["axis_compatibility"], "axis_compatibility_unknown")
        self.assertFalse(row["forbids_configured_straight"])
        self.assertNotEqual(row["axis_compatibility"], "confirmed_curve")

    def test_lateral_offset_does_not_forbid(self):
        row = assess_hypothesis(
            {
                "hypothesis_id": "H2",
                "kind": "consistent_rail_pair_hypothesis",
                "observed_sections": [_section(6, -2.0), _section(10, -2.0), _section(15, -2.0), _section(20, -2.05)],
            },
            SPEC,
        )
        self.assertIn("constant_lateral_offset", row["features"])
        self.assertFalse(row["forbids_configured_straight"])
        self.assertEqual(row["axis_compatibility"], "axis_compatibility_unknown")

    def test_unknown_ownership_is_not_confirmed_curve(self):
        row = assess_hypothesis(
            {
                "hypothesis_id": "H3",
                "kind": "consistent_rail_pair_hypothesis",
                "not_an_own_path": True,
                "observed_sections": [_section(6, 0.0), _section(10, 0.0), _section(15, 1.0), _section(20, 2.4)],
            },
            SPEC,
        )
        self.assertIn("direction_changes_along_support", row["features"])
        self.assertEqual(row["axis_compatibility"], "axis_compatibility_unknown")
        self.assertFalse(row["forbids_configured_straight"])

    def test_assigned_own_path_with_curvature_forbids_decision(self):
        frame = assess_frame(
            [
                {
                    "hypothesis_id": "H4",
                    "kind": "consistent_rail_pair_hypothesis",
                    "own_path_assigned": True,
                    "observed_sections": [_section(6, 0.0), _section(10, 0.0), _section(15, 1.0), _section(20, 2.4)],
                }
            ],
            SPEC,
        )
        self.assertFalse(frame["configured_straight_applicable_for_decision"])
        self.assertEqual(frame["per_hypothesis"][0]["axis_compatibility"], "confirmed_curve")

    def test_axes_are_not_combined(self):
        frame = assess_frame(
            [
                {
                    "hypothesis_id": "A",
                    "kind": "consistent_rail_pair_hypothesis",
                    "observed_sections": [_section(6, -1.5), _section(10, -1.5), _section(15, -1.5), _section(20, -1.5)],
                },
                {
                    "hypothesis_id": "B",
                    "kind": "consistent_rail_pair_hypothesis",
                    "observed_sections": [_section(6, 1.4), _section(10, 1.4), _section(15, 1.4), _section(20, 1.4)],
                },
            ],
            SPEC,
        )
        self.assertTrue(frame["combined_axis_span_not_used"])
        self.assertTrue(frame["configured_straight_applicable_for_decision"])


class BudgetTests(unittest.TestCase):
    def test_timeout_stops_the_worker(self):
        started = time.perf_counter()
        result = call_bounded(sleep_for_budget_test, (3.0,), 0.4)
        elapsed = time.perf_counter() - started
        self.assertEqual(result["status"], "processing_timeout")
        self.assertFalse(result["worker_alive_after_stop"])
        self.assertLess(elapsed, 2.5)
        self.assertTrue(result["empty_result_is_not_clear"])


class IntrusionRuleTests(unittest.TestCase):
    def _face(self):
        return {
            "candidate_id": "L1",
            "branch": "volume_or_suspended",
            "unique_xyz": 483,
            "observed_extent_m": {"s_m": 0.012, "l_m": 1.904, "h_m": 1.443},
            "range_from_lidar_m": 33.050,
            "profile_assumptions": ["orientation_unconfirmed"],
        }

    def test_accepts_large_volume_without_editing_the_candidate(self):
        from obstacle_detector.intrusion_rule import decide

        original = self._face()
        before = dict(original)
        ruled = decide(
            [original],
            corridor_source="configured_straight",
            profile_applicable=True,
            ground_filter_available=True,
            status="ok",
            object_size_m=[0.30, 0.30, 0.10],
            large_min_unique_xyz=8,
            orientation_unconfirmed=True,
        )
        self.assertEqual(ruled["decision"], "conditional_intrusion")
        self.assertEqual(ruled["conditional_intrusions"][0]["range_from_lidar_m"], 33.050)
        self.assertEqual(original, before)
        self.assertTrue(ruled["orientation_doubt_remains"])

    def test_one_point_and_missing_ground_do_not_become_an_intrusion(self):
        from obstacle_detector.intrusion_rule import decide

        point = {"candidate_id": "R1", "branch": "volume_or_suspended", "unique_xyz": 1, "observed_extent_m": {"s_m": 0.0, "l_m": 0.0, "h_m": 0.0}, "range_from_lidar_m": 10.0}
        weak = decide([point], corridor_source="configured_straight", profile_applicable=True, ground_filter_available=True, status="ok", object_size_m=[0.30, 0.30, 0.10], large_min_unique_xyz=8, orientation_unconfirmed=True)
        self.assertEqual(weak["decision"], "candidate_only")
        self.assertEqual(weak["candidate_decisions"][0]["reason"], "single_point_is_weak")
        missing = decide([self._face()], corridor_source="configured_straight", profile_applicable=True, ground_filter_available=False, status="ok", object_size_m=[0.30, 0.30, 0.10], large_min_unique_xyz=8, orientation_unconfirmed=True)
        self.assertEqual(missing["decision"], "insufficient_evidence")
        self.assertEqual(missing["conditional_intrusions"], [])

    def test_tall_partial_face_is_not_dropped_for_a_short_horizontal_span(self):
        from obstacle_detector.intrusion_rule import decide

        face = {
            "candidate_id": "L2",
            "branch": "volume_or_suspended",
            "unique_xyz": 16,
            "observed_extent_m": {"s_m": 0.01, "l_m": 0.26, "h_m": 1.50},
            "observed_s_min_m": 98.0,
            "range_from_lidar_m": 98.7,
        }
        ruled = decide(
            [face],
            corridor_source="configured_straight",
            profile_applicable=True,
            ground_filter_available=True,
            status="ok",
            object_size_m=[0.30, 0.30, 0.10],
            large_min_unique_xyz=8,
            orientation_unconfirmed=True,
        )
        self.assertEqual(ruled["decision"], "conditional_intrusion")
        hit = ruled["conditional_intrusions"][0]
        self.assertEqual(hit["evidence"], "vertical_face")
        self.assertEqual(hit["corridor_segment"], "assumed_continuation")
        self.assertFalse(hit["confirmed_wire"])

    def test_compact_candidate_keeps_position_for_the_segment_label(self):
        from obstacle_detector.frame_pipeline import _compact_candidate
        from obstacle_detector.intrusion_rule import decide

        compact = _compact_candidate(
            {
                "candidate_id": "L9",
                "hypothesis_id": "H1",
                "kind": "large",
                "branch": "volume_or_suspended",
                "unique_xyz": 16,
                "observed_extent_m": {"s_m": 0.2, "l_m": 0.4, "h_m": 1.2},
                "observed_s_min_m": 98.4,
                "observed_h_min_m": -0.4,
                "range_from_lidar_m": 98.7,
                "doubt_reasons": [],
            }
        )
        self.assertEqual(compact["observed_s_min_m"], 98.4)
        self.assertEqual(compact["observed_h_min_m"], -0.4)
        ruled = decide(
            [compact],
            corridor_source="configured_straight",
            profile_applicable=True,
            ground_filter_available=True,
            status="ok",
            object_size_m=[0.30, 0.30, 0.10],
            large_min_unique_xyz=8,
            orientation_unconfirmed=True,
            continuation_from_s_m=25.0,
        )
        self.assertEqual(ruled["conditional_intrusions"][0]["corridor_segment"], "assumed_continuation")

    def test_ten_sparse_points_along_a_line_are_a_fragment_not_a_wire(self):
        from obstacle_detector.intrusion_rule import decide

        fragment = {
            "candidate_id": "L3",
            "branch": "volume_or_suspended",
            "unique_xyz": 10,
            "observed_extent_m": {"s_m": 1.20, "l_m": 0.02, "h_m": 0.02},
            "observed_s_min_m": 12.0,
            "range_from_lidar_m": 12.0,
        }
        ruled = decide(
            [fragment],
            corridor_source="configured_straight",
            profile_applicable=True,
            ground_filter_available=True,
            status="ok",
            object_size_m=[0.30, 0.30, 0.10],
            large_min_unique_xyz=8,
            orientation_unconfirmed=True,
        )
        self.assertEqual(ruled["conditional_intrusions"][0]["evidence"], "thin_fragment")
        self.assertFalse(ruled["conditional_intrusions"][0]["confirmed_wire"])

    def test_full_width_flat_sheet_is_not_a_thin_fragment(self):
        from obstacle_detector.intrusion_rule import decide

        sheet = {
            "candidate_id": "L1",
            "branch": "volume_or_suspended",
            "unique_xyz": 20,
            "observed_extent_m": {"s_m": 0.055, "l_m": 1.946, "h_m": 0.0017},
            "observed_s_min_m": 58.6,
            "observed_h_min_m": 1.885,
            "range_from_lidar_m": 58.676,
        }
        ruled = decide(
            [sheet],
            corridor_source="configured_straight",
            profile_applicable=True,
            ground_filter_available=True,
            status="ok",
            object_size_m=[0.30, 0.30, 0.10],
            large_min_unique_xyz=8,
            orientation_unconfirmed=True,
            thin_floor_h_m=-1.075,
            thin_clearance_m=1.0,
        )
        self.assertEqual(ruled["decision"], "candidate_only")
        self.assertEqual(ruled["candidate_decisions"][0]["reason"], "wide_flat_sheet_not_a_fragment")

    def test_empty_applicable_volume_is_not_a_free_path(self):
        from obstacle_detector.intrusion_rule import decide

        ruled = decide([], corridor_source="configured_straight", profile_applicable=True, ground_filter_available=True, status="ok", object_size_m=[0.30, 0.30, 0.10], large_min_unique_xyz=8, orientation_unconfirmed=True)
        self.assertEqual(ruled["decision"], "no_intrusion_detected")
        self.assertTrue(ruled["configured_volume_is_not_declared_free"])


class IdentityTests(unittest.TestCase):
    def test_profile_and_budget_change_the_identity(self):
        from obstacle_detector.offline_run import run_identity

        base = run_identity("v", "cfg", "/lidar_points", "bag", "reader")
        other_budget = run_identity("v", "cfg", "/lidar_points", "bag", "reader", frame_budget_s=8.0)
        other_profile = run_identity("v", "cfg", "/lidar_points", "bag", "reader", profile_bytes=b"width: 2.1\n")
        self.assertNotEqual(base, other_budget)
        self.assertNotEqual(base, other_profile)


if __name__ == "__main__":
    unittest.main()
