import unittest

import numpy as np

from obstacle_detector.pair_geometry import (
    PAIR_CONFIG,
    compare_pair_features,
    distance_from_corresponding_points,
    link_pair_histories,
    measure_candidate_pairs,
    observation_support,
    pair_consistency,
    pose_at_s_ref,
    rank_pair_hypotheses,
    reference_line_distance,
    sensor_shift_yaw,
    transformed_pair_check,
)


def _pair(left, right, distance, publish_l, k=0.0, straight=True):
    return {
        "candidate_ids": [0, 1],
        "left_l_m": left,
        "right_l_m": right,
        "distance_m": distance,
        "distance_mad_m": 0.01,
        "support_groups": 20,
        "sections": [],
        "midpoint_line": {
            "b_m": publish_l,
            "k_dl_ds": k,
            "approximately_straight": straight,
            "linear_extrapolation_is_calibration": False,
        },
        "publish": {
            "s_m": 15.0,
            "target_s_m": 15.0,
            "target_section_observed": True,
            "midpoint_l_m": publish_l,
            "midpoint_h_m": -1.2,
            "path_midpoint_relative_to_lidar_l_m": publish_l,
            "lidar_l_relative_to_path_midpoint_m": -publish_l,
            "source": "observed",
            "not_a_body_mount_offset": True,
        },
    }


class DistanceTests(unittest.TestCase):
    def test_shift_changes_position_and_keeps_cross_track_distance(self) -> None:
        result = transformed_pair_check(shift_l_m=0.45, yaw_rad=0.0)
        self.assertAlmostEqual(result["before"]["distance_m"], result["after"]["distance_m"], places=6)
        self.assertGreater(
            abs(result["after"]["midpoint_line"]["b_m"] - result["before"]["midpoint_line"]["b_m"]),
            0.40,
        )
        self.assertAlmostEqual(
            result["before"]["midpoint_line"]["k_dl_ds"],
            result["after"]["midpoint_line"]["k_dl_ds"],
            places=6,
        )

    def test_yaw_changes_direction_and_keeps_cross_track_distance(self) -> None:
        yaw = np.deg2rad(20.0)
        result = transformed_pair_check(shift_l_m=0.0, yaw_rad=yaw)
        self.assertAlmostEqual(result["before"]["distance_m"], result["after"]["distance_m"], places=5)
        self.assertGreater(abs(result["after"]["midpoint_line"]["k_dl_ds"]), 0.2)
        before_l = result["before"]["rows"][-1]["midpoint_l_m"]
        after_l = result["after"]["rows"][-1]["midpoint_l_m"]
        self.assertGreater(abs(after_l - before_l), 1.0)
        self.assertGreater(
            abs(result["after"]["same_s_cut_delta_l_m"] - result["after"]["distance_m"]),
            0.05,
        )

    def test_tilt_is_not_the_same_as_delta_l(self) -> None:
        measured = reference_line_distance((10.0, -0.2, -1.34), (10.0, 1.3, -1.04), 0.0, 0.0, 0.2)
        self.assertGreater(measured["reference_line_distance_m"], measured["same_s_delta_l_m"] + 0.01)

    def test_repeated_records_do_not_add_support(self) -> None:
        self.assertEqual(observation_support(10, 12), 22)
        self.assertNotEqual(observation_support(10, 12), 20 + 24)


class MeasureTests(unittest.TestCase):
    def test_pair_distance_follows_the_tilted_base(self) -> None:
        ground = {
            "status": "candidate",
            "local_bins": [{"s_lo": 5.0, "s_hi": 30.0, "slope_dh_dl": 0.2}],
            "plane": {"coef_h_from_s_l": [0.0, 0.2, -1.3]},
        }

        def candidate(lateral):
            samples = []
            for station in (6.0, 10.0, 15.0, 20.0, 25.0):
                samples.append(
                    {
                        "s_m": station,
                        "l_m": lateral,
                        "h_m": -1.3 + 0.2 * lateral,
                        "dh_m": 0.1,
                        "groups": 8,
                        "representative_indices": [int(station)],
                    }
                )
            return {"l_median_of_track_m": lateral, "section_samples": samples}

        pairs = measure_candidate_pairs([candidate(-0.2), candidate(1.3)], ground)
        self.assertEqual(len(pairs), 1)
        self.assertGreater(pairs[0]["distance_m"], 1.5)
        self.assertEqual(pairs[0]["publish"]["source"], "observed")
        self.assertFalse(pairs[0]["midpoint_line"]["linear_extrapolation_is_calibration"])
        self.assertGreater(pairs[0]["support_groups"], 0)


class HistoryTests(unittest.TestCase):
    def test_one_outlier_does_not_move_the_accumulated_distance(self) -> None:
        frames = []
        for index in range(6):
            frames.append(
                {
                    "frame_index": 162 + index,
                    "pairs": [_pair(-0.2, 1.3, 1.52, 0.55)],
                    "line_positions": [-0.2, 1.3],
                    "ground_found": True,
                }
            )
        frames.append(
            {
                "frame_index": 168,
                "pairs": [_pair(-0.2, 1.3, 2.40, 0.55)],
                "line_positions": [-0.2, 1.3],
                "ground_found": True,
            }
        )
        linked = link_pair_histories(frames, PAIR_CONFIG)
        self.assertEqual(len(linked["histories"]), 1)
        history = linked["histories"][0]
        admitted = [item["distance_m"] for item in history["admitted"]]
        self.assertNotIn(2.40, admitted)
        self.assertAlmostEqual(float(np.median(admitted)), 1.52, places=6)

    def test_ambiguous_match_pauses_and_does_not_mix(self) -> None:
        frames = [
            {
                "frame_index": 162,
                "pairs": [_pair(-0.20, 1.30, 1.50, 0.55)],
                "line_positions": [-0.20, 1.30],
                "ground_found": True,
            },
            {
                "frame_index": 163,
                "pairs": [
                    _pair(-0.18, 1.32, 1.50, 0.57),
                    _pair(-0.10, 1.40, 1.70, 0.65),
                ],
                "line_positions": [-0.18, 1.32, -0.10, 1.40],
                "ground_found": True,
            },
        ]
        linked = link_pair_histories(frames, PAIR_CONFIG)
        self.assertEqual(len(linked["histories"]), 1)
        self.assertEqual(len(linked["histories"][0]["admitted"]), 1)
        paused = [event for event in linked["histories"][0]["events"] if event["frame_index"] == 163]
        self.assertEqual(paused[0]["update"], "paused")
        self.assertIn("неоднозначное", paused[0]["reason"])

    def test_missing_line_is_not_a_new_measurement(self) -> None:
        frames = [
            {
                "frame_index": 162,
                "pairs": [_pair(-0.2, 1.3, 1.52, 0.55)],
                "line_positions": [-0.2, 1.3],
                "ground_found": True,
            },
            {
                "frame_index": 163,
                "pairs": [],
                "line_positions": [-0.21],
                "ground_found": True,
            },
        ]
        linked = link_pair_histories(frames, PAIR_CONFIG)
        event = linked["histories"][0]["events"][-1]
        self.assertEqual(event["update"], "paused")
        self.assertFalse(event["raw_is_measurement"])
        self.assertFalse(event["assumed_line"]["is_new_measurement"])
        self.assertEqual(event["assumed_line"]["label"], "предполагаемая")
        self.assertEqual(len(linked["histories"][0]["admitted"]), 1)

    def test_separate_pairs_keep_separate_distances(self) -> None:
        frames = []
        for index in range(4):
            frames.append(
                {
                    "frame_index": 170 + index,
                    "pairs": [
                        _pair(-0.8, 0.7, 1.50, -0.05),
                        _pair(3.2, 4.8, 1.62, 4.0),
                    ],
                    "line_positions": [-0.8, 0.7, 3.2, 4.8],
                    "ground_found": True,
                }
            )
        linked = link_pair_histories(frames, PAIR_CONFIG)
        self.assertEqual(len(linked["histories"]), 2)
        distances = {history["pair_id"]: history["admitted"][-1]["distance_m"] for history in linked["histories"]}
        self.assertIn(1.50, distances.values())
        self.assertIn(1.62, distances.values())

    def test_one_outlier_is_kept_apart_from_the_accepted_history(self) -> None:
        frames = []
        for index in range(4):
            frames.append(
                {
                    "frame_index": index,
                    "pairs": [_pair(-0.2, 1.3, 1.52, 0.55)],
                    "line_positions": [-0.2, 1.3],
                    "ground_found": True,
                }
            )
        frames.append(
            {
                "frame_index": 4,
                "pairs": [_pair(-0.2, 1.3, 2.40, 0.55)],
                "line_positions": [-0.2, 1.3],
                "ground_found": True,
            }
        )
        frames.append(
            {
                "frame_index": 5,
                "pairs": [_pair(-0.2, 1.3, 1.52, 0.55)],
                "line_positions": [-0.2, 1.3],
                "ground_found": True,
            }
        )
        linked = link_pair_histories(frames, PAIR_CONFIG)
        history = linked["histories"][0]
        admitted = [item["distance_m"] for item in history["admitted"]]
        self.assertNotIn(2.40, admitted)
        self.assertEqual(history["conflict"], None)
        self.assertTrue(any(abs(item["distance_m"] - 2.40) < 1e-9 for item in history["outliers"]))

    def test_stable_new_width_opens_a_conflict_and_does_not_replace_the_old_one(self) -> None:
        frames = []
        for index in range(4):
            frames.append(
                {
                    "frame_index": index,
                    "pairs": [_pair(-0.2, 1.3, 1.52, 0.55)],
                    "line_positions": [-0.2, 1.3],
                    "ground_found": True,
                }
            )
        for index in range(4, 7):
            frames.append(
                {
                    "frame_index": index,
                    "pairs": [_pair(-0.2, 1.3, 1.80, 0.55)],
                    "line_positions": [-0.2, 1.3],
                    "ground_found": True,
                }
            )
        linked = link_pair_histories(frames, PAIR_CONFIG)
        history = linked["histories"][0]
        admitted = [item["distance_m"] for item in history["admitted"]]
        self.assertTrue(all(abs(value - 1.52) < 1e-9 for value in admitted))
        self.assertEqual(history["conflict"]["status"], "WIDTH_CONFLICT")
        self.assertAlmostEqual(history["conflict"]["hypothesis_median_m"], 1.80, places=6)
        self.assertTrue(history["conflict"]["source_lines_match"])
        self.assertFalse(history["conflict"]["not_adopted"] is False)
        last = history["events"][-1]
        self.assertEqual(last["update"], "width_conflict")
        self.assertFalse(last["fresh_width_measurement"])
        self.assertGreater(last["width_age_frames"], 0)


class PoseTests(unittest.TestCase):
    def test_alternating_sections_do_not_jump_c15(self) -> None:
        ground = {
            "status": "candidate",
            "local_bins": [{"s_lo": 5.0, "s_hi": 30.0, "slope_dh_dl": 0.0}],
            "plane": {"coef_h_from_s_l": [0.0, 0.0, -1.3]},
        }

        def candidate(side, stations):
            samples = []
            for station in stations:
                lateral = 0.10 + 0.02 * station + side
                samples.append(
                    {
                        "s_m": station,
                        "l_m": lateral,
                        "h_m": -1.2,
                        "dh_m": 0.10,
                        "width_m": 0.08,
                        "groups": 8,
                        "representative_indices": [int(station)],
                    }
                )
            return {"l_median_of_track_m": samples[0]["l_m"], "geometry_key": f"side-{side}", "section_samples": samples}

        full = [6.0, 10.0, 15.0, 20.0, 25.0]
        gap = [6.0, 10.0, 20.0, 25.0]
        frames = []
        old_publish = []
        for index in range(6):
            stations = full if index % 2 == 0 else gap
            pairs = measure_candidate_pairs(
                [candidate(-0.75, stations), candidate(0.75, stations)],
                ground,
                PAIR_CONFIG,
            )
            self.assertEqual(len(pairs), 1)
            pose = pairs[0]["pose_at_s_ref"]
            self.assertFalse(pose["model_is_direct_observation"])
            self.assertTrue(pose["model_reliable"])
            self.assertAlmostEqual(pose["model_c_m"], 0.40, places=6)
            if index % 2 == 0:
                self.assertEqual(pose["model_support"], "inside_observed_interval")
                self.assertIsNotNone(pose["direct_observation"])
            else:
                self.assertEqual(pose["model_support"], "gap")
                self.assertIsNone(pose["direct_observation"])
                self.assertAlmostEqual(pose["nearest_observation"]["s_obs_m"], 10.0, places=6)
            old_publish.append(pairs[0]["publish"]["midpoint_l_m"])
            frames.append(
                {
                    "frame_index": index,
                    "timestamp_ns": 1_000_000_000 * index,
                    "pairs": pairs,
                    "line_positions": [pairs[0]["left_l_m"], pairs[0]["right_l_m"]],
                    "ground_found": True,
                }
            )
        linked = link_pair_histories(frames, PAIR_CONFIG)
        smoothed = [event["smoothed_c_s_ref_m"] for event in linked["histories"][0]["events"]]
        jumps = [abs(smoothed[i] - smoothed[i - 1]) for i in range(1, len(smoothed))]
        self.assertLess(max(jumps), 0.02)
        self.assertTrue(all(abs(value - 0.40) < 0.02 for value in smoothed))
        old_window = []
        for index in range(len(old_publish)):
            window = old_publish[max(0, index - 2) : index + 1]
            old_window.append(float(np.median(window)))
        old_jumps = [abs(old_window[i] - old_window[i - 1]) for i in range(1, len(old_window))]
        self.assertGreater(max(old_jumps), 0.05)

    def test_reordered_lines_keep_the_same_pair(self) -> None:
        ground = {
            "status": "candidate",
            "local_bins": [{"s_lo": 5.0, "s_hi": 30.0, "slope_dh_dl": 0.0}],
            "plane": {"coef_h_from_s_l": [0.0, 0.0, -1.3]},
        }

        def candidate(lateral, key):
            samples = []
            for station in (6.0, 10.0, 15.0, 20.0):
                samples.append(
                    {
                        "s_m": station,
                        "l_m": lateral,
                        "h_m": -1.2,
                        "dh_m": 0.10,
                        "width_m": 0.08,
                        "groups": 8,
                        "representative_indices": [int(station)],
                    }
                )
            return {"l_median_of_track_m": lateral, "geometry_key": key, "section_samples": samples}

        left = candidate(-0.4, "line-left")
        right = candidate(1.1, "line-right")
        forward = measure_candidate_pairs([left, right], ground, PAIR_CONFIG)
        backward = measure_candidate_pairs([right, left], ground, PAIR_CONFIG)
        self.assertEqual(forward[0]["line_keys"], backward[0]["line_keys"])
        self.assertAlmostEqual(forward[0]["distance_m"], backward[0]["distance_m"], places=6)
        self.assertNotEqual(forward[0]["candidate_ids"], backward[0]["candidate_ids"])
        linked = link_pair_histories(
            [
                {"frame_index": 1, "pairs": forward, "line_positions": [-0.4, 1.1], "ground_found": True},
                {"frame_index": 2, "pairs": backward, "line_positions": [-0.4, 1.1], "ground_found": True},
            ],
            PAIR_CONFIG,
        )
        self.assertEqual(len(linked["histories"]), 1)
        self.assertEqual(len(linked["histories"][0]["admitted"]), 2)


class ConsistencyTests(unittest.TestCase):
    def _ground(self):
        return {
            "status": "candidate",
            "local_bins": [{"s_lo": 5.0, "s_hi": 30.0, "slope_dh_dl": 0.0, "l_min": -3.0, "l_max": 3.0}],
            "plane": {"coef_h_from_s_l": [0.0, 0.0, -1.3]},
            "fitting_residual": {"p95_m": 0.05},
        }

    def _candidate(self, samples, key):
        return {
            "l_median_of_track_m": float(np.median([sample["l_m"] for sample in samples])),
            "geometry_key": key,
            "section_samples": samples,
        }

    def test_converging_lines_have_a_straight_midpoint_and_disagree(self) -> None:
        rows = []
        for station in (6.0, 10.0, 15.0, 20.0, 25.0):
            left_l = -1.0 + 0.03 * (station - 6.0)
            right_l = 1.0 - 0.03 * (station - 6.0)
            rows.append(
                {
                    "s_m": station,
                    "direction_dl_ds": 0.0,
                    "reference_line_distance_m": abs(right_l - left_l),
                    "line_a": {"l_m": left_l, "h_m": -1.2, "dh_m": 0.1},
                    "line_b": {"l_m": right_l, "h_m": -1.2, "dh_m": 0.1},
                }
            )
        consistency = pair_consistency(rows, PAIR_CONFIG)
        self.assertLess(consistency["midpoint_direction_mad"], 0.01)
        self.assertTrue(consistency["pair_geometry_inconsistent"])
        self.assertIn("направления линий расходятся", consistency["inconsistency_reasons"])
        self.assertIn("расстояние между линиями меняется вдоль пути", consistency["inconsistency_reasons"])
        self.assertGreater(len(consistency["section_distances_m"]), 2)
        self.assertTrue(consistency["midpoint_direction_is_not_line_agreement"])

    def test_parallel_curve_stays_consistent(self) -> None:
        stations = [6.0, 10.0, 15.0, 20.0, 25.0]
        left = []
        right = []
        for station in stations:
            curve = 0.002 * (station - 15.0) ** 2
            left.append({"s_m": station, "l_m": curve - 0.75, "h_m": -1.2, "dh_m": 0.1, "width_m": 0.08, "groups": 8})
            right.append({"s_m": station, "l_m": curve + 0.75, "h_m": -1.2, "dh_m": 0.1, "width_m": 0.08, "groups": 8})
        pairs = measure_candidate_pairs(
            [self._candidate(left, "left"), self._candidate(right, "right")],
            self._ground(),
            PAIR_CONFIG,
        )
        consistency = pair_consistency(pairs[0]["sections"], PAIR_CONFIG)
        self.assertFalse(consistency["pair_geometry_inconsistent"])
        self.assertGreater(consistency["midpoint_direction_mad"], 0.001)
        self.assertLess(consistency["distance_span_m"], 0.05)
        features = compare_pair_features(pairs, self._ground())
        self.assertIn("midpoint_direction_mad", features["pairs"][0])
        self.assertNotIn("direction_mad", features["pairs"][0])

    def test_shared_line_is_competition_not_two_confirmations(self) -> None:
        stations = [6.0, 10.0, 15.0, 20.0]
        lines = []
        for lateral, key in ((-1.2, "a"), (-0.2, "b"), (0.9, "c")):
            samples = [
                {"s_m": station, "l_m": lateral, "h_m": -1.2, "dh_m": 0.1, "width_m": 0.08, "groups": 8}
                for station in stations
            ]
            lines.append(self._candidate(samples, key))
        pairs = measure_candidate_pairs(lines, self._ground(), PAIR_CONFIG)
        ranked = rank_pair_hypotheses(pairs, lines, self._ground(), PAIR_CONFIG)
        self.assertGreaterEqual(len(ranked["hypotheses"]), 2)
        self.assertTrue(any(item["shared_with"] for item in ranked["hypotheses"]))
        scores = [item["engineering_score"] for item in ranked["hypotheses"]]
        margin = PAIR_CONFIG["score_rival_margin"]
        for item in ranked["hypotheses"]:
            if not item["shared_with"]:
                continue
            rivals = [other for other in ranked["hypotheses"] if other["hypothesis_id"] in item["shared_with"]]
            if any(abs(item["engineering_score"] - other["engineering_score"]) <= margin for other in rivals):
                self.assertNotEqual(item["kind"], "consistent_rail_pair_hypothesis")
            self.assertTrue(any("конкурирующая комбинация" in penalty for penalty in item["penalties"]))
        self.assertFalse(ranked["own_path_selected"])
        self.assertTrue(all(item["assumed_sections"] == [] or all(not row["adds_width_measurement"] for row in item["assumed_sections"]) for item in ranked["hypotheses"]))


class TransformHelperTests(unittest.TestCase):
    def test_shift_yaw_round_numbers(self) -> None:
        s, l, h = sensor_shift_yaw(np.array([10.0]), np.array([1.0]), np.array([-1.0]), 0.0, 0.0)
        self.assertAlmostEqual(float(s[0]), 10.0)
        self.assertAlmostEqual(float(l[0]), 1.0)
        measured = distance_from_corresponding_points(
            np.array([[6.0, 0.0, -1.0], [20.0, 0.0, -1.0]]),
            np.array([[6.0, 1.5, -1.0], [20.0, 1.5, -1.0]]),
        )
        self.assertAlmostEqual(measured["distance_m"], 1.5, places=6)


if __name__ == "__main__":
    unittest.main()
