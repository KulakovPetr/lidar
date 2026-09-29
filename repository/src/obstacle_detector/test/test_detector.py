import inspect
import unittest

import numpy as np

from obstacle_detector.corridor import build_path_corridor, classify_sensor_points, load_config, working_to_sensor
from obstacle_detector.detector import (
    cluster_labels,
    cluster_labels_by_distance,
    detect,
    evaluate_insertion,
    insertion_funnel,
    load_detector_config,
    local_protrusions,
    subsample_grid,
)
from test.test_corridor import _ground, _hypothesis, _mount, _sensor


def _detector_config():
    config = load_detector_config()
    return config


def _corridor(lower=0.0, status="assumption"):
    corridor_config = load_config()
    corridor_config["lower_boundary_above_reference_m"] = {
        "value": lower,
        "source": "manual_estimate",
        "status": status,
        "basis": "test",
    }
    return build_path_corridor(_hypothesis([6, 10, 15, 20]), _ground(), corridor_config, _mount(), lower_boundary=corridor_config["lower_boundary_above_reference_m"])


class DetectorTests(unittest.TestCase):
    def test_detector_does_not_accept_insertion_labels(self):
        self.assertNotIn("label", inspect.signature(detect).parameters)
        self.assertNotIn("inserted", inspect.signature(detect).parameters)

    def test_assumption_floor_can_be_inside_and_stays_conditional(self):
        corridor = _corridor()
        detected = self._box(corridor, dh_shift=0.08, band=0.02)
        self.assertGreater(detected["counts"]["geometric_inside"], 0)
        self.assertTrue(detected["conditional"])
        self.assertIsNone(detected["obstacle_reported"])
        self.assertTrue(detected["absence_of_candidates_is_not_clear"])

    def test_thin_surface_band_does_not_eat_the_whole_10_cm_box(self):
        corridor = _corridor(status="stated")
        detected = self._box(corridor, dh_shift=0.0, band=0.02)
        self.assertGreater(detected["counts"]["object_pool"], 0)
        self.assertEqual(detected["surface_mode"], "thin_band_removed")
        self.assertTrue(detected["candidates_large"] or detected["candidates_rare"])

    def test_comparable_ground_band_keeps_the_box_ambiguous(self):
        corridor = _corridor(status="stated")
        detected = self._box(corridor, dh_shift=0.0, band=0.12)
        self.assertEqual(detected["surface_mode"], "ambiguous_not_deleted")
        self.assertGreater(detected["counts"]["ambiguous_surface"], 0)
        self.assertEqual(detected["counts"]["excluded_thin_surface"], 0)
        funnel = insertion_funnel(np.arange(detected["counts"]["input"]), detected["masks"])
        self.assertGreater(funnel["ambiguous_surface"], 0)

    def test_single_point_remains_a_rare_fragment(self):
        corridor = _corridor(status="stated")
        x, y, z = _sensor([15.0], [0.0], [-0.5])
        detected = detect(
            np.array([15.0]),
            np.array([0.0]),
            np.array([-0.5]),
            np.array([0.5]),
            np.array([True]),
            np.sqrt(x * x + y * y + z * z),
            np.array([7]),
            np.array(["inside"]),
            corridor,
            [],
            0.02,
            _detector_config(),
        )
        self.assertEqual(len(detected["candidates_rare"]), 1)
        self.assertEqual(int(detected["candidates_rare"][0]["source_indices"][0]), 7)
        self.assertEqual(detected["candidates_rare"][0]["unique_xyz"], 1)
        self.assertFalse(detected["candidates_large"])

    def test_outside_box_is_not_a_candidate(self):
        corridor = _corridor(status="stated")
        detected = self._box(corridor, dh_shift=0.05, band=0.02, lateral=2.5)
        self.assertEqual(detected["counts"]["geometric_inside"], 0)
        self.assertFalse(detected["candidates_large"])
        self.assertFalse(detected["candidates_rare"])

    def test_missing_line_height_does_not_delete_the_column(self):
        detected = self._line_points(
            np.array([-0.5, -0.5]),
            [{"s_m": 15.0, "l_m": 0.75, "width_m": 0.10}],
        )
        self.assertEqual(detected["counts"]["excluded_path_structure"], 0)
        self.assertEqual(detected["counts"]["path_possible_not_removed"], 1)
        self.assertEqual(detected["counts"]["object_pool"], 2)

    def test_supported_line_surface_excludes_only_its_height(self):
        sample = [{"s_m": 15.0, "l_m": 0.75, "width_m": 0.10, "h_m": -0.5, "height_support": "supported"}]
        surface = self._line_points(np.array([-0.5, -0.5]), sample)
        self.assertEqual(surface["counts"]["excluded_path_structure"], 1)
        self.assertEqual(int(surface["masks"]["object_pool"][0]), 1)
        above = self._line_points(np.array([-0.5, 0.2]), sample, lateral=np.array([0.75, 0.75]))
        self.assertEqual(int(above["masks"]["path_structure"].sum()), 1)
        self.assertEqual(int(above["masks"]["object_pool"][1]), 1)
        suspended = self._line_points(np.linspace(0.4, 0.6, 5), sample, lateral=0.75, along_s=True)
        self.assertEqual(suspended["counts"]["excluded_path_structure"], 0)
        self.assertGreater(suspended["counts"]["object_pool"], 0)
        stacked = self._line_points(np.array([-0.50, -0.45, -0.40]), sample, lateral=0.75)
        self.assertGreater(stacked["counts"]["excluded_path_structure"], 0)
        self.assertGreater(stacked["counts"]["object_pool"], 0)
        self.assertLess(stacked["counts"]["excluded_path_structure"], 3)

    def test_joint_cluster_is_not_credited_from_a_separate_pool(self):
        corridor = _corridor(status="stated")
        config = _detector_config()
        background_s = np.linspace(14.7, 15.05, 8)
        background = self._cloud(background_s, np.zeros(8), np.full(8, -0.4), corridor, config)
        alone_s = np.array([15.12, 15.18, 15.24])
        alone = self._cloud(alone_s, np.full(3, 0.05), np.full(3, -0.38), corridor, config)
        self.assertTrue(alone["candidates_rare"] or alone["candidates_large"])
        merged_s = np.concatenate([background_s, alone_s])
        joint = self._cloud(merged_s, np.concatenate([np.zeros(8), np.full(3, 0.05)]), np.concatenate([np.full(8, -0.4), np.full(3, -0.38)]), corridor, config)
        score = evaluate_insertion(np.arange(8, 11), joint, background["candidates_large"] + background["candidates_rare"], np.full(11, 15.0))
        self.assertEqual(score["outcome"], "merged_with_background")
        self.assertFalse(score["predominantly_insertion"])
        self.assertTrue(score["pooled_is_not_separation"])
        self.assertGreater(score["passed_filters"], 0)

    def test_ambiguous_layer_is_not_a_detection(self):
        corridor = _corridor(status="stated")
        detected = self._box(corridor, dh_shift=0.0, band=0.12)
        score = evaluate_insertion(np.arange(detected["counts"]["input"]), detected, [], np.full(detected["counts"]["input"], 15.0))
        self.assertEqual(score["outcome"], "ambiguous_layer_only")
        self.assertEqual(score["in_candidate"], 0)
        self.assertTrue(score["diagnostic_expansion_is_not_detection"])
        self.assertGreater(detected["surface"]["points_withheld_ambiguous"], 0)
        self.assertFalse(detected["surface"]["max_with_object_height_applied"])
        expanded = self._box(corridor, dh_shift=0.0, band=0.06)
        self.assertTrue(expanded["surface"]["max_with_object_height_applied"])
        self.assertAlmostEqual(expanded["surface"]["used_band_m"], 0.10)
        self.assertGreater(expanded["surface"]["diagnostic_points_withheld_only_by_expansion"], 0)
        funnel = insertion_funnel(np.arange(detected["counts"]["input"]), detected["masks"])
        self.assertNotIn("separated", funnel)

    def test_adjacent_cells_do_not_link_distant_points(self):
        s = np.array([0.01, 0.29])
        l = np.array([0.01, 0.01])
        h = np.array([0.01, 0.01])
        cells = cluster_labels(s, l, h, 0.15)
        distance = cluster_labels_by_distance(s, l, h, 0.15, 0.15)
        self.assertEqual(int(cells[0]), int(cells[1]))
        self.assertNotEqual(int(distance[0]), int(distance[1]))

    def test_close_points_link_across_a_cell_boundary(self):
        labels = cluster_labels_by_distance(np.array([0.14, 0.16]), np.zeros(2), np.zeros(2), 0.15, 0.15)
        self.assertEqual(int(labels[0]), int(labels[1]))

    def test_a_chain_is_one_component_and_a_sparse_fragment_is_kept(self):
        chain = cluster_labels_by_distance(np.array([0.0, 0.14, 0.28]), np.zeros(3), np.zeros(3), 0.15, 0.15)
        self.assertEqual(len(set(chain.tolist())), 1)
        sparse = cluster_labels_by_distance(np.array([0.0, 0.40, 0.80]), np.zeros(3), np.zeros(3), 0.15, 0.15)
        self.assertEqual(len(set(sparse.tolist())), 3)
        corridor = _corridor(status="stated")
        detected = detect(
            np.array([15.0, 15.4, 15.8]),
            np.zeros(3),
            np.full(3, -0.4),
            np.full(3, 0.6),
            np.ones(3, dtype=bool),
            np.full(3, 15.0),
            np.arange(3),
            np.array(["inside", "inside", "inside"]),
            corridor,
            [],
            0.02,
            _detector_config(),
            connectivity="distance",
        )
        self.assertEqual(len(detected["candidates_rare"]), 3)
        self.assertFalse(detected["candidates_large"])

    def test_local_bump_is_a_protrusion_without_a_10_cm_minimum(self):
        s = np.linspace(0.0, 1.2, 24)
        l = np.zeros(24)
        h = np.zeros(24)
        bump_s = np.linspace(0.45, 0.55, 6)
        points_s = np.concatenate([s, bump_s])
        points_l = np.concatenate([l, np.zeros(6)])
        points_h = np.concatenate([h, np.full(6, 0.04)])
        query = np.concatenate([np.zeros(24, dtype=bool), np.ones(6, dtype=bool)])
        found = local_protrusions(points_s, points_l, points_h, query, _detector_config())
        self.assertGreater(int(found["mask"].sum()), 0)
        self.assertNotIn("label", local_protrusions.__code__.co_varnames)
        self.assertNotIn("inserted", local_protrusions.__code__.co_varnames)
        two = np.concatenate([np.zeros(15), np.full(15, 0.30)])
        levels = local_protrusions(np.linspace(0, 1, 30), np.zeros(30), two, np.ones(30, dtype=bool), _detector_config())
        self.assertEqual(int(levels["mask"].sum()), 0)
        self.assertGreater(levels["skip_counts"].get("несколько высотных уровней в окрестности", 0), 0)

    def test_ulp_below_a_flat_floor_is_not_a_candidate(self):
        corridor = _corridor(status="stated")
        s = np.tile(np.linspace(14.0, 16.0, 5), 5)
        l = np.repeat(np.linspace(-0.4, 0.4, 5), 5)
        h = np.full(s.shape, -1.3)
        dh = np.full(s.shape, -4.440892098500626e-16)
        floor = detect(
            s, l, h, dh, np.ones(s.shape, dtype=bool), np.full(s.shape, 15.0),
            np.arange(s.size), np.array(["inside"] * int(s.size)), corridor, [],
            4.440892098500626e-16, _detector_config(), connectivity="distance",
        )
        self.assertEqual(floor["counts"]["object_pool"], 0)
        self.assertEqual(floor["counts"]["rare"], 0)
        box_s = np.array([15.0, 15.05, 15.10])
        box = detect(
            box_s, np.zeros(3), np.full(3, -1.20), np.full(3, 0.10),
            np.ones(3, dtype=bool), np.full(3, 15.0), np.arange(3),
            np.array(["inside", "inside", "inside"]), corridor, [],
            4.440892098500626e-16, _detector_config(), connectivity="distance",
        )
        self.assertGreater(box["counts"]["object_pool"], 0)

    def test_grid_is_reproducible(self):
        first = subsample_grid((10.0, -0.15, -1.0), (0.3, 0.3, 0.1), (6, 6, 4), 18, 4)
        second = subsample_grid((10.0, -0.15, -1.0), (0.3, 0.3, 0.1), (6, 6, 4), 18, 4)
        self.assertEqual(first.shape, (18, 3))
        self.assertTrue(np.allclose(first, second))

    def _line_points(self, heights, samples, lateral=None, along_s=False):
        h = np.atleast_1d(np.asarray(heights, dtype=np.float64))
        if lateral is None:
            lateral = np.array([0.0, 0.75], dtype=np.float64)[: h.size]
        lateral = np.broadcast_to(np.asarray(lateral, dtype=np.float64), h.size).copy()
        s = np.linspace(14.7, 15.3, h.size) if along_s else np.full(h.size, 15.0)
        return self._cloud(s, lateral, h, _corridor(status="stated"), _detector_config(), samples)

    def _cloud(self, s, l, h, corridor, config, samples=None):
        s = np.asarray(s, dtype=np.float64)
        l = np.asarray(l, dtype=np.float64)
        h = np.asarray(h, dtype=np.float64)
        return detect(
            s,
            l,
            h,
            np.full(s.shape, 0.6),
            np.ones(s.shape, dtype=bool),
            np.full(s.shape, 15.0),
            np.arange(s.size),
            np.array(["inside"] * int(s.size)),
            corridor,
            [] if samples is None else samples,
            0.02,
            config,
        )

    def _box(self, corridor, dh_shift, band, lateral=0.0):
        points = subsample_grid((15.0 - 0.15, lateral - 0.15, -1.0), (0.3, 0.3, 0.1), (4, 4, 3), 48, 4)
        s, l, h = points[:, 0], points[:, 1], points[:, 2]
        x, y, z = working_to_sensor(s, l, h)
        classification = classify_sensor_points(x, y, z, corridor, _mount(), load_config() | {"front_ahead_of_lidar_m": {"value": None}})
        # The file config still has a null floor. Pass the corridor's own floor via relation from a config that matches the corridor.
        return detect(
            s,
            l,
            h,
            h - (-1.0) + dh_shift,
            np.ones(s.shape, dtype=bool),
            np.sqrt(x * x + y * y + z * z),
            np.arange(s.size),
            classification["geometric_relation"],
            corridor,
            [],
            band,
            _detector_config(),
        )


if __name__ == "__main__":
    unittest.main()
