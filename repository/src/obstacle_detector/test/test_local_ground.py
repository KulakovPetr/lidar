import unittest

import numpy as np

from obstacle_detector.local_ground import (
    CONFIG,
    estimate_ground,
    height_above_local,
    link_clusters,
    longitudinal_candidates,
    section_arrays,
    sensor_to_working,
    transform_roundtrip_max_error,
    working_to_sensor,
)
from obstacle_detector.repeat_groups import group_exact_xyz, repeat_summary, support_count, valid_mask


class TransformTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        x = np.array([1.5, -2.0, 0.0])
        y = np.array([-10.0, -4.0, 3.0])
        z = np.array([0.2, -1.0, 2.0])
        self.assertEqual(transform_roundtrip_max_error(x, y, z), 0.0)
        s, l, h = sensor_to_working(x, y, z)
        self.assertTrue(np.array_equal(s, -y))
        self.assertTrue(np.array_equal(l, x))
        self.assertTrue(np.array_equal(h, z))
        xr, yr, zr = working_to_sensor(s, l, h)
        self.assertTrue(np.allclose(xr, x) and np.allclose(yr, y) and np.allclose(zr, z))


class RepeatTests(unittest.TestCase):
    def test_repeated_xyz_is_one_vote_and_keeps_indices(self) -> None:
        x = np.array([1, 1, 1, 2, 0], dtype=np.float32)
        y = np.array([4, 4, 4, 5, 0], dtype=np.float32)
        z = np.array([7, 7, 7, 8, 0], dtype=np.float32)
        intensity = np.array([1, 3, 1, 2, 0], dtype=np.float32)
        ring = np.array([1, 1, 1, 2, 0], dtype=np.uint16)
        timestamp = np.array([10, 10, 11, 12, 0], dtype=np.float64)
        valid = valid_mask(x, y, z)
        self.assertFalse(valid[4])
        summary = repeat_summary(x, y, z, intensity, ring, timestamp, valid)
        self.assertEqual(summary["valid_records"], 4)
        self.assertEqual(summary["unique_exact_xyz"], 2)
        self.assertEqual(support_count(summary["groups"]["size"]), 2)
        self.assertEqual(int(summary["groups"]["size"].max()), 3)
        big = int(np.argmax(summary["groups"]["size"]))
        self.assertCountEqual(summary["groups"]["indices"][big].tolist(), [0, 1, 2])
        self.assertGreater(summary["intensity_span_max"], 0)
        self.assertGreater(summary["timestamp_span_max"], 0)

    def test_origin_is_not_a_partial_zero(self) -> None:
        valid = valid_mask(
            np.array([0.0, 5.0, 0.0]),
            np.array([0.0, 0.0, 4.0]),
            np.array([0.0, 1.0, 0.0]),
        )
        self.assertEqual(valid.tolist(), [False, True, True])

    def test_span_matches_nanmax_minus_nanmin(self) -> None:
        rng = np.random.default_rng(4)
        x = rng.integers(0, 30, 80).astype(np.float32)
        y = rng.integers(0, 30, 80).astype(np.float32)
        z = rng.integers(0, 30, 80).astype(np.float32)
        intensity = rng.random(80).astype(np.float32)
        intensity[rng.choice(80, 8, replace=False)] = np.nan
        timestamp = rng.random(80)
        ring = rng.integers(0, 4, 80)
        valid = np.ones(80, dtype=bool)
        groups = group_exact_xyz(x, y, z, intensity, ring, timestamp, valid)
        for span_name, values in (("intensity_span", intensity), ("timestamp_span", timestamp)):
            expected = []
            for members in groups["indices"]:
                if len(members) <= 1:
                    expected.append(0.0)
                    continue
                chosen = np.asarray(values)[np.asarray(members)]
                if not np.isfinite(chosen).any():
                    expected.append(np.nan)
                else:
                    expected.append(float(np.nanmax(chosen) - np.nanmin(chosen)))
            got = np.asarray(groups[span_name], dtype=np.float64)
            self.assertTrue(np.allclose(got, expected, equal_nan=True))


def _small_config():
    config = dict(CONFIG)
    config["min_groups_per_mode"] = 12
    config["section_min_groups"] = 8
    config["structure_min_groups"] = 4
    return config


class GroundTests(unittest.TestCase):
    def test_larger_ceiling_does_not_replace_the_lower_surface(self) -> None:
        rng = np.random.default_rng(1)
        config = _small_config()
        s_floor = rng.uniform(5.0, 30.0, 250)
        l_floor = rng.uniform(-3.0, 3.0, 250)
        s_ceil = rng.uniform(5.0, 30.0, 500)
        l_ceil = rng.uniform(-3.0, 3.0, 500)
        s = np.r_[s_floor, s_ceil, np.linspace(6, 12, 8)]
        l = np.r_[l_floor, l_ceil, np.zeros(8)]
        h = np.r_[np.full(250, -1.0), np.full(500, 2.5), np.full(8, -4.0)]
        h = h + rng.normal(0, 0.005, h.shape)
        result = estimate_ground(s, l, h, config)
        self.assertEqual(result["status"], "candidate")
        self.assertLess(result["ground"]["h_median"], 0.0)
        self.assertGreater(result["ground"]["h_median"], -1.5)
        self.assertTrue(any(item["role"] == "higher_extended_surface" for item in result["alternatives"]))
        self.assertLess(result["local_residual"]["p95_m"], 0.2)
        dh = height_above_local(s, l, h, result)
        self.assertFalse(np.any(np.isfinite(dh) & (h < -3) & (np.abs(dh) < 0.2)))

    def test_section_keeps_points_and_structure_is_not_named_a_rail(self) -> None:
        config = _small_config()
        rng = np.random.default_rng(2)
        s = rng.uniform(5.0, 30.0, 400)
        l = rng.uniform(-3.0, 3.0, 400)
        h = np.full(400, -1.0) + rng.normal(0.0, 0.004, 400)
        for center in config["section_s_m"]:
            s = np.r_[s, np.full(10, center)]
            l = np.r_[l, np.linspace(-0.04, 0.04, 10)]
            h = np.r_[h, np.full(10, -0.80)]
        ground = estimate_ground(s, l, h, config)
        self.assertEqual(ground["status"], "candidate")
        dh = height_above_local(s, l, h, ground)
        sections = [
            section_arrays(s, l, h, dh, center, config, rep_index=np.arange(s.size))
            for center in config["section_s_m"]
        ]
        self.assertTrue(all(section["status"] == "ok" for section in sections))
        candidates = longitudinal_candidates(sections, config)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertTrue(all(item["not_a_confirmed_rail"] for item in candidates))
        self.assertTrue(all(item["representative_indices"] for item in candidates))

    def test_inclined_line_survives_spatial_link_without_a_wider_gate(self) -> None:
        config = _small_config()
        config["structure_min_sections"] = 3
        centers = [6.0, 10.0, 15.0, 20.0]
        laterals = [0.0, -0.20, -0.50, -0.85]
        sections = []
        for center, lateral in zip(centers, laterals):
            count = config["structure_min_groups"]
            sections.append(
                {
                    "status": "ok",
                    "s_center_m": center,
                    "unique_groups": count,
                    "l": np.linspace(lateral - 0.02, lateral + 0.02, count),
                    "h": np.full(count, -1.2),
                    "dh": np.full(count, 0.12),
                }
            )
        fixed = longitudinal_candidates(sections, config, link="fixed_dl")
        spatial = longitudinal_candidates(sections, config, link="spatial")
        self.assertEqual(fixed, [])
        self.assertEqual(len(spatial), 1)
        self.assertGreaterEqual(spatial[0]["support_sections"], 3)
        self.assertTrue(spatial[0]["not_a_confirmed_rail"])

    def test_init_cone_accepts_a_nonzero_slope_when_two_points_are_not_a_forecast(self) -> None:
        config = _small_config()
        laterals = [0.0, 0.02, 0.55, 0.70, 0.85]
        sections = []
        for center, lateral in zip(config["section_s_m"], laterals):
            count = config["structure_min_groups"]
            sections.append(
                {
                    "status": "ok",
                    "s_center_m": center,
                    "unique_groups": count,
                    "l": np.linspace(lateral - 0.02, lateral + 0.02, count),
                    "h": np.full(count, -1.2),
                    "dh": np.full(count, 0.12),
                }
            )
        spatial = longitudinal_candidates(sections, config, link="spatial")
        robust = longitudinal_candidates(sections, config, link="robust")
        self.assertTrue(all(10.0 not in item["s_m"] or 15.0 not in item["s_m"] for item in spatial))
        self.assertTrue(any(10.0 in item["s_m"] and 15.0 in item["s_m"] for item in robust))

    def test_cluster_input_order_does_not_change_the_track(self) -> None:
        config = _small_config()
        centers = [6.0, 10.0, 15.0, 20.0]

        def cluster(station, lateral):
            return {
                "s_center_m": station,
                "l_median": lateral,
                "l_min": lateral - 0.02,
                "l_max": lateral + 0.02,
                "width_m": 0.04,
                "dh_median": 0.1,
                "h_median": -1.2,
                "groups": 6,
                "rep_indices": [int(station * 10 + lateral)],
            }

        forward = [
            [cluster(6.0, -1.0), cluster(6.0, 1.1)],
            [cluster(10.0, -0.85), cluster(10.0, 1.2)],
            [cluster(15.0, -0.65), cluster(15.0, 1.35)],
            [cluster(20.0, -0.45), cluster(20.0, 1.5)],
        ]
        backward = [list(reversed(row)) for row in forward]
        left = link_clusters(forward, centers, config, "robust")
        right = link_clusters(backward, centers, config, "robust")
        def keys(tracks):
            ready = [track for track in tracks if len(track["clusters"]) >= 3]
            return {
                tuple((round(item["s_center_m"], 1), round(item["l_median"], 3)) for item in track["clusters"])
                for track in ready
            }
        self.assertEqual(keys(left), keys(right))


if __name__ == "__main__":
    unittest.main()
