import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from obstacle_detector.bag_io import choose_pointcloud_topic
from obstacle_detector.batch import append_record, read_completed, select_extra_frames
from obstacle_detector.frame_pipeline import load_batch_config, variant_flags


class Topic(unittest.TestCase):
    def test_one_topic_is_used(self):
        topic = type("T", (), {"name": "/lidar_points", "type": "sensor_msgs/msg/PointCloud2"})()
        name, reason = choose_pointcloud_topic([topic], ["/lidar_points"])
        self.assertEqual(name, "/lidar_points")
        self.assertIn("единственный", reason)

    def test_several_topics_are_not_merged(self):
        def topic(name):
            return type("T", (), {"name": name, "type": "sensor_msgs/msg/PointCloud2"})()

        with self.assertRaises(SystemExit):
            choose_pointcloud_topic(
                [topic("/a"), topic("/b")],
                ["/lidar_points", "/sensing/lidar/hesai128/pointcloud"],
            )

    def test_single_known_topic_is_named(self):
        def topic(name, kind="sensor_msgs/msg/PointCloud2"):
            return type("T", (), {"name": name, "type": kind})()

        name, reason = choose_pointcloud_topic(
            [topic("/lidar_points"), topic("/other")],
            ["/lidar_points"],
        )
        self.assertEqual(name, "/lidar_points")
        self.assertIn("известный", reason)


class Variant(unittest.TestCase):
    def test_b_and_c_come_from_config(self):
        config = load_batch_config()
        self.assertEqual(config["default_variant"], "C")
        b_flags = variant_flags(config, "B")
        c_flags = variant_flags(config, "C")
        self.assertEqual(b_flags["connectivity"], "distance")
        self.assertFalse(b_flags["local_protrusions_enabled"])
        self.assertTrue(c_flags["local_protrusions_enabled"])
        self.assertIn("s=-Y", config["coordinate_prior"]["mapping"])


class Resume(unittest.TestCase):
    def test_completed_line_is_kept(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frames.jsonl"
            append_record(path, {"frame_index": 0, "status": "ok", "candidates": []})
            before = path.read_text(encoding="utf-8")
            found = read_completed(path)
            self.assertEqual(found[0]["status"], "ok")
            self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_partial_tail_is_dropped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frames.jsonl"
            path.write_text('{"frame_index": 0, "status": "ok"}\n{"frame_index": 1, "sta', encoding="utf-8")
            found = read_completed(path)
            self.assertEqual(list(found), [0])
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))


class ResumeIdentity(unittest.TestCase):
    def test_changed_config_refuses_resume(self):
        from obstacle_detector.batch import append_record, read_completed, resume_is_compatible, run_identity

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            config = folder / "config"
            config.mkdir()
            bag = folder / "bag"
            bag.mkdir()
            for name in ("detector.yaml", "corridor.yaml", "batch.yaml", "pipeline.yaml"):
                (config / name).write_text("value: 1\n", encoding="utf-8")
            (bag / "metadata.yaml").write_text("message_count: 1\n", encoding="utf-8")
            meta = {"message_count": 1}
            first = run_identity("abc", config, "C", "/lidar_points", bag, meta)
            path = folder / "frames.jsonl"
            append_record(path, {"frame_index": 0, "run_identity": first, "status": "ok"})
            before = path.read_text(encoding="utf-8")
            (config / "detector.yaml").write_text("value: 2\n", encoding="utf-8")
            second = run_identity("abc", config, "C", "/lidar_points", bag, meta)
            self.assertNotEqual(first, second)
            with self.assertRaises(SystemExit):
                resume_is_compatible(read_completed(path), second)
            self.assertEqual(path.read_text(encoding="utf-8"), before)


class Coordinates(unittest.TestCase):
    def test_confirmed_mount_is_shared_by_geometry_and_membership(self):
        from obstacle_detector.corridor import geometry_working, sensor_to_train, sensor_to_working
        from obstacle_detector.frame_pipeline import build_frame_geometry

        mount = {
            "confirmed": True,
            "source": "manual_estimate",
            "basis": "проверка согласованности, не калибровка проезда",
            "tx_m": 0.4,
            "ty_m": -0.2,
            "tz_m": 0.1,
            "roll_rad": 0.0,
            "pitch_rad": 0.05,
            "yaw_rad": 0.2,
        }
        x = np.array([0.2, 1.0, -0.4], dtype=np.float64)
        y = np.array([-8.0, -12.0, -6.0], dtype=np.float64)
        z = np.array([-1.0, -0.8, -1.2], dtype=np.float64)
        groups = {"x": x, "y": y, "z": z}
        built = build_frame_geometry(groups, mount)
        expected_s, expected_l, expected_h, frame = geometry_working(x, y, z, mount)
        self.assertEqual(frame, "train_working")
        self.assertEqual(built["coordinate_frame"], "train_working")
        np.testing.assert_allclose(built["s"], expected_s)
        np.testing.assert_allclose(built["l"], expected_l)
        np.testing.assert_allclose(built["h"], expected_h)
        raw_s, raw_l, raw_h = sensor_to_working(x, y, z)
        self.assertGreater(float(np.max(np.abs(built["s"] - raw_s))), 0.05)
        moved_x, moved_y, moved_z = sensor_to_train(x, y, z, mount)
        same_s, same_l, same_h, same_frame = geometry_working(moved_x, moved_y, moved_z, {"confirmed": False})
        self.assertEqual(same_frame, "sensor_working_prior_mount_unconfirmed")
        np.testing.assert_allclose(same_s, expected_s)
        np.testing.assert_allclose(same_l, expected_l)
        np.testing.assert_allclose(same_h, expected_h)

    def test_unconfirmed_mount_keeps_the_sensor_prior(self):
        from obstacle_detector.corridor import sensor_to_working
        from obstacle_detector.frame_pipeline import build_frame_geometry

        x = np.array([0.0, 0.5])
        y = np.array([-10.0, -11.0])
        z = np.array([-1.0, -1.0])
        built = build_frame_geometry({"x": x, "y": y, "z": z})
        s, l, h = sensor_to_working(x, y, z)
        np.testing.assert_allclose(built["s"], s)
        np.testing.assert_allclose(built["l"], l)
        self.assertEqual(built["coordinate_frame"], "sensor_working_prior_mount_unconfirmed")


class PictureRule(unittest.TestCase):
    def test_max_is_per_hypothesis(self):
        records = [
            {
                "frame_index": 3,
                "status": "ok",
                "hypotheses": [
                    {"hypothesis_id": "H1", "candidate_counts": {"volume_large": 2, "volume_rare": 0, "protrusion_large": 0, "protrusion_rare": 0}},
                    {"hypothesis_id": "H2", "candidate_counts": {"volume_large": 4, "volume_rare": 0, "protrusion_large": 0, "protrusion_rare": 0}},
                ],
            },
            {
                "frame_index": 1,
                "status": "insufficient_geometry",
                "hypotheses": [],
            },
            {
                "frame_index": 2,
                "status": "insufficient_geometry",
                "hypotheses": [],
            },
        ]
        chosen = select_extra_frames(records)
        self.assertEqual(chosen["max_candidates_on_one_hypothesis"]["count"], 4)
        self.assertEqual(chosen["max_candidates_on_one_hypothesis"]["frame_index"], 3)
        self.assertEqual(chosen["most_common_geometry_failure"]["first_frame_index"], 1)
        self.assertTrue(chosen["candidate_sum_was_not_used"])


if __name__ == "__main__":
    unittest.main()
