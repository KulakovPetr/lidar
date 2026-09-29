import json
import tempfile
import unittest
from pathlib import Path

from obstacle_detector.batch import append_record, read_completed, resume_is_compatible
from obstacle_detector.check_result import compare
from obstacle_detector.offline_run import content_fingerprint, exit_code, load_mount_file, main, run_identity, select_topic


class Topic(object):
    def __init__(self, name, type_name):
        self.name = name
        self.type = type_name


class OfflineContractTests(unittest.TestCase):
    def test_single_pointcloud_is_selected(self):
        name, reason = select_topic([Topic("/sensing/lidar/hesai128/pointcloud", "sensor_msgs/msg/PointCloud2")], None)
        self.assertEqual(name, "/sensing/lidar/hesai128/pointcloud")
        self.assertIn("единственный", reason)

    def test_several_topics_require_an_explicit_choice(self):
        topics = [
            Topic("/lidar_points", "sensor_msgs/msg/PointCloud2"),
            Topic("/sensing/lidar/hesai128/pointcloud", "sensor_msgs/msg/PointCloud2"),
        ]
        with self.assertRaises(SystemExit):
            select_topic(topics, None)
        chosen, _reason = select_topic(topics, "/lidar_points")
        self.assertEqual(chosen, "/lidar_points")

    def test_zero_mount_is_not_a_silent_calibration(self):
        root = Path(__file__).resolve().parents[1]
        corridor, mount = load_mount_file(root / "config" / "mount_unconfirmed.yaml")
        self.assertFalse(mount["applied"])
        self.assertFalse(corridor["mount"]["confirmed"])

    def test_confirmed_zeros_are_rejected(self):
        text = """
mount:
  confirmed: true
  source: organizer
  status: stated
  basis: "проверка"
  translation_m: {x: 0.0, y: 0.0, z: 0.0}
  rpy_rad: {roll: 0.0, pitch: 0.0, yaw: 0.0}
"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mount.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(SystemExit):
                load_mount_file(path)

    def test_path_is_not_part_of_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "one"
            second = Path(folder) / "other place"
            first.mkdir()
            second.mkdir()
            payload = b"same-bytes"
            (first / "a.db3").write_bytes(payload)
            (second / "a.db3").write_bytes(payload)
            self.assertEqual(content_fingerprint(first), content_fingerprint(second))
            self.assertEqual(
                run_identity("v", "c", "/lidar_points", content_fingerprint(first), "reader"),
                run_identity("v", "c", "/lidar_points", content_fingerprint(second), "reader"),
            )
            self.assertNotEqual(
                run_identity("v", "c", "/lidar_points", content_fingerprint(first), "reader-a"),
                run_identity("v", "c", "/lidar_points", content_fingerprint(first), "reader-b"),
            )

    def test_parameter_change_does_not_append(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frames.jsonl"
            append_record(path, {"frame_index": 0, "run_identity": "aaaa"})
            with self.assertRaises(SystemExit):
                resume_is_compatible(read_completed(path), "bbbb")

    def test_partial_line_is_not_a_duplicate_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frames.jsonl"
            append_record(path, {"frame_index": 0, "run_identity": "aaaa"})
            with path.open("a", encoding="utf-8") as handle:
                handle.write("{\"frame_index\": 1")
            completed = read_completed(path)
            self.assertEqual(set(completed), {0})
            text = path.read_text(encoding="utf-8")
            self.assertFalse(text.rstrip().endswith("1"))

    def test_negative_count_is_rejected(self):
        with self.assertRaises(SystemExit):
            main(["--bag", "bag", "--output", "out", "--config", "mount.yaml", "--count", "-1"])

    def test_processing_errors_are_a_failed_exit(self):
        self.assertEqual(exit_code({"errors": 0}), 0)
        self.assertEqual(exit_code({"errors": 2}), 1)


def _record(index, stamp="1", frame="hesai_lidar", config="cfg", range_m=10.0):
    return {
        "frame_index": index,
        "header_stamp_ns": stamp,
        "header_frame_id": frame,
        "config_version": config,
        "status": "ok",
        "coordinate_frame": "sensor_working_prior_mount_unconfirmed",
        "geometry_quality": {"hypotheses": [{"hypothesis_id": "H1", "kind": "ambiguous", "geometry_applicable": True, "observed_s_min_m": 6.0, "observed_s_max_m": 10.0}]},
        "candidates": [{"hypothesis_id": "H1", "candidate_id": "L1", "kind": "large", "branch": "volume_or_suspended", "source_indices": [1, 2], "range_from_lidar_m": range_m}],
    }


class CheckResultTests(unittest.TestCase):
    def _write(self, folder, name, records):
        path = Path(folder) / name
        path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
        return path

    def test_empty_duplicate_and_unexpected_messages_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            empty = Path(folder) / "empty.jsonl"
            empty.write_text("", encoding="utf-8")
            reference = self._write(folder, "reference.jsonl", [_record(0), _record(1)])
            self.assertTrue(any("пустой" in item for item in compare(reference, empty)))
            duplicate = self._write(folder, "duplicate.jsonl", [_record(0), _record(0)])
            self.assertTrue(any("дубль" in item for item in compare(reference, duplicate)))
            other = self._write(folder, "other.jsonl", [_record(0), _record(2)])
            self.assertTrue(any("неожиданный набор" in item for item in compare(reference, other)))

    def test_stamp_frame_and_config_must_match(self):
        with tempfile.TemporaryDirectory() as folder:
            reference = self._write(folder, "reference.jsonl", [_record(0)])
            changed = self._write(folder, "changed.jsonl", [_record(0, stamp="9", frame="other", config="other")])
            errors = " ".join(compare(reference, changed))
            self.assertIn("timestamp", errors)
            self.assertIn("конфигурация", errors)

    def test_non_finite_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            reference = self._write(folder, "reference.jsonl", [_record(0)])
            actual = self._write(folder, "actual.jsonl", [_record(0, range_m=float("nan"))])
            self.assertTrue(any("не конечна" in item for item in compare(reference, actual)))

    def test_matching_pair_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            reference = self._write(folder, "reference.jsonl", [_record(0), _record(1, range_m=11.0)])
            actual = self._write(folder, "actual.jsonl", [_record(0), _record(1, range_m=11.0)])
            self.assertEqual(compare(reference, actual), [])


class _Stamp(object):
    sec = 10
    nanosec = 20


class _Header(object):
    stamp = _Stamp()
    frame_id = "hesai_lidar"


class _Field(object):
    def __init__(self, name, offset):
        self.name = name
        self.offset = offset
        self.datatype = 7
        self.count = 1


class _Cloud(object):
    def __init__(self, payload):
        self.header = _Header()
        self.height = 1
        self.width = 1
        self.fields = [_Field("x", 0), _Field("y", 4), _Field("z", 8), _Field("intensity", 12)]
        self.is_bigendian = False
        self.point_step = 16
        self.row_step = 16
        self.data = payload


class MissingOptionalFieldTests(unittest.TestCase):
    def test_cloud_without_ring_or_point_time_decodes(self):
        import struct

        import numpy as np

        from obstacle_detector.bag_io import decode_message

        payload = np.frombuffer(struct.pack("<ffff", 1.0, -2.0, 0.5, 3.0), dtype=np.uint8).copy()
        decoded = decode_message(_Cloud(payload))
        self.assertIsNone(decoded["ring"])
        self.assertIsNone(decoded["timestamp"])
        self.assertEqual(decoded["absent_fields"], ["ring", "timestamp"])
        self.assertAlmostEqual(float(decoded["y"][0]), -2.0)
        self.assertEqual(decoded["frame_id"], "hesai_lidar")
        from obstacle_detector.repeat_groups import group_exact_xyz, valid_mask

        valid = valid_mask(decoded["x"], decoded["y"], decoded["z"])
        groups = group_exact_xyz(
            decoded["x"], decoded["y"], decoded["z"], decoded["intensity"], decoded["ring"], decoded["timestamp"], valid
        )
        self.assertFalse(groups["ring_observed"])
        self.assertFalse(groups["timestamp_observed"])
        self.assertTrue(np.all(groups["ring_nunique"] < 0))
        self.assertTrue(np.all(np.isnan(groups["timestamp_span"])))
