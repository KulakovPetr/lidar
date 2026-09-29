import struct
import unittest
from pathlib import Path

import numpy as np

from obstacle_detector.angular_projection import (
    AZIMUTH_BIN_DEG,
    bin_center_deg,
    display_row_to_internal,
    indices_in_bin,
    project_ring_rows,
    project_xyz,
    xyz_angles_deg,
)
from obstacle_detector.frame_view import write_png
from obstacle_detector.manual_marks import evidence_rows, mark_from_pixel, validate_mark
from obstacle_detector.struct_cloud import read_points_struct
from obstacle_detector.pointcloud import FLOAT32, FLOAT64, UINT16, FieldSpec, decode_cloud
from obstacle_detector.view_selection import load_run_roles, load_view_frames, require_development


def _cloud():
    # Two returns along +X, one far off the ±20° sector, one zero, one NaN.
    # Ring values are the field, not an elevation index.
    x = np.array([10.0, 12.0, 0.0, 0.0, np.nan, 10.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 10.0, 0.0, 1.0, 0.0], dtype=np.float32)
    z = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    intensity = np.array([1, 2, 3, 0, 0, 4], dtype=np.float32)
    ring = np.array([5, 5, 9, 0, 1, 40], dtype=np.uint16)
    return x, y, z, intensity, ring


class ProjectionTests(unittest.TestCase):
    def test_second_return_is_kept_and_back_projection_drops_nothing(self) -> None:
        x, y, z, intensity, ring = _cloud()
        spec, loss, grid = project_xyz(x, y, z, intensity, ring)
        self.assertEqual(spec["representation"], "xyz_angular_projection")
        self.assertIsNone(spec["calibration"])
        self.assertFalse(spec["permanent_pm20_crop"])
        self.assertEqual(spec["deskew"], "not_applied")
        self.assertFalse(spec["ring_derived_from_xyz_after_deskew"])
        self.assertEqual(loss["zero_range_missing"], 1)
        self.assertEqual(loss["nonfinite"], 1)
        self.assertEqual(loss["valid"], 4)
        self.assertEqual(loss["index_dropped"], 0)
        self.assertEqual(loss["index_duplicated"], 0)
        self.assertGreaterEqual(loss["collision_bins"], 1)
        self.assertGreaterEqual(loss["multiple_return_bins"], 1)
        ids = indices_in_bin(grid, spec["n_cols"], int(grid["row"][0]), int(grid["col"][0]))
        self.assertCountEqual(ids.tolist(), [0, 1])
        self.assertEqual(loss["points_outside_abs_azimuth_20deg"], 1)
        self.assertEqual(loss["points_outside_abs_azimuth_20deg_recovered"], 1)
        self.assertEqual(spec["azimuth_limits_deg"], [-180.0, 180.0])
        self.assertEqual(spec["n_cols"], int(round(360.0 / AZIMUTH_BIN_DEG)))
        self.assertGreater(loss["quantization_error_m"]["max"], 0.0)

    def test_ring_row_is_the_field_not_an_xyz_angle(self) -> None:
        x, y, z, intensity, ring = _cloud()
        _spec, _loss, grid = project_xyz(x, y, z, intensity, ring)
        packed, reason = project_ring_rows(grid["azimuth_deg"], grid["valid"], ring)
        self.assertIn("поля", reason)
        self.assertFalse(packed["spec"]["nominal_manual_angles_applied"])
        self.assertFalse(packed["spec"]["ring_derived_from_xyz_after_deskew"])
        self.assertEqual(int(packed["grid"]["row"][5]), 40)
        self.assertNotEqual(int(grid["row"][5]), 40)
        self.assertEqual(packed["loss"]["index_dropped"], 0)

    def test_display_row_flip_and_bin_center(self) -> None:
        spec = {
            "n_rows": 10,
            "azimuth_bin_deg": 0.2,
            "elevation_bin_deg": 0.2,
            "elevation_min_deg": -1.0,
        }
        self.assertEqual(display_row_to_internal(0, 10), 9)
        azimuth, elevation = bin_center_deg(0, 0, spec)
        self.assertAlmostEqual(azimuth, -180.0 + 0.1)
        self.assertAlmostEqual(elevation, -1.0 + 0.1)

    def test_angle_convention_wraps_negative_x(self) -> None:
        azimuth, elevation = xyz_angles_deg(
            np.array([-10.0]), np.array([0.0]), np.array([0.0])
        )
        self.assertAlmostEqual(float(azimuth[0]), -180.0)
        self.assertAlmostEqual(float(elevation[0]), 0.0)


class DecodeTests(unittest.TestCase):
    def test_one_zero_coordinate_stays_valid(self) -> None:
        from obstacle_detector.angular_projection import classify_ranges

        x = np.array([0.0, 5.0, 0.0, 0.0], dtype=np.float32)
        y = np.array([0.0, 0.0, 7.0, 0.0], dtype=np.float32)
        z = np.array([0.0, 0.0, 0.0, 0.4], dtype=np.float32)
        _finite, zero, valid, _range = classify_ranges(x, y, z)
        self.assertTrue(zero[0] and not valid[0])
        self.assertTrue(valid[1] and valid[2] and valid[3])

    def test_struct_reader_matches_decoder_including_padding(self) -> None:
        point_a = struct.pack("<ffff", 3.0, 0.0, 1.0, 8.0) + struct.pack("<H", 4) + struct.pack("<d", 1.5)
        point_b = struct.pack("<ffff", 0.0, 0.0, 0.0, 0.0) + struct.pack("<H", 9) + struct.pack("<d", 2.5)
        raw = b"".join((point_a, b"\xab\xab", point_b, b"\xab\xab"))
        fields = [
            FieldSpec("x", 0, FLOAT32, 1),
            FieldSpec("y", 4, FLOAT32, 1),
            FieldSpec("z", 8, FLOAT32, 1),
            FieldSpec("intensity", 12, FLOAT32, 1),
            FieldSpec("ring", 16, UINT16, 1),
        ]
        decoded, _layout = decode_cloud(raw, 2, 1, 26, 28, False, fields)
        got = read_points_struct(
            raw, 2, 1, 26, 28, False,
            [(f.name, f.offset, f.datatype, f.count) for f in fields],
            [0, 1],
        )
        self.assertEqual(got[0]["y"], 0.0)
        self.assertNotEqual(got[0]["x"], 0.0)
        self.assertEqual(got[1]["x"], 0.0)
        self.assertEqual(got[1]["y"], 0.0)
        self.assertEqual(got[1]["z"], 0.0)
        for index, record in enumerate(got):
            for name in ("x", "y", "z", "intensity"):
                self.assertEqual(float(decoded[name][index]), float(record[name]))
            self.assertEqual(int(decoded["ring"][index]), int(record["ring"]))

    def test_thin_fragment_and_background_stay_in_the_index_list(self) -> None:
        # Same angular cell: a far return and two closer points of a thin fragment.
        # A point 1° away must land elsewhere. None of the three are replaced by a bin center.
        x = np.array([20.0, 8.0, 8.05, 20.0], dtype=np.float64)
        y = np.array([0.0, 0.01, 0.01, 0.35], dtype=np.float64)
        z = np.zeros(4, dtype=np.float64)
        spec, loss, grid = project_xyz(x, y, z, None, None)
        self.assertEqual(spec["azimuth_bin_deg"], 0.2)
        self.assertEqual(loss["index_dropped"], 0)
        shared = indices_in_bin(grid, spec["n_cols"], int(grid["row"][0]), int(grid["col"][0]))
        self.assertCountEqual(shared.tolist(), [0, 1, 2])
        other = indices_in_bin(grid, spec["n_cols"], int(grid["row"][3]), int(grid["col"][3]))
        self.assertEqual(other.tolist(), [3])
        self.assertTrue(np.array_equal(x[shared], x[np.array(shared, dtype=np.int64)]))
        self.assertGreater(loss["quantization_error_m"]["max"], 0.0)

    def test_model_proposal_is_not_a_human_mark(self) -> None:
        with self.assertRaises(ValueError):
            validate_mark(
                {
                    "source": "model_proposal",
                    "surface": "rail",
                    "visibility": "visible",
                    "confidence": "uncertain",
                    "coordinate_frame": "sensor_xyz",
                    "point_indices": [1],
                    "range_m_by_index": [4.0],
                }
            )


class DecodeLayoutTests(unittest.TestCase):
    def test_unaligned_float64_and_row_padding(self) -> None:
        point = struct.pack("<ffff", 1.5, -2.5, 3.5, 9.0) + struct.pack("<H", 7) + struct.pack("<d", 12.25)
        self.assertEqual(len(point), 26)
        raw = point + point
        fields = [
            FieldSpec("x", 0, FLOAT32, 1),
            FieldSpec("y", 4, FLOAT32, 1),
            FieldSpec("z", 8, FLOAT32, 1),
            FieldSpec("intensity", 12, FLOAT32, 1),
            FieldSpec("ring", 16, UINT16, 1),
            FieldSpec("timestamp", 18, FLOAT64, 1),
        ]
        decoded, layout = decode_cloud(raw, 1, 2, 26, 26, False, fields)
        self.assertEqual(decoded["x"].dtype, np.float32)
        self.assertTrue(np.allclose(decoded["x"], [1.5, 1.5]))
        self.assertEqual(int(decoded["ring"][0]), 7)
        self.assertAlmostEqual(float(decoded["timestamp"][0]), 12.25)
        self.assertEqual(layout[-1]["datatype"], "float64")

        # Big-endian float32 must not be read as little-endian.
        be = struct.pack(">f", 2.0)
        got, _ = decode_cloud(be, 1, 1, 4, 4, True, [FieldSpec("x", 0, FLOAT32, 1), FieldSpec("y", 0, FLOAT32, 1), FieldSpec("z", 0, FLOAT32, 1)])
        # y and z alias x in this fixture; the endian check is on the value.
        self.assertAlmostEqual(float(got["x"][0]), 2.0, places=5)

        row = b"".join(
            (
                struct.pack("<fff", 1.0, 0.0, 0.0),
                b"\xff\xff\xff\xff",
                struct.pack("<fff", 3.0, 0.0, 0.0),
                b"\xff\xff\xff\xff",
            )
        )
        decoded, _ = decode_cloud(
            row,
            2,
            1,
            12,
            16,
            False,
            [FieldSpec("x", 0, FLOAT32, 1), FieldSpec("y", 4, FLOAT32, 1), FieldSpec("z", 8, FLOAT32, 1)],
        )
        self.assertTrue(np.allclose(decoded["x"], [1.0, 3.0]))
        self.assertNotAlmostEqual(float(decoded["x"][1]), struct.unpack("<f", b"\xff\xff\xff\xff")[0])


class MarkTests(unittest.TestCase):
    def test_algorithm_proposal_is_not_a_label(self) -> None:
        with self.assertRaises(ValueError):
            validate_mark(
                {
                    "source": "algorithm",
                    "surface": "rail",
                    "visibility": "visible",
                    "confidence": "certain",
                    "coordinate_frame": "sensor_xyz",
                    "point_indices": [1],
                    "range_m_by_index": [4.0],
                }
            )

    def test_pixel_mark_keeps_both_returns_and_only_its_range_band(self) -> None:
        x, y, z, intensity, ring = _cloud()
        spec, _loss, grid = project_xyz(x, y, z, intensity, ring)
        internal = int(grid["row"][0])
        col = int(grid["col"][0])
        display_row = spec["n_rows"] - 1 - internal
        mark = mark_from_pixel(
            grid, spec, display_row, col, "rail", "visible", "uncertain", "tester", "оба возврата"
        )
        self.assertEqual(mark["point_indices"], [0, 1])
        self.assertEqual(mark["ring_by_index"], [5, 5])
        frame = {
            "frame_key": "dev/msg_000000",
            "run": "dev",
            "message_index": 0,
            "frame_id": "hesai_lidar",
            "header_stamp_ns": 1,
        }
        rows = evidence_rows([frame], {"dev/msg_000000": [mark]})
        rail_10 = [row for row in rows if row["surface"] == "rail" and row["range_band"] == "[10,30)"]
        self.assertEqual(rail_10[0]["status"], "видимая ручная отметка")
        self.assertEqual(rail_10[0]["point_indices"], [0, 1])
        rail_0 = [row for row in rows if row["surface"] == "rail" and row["range_band"] == "[0,10)"]
        self.assertEqual(rail_0[0]["status"], "нет ручной отметки")
        floor = [row for row in rows if row["surface"] == "floor" and row["range_band"] == "[10,30)"]
        self.assertEqual(floor[0]["status"], "нет ручной отметки")

    def test_empty_table_does_not_call_a_surface_invisible(self) -> None:
        frame = {
            "frame_key": "dev/msg_000000",
            "run": "dev",
            "message_index": 0,
            "frame_id": "hesai_lidar",
            "header_stamp_ns": 1,
        }
        rows = evidence_rows([frame], {})
        self.assertTrue(all(row["status"] == "нет ручной отметки" for row in rows))


class SelectionTests(unittest.TestCase):
    def test_locked_and_validation_are_refused(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath("config", "pipeline.yaml").read_text(encoding="utf-8")
        roles = load_run_roles(text)
        self.assertEqual(roles["doubleT_obstacle"], "locked_test")
        self.assertEqual(roles["squareT_platform_squareT_switch"], "validation")
        require_development("doubleT_platform", roles)
        with self.assertRaises(ValueError):
            require_development("doubleT_obstacle", roles)
        with self.assertRaises(ValueError):
            require_development("squareT_platform_squareT_switch", roles)
        frames = load_view_frames(text)
        self.assertGreaterEqual(len(frames), 1)
        for run, _index in frames:
            require_development(run, roles)

    def test_png_signature(self) -> None:
        path = Path(__file__).resolve().parent / "_tiny.png"
        try:
            write_png(path, np.zeros((2, 3, 3), dtype=np.uint8))
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
