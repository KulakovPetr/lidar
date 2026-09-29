import tempfile
import unittest
from pathlib import Path

import numpy as np

from obstacle_detector.corridor import (
    attachment_is_reliable,
    build_path_corridor,
    choose_own_path,
    load_session,
    match_session_attachment,
    session_variant,
    classify_sensor_points,
    config_path,
    coordinate_contract,
    elevated_structure_absence,
    expand_group_results,
    geometry_working,
    load_config,
    mount_from_config,
    mount_is_applied,
    sensor_to_train,
    sensor_to_working,
    surface_object_overlap_m,
    train_to_sensor,
    working_to_sensor,
)


def _ground():
    return {
        "status": "candidate",
        "ambiguous_with_nearby_surface": False,
        "local_bins": [
            {
                "s_lo": 0.0,
                "s_hi": 40.0,
                "intercept": -1.0,
                "slope_dh_dl": 0.0,
                "l_min": -20.0,
                "l_max": 20.0,
            }
        ],
        "fitting_residual": {"p95_m": 0.05},
    }


def _config():
    config = load_config(config_path())
    config["lower_boundary_above_reference_m"] = {
        "value": 0.0,
        "source": "manual_estimate",
        "status": "stated",
        "basis": "test floor",
    }
    return config


def _mount(**changes):
    mount = {
        "confirmed": False,
        "source": "unknown",
        "status": "unconfirmed",
        "basis": "",
        "tx_m": 0.0,
        "ty_m": 0.0,
        "tz_m": 0.0,
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "yaw_rad": 0.0,
    }
    mount.update(changes)
    return mount


def _hypothesis(stations, kind="consistent_rail_pair_hypothesis", lateral=None):
    rows = []
    for s_m in stations:
        center = 0.0 if lateral is None else float(lateral(s_m))
        rows.append(
            {
                "s_m": float(s_m),
                "left_l_m": center - 0.75,
                "right_l_m": center + 0.75,
            }
        )
    return {
        "hypothesis_id": "H1",
        "kind": kind,
        "engineering_score": 0.8,
        "observed_sections": rows,
        "shared_with": [],
        "distance_m": 1.5,
        "distance_scatter_m": 0.02,
    }


def _sensor(s, l, h):
    x, y, z = working_to_sensor(s, l, h)
    return np.asarray(x), np.asarray(y), np.asarray(z)


class CoordinateTests(unittest.TestCase):
    def test_contract_names_the_three_frames(self):
        contract = coordinate_contract()
        self.assertIn("Rz(yaw) @ Ry(pitch) @ Rx(roll)", contract["sensor_to_train"]["formula"])
        self.assertIn("[s, l, h]", contract["sensor_to_working"]["formula"])
        self.assertFalse(contract["train"]["pair_midpoint_is_body_center"])

    def test_working_roundtrip(self):
        rng = np.random.default_rng(4)
        points = rng.normal(size=(20, 3))
        s, l, h = sensor_to_working(points[:, 0], points[:, 1], points[:, 2])
        back = np.column_stack(working_to_sensor(s, l, h))
        self.assertLess(np.max(np.abs(back - points)), 1e-12)

    def test_mount_roundtrip(self):
        rng = np.random.default_rng(5)
        points = rng.normal(size=(30, 3))
        mount = _mount(
            confirmed=True,
            source="manual_estimate",
            basis="стенд",
            tx_m=0.2,
            ty_m=-0.1,
            tz_m=1.4,
            roll_rad=0.05,
            pitch_rad=-0.08,
            yaw_rad=0.2,
        )
        train = np.column_stack(sensor_to_train(points[:, 0], points[:, 1], points[:, 2], mount))
        back = np.column_stack(train_to_sensor(train[:, 0], train[:, 1], train[:, 2], mount))
        self.assertLess(np.max(np.abs(back - points)), 1e-9)

    def test_unconfirmed_numbers_are_not_applied(self):
        mount = _mount(confirmed=False, source="unknown", tx_m=0.4, yaw_rad=0.3)
        self.assertFalse(mount_is_applied(mount))
        s, l, h, frame = geometry_working([0.0], [-15.0], [-1.0], mount)
        self.assertEqual(frame, "sensor_working_prior_mount_unconfirmed")
        self.assertAlmostEqual(float(l[0]), 0.0)
        self.assertAlmostEqual(float(s[0]), 15.0)

    def test_confirmed_shift_and_yaw_move_the_train_frame(self):
        mount = _mount(confirmed=True, source="manual_estimate", basis="проверка", tx_m=0.4, yaw_rad=0.0)
        _s, l, _h, frame = geometry_working([0.0], [-15.0], [-1.0], mount)
        self.assertEqual(frame, "train_working")
        self.assertAlmostEqual(float(l[0]), 0.4, places=6)
        yaw = _mount(confirmed=True, source="organizer", basis="проверка", yaw_rad=0.1)
        s2, l2, _h2, _frame = geometry_working([0.0], [-15.0], [-1.0], yaw)
        self.assertGreater(abs(float(l2[0])), 0.1)
        self.assertNotAlmostEqual(float(s2[0]), 15.0, places=3)

    def test_consistent_coordinate_change_keeps_the_profile_position(self):
        config = _config()
        corridor = build_path_corridor(_hypothesis([6, 10, 15, 20]), _ground(), config, _mount())
        x, y, z = _sensor([15.0], [0.2], [-0.5])
        before = classify_sensor_points(x, y, z, corridor, _mount(), config)
        shift, yaw = 0.35, 0.12
        cosine, sine = float(np.cos(yaw)), float(np.sin(yaw))
        x_s = x + shift
        x2 = cosine * x_s - sine * y
        y2 = sine * x_s + cosine * y
        undone = _mount(
            confirmed=True,
            source="manual_estimate",
            basis="обратное преобразование проверки",
            tx_m=-shift,
            yaw_rad=-yaw,
        )
        after = classify_sensor_points(x2, y2, z, corridor, undone, config)
        self.assertAlmostEqual(float(before["profile_lateral_m"][0]), float(after["profile_lateral_m"][0]), places=6)
        self.assertAlmostEqual(float(before["profile_vertical_m"][0]), float(after["profile_vertical_m"][0]), places=6)
        self.assertEqual(before["relation"][0], after["relation"][0])

    def test_file_has_no_frame_overrides(self):
        load_config(config_path())
        text = "frames:\n  doubleT_platform/0: {}\n"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "corridor.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_zero_mount_in_the_file_is_not_a_calibration(self):
        mount = mount_from_config(load_config(config_path()))
        self.assertFalse(mount["applied"])
        self.assertFalse(mount["identity_is_calibration"])
        self.assertEqual(mount["source"], "unknown")


class ProfileTests(unittest.TestCase):
    def test_half_width_is_not_doubled(self):
        config = _config()
        mount = _mount()
        corridor = build_path_corridor(_hypothesis([6, 10, 15, 20]), _ground(), config, mount)
        self.assertAlmostEqual(corridor["half_width_m"], 1.05)
        self.assertAlmostEqual(corridor["width_m"], 2.1)
        x, y, z = _sensor([15.0, 15.0, 15.0], [1.05, 1.06, 2.1], [-0.5, -0.5, -0.5])
        out = classify_sensor_points(x, y, z, corridor, mount, config)
        self.assertEqual(list(out["relation"]), ["inside", "outside", "outside"])
        self.assertTrue(out["inside_is_not_an_obstacle"])

    def test_vertical_bounds(self):
        config = _config()
        mount = _mount()
        corridor = build_path_corridor(_hypothesis([6, 10, 15, 20]), _ground(), config, mount)
        heights = [-1.0, -0.5, 2.0, 2.01]
        x, y, z = _sensor(np.full(4, 15.0), np.zeros(4), heights)
        out = classify_sensor_points(x, y, z, corridor, mount, config)
        self.assertEqual(list(out["relation"]), ["inside", "inside", "inside", "outside"])

    def test_straight_and_curved_paths(self):
        config = _config()
        mount = _mount()
        straight = build_path_corridor(_hypothesis([6, 10, 15, 20]), _ground(), config, mount)
        center = classify_sensor_points(*_sensor([15.0], [0.0], [-0.5]), straight, mount, config)
        self.assertAlmostEqual(float(center["profile_lateral_m"][0]), 0.0, places=6)
        self.assertEqual(center["relation"][0], "inside")
        curve = build_path_corridor(
            _hypothesis([6, 10, 15, 20, 25], lateral=lambda s: 0.01 * (s - 15.0) ** 2),
            _ground(),
            config,
            mount,
        )
        on_curve = classify_sensor_points(*_sensor([20.0], [0.25], [-0.5]), curve, mount, config)
        self.assertAlmostEqual(float(on_curve["profile_lateral_m"][0]), 0.0, places=5)
        self.assertEqual(on_curve["relation"][0], "inside")
        segment = next(item for item in curve["segments"] if abs(item["s0"] - 15.0) < 1e-6 and abs(item["s1"] - 20.0) < 1e-6)
        ds = segment["s1"] - segment["s0"]
        dl = segment["l1"] - segment["l0"]
        length = float(np.hypot(ds, dl))
        weight = (19.6 - segment["s0"]) / (segment["s1"] - segment["s0"])
        foot_s = 19.6
        foot_l = segment["l0"] + weight * (segment["l1"] - segment["l0"])
        offset = 1.2
        query_s = foot_s + offset * (-dl / length)
        query_l = foot_l + offset * (ds / length)
        side = classify_sensor_points(*_sensor([query_s], [query_l], [-0.5]), curve, mount, config)
        self.assertAlmostEqual(float(side["profile_lateral_m"][0]), offset, places=5)
        self.assertEqual(side["relation"][0], "outside")

    def test_gap_and_unresolved_hypothesis_are_uncertain(self):
        config = _config()
        mount = _mount()
        gap = build_path_corridor(_hypothesis([6, 10, 20]), _ground(), config, mount)
        out = classify_sensor_points(*_sensor([15.0], [0.0], [-0.5]), gap, mount, config)
        self.assertEqual(out["support"][0], "gap_interpolation")
        self.assertEqual(out["relation"][0], "uncertain")
        self.assertEqual(out["uncertain_reason"][0], "gap_interpolation")
        ambiguous = build_path_corridor(
            _hypothesis([6, 10, 15, 20], kind="ambiguous"),
            _ground(),
            config,
            mount,
        )
        unresolved = classify_sensor_points(*_sensor([15.0], [0.0], [-0.5]), ambiguous, mount, config)
        self.assertEqual(unresolved["relation"][0], "inside")
        self.assertTrue(bool(unresolved["conditional"][0]))
        self.assertIn("гипотеза пути не согласована", unresolved["conditional_reasons"])

    def test_observed_station_wins_a_tie_with_extrapolation(self):
        config = _config()
        corridor = build_path_corridor(_hypothesis([6, 10, 15, 20]), _ground(), config, _mount())
        end = classify_sensor_points(*_sensor([6.0], [0.0], [-0.8]), corridor, _mount(), config)
        self.assertEqual(end["support"][0], "observed")
        self.assertEqual(end["relation"][0], "inside")
        past = classify_sensor_points(*_sensor([5.7], [0.0], [-0.8]), corridor, _mount(), config)
        self.assertEqual(past["support"][0], "extrapolated")
        self.assertEqual(past["relation"][0], "uncertain")

    def test_path_is_not_extended_for_hundreds_of_metres(self):
        config = _config()
        corridor = build_path_corridor(_hypothesis([6, 10, 15, 20]), _ground(), config, _mount())
        self.assertLessEqual(max(item["s_m"] for item in corridor["vertices"]), 25.0)
        self.assertIn(200.0, corridor["applicability"]["forbidden_confirmed_horizon_m"])
        far = classify_sensor_points(*_sensor([220.0], [0.0], [-0.5]), corridor, _mount(), config)
        self.assertEqual(far["support"][0], "unsupported")
        self.assertEqual(far["relation"][0], "uncertain")
        both = classify_sensor_points(*_sensor([15.0, 220.0], [0.0, 0.0], [-0.5, -0.5]), corridor, _mount(), config)
        full = classify_sensor_points(
            *_sensor([15.0, 220.0], [0.0, 0.0], [-0.5, -0.5]),
            corridor,
            _mount(),
            config,
            _s_window=False,
        )
        self.assertEqual(list(both["relation"]), list(full["relation"]))
        self.assertEqual(list(both["support"]), list(full["support"]))
        self.assertIsNone(corridor["applicability"]["confirmed_horizon_m"])

    def test_unknown_front_has_no_distance_and_a_set_value_is_explicit(self):
        config = _config()
        corridor = build_path_corridor(_hypothesis([6, 10, 15, 20]), _ground(), config, _mount())
        x, y, z = _sensor([15.0], [0.0], [-0.5])
        unknown = classify_sensor_points(x, y, z, corridor, _mount(), config)
        self.assertIsNone(unknown["distance_from_train_front_m"][0])
        known = classify_sensor_points(x, y, z, corridor, _mount(), config, front_ahead_of_lidar_m=4.0)
        self.assertAlmostEqual(float(known["distance_from_train_front_m"][0]), 11.0, places=6)

    def test_lower_boundary_changes_overlap_of_a_10_cm_object(self):
        self.assertAlmostEqual(surface_object_overlap_m(0.10, 0.0), 0.10)
        self.assertAlmostEqual(surface_object_overlap_m(0.10, 0.05), 0.05)
        self.assertAlmostEqual(surface_object_overlap_m(0.10, 0.10), 0.0)
        self.assertAlmostEqual(surface_object_overlap_m(0.10, 0.15), 0.0)

    def test_assumption_floor_does_not_confirm_an_inside_point(self):
        config = load_config(config_path())
        assumption = config["assumptions"]["lower_boundary_above_reference_m"]
        corridor = build_path_corridor(
            _hypothesis([6, 10, 15, 20]),
            _ground(),
            config,
            _mount(),
            lower_boundary=assumption,
        )
        out = classify_sensor_points(*_sensor([15.0], [0.0], [-0.5]), corridor, _mount(), config)
        self.assertEqual(out["geometric_relation"][0], "inside")
        self.assertTrue(bool(out["conditional"][0]))
        self.assertTrue(out["assumption_is_not_calibration"])
        self.assertIn("допущение", " ".join(out["conditional_reasons"]))

    def test_pair_midpoint_is_not_stored_as_the_body_center(self):
        corridor = build_path_corridor(_hypothesis([6, 10, 15]), _ground(), _config(), _mount())
        self.assertFalse(corridor["reference_midpoint_is_body_center"])
        self.assertTrue(all(item["is_body_center"] is False for item in corridor["vertices"]))

    def test_reliable_attachment_does_not_resolve_a_second_hypothesis(self):
        attachment = {
            "enabled": True,
            "source": "manual_estimate",
            "basis": "известная установка у вагона, не маршрут на стрелке",
            "s_m": 0.0,
            "l_m": 0.1,
        }
        self.assertTrue(attachment_is_reliable(attachment))
        config = _config()
        first = build_path_corridor(_hypothesis([6, 10, 15]), _ground(), config, _mount(), attachment)
        second = build_path_corridor(
            _hypothesis([6, 10, 15], lateral=lambda _s: 3.0),
            _ground(),
            config,
            _mount(),
            attachment,
        )
        self.assertEqual(first["vertices"][0]["role"], "attachment_prior")
        self.assertAlmostEqual(first["vertices"][0]["l_m"], 0.1)
        observed = [item for item in first["vertices"] if item["role"] == "observed"]
        self.assertAlmostEqual(observed[0]["l_m"], 0.0)
        decision = choose_own_path(
            [{"hypothesis_id": "H1", "shared_with": []}, {"hypothesis_id": "H2", "shared_with": []}],
            prefer_distance_m=1.55,
        )
        self.assertIsNone(decision["selected_id"])
        self.assertNotEqual(first["vertices"][0]["l_m"], second["vertices"][-1]["l_m"])

    def test_own_path_ignores_score_axis_and_nominal_distance(self):
        decision = choose_own_path(
            [
                {"hypothesis_id": "Ha", "engineering_score": 0.9, "shared_with": ["Hb"]},
                {"hypothesis_id": "Hb", "engineering_score": 0.2, "shared_with": ["Ha"]},
            ],
            prefer_score=True,
            prefer_sensor_axis=True,
            prefer_distance_m=1.55,
        )
        self.assertFalse(decision["own_path_selected"])
        self.assertIsNone(decision["selected_id"])
        self.assertIn("distance_near_1_55_m", decision["refused_rules"])

    def test_indices_of_rare_points_are_kept(self):
        config = _config()
        corridor = build_path_corridor(_hypothesis([6, 10, 15, 20]), _ground(), config, _mount())
        x, y, z = _sensor([15.0, 15.0], [0.0, 4.0], [-0.5, -0.5])
        grouped = classify_sensor_points(x, y, z, corridor, _mount(), config, source_index=[10, 11])
        expanded = expand_group_results(grouped, [np.array([10, 12]), np.array([11])])
        self.assertEqual(expanded["count"], 3)
        self.assertEqual(list(expanded["source_index"]), [10, 12, 11])
        self.assertEqual(list(expanded["sensor_x_m"]), [float(x[0]), float(x[0]), float(x[1])])

    def _pair(self, hypothesis_id, lateral, stations=(6, 10, 15)):
        center = lateral
        item = _hypothesis(stations, lateral=lambda s, center=center: center(s) if callable(center) else center)
        item["hypothesis_id"] = hypothesis_id
        item["candidate_count"] = 500 if hypothesis_id == "Hfar" else 1
        return item

    def test_attachment_matches_position_not_candidate_count(self):
        attachment = {
            "enabled": True,
            "name": "left",
            "source": "manual_estimate",
            "basis": "проверка",
            "s_m": 6.0,
            "l_m": -1.40,
            "support_s_min_m": 6.0,
            "support_s_max_m": 10.0,
            "lateral_tolerance_m": 0.25,
            "heading_dl_ds": 0.0,
            "heading_tolerance": 0.15,
        }
        near = self._pair("Hnear", -1.40)
        far = self._pair("Hfar", 1.26)
        decision = match_session_attachment([far, near], attachment)
        self.assertEqual(decision["status"], "one")
        self.assertEqual(decision["compatible"][0]["hypothesis_id"], "Hnear")
        self.assertFalse(decision["own_path_selected"])
        self.assertIn("candidate_count", decision["refused_rules"])

    def test_two_compatible_pairs_stay_ambiguous(self):
        attachment = session_variant(load_session(), "central")
        leftish = self._pair("Ha", -0.40)
        rightish = self._pair("Hb", -0.10)
        decision = match_session_attachment([leftish, rightish], attachment)
        self.assertEqual(decision["status"], "several")
        self.assertTrue(decision["route_ambiguous"])
        self.assertFalse(decision["own_path_selected"])

    def test_no_pair_is_a_conflict_not_a_forced_choice(self):
        attachment = session_variant(load_session(), "left")
        decision = match_session_attachment([self._pair("Hx", 3.0)], attachment)
        self.assertEqual(decision["status"], "none")
        self.assertTrue(decision["attachment_conflict"])
        self.assertEqual(decision["compatible"], [])

    def test_sensor_axis_is_not_selected_without_attachment(self):
        attachment = session_variant(load_session(), "left")
        on_axis = self._pair("H0", 0.0)
        decision = match_session_attachment([on_axis], attachment)
        self.assertEqual(decision["status"], "none")

    def test_curve_follows_observed_sections_not_a_far_straight_line(self):
        attachment = session_variant(load_session(), "left")
        curved = self._pair("Hc", lambda s: -1.40 + 0.02 * (s - 6.0), stations=(6, 10, 15, 20))
        decision = match_session_attachment([curved], attachment)
        self.assertEqual(decision["status"], "one")
        corridor = build_path_corridor(curved, _ground(), _config(), _mount(), attachment)
        roles = [item["role"] for item in corridor["vertices"]]
        self.assertNotIn(100.0, [item["s_m"] for item in corridor["vertices"]])
        self.assertLessEqual(max(item["s_m"] for item in corridor["vertices"]), 25.0)
        self.assertIn("observed", roles)

    def test_session_does_not_assign_an_own_path_or_a_mount(self):
        session = load_session()
        self.assertFalse(session["own_path_assigned"])
        self.assertFalse(session["mount"]["confirmed"])
        self.assertTrue(all(item["own_path"] is False for item in session["variants"]))
        self.assertIsNone(session["front_ahead_of_lidar_m"]["value"])
        self.assertFalse(session["front_ahead_of_lidar_m"]["blocks_range_from_lidar"])

    def test_absent_elevation_is_not_an_obstacle(self):
        absence = elevated_structure_absence()
        self.assertFalse(absence["path_removed"])
        self.assertFalse(absence["obstacle"])


if __name__ == "__main__":
    unittest.main()
