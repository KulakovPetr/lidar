"""Contest corridor model on top of a path hypothesis.

Sensor axes are not train axes. The midpoint of a reference-line pair is not
the body center. A point inside the profile is not an obstacle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

SOURCES = ("organizer", "manual_estimate", "algorithm_estimate", "unknown")
APPLIED_MOUNT_SOURCES = ("organizer", "manual_estimate", "algorithm_estimate")

# p_working = W @ p_sensor, with s = -y, l = x, h = z.
WORKING_FROM_SENSOR = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
SENSOR_FROM_WORKING = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "corridor.yaml"


def load_config(path: Path | None = None) -> dict:
    """Load the shared corridor file. Per-frame overrides are rejected."""
    raw = yaml.safe_load((path or config_path()).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("corridor config must be a mapping")
    if "frames" in raw or "per_frame" in raw:
        raise ValueError("per-frame corridor parameters are not allowed")
    for key in raw:
        if isinstance(key, str) and "/" in key:
            raise ValueError(f"frame-like config key {key!r} is not allowed")
    return raw


def coordinate_contract() -> dict:
    """Three frames and the maps between them. Units are metres and radians."""
    return {
        "units": {"length": "m", "angle": "rad"},
        "sensor": {
            "name": "sensor",
            "origin": "начало PointCloud2, как его публикует драйвер",
            "axes": "x, y, z в том виде, в каком они записаны. Это не оси поезда.",
            "handed": "как в облаке",
        },
        "train": {
            "name": "train",
            "origin": "задаётся переносом установки; при неподтверждённой установке начало не привязано к кузову",
            "axes": {
                "x": "вперёд по кузову",
                "y": "влево по кузову",
                "z": "вверх по кузову",
            },
            "handed": "правая",
            "not_equated_to_sensor": True,
            "pair_midpoint_is_body_center": False,
        },
        "working": {
            "name": "working",
            "origin": "то же начало, что у кадра, из которого оно получено",
            "axes": {"s": "−y", "l": "x", "h": "z"},
            "meaning": "экспериментальный prior этапов 4A–4C, не курс и не гравитация",
        },
        "path": {
            "name": "path",
            "origin": "первое наблюдаемое сечение гипотезы, не центр кузова",
            "axes": {
                "u": "вдоль локальной ломаной середины пары",
                "v": "поперёк профиля, вдоль оценённого наклона опорной поверхности",
                "w": "по нормали профиля вверх от нижней границы",
            },
        },
        "sensor_to_train": {
            "direction": "sensor -> train",
            "formula": "p_train = Rz(yaw) @ Ry(pitch) @ Rx(roll) @ p_sensor + t",
            "rotation_order": "сначала крен Rx, затем тангаж Ry, затем рыскание Rz",
            "inverse": "p_sensor = R.T @ (p_train - t)",
            "applied_only_when_confirmed": True,
        },
        "sensor_to_working": {
            "direction": "sensor -> working",
            "formula": "[s, l, h] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]] @ [x, y, z]",
            "inverse": "[x, y, z] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]] @ [s, l, h]",
        },
    }


def _rx(roll: float) -> np.ndarray:
    c, s = float(np.cos(roll)), float(np.sin(roll))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _ry(pitch: float) -> np.ndarray:
    c, s = float(np.cos(pitch)), float(np.sin(pitch))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rz(yaw: float) -> np.ndarray:
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def mount_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return _rz(yaw) @ _ry(pitch) @ _rx(roll)


def mount_is_applied(mount: dict) -> bool:
    """An unconfirmed or unsourced mount is not a calibration and is not applied."""
    if not mount or not mount.get("confirmed"):
        return False
    if mount.get("source") not in APPLIED_MOUNT_SOURCES:
        return False
    return bool(str(mount.get("basis") or "").strip())


def mount_from_config(config: dict) -> dict:
    block = config["mount"]
    translation = block["translation_m"]
    angles = block["rpy_rad"]
    mount = {
        "confirmed": bool(block.get("confirmed")),
        "source": block.get("source"),
        "status": block.get("status"),
        "basis": block.get("basis") or "",
        "tx_m": float(translation["x"]),
        "ty_m": float(translation["y"]),
        "tz_m": float(translation["z"]),
        "roll_rad": float(angles["roll"]),
        "pitch_rad": float(angles["pitch"]),
        "yaw_rad": float(angles["yaw"]),
        "zero_is_not_a_calibration": True,
    }
    mount["applied"] = mount_is_applied(mount)
    mount["identity_is_calibration"] = False
    return mount


def sensor_to_train(x, y, z, mount: dict):
    """Map sensor points into the train frame. Unconfirmed mounts do nothing."""
    points = np.column_stack(
        [
            np.asarray(x, dtype=np.float64).reshape(-1),
            np.asarray(y, dtype=np.float64).reshape(-1),
            np.asarray(z, dtype=np.float64).reshape(-1),
        ]
    )
    if not mount_is_applied(mount):
        out = points
    else:
        rotation = mount_rotation(mount["roll_rad"], mount["pitch_rad"], mount["yaw_rad"])
        shift = np.array([mount["tx_m"], mount["ty_m"], mount["tz_m"]], dtype=np.float64)
        out = points @ rotation.T + shift
    return out[:, 0], out[:, 1], out[:, 2]


def train_to_sensor(x, y, z, mount: dict):
    points = np.column_stack(
        [
            np.asarray(x, dtype=np.float64).reshape(-1),
            np.asarray(y, dtype=np.float64).reshape(-1),
            np.asarray(z, dtype=np.float64).reshape(-1),
        ]
    )
    if not mount_is_applied(mount):
        out = points
    else:
        rotation = mount_rotation(mount["roll_rad"], mount["pitch_rad"], mount["yaw_rad"])
        shift = np.array([mount["tx_m"], mount["ty_m"], mount["tz_m"]], dtype=np.float64)
        out = (points - shift) @ rotation
    return out[:, 0], out[:, 1], out[:, 2]


def sensor_to_working(x, y, z):
    points = np.column_stack(
        [
            np.asarray(x, dtype=np.float64).reshape(-1),
            np.asarray(y, dtype=np.float64).reshape(-1),
            np.asarray(z, dtype=np.float64).reshape(-1),
        ]
    )
    out = points @ WORKING_FROM_SENSOR.T
    return out[:, 0], out[:, 1], out[:, 2]


def working_to_sensor(s, l, h):
    points = np.column_stack(
        [
            np.asarray(s, dtype=np.float64).reshape(-1),
            np.asarray(l, dtype=np.float64).reshape(-1),
            np.asarray(h, dtype=np.float64).reshape(-1),
        ]
    )
    out = points @ SENSOR_FROM_WORKING.T
    return out[:, 0], out[:, 1], out[:, 2]


def geometry_working(x, y, z, mount: dict):
    """Working coordinates used for a corridor.

    A confirmed mount is applied first, so the prior s = -y, l = x, h = z
    then refers to the train frame. An unconfirmed mount leaves the sensor prior.
    """
    if mount_is_applied(mount):
        xt, yt, zt = sensor_to_train(x, y, z, mount)
        s, l, h = sensor_to_working(xt, yt, zt)
        return s, l, h, "train_working"
    s, l, h = sensor_to_working(x, y, z)
    return s, l, h, "sensor_working_prior_mount_unconfirmed"


def attachment_is_reliable(attachment: dict | None) -> bool:
    if not attachment or not attachment.get("enabled"):
        return False
    if attachment.get("source") not in ("organizer", "manual_estimate"):
        return False
    if attachment.get("s_m") is None or attachment.get("l_m") is None:
        return False
    return bool(str(attachment.get("basis") or "").strip())


def _section_midpoints(hypothesis) -> list[tuple[float, float]]:
    points = []
    for row in hypothesis.get("observed_sections") or []:
        points.append(
            (
                float(row["s_m"]),
                0.5 * (float(row["left_l_m"]) + float(row["right_l_m"])),
            )
        )
    points.sort(key=lambda item: item[0])
    return points


def match_session_attachment(hypotheses, attachment: dict | None) -> dict:
    """Match hypotheses to one session attachment by position, heading, and support.

    Candidate counts are not an argument. A local hypothesis id is not the attachment.
    Proximity to the sensor lateral axis is not proof of the own path.
    """
    refused = [
        "candidate_count",
        "max_engineering_score",
        "closest_to_sensor_lateral_axis",
        "distance_near_1_55_m",
    ]
    if not attachment or not attachment.get("enabled"):
        return {
            "status": "not_applied",
            "compatible": [],
            "own_path_selected": False,
            "route_ambiguous": False,
            "attachment_conflict": False,
            "conditional_on_attachment": False,
            "refused_rules": refused,
        }
    support_min = float(attachment["support_s_min_m"])
    support_max = float(attachment["support_s_max_m"])
    lateral = float(attachment["lateral_tolerance_m"])
    anchor_s = float(attachment["s_m"])
    anchor_l = float(attachment["l_m"])
    heading = attachment.get("heading_dl_ds")
    heading_tol = attachment.get("heading_tolerance")
    compatible = []
    for hypothesis in hypotheses or []:
        points = _section_midpoints(hypothesis)
        supported = [item for item in points if support_min - 1e-6 <= item[0] <= support_max + 1e-6]
        if not supported:
            continue
        nearest = min(supported, key=lambda item: abs(item[0] - anchor_s))
        offset = abs(nearest[1] - anchor_l)
        if offset > lateral:
            continue
        direction_checked = False
        if heading is not None and heading_tol is not None and len(supported) >= 2:
            direction_checked = True
            ds = supported[-1][0] - supported[0][0]
            slope = 0.0 if ds == 0.0 else (supported[-1][1] - supported[0][1]) / ds
            if abs(slope - float(heading)) > float(heading_tol):
                continue
        compatible.append(
            {
                "hypothesis_id": hypothesis.get("hypothesis_id"),
                "local_id_is_not_the_attachment": True,
                "anchor_s_m": nearest[0],
                "anchor_l_m": nearest[1],
                "lateral_offset_m": offset,
                "support_sections": len(supported),
                "direction_checked": direction_checked,
                "observed_s_min_m": points[0][0],
                "observed_s_max_m": points[-1][0],
            }
        )
    count = len(compatible)
    status = "one" if count == 1 else "several" if count > 1 else "none"
    return {
        "status": status,
        "compatible": compatible,
        "own_path_selected": False,
        "route_ambiguous": status == "several",
        "attachment_conflict": status == "none",
        "conditional_on_attachment": status == "one",
        "attachment_name": attachment.get("name"),
        "attachment_source": attachment.get("source"),
        "not_organizer_data": attachment.get("source") != "organizer",
        "not_ground_truth": True,
        "continuation_follows_observed_sections": True,
        "straight_line_is_not_a_far_path": True,
        "refused_rules": refused,
        "sensor_axis_is_not_proof": True,
    }


def session_variant(session: dict, name: str) -> dict:
    """One conditional attachment. It does not become the own path."""
    match = session["match"]
    chosen = next(item for item in session["variants"] if item["name"] == name)
    if chosen.get("own_path"):
        raise ValueError("a session variant must not be marked as the own path")
    return {
        "name": chosen["name"],
        "enabled": bool(chosen.get("enabled")),
        "source": chosen.get("source"),
        "basis": chosen.get("basis") or "",
        "s_m": chosen.get("s_m"),
        "l_m": chosen.get("l_m"),
        "uncertainty_l_m": chosen.get("uncertainty_l_m"),
        "support_s_min_m": match["support_s_min_m"],
        "support_s_max_m": match["support_s_max_m"],
        "lateral_tolerance_m": match["lateral_tolerance_m"],
        "heading_dl_ds": match.get("heading_dl_ds"),
        "heading_tolerance": match.get("heading_tolerance"),
        "frame": chosen.get("frame"),
        "not_organizer_data": chosen.get("source") != "organizer",
        "not_ground_truth": True,
        "own_path": False,
    }


def load_session(path: Path | None = None) -> dict:
    """Session mount, path attachment, and profile floor. No per-frame block."""
    file_path = path or (Path(__file__).resolve().parents[1] / "config" / "session.yaml")
    raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("session config must be a mapping")
    if "frames" in raw or "per_frame" in raw:
        raise ValueError("per-frame session parameters are not allowed")
    if raw.get("own_path_assigned"):
        raise ValueError("a session file must not assign one variant as the own path")
    return raw


def choose_own_path(hypotheses, **ignored) -> dict:
    """Own-path selection stays closed. Score, sensor axis, and 1.55 m do not decide it."""
    return {
        "selected_id": None,
        "own_path_selected": False,
        "refused_rules": [
            "max_engineering_score",
            "closest_to_sensor_lateral_axis",
            "distance_near_1_55_m",
        ],
        "ignored_arguments": sorted(ignored),
        "alternatives_remain": len(list(hypotheses or [])) > 1 or any(
            item.get("shared_with") for item in (hypotheses or [])
        ),
    }


def _surface_at(ground, s_m: float, l_m: float):
    if not ground or ground.get("status") != "candidate":
        return None, None, False
    for bin_model in ground.get("local_bins") or []:
        if bin_model["s_lo"] <= s_m < bin_model["s_hi"]:
            href = float(bin_model["intercept"] + bin_model["slope_dh_dl"] * l_m)
            lo = bin_model.get("l_min")
            hi = bin_model.get("l_max")
            supported = lo is not None and hi is not None and lo <= l_m <= hi
            return href, float(bin_model["slope_dh_dl"]), bool(supported)
    return None, None, False


def _interp_l(observed, s_m: float) -> float:
    before = [item for item in observed if item["s_m"] <= s_m]
    after = [item for item in observed if item["s_m"] >= s_m]
    if before and after and before[-1]["s_m"] != after[0]["s_m"]:
        a, b = before[-1], after[0]
        span = b["s_m"] - a["s_m"]
        weight = 0.0 if span == 0.0 else (s_m - a["s_m"]) / span
        return float(a["l_m"] + weight * (b["l_m"] - a["l_m"]))
    pool = before or after or observed
    return float(pool[-1]["l_m"])


def build_path_corridor(hypothesis, ground, config, mount, attachment=None, lower_boundary=None) -> dict:
    """Place the contest profile on one hypothesis. This does not select the own path."""
    profile = config["profile"]
    width = float(profile["width_m"]["value"])
    height = float(profile["height_m"]["value"])
    if lower_boundary is None:
        lower_boundary = config["lower_boundary_above_reference_m"]
    lower_value = lower_boundary.get("value")
    lower_status = lower_boundary.get("status")
    extra_lat = float(config["extra_lateral_m"]["value"])
    extra_vert = float(config["extra_vertical_m"]["value"])
    observed_rows = sorted(hypothesis.get("observed_sections") or [], key=lambda row: float(row["s_m"]))
    observed = []
    for row in observed_rows:
        s_m = float(row["s_m"])
        mid = 0.5 * (float(row["left_l_m"]) + float(row["right_l_m"]))
        href, slope, supported = _surface_at(ground, s_m, mid)
        observed.append(
            {
                "s_m": s_m,
                "l_m": mid,
                "left_l_m": float(row["left_l_m"]),
                "right_l_m": float(row["right_l_m"]),
                "h_ref_m": href,
                "slope_dh_dl": slope,
                "reference_supported": supported,
                "role": "observed",
                "is_body_center": False,
            }
        )
    expected = [float(value) for value in config["path"]["expected_section_s_m"]]
    extra = float(config["path"]["max_extrapolation_m"])
    vertices = []
    if observed and attachment_is_reliable(attachment):
        att_s = float(attachment["s_m"])
        if att_s < observed[0]["s_m"] - 1e-6:
            href, slope, supported = _surface_at(ground, att_s, float(attachment["l_m"]))
            vertices.append(
                {
                    "s_m": att_s,
                    "l_m": float(attachment["l_m"]),
                    "h_ref_m": href,
                    "slope_dh_dl": slope,
                    "reference_supported": supported,
                    "role": "attachment_prior",
                    "is_body_center": False,
                }
            )
    if len(observed) >= 2 and extra > 0.0:
        first, second = observed[0], observed[1]
        ds = second["s_m"] - first["s_m"]
        dl_ds = 0.0 if ds == 0.0 else (second["l_m"] - first["l_m"]) / ds
        s_back = first["s_m"] - extra
        l_back = first["l_m"] - dl_ds * extra
        href, slope, supported = _surface_at(ground, s_back, l_back)
        vertices.append(
            {
                "s_m": s_back,
                "l_m": l_back,
                "h_ref_m": href,
                "slope_dh_dl": slope,
                "reference_supported": supported,
                "role": "extrapolated",
                "is_body_center": False,
            }
        )
    if observed:
        s0, s1 = observed[0]["s_m"], observed[-1]["s_m"]
        present = {round(item["s_m"], 3) for item in observed}
        for s_m in expected:
            if s_m < s0 - 1e-6 or s_m > s1 + 1e-6 or round(s_m, 3) in present:
                continue
            l_m = _interp_l(observed, s_m)
            href, slope, supported = _surface_at(ground, s_m, l_m)
            vertices.append(
                {
                    "s_m": s_m,
                    "l_m": l_m,
                    "h_ref_m": href,
                    "slope_dh_dl": slope,
                    "reference_supported": supported,
                    "role": "gap_interpolation",
                    "is_body_center": False,
                    "adds_width_measurement": False,
                }
            )
        vertices.extend(observed)
    if len(observed) >= 2 and extra > 0.0:
        penultimate, last = observed[-2], observed[-1]
        ds = last["s_m"] - penultimate["s_m"]
        dl_ds = 0.0 if ds == 0.0 else (last["l_m"] - penultimate["l_m"]) / ds
        s_fwd = last["s_m"] + extra
        l_fwd = last["l_m"] + dl_ds * extra
        href, slope, supported = _surface_at(ground, s_fwd, l_fwd)
        vertices.append(
            {
                "s_m": s_fwd,
                "l_m": l_fwd,
                "h_ref_m": href,
                "slope_dh_dl": slope,
                "reference_supported": supported,
                "role": "extrapolated",
                "is_body_center": False,
            }
        )
    vertices.sort(key=lambda item: item["s_m"])
    unique = []
    for item in vertices:
        if unique and abs(unique[-1]["s_m"] - item["s_m"]) < 1e-6:
            if item["role"] == "observed":
                unique[-1] = item
            continue
        unique.append(item)
    vertices = unique
    segments = []
    arc = [0.0]
    for start, stop in zip(vertices, vertices[1:]):
        ds = stop["s_m"] - start["s_m"]
        dl = stop["l_m"] - start["l_m"]
        length = float(np.hypot(ds, dl))
        roles = {start["role"], stop["role"]}
        if "extrapolated" in roles:
            role = "extrapolated"
        elif "gap_interpolation" in roles:
            role = "gap_interpolation"
        elif "attachment_prior" in roles:
            role = "attachment_prior"
        else:
            role = "between_samples"
        segments.append(
            {
                "s0": start["s_m"],
                "l0": start["l_m"],
                "s1": stop["s_m"],
                "l1": stop["l_m"],
                "length_m": length,
                "role": role,
                "h0": start["h_ref_m"],
                "h1": stop["h_ref_m"],
                "slope0": start["slope_dh_dl"],
                "slope1": stop["slope_dh_dl"],
                "reference0": start["reference_supported"],
                "reference1": stop["reference_supported"],
            }
        )
        arc.append(arc[-1] + length)
    observed_s = [item["s_m"] for item in observed]
    first_observed_arc = 0.0
    if observed_s and vertices:
        for vertex, vertex_arc in zip(vertices, arc):
            if abs(vertex["s_m"] - observed_s[0]) < 1e-6 and vertex["role"] == "observed":
                first_observed_arc = vertex_arc
                break
    intervals = _intervals(vertices, segments)
    fitting = None
    if ground:
        residual = ground.get("fitting_residual") or {}
        fitting = residual.get("p95_m")
    scatter = hypothesis.get("distance_scatter_m")
    if scatter is None:
        consistency = hypothesis.get("consistency") or {}
        scatter = consistency.get("distance_span_m")
    return {
        "hypothesis_id": hypothesis.get("hypothesis_id"),
        "kind": hypothesis.get("kind"),
        "engineering_score": hypothesis.get("engineering_score"),
        "score_is_probability": False,
        "shared_with": list(hypothesis.get("shared_with") or []),
        "distance_m": hypothesis.get("distance_m"),
        "width_m": width,
        "height_m": height,
        "half_width_m": 0.5 * width,
        "half_width_was_not_doubled": True,
        "extra_lateral_m": extra_lat,
        "extra_vertical_m": extra_vert,
        "lower_boundary_m": None if lower_value is None else float(lower_value),
        "lower_boundary_status": lower_status,
        "lower_boundary_source": lower_boundary.get("source"),
        "profile_includes_protrusions": True,
        "train_width_is_not_pair_plus_margin": True,
        "reference_surface_is_rail_head": False,
        "reference_surface_is_body_bottom": False,
        "reference_midpoint_is_body_center": False,
        "not_a_dynamic_gauge": True,
        "missing_dynamic_terms": ["body_overhang", "bogie_kinematics"],
        "not_an_obstacle_model": True,
        "mount_applied": mount_is_applied(mount),
        "placement": "hypothetical" if (not mount_is_applied(mount) or lower_status != "stated") else "parameters_stated",
        "vertices": vertices,
        "segments": segments,
        "arc_m": arc,
        "first_observed_arc_m": first_observed_arc,
        "observed_s_m": observed_s,
        "support_intervals": intervals,
        "direction_quality": "available" if len(observed) >= 2 else "insufficient",
        "ground_quality": {
            "status": None if not ground else ground.get("status"),
            "ambiguous_with_nearby_surface": None if not ground else ground.get("ambiguous_with_nearby_surface"),
            "fitting_residual_p95_m": fitting,
            "fitting_residual_is_ground_accuracy": False,
            "vertical_axis_is_sensor_z": not mount_is_applied(mount),
            "vertical_axis_is_gravity": False,
            "transverse_slope_is_body_roll": False,
        },
        "position_uncertainty_m": {
            "lateral_m": None if scatter is None else float(scatter),
            "vertical_m": None if fitting is None else float(fitting),
        },
        "applicability": {
            "observed_s_m": observed_s,
            "max_extrapolation_m": extra,
            "confirmed_horizon_m": None,
            "forbidden_confirmed_horizon_m": list(config["path"]["forbidden_confirmed_horizon_m"]),
            "furthest_cloud_point_is_not_small_object_range": True,
            "missing_elevated_structure_is_not_path_loss": True,
            "missing_elevated_structure_is_not_obstacle": True,
        },
        "assumed_sections_add_width": False,
        "section_half_m": float(config["path"]["section_half_m"]),
    }


def _intervals(vertices, segments) -> list[dict]:
    rows = []
    for segment in segments:
        rows.append(
            {
                "s0_m": segment["s0"],
                "s1_m": segment["s1"],
                "support": segment["role"],
                "length_m": segment["length_m"],
            }
        )
    if not rows and vertices:
        rows.append(
            {
                "s0_m": vertices[0]["s_m"],
                "s1_m": vertices[0]["s_m"],
                "support": "observed" if vertices[0]["role"] == "observed" else vertices[0]["role"],
                "length_m": 0.0,
            }
        )
    return rows


def _profile_axes(slope):
    slope = 0.0 if slope is None or not np.isfinite(slope) else float(slope)
    norm = float(np.hypot(1.0, slope))
    along_l, along_h = 1.0 / norm, slope / norm
    up_l, up_h = -slope / norm, 1.0 / norm
    return along_l, along_h, up_l, up_h


def _empty_classification(n: int, source_index) -> dict:
    empty = np.zeros(n, dtype=np.float64)
    def blank():
        return np.full(n, None, dtype=object)
    none = blank()
    return {
        "path_s_m": empty,
        "profile_lateral_m": empty,
        "profile_vertical_m": empty,
        "relation": blank(),
        "uncertain_reason": blank(),
        "support": blank(),
        "range_from_lidar_m": empty,
        "distance_along_path_m": empty,
        "distance_from_train_front_m": none,
        "source_index": None if source_index is None else np.asarray(source_index),
        "sensor_x_m": empty,
        "sensor_y_m": empty,
        "sensor_z_m": empty,
        "inside_is_not_an_obstacle": True,
        "count": n,
    }


def classify_sensor_points(x, y, z, corridor, mount, config, source_index=None, front_ahead_of_lidar_m=None, _s_window=True) -> dict:
    """Classify original sensor points against one corridor.

    `relation` is inside, outside, or uncertain. Inside is not an obstacle.
    Source indices and XYZ are kept one-for-one. Rare points are not removed.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if source_index is not None and int(np.asarray(source_index).size) != n:
        raise ValueError("source_index length must match the points")
    result = _empty_classification(n, source_index)
    result["sensor_x_m"] = x.copy()
    result["sensor_y_m"] = y.copy()
    result["sensor_z_m"] = z.copy()
    result["range_from_lidar_m"] = np.sqrt(x * x + y * y + z * z)
    if front_ahead_of_lidar_m is None:
        front_ahead_of_lidar_m = config["front_ahead_of_lidar_m"]["value"]
    front_value = None if front_ahead_of_lidar_m is None else float(front_ahead_of_lidar_m)
    if n == 0 or not corridor.get("vertices"):
        result["relation"][:] = "uncertain"
        result["geometric_relation"] = result["relation"]
        result["uncertain_reason"][:] = "outside_applicability"
        result["support"][:] = "unsupported"
        result["conditional"] = np.zeros(n, dtype=bool)
        result["conditional_reasons"] = []
        result["geometry_confidence_separated"] = True
        result["assumption_is_not_calibration"] = True
        result["front_distance_not_required"] = True
        return result
    s, l, h, _frame = geometry_working(x, y, z, mount)
    segments = corridor["segments"]
    if not segments:
        return _classify_single_vertex(s, l, h, corridor, result, front_value)
    if _s_window and corridor.get("vertices"):
        vertex_s = np.array([float(item["s_m"]) for item in corridor["vertices"]], dtype=np.float64)
        window = (s >= float(vertex_s.min())) & (s <= float(vertex_s.max()))
        if not window.all():
            result["relation"][:] = "uncertain"
            result["geometric_relation"] = result["relation"]
            result["uncertain_reason"][:] = "outside_applicability"
            result["support"][:] = "unsupported"
            result["conditional_reasons"] = []
            result["conditional"] = np.zeros(n, dtype=bool)
            result["geometry_confidence_separated"] = True
            result["assumption_is_not_calibration"] = corridor.get("lower_boundary_status") != "stated"
            result["front_distance_not_required"] = front_value is None
            if window.any():
                part = classify_sensor_points(
                    x[window],
                    y[window],
                    z[window],
                    corridor,
                    mount,
                    config,
                    None if source_index is None else np.asarray(source_index)[window],
                    front_ahead_of_lidar_m if front_value is None else front_value,
                    _s_window=False,
                )
                for key in (
                    "path_s_m",
                    "profile_lateral_m",
                    "profile_vertical_m",
                    "relation",
                    "geometric_relation",
                    "uncertain_reason",
                    "support",
                    "distance_along_path_m",
                    "distance_from_train_front_m",
                    "conditional",
                ):
                    result[key][window] = part[key]
                result["conditional_reasons"] = list(part.get("conditional_reasons") or [])
                if result["conditional_reasons"]:
                    result["conditional"][:] = True
            return result
    s0 = np.array([segment["s0"] for segment in segments], dtype=np.float64)
    l0 = np.array([segment["l0"] for segment in segments], dtype=np.float64)
    s1 = np.array([segment["s1"] for segment in segments], dtype=np.float64)
    l1 = np.array([segment["l1"] for segment in segments], dtype=np.float64)
    ds = s1 - s0
    dl = l1 - l0
    length2 = ds * ds + dl * dl
    length2 = np.where(length2 < 1e-18, np.nan, length2)
    delta_s = s[:, None] - s0
    delta_l = l[:, None] - l0
    t = (delta_s * ds + delta_l * dl) / length2
    t_clip = np.clip(np.nan_to_num(t, nan=0.0), 0.0, 1.0)
    foot_s = s0 + t_clip * ds
    foot_l = l0 + t_clip * dl
    dist2 = (s[:, None] - foot_s) ** 2 + (l[:, None] - foot_l) ** 2
    # Shared endpoints tie. Prefer an observed section over the extrapolation that ends there.
    role_preference = {
        "observed": 0,
        "between_samples": 1,
        "gap_interpolation": 2,
        "attachment_prior": 3,
        "extrapolated": 4,
        "unsupported": 5,
    }
    preference = np.array([role_preference.get(segment["role"], 5) for segment in segments], dtype=np.float64)
    best = np.argmin(dist2 + preference * 1e-12, axis=1)
    rows = np.arange(n)
    t_best = t[rows, best]
    t_clip_best = t_clip[rows, best]
    foot_s_best = foot_s[rows, best]
    foot_l_best = foot_l[rows, best]
    beyond = ((best == 0) & (t_best < 0.0)) | ((best == len(segments) - 1) & (t_best > 1.0))
    beyond |= ~np.isfinite(t_best)
    arc0 = np.array(corridor["arc_m"][:-1], dtype=np.float64)
    seg_len = np.array([segment["length_m"] for segment in segments], dtype=np.float64)
    path_s = arc0[best] + t_clip_best * seg_len[best] - float(corridor["first_observed_arc_m"])
    h0 = np.array([np.nan if segment["h0"] is None else segment["h0"] for segment in segments])
    h1 = np.array([np.nan if segment["h1"] is None else segment["h1"] for segment in segments])
    slope0 = np.array([0.0 if segment["slope0"] is None else segment["slope0"] for segment in segments])
    slope1 = np.array([0.0 if segment["slope1"] is None else segment["slope1"] for segment in segments])
    ref0 = np.array([bool(segment["reference0"]) for segment in segments])
    ref1 = np.array([bool(segment["reference1"]) for segment in segments])
    href = h0[best] + t_clip_best * (h1[best] - h0[best])
    slope = slope0[best] + t_clip_best * (slope1[best] - slope0[best])
    reference_ok = ref0[best] & ref1[best] & np.isfinite(href)
    role_code = {
        "observed": 1,
        "between_samples": 2,
        "gap_interpolation": 3,
        "extrapolated": 4,
        "attachment_prior": 5,
        "unsupported": 0,
    }
    support_code = np.array([role_code.get(segment["role"], 0) for segment in segments], dtype=np.int8)[best]
    observed_s = np.array(corridor["observed_s_m"], dtype=np.float64)
    half = float(corridor["section_half_m"])
    if observed_s.size:
        near = np.min(np.abs(foot_s_best[:, None] - observed_s), axis=1) <= half
    else:
        near = np.zeros(n, dtype=bool)
    upgrade = near & ((support_code == 2) | (support_code == 3))
    support_code = support_code.copy()
    support_code[upgrade] = 1
    support_code[beyond] = 0
    support_names = np.array(
        ["unsupported", "observed", "between_samples", "gap_interpolation", "extrapolated", "attachment_prior"]
    )
    support = support_names[support_code]
    seg_length = np.sqrt(np.where(np.isfinite(length2), length2, 1.0))
    n_s = -dl / seg_length
    n_l = ds / seg_length
    horizontal = (s - foot_s_best) * n_s[best] + (l - foot_l_best) * n_l[best]
    slope_across = slope * n_l[best]
    along_l, along_h, up_l, up_h = _profile_axes_batch(slope_across)
    lower = corridor["lower_boundary_m"]
    floor = href if lower is None else href + float(lower)
    d_h = h - floor
    lateral = horizontal * along_l + d_h * along_h
    vertical = horizontal * up_l + d_h * up_h
    lateral = np.where(np.isfinite(lateral), lateral, horizontal)
    result["path_s_m"] = path_s
    result["profile_lateral_m"] = lateral
    result["profile_vertical_m"] = vertical
    result["distance_along_path_m"] = path_s.copy()
    result["support"] = support
    if front_value is None:
        result["distance_from_train_front_m"] = np.full(n, None, dtype=object)
    else:
        result["distance_from_train_front_m"] = foot_s_best - front_value
    half_width = float(corridor["half_width_m"]) + float(corridor["extra_lateral_m"])
    top = float(corridor["height_m"]) + float(corridor["extra_vertical_m"])
    # 0 outside_applicability, 1 extrapolated, 2 gap, 3 attachment, 4 insufficient direction,
    # 5 reference unsupported, 6 lower boundary missing, 7 no geometric doubt
    reason_code = np.full(n, 7, dtype=np.int8)
    floor_known = lower is not None and np.isfinite(float(lower))
    direction_ok = corridor.get("direction_quality") == "available"
    placed = ((support_code == 1) | (support_code == 2)) & direction_ok & ~beyond
    lateral_out = np.abs(lateral) > half_width
    vertical_finite = np.isfinite(vertical)
    vertical_out = vertical_finite & ((vertical < 0.0) | (vertical > top))
    relation_code = np.full(n, 0, dtype=np.int8)  # 0 uncertain, 1 inside, 2 outside
    relation_code[placed & lateral_out] = 2
    vertical_known = placed & ~lateral_out & reference_ok & floor_known & vertical_finite
    relation_code[vertical_known & ~vertical_out] = 1
    relation_code[vertical_known & vertical_out] = 2
    uncertain = relation_code == 0
    reason_code[uncertain & (beyond | (support_code == 0))] = 0
    reason_code[uncertain & (reason_code == 7) & (support_code == 4)] = 1
    reason_code[uncertain & (reason_code == 7) & (support_code == 3)] = 2
    reason_code[uncertain & (reason_code == 7) & (support_code == 5)] = 3
    if not direction_ok:
        reason_code[uncertain & (reason_code == 7)] = 4
    reason_code[uncertain & (reason_code == 7) & ~reference_ok] = 5
    if not floor_known:
        reason_code[uncertain & (reason_code == 7)] = 6
    reason_code[uncertain & (reason_code == 7)] = 0
    reason_names = np.array(
        [
            "outside_applicability",
            "extrapolated",
            "gap_interpolation",
            "attachment_prior",
            "insufficient_direction",
            "reference_surface_unsupported",
            "lower_boundary_not_established",
            "",
        ]
    )
    relation_names = np.array(["uncertain", "inside", "outside"])
    relation = relation_names[relation_code]
    reason = reason_names[reason_code]
    reason[relation_code != 0] = ""
    conditional_reasons = []
    if corridor.get("lower_boundary_status") != "stated":
        conditional_reasons.append("нижняя граница — допущение, не калибровка")
    if corridor.get("kind") != "consistent_rail_pair_hypothesis":
        conditional_reasons.append("гипотеза пути не согласована")
    if not corridor.get("mount_applied", False):
        conditional_reasons.append("установка лидара не подтверждена")
    conditional = np.zeros(n, dtype=bool)
    if conditional_reasons:
        conditional[:] = True
    conditional[support_code == 2] = True
    result["relation"] = relation
    result["geometric_relation"] = relation
    result["uncertain_reason"] = reason
    result["conditional"] = conditional
    result["conditional_reasons"] = conditional_reasons
    result["geometry_confidence_separated"] = True
    result["assumption_is_not_calibration"] = corridor.get("lower_boundary_status") != "stated"
    result["inside_is_not_an_obstacle"] = True
    result["front_distance_not_required"] = front_value is None
    return result


def _profile_axes_batch(slope: np.ndarray):
    slope = np.nan_to_num(np.asarray(slope, dtype=np.float64), nan=0.0)
    norm = np.hypot(1.0, slope)
    along_l = 1.0 / norm
    along_h = slope / norm
    up_l = -slope / norm
    up_h = 1.0 / norm
    return along_l, along_h, up_l, up_h


def _classify_single_vertex(s, l, h, corridor, result, front_value):
    vertex = corridor["vertices"][0]
    n = int(s.size)
    half = float(corridor["section_half_m"])
    near = np.abs(s - vertex["s_m"]) <= half
    support = np.full(n, "unsupported", dtype=object)
    support[near] = "observed" if vertex["role"] == "observed" else vertex["role"]
    result["support"] = support
    result["path_s_m"] = np.zeros(n, dtype=np.float64)
    result["distance_along_path_m"] = np.zeros(n, dtype=np.float64)
    along_l, along_h, up_l, up_h = _profile_axes(vertex.get("slope_dh_dl"))
    href = vertex.get("h_ref_m")
    lower = corridor["lower_boundary_m"]
    floor = None if href is None or lower is None else float(href) + float(lower)
    if floor is None:
        lateral = l - vertex["l_m"]
        vertical = np.full(n, np.nan)
    else:
        d_l = l - vertex["l_m"]
        d_h = h - floor
        lateral = d_l * along_l + d_h * along_h
        vertical = d_l * up_l + d_h * up_h
    result["profile_lateral_m"] = lateral
    result["profile_vertical_m"] = vertical
    result["relation"] = np.full(n, "uncertain", dtype=object)
    result["geometric_relation"] = result["relation"]
    reason = np.full(n, "outside_applicability", dtype=object)
    reason[near] = "insufficient_direction"
    result["uncertain_reason"] = reason
    result["conditional"] = np.ones(n, dtype=bool)
    result["conditional_reasons"] = ["направления пути недостаточно"]
    result["geometry_confidence_separated"] = True
    result["assumption_is_not_calibration"] = corridor.get("lower_boundary_status") != "stated"
    result["front_distance_not_required"] = front_value is None
    if front_value is None:
        result["distance_from_train_front_m"] = np.full(n, None, dtype=object)
    else:
        values = np.full(n, None, dtype=object)
        values[near] = float(vertex["s_m"] - front_value)
        result["distance_from_train_front_m"] = values
    return result


def expand_group_results(classification: dict, group_indices) -> dict:
    """Repeat one geometric vote onto every original index. Size-1 groups stay."""
    pieces = []
    index_pieces = []
    for row, members in enumerate(group_indices):
        members = np.asarray(members, dtype=np.int64).reshape(-1)
        if members.size == 0:
            continue
        pieces.append(np.full(members.size, row, dtype=np.int64))
        index_pieces.append(members)
    if not pieces:
        out = _empty_classification(0, np.zeros(0, dtype=np.int64))
        return out
    rows = np.concatenate(pieces)
    out = {"source_index": np.concatenate(index_pieces), "inside_is_not_an_obstacle": True, "count": int(rows.size)}
    for key, value in classification.items():
        if key in ("source_index", "inside_is_not_an_obstacle", "count", "unresolved_blocks_firm_answer"):
            continue
        array = np.asarray(value)
        if array.shape[:1] == (classification["count"],):
            out[key] = array[rows]
    return out


def surface_object_overlap_m(object_height_m: float, lower_boundary_m: float, profile_height_m: float = 3.0) -> float:
    """Overlap of a box standing on the reference surface with the profile.

    This is geometry of the assumed floor. It does not show that a lidar can see the object.
    """
    a0, a1 = 0.0, float(object_height_m)
    b0 = float(lower_boundary_m)
    b1 = b0 + float(profile_height_m)
    return float(max(0.0, min(a1, b1) - max(a0, b0)))


def elevated_structure_absence() -> dict:
    """Missing raised rails are not a removed path and not an obstacle."""
    return {
        "path_removed": False,
        "obstacle": False,
        "note": "возвышение может отсутствовать, если рельс вровень с поверхностью; это не конец пути и не препятствие",
    }


def parameter_table(config: dict) -> list[dict]:
    """Flat parameter rows for the report and a later launch file."""
    profile = config["profile"]
    mount = config["mount"]
    assumption = config["assumptions"]["lower_boundary_above_reference_m"]
    rows = [
        _row("contest_width_m", profile["width_m"]),
        _row("contest_height_m", profile["height_m"]),
        _row("profile_includes_protrusions", profile["includes_protrusions"]),
        _row("test_object_m", config["test_object_m"]),
        _row(
            "mount_tx_m",
            {"value": mount["translation_m"]["x"], "source": mount["source"], "status": mount["status"], "basis": mount["basis"]},
        ),
        _row(
            "mount_ty_m",
            {"value": mount["translation_m"]["y"], "source": mount["source"], "status": mount["status"], "basis": mount["basis"]},
        ),
        _row(
            "mount_tz_m",
            {"value": mount["translation_m"]["z"], "source": mount["source"], "status": mount["status"], "basis": mount["basis"]},
        ),
        _row(
            "mount_roll_rad",
            {"value": mount["rpy_rad"]["roll"], "source": mount["source"], "status": mount["status"], "basis": mount["basis"]},
        ),
        _row(
            "mount_pitch_rad",
            {"value": mount["rpy_rad"]["pitch"], "source": mount["source"], "status": mount["status"], "basis": mount["basis"]},
        ),
        _row(
            "mount_yaw_rad",
            {"value": mount["rpy_rad"]["yaw"], "source": mount["source"], "status": mount["status"], "basis": mount["basis"]},
        ),
        _row("lower_boundary_above_reference_m", config["lower_boundary_above_reference_m"]),
        _row("demonstration_lower_boundary_above_reference_m", assumption),
        _row("extra_lateral_m", config["extra_lateral_m"]),
        _row("extra_vertical_m", config["extra_vertical_m"]),
        _row("front_ahead_of_lidar_m", config["front_ahead_of_lidar_m"]),
        _row("reference_surface", config["reference_surface"]),
        {
            "name": "max_extrapolation_m",
            "value": config["path"]["max_extrapolation_m"],
            "source": config["path"]["max_extrapolation_source"],
            "status": "limit",
            "basis": "Предел продолжения модели. Не дальность, на которой гарантированно виден мелкий объект.",
        },
        {
            "name": "working_prior",
            "value": "s=-Y, l=X, h=Z",
            "source": config["working_prior"]["source"],
            "status": config["working_prior"]["status"],
            "basis": config["working_prior"]["basis"],
        },
    ]
    return rows


def _row(name: str, block: dict) -> dict:
    return {
        "name": name,
        "value": block.get("value"),
        "source": block.get("source"),
        "status": block.get("status"),
        "basis": block.get("basis"),
    }


def launch_parameter_names(config: dict) -> list[str]:
    return list(config.get("launch_parameters") or [])
