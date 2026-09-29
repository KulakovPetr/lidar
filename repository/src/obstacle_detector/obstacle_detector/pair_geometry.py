"""Distance between reference-line hypotheses and path position relative to the lidar.

The quantity is the distance between reference lines across the local path
direction. It is not a confirmed rail gauge and not a mounting offset of the
lidar on the vehicle body. The own track is not selected here.
"""

from __future__ import annotations

import numpy as np

PAIR_CONFIG = {
    "min_common_sections": 2,
    "min_distance_m": 0.40,
    "max_distance_m": 2.80,
    "section_agree_m": 0.25,
    "min_agreeing_sections": 2,
    "publish_s_m": 15.0,
    "straight_residual_p95_m": 0.08,
    "min_straight_sections": 3,
    "width_window": 9,
    "pose_window": 3,
    "width_admit_m": 0.12,
    "match_line_m": 0.40,
    "match_ambiguity_margin_m": 0.15,
    "s_ref_m": 15.0,
    "s_ref_max_extrapolation_m": 5.0,
    "width_conflict_run": 3,
    "width_conflict_agree_m": 0.12,
    # Exploratory search range from stage 4B. Not a normative rail gauge.
    "exploratory_distance_is_normative_gauge": False,
    "pair_direction_diff_inconsistent": 0.02,
    "pair_distance_change_inconsistent_m": 0.25,
    "direction_diff_score_scale": 0.08,
    "distance_change_score_scale_m": 0.25,
    "score_rival_margin": 0.15,
    "history_stability_in_score": False,
}


def ground_slopes(ground, s_m: float):
    """Local lateral slope of the base and along-track slope, if a base exists."""
    if not ground or ground.get("status") != "candidate":
        return None
    slope_l = None
    for bin_model in ground["local_bins"]:
        if bin_model["s_lo"] <= s_m < bin_model["s_hi"]:
            slope_l = float(bin_model["slope_dh_dl"])
            break
    if slope_l is None and ground["local_bins"]:
        nearest = min(ground["local_bins"], key=lambda item: abs(0.5 * (item["s_lo"] + item["s_hi"]) - s_m))
        if abs(0.5 * (nearest["s_lo"] + nearest["s_hi"]) - s_m) <= 1.0:
            slope_l = float(nearest["slope_dh_dl"])
    plane = ground.get("plane") or {}
    coef = plane.get("coef_h_from_s_l")
    slope_s = None if not coef else float(coef[0])
    if slope_l is None or slope_s is None:
        return None
    return slope_s, slope_l


def cross_track_frame(heading_dl_ds: float, slope_s: float, slope_l: float):
    """Unit path direction and across-base direction in working coordinates (s, l, h)."""
    dh = slope_s + slope_l * heading_dl_ds
    tangent = np.array([1.0, heading_dl_ds, dh], dtype=np.float64)
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm <= 0:
        return None
    tangent = tangent / tangent_norm
    normal = np.array([-slope_s, -slope_l, 1.0], dtype=np.float64)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 0:
        return None
    normal = normal / normal_norm
    lateral = np.cross(normal, tangent)
    lateral_norm = np.linalg.norm(lateral)
    if lateral_norm <= 1e-9:
        return None
    lateral = lateral / lateral_norm
    return tangent, lateral


def reference_line_distance(point_a, point_b, heading_dl_ds: float, slope_s: float, slope_l: float):
    """Distance along the base, perpendicular to the local path direction.

    `point_a` and `point_b` are (s, l, h). A difference of l at the same s is
    kept only as a diagnostic: it is not the distance when the path is angled
    or the base is tilted.
    """
    frame = cross_track_frame(heading_dl_ds, slope_s, slope_l)
    if frame is None:
        return None
    tangent, lateral = frame
    delta = np.asarray(point_b, dtype=np.float64) - np.asarray(point_a, dtype=np.float64)
    signed = float(np.dot(delta, lateral))
    midpoint = 0.5 * (np.asarray(point_a, dtype=np.float64) + np.asarray(point_b, dtype=np.float64))
    return {
        "reference_line_distance_m": abs(signed),
        "signed_cross_track_m": signed,
        "same_s_delta_l_m": abs(float(delta[1])),
        "midpoint_s_m": float(midpoint[0]),
        "midpoint_l_m": float(midpoint[1]),
        "midpoint_h_m": float(midpoint[2]),
        "direction_dl_ds": float(heading_dl_ds),
        "direction_heading_deg": float(np.degrees(np.arctan(heading_dl_ds))),
        "direction_vector_slh": [float(v) for v in tangent],
        "lateral_vector_slh": [float(v) for v in lateral],
        "base_slope_ds": float(slope_s),
        "base_slope_dl": float(slope_l),
    }


def _local_heading(s_values, l_values, index: int):
    s_values = np.asarray(s_values, dtype=np.float64)
    l_values = np.asarray(l_values, dtype=np.float64)
    if s_values.size < 2:
        return None
    if 0 < index < s_values.size - 1:
        left, right = index - 1, index + 1
    elif index == 0:
        left, right = 0, 1
    else:
        left, right = s_values.size - 2, s_values.size - 1
    ds = float(s_values[right] - s_values[left])
    if abs(ds) < 1e-9:
        return None
    return float((l_values[right] - l_values[left]) / ds)


def fit_midpoint_line(s_values, l_values, config=None):
    """Robust c(s)=b+k*s. b is extrapolated to s=0 and is not a body calibration."""
    config = PAIR_CONFIG if config is None else config
    s_values = np.asarray(s_values, dtype=np.float64)
    l_values = np.asarray(l_values, dtype=np.float64)
    order = np.argsort(s_values, kind="mergesort")
    s_values = s_values[order]
    l_values = l_values[order]
    if s_values.size < 2:
        return None
    slopes = []
    for i in range(s_values.size):
        for j in range(i + 1, s_values.size):
            ds = float(s_values[j] - s_values[i])
            if abs(ds) < 1e-9:
                continue
            slopes.append((float(l_values[j] - l_values[i])) / ds)
    if not slopes:
        return None
    k = float(np.median(slopes))
    b = float(np.median(l_values - k * s_values))
    residual = l_values - (b + k * s_values)
    abs_residual = np.abs(residual)
    p95 = float(np.percentile(abs_residual, 95))
    straight = bool(s_values.size >= config["min_straight_sections"] and p95 <= config["straight_residual_p95_m"])
    return {
        "model": "c(s)=b+k*s",
        "b_m": b,
        "k_dl_ds": k,
        "heading_deg": float(np.degrees(np.arctan(k))),
        "residual_p50_m": float(np.median(abs_residual)),
        "residual_p95_m": p95,
        "n_sections": int(s_values.size),
        "s_min_m": float(s_values.min()),
        "s_max_m": float(s_values.max()),
        "observed_span_m": float(s_values.max() - s_values.min()),
        "extrapolation_length_m": float(s_values.min()),
        "extrapolation_target_s_m": 0.0,
        "approximately_straight": straight,
        "b_is_observed": False,
        "linear_extrapolation_is_calibration": False,
        "linear_model_status": (
            "приблизительно прямой участок; b экстраполирован к s=0 и не является калибровкой крепления"
            if straight
            else "участок не принят прямым; линейная экстраполяция не используется как калибровка"
        ),
    }


def pose_at_s_ref(s_values, l_values, line, config=None):
    """Separate a direct midpoint observation from the model value at s_ref.

    The model value is never labeled as the observation. An unreliable model
    does not supply a position update.
    """
    config = PAIR_CONFIG if config is None else config
    s_ref = float(config.get("s_ref_m", config["publish_s_m"]))
    s_values = np.asarray(s_values, dtype=np.float64)
    l_values = np.asarray(l_values, dtype=np.float64)
    order = np.argsort(s_values, kind="mergesort")
    s_values = s_values[order]
    l_values = l_values[order]
    direct = None
    nearest = None
    if s_values.size:
        nearest_i = int(np.argmin(np.abs(s_values - s_ref)))
        nearest = {
            "s_obs_m": float(s_values[nearest_i]),
            "midpoint_l_m": float(l_values[nearest_i]),
            "is_model_value": False,
        }
        for s_i, l_i in zip(s_values, l_values):
            if abs(float(s_i) - s_ref) <= 1e-6:
                direct = {
                    "s_obs_m": float(s_i),
                    "midpoint_l_m": float(l_i),
                    "is_model_value": False,
                }
                break
    if s_values.size == 0:
        support = "extrapolation"
        outside = None
    elif float(s_values[0]) - 1e-9 > s_ref or float(s_values[-1]) + 1e-9 < s_ref:
        support = "extrapolation"
        outside = float(s_values[0] - s_ref) if s_ref < s_values[0] else float(s_ref - s_values[-1])
    elif direct is not None:
        support = "inside_observed_interval"
        outside = 0.0
    else:
        support = "gap"
        outside = 0.0
    model_c = None
    reliable = False
    reason = "пространственная модель не построена"
    quality = None
    if line is not None:
        quality = {
            "residual_p95_m": line.get("residual_p95_m"),
            "n_sections": line.get("n_sections"),
            "approximately_straight": line.get("approximately_straight"),
            "support": support,
        }
        if not line.get("approximately_straight"):
            reason = "участок не принят прямым, обновление положения на s_ref пропущено"
        elif int(line.get("n_sections") or 0) < config["min_straight_sections"]:
            reason = "мало сечений для модели, обновление положения на s_ref пропущено"
        elif support == "extrapolation" and (outside is None or outside > config["s_ref_max_extrapolation_m"]):
            reason = "экстраполяция дальше допуска, обновление положения на s_ref пропущено"
        else:
            reliable = True
            model_c = float(line["b_m"] + line["k_dl_ds"] * s_ref)
            reason = "модель на фиксированном s_ref принята"
    return {
        "s_ref_m": s_ref,
        "direct_observation": direct,
        "nearest_observation": nearest,
        "model_c_m": model_c,
        "model_support": support,
        "model_extrapolation_m": outside,
        "model_is_direct_observation": False,
        "model_reliable": reliable,
        "model_reason": reason,
        "quality": quality,
    }


def _lines_correspond(left_a, right_a, left_b, right_b, config) -> bool:
    return max(abs(float(left_a) - float(left_b)), abs(float(right_a) - float(right_b))) <= config["match_line_m"]


def observation_support(groups_a: int, groups_b: int) -> int:
    """Unique geometric groups on the two lines. Repeated records are not votes."""
    return int(groups_a) + int(groups_b)


def _sample_index(candidate):
    return {round(float(sample["s_m"]), 3): sample for sample in candidate.get("section_samples", [])}


def measure_candidate_pairs(candidates, ground, config=None):
    """Pair hypotheses inside one frame. No history and no own-track choice."""
    config = PAIR_CONFIG if config is None else config
    ordered = sorted(enumerate(candidates), key=lambda item: item[1].get("l_median_of_track_m", 0.0))
    measurements = []
    for left_pos in range(len(ordered)):
        for right_pos in range(left_pos + 1, len(ordered)):
            left_id, left = ordered[left_pos]
            right_id, right = ordered[right_pos]
            left_samples = _sample_index(left)
            right_samples = _sample_index(right)
            common = sorted(set(left_samples) & set(right_samples))
            if len(common) < config["min_common_sections"]:
                continue
            rough = []
            usable = []
            for key in common:
                sample_a = left_samples[key]
                sample_b = right_samples[key]
                rough.append((key, sample_a, sample_b))
            midpoint_s = [0.5 * (a["s_m"] + b["s_m"]) for _key, a, b in rough]
            midpoint_l = [0.5 * (a["l_m"] + b["l_m"]) for _key, a, b in rough]
            for index, (key, sample_a, sample_b) in enumerate(rough):
                heading = _local_heading(midpoint_s, midpoint_l, index)
                slopes = ground_slopes(ground, float(sample_a["s_m"]))
                if heading is None or slopes is None:
                    continue
                measured = reference_line_distance(
                    (sample_a["s_m"], sample_a["l_m"], sample_a["h_m"]),
                    (sample_b["s_m"], sample_b["l_m"], sample_b["h_m"]),
                    heading,
                    slopes[0],
                    slopes[1],
                )
                if measured is None:
                    continue
                measured.update(
                    {
                        "s_m": float(sample_a["s_m"]),
                        "line_a": {
                            "candidate_id": int(left_id),
                            "geometry_key": left.get("geometry_key"),
                            "l_m": float(sample_a["l_m"]),
                            "h_m": float(sample_a["h_m"]),
                            "dh_m": None if sample_a.get("dh_m") is None else float(sample_a["dh_m"]),
                            "width_m": None if sample_a.get("width_m") is None else float(sample_a["width_m"]),
                            "height_support": sample_a.get("height_support", "unknown"),
                            "groups": int(sample_a["groups"]),
                            "representative_indices": [int(v) for v in sample_a.get("representative_indices", [])],
                        },
                        "line_b": {
                            "candidate_id": int(right_id),
                            "geometry_key": right.get("geometry_key"),
                            "l_m": float(sample_b["l_m"]),
                            "h_m": float(sample_b["h_m"]),
                            "dh_m": None if sample_b.get("dh_m") is None else float(sample_b["dh_m"]),
                            "width_m": None if sample_b.get("width_m") is None else float(sample_b["width_m"]),
                            "height_support": sample_b.get("height_support", "unknown"),
                            "groups": int(sample_b["groups"]),
                            "representative_indices": [int(v) for v in sample_b.get("representative_indices", [])],
                        },
                        "support_groups": observation_support(sample_a["groups"], sample_b["groups"]),
                        "both_observed": True,
                        "assumed": False,
                        "is_new_measurement": True,
                    }
                )
                usable.append(measured)
            if len(usable) < config["min_agreeing_sections"]:
                continue
            distances = np.array([row["reference_line_distance_m"] for row in usable], dtype=np.float64)
            center = float(np.median(distances))
            if not (config["min_distance_m"] <= center <= config["max_distance_m"]):
                continue
            for row in usable:
                row["agrees_with_pair_median"] = abs(row["reference_line_distance_m"] - center) <= config["section_agree_m"]
            agreeing = [row for row in usable if row["agrees_with_pair_median"]]
            if len(agreeing) < config["min_agreeing_sections"]:
                continue
            agreed = np.array([row["reference_line_distance_m"] for row in agreeing], dtype=np.float64)
            line = fit_midpoint_line(
                [row["midpoint_s_m"] for row in agreeing],
                [row["midpoint_l_m"] for row in agreeing],
                config,
            )
            publish = min(agreeing, key=lambda row: abs(row["s_m"] - config["publish_s_m"]))
            pose = pose_at_s_ref(
                [row["midpoint_s_m"] for row in agreeing],
                [row["midpoint_l_m"] for row in agreeing],
                line,
                config,
            )
            measurements.append(
                {
                    "candidate_ids": [int(left_id), int(right_id)],
                    "local_ids_are_not_persistent": True,
                    "line_keys": [left.get("geometry_key"), right.get("geometry_key")],
                    "left_l_m": float(np.median([row["line_a"]["l_m"] for row in agreeing])),
                    "right_l_m": float(np.median([row["line_b"]["l_m"] for row in agreeing])),
                    "distance_m": float(np.median(agreed)),
                    "distance_mad_m": float(np.median(np.abs(agreed - np.median(agreed)))),
                    "support_groups": int(sum(row["support_groups"] for row in agreeing)),
                    "sections": usable,
                    "midpoint_line": line,
                    "publish": {
                        "s_m": publish["s_m"],
                        "target_s_m": config["publish_s_m"],
                        "target_section_observed": abs(publish["s_m"] - config["publish_s_m"]) < 1e-6,
                        "midpoint_l_m": publish["midpoint_l_m"],
                        "midpoint_h_m": publish["midpoint_h_m"],
                        "path_midpoint_relative_to_lidar_l_m": publish["midpoint_l_m"],
                        "lidar_l_relative_to_path_midpoint_m": -publish["midpoint_l_m"],
                        "source": "observed",
                        "not_a_body_mount_offset": True,
                    },
                    "pose_at_s_ref": pose,
                    "not_a_confirmed_gauge": True,
                    "quantity_name": "расстояние между опорными линиями",
                }
            )
    return measurements


def _median_tail(values, window: int):
    if not values:
        return None
    tail = values[-window:]
    return float(np.median(tail))


def _new_history(pair_id: str, frame_index: int, pair: dict):
    return {
        "pair_id": pair_id,
        "born_frame": frame_index,
        "last_left_l_m": pair["left_l_m"],
        "last_right_l_m": pair["right_l_m"],
        "admitted": [],
        "outliers": [],
        "pending_width": [],
        "conflict": None,
        "last_width_frame": None,
        "last_pose_frame": None,
        "pose_samples": [],
        "direction_samples": [],
        "events": [],
    }


def _pose_fields(history, frame_index: int, pair, config, allow_update: bool, timestamp_ns):
    pose = None if pair is None else pair.get("pose_at_s_ref")
    updated = False
    skip_reason = None
    if not allow_update:
        skip_reason = "положение на s_ref не обновляется: ширина этого кадра не принята"
    elif not pose or not pose.get("model_reliable") or pose.get("model_c_m") is None:
        skip_reason = None if not pose else pose.get("model_reason")
        if skip_reason is None:
            skip_reason = "нет надёжной модели на s_ref"
    else:
        history["pose_samples"].append(
            {
                "frame_index": frame_index,
                "s_ref_m": pose["s_ref_m"],
                "c_m": pose["model_c_m"],
                "support": pose["model_support"],
                "quality": pose.get("quality"),
                "timestamp_ns": timestamp_ns,
                "is_direct_observation": False,
            }
        )
        history["last_pose_frame"] = frame_index
        updated = True
    smoothed = _median_tail([item["c_m"] for item in history["pose_samples"]], config["pose_window"])
    last = history["last_pose_frame"]
    age = None if last is None else int(frame_index - last)
    return {
        "s_ref_m": None if not pose else pose.get("s_ref_m", config.get("s_ref_m", config["publish_s_m"])),
        "direct_observation": None if not pose else pose.get("direct_observation"),
        "nearest_observation": None if not pose else pose.get("nearest_observation"),
        "model_c_m": None if not pose else pose.get("model_c_m"),
        "model_support": None if not pose else pose.get("model_support"),
        "model_is_direct_observation": False,
        "model_reliable": False if not pose else bool(pose.get("model_reliable")),
        "estimate_c_s_ref_m": None if not updated else pose["model_c_m"],
        "smoothed_c_s_ref_m": smoothed,
        "pose_update": "accepted" if updated else "skipped",
        "pose_skip_reason": None if updated else skip_reason,
        "pose_age_frames": age,
        "pose_timestamp_ns": timestamp_ns,
        "pose_quality": None if not pose else pose.get("quality"),
        "smoothing_is_state_filter_at_s_ref": True,
        "smoothing_is_same_tunnel_place": False,
        "pose_l_smoothed_m": smoothed,
    }


def _width_decision(history, frame_index: int, pair, config):
    distance = float(pair["distance_m"])
    current = _median_tail([item["distance_m"] for item in history["admitted"]], config["width_window"])
    outside = current is not None and len(history["admitted"]) >= 3 and abs(distance - current) > config["width_admit_m"]
    sample = {
        "frame_index": frame_index,
        "distance_m": distance,
        "left_l_m": pair["left_l_m"],
        "right_l_m": pair["right_l_m"],
    }
    if not outside:
        history["pending_width"] = []
        return "admitted", "измерение принято"
    pending = history["pending_width"]
    if pending:
        prev = pending[-1]
        agrees = abs(distance - prev["distance_m"]) <= config["width_conflict_agree_m"] and _lines_correspond(
            pair["left_l_m"], pair["right_l_m"], prev["left_l_m"], prev["right_l_m"], config
        )
        if not agrees:
            if len(pending) < config["width_conflict_run"]:
                history["outliers"].extend(pending)
            pending = []
    if not _lines_correspond(pair["left_l_m"], pair["right_l_m"], history["last_left_l_m"], history["last_right_l_m"], config):
        history["outliers"].append(sample)
        history["pending_width"] = pending
        return "outlier", "линии не совпали с историей пары; измерение не усредняется со старой шириной"
    pending.append(sample)
    history["pending_width"] = pending
    if len(pending) >= config["width_conflict_run"]:
        frames = {item["frame_index"] for item in pending}
        history["outliers"] = [item for item in history["outliers"] if item["frame_index"] not in frames]
        hypothesis = [item["distance_m"] for item in pending]
        history["conflict"] = {
            "status": "WIDTH_CONFLICT",
            "frames": [item["frame_index"] for item in pending],
            "distances_m": hypothesis,
            "hypothesis_median_m": float(np.median(hypothesis)),
            "source_lines_match": True,
            "left_l_m": pair["left_l_m"],
            "right_l_m": pair["right_l_m"],
            "not_averaged_with_previous": True,
            "not_adopted": True,
            "adopted_only_by_frame_count": False,
        }
        return "width_conflict", "WIDTH_CONFLICT"
    history["outliers"].append(sample)
    return "outlier", "измерение далеко от накопленной медианы и сохранено отдельно как выброс"


def _width_event(history, frame_index: int, pair: dict, config, timestamp_ns=None):
    status, reason = _width_decision(history, frame_index, pair, config)
    admit = status == "admitted"
    if admit:
        history["admitted"].append(
            {
                "frame_index": frame_index,
                "distance_m": pair["distance_m"],
                "distance_mad_m": pair["distance_mad_m"],
                "support_groups": pair["support_groups"],
            }
        )
        history["last_left_l_m"] = pair["left_l_m"]
        history["last_right_l_m"] = pair["right_l_m"]
        history["last_width_frame"] = frame_index
        if pair.get("midpoint_line") and pair["midpoint_line"].get("k_dl_ds") is not None:
            history["direction_samples"].append(
                {
                    "frame_index": frame_index,
                    "k_dl_ds": pair["midpoint_line"]["k_dl_ds"],
                    "within_frame_only": True,
                }
            )
    smoothed = _median_tail([item["distance_m"] for item in history["admitted"]], config["width_window"])
    last_width = history["last_width_frame"]
    width_age = None if last_width is None else int(frame_index - last_width)
    pose = _pose_fields(history, frame_index, pair, config, admit, timestamp_ns)
    update = "accepted" if admit else "paused"
    if status == "width_conflict":
        update = "width_conflict"
    return {
        "pair_id": history["pair_id"],
        "frame_index": frame_index,
        "update": update,
        "width_status": status,
        "reason": reason,
        "raw_distance_m": pair["distance_m"],
        "smoothed_distance_m": smoothed,
        "raw_is_measurement": True,
        "fresh_width_measurement": admit,
        "width_age_frames": width_age,
        "smoothed_uses_this_frame": admit,
        "scatter_is_not_bias_removal": True,
        "width_conflict": history["conflict"] if status == "width_conflict" else None,
        "publish": pair["publish"],
        "midpoint_line": pair["midpoint_line"],
        "direction_k_smoothed": _median_tail(
            [item["k_dl_ds"] for item in history.get("direction_samples", [])],
            config["pose_window"],
        ),
        "support_groups": pair["support_groups"],
        "candidate_ids": pair["candidate_ids"],
        "line_keys": pair.get("line_keys"),
        "sections": pair["sections"],
        **pose,
    }


def _pause_event(history, frame_index: int, reason: str, config, assumed=None, raw_pair=None, timestamp_ns=None):
    smoothed = _median_tail([item["distance_m"] for item in history["admitted"]], config["width_window"])
    pose_k = [item["k_dl_ds"] for item in history["direction_samples"]]
    last_width = history["last_width_frame"]
    event = {
        "pair_id": history["pair_id"],
        "frame_index": frame_index,
        "update": "paused",
        "width_status": "paused",
        "reason": reason,
        "raw_distance_m": None if raw_pair is None else raw_pair["distance_m"],
        "smoothed_distance_m": smoothed,
        "raw_is_measurement": False,
        "fresh_width_measurement": False,
        "width_age_frames": None if last_width is None else int(frame_index - last_width),
        "smoothed_uses_this_frame": False,
        "scatter_is_not_bias_removal": True,
        "publish": None,
        "midpoint_line": None,
        "direction_k_smoothed": _median_tail(pose_k, config["pose_window"]),
        "support_groups": 0,
        "candidate_ids": [] if raw_pair is None else raw_pair["candidate_ids"],
        "sections": [] if raw_pair is None else raw_pair["sections"],
        "assumed_line": assumed,
        **_pose_fields(history, frame_index, raw_pair, config, False, timestamp_ns),
    }
    history["events"].append(event)
    return event


def _match_costs(histories, pairs, config):
    costs = []
    for h_index, history in enumerate(histories):
        for p_index, pair in enumerate(pairs):
            cost = max(
                abs(pair["left_l_m"] - history["last_left_l_m"]),
                abs(pair["right_l_m"] - history["last_right_l_m"]),
            )
            if cost <= config["match_line_m"]:
                costs.append((cost, h_index, p_index))
    return costs


def _ambiguous_indices(costs, config):
    by_history = {}
    by_pair = {}
    for cost, h_index, p_index in costs:
        by_history.setdefault(h_index, []).append(cost)
        by_pair.setdefault(p_index, []).append(cost)
    ambiguous_h = set()
    ambiguous_p = set()
    margin = config["match_ambiguity_margin_m"]
    for h_index, values in by_history.items():
        ordered = sorted(values)
        if len(ordered) >= 2 and ordered[1] - ordered[0] < margin:
            ambiguous_h.add(h_index)
    for p_index, values in by_pair.items():
        ordered = sorted(values)
        if len(ordered) >= 2 and ordered[1] - ordered[0] < margin:
            ambiguous_p.add(p_index)
    return ambiguous_h, ambiguous_p


def _one_line_assumption(history, line_positions, config):
    if not line_positions:
        return None
    positions = np.asarray(line_positions, dtype=np.float64)
    left_hit = positions[np.abs(positions - history["last_left_l_m"]) <= config["match_line_m"]]
    right_hit = positions[np.abs(positions - history["last_right_l_m"]) <= config["match_line_m"]]
    if left_hit.size and not right_hit.size:
        observed_l = float(np.median(left_hit))
        missing = "right"
        assumed_l = observed_l + (history["last_right_l_m"] - history["last_left_l_m"])
    elif right_hit.size and not left_hit.size:
        observed_l = float(np.median(right_hit))
        missing = "left"
        assumed_l = observed_l - (history["last_right_l_m"] - history["last_left_l_m"])
    else:
        return None
    previous = _median_tail([item["distance_m"] for item in history["admitted"]], config["width_window"])
    return {
        "label": "предполагаемая",
        "missing_side": missing,
        "observed_line_l_m": observed_l,
        "assumed_line_l_m": float(assumed_l),
        "previous_distance_used_as_prior_m": previous,
        "is_new_measurement": False,
    }


def link_pair_histories(frames, config=None):
    """Keep a separate history per pair. Ambiguous matches do not update it."""
    config = PAIR_CONFIG if config is None else config
    histories = []
    next_id = 1
    timeline = []
    for frame in frames:
        pairs = list(frame["pairs"])
        costs = _match_costs(histories, pairs, config)
        ambiguous_h, ambiguous_p = _ambiguous_indices(costs, config)
        free_h = set(range(len(histories))) - ambiguous_h
        free_p = set(range(len(pairs))) - ambiguous_p
        assigned_h = set()
        assigned_p = set()
        touched_by_ambiguous = {p_index for _cost, h_index, p_index in costs if h_index in ambiguous_h or p_index in ambiguous_p}
        for _cost, h_index, p_index in sorted(costs):
            if h_index in ambiguous_h or p_index in ambiguous_p:
                continue
            if h_index not in free_h or p_index not in free_p:
                continue
            if h_index in assigned_h or p_index in assigned_p:
                continue
            event = _width_event(
                histories[h_index], frame["frame_index"], pairs[p_index], config, frame.get("timestamp_ns")
            )
            histories[h_index]["events"].append(event)
            assigned_h.add(h_index)
            assigned_p.add(p_index)
        for h_index in sorted(ambiguous_h):
            competing = [pairs[p_index]["distance_m"] for _cost, hist_i, p_index in costs if hist_i == h_index]
            raw = None
            if competing:
                raw = {"distance_m": float(np.median(competing)), "candidate_ids": [], "sections": []}
            _pause_event(
                histories[h_index],
                frame["frame_index"],
                "неоднозначное соответствие",
                config,
                raw_pair=raw,
                timestamp_ns=frame.get("timestamp_ns"),
            )
        for h_index, history in enumerate(histories):
            if h_index in assigned_h or h_index in ambiguous_h:
                continue
            assumed = _one_line_assumption(history, frame.get("line_positions", []), config)
            if assumed is not None:
                reason = "одна опорная линия не наблюдалась; прежняя ширина — prior, не новое измерение"
            elif frame.get("ground_found", True):
                reason = "пара не наблюдалась"
            else:
                reason = "основание не выделено, пары не измерялись"
            _pause_event(
                history,
                frame["frame_index"],
                reason,
                config,
                assumed=assumed,
                timestamp_ns=frame.get("timestamp_ns"),
            )
        for p_index, pair in enumerate(pairs):
            if p_index in assigned_p or p_index in ambiguous_p or p_index in touched_by_ambiguous:
                continue
            history = _new_history(f"pair-{next_id}", frame["frame_index"], pair)
            next_id += 1
            event = _width_event(history, frame["frame_index"], pair, config, frame.get("timestamp_ns"))
            history["events"].append(event)
            histories.append(history)
        for p_index in sorted(touched_by_ambiguous - assigned_p):
            timeline.append(
                {
                    "pair_id": None,
                    "frame_index": frame["frame_index"],
                    "update": "paused",
                    "reason": "неоднозначное соответствие, новая история не открыта",
                    "raw_distance_m": pairs[p_index]["distance_m"],
                    "raw_is_measurement": False,
                    "candidate_ids": pairs[p_index]["candidate_ids"],
                    "sections": pairs[p_index]["sections"],
                }
            )
    for history in histories:
        timeline.extend(history["events"])
    timeline.sort(key=lambda item: (item["frame_index"], item["pair_id"] or ""))
    return {"histories": histories, "timeline": timeline}


def _segment_slopes(samples):
    ordered = sorted(samples, key=lambda item: float(item["s_m"]))
    slopes = []
    for left, right in zip(ordered, ordered[1:]):
        ds = float(right["s_m"] - left["s_m"])
        if abs(ds) < 1e-9:
            continue
        slopes.append(
            {
                "s_from_m": float(left["s_m"]),
                "s_to_m": float(right["s_m"]),
                "dl_ds": float(right["l_m"] - left["l_m"]) / ds,
            }
        )
    return slopes


def pair_consistency(sections, config=None):
    """Line agreement from section coordinates. Midpoint curvature is separate."""
    config = PAIR_CONFIG if config is None else config
    rows = [row for row in sections if row.get("line_a") and row.get("line_b")]
    rows = sorted(rows, key=lambda row: float(row["s_m"]))
    left = [{"s_m": row["s_m"], "l_m": row["line_a"]["l_m"]} for row in rows]
    right = [{"s_m": row["s_m"], "l_m": row["line_b"]["l_m"]} for row in rows]
    left_slopes = _segment_slopes(left)
    right_slopes = _segment_slopes(right)
    by_span = {
        (item["s_from_m"], item["s_to_m"]): item["dl_ds"]
        for item in right_slopes
    }
    differences = []
    for item in left_slopes:
        other = by_span.get((item["s_from_m"], item["s_to_m"]))
        if other is None:
            continue
        differences.append(
            {
                "s_from_m": item["s_from_m"],
                "s_to_m": item["s_to_m"],
                "left_dl_ds": item["dl_ds"],
                "right_dl_ds": other,
                "difference_dl_ds": item["dl_ds"] - other,
            }
        )
    distances = [
        {
            "s_m": float(row["s_m"]),
            "distance_m": float(row["reference_line_distance_m"])
            if row.get("reference_line_distance_m") is not None
            else abs(float(row["line_b"]["l_m"]) - float(row["line_a"]["l_m"])),
        }
        for row in rows
    ]
    distance_values = np.array([item["distance_m"] for item in distances], dtype=np.float64)
    if distance_values.size >= 2:
        ds = np.array([distances[-1]["s_m"] - distances[0]["s_m"]], dtype=np.float64)
        change = float(distance_values[-1] - distance_values[0])
        along = None if abs(float(ds[0])) < 1e-9 else change / float(ds[0])
        distance_span = float(np.max(distance_values) - np.min(distance_values))
    else:
        along = None
        distance_span = 0.0
    diff_values = np.array([item["difference_dl_ds"] for item in differences], dtype=np.float64)
    median_diff = None if diff_values.size == 0 else float(np.median(np.abs(diff_values)))
    midpoint = []
    for row in rows:
        midpoint.append(float(row["direction_dl_ds"]) if row.get("direction_dl_ds") is not None else None)
    midpoint = [value for value in midpoint if value is not None]
    midpoint_mad = None
    if midpoint:
        arr = np.asarray(midpoint, dtype=np.float64)
        midpoint_mad = float(np.median(np.abs(arr - np.median(arr))))
    support_s = [float(row["s_m"]) for row in rows]
    inconsistent = False
    reasons = []
    if median_diff is not None and median_diff > config["pair_direction_diff_inconsistent"]:
        inconsistent = True
        reasons.append("направления линий расходятся")
    if distance_span > config["pair_distance_change_inconsistent_m"]:
        inconsistent = True
        reasons.append("расстояние между линиями меняется вдоль пути")
    return {
        "left_direction_dl_ds": left_slopes,
        "right_direction_dl_ds": right_slopes,
        "direction_difference": differences,
        "median_abs_direction_difference": median_diff,
        "section_distances_m": distances,
        "distance_change_along_path": along,
        "distance_span_m": distance_span,
        "common_support_s_m": support_s,
        "common_support_length_m": 0.0 if len(support_s) < 2 else float(max(support_s) - min(support_s)),
        "midpoint_direction_mad": midpoint_mad,
        "midpoint_direction_is_not_line_agreement": True,
        "pair_geometry_inconsistent": inconsistent,
        "inconsistency_reasons": reasons,
    }


def compare_pair_features(pairs, ground=None):
    """Independent pair components. No component selects the own track."""
    fitting = None
    if ground:
        residual = ground.get("fitting_residual") or ground.get("local_residual") or {}
        fitting = residual.get("p95_m")
    rows = []
    for index, pair in enumerate(pairs):
        sections = [row for row in pair.get("sections", []) if row.get("agrees_with_pair_median", True)]
        headings = np.array([row["direction_dl_ds"] for row in sections], dtype=np.float64) if sections else np.array([])
        dh_a = [row["line_a"]["dh_m"] for row in sections if row["line_a"].get("dh_m") is not None]
        dh_b = [row["line_b"]["dh_m"] for row in sections if row["line_b"].get("dh_m") is not None]
        width_a = [row["line_a"]["width_m"] for row in sections if row["line_a"].get("width_m") is not None]
        width_b = [row["line_b"]["width_m"] for row in sections if row["line_b"].get("width_m") is not None]
        s_vals = [row["s_m"] for row in sections]
        keys = [key for key in (pair.get("line_keys") or []) if key]
        shared = []
        for other_i, other in enumerate(pairs):
            if other_i == index:
                continue
            other_keys = {key for key in (other.get("line_keys") or []) if key}
            common = [key for key in keys if key in other_keys]
            common_ids = sorted(set(pair.get("candidate_ids") or []) & set(other.get("candidate_ids") or []))
            if common or common_ids:
                shared.append(
                    {
                        "other_candidate_ids": other.get("candidate_ids"),
                        "shared_line_keys": common,
                        "shared_local_ids": common_ids,
                    }
                )
        dh_diff = None
        if dh_a and dh_b and len(dh_a) == len(dh_b):
            dh_diff = float(np.median(np.asarray(dh_b, dtype=np.float64) - np.asarray(dh_a, dtype=np.float64)))
        pose = pair.get("pose_at_s_ref") or {}
        line = pair.get("midpoint_line") or {}
        rows.append(
            {
                "candidate_ids": pair.get("candidate_ids"),
                "line_keys": keys,
                "local_ids_are_not_persistent": True,
                "distance_m": pair.get("distance_m"),
                "distance_scatter_m": pair.get("distance_mad_m"),
                "midpoint_direction_mad": None
                if headings.size == 0
                else float(np.median(np.abs(headings - np.median(headings)))),
                "consistency": pair_consistency(sections),
                "direction_residual_p95_m": line.get("residual_p95_m"),
                "dh_left_by_section_m": dh_a,
                "dh_right_by_section_m": dh_b,
                "dh_difference_m": dh_diff,
                "height_difference_uses_local_ground": True,
                "same_sensor_z_not_required": True,
                "fitting_residual_p95_m": fitting,
                "width_left_m": None if not width_a else float(np.median(width_a)),
                "width_right_m": None if not width_b else float(np.median(width_b)),
                "joint_support_sections": len(sections),
                "joint_s_min_m": None if not s_vals else float(min(s_vals)),
                "joint_s_max_m": None if not s_vals else float(max(s_vals)),
                "joint_length_m": None if len(s_vals) < 2 else float(max(s_vals) - min(s_vals)),
                "gaps_or_extrapolation": pose.get("model_support"),
                "model_c_s_ref_m": pose.get("model_c_m"),
                "model_is_direct_observation": False,
                "shared_lines": shared,
                "not_a_confirmed_rail_pair": True,
            }
        )
    return {
        "pairs": rows,
        "competition": "unresolved" if len(rows) > 1 else "single_pair_not_selected",
        "nearest_to_x0_not_selected": True,
        "nominal_gauge_not_used": True,
        "smallest_mad_is_not_sufficient": True,
        "most_frames_is_not_sufficient": True,
        "own_path_selected": False,
    }


def _assumed_sections(left_samples, right_samples):
    """A missing side inside the common span is a hint, not an observation."""
    left = {round(float(sample["s_m"]), 3): sample for sample in left_samples}
    right = {round(float(sample["s_m"]), 3): sample for sample in right_samples}
    common = sorted(set(left) & set(right))
    if len(common) < 2:
        return []
    separation = float(np.median([right[s]["l_m"] - left[s]["l_m"] for s in common]))
    assumed = []
    for station, sample in left.items():
        if station in right or not (common[0] <= station <= common[-1]):
            continue
        assumed.append(
            {
                "s_m": float(sample["s_m"]),
                "side": "right",
                "l_m": float(sample["l_m"] + separation),
                "observed": False,
                "adds_width_measurement": False,
            }
        )
    for station, sample in right.items():
        if station in left or not (common[0] <= station <= common[-1]):
            continue
        assumed.append(
            {
                "s_m": float(sample["s_m"]),
                "side": "left",
                "l_m": float(sample["l_m"] - separation),
                "observed": False,
                "adds_width_measurement": False,
            }
        )
    return assumed


def rank_pair_hypotheses(pairs, candidates, ground=None, config=None):
    """Engineering score. Not a probability and not an own-path choice."""
    config = PAIR_CONFIG if config is None else config
    fitting = None
    if ground:
        residual = ground.get("fitting_residual") or ground.get("local_residual") or {}
        fitting = residual.get("p95_m")
    by_id = {index: candidate for index, candidate in enumerate(candidates)}
    ranked = []
    for index, pair in enumerate(pairs):
        observed = [row for row in pair.get("sections", []) if row.get("agrees_with_pair_median", True)]
        consistency = pair_consistency(observed, config)
        left = by_id.get(pair["candidate_ids"][0], {})
        right = by_id.get(pair["candidate_ids"][1], {})
        assumed = _assumed_sections(left.get("section_samples") or [], right.get("section_samples") or [])
        widths = [
            row["line_a"].get("width_m")
            for row in observed
            if row["line_a"].get("width_m") is not None
        ] + [
            row["line_b"].get("width_m")
            for row in observed
            if row["line_b"].get("width_m") is not None
        ]
        dh_a = [row["line_a"]["dh_m"] for row in observed if row["line_a"].get("dh_m") is not None]
        dh_b = [row["line_b"]["dh_m"] for row in observed if row["line_b"].get("dh_m") is not None]
        dh_diff = None
        if dh_a and dh_b and len(dh_a) == len(dh_b):
            dh_diff = float(np.median(np.asarray(dh_b) - np.asarray(dh_a)))
        support_n = len(consistency["common_support_s_m"])
        support_c = min(1.0, support_n / 3.0)
        dk = consistency["median_abs_direction_difference"]
        dir_c = 1.0 if dk is None else 1.0 - min(1.0, abs(dk) / config["direction_diff_score_scale"])
        dist_c = 1.0 - min(1.0, consistency["distance_span_m"] / config["distance_change_score_scale_m"])
        height_scale = 0.05 if fitting is None else max(float(fitting), 0.05)
        height_c = 1.0 if dh_diff is None else 1.0 - min(1.0, abs(dh_diff) / height_scale)
        narrow_c = 1.0 if widths and max(widths) <= 0.60 else 0.4
        observed_c = 1.0 if not assumed else max(0.0, 1.0 - len(assumed) / max(support_n, 1))
        components = {
            "both_sides_support": support_c,
            "direction_agreement": dir_c,
            "distance_stability": dist_c,
            "height_agreement": height_c,
            "transverse_width": narrow_c,
            "observed_not_assumed": observed_c,
        }
        score = float(min(components.values()))
        penalties = []
        if consistency["pair_geometry_inconsistent"]:
            penalties.extend(consistency["inconsistency_reasons"])
        if support_c < 1.0:
            penalties.append("общая поддержка короче трёх сечений")
        if dir_c < 0.5:
            penalties.append("направления линий согласованы слабо")
        if dist_c < 0.5:
            penalties.append("расстояние заметно меняется вдоль пути")
        if height_c < 0.5:
            penalties.append("высоты над основанием расходятся сильнее остатка подгонки")
        if narrow_c < 1.0:
            penalties.append("поперечная структура шире порога сравнения 0,60 м")
        if assumed:
            penalties.append("есть предполагаемый участок, он не измеряет ширину")
        if consistency["pair_geometry_inconsistent"] or score < 0.35:
            kind = "insufficient_rail_features"
        elif score >= 0.6 and not penalties:
            kind = "consistent_rail_pair_hypothesis"
        else:
            kind = "ambiguous"
        ranked.append(
            {
                "hypothesis_id": f"H{index + 1}",
                "candidate_ids": pair.get("candidate_ids"),
                "local_ids_are_not_persistent": True,
                "line_keys": pair.get("line_keys"),
                "distance_m": pair.get("distance_m"),
                "distance_scatter_m": pair.get("distance_mad_m"),
                "engineering_score": score,
                "score_is_probability": False,
                "history_stability_in_score": False,
                "components": components,
                "penalties": penalties,
                "kind": kind,
                "consistency": consistency,
                "assumed_sections": assumed,
                "observed_sections": [
                    {
                        "s_m": row["s_m"],
                        "distance_m": row["reference_line_distance_m"],
                        "left_l_m": row["line_a"]["l_m"],
                        "right_l_m": row["line_b"]["l_m"],
                        "left_h_m": row["line_a"].get("h_m"),
                        "right_h_m": row["line_b"].get("h_m"),
                    }
                    for row in observed
                ],
                "shared_with": [],
                "not_an_own_path": True,
            }
        )
    for i, left in enumerate(ranked):
        for right in ranked[i + 1 :]:
            shared = False
            if left.get("line_keys") and right.get("line_keys"):
                shared = bool(set(left["line_keys"]) & set(right["line_keys"]))
            if not shared and left.get("candidate_ids") and right.get("candidate_ids"):
                shared = bool(set(left["candidate_ids"]) & set(right["candidate_ids"]))
            if not shared:
                continue
            left["shared_with"].append(right["hypothesis_id"])
            right["shared_with"].append(left["hypothesis_id"])
            note = "общая линия с другой гипотезой: конкурирующая комбинация, не отдельный путь"
            for item in (left, right):
                if note not in item["penalties"]:
                    item["penalties"].append(note)
            if abs(left["engineering_score"] - right["engineering_score"]) <= config["score_rival_margin"]:
                for item in (left, right):
                    if item["kind"] == "consistent_rail_pair_hypothesis":
                        item["kind"] = "ambiguous"
    return {
        "hypotheses": ranked,
        "own_path_selected": False,
        "nearest_to_x0_not_selected": True,
        "nominal_gauge_not_used": True,
        "exploratory_distance_m": [config["min_distance_m"], config["max_distance_m"]],
        "exploratory_distance_is_normative_gauge": False,
        "score_is_probability": False,
    }


def distance_from_corresponding_points(points_a, points_b):
    """Measure one geometric pair from corresponding 3D samples in working coordinates."""
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
    stacked = np.vstack([points_a, points_b])
    design = np.column_stack([stacked[:, 0], stacked[:, 1], np.ones(stacked.shape[0])])
    slope_s, slope_l, _intercept = np.linalg.lstsq(design, stacked[:, 2], rcond=None)[0]
    midpoint = 0.5 * (points_a + points_b)
    order = np.argsort(midpoint[:, 0], kind="mergesort")
    midpoint = midpoint[order]
    points_a = points_a[order]
    points_b = points_b[order]
    rows = []
    for index in range(points_a.shape[0]):
        heading = _local_heading(midpoint[:, 0], midpoint[:, 1], index)
        measured = reference_line_distance(points_a[index], points_b[index], heading, float(slope_s), float(slope_l))
        measured["same_s_delta_l_m"] = abs(float(points_b[index, 1] - points_a[index, 1]))
        rows.append(measured)
    line = fit_midpoint_line(midpoint[:, 0], midpoint[:, 1])
    distances = np.array([row["reference_line_distance_m"] for row in rows], dtype=np.float64)
    return {
        "distance_m": float(np.median(distances)),
        "same_s_delta_l_median_m": float(np.median([row["same_s_delta_l_m"] for row in rows])),
        "midpoint_line": line,
        "rows": rows,
    }


def sensor_shift_yaw(s, l, h, shift_l_m: float, yaw_rad: float):
    """Lateral sensor shift, then yaw about the sensor Z axis. Working prior is reapplied."""
    x = np.asarray(l, dtype=np.float64) + shift_l_m
    y = -np.asarray(s, dtype=np.float64)
    z = np.asarray(h, dtype=np.float64)
    cosine = float(np.cos(yaw_rad))
    sine = float(np.sin(yaw_rad))
    x2 = cosine * x - sine * y
    y2 = sine * x + cosine * y
    return -y2, x2, z


def transformed_pair_check(shift_l_m: float, yaw_rad: float):
    """One pair on a tilted plane. Shift and yaw must preserve the cross-track distance."""
    s_values = np.array([6.0, 10.0, 15.0, 20.0, 25.0])
    left_l = np.full(s_values.shape, -0.25)
    right_l = np.full(s_values.shape, 1.27)
    slope_s = 0.001
    slope_l = 0.04

    def pack(s_line, l_line):
        h_line = -1.30 + slope_s * s_line + slope_l * l_line
        return np.column_stack([s_line, l_line, h_line])

    before = distance_from_corresponding_points(pack(s_values, left_l), pack(s_values, right_l))
    s_left, l_left, h_left = sensor_shift_yaw(s_values, left_l, pack(s_values, left_l)[:, 2], shift_l_m, yaw_rad)
    s_right, l_right, h_right = sensor_shift_yaw(s_values, right_l, pack(s_values, right_l)[:, 2], shift_l_m, yaw_rad)
    after = distance_from_corresponding_points(
        np.column_stack([s_left, l_left, h_left]),
        np.column_stack([s_right, l_right, h_right]),
    )
    before["same_s_cut_delta_l_m"] = _same_s_cut_delta_l(s_values, left_l, s_values, right_l)
    after["same_s_cut_delta_l_m"] = _same_s_cut_delta_l(s_left, l_left, s_right, l_right)
    return {"before": before, "after": after, "shift_l_m": shift_l_m, "yaw_rad": yaw_rad}


def _same_s_cut_delta_l(s_a, l_a, s_b, l_b) -> float:
    """Horizontal l difference on a cut of constant s. This is not the cross-track distance."""
    s_a = np.asarray(s_a, dtype=np.float64)
    l_a = np.asarray(l_a, dtype=np.float64)
    s_b = np.asarray(s_b, dtype=np.float64)
    l_b = np.asarray(l_b, dtype=np.float64)
    order_a = np.argsort(s_a, kind="mergesort")
    order_b = np.argsort(s_b, kind="mergesort")
    s_a, l_a = s_a[order_a], l_a[order_a]
    s_b, l_b = s_b[order_b], l_b[order_b]
    lo = float(max(s_a[0], s_b[0]))
    hi = float(min(s_a[-1], s_b[-1]))
    grid = np.linspace(lo, hi, 5)
    return float(np.median(np.abs(np.interp(grid, s_b, l_b) - np.interp(grid, s_a, l_a))))
