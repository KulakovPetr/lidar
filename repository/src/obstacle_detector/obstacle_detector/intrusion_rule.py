"""Decide after candidates exist. Candidates themselves are not edited.

The thresholds are the existing detector values. This is not an E01 lattice test.
"""

from __future__ import annotations

import time


RULE_ID = "configured_volume_unexplained_cluster"


def _extent(candidate: dict) -> tuple[float, float, float]:
    observed = candidate.get("observed_extent_m") or {}
    return (
        float(observed.get("s_m") or 0.0),
        float(observed.get("l_m") or 0.0),
        float(observed.get("h_m") or 0.0),
    )


def _reject(candidate: dict, reason: str) -> dict:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "classification": "candidate_only",
        "reason": reason,
    }


def decide(
    candidates,
    *,
    corridor_source: str,
    profile_applicable: bool | None,
    ground_filter_available: bool,
    status: str,
    object_size_m: list,
    large_min_unique_xyz: int,
    orientation_unconfirmed: bool,
    thin_length_m: float = 0.80,
    thin_min_points: int = 10,
    continuation_from_s_m: float | None = 25.0,
    thin_floor_h_m: float | None = None,
    thin_clearance_m: float = 1.0,
) -> dict:
    """Classify the frame. `candidates` is read and returned unchanged."""
    started = time.perf_counter()
    listed = list(candidates or [])
    base = {
        "rule_id": RULE_ID,
        "intrusion_rule_present": True,
        "temporal_confirmation_required_for_single_frame": False,
        "orientation": "unconfirmed" if orientation_unconfirmed else "mount_applied",
        "orientation_doubt_remains": bool(orientation_unconfirmed),
        "not_a_proven_danger_to_the_train": True,
        "conditional_intrusions": [],
        "candidate_decisions": [],
        "candidates_not_modified": True,
        "parameters": {
            "large_min_unique_xyz": int(large_min_unique_xyz),
            "large_min_origin": "config/detector.yaml manual_estimate fixed",
            "object_size_m": [float(value) for value in object_size_m],
            "object_size_origin": "config/detector.yaml organizer stated",
            "object_size_is_a_minimum_not_a_maximum": True,
            "thin_length_m": float(thin_length_m),
            "thin_min_points": int(thin_min_points),
            "thin_origin": "config/detector.yaml existing thin_length_m and thin_points",
            "vertical_face_height_m": float(object_size_m[0]) if object_size_m else 0.30,
            "wide_flat_sheet_h_m": 0.05,
            "wide_flat_sheet_l_m": 1.50,
            "wide_flat_sheet_origin": "measured full-width sheet a few millimetres thick, not a wire",
            "one_return_is_not_a_wire": True,
        },
    }
    if status in ("processing_timeout", "processing_error"):
        base["decision"] = status
        base["reason"] = "frame_did_not_finish"
        base["timing_s"] = time.perf_counter() - started
        return base
    if status == "insufficient_geometry":
        base["decision"] = "insufficient_geometry"
        base["reason"] = "ground_geometry_not_built"
        base["timing_s"] = time.perf_counter() - started
        return base
    if corridor_source != "configured_straight":
        base["decision"] = "insufficient_evidence"
        base["reason"] = "explicit_profile_not_selected"
        base["timing_s"] = time.perf_counter() - started
        return base
    if profile_applicable is False:
        base["decision"] = "insufficient_evidence"
        base["reason"] = "configured_straight_not_applicable_for_decision"
        base["timing_s"] = time.perf_counter() - started
        return base
    if not ground_filter_available:
        base["decision"] = "insufficient_evidence"
        base["reason"] = "ground_filter_unavailable_volume_points_are_not_obstacles"
        for candidate in listed:
            base["candidate_decisions"].append(_reject(candidate, "ground_filter_unavailable"))
        base["timing_s"] = time.perf_counter() - started
        return base
    size_s, size_l, size_h = (float(value) for value in object_size_m)
    minimum = int(large_min_unique_xyz)
    thin_need = int(thin_min_points)
    thin_len = float(thin_length_m)
    face_h = size_s
    for candidate in listed:
        unique = int(candidate.get("unique_xyz") or 0)
        span_s, span_l, span_h = _extent(candidate)
        branch = candidate.get("branch")
        if unique <= 1:
            base["candidate_decisions"].append(_reject(candidate, "single_point_is_weak"))
            continue
        if branch not in ("volume_or_suspended", "local_protrusion"):
            base["candidate_decisions"].append(_reject(candidate, "not_an_unexplained_volume_cluster"))
            continue
        horizontal = span_s + 1e-9 >= size_s or span_l + 1e-9 >= size_l
        tall = span_h + 1e-9 >= face_h
        height_ok = span_h + 1e-9 >= size_h
        h_min = candidate.get("observed_h_min_m")
        elevated = thin_floor_h_m is None or (
            h_min is not None and float(h_min) + 1e-9 >= float(thin_floor_h_m) + float(thin_clearance_m)
        )
        wide_flat_sheet = span_h + 1e-9 < 0.05 and span_l + 1e-9 >= 1.50
        long_thin = max(span_s, span_l) + 1e-9 >= thin_len and unique >= thin_need and elevated and not wide_flat_sheet
        evidence = None
        if branch == "local_protrusion" and unique >= minimum and height_ok and (horizontal or tall):
            evidence = "low_protrusion"
        elif branch == "volume_or_suspended" and unique >= minimum and height_ok and horizontal:
            evidence = "volume"
        elif branch == "volume_or_suspended" and unique >= minimum and tall:
            evidence = "vertical_face"
        elif branch == "volume_or_suspended" and long_thin:
            evidence = "thin_fragment"
        if evidence is None:
            if branch == "local_protrusion":
                reason = "low_protrusion_seen_not_accepted"
            elif wide_flat_sheet and unique >= thin_need and max(span_s, span_l) + 1e-9 >= thin_len:
                reason = "wide_flat_sheet_not_a_fragment"
            elif unique < minimum and not long_thin:
                reason = "support_below_existing_large_min"
            else:
                reason = "extent_below_stated_object_size"
            base["candidate_decisions"].append(_reject(candidate, reason))
            continue
        s_min = candidate.get("observed_s_min_m")
        if s_min is None:
            segment = "position_unknown"
        elif continuation_from_s_m is not None and float(s_min) >= float(continuation_from_s_m):
            segment = "assumed_continuation"
        else:
            segment = "near_straight_assumption"
        base["conditional_intrusions"].append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "evidence": evidence,
                "confirmed_wire": False,
                "corridor_segment": segment,
                "corridor_segment_is_not_a_confirmed_path": True,
                "range_from_lidar_m": candidate.get("range_from_lidar_m"),
                "range_points": "minimum range of this candidate's own points",
                "range_not_taken_from_an_evaluation_mask": True,
                "features_used": [
                    "geometric_inside_configured_volume",
                    evidence,
                    "support_rule_for_this_evidence",
                ],
                "observed_extent_m": candidate.get("observed_extent_m"),
                "observed_extent_is_not_the_physical_object_size": True,
                "unique_xyz": unique,
                "profile_assumptions_remain": list(candidate.get("profile_assumptions") or []),
                "orientation": base["orientation"],
                "temporal_confirmation_used": False,
                "not_a_proven_danger_to_the_train": True,
            }
        )
    if base["conditional_intrusions"]:
        base["decision"] = "conditional_intrusion"
        base["reason"] = "unexplained_volume_cluster_inside_configured_profile"
    elif listed:
        base["decision"] = "candidate_only"
        base["reason"] = "observations_present_rule_did_not_accept_them"
    else:
        base["decision"] = "no_intrusion_detected"
        base["reason"] = "rule_applicable_no_intrusion_in_configured_volume"
        base["configured_volume_is_not_declared_free"] = True
        base["outside_configured_range_is_not_free"] = True
    base["timing_s"] = time.perf_counter() - started
    return base
