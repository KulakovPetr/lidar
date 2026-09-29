"""Configured straight corridor. One ground estimate is supplied by the caller.

Geometric membership, justification of the volume, ground-filter availability
and candidate quality are separate fields. A missing rail pair does not stop
this profile.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from obstacle_detector.axis_compatibility import assess_frame
from obstacle_detector.detector import detect
from obstacle_detector.local_ground import height_above_local, lateral_support_mask


def load_profile(path: Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    prior = raw["prior"]
    profile = raw["profile"]
    longi = raw["longitudinal"]
    axis = raw["axis_check"]
    return {
        "raw": raw,
        "source_path_name": Path(path).name,
        "s_min_m": float(longi["s_min_m"]),
        "s_max_m": float(longi["s_max_m"]),
        "far_from_s_m": float(longi["far_assumption_from_s_m"]),
        "width_m": float(profile["width_m"]),
        "height_m": float(profile["height_m"]),
        "center_l_m": float(profile["center_l_m"]),
        "lower_above_rail_m": float(profile["lower_boundary_above_rail_head_m"]),
        "rail_head_h_m": float(prior["rail_head_h_if_z_up_and_angles_zero_m"]),
        "heading_tolerance_dl_ds": float(axis["heading_tolerance_dl_ds"]),
        "heading_tolerance_source": axis["heading_tolerance_source"],
        "min_support_span_m": float(axis["min_support_span_m"]),
        "min_support_span_source": axis["min_support_span_source"],
        "lower_boundary_is_not_zero_above_estimated_ground": True,
    }


def geometric_membership(s, l, h, spec: dict) -> dict:
    """Hit test against the configured volume. s-overlap alone is not inside."""
    s_min = float(spec["s_min_m"])
    s_max = float(spec["s_max_m"])
    half = 0.5 * float(spec["width_m"])
    floor_h = float(spec["rail_head_h_m"]) + float(spec["lower_above_rail_m"])
    top_h = floor_h + float(spec["height_m"])
    in_s = (s >= s_min) & (s <= s_max)
    in_l = np.abs(l - float(spec["center_l_m"])) <= half
    in_h = (h >= floor_h) & (h <= top_h)
    inside = in_s & in_l & in_h
    return {
        "inside": inside,
        "outside_configured_range": ~in_s,
        "floor_h_m": floor_h,
        "top_h_m": top_h,
        "half_width_m": half,
        "vertical_reference": "rail_head_assumption_not_estimated_ground",
        "far_assumption": s >= float(spec["far_from_s_m"]),
    }


def run_configured_straight(
    groups, geometry, spec: dict, detector_config, flags, precomputed_dh=None, precomputed_support=None
) -> dict:
    s = np.asarray(geometry["s"], dtype=np.float64)
    l = np.asarray(geometry["l"], dtype=np.float64)
    h = np.asarray(geometry["h"], dtype=np.float64)
    started_axis = __import__("time").perf_counter()
    axes = assess_frame(list(geometry["ranked"]["hypotheses"]), spec)
    axis_s = __import__("time").perf_counter() - started_axis
    volume = geometric_membership(s, l, h, spec)
    ground = geometry["ground"]
    ground_ok = ground.get("status") == "candidate"
    profile_record = {
        "corridor_source": "configured_straight",
        "coordinate_frame": geometry.get("coordinate_frame"),
        "prior": "s=-Y, l=X, h=Z",
        "prior_is_assumption": True,
        "s_min_m": spec["s_min_m"],
        "s_max_m": spec["s_max_m"],
        "far_assumption_from_s_m": spec["far_from_s_m"],
        "search_horizon_is_not_a_confirmed_path": True,
        "estimated_trajectory": "straight_prior_not_a_fitted_path",
        "position_uncertainty": "orientation_and_heading_unconfirmed",
        "width_m": spec["width_m"],
        "height_m": spec["height_m"],
        "center_l_m": spec["center_l_m"],
        "parameter_source": spec["source_path_name"],
        "rail_head_h_m": spec["rail_head_h_m"],
        "rail_head_h_requires_unconfirmed_zero_angles": True,
        "lower_boundary_above_rail_head_m": spec["lower_above_rail_m"],
        "lower_boundary_is_not_zero_above_estimated_ground": True,
        "outside_configured_range_is_not_free": True,
        "volume_justification": "assumption_not_observed_path",
        "ground_filter_available": ground_ok,
        "ground_status": ground.get("status"),
        "ground_reason": ground.get("reason"),
        "equivalent_to_supported_detection": False,
    }
    if not axes["configured_straight_applicable_for_decision"]:
        return {
            "applicable_for_decision": False,
            "axis_assessment": axes,
            "profile": profile_record,
            "geometric_inside": int(volume["inside"].sum()),
            "outside_configured_range": int(volume["outside_configured_range"].sum()),
            "detected": None,
            "skipped_branches": ["clustering_not_a_reliable_decision"],
            "timing_s": {"axis": axis_s, "filters": 0.0, "connectivity": 0.0},
            "degraded_without_ground": not ground_ok,
        }
    relation = np.full(s.shape, "outside", dtype=object)
    relation[volume["inside"]] = "inside"
    skipped = ["path_line_filter_no_pair_supplied"]
    if ground_ok:
        dh = height_above_local(s, l, h, ground) if precomputed_dh is None else precomputed_dh
        support = lateral_support_mask(s, l, ground) if precomputed_support is None else precomputed_support
        residual = (ground.get("fitting_residual") or {}).get("p95_m")
        local_enabled = bool(flags["local_protrusions_enabled"])
    else:
        dh = np.full(s.shape, np.nan) if precomputed_dh is None else precomputed_dh
        support = np.zeros(s.shape, dtype=bool) if precomputed_support is None else precomputed_support
        residual = None
        local_enabled = False
        skipped.extend(["thin_surface_filter", "ambiguous_surface_layer", "local_protrusion_branch"])
    ranges = np.sqrt(np.asarray(groups["x"], dtype=np.float64) ** 2 + np.asarray(groups["y"], dtype=np.float64) ** 2 + np.asarray(groups["z"], dtype=np.float64) ** 2)
    if "first_index" in groups:
        indices = np.asarray(groups["first_index"], dtype=np.int64).reshape(-1)
    else:
        indices = np.array([int(np.asarray(members)[0]) for members in groups["indices"]], dtype=np.int64)
    corridor = {
        "hypothesis_id": "configured_straight",
        "kind": "configured_straight",
        "section_half_m": 0.5,
        "lower_boundary_status": "assumption",
        "lower_boundary_m": None,
        "lower_boundary_is_rail_head_not_ground": True,
    }
    detected = detect(
        s, l, h, dh, support, ranges, indices, relation, corridor, [], residual, detector_config,
        connectivity=flags["connectivity"],
        local_protrusions_enabled=local_enabled,
    )
    return {
        "applicable_for_decision": True,
        "axis_assessment": axes,
        "profile": profile_record,
        "geometric_inside": int(volume["inside"].sum()),
        "outside_configured_range": int(volume["outside_configured_range"].sum()),
        "detected": detected,
        "skipped_branches": skipped,
        "timing_s": {
            "axis": axis_s,
            "filters": float(detected["timing_s"]["filter"]),
            "connectivity": float(detected["timing_s"]["link"]),
        },
        "degraded_without_ground": not ground_ok,
        "candidate_quality_not_raised_without_ground": not ground_ok,
    }
