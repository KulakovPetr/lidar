"""Conditional straight corridor for one diagnostic comparison.

It does not select a path, does not invent a ground plane, and does not
change clustering thresholds. The longitudinal limits come from the caller.
"""

from __future__ import annotations

import numpy as np

from obstacle_detector.detector import detect
from obstacle_detector.local_ground import height_above_local, lateral_support_mask


def conditional_volume(s, l, h, spec: dict) -> dict:
    """Cross-section test. An s-interval overlap alone is not inside."""
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
        "in_s_only": in_s & ~(in_l & in_h),
        "floor_h_m": floor_h,
        "top_h_m": top_h,
        "half_width_m": half,
        "support": "assumption_not_observed_path",
        "far_assumption": s >= float(spec["far_from_s_m"]),
    }


def run_diagnostic(groups, geometry, spec: dict, detector_config, flags) -> dict:
    """Same detector on a rail-independent volume. Ground filters run only if a real ground exists."""
    s = np.asarray(geometry["s"], dtype=np.float64)
    l = np.asarray(geometry["l"], dtype=np.float64)
    h = np.asarray(geometry["h"], dtype=np.float64)
    volume = conditional_volume(s, l, h, spec)
    relation = np.full(s.shape, "outside", dtype=object)
    relation[volume["inside"]] = "inside"
    ground = geometry["ground"]
    ground_ok = ground.get("status") == "candidate"
    skipped = []
    if ground_ok:
        dh = height_above_local(s, l, h, ground)
        support = lateral_support_mask(s, l, ground)
        residual = (ground.get("fitting_residual") or {}).get("p95_m")
        local_enabled = bool(flags["local_protrusions_enabled"])
    else:
        dh = np.full(s.shape, np.nan)
        support = np.zeros(s.shape, dtype=bool)
        residual = None
        local_enabled = False
        skipped.append("thin_surface_filter")
        skipped.append("ambiguous_surface_layer")
        skipped.append("local_protrusion_branch")
    # No rail pair is supplied. The path-line filter therefore has nothing to apply.
    skipped.append("path_line_filter_no_pair_supplied")
    ranges = np.sqrt(groups["x"] ** 2 + groups["y"] ** 2 + groups["z"] ** 2)
    indices = np.array([int(np.asarray(members)[0]) for members in groups["indices"]], dtype=np.int64)
    corridor = {
        "hypothesis_id": "B",
        "kind": "diagnostic_conditional_straight",
        "section_half_m": 0.5,
        "lower_boundary_status": "assumption",
        "lower_boundary_m": float(spec["lower_above_rail_m"]),
    }
    detected = detect(
        s,
        l,
        h,
        dh,
        support,
        ranges,
        indices,
        relation,
        corridor,
        [],
        residual,
        detector_config,
        connectivity=flags["connectivity"],
        local_protrusions_enabled=local_enabled,
    )
    for item in (
        detected["candidates_large"]
        + detected["candidates_rare"]
        + detected["candidates_protrusion_large"]
        + detected["candidates_protrusion_rare"]
    ):
        item["doubt_reasons"] = list(item.get("doubt_reasons") or []) + [
            "условный прямой коридор, не подтверждённый путь",
            "ориентация установки не подтверждена",
        ]
        if not ground_ok:
            item["doubt_reasons"].append("основание недоступно, фильтр поверхности не выполнен")
        item["degraded_without_ground"] = not ground_ok
        item["confidence_not_raised"] = True
    return {
        "volume": {key: value for key, value in volume.items() if key != "inside" and key != "in_s_only" and key != "far_assumption"},
        "inside_count": int(volume["inside"].sum()),
        "ground_status": ground.get("status"),
        "ground_used": ground_ok,
        "skipped_branches": skipped,
        "surface_mode": detected["surface_mode"],
        "detected": detected,
        "equivalent_to_supported_detection": False,
    }
