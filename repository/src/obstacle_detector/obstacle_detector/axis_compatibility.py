"""Per-hypothesis comparison with a straight central corridor.

Curvature is not the spread of every axis taken together.
A constant offset, a constant heading and a change of heading are different.
An adjacent track or a weak line combination does not forbid the corridor.
Unknown ownership is axis_compatibility_unknown, not confirmed_curve.
"""

from __future__ import annotations


def _mid_l(section: dict) -> float:
    return 0.5 * (float(section["left_l_m"]) + float(section["right_l_m"]))


def _slope(sections: list) -> float | None:
    if len(sections) < 2:
        return None
    s0 = float(sections[0]["s_m"])
    s1 = float(sections[-1]["s_m"])
    if abs(s1 - s0) < 1e-6:
        return None
    return (_mid_l(sections[-1]) - _mid_l(sections[0])) / (s1 - s0)


def assess_hypothesis(hypothesis: dict, spec: dict) -> dict:
    sections = sorted(hypothesis.get("observed_sections") or [], key=lambda row: float(row["s_m"]))
    kind = hypothesis.get("kind")
    lateral = None if not sections else float(sum(_mid_l(row) for row in sections) / len(sections))
    span = None if len(sections) < 2 else float(sections[-1]["s_m"]) - float(sections[0]["s_m"])
    direction = _slope(sections)
    change = None
    if len(sections) >= 3 and span is not None and span + 1e-9 >= float(spec["min_support_span_m"]):
        half = max(2, len(sections) // 2)
        change = None if _slope(sections[:half]) is None or _slope(sections[half - 1 :]) is None else abs(
            _slope(sections[half - 1 :]) - _slope(sections[:half])
        )
    heading = float(spec["heading_tolerance_dl_ds"])
    half_width = 0.5 * float(spec["width_m"])
    center = float(spec["center_l_m"])
    features = []
    if lateral is not None and abs(lateral - center) > half_width:
        features.append("constant_lateral_offset")
    if direction is not None and abs(direction) > heading and (change is None or change <= heading):
        features.append("constant_heading")
    if change is not None and change > heading:
        features.append("direction_changes_along_support")
    weak = kind != "consistent_rail_pair_hypothesis" or len(sections) < 3 or span is None or span < float(spec["min_support_span_m"])
    offset = "constant_lateral_offset" in features
    shared = bool(hypothesis.get("shared_with"))
    ownership_known = bool(hypothesis.get("own_path_assigned"))
    contradicts = (
        (not weak)
        and (not offset)
        and (not shared)
        and ("direction_changes_along_support" in features)
        and ownership_known
    )
    if weak or shared:
        reason = "weak_or_shared_combination_does_not_forbid"
    elif offset:
        reason = "lateral_offset_is_not_evidence_of_the_central_path"
    elif "direction_changes_along_support" in features and not ownership_known:
        reason = "direction_change_seen_but_axis_ownership_unknown"
    elif "constant_heading" in features:
        reason = "constant_heading_is_not_curvature"
    else:
        reason = "no_established_contradiction"
    return {
        "hypothesis_id": hypothesis.get("hypothesis_id"),
        "kind": kind,
        "not_an_own_path": True,
        "lateral_position_m": lateral,
        "direction_dl_ds": direction,
        "direction_change_dl_ds": change,
        "support_length_m": span,
        "observed_samples": len(sections),
        "support_quality": kind,
        "features": features,
        "axis_compatibility": "confirmed_curve" if contradicts else "axis_compatibility_unknown",
        "reason": reason,
        "forbids_configured_straight": bool(contradicts),
        "tolerances": {
            "heading_tolerance_dl_ds": heading,
            "heading_tolerance_source": spec.get("heading_tolerance_source"),
            "min_support_span_m": float(spec["min_support_span_m"]),
            "min_support_span_source": spec.get("min_support_span_source"),
            "half_width_m": half_width,
            "half_width_source": "half of organizer profile width",
            "stage2_combined_axis_span_not_used": True,
        },
    }


def assess_frame(hypotheses: list, spec: dict) -> dict:
    rows = [assess_hypothesis(item, spec) for item in hypotheses]
    contradicted = [row["hypothesis_id"] for row in rows if row["forbids_configured_straight"]]
    return {
        "per_hypothesis": rows,
        "combined_axis_span_not_used": True,
        "configured_straight_applicable_for_decision": not contradicted,
        "contradicting_hypotheses": contradicted,
        "outside_profile_not_widened": True,
    }
