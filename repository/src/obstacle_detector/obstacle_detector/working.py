"""One-frame working output. Does not read earlier reports.

New options default off: KISS-ICP and accumulated local context.
Absence of candidates is not CLEAR. Competing corridors stay separate.
"""

from __future__ import annotations

import hashlib
import json

from obstacle_detector.frame_pipeline import algorithm_version, process_grouped_cloud
from obstacle_detector.intrusion_rule import decide
from obstacle_detector.tracking import TrackSession


def config_version(detector_config, corridor_config, mount, flags, session_attachment=None) -> str:
    """Hash the parameters actually used, not the default files on disk."""
    payload = {
        "detector": detector_config,
        "corridor": corridor_config,
        "mount": mount,
        "variant": flags,
        "session_attachment": session_attachment,
    }
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def options_from_config(detector_config: dict) -> dict:
    block = detector_config.get("temporal") or {}
    return {
        "kiss_icp": bool(block.get("kiss_icp", False)),
        "local_context": bool(block.get("local_context", False)),
        "temporal_enabled": bool(block.get("enabled", True)),
        "confirm_frames": int(block.get("confirm_frames", 2)),
        "confirm_window": int(block.get("confirm_window", 3)),
        "train_nose_offset_m": block.get("train_nose_offset_m"),
    }


def process_working_frame(groups, corridor_config, detector_config, mount, flags, stamp_ns, frame_id, tracker: TrackSession | None = None, pose=None, registration_reliable=None, local_context_slh=None, frame_key=0, session_attachment=None, message_index=None, header_frame_id=None, straight_profile=None):
    options = options_from_config(detector_config)
    context = local_context_slh if options["local_context"] else None
    used_pose = pose if options["kiss_icp"] else None
    result = process_grouped_cloud(
        groups,
        corridor_config,
        detector_config,
        mount,
        flags,
        local_context_slh=context,
        detailed=True,
        session_attachment=session_attachment,
        straight_profile=straight_profile,
    )
    candidates = []
    for item in result["candidates"]:
        item = dict(item)
        item["range_from_lidar_m"] = item.get("range_from_lidar_m")
        item["range_from_train_nose_m"] = None
        item["range_from_train_nose_geometry_not_defined"] = True
        candidates.append(item)
    views = []
    state = {"mode": "not_run", "tracks": []}
    if tracker is not None and options["temporal_enabled"]:
        linkable = [item for item in candidates if "position_m" in item]
        views, state = tracker.update(
            linkable,
            frame_key,
            pose=used_pose,
            registration_reliable=registration_reliable if options["kiss_icp"] else None,
            coordinate_frame=result.get("coordinate_frame"),
            stamp_ns=stamp_ns,
        )
        by_id = {(view["hypothesis_id"], view["candidate_id"]): view for view in views}
        for item in candidates:
            item["tracking"] = by_id.get((item["hypothesis_id"], item["candidate_id"]), {"association": "unmatched", "single_frame_published": True})
    record = {
        "header_stamp_ns": int(stamp_ns),
        "header_frame_id": header_frame_id if header_frame_id is not None else frame_id,
        "message_index": message_index if message_index is not None else frame_key,
        "coordinate_frame": result.get("coordinate_frame"),
        "algorithm_version": algorithm_version(),
        "config_version": config_version(detector_config, corridor_config, mount, flags, session_attachment),
        "session_attachment_is_conditional": True,
        "options": options,
        "status": result["status"],
        "own_path_selected": False,
        "conditional_per_hypothesis": True,
        "competing_corridors_are_not_one_path": True,
        "absence_of_candidates_is_not_clear": True,
        "candidate_sum_is_not_an_obstacle_count": True,
        "geometry_quality": {
            "status": result["status"],
            "assumptions": result.get("assumptions"),
            "hypotheses": [
                {
                    "hypothesis_id": item.get("hypothesis_id"),
                    "kind": item.get("kind"),
                    "not_an_own_path": True,
                    "geometry_applicable": item.get("geometry_applicable"),
                    "observed_s_min_m": item.get("observed_s_min_m"),
                    "observed_s_max_m": item.get("observed_s_max_m"),
                    "uncertainty": item.get("uncertainty"),
                    "support": item.get("candidate_counts"),
                }
                for item in result.get("hypotheses") or []
            ],
        },
        "candidates": candidates,
        "tracking": state,
        "timing_s": dict(result.get("timing_s") or {}),
        "corridor_source": result.get("corridor_source", "inferred"),
        "temporal_confirmation_required_for_single_frame": False,
        "state_reasons": result.get(
            "state_reasons",
            {
                "ground": "ground_unavailable" if result.get("geometry_status") != "candidate" else "ground_filter_available",
                "temporal_confirmation": "not_required_for_single_frame",
                "orientation": "unconfirmed" if not mount.get("applied") else "mount_applied",
            },
        ),
        "profile": result.get("profile"),
        "axis_assessment": result.get("axis_assessment"),
        "straight_applicable_for_decision": result.get("straight_applicable_for_decision"),
        "degraded_without_ground": result.get("degraded_without_ground"),
        "geometric_inside": result.get("geometric_inside"),
        "outside_configured_range_is_not_free": result.get("outside_configured_range_is_not_free", True),
    }
    record["timing_s"]["bag_read_not_included"] = True
    profile = result.get("profile") or {}
    ruled = decide(
        candidates,
        corridor_source=record["corridor_source"],
        profile_applicable=result.get("straight_applicable_for_decision"),
        ground_filter_available=bool(profile.get("ground_filter_available")),
        status=str(result.get("status") or ""),
        object_size_m=list(detector_config["object_size_m"]),
        large_min_unique_xyz=int(detector_config["large_min_unique_xyz"]["value"]),
        orientation_unconfirmed=not bool(mount.get("applied")),
        thin_length_m=float(detector_config.get("thin_length_m", 0.80)),
        thin_min_points=int(detector_config.get("thin_points", 10)),
        continuation_from_s_m=None if profile.get("far_assumption_from_s_m") is None else float(profile["far_assumption_from_s_m"]),
        thin_floor_h_m=None if profile.get("rail_head_h_m") is None else float(profile["rail_head_h_m"]),
        thin_clearance_m=float(detector_config.get("thin_height_above_floor_m", 1.0)),
    )
    record["decision"] = ruled["decision"]
    record["conditional_intrusions"] = ruled["conditional_intrusions"]
    record["candidate_decisions"] = ruled["candidate_decisions"]
    record["intrusion_rule_present"] = True
    record["intrusion_rule_id"] = ruled["rule_id"]
    record["intrusion_rule_reason"] = ruled["reason"]
    record["intrusion_rule_parameters"] = ruled["parameters"]
    record["orientation_doubt_remains"] = ruled["orientation_doubt_remains"]
    record["not_a_proven_danger_to_the_train"] = True
    record["candidates_not_modified_by_intrusion_rule"] = True
    record["timing_s"]["intrusion_rule"] = ruled["timing_s"]
    return record
