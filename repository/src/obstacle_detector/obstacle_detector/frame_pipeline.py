"""One PointCloud2: geometry, corridors, candidates.

This path does not read earlier reports, insertion tables, or saved hypotheses.
There is no state carried from one frame to the next.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml

from obstacle_detector.corridor import (
    build_path_corridor,
    classify_sensor_points,
    geometry_working,
    load_config,
    match_session_attachment,
    mount_from_config,
)
from obstacle_detector.detector import _spatial_grid, detect, line_samples, load_detector_config
from obstacle_detector.local_ground import (
    CONFIG,
    estimate_ground,
    height_above_local,
    lateral_support_mask,
    section_arrays,
    structure_selection,
)
from obstacle_detector.pair_geometry import PAIR_CONFIG, measure_candidate_pairs, rank_pair_hypotheses
from obstacle_detector.repeat_groups import group_exact_xyz
from obstacle_detector.straight_corridor import run_configured_straight

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def config_dir() -> Path:
    """Source checkout keeps config next to the package. A colcon install uses share."""
    sibling = PACKAGE_ROOT / "config"
    if (sibling / "detector.yaml").is_file():
        return sibling
    installed = PACKAGE_ROOT.parents[2] / "share" / "obstacle_detector" / "config"
    if (installed / "detector.yaml").is_file():
        return installed
    try:
        from ament_index_python.packages import get_package_share_directory

        shared = Path(get_package_share_directory("obstacle_detector")) / "config"
        if (shared / "detector.yaml").is_file():
            return shared
    except Exception:
        pass
    return sibling


def algorithm_version() -> str:
    import hashlib

    digest = hashlib.sha256()
    for relative in (
        "obstacle_detector/detector.py",
        "obstacle_detector/corridor.py",
        "obstacle_detector/local_ground.py",
        "obstacle_detector/pair_geometry.py",
        "obstacle_detector/frame_pipeline.py",
        "obstacle_detector/repeat_groups.py",
        "obstacle_detector/tracking.py",
        "obstacle_detector/working.py",
        "obstacle_detector/intrusion_rule.py",
        "obstacle_detector/straight_corridor.py",
        "config/detector.yaml",
        "config/intrusion_rule.yaml",
        "config/corridor.yaml",
        "config/batch.yaml",
    ):
        digest.update(relative.encode())
        path = (config_dir() / relative.split("/", 1)[1]) if relative.startswith("config/") else (PACKAGE_ROOT / relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def load_batch_config(path: Path | None = None) -> dict:
    raw = yaml.safe_load((path or (PACKAGE_ROOT / "config" / "batch.yaml")).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "variants" not in raw:
        raise ValueError("batch config must name variants")
    return raw


def variant_flags(batch_config: dict, name: str) -> dict:
    block = (batch_config.get("variants") or {}).get(name)
    if not isinstance(block, dict):
        raise SystemExit(f"вариант {name} не задан в конфигурации")
    connectivity = block.get("connectivity")
    if connectivity not in ("cells", "distance"):
        raise SystemExit(f"вариант {name}: неизвестная связность {connectivity}")
    return {
        "name": name,
        "connectivity": connectivity,
        "local_protrusions_enabled": bool(block.get("local_protrusions")),
    }


def build_frame_geometry(groups, mount=None) -> dict:
    """Ground and pairs in the same working frame as corridor membership.

    An unconfirmed mount leaves the sensor prior. A confirmed mount is applied
    before s=-Y, l=X, h=Z, matching geometry_working.
    """
    started = time.perf_counter()
    s, l, h, frame_name = geometry_working(groups["x"], groups["y"], groups["z"], mount or {"confirmed": False})
    region = (s >= CONFIG["s_min_m"]) & (s <= CONFIG["s_max_m"])
    ground = estimate_ground(s[region], l[region], h[region], CONFIG)
    surface_started = time.perf_counter()
    dh = height_above_local(s, l, h, ground)
    support = lateral_support_mask(s, l, ground)
    surface_s = time.perf_counter() - surface_started
    sections = [section_arrays(s[region], l[region], h[region], dh[region], center, CONFIG) for center in CONFIG["section_s_m"]]
    selected = structure_selection(sections, CONFIG, ground)
    fragments = selected["fragments"]
    ranked = rank_pair_hypotheses(measure_candidate_pairs(fragments, ground, PAIR_CONFIG), fragments, ground, PAIR_CONFIG)
    return {
        "s": s,
        "l": l,
        "h": h,
        "ground": ground,
        "fragments": fragments,
        "ranked": ranked,
        "coordinate_frame": frame_name,
        "dh": dh,
        "lateral_support": support,
        "surface_s": surface_s,
        "elapsed_s": time.perf_counter() - started,
    }


def _first_index(groups) -> np.ndarray:
    if "first_index" in groups:
        return np.asarray(groups["first_index"], dtype=np.int64).reshape(-1)
    return np.array([int(np.asarray(members)[0]) for members in groups["indices"]], dtype=np.int64)


def _observed_span(corridor) -> tuple[float, float] | None:
    observed = [
        float(vertex["s_m"])
        for vertex in corridor["vertices"]
        if vertex["role"] == "observed" and vertex.get("h_ref_m") is not None
    ]
    if not observed:
        return None
    return min(observed), max(observed)


def _support_range(ground) -> tuple[float | None, float | None]:
    bins = ground.get("local_bins") or []
    lows = [float(item["s_lo"]) for item in bins if item.get("s_lo") is not None]
    highs = [float(item["s_hi"]) for item in bins if item.get("s_hi") is not None]
    if not lows or not highs:
        return None, None
    return min(lows), max(highs)


def _compact_candidate(item) -> dict:
    return {
        "candidate_id": item["candidate_id"],
        "hypothesis_id": item["hypothesis_id"],
        "kind": item["kind"],
        "branch": item.get("branch", "volume_or_suspended"),
        "unique_xyz": int(item["unique_xyz"]),
        "observed_extent_m": item["observed_extent_m"],
        "observed_s_min_m": item.get("observed_s_min_m"),
        "observed_h_min_m": item.get("observed_h_min_m"),
        "range_from_lidar_m": item["range_from_lidar_m"],
        "conditional": True,
        "doubt_reasons": list(item.get("doubt_reasons") or []),
        "not_a_confirmed_obstacle": True,
    }


def _candidate_counts(detected) -> dict:
    return {
        "volume_large": len(detected["candidates_large"]),
        "volume_rare": len(detected["candidates_rare"]),
        "protrusion_large": len(detected["candidates_protrusion_large"]),
        "protrusion_rare": len(detected["candidates_protrusion_rare"]),
    }


def _observation_gap() -> dict:
    return {
        "candidate_meaning": "diagnostic_fragment",
        "conditional_intrusions": [],
        "intrusion_rule_present": False,
        "intrusion_rule_gap": "В ядре нет самостоятельного правила, которое признаёт наблюдение вторжением. inside не является препятствием. Кандидат не переименовывается во вторжение.",
        "temporal_confirmation_required_for_single_frame": False,
        "decision": "insufficient_evidence",
    }


def _straight_frame(groups, geometry, base, spec, detector_config, flags, detailed: bool) -> dict:
    """Use the ground already estimated for this frame. Do not estimate it again."""
    started = time.perf_counter()
    outcome = run_configured_straight(
        groups,
        geometry,
        spec,
        detector_config,
        flags,
        precomputed_dh=geometry.get("dh"),
        precomputed_support=geometry.get("lateral_support"),
    )
    base["corridor_source"] = "configured_straight"
    base["profile"] = outcome["profile"]
    base["axis_assessment"] = outcome["axis_assessment"]
    base["geometric_inside"] = outcome["geometric_inside"]
    base["outside_configured_range"] = outcome["outside_configured_range"]
    base["outside_configured_range_is_not_free"] = True
    base["volume_justification"] = outcome["profile"]["volume_justification"]
    base["ground_filter_available"] = outcome["profile"]["ground_filter_available"]
    base["degraded_without_ground"] = outcome["degraded_without_ground"]
    base["skipped_branches"] = outcome["skipped_branches"]
    base["straight_applicable_for_decision"] = outcome["applicable_for_decision"]
    base["timing_s"]["surface"] = float(geometry.get("surface_s") or 0.0)
    base["timing_s"]["surface_evaluations"] = 1
    base["timing_s"]["corridor"] = float(outcome["timing_s"]["axis"])
    base["timing_s"]["filters"] = float(outcome["timing_s"]["filters"])
    base["timing_s"]["connectivity"] = float(outcome["timing_s"]["connectivity"])
    base.update(_observation_gap())
    if outcome["degraded_without_ground"]:
        base["state_reasons"] = {
            "ground": "ground_unavailable",
            "temporal_confirmation": "not_required_for_single_frame",
            "orientation": "unconfirmed",
        }
    else:
        base["state_reasons"] = {
            "ground": "ground_filter_available",
            "temporal_confirmation": "not_required_for_single_frame",
            "orientation": "unconfirmed",
        }
    if not outcome["applicable_for_decision"]:
        base["status"] = "configured_straight_inapplicable"
        base["diagnostic_observations"] = {"axis_assessment": outcome["axis_assessment"]}
        base["timing_s"]["result"] = time.perf_counter() - started
        base["timing_s"]["frame"] = base["timing_s"]["geometry"] + base["timing_s"]["result"]
        return base
    detected = outcome["detected"]
    base["surface_mode"] = detected["surface_mode"]
    listed = (
        detected["candidates_large"]
        + detected["candidates_rare"]
        + detected["candidates_protrusion_large"]
        + detected["candidates_protrusion_rare"]
    )
    for item in listed:
        compact = _compact_candidate(item)
        compact["profile_assumptions"] = [
            "prior s=-Y l=X h=Z не подтверждён",
            "нижняя граница 0 м над головкой рельса, не над оценённым основанием",
            "ориентация углов неизвестна",
        ]
        compact["range_points"] = "minimum range of points in this candidate"
        if outcome["degraded_without_ground"]:
            compact["doubt_reasons"] = list(compact["doubt_reasons"]) + ["основание недоступно, фильтр поверхности не выполнен"]
            compact["degraded_without_ground"] = True
        if detailed:
            rows = np.asarray(item["rows"], dtype=np.int64)
            compact["source_indices"] = [int(value) for value in np.asarray(item["source_indices"]).tolist()]
            compact["sensor_position_m"] = {
                "x": float(np.median(np.asarray(groups["x"])[rows])),
                "y": float(np.median(np.asarray(groups["y"])[rows])),
                "z": float(np.median(np.asarray(groups["z"])[rows])),
            }
        base["candidates"].append(compact)
    base["candidate_quality"] = {
        "large": len(detected["candidates_large"]),
        "rare": len(detected["candidates_rare"]),
        "not_an_obstacle_count": True,
        "not_raised_without_ground": outcome["degraded_without_ground"],
    }
    base["status"] = "ok"
    base["timing_s"]["detector"] = float(detected["timing_s"]["detector"])
    base["timing_s"]["result"] = time.perf_counter() - started
    base["timing_s"]["frame"] = base["timing_s"]["geometry"] + base["timing_s"]["result"]
    return base


def process_grouped_cloud(groups, corridor_config, detector_config, mount, flags, keep_plot: bool = False, session_attachment=None, local_context_slh=None, detailed: bool = False, straight_profile=None) -> dict:
    """Detect on one already grouped cloud. Insertion labels are not an argument."""
    started = time.perf_counter()
    geometry = build_frame_geometry(groups, mount)
    ground = geometry["ground"]
    support_min, support_max = _support_range(ground)
    assumptions = [
        "Рабочий prior s=-Y, l=X, h=Z — допущение, не калибровка установки.",
        "Отсутствие кандидатов не означает, что коридор свободен.",
        "Сумма кандидатов по гипотезам не является числом физических препятствий.",
        "Инженерная оценка гипотезы не является вероятностью.",
    ]
    lower = detector_config["lower_boundary_above_reference_m"]
    if lower.get("status") != "stated":
        assumptions.append("Нижняя граница профиля — допущение, не калибровка.")
    base = {
        "geometry_status": ground.get("status"),
        "geometry_reason": ground.get("reason"),
        "support_s_min_m": support_min,
        "support_s_max_m": support_max,
        "fitting_residual_p95_m": (ground.get("fitting_residual") or {}).get("p95_m"),
        "fitting_residual_is_not_surface_accuracy": True,
        "hypotheses": [],
        "candidates": [],
        "assumptions": assumptions,
        "absence_of_candidates_is_not_clear": True,
        "obstacle_reported": None,
        "candidate_sum_is_not_an_obstacle_count": True,
        "coordinate_frame": geometry["coordinate_frame"],
        "variant": flags["name"],
        "timing_s": {"geometry": float(geometry["elapsed_s"]), "corridor_match": 0.0, "membership": 0.0, "detector": 0.0},
        "error": None,
        "_plot": None,
        "corridor_source": "configured_straight" if straight_profile is not None else "inferred",
    }
    if straight_profile is not None:
        return _straight_frame(groups, geometry, base, straight_profile, detector_config, flags, detailed)
    if ground.get("status") != "candidate":
        base["status"] = "insufficient_geometry"
        base["timing_s"]["frame"] = time.perf_counter() - started
        return base
    hypotheses = list(geometry["ranked"]["hypotheses"])
    if not hypotheses:
        base["status"] = "attachment_conflict" if session_attachment and session_attachment.get("enabled") else "no_hypotheses"
        base["attachment_match"] = match_session_attachment([], session_attachment)
        base["timing_s"]["frame"] = time.perf_counter() - started
        return base
    match_started = time.perf_counter()
    attachment_match = match_session_attachment(hypotheses, session_attachment)
    base["timing_s"]["corridor_match"] = time.perf_counter() - match_started
    base["attachment_match"] = attachment_match
    allowed = None
    if session_attachment is not None and session_attachment.get("enabled"):
        allowed = {item["hypothesis_id"] for item in attachment_match["compatible"]}
    indices = _first_index(groups)
    ranges = np.sqrt(groups["x"] ** 2 + groups["y"] ** 2 + groups["z"] ** 2)
    shared_grid = None
    if flags.get("local_protrusions_enabled") and local_context_slh is None:
        finite = np.flatnonzero(np.isfinite(geometry["s"]) & np.isfinite(geometry["l"]) & np.isfinite(geometry["h"]))
        shared_grid = _spatial_grid(geometry["s"], geometry["l"], finite, float(detector_config["local_anchor_step_m"]))
    base["timing_s"]["surface"] = float(geometry.get("surface_s") or 0.0)
    base["timing_s"]["surface_evaluations"] = 1
    dh_frame = geometry["dh"]
    support_frame = geometry["lateral_support"]
    applicable = 0
    for hypothesis in hypotheses:
        if allowed is not None and hypothesis["hypothesis_id"] not in allowed:
            base["hypotheses"].append(
                {
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "kind": hypothesis.get("kind"),
                    "not_an_own_path": True,
                    "attachment_compatible": False,
                    "candidate_counts": {"volume_large": 0, "volume_rare": 0, "protrusion_large": 0, "protrusion_rare": 0},
                    "detected": False,
                }
            )
            continue
        corridor = build_path_corridor(
            hypothesis,
            ground,
            corridor_config,
            mount,
            attachment=session_attachment if allowed is not None else None,
            lower_boundary=lower,
        )
        span = _observed_span(corridor)
        summary = {
            "hypothesis_id": hypothesis["hypothesis_id"],
            "kind": hypothesis.get("kind"),
            "engineering_score": hypothesis.get("engineering_score"),
            "score_is_probability": False,
            "not_an_own_path": True,
            "penalties": list(hypothesis.get("penalties") or []),
            "observed_s_min_m": None if span is None else span[0],
            "observed_s_max_m": None if span is None else span[1],
            "geometry_applicable": span is not None,
            "attachment_compatible": True if allowed is None else True,
            "candidate_counts": {"volume_large": 0, "volume_rare": 0, "protrusion_large": 0, "protrusion_rare": 0},
        }
        if span is None:
            summary["uncertainty"] = "нет наблюдаемого участка с опорой поверхности"
            base["hypotheses"].append(summary)
            continue
        applicable += 1
        membership_started = time.perf_counter()
        relation = classify_sensor_points(groups["x"], groups["y"], groups["z"], corridor, mount, corridor_config)["geometric_relation"]
        base["timing_s"]["membership"] += time.perf_counter() - membership_started
        detect_started = time.perf_counter()
        detected = detect(
            geometry["s"],
            geometry["l"],
            geometry["h"],
            dh_frame,
            support_frame,
            ranges,
            indices,
            relation,
            corridor,
            line_samples(hypothesis, geometry["fragments"]),
            (ground.get("fitting_residual") or {}).get("p95_m"),
            detector_config,
            connectivity=flags["connectivity"],
            local_protrusions_enabled=flags["local_protrusions_enabled"],
            local_context_slh=local_context_slh,
            spatial_grid=shared_grid,
        )
        base["timing_s"]["detector"] += time.perf_counter() - detect_started
        summary["candidate_counts"] = _candidate_counts(detected)
        summary["surface_mode"] = detected["surface_mode"]
        summary["uncertainty"] = list(detected["doubt_reasons"])
        listed = (
            detected["candidates_large"]
            + detected["candidates_rare"]
            + detected["candidates_protrusion_large"]
            + detected["candidates_protrusion_rare"]
        )
        for item in listed:
            compact = _compact_candidate(item)
            if detailed:
                rows = np.asarray(item["rows"], dtype=np.int64)
                compact["source_indices"] = [int(value) for value in np.asarray(item["source_indices"]).tolist()]
                compact["position_m"] = {
                    "s": float(np.median(geometry["s"][rows])),
                    "l": float(np.median(geometry["l"][rows])),
                    "h": float(np.median(geometry["h"][rows])),
                }
                compact["sensor_position_m"] = {
                    "x": float(np.median(np.asarray(groups["x"])[rows])),
                    "y": float(np.median(np.asarray(groups["y"])[rows])),
                    "z": float(np.median(np.asarray(groups["z"])[rows])),
                }
                compact["observed_extent_is_not_true_object_size"] = True
                compact["rare_fragment_is_not_a_wire"] = compact["kind"] == "rare"
            base["candidates"].append(compact)
        if keep_plot and listed:
            rows = np.concatenate([np.asarray(item["rows"], dtype=np.int64) for item in listed])
            current = {"hypothesis_id": hypothesis["hypothesis_id"], "count": int(rows.size), "s": geometry["s"][rows], "l": geometry["l"][rows]}
            previous = base.get("_plot_rows")
            if not previous or current["count"] > previous[0]["count"]:
                base["_plot_rows"] = [current]
        del detected
        base["hypotheses"].append(summary)
    if session_attachment is not None and session_attachment.get("enabled") and attachment_match["status"] == "none":
        base["status"] = "attachment_conflict"
    elif applicable == 0:
        base["status"] = "insufficient_geometry"
    else:
        base["status"] = "ok"
    base["timing_s"]["frame"] = time.perf_counter() - started
    return base


def process_arrays(x, y, z, intensity, ring, timestamp, valid, corridor_config, detector_config, mount, flags, keep_plot: bool = False) -> dict:
    groups = group_exact_xyz(x, y, z, intensity, ring, timestamp, valid)
    return process_grouped_cloud(groups, corridor_config, detector_config, mount, flags, keep_plot=keep_plot)


def load_runtime(config_directory: Path | None = None):
    root = config_directory if config_directory is not None else config_dir()
    corridor_config = load_config(root / "corridor.yaml")
    detector_config = load_detector_config(root / "detector.yaml")
    batch_config = load_batch_config(root / "batch.yaml")
    return corridor_config, detector_config, batch_config, mount_from_config(corridor_config)
