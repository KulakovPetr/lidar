"""Limited A/B on named frames. The detector does not receive event labels."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

from obstacle_detector.bag_io import decode_message
from obstacle_detector.corridor import build_path_corridor, classify_sensor_points
from obstacle_detector.detector import detect, line_samples
from obstacle_detector.diagnostic_corridor import run_diagnostic
from obstacle_detector.frame_pipeline import (
    _first_index,
    _observed_span,
    algorithm_version,
    build_frame_geometry,
    load_runtime,
    process_grouped_cloud,
    variant_flags,
)
from obstacle_detector.local_ground import height_above_local, lateral_support_mask
from obstacle_detector.offline_run import load_mount_file, reader_version
from obstacle_detector.repeat_groups import group_exact_xyz, valid_mask


E01_FRAMES = [194, 203, 209, 212, 216, 220, 225, 230]
E01_TRACE = [203, 212, 220]
E01_EARLY = [225, 230]
E03_FRAMES = [611, 616, 620, 630, 636]
QUIET_FRAMES = list(range(240, 261))
PLATFORM_FRAMES = [0, 172]
CURVE_FRAMES = [0, 126]


def load_spec(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    longi = raw["longitudinal"]
    profile = raw["profile"]
    prior = raw["prior"]
    return {
        "s_min_m": float(longi["s_min_m"]),
        "s_max_m": float(longi["s_max_m"]),
        "far_from_s_m": float(longi["far_assumption_from_s_m"]),
        "width_m": float(profile["width_m"]),
        "height_m": float(profile["height_m"]),
        "center_l_m": float(profile["center_l_m"]),
        "lower_above_rail_m": float(profile["lower_boundary_above_rail_head_m"]),
        "rail_head_h_m": float(prior["rail_head_h_if_z_up_and_angles_zero_m"]),
        "raw": raw,
    }


def read_cloud(connection, ids, index):
    payload, bag_ns = connection.execute(
        "SELECT data, timestamp FROM messages WHERE id = ?", (ids[index],)
    ).fetchone()
    msg = deserialize_message(payload, PointCloud2)
    decoded = decode_message(msg)
    return decoded, int(bag_ns)


def groups_from(decoded):
    valid = valid_mask(decoded["x"], decoded["y"], decoded["z"])
    return group_exact_xyz(
        decoded["x"], decoded["y"], decoded["z"], decoded["intensity"], decoded["ring"], decoded["timestamp"], valid
    )


def locate_face(s, l, h, gate, min_l_span, min_h_span):
    """Dense thin slab inside a broad gate. Not a detector cluster."""
    chosen = np.flatnonzero(gate)
    if chosen.size < 40:
        return None
    bins = np.floor(s[chosen] / 0.25).astype(np.int32)
    best = None
    for value in np.unique(bins):
        member = chosen[bins == value]
        if member.size < 40:
            continue
        l_span = float(l[member].max() - l[member].min())
        h_span = float(h[member].max() - h[member].min())
        if l_span < min_l_span or h_span < min_h_span:
            continue
        if best is None or member.size > best[0]:
            best = (int(member.size), int(value))
    if best is None:
        return None
    s_lo = best[1] * 0.25
    slab = gate & (s >= s_lo) & (s < s_lo + 0.40)
    if int(slab.sum()) < 40:
        return None
    if float(l[slab].max() - l[slab].min()) > 3.6:
        return {"separated": False, "reason": "slab spans the tunnel, not a separable face", "s_lo_m": s_lo}
    h_cut = float(np.percentile(h[slab], 12))
    core = slab & (h >= h_cut + 0.05)
    if int(core.sum()) < 30 or float(h[core].max() - h[core].min()) < 0.45:
        return {"separated": False, "reason": "face height is not separated from the floor band", "s_lo_m": s_lo}
    l_lo, l_hi = np.percentile(l[core], [4, 96])
    h_lo, h_hi = np.percentile(h[core], [4, 96])
    confident = core & (l >= l_lo) & (l <= l_hi) & (h >= h_lo) & (h <= h_hi)
    near = (
        (np.abs(s - (s_lo + 0.2)) <= 0.55)
        & (l >= l_lo - 0.20)
        & (l <= l_hi + 0.20)
        & (h >= h_lo - 0.20)
        & (h <= h_hi + 0.20)
    )
    uncertain = near & ~confident & gate
    return {
        "separated": True,
        "s_lo_m": float(s_lo),
        "s_hi_m": float(s_lo + 0.40),
        "l_m": [float(l_lo), float(l_hi)],
        "h_m": [float(h_lo), float(h_hi)],
        "confident": confident,
        "uncertain": uncertain,
    }


def reference_range(x, y, z, mask):
    if not np.any(mask):
        return None
    ranges = np.sqrt(x[mask] ** 2 + y[mask] ** 2 + z[mask] ** 2)
    return {
        "source": "minimum range of the confident visible face",
        "min_m": float(ranges.min()),
        "max_m": float(ranges.max()),
        "uncertainty_m": float(ranges.max() - ranges.min()),
        "uncertainty_is_face_thickness_not_object_depth": True,
        "coarse_window_min_is_not_this_value": True,
    }


def row_lookup(groups):
    mapping = {}
    for row, members in enumerate(groups["indices"]):
        for index in np.asarray(members).tolist():
            mapping[int(index)] = row
    return mapping


def section_at(corridor, s_query):
    vertices = list(corridor.get("vertices") or [])
    if len(vertices) < 2:
        return {"available": False, "reason": "fewer than two corridor vertices"}
    for start, stop in zip(vertices, vertices[1:]):
        if start["s_m"] - 1e-6 <= s_query <= stop["s_m"] + 1e-6:
            span = stop["s_m"] - start["s_m"]
            t = 0.0 if span == 0 else (s_query - start["s_m"]) / span
            l_axis = float(start["l_m"] + t * (stop["l_m"] - start["l_m"]))
            h0, h1 = start.get("h_ref_m"), stop.get("h_ref_m")
            if h0 is None or h1 is None:
                href = None
            else:
                href = float(h0 + t * (h1 - h0))
            lower = corridor.get("lower_boundary_m")
            floor = None if href is None or lower is None else href + float(lower)
            top = None if floor is None else floor + float(corridor["height_m"])
            half = float(corridor["half_width_m"]) + float(corridor["extra_lateral_m"])
            return {
                "available": True,
                "s_m": s_query,
                "axis_l_m": l_axis,
                "h_ref_m": href,
                "lower_h_m": floor,
                "upper_h_m": top,
                "l_min_m": l_axis - half,
                "l_max_m": l_axis + half,
                "segment_role": "extrapolated" if "extrapolated" in {start["role"], stop["role"]} else start["role"],
                "observed_s_m": list(corridor.get("observed_s_m") or []),
                "s_overlap_is_not_inside_gauge": True,
            }
    return {"available": False, "reason": "object range is outside the corridor vertices"}


def count_mask(rows_of_interest, mask):
    if not rows_of_interest.size:
        return 0
    return int(np.count_nonzero(mask[rows_of_interest]))


def trace_hypotheses(groups, geometry, corridor_config, detector_config, mount, flags, confident_idx, s_face):
    ground = geometry["ground"]
    if ground.get("status") != "candidate":
        return {
            "early_return": "insufficient_geometry",
            "missing": "ground",
            "ground_status": ground.get("status"),
            "ground_reason": ground.get("reason"),
            "hypotheses": [],
        }
    hypotheses = list(geometry["ranked"]["hypotheses"])
    fragments = geometry.get("fragments") or []
    if not hypotheses:
        return {
            "early_return": "no_hypotheses",
            "missing": "lines" if not fragments else "pair",
            "ground_status": ground.get("status"),
            "fragment_count": len(fragments),
            "hypothesis_count": 0,
            "hypotheses": [],
        }
    lookup = row_lookup(groups)
    rows = np.array([lookup[int(index)] for index in confident_idx if int(index) in lookup], dtype=np.int64)
    rows = np.unique(rows) if rows.size else rows
    indices = _first_index(groups)
    ranges = np.sqrt(groups["x"] ** 2 + groups["y"] ** 2 + groups["z"] ** 2)
    listed = []
    applicable = 0
    for hypothesis in hypotheses:
        corridor = build_path_corridor(hypothesis, ground, corridor_config, mount, lower_boundary=detector_config["lower_boundary_above_reference_m"])
        span = _observed_span(corridor)
        section = section_at(corridor, s_face) if s_face is not None else {"available": False}
        item = {
            "hypothesis_id": hypothesis.get("hypothesis_id"),
            "kind": hypothesis.get("kind"),
            "observed_s_min_m": None if span is None else span[0],
            "observed_s_max_m": None if span is None else span[1],
            "geometry_applicable": span is not None,
            "section_at_face": section,
            "axis_l_m": [round(float(vertex["l_m"]), 3) for vertex in (corridor.get("vertices") or [])],
        }
        if span is None:
            item["early_note"] = "нет наблюдаемого участка с опорой поверхности"
            item["selected_points"] = int(rows.size)
            listed.append(item)
            continue
        applicable += 1
        relation = classify_sensor_points(groups["x"], groups["y"], groups["z"], corridor, mount, corridor_config)["geometric_relation"]
        detected = detect(
            geometry["s"],
            geometry["l"],
            geometry["h"],
            height_above_local(geometry["s"], geometry["l"], geometry["h"], ground),
            lateral_support_mask(geometry["s"], geometry["l"], ground),
            ranges,
            indices,
            relation,
            corridor,
            line_samples(hypothesis, fragments),
            (ground.get("fitting_residual") or {}).get("p95_m"),
            detector_config,
            connectivity=flags["connectivity"],
            local_protrusions_enabled=flags["local_protrusions_enabled"],
        )
        masks = detected["masks"]
        names = np.asarray(relation).astype(str)
        outside_support = np.zeros(geometry["s"].shape, dtype=bool)
        if rows.size:
            outside_support[rows] = (geometry["s"][rows] < span[0]) | (geometry["s"][rows] > span[1])
        in_candidate_rows = np.zeros(geometry["s"].shape, dtype=bool)
        candidates = []
        for candidate in (
            detected["candidates_large"]
            + detected["candidates_rare"]
            + detected["candidates_protrusion_large"]
            + detected["candidates_protrusion_rare"]
        ):
            member = np.asarray(candidate["rows"], dtype=np.int64)
            in_candidate_rows[member] = True
            if rows.size and np.count_nonzero(np.isin(member, rows)):
                candidates.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "branch": candidate["branch"],
                        "kind": candidate["kind"],
                        "roi_rows": int(np.count_nonzero(np.isin(member, rows))),
                        "candidate_rows": int(member.size),
                        "range_from_lidar_m": candidate["range_from_lidar_m"],
                    }
                )
        item["selected_points"] = int(rows.size)
        item["inside"] = count_mask(rows, names == "inside")
        item["outside"] = count_mask(rows, names == "outside")
        item["uncertain"] = count_mask(rows, names == "uncertain")
        item["outside_longitudinal_support"] = int(np.count_nonzero(outside_support))
        item["excluded_surface"] = count_mask(rows, masks["thin_surface"])
        item["excluded_path_line"] = count_mask(rows, masks["path_structure"])
        item["ambiguous_layer"] = count_mask(rows, masks["ambiguous_surface"])
        item["volume_pool"] = count_mask(rows, masks["object_pool"])
        item["local_branch_input"] = count_mask(rows, masks["ambiguous_surface"])
        item["in_candidate"] = count_mask(rows, in_candidate_rows)
        item["candidates_touching_roi"] = candidates
        item["surface_mode"] = detected["surface_mode"]
        listed.append(item)
    status = "ok" if applicable else "insufficient_geometry"
    missing = None if applicable else "applicable_corridor"
    return {
        "early_return": None if applicable else status,
        "missing": missing,
        "ground_status": ground.get("status"),
        "fragment_count": len(fragments),
        "hypothesis_count": len(hypotheses),
        "hypotheses": listed,
    }


def candidate_brief(detected, lookup, confident_idx):
    rows = np.array([lookup[int(index)] for index in confident_idx if int(index) in lookup], dtype=np.int64)
    rows = np.unique(rows) if rows.size else rows
    best = None
    extra = 0
    for candidate in (
        detected["candidates_large"]
        + detected["candidates_rare"]
        + detected["candidates_protrusion_large"]
        + detected["candidates_protrusion_rare"]
    ):
        extra += 1
        member = np.asarray(candidate["rows"], dtype=np.int64)
        if not rows.size or not member.size:
            hit = 0
        else:
            hit = int(np.count_nonzero(np.isin(member, rows)))
        share = hit / float(member.size)
        record = {
            "candidate_id": candidate["candidate_id"],
            "branch": candidate["branch"],
            "roi_rows": hit,
            "candidate_rows": int(member.size),
            "roi_share": share,
            "surrounding_share": 1.0 - share,
            "range_from_lidar_m": candidate["range_from_lidar_m"],
            "localizing": hit >= 4 and share >= 0.5,
        }
        if best is None or (record["localizing"], hit) > (best["localizing"], best["roi_rows"]):
            best = record
    return {
        "candidate_count": extra,
        "best": best,
        "volume_pool_roi_rows": count_mask(rows, detected["masks"]["object_pool"]) if rows.size else 0,
        "inside_roi_rows": count_mask(rows, detected["masks"]["inside"]) if rows.size else 0,
    }


def save_view(path, s, l, h, confident, uncertain):
    rng = np.random.default_rng(0)
    other = ~(confident | uncertain)
    parts = []
    labels = []
    for mask, label, limit in ((other, 0, 6000), (uncertain, 1, 4000), (confident, 2, 8000)):
        idx = np.flatnonzero(mask)
        if idx.size > limit:
            idx = rng.choice(idx, limit, replace=False)
        parts.append(idx)
        labels.append(np.full(idx.size, label, dtype=np.int8))
    if not parts:
        return
    chosen = np.concatenate(parts)
    np.savez_compressed(
        path,
        s=s[chosen].astype(np.float32),
        l=l[chosen].astype(np.float32),
        h=h[chosen].astype(np.float32),
        label=np.concatenate(labels),
    )


def analyze_frame(connection, ids, index, search, mount, corridor_config, detector_config, flags, spec, trace, out_views):
    decoded, bag_ns = read_cloud(connection, ids, index)
    groups = groups_from(decoded)
    started = time.perf_counter()
    geometry = build_frame_geometry(groups, mount)
    geometry_s = time.perf_counter() - started
    s, l, h = geometry["s"], geometry["l"], geometry["h"]
    # Group arrays match groups, not the original cloud. Search on group coordinates.
    gate = search(s, l, h)
    face = locate_face(s, l, h, gate, 1.0, 0.7)
    separated = bool(face and face.get("separated"))
    confident = face["confident"] if separated else np.zeros(s.shape, dtype=bool)
    uncertain = face["uncertain"] if separated else np.zeros(s.shape, dtype=bool)
    x = np.asarray(groups["x"], dtype=np.float64)
    y = np.asarray(groups["y"], dtype=np.float64)
    z = np.asarray(groups["z"], dtype=np.float64)
    # Map confident group rows back to one source index for the range of those groups.
    lookup = row_lookup(groups)
    confident_idx = []
    for row in np.flatnonzero(confident).tolist():
        members = np.asarray(groups["indices"][row]).tolist()
        if members:
            confident_idx.append(int(members[0]))
    result = {
        "index": index,
        "header_stamp_ns": int(decoded["stamp_ns"]),
        "bag_record_time_ns": bag_ns,
        "absent_fields": list(decoded["absent_fields"]),
        "ring_observed": bool(groups["ring_observed"]),
        "timestamp_observed": bool(groups["timestamp_observed"]),
        "face": {key: value for key, value in (face or {"separated": False}).items() if key not in {"confident", "uncertain"}},
        "roi_points": int(np.count_nonzero(confident | uncertain)),
        "confident_points": int(np.count_nonzero(confident)),
        "uncertain_points": int(np.count_nonzero(uncertain)),
        "reference_range": reference_range(x, y, z, confident) if separated else None,
        "geometry_s": geometry_s,
    }
    if separated:
        save_view(out_views / f"frame_{index:04d}.npz", s, l, h, confident, uncertain)
    if trace and separated:
        result["trace"] = trace_hypotheses(
            groups, geometry, corridor_config, detector_config, mount, flags, confident_idx, face["s_lo_m"] + 0.20
        )
    elif trace:
        result["trace"] = trace_hypotheses(
            groups, geometry, corridor_config, detector_config, mount, flags, [], None
        )
    a_started = time.perf_counter()
    official = process_grouped_cloud(groups, corridor_config, detector_config, mount, flags, detailed=True)
    result["a_s"] = time.perf_counter() - a_started
    result["a_status"] = official["status"]
    listed = official.get("candidates") or []
    result["a_candidate_count"] = len(listed)
    conf_set = set(confident_idx)
    best = None
    for candidate in listed:
        sources = {int(value) for value in candidate.get("source_indices") or []}
        hit = len(sources & conf_set)
        share = hit / float(len(sources)) if sources else 0.0
        record = {
            "candidate_id": candidate.get("candidate_id"),
            "hypothesis_id": candidate.get("hypothesis_id"),
            "branch": candidate.get("branch"),
            "roi_rows": hit,
            "candidate_rows": len(sources),
            "roi_share": share,
            "surrounding_share": 1.0 - share,
            "range_from_lidar_m": candidate.get("range_from_lidar_m"),
            "localizing": hit >= 4 and share >= 0.5,
        }
        if best is None or (record["localizing"], record["roi_rows"]) > (best["localizing"], best["roi_rows"]):
            best = record
    result["a"] = {"best": best, "status": official["status"]}
    result["a_geometry_status"] = official.get("geometry_status")
    result["a_geometry_reason"] = official.get("geometry_reason")
    result["a_hypothesis_count"] = len(official.get("hypotheses") or [])
    b_started = time.perf_counter()
    diagnostic = run_diagnostic(groups, geometry, spec, detector_config, flags)
    result["b_s"] = time.perf_counter() - b_started
    result["b"] = candidate_brief(diagnostic["detected"], lookup, confident_idx if separated else [])
    result["b"]["ground_status"] = diagnostic["ground_status"]
    result["b"]["ground_used"] = diagnostic["ground_used"]
    result["b"]["skipped_branches"] = diagnostic["skipped_branches"]
    result["b"]["surface_mode"] = diagnostic["surface_mode"]
    result["b"]["inside_volume_groups"] = diagnostic["inside_count"]
    result["b"]["equivalent_to_supported_detection"] = False
    # Mode A candidate touch count from the trace when present; otherwise a short A pass.
    if "trace" in result and result["trace"].get("hypotheses"):
        touching = 0
        best = None
        for item in result["trace"]["hypotheses"]:
            for candidate in item.get("candidates_touching_roi") or []:
                touching += 1
                share = candidate["roi_rows"] / float(candidate["candidate_rows"]) if candidate["candidate_rows"] else 0.0
                record = dict(candidate)
                record["roi_share"] = share
                record["localizing"] = candidate["roi_rows"] >= 4 and share >= 0.5
                if best is None or (record["localizing"], record["roi_rows"]) > (best["localizing"], best["roi_rows"]):
                    best = record
        result["a_from_trace"] = {"touching_candidates": touching, "best": best}
        axes = []
        for item in result["trace"]["hypotheses"]:
            axes.extend(item.get("axis_l_m") or [])
        if axes and (max(axes) - min(axes) > 0.40 or max(abs(value) for value in axes) > 0.50):
            result["straight_corridor_ambiguous"] = True
            result["b"]["straight_continuation_contradicted"] = True
            result["b"]["ambiguity_reason"] = "ось рельсовой гипотезы не совпадает с прямым l=0; коридор не расширялся"
        elif not axes:
            result["straight_corridor_ambiguous"] = None
            result["b"]["ambiguity_reason"] = "рельсовая ось на кадре не измерена; прямое продолжение остаётся допущением"
    return result


def gate_e01(s, l, h):
    return (s > 0.4) & (s < 40.0) & (np.abs(l) < 1.7) & (h > -1.7) & (h < 1.4)


def gate_e03(s, l, h):
    return (s > 1.0) & (s < 30.0) & (l > -3.0) & (l < -0.4) & (h > -1.7) & (h < 2.2)


def gate_none(s, l, h):
    return np.zeros(s.shape, dtype=bool)


def open_bag(path: Path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    ids = [int(row[0]) for row in connection.execute("SELECT id FROM messages ORDER BY id")]
    return connection, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mount", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--synthetic-bag", type=Path, required=True)
    parser.add_argument("--platform-bag", type=Path, required=True)
    parser.add_argument("--curve-bag", type=Path, required=True)
    parser.add_argument("--smoke-bag", type=Path, required=True)
    args = parser.parse_args()
    spec = load_spec(args.config)
    corridor_config, mount = load_mount_file(args.mount)
    _packaged, detector_config, batch_config, _mount = load_runtime()
    flags = variant_flags(batch_config, batch_config["default_variant"])
    out = args.out
    views = out / "views"
    views.mkdir(parents=True, exist_ok=True)
    synthetic, synthetic_ids = open_bag(args.synthetic_bag)
    payload = {
        "algorithm_version": algorithm_version(),
        "reader_version": reader_version(),
        "diagnostic_spec": spec["raw"],
        "e01": [],
        "e03": [],
        "quiet": [],
        "platform": [],
        "curve": [],
        "smoke": [],
    }
    for index in E01_FRAMES:
        print("e01", index, flush=True)
        payload["e01"].append(
            analyze_frame(
                synthetic, synthetic_ids, index, gate_e01, mount, corridor_config, detector_config, flags, spec,
                trace=index in set(E01_TRACE + E01_EARLY), out_views=views,
            )
        )
    for index in E03_FRAMES:
        print("e03", index, flush=True)
        payload["e03"].append(
            analyze_frame(
                synthetic, synthetic_ids, index, gate_e03, mount, corridor_config, detector_config, flags, spec,
                trace=False, out_views=views,
            )
        )
    for index in QUIET_FRAMES:
        print("quiet", index, flush=True)
        payload["quiet"].append(
            analyze_frame(
                synthetic, synthetic_ids, index, gate_none, mount, corridor_config, detector_config, flags, spec,
                trace=False, out_views=views,
            )
        )
    synthetic.close()
    for name, bag, frames in (
        ("platform", args.platform_bag, PLATFORM_FRAMES),
        ("curve", args.curve_bag, CURVE_FRAMES),
        ("smoke", args.smoke_bag, [0, 1]),
    ):
        connection, ids = open_bag(bag)
        for index in frames:
            print(name, index, flush=True)
            payload[name].append(
                analyze_frame(
                    connection, ids, index, gate_none, mount, corridor_config, detector_config, flags, spec,
                    trace=name == "curve", out_views=views,
                )
            )
        connection.close()
    def convert(value):
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    (out / "ab_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=convert), encoding="utf-8"
    )
    print("wrote", out / "ab_results.json", flush=True)


if __name__ == "__main__":
    main()
