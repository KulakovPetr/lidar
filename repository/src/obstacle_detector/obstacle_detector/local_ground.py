"""Local ground candidates in working coordinates.

s = -Y, l = X, h = Z. This is a prior for the experiment, not a measured
velocity or gravity. The sensor XYZ stay available beside these coordinates.
"""

from __future__ import annotations

import numpy as np

# Fixed before the transfer frames are interpreted. Not tuned per bag name.
CONFIG = {
    "s_min_m": 5.0,
    "s_max_m": 30.0,
    "segment_m": 2.0,
    "height_bin_m": 0.05,
    "min_groups_per_mode": 40,
    "min_lateral_span_m": 1.5,
    "continuity_m": 0.30,
    "plane_inlier_m": 0.08,
    "section_s_m": [6.0, 10.0, 15.0, 20.0, 25.0],
    "section_half_m": 0.5,
    "section_half_fallback_m": 1.0,
    "section_min_groups": 30,
    "zoom_below_m": 0.15,
    "zoom_above_m": 0.50,
    "structure_dh_min_m": 0.04,
    "structure_dh_max_m": 0.40,
    "structure_l_gap_m": 0.15,
    "structure_max_width_m": 0.60,
    "structure_min_groups": 6,
    "structure_min_sections": 3,
    "structure_match_m": 0.25,
    "structure_max_missed_sections": 1,
    # Two continuations are both kept when their prediction residuals are this close.
    # This does not widen structure_match_m.
    "structure_link_ambiguity_m": 0.10,
    # Robust forecast uses at least this many sections. Two points are not a forecast.
    "structure_forecast_min_sections": 3,
    "structure_forecast_window": 4,
    # Before a forecast exists, a continuation may have a non-zero slope inside this cone.
    # The bound is global. It is not chosen per frame.
    "structure_init_max_abs_dl_ds": 0.15,
    "structure_max_alternatives": 3,
    "structure_max_live_tracks": 32,
    # A wide group is split only when its own profile has two peaks and a valley.
    "structure_wide_hist_bin_m": 0.08,
    "structure_wide_valley_ratio": 0.40,
    "structure_wide_peak_dh_m": 0.05,
}


def sensor_to_working(x, y, z):
    """s along -Y, l along X, h along Z."""
    return -np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)


def working_to_sensor(s, l, h):
    return np.asarray(l, dtype=np.float64), -np.asarray(s, dtype=np.float64), np.asarray(h, dtype=np.float64)


def transform_roundtrip_max_error(x, y, z) -> float:
    s, l, h = sensor_to_working(x, y, z)
    xr, yr, zr = working_to_sensor(s, l, h)
    return float(np.max(np.abs(np.stack([xr - x, yr - y, zr - z]))))


def _segments(config):
    edges = np.arange(config["s_min_m"], config["s_max_m"] + 1e-9, config["segment_m"])
    if edges[-1] < config["s_max_m"]:
        edges = np.r_[edges, config["s_max_m"]]
    return edges


def _modes_in_segment(s, l, h, lo, hi, config):
    sel = (s >= lo) & (s < hi)
    if int(sel.sum()) < config["min_groups_per_mode"]:
        return []
    hs = h[sel]
    ls = l[sel]
    bin_m = config["height_bin_m"]
    h0 = np.floor(np.min(hs) / bin_m) * bin_m
    h1 = np.ceil(np.max(hs) / bin_m) * bin_m
    if h1 <= h0:
        h1 = h0 + bin_m
    nb = int(round((h1 - h0) / bin_m))
    nb = max(nb, 1)
    bin_id = np.clip(np.floor((hs - h0) / bin_m).astype(np.int32), 0, nb - 1)
    counts = np.bincount(bin_id, minlength=nb)
    modes = []
    for b in range(nb):
        if int(counts[b]) < config["min_groups_per_mode"]:
            continue
        if 0 < b < nb - 1 and counts[b] < counts[b - 1] and counts[b] < counts[b + 1]:
            continue
        if b > 0 and counts[b] < counts[b - 1]:
            continue
        if b + 1 < nb and counts[b] < counts[b + 1]:
            continue
        members = bin_id == b
        span = float(ls[members].max() - ls[members].min())
        if span < config["min_lateral_span_m"]:
            continue
        modes.append(
            {
                "s_lo": float(lo),
                "s_hi": float(hi),
                "s_mid": float(0.5 * (lo + hi)),
                "h": float(np.median(hs[members])),
                "groups": int(counts[b]),
                "l_span": span,
                "l_min": float(ls[members].min()),
                "l_max": float(ls[members].max()),
                "member_mask_in_segment": members,
            }
        )
    modes.sort(key=lambda item: item["h"])
    kept = []
    for mode in modes:
        if any(abs(mode["h"] - other["h"]) < 3 * bin_m for other in kept):
            continue
        kept.append(mode)
    return kept


def _link_tracks(segment_modes, config):
    tracks = []
    for modes in segment_modes:
        used = set()
        for track in tracks:
            if track["closed"]:
                continue
            best = None
            best_dh = None
            for i, mode in enumerate(modes):
                if i in used:
                    continue
                dh = abs(mode["h"] - track["h"][-1])
                gap = mode["s_mid"] - track["s_mid"][-1]
                if gap <= 0 or gap > config["segment_m"] * 2.5:
                    continue
                if dh <= config["continuity_m"] and (best_dh is None or dh < best_dh):
                    best = i
                    best_dh = dh
            if best is None:
                if modes and (modes[0]["s_mid"] - track["s_mid"][-1]) > config["segment_m"] * 2.5:
                    track["closed"] = True
                continue
            used.add(best)
            mode = modes[best]
            track["h"].append(mode["h"])
            track["s_mid"].append(mode["s_mid"])
            track["s_lo"].append(mode["s_lo"])
            track["s_hi"].append(mode["s_hi"])
            track["groups"].append(mode["groups"])
            track["l_span"].append(mode["l_span"])
            track["l_min"].append(mode["l_min"])
            track["l_max"].append(mode["l_max"])
        for i, mode in enumerate(modes):
            if i in used:
                continue
            tracks.append(
                {
                    "h": [mode["h"]],
                    "s_mid": [mode["s_mid"]],
                    "s_lo": [mode["s_lo"]],
                    "s_hi": [mode["s_hi"]],
                    "groups": [mode["groups"]],
                    "l_span": [mode["l_span"]],
                    "l_min": [mode["l_min"]],
                    "l_max": [mode["l_max"]],
                    "closed": False,
                }
            )
    ready = []
    for track in tracks:
        if len(track["h"]) < 4:
            continue
        ready.append(
            {
                "n_segments": len(track["h"]),
                "h_median": float(np.median(track["h"])),
                "h_series": [float(v) for v in track["h"]],
                "s_mid": [float(v) for v in track["s_mid"]],
                "s_lo": [float(v) for v in track["s_lo"]],
                "s_hi": [float(v) for v in track["s_hi"]],
                "groups_per_segment": [int(v) for v in track["groups"]],
                "l_span_median": float(np.median(track["l_span"])),
                "l_min": float(np.min(track["l_min"])),
                "l_max": float(np.max(track["l_max"])),
                "support_groups": int(np.sum(track["groups"])),
            }
        )
    return ready


def _assign_roles(tracks):
    if not tracks:
        return None, []
    ordered = sorted(tracks, key=lambda item: item["h_median"])
    ground = ordered[0]
    ground["role"] = "low_extended_surface"
    alternatives = []
    for track in ordered[1:]:
        track = dict(track)
        if track["h_median"] - ground["h_median"] > 0.8:
            track["role"] = "higher_extended_surface"
        else:
            track["role"] = "nearby_extended_surface"
        alternatives.append(track)
    ambiguous = any(abs(track["h_median"] - ground["h_median"]) <= 0.25 for track in ordered[1:])
    ground["ambiguous_with_nearby_surface"] = ambiguous
    return ground, alternatives


def _local_line(l, h):
    if h.size < 8:
        return float(np.median(h)) if h.size else 0.0, 0.0, np.zeros(h.shape, dtype=bool)
    med = float(np.median(h))
    mad = float(np.median(np.abs(h - med)))
    limit = max(3.0 * 1.4826 * mad, 0.08)
    keep = np.abs(h - med) <= limit
    if int(keep.sum()) < 8:
        keep = np.ones(h.shape, dtype=bool)
    design = np.stack([l[keep], np.ones(int(keep.sum()))], axis=1)
    slope, intercept = np.linalg.lstsq(design, h[keep], rcond=None)[0]
    return float(intercept), float(slope), keep


def _fit_summary_plane(s, l, h, threshold):
    rng = np.random.default_rng(4)
    n = int(s.size)
    if n < 30:
        return None
    best = None
    best_count = -1
    for _ in range(80):
        pick = rng.choice(n, 3, replace=False)
        design = np.stack([s[pick], l[pick], np.ones(3)], axis=1)
        coef, *_ = np.linalg.lstsq(design, h[pick], rcond=None)
        resid = np.abs(h - (coef[0] * s + coef[1] * l + coef[2]))
        count = int(np.count_nonzero(resid <= threshold))
        if count > best_count:
            best_count = count
            best = coef
    resid = np.abs(h - (best[0] * s + best[1] * l + best[2]))
    inliers = resid <= threshold
    if int(inliers.sum()) >= 30:
        design = np.stack([s[inliers], l[inliers], np.ones(int(inliers.sum()))], axis=1)
        best, *_ = np.linalg.lstsq(design, h[inliers], rcond=None)
        resid = np.abs(h - (best[0] * s + best[1] * l + best[2]))
    # Plane: h = b_s * s + b_l * l + c, with s=-y, l=x, h=z.
    # z = b_s*(-y) + b_l*x + c => -b_l x + b_s y + z = c
    normal = np.array([-best[1], best[0], 1.0], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    return {
        "coef_h_from_s_l": [float(best[0]), float(best[1]), float(best[2])],
        "normal_sensor_xyz": [float(v) for v in normal],
        "tilt_from_sensor_plus_z_deg": tilt,
        "residual_p50_m": float(np.percentile(resid, 50)),
        "residual_p95_m": float(np.percentile(resid, 95)),
        "inlier_groups": int(np.count_nonzero(resid <= threshold)),
        "groups_in_fit": n,
    }


def estimate_ground(s, l, h, config=None):
    """Several extended surfaces. The low wide one is the ground candidate.

    A larger high surface is kept as an alternative and is not selected just
    because it contains more points. A lone low point is not a surface.
    """
    config = CONFIG if config is None else config
    edges = _segments(config)
    segment_modes = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        segment_modes.append(_modes_in_segment(s, l, h, lo, hi, config))
    tracks = _link_tracks(segment_modes, config)
    ground, alternatives = _assign_roles(tracks)
    if ground is None:
        return {
            "status": "not_found",
            "reason": "нет протяжённой поверхности с боковым охватом в 5–30 м",
            "alternatives": [],
            "local_bins": [],
            "plane": None,
        }
    bins = []
    resid_chunks = []
    for lo, hi, h_mode in zip(ground["s_lo"], ground["s_hi"], ground["h_series"]):
        sel = (s >= lo) & (s < hi) & (np.abs(h - h_mode) <= 2 * config["height_bin_m"])
        intercept, slope, keep = _local_line(l[sel], h[sel])
        if int(sel.sum()):
            pred = intercept + slope * l[sel]
            resid = np.abs(h[sel] - pred)
            resid_chunks.append(resid[keep] if keep.size == resid.size else resid)
            local_p50 = float(np.percentile(resid, 50))
            local_p95 = float(np.percentile(resid, 95))
            lateral_lo = float(np.min(l[sel]))
            lateral_hi = float(np.max(l[sel]))
        else:
            local_p50 = None
            local_p95 = None
            lateral_lo = None
            lateral_hi = None
        bins.append(
            {
                "s_lo": lo,
                "s_hi": hi,
                "intercept": intercept,
                "slope_dh_dl": slope,
                "groups": int(sel.sum()),
                "l_min": lateral_lo,
                "l_max": lateral_hi,
                "residual_p50_m": local_p50,
                "residual_p95_m": local_p95,
                "fitting_residual_p50_m": local_p50,
                "fitting_residual_p95_m": local_p95,
                "residual_name": "fitting_residual",
                "residual_is_not_ground_accuracy": True,
            }
        )
    if resid_chunks:
        all_resid = np.concatenate(resid_chunks)
        local_summary = {
            "p50_m": float(np.percentile(all_resid, 50)),
            "p95_m": float(np.percentile(all_resid, 95)),
            "groups": int(all_resid.size),
            "name": "fitting_residual",
            "is_ground_accuracy": False,
            "note": "остаток подгонки на предварительно выбранной полосе, не точность основания",
        }
    else:
        local_summary = {
            "p50_m": None,
            "p95_m": None,
            "groups": 0,
            "name": "fitting_residual",
            "is_ground_accuracy": False,
            "note": "остаток подгонки на предварительно выбранной полосе, не точность основания",
        }
    band = np.zeros(s.shape, dtype=bool)
    for lo, hi, h_mode in zip(ground["s_lo"], ground["s_hi"], ground["h_series"]):
        band |= (s >= lo) & (s < hi) & (np.abs(h - h_mode) <= 2 * config["height_bin_m"])
    plane = _fit_summary_plane(s[band], l[band], h[band], config["plane_inlier_m"]) if band.any() else None
    return {
        "status": "candidate",
        "reason": "нижняя из протяжённых поверхностей с боковым охватом; более высокие сохранены отдельно",
        "ground": {key: value for key, value in ground.items() if key != "member_mask_in_segment"},
        "alternatives": alternatives,
        "local_bins": bins,
        "local_residual": local_summary,
        "fitting_residual": local_summary,
        "plane": plane,
        "ambiguous_with_nearby_surface": ground["ambiguous_with_nearby_surface"],
    }


def height_above_local(s, l, h, ground) -> np.ndarray:
    """Height above the local bin model. NaN where that s is outside support."""
    out = np.full(s.shape, np.nan, dtype=np.float64)
    if ground.get("status") != "candidate":
        return out
    for bin_model in ground["local_bins"]:
        sel = (s >= bin_model["s_lo"]) & (s < bin_model["s_hi"])
        pred = bin_model["intercept"] + bin_model["slope_dh_dl"] * l[sel]
        out[sel] = h[sel] - pred
    return out


def lateral_support_mask(s, l, ground):
    """True where (s, l) lies on the fitted local surface, not beside it."""
    supported = np.zeros(s.shape, dtype=bool)
    if not ground or ground.get("status") != "candidate":
        return supported
    for bin_model in ground["local_bins"]:
        lo = bin_model.get("l_min")
        hi = bin_model.get("l_max")
        if lo is None or hi is None:
            continue
        sel = (s >= bin_model["s_lo"]) & (s < bin_model["s_hi"]) & (l >= lo) & (l <= hi)
        supported[sel] = True
    return supported


def _bin_at(ground, s_m: float):
    if not ground or ground.get("status") != "candidate":
        return None
    for bin_model in ground["local_bins"]:
        if bin_model["s_lo"] <= s_m < bin_model["s_hi"]:
            return bin_model
    return None


def section_arrays(s, l, h, dh, center, config, rep_index=None):
    half = config["section_half_m"]
    sel = np.abs(s - center) <= half
    thickness = 2 * half
    expanded = False
    if int(sel.sum()) < config["section_min_groups"]:
        half = config["section_half_fallback_m"]
        sel = np.abs(s - center) <= half
        thickness = 2 * half
        expanded = True
    status = "ok" if int(sel.sum()) >= config["section_min_groups"] else "insufficient"
    out = {
        "s_center_m": float(center),
        "thickness_m": float(thickness),
        "expanded_to_2m": expanded,
        "status": status,
        "unique_groups": int(sel.sum()),
        "l": l[sel],
        "h": h[sel],
        "dh": dh[sel],
        "s": s[sel],
    }
    if rep_index is not None:
        out["rep_index"] = np.asarray(rep_index)[sel]
    return out


def section_group_roles(section, config=None):
    """Label each point of one section. Roles are not a new threshold."""
    config = CONFIG if config is None else config
    n = int(section["l"].size) if "l" in section else 0
    roles = np.full(n, "no_dh", dtype=object)
    if n == 0 or section.get("status") != "ok":
        return roles
    dh = section["dh"]
    finite = np.isfinite(dh)
    roles[finite & (dh < config["structure_dh_min_m"])] = "below_height_band"
    roles[finite & (dh > config["structure_dh_max_m"])] = "above_height_band"
    band = finite & (dh >= config["structure_dh_min_m"]) & (dh <= config["structure_dh_max_m"])
    roles[band] = "unassigned_band"
    if int(band.sum()) < config["structure_min_groups"]:
        roles[band] = "band_below_min_groups"
        return roles
    order = np.argsort(section["l"][band], kind="mergesort")
    band_index = np.flatnonzero(band)[order]
    ll = section["l"][band_index]
    splits = np.flatnonzero(np.diff(ll) > config["structure_l_gap_m"])
    bounds = np.r_[0, splits + 1, ll.size]
    for a, b in zip(bounds[:-1], bounds[1:]):
        width = float(ll[b - 1] - ll[a])
        if b - a < config["structure_min_groups"]:
            roles[band_index[a:b]] = "rejected_few_groups"
        elif width > config["structure_max_width_m"]:
            roles[band_index[a:b]] = "rejected_wide"
        else:
            roles[band_index[a:b]] = "kept"
    return roles


def _section_clusters(section, config):
    """Clusters in one section, plus the counts that removed points."""
    empty = {
        "s_m": float(section.get("s_center_m", 0.0)),
        "status": section.get("status"),
        "unique_groups": int(section.get("unique_groups", 0)),
        "finite_dh": 0,
        "in_height_band": 0,
        "below_height_band": 0,
        "above_height_band": 0,
        "rejected_few_groups": 0,
        "rejected_few_points": 0,
        "rejected_wide": 0,
        "rejected_wide_points": 0,
        "clusters_kept": 0,
        "wide": [],
        "kept": [],
    }
    if section.get("status") != "ok":
        empty["stop"] = "section_not_ok"
        return empty, []
    dh = section["dh"]
    finite = np.isfinite(dh)
    empty["finite_dh"] = int(finite.sum())
    empty["below_height_band"] = int((finite & (dh < config["structure_dh_min_m"])).sum())
    empty["above_height_band"] = int((finite & (dh > config["structure_dh_max_m"])).sum())
    band = finite & (dh >= config["structure_dh_min_m"]) & (dh <= config["structure_dh_max_m"])
    empty["in_height_band"] = int(band.sum())
    if int(band.sum()) < config["structure_min_groups"]:
        empty["stop"] = "height_band_below_min_groups"
        return empty, []
    order = np.argsort(section["l"][band], kind="mergesort")
    ll = section["l"][band][order]
    hh = section["h"][band][order]
    dd = dh[band][order]
    reps = None if "rep_index" not in section else section["rep_index"][band][order]
    splits = np.flatnonzero(np.diff(ll) > config["structure_l_gap_m"])
    bounds = np.r_[0, splits + 1, ll.size]
    clusters = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        width = float(ll[b - 1] - ll[a])
        row = {
            "s_center_m": float(section["s_center_m"]),
            "l_median": float(np.median(ll[a:b])),
            "l_min": float(ll[a]),
            "l_max": float(ll[b - 1]),
            "width_m": width,
            "dh_median": float(np.median(dd[a:b])),
            "h_median": float(np.median(hh[a:b])),
            "groups": int(b - a),
            "rep_indices": [] if reps is None else [int(v) for v in reps[a:b]],
        }
        if b - a < config["structure_min_groups"]:
            empty["rejected_few_groups"] += 1
            empty["rejected_few_points"] += int(b - a)
            continue
        if width > config["structure_max_width_m"]:
            empty["rejected_wide"] += 1
            empty["rejected_wide_points"] += int(b - a)
            empty["wide"].append(
                {"l_median": row["l_median"], "width_m": width, "groups": row["groups"], "dh_median": row["dh_median"]}
            )
            continue
        clusters.append(row)
        empty["kept"].append(
            {"l_median": row["l_median"], "width_m": width, "groups": row["groups"], "dh_median": row["dh_median"]}
        )
    empty["clusters_kept"] = len(clusters)
    return empty, clusters


def _track_slope(track):
    if len(track["s"]) < 2:
        return None
    ds = float(track["s"][-1] - track["s"][-2])
    if abs(ds) < 1e-9:
        return None
    return float((track["l_median"][-1] - track["l_median"][-2]) / ds)


def _copy_track(track):
    return {
        "clusters": list(track["clusters"]),
        "l_median": list(track["l_median"]),
        "s": list(track["s"]),
        "alternative_l_m": list(track.get("alternative_l_m", [])),
    }


def _extend_track(track, cluster):
    track["clusters"].append(cluster)
    track["l_median"].append(cluster["l_median"])
    track["s"].append(cluster["s_center_m"])


def _link_section_clusters(per_section, centers, config, link: str):
    """Connect clusters along one frame.

    `fixed_dl` compares raw lateral positions. `spatial` predicts the next
    position from the local slope dl/ds and the section step. The slope is a
    geometric forecast inside the frame, not a speed between frames.
    """
    center_index = {value: i for i, value in enumerate(centers)}
    tracks = []
    for section_i, clusters in enumerate(per_section):
        section_s = centers[section_i]
        used = set()
        extended = []
        for track in tracks:
            missed = section_i - center_index[track["s"][-1]] - 1
            if missed > config["structure_max_missed_sections"]:
                extended.append(track)
                continue
            slope = _track_slope(track)
            ds = float(section_s - track["s"][-1])
            if link == "spatial" and slope is not None:
                predicted = float(track["l_median"][-1] + slope * ds)
            else:
                predicted = float(track["l_median"][-1])
            hits = []
            for i, cluster in enumerate(clusters):
                if i in used:
                    continue
                residual = abs(float(cluster["l_median"]) - predicted)
                if residual <= config["structure_match_m"]:
                    hits.append((residual, i))
            hits.sort()
            if not hits:
                extended.append(track)
                continue
            chosen = [hits[0]]
            margin = config["structure_link_ambiguity_m"]
            if link == "spatial" and len(hits) >= 2 and hits[1][0] - hits[0][0] < margin:
                chosen.append(hits[1])
            snapshot = _copy_track(track) if len(chosen) > 1 else None
            if snapshot is not None:
                track.setdefault("alternative_l_m", []).append(float(clusters[chosen[1][1]]["l_median"]))
                snapshot["alternative_l_m"].append(float(clusters[chosen[0][1]]["l_median"]))
            for n, (_residual, index) in enumerate(chosen):
                used.add(index)
                branch = track if n == 0 else snapshot
                _extend_track(branch, clusters[index])
                extended.append(branch)
        for i, cluster in enumerate(clusters):
            if i in used:
                continue
            extended.append(
                {
                    "clusters": [cluster],
                    "l_median": [cluster["l_median"]],
                    "s": [cluster["s_center_m"]],
                    "alternative_l_m": [],
                }
            )
        tracks = extended
    return tracks


def _pairwise_slope(s_values, l_values):
    slopes = []
    for i in range(len(s_values)):
        for j in range(i + 1, len(s_values)):
            ds = float(s_values[j] - s_values[i])
            if abs(ds) < 1e-9:
                continue
            slopes.append(float(l_values[j] - l_values[i]) / ds)
    if not slopes:
        return None
    return float(np.median(slopes))


def _forecast_track(track, section_s, config):
    """Local slope from several sections. Two observations are not a forecast."""
    s_values = [float(v) for v in track["s"]]
    l_values = [float(v) for v in track["l_median"]]
    order = np.argsort(s_values, kind="mergesort")
    s_values = [s_values[i] for i in order][-int(config["structure_forecast_window"]) :]
    l_values = [l_values[i] for i in order][-int(config["structure_forecast_window"]) :]
    ds = float(section_s - s_values[-1])
    if len(s_values) < int(config["structure_forecast_min_sections"]) or abs(ds) < 1e-9:
        return {
            "mode": "insufficient_observations",
            "n_sections": len(s_values),
            "slope_dl_ds": None,
            "predicted_l_m": None,
            "ds_m": ds,
        }
    slope = _pairwise_slope(s_values, l_values)
    if slope is None:
        return {
            "mode": "insufficient_observations",
            "n_sections": len(s_values),
            "slope_dl_ds": None,
            "predicted_l_m": None,
            "ds_m": ds,
        }
    return {
        "mode": "robust_local_fit",
        "n_sections": len(s_values),
        "slope_dl_ds": slope,
        "predicted_l_m": float(l_values[-1] + slope * ds),
        "ds_m": ds,
    }


def _track_sort_key(track):
    pairs = tuple((round(float(s), 3), round(float(l), 3)) for s, l in zip(track["s"], track["l_median"]))
    return tuple(sorted(pairs))


def link_clusters(per_section, centers, config=None, link="robust"):
    """Link clusters. `robust` does not follow the input order of a section."""
    config = CONFIG if config is None else config
    if link != "robust":
        return _link_section_clusters(per_section, centers, config, link)
    center_index = {value: i for i, value in enumerate(centers)}
    tracks = []
    for section_i, clusters in enumerate(per_section):
        section_s = float(centers[section_i])
        proposals = []
        for t_index, track in enumerate(tracks):
            missed = section_i - center_index[track["s"][-1]] - 1
            if missed > config["structure_max_missed_sections"]:
                continue
            forecast = _forecast_track(track, section_s, config)
            ds = float(section_s - track["s"][-1])
            if abs(ds) < 1e-9:
                continue
            for c_index, cluster in enumerate(clusters):
                implied = float(cluster["l_median"] - track["l_median"][-1]) / ds
                if forecast["mode"] == "robust_local_fit":
                    residual = abs(float(cluster["l_median"]) - forecast["predicted_l_m"])
                    if residual <= config["structure_match_m"]:
                        proposals.append(
                            {
                                "cost": residual,
                                "track_key": _track_sort_key(track),
                                "l_m": float(cluster["l_median"]),
                                "t_index": t_index,
                                "c_index": c_index,
                                "kind": "forecast",
                                "implied_dl_ds": implied,
                                "forecast": forecast,
                                "residual_m": residual,
                            }
                        )
                elif abs(implied) <= config["structure_init_max_abs_dl_ds"]:
                    proposals.append(
                        {
                            "cost": 0.0,
                            "track_key": _track_sort_key(track),
                            "l_m": float(cluster["l_median"]),
                            "t_index": t_index,
                            "c_index": c_index,
                            "kind": "init",
                            "implied_dl_ds": implied,
                            "forecast": forecast,
                            "residual_m": abs(float(cluster["l_median"]) - track["l_median"][-1]),
                        }
                    )
        proposals.sort(key=lambda item: (item["cost"], item["track_key"], item["l_m"], item["c_index"]))
        assigned_tracks = set()
        assigned_clusters = set()
        primary = {}
        for item in proposals:
            if item["t_index"] in assigned_tracks or item["c_index"] in assigned_clusters:
                continue
            assigned_tracks.add(item["t_index"])
            assigned_clusters.add(item["c_index"])
            primary[item["t_index"]] = item
        alternatives = {t_index: [] for t_index in primary}
        for item in proposals:
            first = primary.get(item["t_index"])
            if first is None or item["c_index"] == first["c_index"] or item["c_index"] in assigned_clusters:
                continue
            if item["kind"] != first["kind"]:
                continue
            close = abs(item["cost"] - first["cost"]) < config["structure_link_ambiguity_m"]
            if first["kind"] == "init":
                close = True
            if not close:
                continue
            if len(alternatives[item["t_index"]]) >= config["structure_max_alternatives"] - 1:
                continue
            if len(tracks) + len(alternatives[item["t_index"]]) >= config["structure_max_live_tracks"]:
                continue
            assigned_clusters.add(item["c_index"])
            alternatives[item["t_index"]].append(item)
        extended = []
        for t_index, track in enumerate(tracks):
            first = primary.get(t_index)
            if first is None:
                extended.append(track)
                continue
            chosen = [first, *alternatives[t_index]]
            snapshot = _copy_track(track) if len(chosen) > 1 else None
            for n, item in enumerate(chosen):
                branch = track if n == 0 else _copy_track(snapshot)
                if len(chosen) > 1:
                    others = [float(clusters[other["c_index"]]["l_median"]) for other in chosen if other is not item]
                    branch.setdefault("alternative_l_m", []).extend(others)
                branch["last_link"] = {
                    "kind": item["kind"],
                    "forecast_mode": item["forecast"]["mode"],
                    "forecast_sections": item["forecast"]["n_sections"],
                    "slope_dl_ds": item["forecast"]["slope_dl_ds"],
                    "predicted_l_m": item["forecast"]["predicted_l_m"],
                    "implied_dl_ds": item["implied_dl_ds"],
                    "residual_m": item["residual_m"],
                    "observed_l_m": float(clusters[item["c_index"]]["l_median"]),
                    "s_m": section_s,
                }
                _extend_track(branch, clusters[item["c_index"]])
                extended.append(branch)
        for c_index, cluster in enumerate(clusters):
            if c_index in assigned_clusters:
                continue
            extended.append(
                {
                    "clusters": [cluster],
                    "l_median": [cluster["l_median"]],
                    "s": [cluster["s_center_m"]],
                    "alternative_l_m": [],
                    "last_link": None,
                }
            )
        tracks = extended
    return tracks


def explain_spatial_break(clusters_along_track, config=None):
    """Replay the two-point slope on one known track and report the first rejected hop."""
    config = CONFIG if config is None else config
    if len(clusters_along_track) < 2:
        return None
    for index in range(1, len(clusters_along_track)):
        previous = clusters_along_track[: index]
        nxt = clusters_along_track[index]
        ds = float(nxt["s_center_m"] - previous[-1]["s_center_m"])
        if abs(ds) < 1e-9:
            continue
        if len(previous) >= 2:
            slope = float(previous[-1]["l_median"] - previous[-2]["l_median"]) / float(
                previous[-1]["s_center_m"] - previous[-2]["s_center_m"]
            )
            predicted = float(previous[-1]["l_median"] + slope * ds)
            mode = "last_two_centers"
        else:
            slope = None
            predicted = float(previous[-1]["l_median"])
            mode = "no_slope_yet"
        residual = abs(float(nxt["l_median"]) - predicted)
        implied = float(nxt["l_median"] - previous[-1]["l_median"]) / ds
        if residual > config["structure_match_m"]:
            return {
                "broke": True,
                "at_s_m": float(nxt["s_center_m"]),
                "from_s_m": float(previous[-1]["s_center_m"]),
                "mode": mode,
                "slope_dl_ds": slope,
                "predicted_l_m": predicted,
                "observed_l_m": float(nxt["l_median"]),
                "residual_m": residual,
                "implied_dl_ds": implied,
                "match_m": config["structure_match_m"],
                "representative_indices": [int(v) for v in nxt.get("rep_indices", [])[:8]],
            }
    return {"broke": False}


def _wide_profiles(section, config):
    """Points inside groups wider than the comparison threshold. The threshold stays."""
    if section.get("status") != "ok":
        return []
    dh = section["dh"]
    band = np.isfinite(dh) & (dh >= config["structure_dh_min_m"]) & (dh <= config["structure_dh_max_m"])
    if int(band.sum()) < config["structure_min_groups"]:
        return []
    order = np.argsort(section["l"][band], kind="mergesort")
    ll = section["l"][band][order]
    dd = dh[band][order]
    hh = section["h"][band][order]
    reps = None if "rep_index" not in section else section["rep_index"][band][order]
    splits = np.flatnonzero(np.diff(ll) > config["structure_l_gap_m"])
    bounds = np.r_[0, splits + 1, ll.size]
    groups = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        width = float(ll[b - 1] - ll[a])
        if b - a < config["structure_min_groups"] or width <= config["structure_max_width_m"]:
            continue
        groups.append(
            {
                "s_center_m": float(section["s_center_m"]),
                "l": ll[a:b].copy(),
                "dh": dd[a:b].copy(),
                "h": hh[a:b].copy(),
                "rep_indices": [] if reps is None else [int(v) for v in reps[a:b]],
                "width_m": width,
                "groups": int(b - a),
                "l_median": float(np.median(ll[a:b])),
                "dh_median": float(np.median(dd[a:b])),
                "h_median": float(np.median(hh[a:b])),
            }
        )
    return groups


def _substructures(group, config):
    """Split only on the group's own lateral profile. Expecting rails is not a reason."""
    ll = np.asarray(group["l"], dtype=np.float64)
    dd = np.asarray(group["dh"], dtype=np.float64)
    if ll.size < 2 * config["structure_min_groups"]:
        return {"split_justified": False, "reason": "мало точек для двух подструктур", "peaks": []}
    internal_gap = float(np.max(np.diff(ll)))
    bin_m = config["structure_wide_hist_bin_m"]
    start = float(ll[0])
    stop = float(ll[-1]) + bin_m
    edges = np.arange(start, stop + 1e-9, bin_m)
    if edges.size < 4:
        return {"split_justified": False, "reason": "профиль слишком узкий для двух пиков", "peaks": [], "internal_gap_m": internal_gap}
    counts, _ = np.histogram(ll, bins=edges)
    peaks = []
    for index in range(1, counts.size - 1):
        if counts[index] >= counts[index - 1] and counts[index] >= counts[index + 1] and counts[index] > 0:
            peaks.append(index)
    if len(peaks) < 2:
        return {
            "split_justified": False,
            "reason": "нет двух пиков поперечного профиля",
            "internal_gap_m": internal_gap,
            "peaks": [],
        }
    peaks = sorted(peaks, key=lambda index: int(counts[index]), reverse=True)[:2]
    peaks = sorted(peaks)
    left, right = peaks
    valley = int(np.min(counts[left + 1 : right])) if right > left + 1 else int(counts[left])
    smaller = int(min(counts[left], counts[right]))
    deep = smaller > 0 and valley <= config["structure_wide_valley_ratio"] * smaller
    left_sel = (ll >= edges[left]) & (ll < edges[left + 1])
    right_sel = (ll >= edges[right]) & (ll < edges[right + 1])
    # Include the shoulder out to the valley, not only the peak bin.
    valley_l = float(0.5 * (edges[left + 1] + edges[right]))
    left_sel = ll <= valley_l
    right_sel = ll > valley_l
    dh_gap = abs(float(np.median(dd[left_sel])) - float(np.median(dd[right_sel])))
    enough = int(left_sel.sum()) >= config["structure_min_groups"] and int(right_sel.sum()) >= config["structure_min_groups"]
    justified = bool(deep and enough and (dh_gap >= config["structure_wide_peak_dh_m"] or internal_gap >= 0.8 * config["structure_l_gap_m"]))
    reason = "два пика разделены провалом профиля" if justified else "провал профиля или различие высот недостаточны"
    parts = []
    if justified:
        for sel in (left_sel, right_sel):
            parts.append(
                {
                    "l_median": float(np.median(ll[sel])),
                    "l_min": float(ll[sel].min()),
                    "l_max": float(ll[sel].max()),
                    "dh_median": float(np.median(dd[sel])),
                    "h_median": float(np.median(np.asarray(group["h"])[sel])),
                    "groups": int(sel.sum()),
                    "width_m": float(ll[sel].max() - ll[sel].min()),
                    "s_center_m": group["s_center_m"],
                    "rep_indices": [int(v) for v in np.asarray(group["rep_indices"])[sel][:8]] if group["rep_indices"] else [],
                }
            )
    return {
        "split_justified": justified,
        "reason": reason,
        "internal_gap_m": internal_gap,
        "valley_count": valley,
        "dh_peak_difference_m": dh_gap,
        "peaks": parts,
    }


def analyze_wide_groups(sections, config=None, kept_by_s=None):
    """Inspect groups rejected by the unchanged 0.60 m width gate."""
    config = CONFIG if config is None else config
    found = []
    for section in sections:
        for group in _wide_profiles(section, config):
            profile = _substructures(group, config)
            found.append({"group": group, "profile": profile})
    by_s = {}
    for item in found:
        by_s.setdefault(item["group"]["s_center_m"], []).append(item)
    stations = sorted(by_s)
    for item in found:
        item["continued"] = False
        if not item["profile"]["split_justified"]:
            continue
        s_now = item["group"]["s_center_m"]
        later = [station for station in stations if station > s_now]
        if not later:
            item["profile"]["reason"] = "пики есть только на одном сечении, продольной поддержки нет"
            item["profile"]["split_justified"] = False
            item["profile"]["peaks"] = []
            continue
        nxt = by_s[later[0]]
        continued = 0
        extra = [] if not kept_by_s else list(kept_by_s.get(later[0], []))
        for peak in item["profile"]["peaks"]:
            target_l = list(extra)
            for other in nxt:
                target_l.append(other["group"]["l_median"])
                if other["profile"]["split_justified"]:
                    target_l.extend(part["l_median"] for part in other["profile"]["peaks"])
            ds = float(later[0] - s_now)
            limit = max(config["structure_match_m"], abs(ds) * config["structure_init_max_abs_dl_ds"])
            if any(abs(peak["l_median"] - value) <= limit for value in target_l):
                continued += 1
        item["continued"] = continued == len(item["profile"]["peaks"])
        if not item["continued"]:
            item["profile"]["split_justified"] = False
            item["profile"]["reason"] = "пики не продолжаются на соседнем сечении"
            item["profile"]["peaks"] = []
    summary = []
    for item in found:
        group = item["group"]
        summary.append(
            {
                "s_m": group["s_center_m"],
                "l_median": group["l_median"],
                "width_m": group["width_m"],
                "groups": group["groups"],
                "dh_median": group["dh_median"],
                "internal_gap_m": item["profile"].get("internal_gap_m"),
                "dh_peak_difference_m": item["profile"].get("dh_peak_difference_m"),
                "split_justified": item["profile"]["split_justified"],
                "reason": item["profile"]["reason"],
                "substructures": item["profile"]["peaks"],
                "representative_indices": group["rep_indices"][:8],
            }
        )
    return {"kept_width_gate_m": config["structure_max_width_m"], "groups": summary, "plot_groups": found}


def _candidates_from_tracks(tracks, expected, config, ground=None, min_sections=None):
    candidates = []
    minimum = config["structure_min_sections"] if min_sections is None else min_sections
    for track in tracks:
        if len(track["clusters"]) < minimum:
            continue
        present = set(track["s"])
        gaps = [s for s in expected if min(present) <= s <= max(present) and s not in present]
        widths = [c["width_m"] for c in track["clusters"]]
        samples = []
        for cluster in track["clusters"]:
            support = "unknown"
            bin_model = _bin_at(ground, cluster["s_center_m"]) if ground is not None else None
            if bin_model is not None and bin_model.get("l_min") is not None:
                inside = bin_model["l_min"] <= cluster["l_median"] <= bin_model["l_max"]
                support = "supported" if inside else "extrapolated"
            samples.append(
                {
                    "s_m": cluster["s_center_m"],
                    "l_m": cluster["l_median"],
                    "h_m": cluster["h_median"],
                    "dh_m": cluster["dh_median"],
                    "width_m": cluster["width_m"],
                    "groups": cluster["groups"],
                    "height_support": support,
                    "representative_indices": [int(v) for v in cluster.get("rep_indices", [])],
                }
            )
        key = "|".join(f"{sample['s_m']:.1f}:{sample['l_m']:.3f}" for sample in samples)
        candidates.append(
            {
                "label": "продольный кандидат",
                "geometry_key": key,
                "local_index_is_not_persistent": True,
                "s_m": track["s"],
                "l_median_m": track["l_median"],
                "l_median_of_track_m": float(np.median(track["l_median"])),
                "dh_median_m": float(np.median([c["dh_median"] for c in track["clusters"]])),
                "h_by_section_m": [c["h_median"] for c in track["clusters"]],
                "observed_width_m": {
                    "min": float(np.min(widths)),
                    "median": float(np.median(widths)),
                    "max": float(np.max(widths)),
                },
                "support_sections": len(track["clusters"]),
                "gaps_s_m": gaps,
                "alternative_continuations_l_m": [float(v) for v in track.get("alternative_l_m", [])],
                "unique_groups": int(sum(c["groups"] for c in track["clusters"])),
                "representative_indices": [
                    index for cluster in track["clusters"] for index in cluster.get("rep_indices", [])
                ],
                "section_samples": samples,
                "not_a_confirmed_rail": True,
                "width_is_not_a_physical_rail_width": True,
                "below_single_line_minimum": len(track["clusters"]) < config["structure_min_sections"],
                "last_link": track.get("last_link"),
            }
        )
    candidates.sort(key=lambda item: item["l_median_of_track_m"])
    for index, candidate in enumerate(candidates):
        candidate["local_index"] = index
    return candidates


def structure_selection(sections, config=None, ground=None):
    """Funnel plus the fixed-dl baseline and the spatial continuation."""
    config = CONFIG if config is None else config
    per_section = []
    trace_sections = []
    for section in sections:
        info, clusters = _section_clusters(section, config)
        trace_sections.append(info)
        per_section.append(clusters)
    centers = [float(section["s_center_m"]) for section in sections]
    fixed_tracks = link_clusters(per_section, centers, config, "fixed_dl")
    spatial_tracks = link_clusters(per_section, centers, config, "spatial")
    robust_tracks = link_clusters(per_section, centers, config, "robust")
    fixed = _candidates_from_tracks(fixed_tracks, centers, config, ground)
    spatial = _candidates_from_tracks(spatial_tracks, centers, config, ground)
    robust = _candidates_from_tracks(robust_tracks, centers, config, ground)
    fragments = _candidates_from_tracks(robust_tracks, centers, config, ground, min_sections=2)
    trace = {
        "steps": [
            "unique_groups",
            "finite_height_above_supported_ground",
            "height_band",
            "lateral_groups",
            "width_and_count",
            "section_links",
            "min_length",
        ],
        "sections": [
            {key: value for key, value in row.items() if key not in ("kept", "wide")}
            | {"kept_clusters": row["kept"], "rejected_wide_clusters": row["wide"]}
            for row in trace_sections
        ],
        "fixed_dl_track_lengths": [len(track["clusters"]) for track in fixed_tracks],
        "spatial_track_lengths": [len(track["clusters"]) for track in spatial_tracks],
        "robust_track_lengths": [len(track["clusters"]) for track in robust_tracks],
        "fixed_dl_candidates": len(fixed),
        "spatial_candidates": len(spatial),
        "robust_candidates": len(robust),
        "link_gate_m": config["structure_match_m"],
        "link_gate_was_not_widened": True,
    }
    return {
        "trace": trace,
        "baseline_candidates": fixed,
        "candidates": spatial,
        "robust_candidates": robust,
        "fragments": fragments,
        "fixed_tracks": fixed_tracks,
        "spatial_tracks": spatial_tracks,
        "robust_tracks": robust_tracks,
        "per_section": per_section,
        "centers": centers,
    }


def longitudinal_candidates(sections, config=None, link="spatial", ground=None):
    """Elevated clusters tracked across sections. Not confirmed rails."""
    config = CONFIG if config is None else config
    selected = structure_selection(sections, config, ground)
    if link == "fixed_dl":
        return selected["baseline_candidates"]
    if link == "robust":
        return selected["robust_candidates"]
    return selected["candidates"]
