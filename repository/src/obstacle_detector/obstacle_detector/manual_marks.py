"""Manual rail / floor / wall marks. Algorithm proposals are refused."""

from __future__ import annotations

import numpy as np

from obstacle_detector.angular_projection import (
    bin_center_deg,
    display_row_to_internal,
    indices_in_bin,
)

SURFACES = ("rail", "floor", "wall")
RANGE_BANDS = (
    (0.0, 10.0),
    (10.0, 30.0),
    (30.0, 50.0),
    (50.0, 100.0),
    (100.0, 200.0),
    (200.0, None),
)
FORBIDDEN_SOURCES = frozenset(
    {
        "algorithm",
        "proposal",
        "detector",
        "auto",
        "model",
        "model_proposal",
        "heuristic",
    }
)


def band_label(lo: float, hi: float | None) -> str:
    if hi is None:
        return f">={lo:g}"
    return f"[{lo:g},{hi:g})"


def range_band(range_m: float) -> tuple[float, float | None] | None:
    if range_m is None or not np.isfinite(range_m) or range_m < 0.0:
        return None
    for lo, hi in RANGE_BANDS:
        if hi is None or (range_m >= lo and range_m < hi):
            if hi is None and range_m >= lo:
                return lo, hi
            if hi is not None and lo <= range_m < hi:
                return lo, hi
    return None


def validate_mark(mark: dict) -> dict:
    source = str(mark.get("source", ""))
    if mark.get("algorithm_proposal") or source.lower() in FORBIDDEN_SOURCES:
        raise ValueError("предложение алгоритма не является разметкой")
    if source != "manual":
        raise ValueError("в таблицу видимости входит только source=manual")
    surface = mark.get("surface")
    if surface not in SURFACES:
        raise ValueError(f"поверхность должна быть одной из {SURFACES}")
    visibility = mark.get("visibility")
    if visibility not in ("visible", "not_visible"):
        raise ValueError("visibility должен быть visible или not_visible")
    confidence = mark.get("confidence")
    if confidence not in ("certain", "uncertain"):
        raise ValueError("confidence должен быть certain или uncertain")
    if mark.get("coordinate_frame") != "sensor_xyz":
        raise ValueError("нужна система координат sensor_xyz")
    indices = [int(i) for i in mark.get("point_indices", [])]
    ranges = mark.get("range_m_by_index")
    if ranges is not None:
        ranges = [float(v) for v in ranges]
    if visibility == "visible" and (not indices or not ranges):
        raise ValueError("видимая отметка должна содержать индексы точек и их дальности")
    if ranges is not None and len(ranges) != len(indices):
        raise ValueError("дальности и индексы разной длины")
    return {
        "source": "manual",
        "author": str(mark.get("author", "")),
        "surface": surface,
        "visibility": visibility,
        "confidence": confidence,
        "coordinate_frame": "sensor_xyz",
        "point_indices": indices,
        "range_m_by_index": None if ranges is None else [float(v) for v in ranges],
        "internal_row": mark.get("internal_row"),
        "col": mark.get("col"),
        "note": str(mark.get("note", "")),
    }


def mark_from_pixel(grid, spec, display_row, col, surface, visibility, confidence, author, note):
    """Attach every point index in the angular bin. An empty bin cannot be a visible mark."""
    if not str(author).strip():
        raise ValueError("у ручной отметки должен быть author")
    n_rows = int(spec["n_rows"])
    n_cols = int(spec["n_cols"])
    internal = display_row_to_internal(int(display_row), n_rows)
    col = int(col)
    if internal < 0 or internal >= n_rows or col < 0 or col >= n_cols:
        raise ValueError("пиксель вне угловой сетки")
    flat = internal * n_cols + col
    start = int(grid["indptr"][flat])
    stop = int(grid["indptr"][flat + 1])
    ids = indices_in_bin(grid, n_cols, internal, col)
    ranges = [float(v) for v in grid["range_for_indices"][start:stop]]
    rings = [int(v) for v in grid["ring_for_indices"][start:stop]]
    azimuth, elevation = bin_center_deg(internal, col, spec)
    mark = validate_mark(
        {
            "source": "manual",
            "author": author,
            "surface": surface,
            "visibility": visibility,
            "confidence": confidence,
            "coordinate_frame": "sensor_xyz",
            "point_indices": [int(i) for i in ids],
            "range_m_by_index": ranges,
            "internal_row": internal,
            "col": col,
            "note": note,
        }
    )
    mark["display_row"] = int(display_row)
    mark["ring_by_index"] = rings
    mark["bin_center_azimuth_deg"] = azimuth
    mark["bin_center_elevation_deg"] = elevation
    mark["ring_source"] = "point_field"
    return mark


def evidence_rows(frames: list[dict], marks_by_frame: dict[str, list[dict]]) -> list[dict]:
    """One row per frame, surface, and range band.

    A missing mark stays «нет ручной отметки». It is not treated as «не видно».
    """
    rows = []
    for frame in frames:
        key = frame["frame_key"]
        marks = [validate_mark(item) for item in marks_by_frame.get(key, [])]
        for surface in SURFACES:
            for lo, hi in RANGE_BANDS:
                matched = []
                for mark in marks:
                    if mark["surface"] != surface or not mark["range_m_by_index"]:
                        continue
                    hit_ranges = [
                        r
                        for r in mark["range_m_by_index"]
                        if range_band(r) == (lo, hi)
                    ]
                    if hit_ranges:
                        matched.append((mark, hit_ranges))
                if not matched:
                    status = "нет ручной отметки"
                    source = ""
                    confidence = ""
                    indices = []
                    note = ""
                else:
                    flags = sorted({mark["visibility"] for mark, _ in matched})
                    if flags == ["visible"]:
                        status = "видимая ручная отметка"
                    elif flags == ["not_visible"]:
                        status = "ручная отметка: не видно"
                    else:
                        status = "есть и видимая отметка, и отметка «не видно»"
                    source = "manual"
                    confidence = ",".join(sorted({mark["confidence"] for mark, _ in matched}))
                    indices = []
                    notes = []
                    for mark, hit_ranges in matched:
                        for index, distance in zip(mark["point_indices"], mark["range_m_by_index"]):
                            if distance in hit_ranges:
                                indices.append(index)
                        if mark["note"]:
                            notes.append(mark["note"])
                    note = "; ".join(notes)
                rows.append(
                    {
                        "run": frame["run"],
                        "message_index": frame["message_index"],
                        "frame_id": frame["frame_id"],
                        "header_stamp_ns": frame["header_stamp_ns"],
                        "surface": surface,
                        "range_band": band_label(lo, hi),
                        "status": status,
                        "source": source,
                        "confidence": confidence,
                        "coordinate_frame": "sensor_xyz" if source else "",
                        "point_indices": indices,
                        "note": note,
                    }
                )
    return rows
