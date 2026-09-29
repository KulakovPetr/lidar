"""XYZ angular projection with an explicit reverse index.

This is not a factory ray grid. Channel angles from a manual are not applied.
Ring, when used, is copied from the point field. It is not recomputed from
elevation, including after a motion correction that this module does not do.
"""

from __future__ import annotations

import numpy as np

AZIMUTH_BIN_DEG = 0.2
ELEVATION_BIN_DEG = 0.2
ZERO_RANGE_M = 1e-3
# Two occupied ranges in one angular bin are recorded as multiple returns
# when they differ by more than this gap. This is bookkeeping, not a threshold
# of a detector and not a sensor return-mode register.
MULTIPLE_RETURN_GAP_M = 0.2
REPRESENTATION_XYZ = "xyz_angular_projection"
REPRESENTATION_RING = "ring_row_xyz_azimuth"


def classify_ranges(x, y, z, zero_range_m: float = ZERO_RANGE_M):
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    range_m = np.full(x.shape, np.nan, dtype=np.float64)
    range_m[finite] = np.hypot(np.hypot(x[finite], y[finite]), z[finite])
    zero = finite & (range_m < zero_range_m)
    valid = finite & ~zero
    return finite, zero, valid, range_m


def xyz_angles_deg(x, y, z):
    """Azimuth atan2(y, x) in [-180, 180) and elevation atan2(z, hypot(x, y))."""
    azimuth = np.degrees(np.arctan2(y, x))
    azimuth = np.where(azimuth >= 180.0, azimuth - 360.0, azimuth)
    elevation = np.degrees(np.arctan2(z, np.hypot(x, y)))
    return azimuth, elevation


def _csr(flat: np.ndarray, point_ids: np.ndarray, n_bins: int):
    order = np.argsort(flat, kind="mergesort")
    flat_sorted = flat[order]
    indices = point_ids[order]
    counts = np.bincount(flat_sorted, minlength=n_bins).astype(np.int64)
    indptr = np.zeros(n_bins + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, indices, counts


def _range_span_by_bin(indptr, counts, ranges_for_indices):
    occupied = np.flatnonzero(counts > 0)
    if occupied.size == 0:
        empty = np.zeros(0, dtype=bool)
        return occupied, empty
    starts = indptr[occupied]
    span = np.maximum.reduceat(ranges_for_indices, starts) - np.minimum.reduceat(
        ranges_for_indices, starts
    )
    return occupied, span > MULTIPLE_RETURN_GAP_M


def quantization_error_m(x, y, z, valid, azimuth, elevation, row, col, az_bin, el_bin, el_min):
    ids = np.flatnonzero(valid)
    if ids.size == 0:
        return np.zeros(0, dtype=np.float64)
    az_c = np.radians(-180.0 + (col[ids].astype(np.float64) + 0.5) * az_bin)
    el_c = np.radians(el_min + (row[ids].astype(np.float64) + 0.5) * el_bin)
    # Range is the original range. The only change is the bin-center angle.
    radius = np.hypot(np.hypot(x[ids], y[ids]), z[ids])
    horiz = radius * np.cos(el_c)
    dx = horiz * np.cos(az_c) - x[ids]
    dy = horiz * np.sin(az_c) - y[ids]
    dz = radius * np.sin(el_c) - z[ids]
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def _error_summary(error: np.ndarray) -> dict:
    if error.size == 0:
        return {"p50": None, "p95": None, "max": None}
    return {
        "p50": float(np.percentile(error, 50)),
        "p95": float(np.percentile(error, 95)),
        "max": float(np.max(error)),
    }


def project_xyz(
    x,
    y,
    z,
    intensity,
    ring,
    zero_range_m: float = ZERO_RANGE_M,
    az_bin_deg: float = AZIMUTH_BIN_DEG,
    el_bin_deg: float = ELEVATION_BIN_DEG,
):
    """Project valid points into azimuth × elevation bins. Every valid index is kept.

    Bins that contain more than one point stay as lists. Nothing is overwritten.
    The default step stays 0.2°. A caller may pass another step for a comparison;
    that does not change the project default.
    """
    finite, zero, valid, range_m = classify_ranges(x, y, z, zero_range_m)
    azimuth, elevation = xyz_angles_deg(x, y, z)
    n_cols = int(round(360.0 / az_bin_deg))
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size == 0:
        el_min = 0.0
        n_rows = 1
        row = np.zeros(x.shape, dtype=np.int32)
        col = np.zeros(x.shape, dtype=np.int32)
    else:
        el_valid = elevation[valid]
        el_min = float(np.floor(np.min(el_valid) / el_bin_deg) * el_bin_deg)
        el_max = float(np.ceil(np.max(el_valid) / el_bin_deg) * el_bin_deg)
        if el_max <= el_min:
            el_max = el_min + el_bin_deg
        n_rows = int(round((el_max - el_min) / el_bin_deg))
        n_rows = max(n_rows, 1)
        col = np.floor((azimuth - (-180.0)) / az_bin_deg).astype(np.int32)
        row = np.floor((elevation - el_min) / el_bin_deg).astype(np.int32)
        # The right and top edges fall on the exclusive bound. Keep them in the last bin.
        np.clip(col, 0, n_cols - 1, out=col)
        np.clip(row, 0, n_rows - 1, out=row)
    flat = row.astype(np.int64) * n_cols + col.astype(np.int64)
    indptr, indices, counts = _csr(flat[valid], valid_ids.astype(np.int64), n_rows * n_cols)
    ranges_for_indices = range_m[indices].astype(np.float32) if indices.size else np.zeros(0, np.float32)
    if intensity is None:
        intensity_for_indices = np.full(indices.shape, np.nan, dtype=np.float32)
    else:
        intensity_for_indices = intensity[indices].astype(np.float32) if indices.size else np.zeros(0, np.float32)
    if ring is None:
        ring_for_indices = np.full(indices.shape, -1, dtype=np.int32)
        ring_source = "absent"
    else:
        ring_for_indices = ring[indices].astype(np.int32) if indices.size else np.zeros(0, np.int32)
        ring_source = "point_field"
    occupied, multi_flag = _range_span_by_bin(indptr, counts, ranges_for_indices.astype(np.float64))
    collision_bins = int(np.count_nonzero(counts >= 2))
    multi_bins = int(np.count_nonzero(multi_flag))
    outside = valid & (np.abs(azimuth) > 20.0)
    outside_ids = np.flatnonzero(outside)
    unique_ids = np.unique(indices) if indices.size else np.zeros(0, dtype=np.int64)
    dropped = int(valid_ids.size - unique_ids.size)
    duplicated = int(indices.size - unique_ids.size)
    outside_recovered = int(np.isin(outside_ids, unique_ids).sum()) if outside_ids.size else 0
    error = quantization_error_m(
        x, y, z, valid, azimuth, elevation, row, col, az_bin_deg, el_bin_deg, el_min
    )
    points_in_collision = int(counts[counts >= 2].sum())
    spec = {
        "representation": REPRESENTATION_XYZ,
        "calibration": None,
        "calibration_note": (
            "Файла индивидуальной калибровки нет. Углы каналов из мануала номинальные "
            "и в эту сетку не подставляются."
        ),
        "return_type_field": None,
        "deskew": "not_applied",
        "ring_derived_from_xyz_after_deskew": False,
        "ring_source_on_points": ring_source,
        "permanent_pm20_crop": False,
        "azimuth_bin_deg": float(az_bin_deg),
        "elevation_bin_deg": float(el_bin_deg),
        "azimuth_limits_deg": [-180.0, 180.0],
        "elevation_min_deg": el_min,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "row0_is": "lowest_elevation",
        "png_row0_is": "highest_elevation",
        "multiple_return_gap_m": MULTIPLE_RETURN_GAP_M,
        "zero_range_m": zero_range_m,
    }
    loss = {
        "input_points": int(x.shape[0]),
        "nonfinite": int((~finite).sum()),
        "zero_range_missing": int(zero.sum()),
        "valid": int(valid_ids.size),
        "index_recovered": int(indices.size - duplicated),
        "index_dropped": dropped,
        "index_duplicated": duplicated,
        "occupied_bins": int(occupied.size),
        "missing_bins": int(counts.size - occupied.size),
        "collision_bins": collision_bins,
        "points_in_collision_bins": points_in_collision,
        "multiple_return_bins": multi_bins,
        "points_outside_abs_azimuth_20deg": int(outside.sum()),
        "points_outside_abs_azimuth_20deg_recovered": int(outside_recovered),
        "quantization_error_m": _error_summary(error),
        "note": (
            "Обратная проекция по индексам возвращает исходные точки. "
            "Ошибка квантования — отдельное сравнение с центром угловой ячейки, "
            "а не выброшенные точки. Все индексы ячейки сохранены. "
            "Несколько точек в ячейке не являются полем режима dual return."
        ),
    }
    grid = {
        "row": row,
        "col": col,
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "range_m": range_m,
        "valid": valid,
        "indptr": indptr,
        "indices": indices,
        "counts": counts,
        "range_for_indices": ranges_for_indices,
        "intensity_for_indices": intensity_for_indices,
        "ring_for_indices": ring_for_indices,
    }
    return spec, loss, grid


def ring_rows_usable(ring, valid) -> tuple[bool, str]:
    if ring is None:
        return False, "поля ring нет"
    if ring.dtype != np.uint16:
        return False, f"ring имеет dtype {ring.dtype}, для строк сетки нужен uint16"
    if valid.any() and int(ring[valid].max()) > 127:
        return False, "среди годных точек есть ring вне 0..127"
    if not valid.any():
        return False, "нет годных точек"
    return True, "ring скопирован из поля точки"


def project_ring_rows(azimuth, valid, ring):
    """Rows are the ring field. Columns are XYZ azimuth. Not a calibrated firing grid."""
    ok, reason = ring_rows_usable(ring, valid)
    n_cols = int(round(360.0 / AZIMUTH_BIN_DEG))
    if not ok:
        return None, reason
    row = ring.astype(np.int32)
    col = np.floor((azimuth - (-180.0)) / AZIMUTH_BIN_DEG).astype(np.int32)
    np.clip(col, 0, n_cols - 1, out=col)
    n_rows = 128
    flat = row.astype(np.int64) * n_cols + col.astype(np.int64)
    valid_ids = np.flatnonzero(valid)
    indptr, indices, counts = _csr(flat[valid], valid_ids.astype(np.int64), n_rows * n_cols)
    # Ring row must stay the field value for every stored point.
    if indices.size and not np.array_equal(row[indices], ring[indices].astype(np.int32)):
        raise RuntimeError("ring row diverged from the point field")
    spec = {
        "representation": REPRESENTATION_RING,
        "rows": "ring field 0..127",
        "columns": "xyz azimuth",
        "not": "factory firing grid",
        "nominal_manual_angles_applied": False,
        "ring_derived_from_xyz_after_deskew": False,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "azimuth_bin_deg": AZIMUTH_BIN_DEG,
        "reason": reason,
    }
    loss = {
        "valid": int(valid_ids.size),
        "index_recovered": int(indices.size),
        "index_dropped": int(valid_ids.size - np.unique(indices).size) if indices.size else int(valid_ids.size),
        "collision_bins": int(np.count_nonzero(counts >= 2)),
    }
    grid = {"indptr": indptr, "indices": indices, "counts": counts, "row": row, "col": col}
    return {"spec": spec, "loss": loss, "grid": grid}, reason


def bin_center_deg(internal_row: int, col: int, spec: dict) -> tuple[float, float]:
    azimuth = -180.0 + (col + 0.5) * float(spec["azimuth_bin_deg"])
    elevation = float(spec["elevation_min_deg"]) + (internal_row + 0.5) * float(spec["elevation_bin_deg"])
    return azimuth, elevation


def display_row_to_internal(display_row: int, n_rows: int) -> int:
    """PNG row 0 is the highest elevation bin."""
    return int(n_rows) - 1 - int(display_row)


def indices_in_bin(grid, n_cols: int, internal_row: int, col: int) -> np.ndarray:
    flat = int(internal_row) * int(n_cols) + int(col)
    start = int(grid["indptr"][flat])
    stop = int(grid["indptr"][flat + 1])
    return grid["indices"][start:stop]
