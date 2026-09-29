"""Orthographic and angular pictures of one cloud.

Spatial cells show the nearest return only. That reduction is labeled on the
picture. The angular index map, not these pixels, keeps every return.
"""

from __future__ import annotations

import zlib
import struct
from pathlib import Path

import numpy as np

DISPLAY_CELL_M = 0.25
MAX_AXIS_PIXELS = 1400


def _lerp(stops, t):
    stops = np.asarray(stops, dtype=np.float64)
    t = np.clip(t, 0.0, 1.0)
    pos = t * (len(stops) - 1)
    i0 = np.floor(pos).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, len(stops) - 1)
    f = (pos - i0)[..., None]
    return stops[i0] * (1.0 - f) + stops[i1] * f


RANGE_STOPS = np.array(
    [[30, 60, 140], [40, 160, 180], [80, 180, 80], [230, 190, 40], [200, 50, 40]],
    dtype=np.float64,
)
INTENSITY_STOPS = np.array([[20, 20, 28], [70, 70, 90], [240, 210, 120]], dtype=np.float64)


def colorize(values, vmin, vmax, stops):
    span = float(vmax) - float(vmin)
    if span <= 0.0:
        span = 1.0
    t = (values.astype(np.float64) - float(vmin)) / span
    rgb = _lerp(stops, t)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def finite_limits(values, valid, pad):
    if not np.any(valid):
        return (-1.0, 1.0)
    lo = float(np.min(values[valid]))
    hi = float(np.max(values[valid]))
    if hi <= lo:
        hi = lo + 1.0
    return (lo - pad, hi + pad)


def choose_cell(span_m: float, requested: float = DISPLAY_CELL_M) -> float:
    cell = float(requested)
    if span_m <= 0.0:
        return cell
    while span_m / cell > MAX_AXIS_PIXELS:
        cell *= 2.0
    return cell


def orthographic(
    h, v, display_values, sort_values, valid, h_name, v_name, color_name, vmin, vmax, stops
):
    """Equal-aspect image. Row 0 is the high end of the vertical axis.

    The painted value belongs to the point with the smallest sort_values in
    that cell (range, when the caller passes range). Other points stay unpainted.
    """
    h_lim = finite_limits(h, valid, 1.0)
    v_lim = finite_limits(v, valid, 1.0)
    cell = max(choose_cell(h_lim[1] - h_lim[0]), choose_cell(v_lim[1] - v_lim[0]))
    nx = max(int(np.ceil((h_lim[1] - h_lim[0]) / cell)), 1)
    ny = max(int(np.ceil((v_lim[1] - v_lim[0]) / cell)), 1)
    image = np.zeros((ny, nx, 3), dtype=np.uint8)
    image[:] = (18, 20, 24)
    counts = np.zeros((ny, nx), dtype=np.int32)
    if np.any(valid):
        hh = h[valid]
        vv = v[valid]
        shown = display_values[valid]
        sort_key = sort_values[valid]
        col = np.floor((hh - h_lim[0]) / cell).astype(np.int32)
        row_from_bottom = np.floor((vv - v_lim[0]) / cell).astype(np.int32)
        np.clip(col, 0, nx - 1, out=col)
        np.clip(row_from_bottom, 0, ny - 1, out=row_from_bottom)
        display_row = ny - 1 - row_from_bottom
        cell_id = display_row.astype(np.int64) * nx + col.astype(np.int64)
        order = np.lexsort((sort_key, cell_id))
        sorted_cells = cell_id[order]
        starts = np.flatnonzero(np.r_[True, sorted_cells[1:] != sorted_cells[:-1]])
        nearest = order[starts]
        rows = display_row[nearest]
        cols = col[nearest]
        image[rows, cols] = colorize(shown[nearest], vmin, vmax, stops)
        flat_counts = np.bincount(cell_id, minlength=ny * nx)
        counts = flat_counts.reshape(ny, nx)
    spec = {
        "h_axis": h_name,
        "v_axis": v_name,
        "h_lim_m": [h_lim[0], h_lim[0] + nx * cell],
        "v_lim_m": [v_lim[0], v_lim[0] + ny * cell],
        "cell_m": cell,
        "equal_aspect": True,
        "color": color_name,
        "vmin": float(vmin),
        "vmax": float(vmax),
        "display_reduction": "nearest point in the spatial cell",
        "spatial_cells_with_multiple_points": int(np.count_nonzero(counts >= 2)),
        "png_row0": f"high {v_name}",
        "png_col0": f"low {h_name}",
    }
    return image, spec


def angular_image(grid, spec, values_for_indices, vmin, vmax, stops, reduction: str, flip_vertical: bool):
    """One pixel per angular bin. XYZ images are flipped so high elevation is on top."""
    n_rows = int(spec["n_rows"])
    n_cols = int(spec["n_cols"])
    image = np.zeros((n_rows, n_cols, 3), dtype=np.uint8)
    image[:] = (18, 20, 24)
    indptr = grid["indptr"]
    counts = grid["counts"]
    occupied = np.flatnonzero(counts > 0)
    if occupied.size:
        if reduction == "count":
            shown = counts[occupied].astype(np.float64)
            scale_max = float(max(float(shown.max()), 1.0))
            colors = colorize(shown, 0.0, scale_max, stops)
            vmin, vmax = 0.0, scale_max
        elif reduction == "min":
            starts = indptr[occupied]
            shown = np.minimum.reduceat(values_for_indices.astype(np.float64), starts)
            colors = colorize(shown, vmin, vmax, stops)
        elif reduction == "nearest_intensity":
            shown = _nearest_value(indptr, counts, grid["range_for_indices"], values_for_indices)[occupied]
            colors = colorize(shown, vmin, vmax, stops)
        else:
            raise ValueError(reduction)
        rows = occupied // n_cols
        cols = occupied % n_cols
        image[rows, cols] = colors
    png = image[::-1] if flip_vertical else image
    view = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "png_row0_is": "highest_elevation" if flip_vertical else spec.get("png_row0_is", "row 0"),
        "reduction": reduction,
        "vmin": None if vmin is None else float(vmin),
        "vmax": None if vmax is None else float(vmax),
        "azimuth_bin_deg": spec.get("azimuth_bin_deg"),
        "elevation_bin_deg": spec.get("elevation_bin_deg"),
        "elevation_min_deg": spec.get("elevation_min_deg"),
        "azimuth_limits_deg": spec.get("azimuth_limits_deg"),
    }
    return png, view


def _nearest_value(indptr, counts, ranges_for_indices, values_for_indices):
    n_bins = counts.shape[0]
    out = np.full(n_bins, np.nan, dtype=np.float64)
    occupied = np.flatnonzero(counts > 0)
    if occupied.size == 0:
        return out
    ranges = ranges_for_indices.astype(np.float64)
    values = values_for_indices.astype(np.float64)
    # Indices are packed in bin order. The minimum range in each occupied slice
    # selects the value. reduceat gives the min; argmin needs a short scan only
    # when a bin has several points. Vectorized path: repeat bin ids.
    bin_of_point = np.repeat(np.arange(n_bins, dtype=np.int64), counts)
    order = np.lexsort((ranges, bin_of_point))
    sorted_bins = bin_of_point[order]
    starts = np.flatnonzero(np.r_[True, sorted_bins[1:] != sorted_bins[:-1]])
    winners = order[starts]
    out[sorted_bins[starts]] = values[winners]
    return out


def write_png(path: Path, rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected uint8 HxWx3")
    height, width, _ = rgb.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + np.ascontiguousarray(rgb[i]).tobytes() for i in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg(path: Path, png_name: str, lines: list[str], width: int, height: int) -> None:
    text_h = 20 * (len(lines) + 1)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{text_h + height}" '
        f'viewBox="0 0 {width} {text_h + height}">',
        '<rect width="100%" height="100%" fill="#12141a"/>',
    ]
    for i, line in enumerate(lines):
        parts.append(
            f'<text x="8" y="{22 + 20 * i}" fill="#e8e6df" '
            f'font-family="Consolas, DejaVu Sans Mono, sans-serif" font-size="14">'
            f"{xml_escape(line)}</text>"
        )
    parts.append(
        f'<image x="0" y="{text_h}" width="{width}" height="{height}" href="{xml_escape(png_name)}"/>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
