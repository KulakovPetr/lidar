"""Exact XYZ groups for geometry votes.

Original records stay in place. A repeated coordinate is one geometric vote
and keeps every source index. Different coordinates are not voxel-merged.
"""

from __future__ import annotations

import numpy as np


def _segment_span(values, change, sizes):
    span = np.zeros(change.size, dtype=np.float64)
    if values is None or change.size == 0:
        return span
    finite = np.isfinite(values)
    multi = sizes > 1
    if not np.any(multi):
        return span
    filled_hi = np.where(finite, values, -np.inf)
    filled_lo = np.where(finite, values, np.inf)
    width = np.maximum.reduceat(filled_hi, change) - np.minimum.reduceat(filled_lo, change)
    has_finite = np.add.reduceat(finite.astype(np.int8), change) > 0
    ok = multi & has_finite & np.isfinite(width)
    span[ok] = width[ok]
    span[multi & ~has_finite] = np.nan
    return span


class GroupMembers:
    """Flat source indices. Indexing a row returns that group's indices, not a Python object per group."""

    def __init__(self, member_index: np.ndarray, member_offset: np.ndarray) -> None:
        self.member_index = member_index
        self.member_offset = member_offset

    def __len__(self) -> int:
        return int(self.member_offset.size - 1)

    def __iter__(self):
        for row in range(len(self)):
            yield self[row]

    def __getitem__(self, row):
        if isinstance(row, np.ndarray):
            chosen = np.flatnonzero(row) if row.dtype == bool else np.asarray(row, dtype=np.int64)
            return [self[int(index)] for index in chosen]
        start = int(self.member_offset[int(row)])
        stop = int(self.member_offset[int(row) + 1])
        return self.member_index[start:stop]


def valid_mask(x, y, z) -> np.ndarray:
    """Finite points that are not the empty return XYZ=(0,0,0)."""
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    origin = (x == 0) & (y == 0) & (z == 0)
    return finite & ~origin


def group_exact_xyz(x, y, z, intensity, ring, timestamp, valid):
    """Group valid records by exact float32 XYZ.

    Returns one row per distinct coordinate. Members are one flat index array
    plus offsets. Coordinates stay the original float32 values.
    """
    x32 = np.asarray(x, dtype=np.float32)
    y32 = np.asarray(y, dtype=np.float32)
    z32 = np.asarray(z, dtype=np.float32)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        empty = np.zeros(0, dtype=np.float32)
        return {
            "x": empty,
            "y": empty,
            "z": empty,
            "indices": GroupMembers(np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64)),
            "member_index": np.zeros(0, dtype=np.int64),
            "member_offset": np.zeros(1, dtype=np.int64),
            "size": np.zeros(0, dtype=np.int32),
            "intensity_span": np.zeros(0, dtype=np.float32),
            "timestamp_span": np.zeros(0, dtype=np.float64),
            "ring_nunique": np.zeros(0, dtype=np.int32),
        }
    packed = np.ascontiguousarray(np.stack([x32[idx], y32[idx], z32[idx]], axis=1))
    keys = packed.view(np.dtype((np.void, packed.dtype.itemsize * 3))).ravel()
    order = np.argsort(keys, kind="mergesort")
    keys_s = keys[order]
    idx_s = idx[order]
    change = np.flatnonzero(np.r_[True, keys_s[1:] != keys_s[:-1]])
    sizes = np.diff(np.r_[change, keys_s.size]).astype(np.int32)
    first_index = idx_s[change].astype(np.int64, copy=False)
    groups_x = packed[order[change], 0]
    groups_y = packed[order[change], 1]
    groups_z = packed[order[change], 2]
    member_offset = np.r_[change, np.int64(keys_s.size)].astype(np.int64, copy=False)
    member_index = np.asarray(idx_s, dtype=np.int64)
    if intensity is None:
        intensity_span = np.full(change.size, np.nan, dtype=np.float32)
    else:
        intensity_span = _segment_span(np.asarray(intensity)[idx_s], change, sizes).astype(np.float32)
    if timestamp is None:
        timestamp_span = np.full(change.size, np.nan, dtype=np.float64)
    else:
        timestamp_span = _segment_span(np.asarray(timestamp, dtype=np.float64)[idx_s], change, sizes).astype(np.float64)
    if ring is None:
        # Absent ring is unknown. A value of 1 would be a fictitious measurement.
        ring_nunique = np.full(change.size, -1, dtype=np.int32)
    else:
        ring_nunique = np.ones(change.size, dtype=np.int32)
        rings = np.asarray(ring)[idx_s]
        for k in np.flatnonzero(sizes > 1):
            start = int(change[int(k)])
            ring_nunique[int(k)] = int(np.unique(rings[start : start + int(sizes[int(k)])]).size)
    return {
        "x": groups_x,
        "y": groups_y,
        "z": groups_z,
        "indices": GroupMembers(member_index, member_offset),
        "member_index": member_index,
        "member_offset": member_offset,
        "first_index": first_index,
        "size": sizes,
        "intensity_span": intensity_span,
        "timestamp_span": timestamp_span,
        "ring_nunique": ring_nunique,
        "intensity_observed": intensity is not None,
        "ring_observed": ring is not None,
        "timestamp_observed": timestamp is not None,
    }


def repeat_summary(x, y, z, intensity, ring, timestamp, valid) -> dict:
    groups = group_exact_xyz(x, y, z, intensity, ring, timestamp, valid)
    sizes = groups["size"]
    hist = {}
    if sizes.size:
        uniq, counts = np.unique(sizes, return_counts=True)
        hist = {str(int(k)): int(v) for k, v in zip(uniq, counts)}
    multi = sizes > 1
    index_diffs = []
    for members in groups["indices"][multi]:
        ordered = np.sort(np.asarray(members, dtype=np.int64))
        index_diffs.append(np.diff(ordered))
    if index_diffs:
        diffs = np.concatenate(index_diffs)
        values, counts = np.unique(diffs, return_counts=True)
        order = np.argsort(counts)[::-1][:12]
        top_diffs = [
            {"index_diff": int(values[i]), "count": int(counts[i])} for i in order
        ]
    else:
        top_diffs = []
    n = int(np.asarray(x).shape[0])
    pair = _pairs_i_plus_128(x, y, z, ring, timestamp, valid, n)
    xyz_ring = None
    if ring is not None:
        idx = np.flatnonzero(valid)
        ring32 = np.asarray(ring, dtype=np.int32)[idx]
        packed_ring = np.ascontiguousarray(
            np.stack(
                [
                    np.asarray(x, dtype=np.float32)[idx],
                    np.asarray(y, dtype=np.float32)[idx],
                    np.asarray(z, dtype=np.float32)[idx],
                    ring32.view(np.float32),
                ],
                axis=1,
            )
        )
        void = packed_ring.view(np.dtype((np.void, packed_ring.dtype.itemsize * 4))).ravel()
        xyz_ring = int(np.unique(void).size)
    return {
        "valid_records": int(np.count_nonzero(valid)),
        "unique_exact_xyz": int(sizes.size),
        "unique_exact_xyz_ring": xyz_ring,
        "group_size_histogram": hist,
        "groups_with_repeats": int(np.count_nonzero(multi)),
        "intensity_observed": intensity is not None,
        "ring_observed": ring is not None,
        "timestamp_observed": timestamp is not None,
        "intensity_span_nonzero_groups": None if intensity is None else int(np.count_nonzero(groups["intensity_span"] > 0)),
        "timestamp_span_nonzero_groups": None if timestamp is None else int(np.count_nonzero(groups["timestamp_span"] > 0)),
        "timestamp_span_max": None if timestamp is None or not sizes.size else float(groups["timestamp_span"].max()),
        "intensity_span_max": None if intensity is None or not sizes.size else float(groups["intensity_span"].max()),
        "groups_with_several_rings": None if ring is None else int(np.count_nonzero(groups["ring_nunique"] > 1)),
        "top_index_diffs": top_diffs,
        "pairs_i_iplus128": pair,
        "groups": groups,
    }


def _pairs_i_plus_128(x, y, z, ring, timestamp, valid, n: int) -> dict:
    if n <= 128:
        return {"pairs": 0}
    i = np.arange(0, n - 128, dtype=np.int64)
    both = valid[i] & valid[i + 128]
    same_xyz = (x[i] == x[i + 128]) & (y[i] == y[i + 128]) & (z[i] == z[i + 128])
    same_ring = np.ones(i.shape, dtype=bool) if ring is None else np.asarray(ring)[i] == np.asarray(ring)[i + 128]
    if timestamp is None:
        same_time = np.ones(i.shape, dtype=bool)
    else:
        ts = np.asarray(timestamp, dtype=np.float64)
        same_time = ts[i] == ts[i + 128]
    both_same_xyz = both & same_xyz
    both_same_ring_time_diff_xyz = both & same_ring & same_time & ~same_xyz
    return {
        "both_valid_pairs": int(both.sum()),
        "both_valid_same_xyz": int(both_same_xyz.sum()),
        "both_valid_same_ring_and_time_different_xyz": int(both_same_ring_time_diff_xyz.sum()),
    }


def support_count(group_sizes: np.ndarray) -> int:
    """One vote per geometric point, not per repeated record."""
    return int(group_sizes.size)
