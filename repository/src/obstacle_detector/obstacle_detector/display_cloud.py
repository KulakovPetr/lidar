"""Display sample. This does not change the cloud the detector receives."""

from __future__ import annotations

import numpy as np


def _index_array(item: dict) -> np.ndarray | None:
    raw = item.get("source_indices") or []
    if len(raw) == 0:
        return None
    return np.asarray(raw, dtype=np.int64)


def highlight_rows(record: dict) -> np.ndarray:
    """Every accepted cluster row. Not the first cluster only.

    Intrusion cards do not repeat source_indices. Those rows live on the
    matching candidate. Display uses that list and does not change detection.
    """
    chunks = []
    for item in record.get("conditional_intrusions") or []:
        array = _index_array(item)
        if array is not None:
            chunks.append(array)
    if not chunks:
        accepted = {
            item.get("candidate_id")
            for item in record.get("conditional_intrusions") or []
        }
        for item in record.get("candidates") or []:
            if item.get("candidate_id") not in accepted:
                continue
            array = _index_array(item)
            if array is not None:
                chunks.append(array)
    if not chunks:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(chunks))


def sample_display(xyz: np.ndarray, highlight: np.ndarray, background_limit: int = 3500):
    """Stride the background. Keep every highlight row.

    Returns the chosen rows and a float label, 1 for an accepted point.
    """
    points = np.asarray(xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must be N x 3")
    total = int(points.shape[0])
    chosen_highlight = np.asarray(highlight, dtype=np.int64).reshape(-1)
    if chosen_highlight.size:
        chosen_highlight = np.unique(chosen_highlight)
        chosen_highlight = chosen_highlight[(chosen_highlight >= 0) & (chosen_highlight < total)]
    else:
        chosen_highlight = np.zeros(0, dtype=np.int64)
    if total == 0:
        return points.reshape(0, 3), np.zeros(0, dtype=np.float32)
    if total <= background_limit:
        background = np.arange(total, dtype=np.int64)
    else:
        background = np.linspace(0, total - 1, int(background_limit), dtype=np.int64)
    if chosen_highlight.size:
        rows = np.unique(np.concatenate([background, chosen_highlight]))
    else:
        rows = np.unique(background)
    labels = np.zeros(rows.shape[0], dtype=np.float32)
    if chosen_highlight.size:
        labels[np.isin(rows, chosen_highlight)] = 1.0
    return points[rows], labels
