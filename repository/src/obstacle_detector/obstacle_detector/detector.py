"""Single-frame geometric candidates inside a conditional corridor.

Exclusion rules are explicit. A missing candidate list is not a clearance.
Insertion labels are not an input.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

from obstacle_detector.corridor import working_to_sensor


def detector_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "detector.yaml"


def load_detector_config(path: Path | None = None) -> dict:
    raw = yaml.safe_load((path or detector_config_path()).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("detector config must be a mapping")
    if "frames" in raw or any(isinstance(key, str) and "/" in key for key in raw):
        raise ValueError("per-frame detector parameters are not allowed")
    return raw


RULES = (
    "Вне условного профиля или при неопределённой геометрии коридора точка не становится кандидатом.",
    "Линия пути исключается только как трёхмерная область: продольная поддержка, собственная ширина и наблюдаемая высота с явным допуском.",
    "Если высотный профиль неизвестен или выборка линии не поддержана, совпадение по s и l отмечается как возможное объяснение и не удаляет точку.",
    "Точка опорной поверхности исключается только если остаток подгонки меньше половины высоты 0,10 м. Иначе она остаётся в неоднозначном слое и не удаляется.",
    "Расширение неоднозначной полосы до высоты 0,10 м не заменяется более узким порогом по результату вставок. Диагностический массив не является тревогой.",
    "Платформа, стена и кабель не являются причинами исключения.",
    "Редкий фрагмент хранится отдельно и не отбрасывается из-за малого числа точек.",
    "Попадание в пул точек не означает, что объект отделён от фона. Отсутствие кандидатов не означает, что путь свободен.",
)


def path_structure_masks(s, l, h, line_samples, section_half_m: float, height_tolerance_m: float):
    """Split a lateral footprint into a removable 3D volume and an unmarked explanation.

    `excluded` is the supported volume around an observed height.
    `possible` is the same footprint when height support is missing. It is not removed.
    """
    s = np.asarray(s, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    excluded = np.zeros(s.shape, dtype=bool)
    possible = np.zeros(s.shape, dtype=bool)
    half_s = float(section_half_m)
    tolerance = float(height_tolerance_m)
    for sample in line_samples or []:
        width = sample.get("width_m")
        if width is None or sample.get("s_m") is None or sample.get("l_m") is None:
            continue
        foot = (np.abs(s - float(sample["s_m"])) <= half_s) & (np.abs(l - float(sample["l_m"])) <= 0.5 * float(width))
        observed_h = sample.get("h_m")
        supported = sample.get("height_support") == "supported" and observed_h is not None and np.isfinite(float(observed_h))
        if not supported:
            possible |= foot
            continue
        excluded |= foot & (np.abs(h - float(observed_h)) <= tolerance)
    possible &= ~excluded
    return excluded, possible


def cluster_labels(s, l, h, gap_m: float) -> np.ndarray:
    """26-connected cells of size gap_m. Empty input stays empty."""
    s = np.asarray(s, dtype=np.float64)
    n = int(s.size)
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    cells = np.floor(np.column_stack([s, np.asarray(l), np.asarray(h)]) / float(gap_m)).astype(np.int64)
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    parent = np.arange(unique.shape[0], dtype=np.int32)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    lookup = {tuple(int(value) for value in row): index for index, row in enumerate(unique)}
    for index, row in enumerate(unique):
        x0, y0, z0 = (int(value) for value in row)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    other = lookup.get((x0 + dx, y0 + dy, z0 + dz))
                    if other is None:
                        continue
                    left, right = find(index), find(other)
                    if left != right:
                        parent[right] = left
    roots = np.array([find(index) for index in range(unique.shape[0])], dtype=np.int32)
    _, compact = np.unique(roots, return_inverse=True)
    return compact[inverse].astype(np.int32)


def _native_components():
    import ctypes
    from pathlib import Path

    path = Path("/usr/local/lib/libobstacle_components.so")
    if not path.is_file():
        return None
    lib = ctypes.CDLL(str(path))
    lib.components_from_pairs.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.components_from_pairs.restype = ctypes.c_int
    lib.labels_fixed_radius.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int64),
    ]
    lib.labels_fixed_radius.restype = ctypes.c_int
    lib.labels_max_radius.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int64),
    ]
    lib.labels_max_radius.restype = ctypes.c_int
    return lib


_NATIVE_COMPONENTS = None
_NATIVE_TRIED = False


def _components_from_pairs(n: int, pairs: np.ndarray) -> np.ndarray | None:
    import ctypes

    global _NATIVE_COMPONENTS, _NATIVE_TRIED
    if not _NATIVE_TRIED:
        _NATIVE_TRIED = True
        try:
            _NATIVE_COMPONENTS = _native_components()
        except OSError:
            _NATIVE_COMPONENTS = None
    lib = _NATIVE_COMPONENTS
    if lib is None or n > int(np.iinfo(np.int32).max):
        return None
    labels = np.empty(n, dtype=np.int32)
    raw = np.ascontiguousarray(pairs, dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raw = np.zeros((0, 2), dtype=np.int64)
    code = lib.components_from_pairs(
        ctypes.c_int64(int(n)),
        ctypes.c_int64(int(raw.shape[0])),
        raw.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if code != 0:
        return None
    return labels


def _native_library():
    global _NATIVE_COMPONENTS, _NATIVE_TRIED
    if not _NATIVE_TRIED:
        _NATIVE_TRIED = True
        try:
            _NATIVE_COMPONENTS = _native_components()
        except OSError:
            _NATIVE_COMPONENTS = None
    return _NATIVE_COMPONENTS


def _labels_from_grid(points: np.ndarray, radii: np.ndarray) -> np.ndarray | None:
    """Radius components. The edge list is not allocated."""
    import ctypes

    lib = _native_library()
    n = int(points.shape[0])
    if lib is None or n > int(np.iinfo(np.int32).max):
        return None
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    xyz = np.ascontiguousarray(points, dtype=np.float64)
    radius = np.ascontiguousarray(radii, dtype=np.float64)
    labels = np.empty(n, dtype=np.int32)
    checks = ctypes.c_int64(0)
    started = time.perf_counter()
    code = lib.labels_max_radius(
        ctypes.c_int64(n),
        xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        radius.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(checks),
    )
    if code != 0:
        return None
    _labels_from_grid.last = {
        "points": n,
        "edges": None,
        "edge_array_materialised": False,
        "pair_checks": int(checks.value),
        "components": "cpu_grid",
        "components_s": time.perf_counter() - started,
    }
    return labels


def _labels_fixed_radius(points: np.ndarray, limit: float) -> np.ndarray | None:
    import ctypes

    lib = _native_library()
    n = int(points.shape[0])
    if lib is None or n > int(np.iinfo(np.int32).max):
        return None
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    xyz = np.ascontiguousarray(points, dtype=np.float64)
    labels = np.empty(n, dtype=np.int32)
    checks = ctypes.c_int64(0)
    code = lib.labels_fixed_radius(
        ctypes.c_int64(n),
        xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(float(limit)),
        labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(checks),
    )
    if code != 0:
        return None
    return labels


def _labels_from_kdtree(points: np.ndarray, limit: float):
    """Connected components of pairs within `limit`. None if SciPy is not installed.

    The sparse graph stores only pairs inside the radius. Non-edges are not materialised.
    """
    try:
        from scipy.spatial import cKDTree
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError:
        return None
    n = int(points.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    try:
        pairs = cKDTree(points).query_pairs(float(limit), output_type="ndarray")
    except TypeError:
        found = cKDTree(points).query_pairs(float(limit))
        pairs = np.array(list(found), dtype=np.int64).reshape(-1, 2) if found else np.zeros((0, 2), dtype=np.int64)
    if int(pairs.size) == 0:
        _labels_from_kdtree.last = {"points": n, "edges": 0, "components": "none"}
        return np.arange(n, dtype=np.int32)
    native_started = time.perf_counter()
    labels = _components_from_pairs(n, pairs)
    if labels is not None:
        _labels_from_kdtree.last = {
            "points": n,
            "edges": int(pairs.shape[0]),
            "components": "cpu_native",
            "components_s": time.perf_counter() - native_started,
        }
        return labels
    graph = coo_matrix(
        (np.ones(int(pairs.shape[0]), dtype=np.uint8), (pairs[:, 0], pairs[:, 1])),
        shape=(n, n),
    )
    _count, labels = connected_components(graph, directed=False, return_labels=True)
    _labels_from_kdtree.last = {
        "points": n,
        "edges": int(pairs.shape[0]),
        "components": "scipy",
        "components_s": time.perf_counter() - native_started,
    }
    return np.asarray(labels, dtype=np.int32)


def link_radius_per_point(points: np.ndarray, link_distance_m: float) -> np.ndarray:
    """Near points use the configured radius. At 70 m and beyond the radius is 0.23 m.

    A pair links when distance <= max(radius_i, radius_j). The boundary is included.
    The larger radius belongs to the farther point, so a 0.215 m ring step on a far
    face still links, and a pair that crosses 70 m links when that same max covers it.
    """
    sensor_range = np.sqrt(np.einsum("ij,ij->i", points, points))
    radius = np.full(int(points.shape[0]), float(link_distance_m), dtype=np.float64)
    radius[sensor_range >= 70.0] = max(float(link_distance_m), 0.23)
    return radius


def cluster_labels_split_halves(points: np.ndarray, link_distance_m: float) -> np.ndarray | None:
    """Previous behaviour: points nearer and farther than 70 m are separate graphs.

    Kept so the grid can be compared with that partition. It is not the production rule.
    """
    points = np.asarray(points, dtype=np.float64)
    n = int(points.shape[0])
    limit = float(link_distance_m)
    sensor_range = np.sqrt(np.einsum("ij,ij->i", points, points))
    far = sensor_range >= 70.0
    labels = np.empty(n, dtype=np.int32)
    next_id = 0
    for mask, radius in ((~far, limit), (far, max(limit, 0.23))):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        sub = _labels_fixed_radius(points[idx], radius)
        if sub is None:
            return None
        labels[idx] = sub + next_id
        next_id += int(sub.max()) + 1 if sub.size else 0
    if n and next_id == 0:
        return np.zeros(n, dtype=np.int32)
    _roots, compact = np.unique(labels, return_inverse=True)
    return compact.astype(np.int32)


def cluster_labels_by_distance(s, l, h, link_distance_m: float, cell_size_m: float) -> np.ndarray:
    """Link original points. A cell is not a link; coordinates are not replaced.

    Production uses the compiled grid. `cell_size_m` is only the Python fallback.
    """
    s = np.asarray(s, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    n = int(s.size)
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    cell = float(cell_size_m)
    limit = float(link_distance_m)
    if cell <= 0.0 or limit < 0.0:
        raise ValueError("cell size must be positive and link distance must be non-negative")
    points = np.column_stack([s, l, h])
    grid_labels = _labels_from_grid(points, link_radius_per_point(points, limit))
    if grid_labels is not None:
        return grid_labels
    keys = np.floor(points / cell).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(map(tuple, keys)):
        buckets.setdefault(key, []).append(index)
    parent = np.arange(n, dtype=np.int32)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def unite(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    limit2 = limit * limit
    reach = int(np.ceil(limit / cell - 1e-12))
    progress_path = os.environ.get("OBSTACLE_CLUSTER_PROGRESS") or ""
    progress_at = time.perf_counter()
    pair_tests = 0
    largest_bucket = max((len(members) for members in buckets.values()), default=0)
    link_started = progress_at

    def write_progress(force: bool = False) -> None:
        nonlocal progress_at
        if not progress_path:
            return
        now = time.perf_counter()
        if not force and now - progress_at < 1.0:
            return
        progress_at = now
        Path(progress_path).write_text(
            json.dumps(
                {
                    "pool_points": n,
                    "buckets": len(buckets),
                    "largest_bucket": largest_bucket,
                    "pair_tests": pair_tests,
                    "link_elapsed_s": now - link_started,
                }
            ),
            encoding="utf-8",
        )

    write_progress(force=True)

    def link_sets(left_ids, right_ids, same_set: bool) -> None:
        nonlocal pair_tests
        left_ids = np.asarray(left_ids, dtype=np.int64)
        right_ids = np.asarray(right_ids, dtype=np.int64)
        if left_ids.size == 0 or right_ids.size == 0:
            return
        block = 48
        for left_start in range(0, int(left_ids.size), block):
            left_block = left_ids[left_start : left_start + block]
            left_points = points[left_block]
            for right_start in range(0, int(right_ids.size), block):
                right_block = right_ids[right_start : right_start + block]
                diff = left_points[:, None, :] - points[right_block][None, :, :]
                close = np.einsum("ijk,ijk->ij", diff, diff) <= limit2
                pairs_i, pairs_j = np.nonzero(close)
                pair_tests += int(left_block.size * right_block.size)
                for row, col in zip(pairs_i.tolist(), pairs_j.tolist()):
                    i = int(left_block[row])
                    j = int(right_block[col])
                    if i == j or (same_set and j <= i):
                        continue
                    unite(i, j)
            write_progress()

    for key, members in buckets.items():
        link_sets(members, members, True)
        x0, y0, z0 = key
        for dx in range(0, reach + 1):
            for dy in range(-reach, reach + 1):
                for dz in range(-reach, reach + 1):
                    if dx == 0 and (dy < 0 or (dy == 0 and dz <= 0)):
                        continue
                    other = buckets.get((x0 + dx, y0 + dy, z0 + dz))
                    if other is not None:
                        link_sets(members, other, False)
    write_progress(force=True)
    roots = np.array([find(index) for index in range(n)], dtype=np.int32)
    _, compact = np.unique(roots, return_inverse=True)
    return compact.astype(np.int32)


def _local_config(config: dict) -> dict:
    return {
        "radii": [float(value) for value in config["local_radius_m"]],
        "step": float(config["local_anchor_step_m"]),
        "min_support": int(config["local_min_support"]),
        "lower_fraction": float(config["local_lower_fraction"]),
        "level_gap": float(config["local_level_gap_m"]),
        "unstable_ratio": float(config["local_unstable_ratio"]),
        "fit_min": int(config["local_fit_min"]),
        "holdout_min": int(config["local_holdout_min"]),
        "min_span_m": float(config["link_distance_m"]["value"]),
        "fit": "baseline",
        # tan(30°). Working assumption that h is vertical. Not a calibration.
        "max_abs_slope": 0.57735026919,
        "ransac_iterations": 40,
        "ransac_seed": 4,
    }


def _triangle_area(s, l):
    return 0.5 * abs(s[0] * (l[1] - l[2]) + s[1] * (l[2] - l[0]) + s[2] * (l[0] - l[1]))


def _ransac_local_plane(idx, s, l, h, settings):
    """One plane from a neighborhood. Insertion labels are not an input.

    Parameters are the existing level gap, link distance and a fixed tilt limit.
    They were not chosen by counting insertion points.
    """
    failed = {
        "usable": False,
        "plane_asl": None,
        "fit_residual_p95_m": None,
        "holdout_residual_p95_m": None,
        "reason": "insufficient_local_support",
        "explained_fraction": 0.0,
        "unexplained_count": int(idx.size),
        "support_s_m": [None, None],
        "support_l_m": [None, None],
        "support_unique_xyz": 0,
        "fit_rows": [],
        "hold_rows": [],
        "normal_slope": None,
    }
    if idx.size < settings["min_support"]:
        failed["reason"] = "недостаточно точек для локальной плоскости"
        return failed
    rng = np.random.default_rng(settings["ransac_seed"])
    best_inliers = None
    best_count = 0
    inlier_m = settings["level_gap"]
    for _ in range(settings["ransac_iterations"]):
        pick = rng.choice(idx.size, 3, replace=False)
        tri = idx[pick]
        if _triangle_area(s[tri], l[tri]) < 1e-6:
            continue
        design = np.column_stack([s[tri], l[tri], np.ones(3)])
        coef, *_ = np.linalg.lstsq(design, h[tri], rcond=None)
        if float(np.hypot(coef[0], coef[1])) > settings["max_abs_slope"]:
            continue
        residual = np.abs(h[idx] - (coef[0] * s[idx] + coef[1] * l[idx] + coef[2]))
        inliers = residual <= inlier_m
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < settings["min_support"]:
        return failed
    support = idx[best_inliers]
    span_s = float(np.max(s[support]) - np.min(s[support]))
    span_l = float(np.max(l[support]) - np.min(l[support]))
    if min(span_s, span_l) < settings["min_span_m"]:
        failed["reason"] = "узкая линия не подтверждает плоскость"
        failed["explained_fraction"] = best_count / float(idx.size)
        return failed
    order = np.argsort(s[support])
    support = support[order]
    mid = support.size // 2
    fit_idx = support[:mid]
    hold_idx = support[mid:]
    if fit_idx.size < settings["fit_min"] or hold_idx.size < settings["holdout_min"]:
        failed["reason"] = "недостаточно отложенных точек для проверки плоскости"
        return failed
    design = np.column_stack([s[fit_idx], l[fit_idx], np.ones(fit_idx.size)])
    coef, *_ = np.linalg.lstsq(design, h[fit_idx], rcond=None)
    if float(np.hypot(coef[0], coef[1])) > settings["max_abs_slope"]:
        failed["reason"] = "вертикальная поверхность не является опорой h(s,l)"
        return failed
    fit_res = np.abs(h[fit_idx] - design @ coef)
    hold_res = np.abs(h[hold_idx] - (coef[0] * s[hold_idx] + coef[1] * l[hold_idx] + coef[2]))
    fit_p95 = float(np.quantile(fit_res, 0.95))
    hold_p95 = float(np.quantile(hold_res, 0.95))
    unexplained = idx[~best_inliers]
    result = {
        "usable": False,
        "plane_asl": [float(coef[0]), float(coef[1]), float(coef[2])],
        "fit_residual_p95_m": fit_p95,
        "holdout_residual_p95_m": hold_p95,
        "reason": "",
        "explained_fraction": best_count / float(idx.size),
        "unexplained_count": int(unexplained.size),
        "support_s_m": [float(np.min(s[support])), float(np.max(s[support]))],
        "support_l_m": [float(np.min(l[support])), float(np.max(l[support]))],
        "support_unique_xyz": int(support.size),
        "fit_rows": [int(value) for value in fit_idx.tolist()],
        "hold_rows": [int(value) for value in hold_idx.tolist()],
        "normal_slope": float(np.hypot(coef[0], coef[1])),
        "inlier_residual_is_not_accuracy": True,
    }
    if hold_p95 > settings["level_gap"]:
        result["reason"] = "unknown"
        return result
    rest = idx[~best_inliers]
    if rest.size >= settings["min_support"]:
        other = _ransac_consensus(rest, s, l, h, settings, iterations=20)
        if other is not None:
            other_h = other[0] * float(np.median(s[idx])) + other[1] * float(np.median(l[idx])) + other[2]
            this_h = coef[0] * float(np.median(s[idx])) + coef[1] * float(np.median(l[idx])) + coef[2]
            if abs(other_h - this_h) > settings["level_gap"]:
                result["reason"] = "несколько опор, выбор не сделан"
                return result
    result["usable"] = True
    return result


def _ransac_consensus(idx, s, l, h, settings, iterations):
    rng = np.random.default_rng(settings["ransac_seed"] + 1)
    best = None
    best_count = 0
    for _ in range(iterations):
        pick = rng.choice(idx.size, 3, replace=False)
        tri = idx[pick]
        if _triangle_area(s[tri], l[tri]) < 1e-6:
            continue
        design = np.column_stack([s[tri], l[tri], np.ones(3)])
        coef, *_ = np.linalg.lstsq(design, h[tri], rcond=None)
        if float(np.hypot(coef[0], coef[1])) > settings["max_abs_slope"]:
            continue
        residual = np.abs(h[idx] - (coef[0] * s[idx] + coef[1] * l[idx] + coef[2]))
        count = int(np.sum(residual <= settings["level_gap"]))
        if count > best_count:
            best_count = count
            best = coef
    if best is None or best_count < settings["min_support"]:
        return None
    return best


def _spatial_grid(s, l, finite, step: float) -> dict:
    """Bin point indices by floor(s/step), floor(l/step).

    Indices inside one cell stay in ascending order, the same order as walking
    ``finite`` from low to high. The bins do not depend on which points are queries.
    """
    finite = np.asarray(finite, dtype=np.int64)
    if finite.size == 0:
        return {}
    cell_s = np.floor(np.asarray(s, dtype=np.float64)[finite] / float(step)).astype(np.int64)
    cell_l = np.floor(np.asarray(l, dtype=np.float64)[finite] / float(step)).astype(np.int64)
    order = np.lexsort((finite, cell_l, cell_s))
    cell_s = cell_s[order]
    cell_l = cell_l[order]
    ordered = finite[order]
    if ordered.size == 1:
        return {(int(cell_s[0]), int(cell_l[0])): ordered}
    change = np.flatnonzero((cell_s[1:] != cell_s[:-1]) | (cell_l[1:] != cell_l[:-1])) + 1
    starts = np.concatenate((np.zeros(1, dtype=np.int64), change))
    ends = np.concatenate((change, np.asarray([ordered.size], dtype=np.int64)))
    grid = {}
    for start, end in zip(starts.tolist(), ends.tolist()):
        grid[(int(cell_s[start]), int(cell_l[start]))] = ordered[start:end]
    return grid


def _anchor_keys(s, l, query_index, step: float) -> list:
    """First-seen anchor cells, in the order a scan of ``query_index`` would meet them."""
    if int(query_index.size) == 0:
        return []
    ks = np.floor(np.asarray(s, dtype=np.float64)[query_index] / float(step)).astype(np.int64)
    kl = np.floor(np.asarray(l, dtype=np.float64)[query_index] / float(step)).astype(np.int64)
    position = np.arange(ks.size)
    order = np.lexsort((position, kl, ks))
    grouped_s = ks[order]
    grouped_l = kl[order]
    change = np.empty(ks.size, dtype=bool)
    change[0] = True
    if ks.size > 1:
        change[1:] = (grouped_s[1:] != grouped_s[:-1]) | (grouped_l[1:] != grouped_l[:-1])
    first = np.sort(order[change])
    return list(zip(ks[first].tolist(), kl[first].tolist()))


def local_protrusions(s, l, h, query_mask, config: dict, local_fit: str = "baseline", context_slh=None, spatial_grid=None) -> dict:
    """Positive deviations from local planes. Insertion labels are not an input.

    Context is every finite point, including points outside the corridor.
    Extra context points can support a plane. They are not query points and
    cannot become candidate points. A small fit residual is not proved accuracy.
    """
    started = time.perf_counter()
    s = np.asarray(s, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    n_current = int(s.shape[0])
    query_in = np.asarray(query_mask, dtype=bool)
    if context_slh is not None:
        extra_s, extra_l, extra_h = (np.asarray(part, dtype=np.float64).reshape(-1) for part in context_slh)
        if extra_s.size:
            s = np.concatenate([s, extra_s])
            l = np.concatenate([l, extra_l])
            h = np.concatenate([h, extra_h])
            query_in = np.concatenate([query_in, np.zeros(extra_s.size, dtype=bool)])
    query = query_in & np.isfinite(s) & np.isfinite(l) & np.isfinite(h)
    selected = np.zeros(s.shape, dtype=bool)
    reasons = np.full(s.shape, "", dtype=object)
    empty = {
        "mask": selected,
        "reasons": reasons,
        "models": [],
        "skip_counts": {},
        "timing_s": time.perf_counter() - started,
    }
    query_index = np.flatnonzero(query)
    if query_index.size == 0:
        return empty
    settings = _local_config(config)
    settings["fit"] = local_fit
    step = settings["step"]
    if spatial_grid is None or context_slh is not None:
        finite = np.flatnonzero(np.isfinite(s) & np.isfinite(l) & np.isfinite(h))
        grid = _spatial_grid(s, l, finite, step)
    else:
        grid = spatial_grid
    anchors = _anchor_keys(s, l, query_index, step)
    anchor_s = []
    anchor_l = []
    for key in anchors:
        cell = grid.get(key)
        if cell is None:
            members = np.asarray([0], dtype=np.int64)
        else:
            chosen = cell[query[cell]]
            members = chosen if chosen.size else cell
        anchor_s.append(float(np.median(s[members])))
        anchor_l.append(float(np.median(l[members])))
    anchor_s = np.asarray(anchor_s)
    anchor_l = np.asarray(anchor_l)

    def gather(center_s, center_l, radius):
        reach = int(np.ceil(radius / step))
        key_s = int(np.floor(center_s / step))
        key_l = int(np.floor(center_l / step))
        parts = []
        for ds in range(-reach, reach + 1):
            for dl in range(-reach, reach + 1):
                part = grid.get((key_s + ds, key_l + dl))
                if part is not None and int(part.size):
                    parts.append(part)
        if not parts:
            return np.zeros(0, dtype=np.int64)
        idx = np.concatenate(parts)
        return idx[np.hypot(s[idx] - center_s, l[idx] - center_l) <= radius]

    models = []
    usable = []
    for radius in settings["radii"]:
        row = []
        for center_s, center_l in zip(anchor_s.tolist(), anchor_l.tolist()):
            idx = gather(center_s, center_l, radius)
            model = {
                "radius_m": float(radius),
                "anchor_s_m": float(center_s),
                "anchor_l_m": float(center_l),
                "support_unique_xyz": int(idx.size),
                "support_s_m": [None, None],
                "support_l_m": [None, None],
                "fit_residual_p95_m": None,
                "holdout_residual_p95_m": None,
                "fit_residual_is_not_surface_accuracy": True,
                "insufficient_data": False,
                "multiple_levels": False,
                "unstable": False,
                "usable": False,
                "plane_asl": None,
                "reason": "",
            }
            if idx.size < settings["min_support"]:
                model["insufficient_data"] = True
                model["reason"] = "недостаточно точек для локальной плоскости"
            else:
                order = np.argsort(h[idx])
                ordered = idx[order]
                heights = h[ordered]
                gaps = np.diff(heights)
                half = max(4, settings["min_support"] // 2)
                multiple = False
                for cut in np.flatnonzero(gaps > settings["level_gap"]).tolist():
                    if cut + 1 >= half and heights.size - cut - 1 >= half:
                        multiple = True
                        break
                if settings["fit"] == "ransac":
                    fitted = _ransac_local_plane(idx, s, l, h, settings)
                    model.update(fitted)
                    model["multiple_levels"] = fitted["reason"] == "несколько опор, выбор не сделан"
                elif multiple:
                    model["multiple_levels"] = True
                    model["reason"] = "несколько высотных уровней в окрестности"
                else:
                    limit = float(np.quantile(heights, settings["lower_fraction"]))
                    lower = idx[h[idx] <= limit]
                    model["support_unique_xyz"] = int(lower.size)
                    if lower.size:
                        model["support_s_m"] = [float(np.min(s[lower])), float(np.max(s[lower]))]
                        model["support_l_m"] = [float(np.min(l[lower])), float(np.max(l[lower]))]
                    support_height_span = float(np.max(h[lower]) - np.min(h[lower])) if lower.size else 0.0
                    model["support_height_span_m"] = support_height_span
                    if settings["fit"] == "span_limit" and support_height_span > settings["level_gap"]:
                        model["insufficient_data"] = True
                        model["reason"] = "insufficient_local_support"
                        model["usable"] = False
                        fit_idx = np.zeros(0, dtype=np.int64)
                        hold_idx = np.zeros(0, dtype=np.int64)
                    else:
                        lower = lower[np.argsort(s[lower])]
                        fit_idx = lower[0::2]
                        hold_idx = lower[1::2]
                    if model["reason"] == "insufficient_local_support":
                        pass
                    elif fit_idx.size < settings["fit_min"] or hold_idx.size < settings["holdout_min"]:
                        model["insufficient_data"] = True
                        model["reason"] = "недостаточно отложенных точек для проверки плоскости"
                    else:
                        design = np.column_stack([s[fit_idx], l[fit_idx], np.ones(fit_idx.size)])
                        coef, *_ = np.linalg.lstsq(design, h[fit_idx], rcond=None)
                        fit_res = np.abs(h[fit_idx] - design @ coef)
                        hold_res = np.abs(h[hold_idx] - (coef[0] * s[hold_idx] + coef[1] * l[hold_idx] + coef[2]))
                        fit_p95 = float(np.quantile(fit_res, 0.95))
                        hold_p95 = float(np.quantile(hold_res, 0.95))
                        model["fit_residual_p95_m"] = fit_p95
                        model["holdout_residual_p95_m"] = hold_p95
                        model["plane_asl"] = [float(coef[0]), float(coef[1]), float(coef[2])]
                        model["fit_rows"] = [int(value) for value in fit_idx.tolist()]
                        model["hold_rows"] = [int(value) for value in hold_idx.tolist()]
                        if hold_p95 > settings["unstable_ratio"] * max(fit_p95, 1e-4):
                            model["unstable"] = True
                            model["reason"] = "проверка на отложенных точках не подтверждает остаток подгонки"
                        else:
                            model["usable"] = True
            model["id"] = len(models)
            models.append(model)
            row.append(model)
        usable.append(row)
    assigned = np.full(s.shape, -1, dtype=np.int32)
    pending = np.arange(query_index.size, dtype=np.int64)
    pending_failure = np.full(query_index.shape, "нет локальной модели", dtype=object)
    decision = [None] * int(s.shape[0])
    rejected = [[] for _ in range(int(s.shape[0]))]
    query_s = s[query_index]
    query_l = l[query_index]
    query_h = h[query_index]
    for radius_models in usable:
        if pending.size == 0 or not radius_models:
            break
        centers_s = np.asarray([model["anchor_s_m"] for model in radius_models], dtype=np.float64)
        centers_l = np.asarray([model["anchor_l_m"] for model in radius_models], dtype=np.float64)
        radius = float(radius_models[0]["radius_m"])
        distance = np.hypot(query_s[pending][:, None] - centers_s[None, :], query_l[pending][:, None] - centers_l[None, :])
        distance[distance > radius] = np.inf
        nearest = np.argmin(distance, axis=1)
        valid = np.isfinite(distance[np.arange(pending.size), nearest])
        still = []
        for local_row, point_row, model_index, is_valid in zip(
            np.arange(pending.size), pending.tolist(), nearest.tolist(), valid.tolist()
        ):
            point = int(query_index[point_row])
            if not is_valid:
                still.append(local_row)
                continue
            model = radius_models[model_index]
            if model["usable"] and model.get("normal_slope") is not None:
                span_s = model["support_s_m"]
                span_l = model["support_l_m"]
                inside_support = (
                    span_s[0] is not None
                    and span_s[0] <= query_s[point_row] <= span_s[1]
                    and span_l[0] <= query_l[point_row] <= span_l[1]
                )
                if not inside_support:
                    pending_failure[point_row] = "unknown"
                    still.append(local_row)
                    continue
            if not model["usable"]:
                pending_failure[point_row] = model["reason"] or pending_failure[point_row]
                rejected[point].append(
                    {
                        "model_id": int(model["id"]),
                        "usable": False,
                        "reason": model["reason"],
                        "radius_m": model["radius_m"],
                        "anchor_s_m": model["anchor_s_m"],
                        "anchor_l_m": model["anchor_l_m"],
                    }
                )
                still.append(local_row)
                continue
            plane = model["plane_asl"]
            deviation = float(query_h[point_row] - (plane[0] * query_s[point_row] + plane[1] * query_l[point_row] + plane[2]))
            threshold = float(model["holdout_residual_p95_m"])
            accepted = deviation > threshold and deviation > 0.0
            decision[point] = {
                "model_id": int(model["id"]),
                "plane_asl": list(plane),
                "support_s_m": list(model["support_s_m"]),
                "support_l_m": list(model["support_l_m"]),
                "fit_residual_p95_m": model["fit_residual_p95_m"],
                "holdout_residual_p95_m": model["holdout_residual_p95_m"],
                "threshold_m": threshold,
                "deviation_m": deviation,
                "accepted": bool(accepted),
                "reason": "положительное отклонение выше порога" if accepted else "отклонение не превышает разброс отложенных точек",
            }
            if accepted:
                selected[point] = True
                assigned[point] = int(model["id"])
            else:
                reasons[point] = "отклонение не превышает разброс отложенных точек"
        if still:
            pending = pending[np.asarray(still, dtype=np.int64)]
        else:
            pending = pending[:0]
    for point_row in pending.tolist():
        reasons[int(query_index[point_row])] = pending_failure[point_row]
    skip_counts: dict[str, int] = {}
    for reason in reasons[query].tolist():
        if reason:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
    for model in models:
        for key in ("fit_rows", "hold_rows"):
            if key in model:
                model[key] = [int(value) for value in model[key] if int(value) < n_current]
    return {
        "mask": selected[:n_current],
        "reasons": reasons[:n_current],
        "assigned_model": assigned[:n_current],
        "models": models,
        "decision": decision[:n_current],
        "rejected_models": rejected[:n_current],
        "skip_counts": skip_counts,
        "context_points_are_not_candidates": True,
        "timing_s": time.perf_counter() - started,
    }


def _extent(s, l, h) -> dict:
    return {
        "s_m": float(np.max(s) - np.min(s)) if s.size else 0.0,
        "l_m": float(np.max(l) - np.min(l)) if l.size else 0.0,
        "h_m": float(np.max(h) - np.min(h)) if h.size else 0.0,
        "extent_is_not_full_object_size": True,
    }


def _candidate(kind, ordinal, hypothesis_id, rows, s, l, h, ranges, indices, doubts) -> dict:
    return {
        "candidate_id": f"{kind[0].upper()}{ordinal}",
        "kind": kind,
        "hypothesis_id": hypothesis_id,
        "rows": np.asarray(rows, dtype=np.int64),
        "source_indices": np.asarray(indices, dtype=np.int64),
        "unique_xyz": int(rows.size),
        "observed_extent_m": _extent(s[rows], l[rows], h[rows]),
        "observed_s_min_m": float(np.min(s[rows])) if rows.size else None,
        "observed_h_min_m": float(np.min(h[rows])) if rows.size else None,
        "range_from_lidar_m": float(np.min(ranges[rows])) if rows.size else None,
        "conditional": True,
        "doubt_reasons": list(doubts),
        "not_a_confirmed_obstacle": True,
    }


def detect(
    s,
    l,
    h,
    dh,
    lateral_support,
    ranges,
    source_indices,
    geometric_relation,
    corridor,
    line_samples,
    ground_p95_m,
    config,
    connectivity="cells",
    local_protrusions_enabled=False,
    local_fit: str = "baseline",
    local_context_slh=None,
    spatial_grid=None,
) -> dict:
    """Candidates for one hypothesis. `source_indices` are cloud indices, not insertion labels."""
    started = time.perf_counter()
    s = np.asarray(s, dtype=np.float64)
    relation = np.asarray(geometric_relation).astype(str)
    n = int(s.size)
    inside = relation == "inside"
    h = np.asarray(h, dtype=np.float64)
    tolerance = float(config["structure_height_tolerance_m"]["value"])
    on_line, path_possible = path_structure_masks(
        s,
        l,
        h,
        line_samples,
        float(corridor.get("section_half_m", 0.5)),
        tolerance,
    )
    on_line &= inside
    path_possible &= inside
    height = float(config["comparable_object_height_m"]["value"])
    fraction = float(config["surface_comparable_fraction"]["value"])
    delete_below = fraction * height
    band = None if ground_p95_m is None else float(ground_p95_m)
    dh = np.asarray(dh, dtype=np.float64)
    supported = np.asarray(lateral_support, dtype=bool)
    # One ulp of a metre-scale height is about 4e-16 m. A fitted plane of a
    # constant floor lands on both sides of zero by that amount. This is not
    # a centimetre exclusion band.
    numerical_tol = 32.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(h))
    near_surface = inside & supported & np.isfinite(dh) & (dh >= -numerical_tol)
    expanded = False
    used_band = None
    if band is None:
        ambiguous = np.zeros(n, dtype=bool)
        explained = np.zeros(n, dtype=bool)
        surface_mode = "no_fitting_residual"
    elif band >= delete_below:
        used_band = max(band, height)
        expanded = used_band > band + 1e-12
        ambiguous = near_surface & (dh <= used_band)
        explained = np.zeros(n, dtype=bool)
        surface_mode = "ambiguous_not_deleted"
    else:
        used_band = band
        ambiguous = np.zeros(n, dtype=bool)
        explained = near_surface & (dh <= band + numerical_tol)
        surface_mode = "thin_band_removed"
    expansion_only = ambiguous & (dh > band) if expanded else np.zeros(n, dtype=bool)
    object_pool = inside & ~on_line & ~explained & ~ambiguous
    filter_s = time.perf_counter() - started
    started = time.perf_counter()
    l_values = np.asarray(l, dtype=np.float64)
    if connectivity == "distance":
        labels = cluster_labels_by_distance(
            s[object_pool],
            l_values[object_pool],
            h[object_pool],
            float(config["link_distance_m"]["value"]),
            float(config["link_cell_size_m"]["value"]),
        )
    elif connectivity == "cells":
        labels = cluster_labels(s[object_pool], l_values[object_pool], h[object_pool], float(config["cluster_gap_m"]["value"]))
    else:
        raise ValueError("connectivity must be cells or distance")
    link_s = time.perf_counter() - started
    pool_index = np.flatnonzero(object_pool)
    large = []
    rare = []
    doubts = ["результат условный"]
    if corridor.get("lower_boundary_status") != "stated":
        doubts.append("нижняя граница — допущение, не калибровка")
    if corridor.get("kind") != "consistent_rail_pair_hypothesis":
        doubts.append("гипотеза пути не согласована")
    minimum = int(config["large_min_unique_xyz"]["value"])
    indices = np.asarray(source_indices)
    ranges = np.asarray(ranges, dtype=np.float64)
    if labels.size:
        for label in range(int(labels.max()) + 1):
            rows = pool_index[labels == label]
            item = _candidate(
                "large" if rows.size >= minimum else "rare",
                len(large) + 1 if rows.size >= minimum else len(rare) + 1,
                corridor.get("hypothesis_id"),
                rows,
                s,
                np.asarray(l),
                np.asarray(h),
                ranges,
                indices[rows],
                doubts,
            )
            item["branch"] = "volume_or_suspended"
            if rows.size >= minimum:
                large.append(item)
            else:
                rare.append(item)
    candidate_s = time.perf_counter() - started - link_s
    local_s = 0.0
    protrusion_large = []
    protrusion_rare = []
    local_info = {"skip_counts": {}, "models": []}
    if local_protrusions_enabled:
        local_info = local_protrusions(
            s, l_values, h, ambiguous & ~on_line, config, local_fit=local_fit, context_slh=local_context_slh, spatial_grid=spatial_grid
        )
        local_s = float(local_info["timing_s"])
        started = time.perf_counter()
        protrusion_index = np.flatnonzero(local_info["mask"])
        if protrusion_index.size:
            protrusion_labels = cluster_labels_by_distance(
                s[protrusion_index],
                l_values[protrusion_index],
                h[protrusion_index],
                float(config["link_distance_m"]["value"]),
                float(config["link_cell_size_m"]["value"]),
            )
            link_s += time.perf_counter() - started
            started = time.perf_counter()
            for label in range(int(protrusion_labels.max()) + 1):
                rows = protrusion_index[protrusion_labels == label]
                item = _candidate(
                    "large" if rows.size >= minimum else "rare",
                    len(protrusion_large) + 1 if rows.size >= minimum else len(protrusion_rare) + 1,
                    corridor.get("hypothesis_id"),
                    rows,
                    s,
                    l_values,
                    h,
                    ranges,
                    indices[rows],
                    doubts + ["локальный выступ, не весь неоднозначный слой"],
                )
                model_ids = local_info["assigned_model"][rows]
                model_ids = model_ids[model_ids >= 0]
                if model_ids.size:
                    model_id = int(np.bincount(model_ids).argmax())
                    model = dict(local_info["models"][model_id])
                    model.pop("plane_asl", None)
                    item["local_model"] = model
                    plane = local_info["models"][model_id].get("plane_asl")
                    if plane is not None:
                        item["local_plane_asl"] = plane
                item["branch"] = "local_protrusion"
                item["candidate_id"] = ("P" if rows.size >= minimum else "Q") + item["candidate_id"][1:]
                if rows.size >= minimum:
                    protrusion_large.append(item)
                else:
                    protrusion_rare.append(item)
            candidate_s += time.perf_counter() - started
    ambiguous_rows = np.flatnonzero(ambiguous & ~on_line)
    return {
        "hypothesis_id": corridor.get("hypothesis_id"),
        "rules": list(RULES),
        "surface_mode": surface_mode,
        "surface_band_m": band,
        "surface_delete_below_m": delete_below,
        "surface": {
            "fitting_residual_p95_m": band,
            "comparable_object_height_m": height,
            "used_band_m": used_band,
            "max_with_object_height_applied": expanded,
            "points_withheld_from_main_search": int((ambiguous & ~on_line).sum() + explained.sum()),
            "points_withheld_ambiguous": int((ambiguous & ~on_line).sum()),
            "points_removed_by_thin_band": int(explained.sum()),
            "diagnostic_residual_only_band_m": band,
            "diagnostic_points_withheld_only_by_expansion": int((expansion_only & ~on_line).sum()),
            "diagnostic_array_is_not_detection": True,
            "ambiguous_layer_is_not_one_alarm": True,
        },
        "counts": {
            "input": n,
            "geometric_inside": int(inside.sum()),
            "excluded_path_structure": int(on_line.sum()),
            "path_possible_not_removed": int(path_possible.sum()),
            "excluded_thin_surface": int(explained.sum()),
            "ambiguous_surface": int(ambiguous_rows.size),
            "object_pool": int(object_pool.sum()),
            "large": len(large),
            "rare": len(rare),
            "protrusion_large": len(protrusion_large),
            "protrusion_rare": len(protrusion_rare),
        },
        "candidates_large": large,
        "candidates_rare": rare,
        "candidates_protrusion_large": protrusion_large,
        "candidates_protrusion_rare": protrusion_rare,
        "local_skip_counts": local_info["skip_counts"],
        "local_reasons": local_info.get("reasons"),
        "local_decision": local_info.get("decision"),
        "local_models": local_info.get("models"),
        "local_rejected": local_info.get("rejected_models"),
        "connectivity": connectivity,
        "local_protrusions_enabled": bool(local_protrusions_enabled),
        "ambiguous_surface_indices": indices[ambiguous_rows],
        "masks": {
            "inside": inside,
            "path_structure": on_line,
            "path_possible": path_possible,
            "thin_surface": explained,
            "ambiguous_surface": ambiguous & ~on_line,
            "expansion_only": expansion_only & ~on_line,
            "object_pool": object_pool,
        },
        "absence_of_candidates_is_not_clear": True,
        "obstacle_reported": None,
        "own_path_selected": False,
        "conditional": True,
        "doubt_reasons": doubts,
        "timing_s": {
            "filter": filter_s,
            "cluster": link_s,
            "link": link_s,
            "local_model": local_s,
            "candidates": candidate_s,
            "detector": filter_s + link_s + local_s + candidate_s,
        },
    }


def line_samples(hypothesis, fragments) -> list[dict]:
    """Observed samples of the two lines that formed the hypothesis."""
    samples = []
    fragments = list(fragments or [])
    for candidate_id in hypothesis.get("candidate_ids") or []:
        if not isinstance(candidate_id, int) or candidate_id < 0 or candidate_id >= len(fragments):
            continue
        for sample in fragments[candidate_id].get("section_samples") or []:
            if sample.get("width_m") is None or sample.get("l_m") is None:
                continue
            observed_h = sample.get("h_m")
            samples.append(
                {
                    "s_m": float(sample["s_m"]),
                    "l_m": float(sample["l_m"]),
                    "width_m": float(sample["width_m"]),
                    "h_m": None if observed_h is None else float(observed_h),
                    "height_support": sample.get("height_support") or "unknown",
                }
            )
    return samples


def subsample_grid(origin, size, shape, keep, seed: int) -> np.ndarray:
    """Deterministic points inside a box. `origin` is the minimum corner."""
    axes = [np.linspace(origin[axis], origin[axis] + size[axis], int(shape[axis])) for axis in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    keep = int(min(keep, grid.shape[0]))
    rng = np.random.default_rng(int(seed))
    chosen = np.sort(rng.choice(grid.shape[0], keep, replace=False))
    return grid[chosen]


def sensor_from_working_points(points) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return working_to_sensor(points[:, 0], points[:, 1], points[:, 2])


def insertion_funnel(inserted_rows, masks) -> dict:
    """Count inserted rows after each rule. Labels stay outside `detect`."""
    present = np.zeros(masks["inside"].shape[0], dtype=bool)
    present[np.asarray(inserted_rows, dtype=np.int64)] = True
    inside = int((masks["inside"] & present).sum())
    path = int((masks["path_structure"] & present).sum())
    thin = int((masks["thin_surface"] & present).sum())
    ambiguous = int((masks["ambiguous_surface"] & present).sum())
    pooled = int((masks["object_pool"] & present).sum())
    return {
        "inserted": int(present.sum()) if inserted_rows is not None else 0,
        "geometric_inside": inside,
        "excluded_by_path": path,
        "excluded_by_thin_surface": thin,
        "ambiguous_surface": ambiguous,
        "object_pool": pooled,
        "pooled_is_not_separation": True,
    }


def evaluate_insertion(inserted_rows, detected, before_candidates, ranges) -> dict:
    """Score an insertion after detect. Labels are not an input of `detect`.

    A positive object-pool count is not separation from the background.
    A retained single point is not a confirmed wire detection.
    """
    rows = np.asarray(inserted_rows, dtype=np.int64)
    masks = detected["masks"]
    present = np.zeros(masks["inside"].shape[0], dtype=bool)
    if rows.size:
        present[rows] = True
    ranges = np.asarray(ranges, dtype=np.float64)
    inserted = int(present.sum())
    inside = int((masks["inside"] & present).sum())
    passed = int((masks["object_pool"] & present).sum())
    ambiguous = int((masks["ambiguous_surface"] & present).sum())
    path = int((masks["path_structure"] & present).sum())
    expansion = int((masks["expansion_only"] & present).sum())
    prior = []
    for candidate in before_candidates or []:
        prior.append({int(value) for value in np.asarray(candidate["source_indices"]).tolist()})
    hits = []
    covered = np.zeros(present.shape, dtype=bool)
    listed = list(detected["candidates_large"]) + list(detected["candidates_rare"])
    listed += list(detected.get("candidates_protrusion_large") or [])
    listed += list(detected.get("candidates_protrusion_rare") or [])
    for candidate in listed:
        member_rows = np.asarray(candidate["rows"], dtype=np.int64)
        member = np.zeros(present.shape, dtype=bool)
        member[member_rows] = True
        inserted_here = int((member & present).sum())
        if inserted_here == 0:
            continue
        covered |= member & present
        original_here = int(member.sum()) - inserted_here
        original_sources = [
            int(index)
            for index, row in zip(np.asarray(candidate["source_indices"]).tolist(), member_rows.tolist())
            if not present[row]
        ]
        overlaps = any(set(original_sources) & earlier for earlier in prior)
        share = inserted_here / float(member.sum())
        inserted_member = member_rows[present[member_rows]]
        hits.append(
            {
                "candidate_id": candidate["candidate_id"],
                "kind": candidate["kind"],
                "insertion_points": inserted_here,
                "original_points": original_here,
                "retained_fraction": inserted_here / inserted if inserted else 0.0,
                "insertion_share": share,
                "observed_extent_m": candidate["observed_extent_m"],
                "range_from_lidar_m": candidate["range_from_lidar_m"],
                "nearest_insertion_range_m": float(np.min(ranges[inserted_member])),
                "range_error_m": abs(float(candidate["range_from_lidar_m"]) - float(np.min(ranges[inserted_member]))),
                "branch": candidate.get("branch", "volume_or_suspended"),
                "overlaps_prior_candidate": overlaps,
                "majority_insertion": share > 0.5,
                "extent_includes_every_point_in_the_candidate": True,
            }
        )
    in_candidate = int(covered.sum())
    majority = [item for item in hits if item["majority_insertion"]]
    merged_with_background = any(
        item["overlaps_prior_candidate"] or (item["original_points"] > 0 and not item["majority_insertion"])
        for item in hits
    )
    if inside == 0:
        outcome = "outside_corridor"
    elif majority:
        outcome = "predominantly_insertion"
    elif merged_with_background:
        outcome = "merged_with_background"
    elif ambiguous and in_candidate == 0:
        outcome = "ambiguous_layer_only"
    elif path and in_candidate == 0:
        outcome = "excluded_by_path_volume"
    else:
        outcome = "no_candidate"
    return {
        "inserted": inserted,
        "geometric_inside": inside,
        "passed_filters": passed,
        "in_candidate": in_candidate,
        "excluded_by_path": path,
        "ambiguous_surface": ambiguous,
        "diagnostic_expansion_only": expansion,
        "diagnostic_expansion_is_not_detection": True,
        "predominantly_insertion": bool(majority),
        "merged_with_background": bool(merged_with_background),
        "ambiguous_layer_only": outcome == "ambiguous_layer_only",
        "outcome": outcome,
        "pooled_is_not_separation": True,
        "single_point_is_not_wire_detection": inserted == 1,
        "hits": hits,
        "prior_candidate_in_the_same_place_is_not_detection": True,
    }
