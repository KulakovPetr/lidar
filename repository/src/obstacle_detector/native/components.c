/* Radius components without storing the edge list.
   components_from_pairs remains for tests that already hold a small pair array. */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static int32_t find_root(int32_t *parent, int32_t index) {
    int32_t root = index;
    while (parent[root] != root) {
        root = parent[root];
    }
    while (parent[index] != root) {
        int32_t next = parent[index];
        parent[index] = root;
        index = next;
    }
    return root;
}

static void unite(int32_t *parent, int8_t *rank, int32_t left, int32_t right) {
    int32_t a = find_root(parent, left);
    int32_t b = find_root(parent, right);
    if (a == b) {
        return;
    }
    if (rank[a] < rank[b]) {
        parent[a] = b;
    } else if (rank[a] > rank[b]) {
        parent[b] = a;
    } else {
        parent[b] = a;
        rank[a] = (int8_t)(rank[a] + 1);
    }
}

static int remap_labels(int64_t n, int32_t *parent, int32_t *labels) {
    int64_t i;
    int32_t *map = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    int32_t next = 0;
    if (map == NULL) {
        return 2;
    }
    for (i = 0; i < n; ++i) {
        map[i] = -1;
    }
    for (i = 0; i < n; ++i) {
        int32_t root = find_root(parent, (int32_t)i);
        if (map[root] < 0) {
            map[root] = next;
            next += 1;
        }
        labels[i] = map[root];
    }
    free(map);
    return 0;
}

int components_from_pairs(int64_t n, int64_t n_pairs, const int64_t *pairs, int32_t *labels) {
    int64_t i;
    int32_t *parent;
    int8_t *rank;
    int code;
    if (n < 0 || n_pairs < 0 || labels == NULL) {
        return 1;
    }
    if (n == 0) {
        return 0;
    }
    parent = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    rank = (int8_t *)malloc((size_t)n * sizeof(int8_t));
    if (parent == NULL || rank == NULL) {
        free(parent);
        free(rank);
        return 2;
    }
    for (i = 0; i < n; ++i) {
        parent[i] = (int32_t)i;
        rank[i] = 0;
    }
    if (pairs != NULL) {
        for (i = 0; i < n_pairs; ++i) {
            int32_t left = (int32_t)pairs[2 * i];
            int32_t right = (int32_t)pairs[2 * i + 1];
            if (left < 0 || right < 0 || left >= n || right >= n || left == right) {
                continue;
            }
            unite(parent, rank, left, right);
        }
    }
    code = remap_labels(n, parent, labels);
    free(parent);
    free(rank);
    return code;
}

typedef struct {
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t index;
} GridPoint;

typedef struct {
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t begin;
    int32_t end;
} GridCell;

typedef struct {
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t cell;
    uint8_t used;
} HashSlot;

static int cmp_grid(const void *left, const void *right) {
    const GridPoint *a = (const GridPoint *)left;
    const GridPoint *b = (const GridPoint *)right;
    if (a->x != b->x) {
        return a->x < b->x ? -1 : 1;
    }
    if (a->y != b->y) {
        return a->y < b->y ? -1 : 1;
    }
    if (a->z != b->z) {
        return a->z < b->z ? -1 : 1;
    }
    if (a->index != b->index) {
        return a->index < b->index ? -1 : 1;
    }
    return 0;
}

static uint32_t hash_cell(int32_t x, int32_t y, int32_t z) {
    uint32_t h = 2166136261u;
    h = (h ^ (uint32_t)x) * 16777619u;
    h = (h ^ (uint32_t)y) * 16777619u;
    h = (h ^ (uint32_t)z) * 16777619u;
    return h;
}

static int lookup_cell(const HashSlot *table, uint32_t mask, int32_t x, int32_t y, int32_t z) {
    uint32_t slot = hash_cell(x, y, z) & mask;
    uint32_t step = 0;
    while (table[slot].used) {
        if (table[slot].x == x && table[slot].y == y && table[slot].z == z) {
            return table[slot].cell;
        }
        step += 1;
        slot = (slot + step) & mask;
    }
    return -1;
}

static int insert_cell(HashSlot *table, uint32_t mask, int32_t x, int32_t y, int32_t z, int32_t cell) {
    uint32_t slot = hash_cell(x, y, z) & mask;
    uint32_t step = 0;
    while (table[slot].used) {
        step += 1;
        slot = (slot + step) & mask;
    }
    table[slot].used = 1;
    table[slot].x = x;
    table[slot].y = y;
    table[slot].z = z;
    table[slot].cell = cell;
    return 0;
}

static int cell_before(int32_t ax, int32_t ay, int32_t az, int32_t bx, int32_t by, int32_t bz) {
    if (ax != bx) {
        return ax < bx;
    }
    if (ay != by) {
        return ay < by;
    }
    return az < bz;
}

static double axis_gap(int delta, double cell) {
    int gap = delta < 0 ? -delta : delta;
    if (gap == 0) {
        return 0.0;
    }
    return (double)(gap - 1) * cell;
}

static int prepare_grid(
    int64_t n,
    const double *xyz,
    double cell,
    GridPoint **points_out,
    GridCell **cells_out,
    int32_t *n_cells_out,
    HashSlot **table_out,
    uint32_t *mask_out
) {
    int64_t i;
    int32_t cells_n = 0;
    int32_t c;
    uint32_t cap = 1;
    double inv = 1.0 / cell;
    GridPoint *points = (GridPoint *)malloc((size_t)n * sizeof(GridPoint));
    GridCell *cells;
    HashSlot *table;
    if (points == NULL) {
        return 2;
    }
    for (i = 0; i < n; ++i) {
        points[i].x = (int32_t)floor(xyz[3 * i] * inv);
        points[i].y = (int32_t)floor(xyz[3 * i + 1] * inv);
        points[i].z = (int32_t)floor(xyz[3 * i + 2] * inv);
        points[i].index = (int32_t)i;
    }
    qsort(points, (size_t)n, sizeof(GridPoint), cmp_grid);
    for (i = 0; i < n;) {
        int64_t j = i + 1;
        while (j < n && points[j].x == points[i].x && points[j].y == points[i].y && points[j].z == points[i].z) {
            j += 1;
        }
        cells_n += 1;
        i = j;
    }
    cells = (GridCell *)malloc((size_t)cells_n * sizeof(GridCell));
    if (cells == NULL) {
        free(points);
        return 2;
    }
    c = 0;
    for (i = 0; i < n;) {
        int64_t j = i + 1;
        while (j < n && points[j].x == points[i].x && points[j].y == points[i].y && points[j].z == points[i].z) {
            j += 1;
        }
        cells[c].x = points[i].x;
        cells[c].y = points[i].y;
        cells[c].z = points[i].z;
        cells[c].begin = (int32_t)i;
        cells[c].end = (int32_t)j;
        c += 1;
        i = j;
    }
    while (cap < (uint32_t)cells_n * 2u) {
        cap <<= 1;
    }
    table = (HashSlot *)calloc(cap, sizeof(HashSlot));
    if (table == NULL) {
        free(points);
        free(cells);
        return 2;
    }
    for (c = 0; c < cells_n; ++c) {
        insert_cell(table, cap - 1u, cells[c].x, cells[c].y, cells[c].z, c);
    }
    *points_out = points;
    *cells_out = cells;
    *n_cells_out = cells_n;
    *table_out = table;
    *mask_out = cap - 1u;
    return 0;
}

static int link_pair(
    const double *xyz,
    int32_t left,
    int32_t right,
    double limit2,
    int32_t *parent,
    int8_t *rank,
    int64_t *pair_checks
) {
    double dx;
    double dy;
    double dz;
    double dist2;
    if (find_root(parent, left) == find_root(parent, right)) {
        return 0;
    }
    dx = xyz[3 * left] - xyz[3 * right];
    dy = xyz[3 * left + 1] - xyz[3 * right + 1];
    dz = xyz[3 * left + 2] - xyz[3 * right + 2];
    dist2 = dx * dx + dy * dy + dz * dz;
    if (pair_checks != NULL) {
        *pair_checks += 1;
    }
    if (dist2 <= limit2) {
        unite(parent, rank, left, right);
        return 1;
    }
    return 0;
}

int labels_fixed_radius(int64_t n, const double *xyz, double radius, int32_t *labels, int64_t *pair_checks) {
    int32_t *parent;
    int8_t *rank;
    GridPoint *points = NULL;
    GridCell *cells = NULL;
    HashSlot *table = NULL;
    int32_t n_cells = 0;
    uint32_t mask = 0;
    int code;
    int32_t c;
    double cell;
    double limit2;
    if (n < 0 || xyz == NULL || labels == NULL || !(radius >= 0.0)) {
        return 1;
    }
    if (pair_checks != NULL) {
        *pair_checks = 0;
    }
    if (n == 0) {
        return 0;
    }
    parent = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    rank = (int8_t *)malloc((size_t)n * sizeof(int8_t));
    if (parent == NULL || rank == NULL) {
        free(parent);
        free(rank);
        return 2;
    }
    for (c = 0; c < n; ++c) {
        parent[c] = c;
        rank[c] = 0;
    }
    if (radius == 0.0) {
        cell = 1.0;
    } else {
        /* Slightly under radius/sqrt(3), so the cell diagonal is strictly below the radius. */
        cell = (radius / sqrt(3.0)) * (1.0 - 1e-6);
    }
    limit2 = radius * radius;
    code = prepare_grid(n, xyz, cell, &points, &cells, &n_cells, &table, &mask);
    if (code != 0) {
        free(parent);
        free(rank);
        return code;
    }
    if (radius > 0.0) {
        for (c = 0; c < n_cells; ++c) {
            int32_t i;
            int32_t head = points[cells[c].begin].index;
            for (i = cells[c].begin + 1; i < cells[c].end; ++i) {
                unite(parent, rank, head, points[i].index);
            }
        }
    } else {
        for (c = 0; c < n_cells; ++c) {
            int32_t i;
            int32_t j;
            for (i = cells[c].begin; i < cells[c].end; ++i) {
                for (j = i + 1; j < cells[c].end; ++j) {
                    link_pair(xyz, points[i].index, points[j].index, 0.0, parent, rank, pair_checks);
                }
            }
        }
    }
    for (c = 0; c < n_cells; ++c) {
        int dx;
        int dy;
        int dz;
        for (dx = -2; dx <= 2; ++dx) {
            for (dy = -2; dy <= 2; ++dy) {
                for (dz = -2; dz <= 2; ++dz) {
                    int32_t nx = cells[c].x + dx;
                    int32_t ny = cells[c].y + dy;
                    int32_t nz = cells[c].z + dz;
                    int other;
                    double gap2;
                    int32_t i;
                    int32_t j;
                    int linked = 0;
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    if (!cell_before(cells[c].x, cells[c].y, cells[c].z, nx, ny, nz)) {
                        continue;
                    }
                    gap2 = 0.0;
                    {
                        double gx = axis_gap(dx, cell);
                        double gy = axis_gap(dy, cell);
                        double gz = axis_gap(dz, cell);
                        gap2 = gx * gx + gy * gy + gz * gz;
                    }
                    if (gap2 > limit2) {
                        continue;
                    }
                    other = lookup_cell(table, mask, nx, ny, nz);
                    if (other < 0) {
                        continue;
                    }
                    if (find_root(parent, points[cells[c].begin].index) == find_root(parent, points[cells[other].begin].index)) {
                        continue;
                    }
                    for (i = cells[c].begin; i < cells[c].end && !linked; ++i) {
                        for (j = cells[other].begin; j < cells[other].end; ++j) {
                            if (link_pair(xyz, points[i].index, points[j].index, limit2, parent, rank, pair_checks)) {
                                linked = 1;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    code = remap_labels(n, parent, labels);
    free(points);
    free(cells);
    free(table);
    free(parent);
    free(rank);
    return code;
}

int labels_max_radius(int64_t n, const double *xyz, const double *point_radius, int32_t *labels, int64_t *pair_checks) {
    int32_t *parent;
    int8_t *rank;
    GridPoint *points = NULL;
    GridCell *cells = NULL;
    HashSlot *table = NULL;
    int32_t n_cells = 0;
    uint32_t mask = 0;
    int code;
    int32_t c;
    double max_radius = 0.0;
    double min_radius = 0.0;
    double cell;
    int reach;
    int64_t i;
    if (n < 0 || xyz == NULL || point_radius == NULL || labels == NULL) {
        return 1;
    }
    if (pair_checks != NULL) {
        *pair_checks = 0;
    }
    if (n == 0) {
        return 0;
    }
    for (i = 0; i < n; ++i) {
        if (!(point_radius[i] >= 0.0)) {
            return 1;
        }
        if (i == 0 || point_radius[i] < min_radius) {
            min_radius = point_radius[i];
        }
        if (point_radius[i] > max_radius) {
            max_radius = point_radius[i];
        }
    }
    parent = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    rank = (int8_t *)malloc((size_t)n * sizeof(int8_t));
    if (parent == NULL || rank == NULL) {
        free(parent);
        free(rank);
        return 2;
    }
    for (c = 0; c < n; ++c) {
        parent[c] = c;
        rank[c] = 0;
    }
    /* Cell diagonal stays below the smallest radius, so every pair inside a cell is a real link. */
    cell = min_radius > 0.0 ? (min_radius / sqrt(3.0)) * (1.0 - 1e-6) : 1.0;
    reach = min_radius > 0.0 ? (int)floor(max_radius / cell) + 1 : 0;
    code = prepare_grid(n, xyz, cell, &points, &cells, &n_cells, &table, &mask);
    if (code != 0) {
        free(parent);
        free(rank);
        return code;
    }
    if (min_radius > 0.0) {
        for (c = 0; c < n_cells; ++c) {
            int32_t i0;
            int32_t head = points[cells[c].begin].index;
            for (i0 = cells[c].begin + 1; i0 < cells[c].end; ++i0) {
                unite(parent, rank, head, points[i0].index);
            }
        }
    }
    for (c = 0; c < n_cells; ++c) {
        int dx;
        int dy;
        int dz;
        for (dx = -reach; dx <= reach; ++dx) {
            for (dy = -reach; dy <= reach; ++dy) {
                for (dz = -reach; dz <= reach; ++dz) {
                    int32_t nx = cells[c].x + dx;
                    int32_t ny = cells[c].y + dy;
                    int32_t nz = cells[c].z + dz;
                    int other;
                    double gap2;
                    int32_t ia;
                    int32_t ib;
                    int linked = 0;
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    if (!cell_before(cells[c].x, cells[c].y, cells[c].z, nx, ny, nz)) {
                        continue;
                    }
                    {
                        double gx = axis_gap(dx, cell);
                        double gy = axis_gap(dy, cell);
                        double gz = axis_gap(dz, cell);
                        gap2 = gx * gx + gy * gy + gz * gz;
                    }
                    if (gap2 > max_radius * max_radius) {
                        continue;
                    }
                    other = lookup_cell(table, mask, nx, ny, nz);
                    if (other < 0) {
                        continue;
                    }
                    if (find_root(parent, points[cells[c].begin].index) == find_root(parent, points[cells[other].begin].index)) {
                        continue;
                    }
                    for (ia = cells[c].begin; ia < cells[c].end && !linked; ++ia) {
                        for (ib = cells[other].begin; ib < cells[other].end; ++ib) {
                            int32_t a = points[ia].index;
                            int32_t b = points[ib].index;
                            double allow = point_radius[a] > point_radius[b] ? point_radius[a] : point_radius[b];
                            if (link_pair(xyz, a, b, allow * allow, parent, rank, pair_checks)) {
                                linked = 1;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    code = remap_labels(n, parent, labels);
    free(points);
    free(cells);
    free(table);
    free(parent);
    free(rank);
    return code;
}
