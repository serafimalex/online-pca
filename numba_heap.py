import numpy as np
from numba import njit, prange, set_num_threads, int64, float32
from line_profiler import profile
NEG_INF32 = np.finfo(np.float32).min

@njit(parallel=True)
def build_row_max_parallel(matrix, row_max_values, row_max_indices):
    """
    Parallel scan over rows to compute per-row maxima.
    matrix: float32 2D
    row_max_values: float32 1D (output)
    row_max_indices: int64 1D (output)
    """
    rows, cols = matrix.shape
    for i in prange(rows):
        # local variables for speed
        maxv = NEG_INF32
        maxidx = -1
        # iterate columns
        for j in range(cols):
            v = matrix[i, j]
            if v > maxv:
                maxv = v
                maxidx = j
        row_max_values[i] = maxv
        row_max_indices[i] = maxidx

@njit
def _next_pow2(n):
    size = 1
    while size < n:
        size <<= 1
    return size

@njit
def segtree_build(row_max_values):
    n = row_max_values.shape[0]
    size = _next_pow2(n)
    vals = np.empty(2 * size, dtype=np.float32)
    idxs = np.empty(2 * size, dtype=np.int64)

    # initialize leaves
    for i in range(size):
        if i < n:
            vals[size + i] = row_max_values[i]
            idxs[size + i] = i
        else:
            vals[size + i] = NEG_INF32
            idxs[size + i] = -1

    # build parent nodes
    for i in range(size - 1, 0, -1):
        left = 2 * i
        right = left + 1
        if vals[left] >= vals[right]:
            vals[i] = vals[left]; idxs[i] = idxs[left]
        else:
            vals[i] = vals[right]; idxs[i] = idxs[right]

    return vals, idxs, size

@njit
def segtree_update(vals, idxs, size, pos, newval):
    idx = size + pos
    vals[idx] = newval
    idxs[idx] = pos
    idx //= 2
    while idx >= 1:
        left = 2 * idx
        right = left + 1
        if vals[left] >= vals[right]:
            vals[idx] = vals[left]; idxs[idx] = idxs[left]
        else:
            vals[idx] = vals[right]; idxs[idx] = idxs[right]
        idx //= 2

@njit
def segtree_get_max(vals, idxs):
    return vals[1], idxs[1]

@njit
def update_row_inplace_f32(matrix, row_idx, new_values,
                           row_max_values, row_max_indices,
                           tree_vals, tree_idxs, tree_size):
    cols = matrix.shape[1]
    for j in range(cols):
        matrix[row_idx, j] = new_values[j]

    maxv = NEG_INF32
    maxidx = -1
    for j in range(cols):
        v = matrix[row_idx, j]
        if v > maxv:
            maxv = v
            maxidx = j

    row_max_values[row_idx] = maxv
    row_max_indices[row_idx] = maxidx
    segtree_update(tree_vals, tree_idxs, tree_size, row_idx, maxv)

@njit
def update_cell_inplace_f32(matrix, row_idx, col_idx, new_value,
                            row_max_values, row_max_indices,
                            tree_vals, tree_idxs, tree_size):
    matrix[row_idx, col_idx] = new_value
    current_max = row_max_values[row_idx]

    if new_value > current_max:
        # new value becomes the row max
        row_max_values[row_idx] = new_value
        row_max_indices[row_idx] = col_idx
        segtree_update(tree_vals, tree_idxs, tree_size, row_idx, new_value)
    elif col_idx == row_max_indices[row_idx] and new_value < current_max:
        # previous maximum decreased; recompute row max
        maxv = NEG_INF32
        maxidx = -1
        cols = matrix.shape[1]
        for j in range(cols):
            v = matrix[row_idx, j]
            if v > maxv:
                maxv = v
                maxidx = j
        row_max_values[row_idx] = maxv
        row_max_indices[row_idx] = maxidx
        segtree_update(tree_vals, tree_idxs, tree_size, row_idx, maxv)

@njit(parallel=True)
def update_col_inplace_f32(matrix, new_vals, rows, col_idx,
                           row_max_values, row_max_indices,
                           tree_vals, tree_idxs, tree_size):
    
    cols = matrix.shape[1]
    m = rows.shape[0]
    for k in prange(m):
        r = rows[k]
        v = new_vals[k]
        matrix[r, col_idx] = v
        current_max = row_max_values[r]

        if v > current_max:
            row_max_values[r] = v
            row_max_indices[r] = col_idx
            segtree_update(tree_vals, tree_idxs, tree_size, r, v)
        elif col_idx == row_max_indices[r] and v < current_max:
            # recompute row max
            maxv = NEG_INF32
            maxidx = -1
            for j in range(cols):
                vv = matrix[r, j]
                if vv > maxv:
                    maxv = vv
                    maxidx = j
            row_max_values[r] = maxv
            row_max_indices[r] = maxidx
            segtree_update(tree_vals, tree_idxs, tree_size, r, maxv)

class MatrixHeap:

    @profile
    def __init__(self, matrix):
        # Ensure contiguous float32 matrix
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.rows, self.cols = self.matrix.shape

        # allocate caches in float32/int64
        self.row_max_values = np.empty(self.rows, dtype=np.float32)
        self.row_max_indices = np.empty(self.rows, dtype=np.int64)

        # build row maxima in parallel
        build_row_max_parallel(self.matrix, self.row_max_values, self.row_max_indices)

        # build segtree using float32 vals
        self.tree_vals, self.tree_idxs, self.tree_size = segtree_build(self.row_max_values)

    @profile
    def get_max(self):
        val, row_idx = segtree_get_max(self.tree_vals, self.tree_idxs)
        if row_idx < 0:
            return float(NEG_INF32), -1, -1
        col_idx = int(self.row_max_indices[row_idx])
        return float(val), int(row_idx), int(col_idx)

    @profile
    def update_cell(self, row_idx, col_idx, new_value):
        # ensure new_value is float32
        nv = np.float32(new_value)
        update_cell_inplace_f32(self.matrix, row_idx, col_idx, nv,
                                self.row_max_values, self.row_max_indices,
                                self.tree_vals, self.tree_idxs, self.tree_size)

    # FAST versions: no copies or dtype coercion; you promise inputs are correct.
    def update_row_fast(self, row_idx: int, new_values: np.ndarray):
        # REQUIRE: new_values.dtype == np.float32 and C-contiguous
        update_row_inplace_f32(self.matrix, row_idx, new_values,
                               self.row_max_values, self.row_max_indices,
                               self.tree_vals, self.tree_idxs, self.tree_size)

    def update_col_fast(self, new_vals: np.ndarray, rows: np.ndarray, col_idx: int):
        # REQUIRE: new_vals.dtype == np.float32, rows.dtype == np.int64, both C-contiguous
        update_col_inplace_f32(self.matrix, new_vals, rows, col_idx,
                               self.row_max_values, self.row_max_indices,
                               self.tree_vals, self.tree_idxs, self.tree_size)

    # Keep the existing safe versions for debug or when types are unknown:
    def update_row(self, row_idx, new_values):
        if new_values.dtype != np.float32 or not new_values.flags.c_contiguous:
            new_values = np.ascontiguousarray(new_values, dtype=np.float32)
        self.update_row_fast(row_idx, new_values)

    def update_col(self, new_vals, rows, col_idx):
        if not isinstance(rows, np.ndarray) or rows.dtype != np.int64 or not rows.flags.c_contiguous:
            rows = np.ascontiguousarray(rows, dtype=np.int64)
        if not isinstance(new_vals, np.ndarray) or new_vals.dtype != np.float32 or not new_vals.flags.c_contiguous:
            new_vals = np.ascontiguousarray(new_vals, dtype=np.float32)
        self.update_col_fast(new_vals, rows, col_idx)

    @profile
    def get_score_matrix(self):
        return self.matrix

    @profile
    def get_row(self, row_idx):
        return self.matrix[row_idx]
