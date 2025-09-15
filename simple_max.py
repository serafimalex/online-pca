import numpy as np
from numba import njit, prange
from line_profiler import profile
# tuned negative infinity for float32
NEG_INF32 = np.finfo(np.float32).min

# ----------------------------
# Numba kernels (no coercion)
# ----------------------------

@njit
def recompute_row_max(row):
    """Compute max and argmax for a 1D float32 row view."""
    max_val = NEG_INF32
    max_idx = -1
    for j in range(row.shape[0]):
        v = row[j]
        if v > max_val:
            max_val = v
            max_idx = j
    return max_val, max_idx


@njit
def update_cell_kernel(matrix, r, c, v, row_max_values, row_max_indices):
    """Update matrix[r,c] and adjust row max if needed."""
    matrix[r, c] = v
    if v > row_max_values[r]:
        row_max_values[r] = v
        row_max_indices[r] = c
    elif row_max_indices[r] == c:
        # previous max overwritten -> recompute
        maxv, maxi = recompute_row_max(matrix[r])
        row_max_values[r] = maxv
        row_max_indices[r] = maxi


@njit
def update_row_kernel(matrix, row_idx, new_vals, row_max_values, row_max_indices):
    """Overwrites a full row (new_vals is a 1D float32 view) and recomputes its max."""
    row = matrix[row_idx]
    max_val = NEG_INF32
    max_idx = -1
    n = row.shape[0]
    for j in range(n):
        v = new_vals[j]
        row[j] = v
        if v > max_val:
            max_val = v
            max_idx = j
    row_max_values[row_idx] = max_val
    row_max_indices[row_idx] = max_idx


@njit(parallel=True, fastmath=True)
def update_column_kernel(matrix, col_idx, rows, new_vals, row_max_values, row_max_indices):
    """
    Updates one column of the matrix and maintains row-wise maxima.
    
    matrix: 2D float32 array
    col_idx: int
    rows: 1D int64 array of row indices to update
    new_vals: 1D float32 array of new values (aligned with rows)
    row_max_values: 1D float32 array storing per-row maxima
    row_max_indices: 1D int64 array storing column index of maxima
    """
    n_cols = matrix.shape[1]

    for k in prange(rows.shape[0]):
        r = rows[k]
        v = new_vals[k]
        matrix[r, col_idx] = v

        # Case 1: New value improves row maximum
        if v > row_max_values[r]:
            row_max_values[r] = v
            row_max_indices[r] = col_idx

        # Case 2: Overwrote the current maximum -> recompute row maximum
        elif row_max_indices[r] == col_idx:
            row_vals = matrix[r, :]  # full row view
            max_val = row_vals[0]
            max_idx = 0
            for j in range(1, n_cols):
                val = row_vals[j]
                if val > max_val:
                    max_val = val
                    max_idx = j
            row_max_values[r] = max_val
            row_max_indices[r] = max_idx


@njit
def get_global_max_kernel(row_max_values, row_max_indices):
    """Return (value, row_idx, col_idx)"""
    # Use np.argmax (numba supports it here)
    ridx = np.argmax(row_max_values)
    return row_max_values[ridx], ridx, row_max_indices[ridx]


# ----------------------------
# Python wrapper (no coercion)
# ----------------------------

class MatrixMax:
    """
    High-performance matrix maxima tracker.
    Assumptions (caller responsibility):
      - matrix is np.float32, C-contiguous
      - rows arrays passed to update_col are np.int64, C-contiguous
      - new_vals arrays passed are np.float32, C-contiguous
      - new_vals for update_row is length == n_cols and float32
      - col_idx is an int scalar
    """
    @profile
    def __init__(self, matrix, enable_global_cache=False):
        # Caller provides float32 contiguous matrix
        self.matrix = matrix
        self.n_rows, self.n_cols = matrix.shape

        # Per-row caches
        self.row_max_values = np.full(self.n_rows, NEG_INF32, dtype=np.float32)
        self.row_max_indices = np.full(self.n_rows, -1, dtype=np.int64)

        # initialize row maxima (uses kernel)
        for r in range(self.n_rows):
            vmax, cidx = recompute_row_max(self.matrix[r])
            self.row_max_values[r] = vmax
            self.row_max_indices[r] = cidx

        # buffers for callers to reuse (optional)
        self._rows_buf = np.empty(self.n_rows, dtype=np.int64)
        self._vals_buf = np.empty(self.n_cols, dtype=np.float32)

        # optional incremental global max caching
        self._global_cache_enabled = bool(enable_global_cache)
        if self._global_cache_enabled:
            vm, ridx, cidx = get_global_max_kernel(self.row_max_values, self.row_max_indices)
            self._global_max_val = vm
            self._global_max_row = ridx
            self._global_max_col = cidx

    # ---- Direct kernels (fast paths) ----
    @profile
    def update_cell(self, r, c, v):
        """Fast kernel call. Caller must pass valid types."""
        update_cell_kernel(self.matrix, int(r), int(c), np.float32(v),
                           self.row_max_values, self.row_max_indices)
        # update cache if enabled
        if self._global_cache_enabled:
            self._maybe_update_global_cache_after_row_change(int(r))

    @profile
    def update_row(self, row_idx, new_vals):
        """
        new_vals: 1D float32 array (length == n_cols). No conversion performed.
        """
        update_row_kernel(self.matrix, int(row_idx), new_vals,
                          self.row_max_values, self.row_max_indices)
        if self._global_cache_enabled:
            self._maybe_update_global_cache_after_row_change(int(row_idx))

    @profile
    def update_col(self, col_idx, rows, new_vals):
        """
        rows: 1D int64 array
        new_vals: 1D float32 array
        col_idx: int scalar
        """
        update_column_kernel(self.matrix, int(col_idx), rows, new_vals,
                             self.row_max_values, self.row_max_indices)
        if self._global_cache_enabled:
            # rows is the set of rows that might have changed, so re-evaluate those
            for r in rows:
                self._maybe_update_global_cache_after_row_change(int(r))

    # ---- Global max access ----
    @profile
    def get_max(self):
        """Return (value, row_idx, col_idx)."""
        if self._global_cache_enabled:
            return float(self._global_max_val), int(self._global_max_row), int(self._global_max_col)
        else:
            v, r, c = get_global_max_kernel(self.row_max_values, self.row_max_indices)
            return float(v), int(r), int(c)

    def enable_global_cache(self):
        vm, r, c = get_global_max_kernel(self.row_max_values, self.row_max_indices)
        self._global_cache_enabled = True
        self._global_max_val = vm
        self._global_max_row = r
        self._global_max_col = c

    def disable_global_cache(self):
        self._global_cache_enabled = False

    @profile
    def _maybe_update_global_cache_after_row_change(self, row_idx):
        """Internal: keep incremental global max cache correct."""
        # If the changed row improved above current global, update.
        rv = self.row_max_values[row_idx]
        if rv > self._global_max_val:
            self._global_max_val = rv
            self._global_max_row = row_idx
            self._global_max_col = self.row_max_indices[row_idx]
            return
        # If the changed row *was* the global row and decreased, we must rescan all rows
        if row_idx == self._global_max_row and rv < self._global_max_val:
            vm, r, c = get_global_max_kernel(self.row_max_values, self.row_max_indices)
            self._global_max_val = vm
            self._global_max_row = r
            self._global_max_col = c

    # ---- helpers for caller reuse of buffers ----
    def rows_buf(self, n):
        """Return a view of internal int64 buffer of length n to fill indices into."""
        return self._rows_buf[:n]

    def vals_buf(self, n_cols):
        """Return a view of internal float32 buffer to fill column or row values."""
        return self._vals_buf[:n_cols]
