import numpy as np
from numba import njit, prange, float32, int64

NEG_INF32 = np.finfo(np.float32).min

# -------------------------------
# Numba kernels
# -------------------------------

@njit
def compute_row_max(matrix, row_idx):
    """Compute maximum value and its column index for a single row."""
    max_val = NEG_INF32
    max_idx = -1
    n_cols = matrix.shape[1]
    for j in range(n_cols):
        val = matrix[row_idx, j]
        if val > max_val:
            max_val = val
            max_idx = j
    return max_val, max_idx

@njit
def update_row(matrix, row_idx, new_values, row_max_values, row_max_indices):
    """Update an entire row and recompute its maximum."""
    n_cols = matrix.shape[1]
    for j in range(n_cols):
        matrix[row_idx, j] = new_values[j]
    max_val, max_idx = compute_row_max(matrix, row_idx)
    row_max_values[row_idx] = max_val
    row_max_indices[row_idx] = max_idx

@njit
def update_cell(matrix, row_idx, col_idx, new_value, row_max_values, row_max_indices):
    """Update a single cell and update the row maximum if needed."""
    matrix[row_idx, col_idx] = new_value
    old_max = row_max_values[row_idx]
    old_max_col = row_max_indices[row_idx]

    if new_value > old_max:
        # New max in this row
        row_max_values[row_idx] = new_value
        row_max_indices[row_idx] = col_idx
    elif col_idx == old_max_col and new_value < old_max:
        # Old max decreased, recompute row max
        max_val, max_idx = compute_row_max(matrix, row_idx)
        row_max_values[row_idx] = max_val
        row_max_indices[row_idx] = max_idx
    # else: old max still valid, do nothing

@njit(parallel=True)
def update_column(matrix, col_idx, rows, new_vals, row_max_values, row_max_indices):
    """
    Update a column for multiple rows.
    rows: array of row indices
    new_vals: array of new values corresponding to rows
    """
    n_cols = matrix.shape[1]
    n_rows = len(rows)

    for k in prange(n_rows):
        r = rows[k]
        v = new_vals[k]
        matrix[r, col_idx] = v
        old_max = row_max_values[r]
        old_max_col = row_max_indices[r]

        if v > old_max:
            row_max_values[r] = v
            row_max_indices[r] = col_idx
        elif col_idx == old_max_col and v < old_max:
            # Old max decreased, recompute row max
            max_val = NEG_INF32
            max_idx = -1
            for j in range(n_cols):
                val = matrix[r, j]
                if val > max_val:
                    max_val = val
                    max_idx = j
            row_max_values[r] = max_val
            row_max_indices[r] = max_idx
        # else: old max still valid

@njit
def get_global_max(row_max_values, row_max_indices):
    """Return global maximum and its position (row, col)."""
    row_idx = np.argmax(row_max_values)
    return row_max_values[row_idx], row_idx, row_max_indices[row_idx]

# -------------------------------
# Python wrapper class
# -------------------------------

class MatrixMax:
    """
    Optimized matrix maximum tracker.
    Only keeps per-row maxima (no heap/tree).
    Supports row and column updates.
    """

    def __init__(self, matrix):
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.num_rows, self.num_cols = self.matrix.shape

        # Row maxima
        self.row_max_values = np.empty(self.num_rows, dtype=np.float32)
        self.row_max_indices = np.empty(self.num_rows, dtype=np.int64)

        # Initialize row maxima
        for i in range(self.num_rows):
            max_val, max_idx = compute_row_max(self.matrix, i)
            self.row_max_values[i] = max_val
            self.row_max_indices[i] = max_idx

    def get_max(self):
        """Return global maximum (value, row_idx, col_idx)."""
        return get_global_max(self.row_max_values, self.row_max_indices)

    def update_row(self, row_idx, new_values):
        """Update an entire row."""
        new_values = np.ascontiguousarray(new_values, dtype=np.float32)
        update_row(self.matrix, row_idx, new_values, self.row_max_values, self.row_max_indices)

    def update_cell(self, row_idx, col_idx, new_value):
        """Update a single cell."""
        update_cell(self.matrix, row_idx, col_idx, np.float32(new_value),
                    self.row_max_values, self.row_max_indices)

    def update_col(self, new_vals, rows, col_idx):
        """Update a column for multiple rows."""
        rows_arr = np.ascontiguousarray(rows, dtype=np.int64)
        vals_arr = np.ascontiguousarray(new_vals, dtype=np.float32)
        update_column(self.matrix, col_idx, rows_arr, vals_arr, self.row_max_values, self.row_max_indices)

    def get_row(self, row_idx):
        """Return a view of a row (no copy)."""
        return self.matrix[row_idx]

    def get_matrix(self):
        """Return the whole matrix (copy)."""
        return self.matrix.copy()
