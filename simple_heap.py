import numpy as np
import numba as nb

# --- compiled helpers ---

@nb.njit
def _init_row_max(matrix):
    R, C = matrix.shape
    row_max_val = np.empty(R, dtype=matrix.dtype)
    row_max_j   = np.empty(R, dtype=np.int64)
    for i in range(R):
        max_val = matrix[i, 0]
        max_j = 0
        for j in range(1, C):
            v = matrix[i, j]
            if v > max_val:
                max_val = v
                max_j = j
        row_max_val[i] = max_val
        row_max_j[i] = max_j
    return row_max_val, row_max_j

@nb.njit
def _update_point(matrix, i, j, val, row_max_val, row_max_j):
    matrix[i, j] = val
    # check row max
    if j == row_max_j[i]:
        if val >= row_max_val[i]:
            row_max_val[i] = val
        else:
            # rescan row
            C = matrix.shape[1]
            max_val = matrix[i, 0]
            max_j = 0
            for jj in range(1, C):
                v = matrix[i, jj]
                if v > max_val:
                    max_val = v
                    max_j = jj
            row_max_val[i] = max_val
            row_max_j[i] = max_j
    else:
        if val > row_max_val[i]:
            row_max_val[i] = val
            row_max_j[i] = j

@nb.njit
def _get_global_max(row_max_val, row_max_j):
    R = row_max_val.shape[0]
    max_val = row_max_val[0]
    max_i = 0
    for i in range(1, R):
        v = row_max_val[i]
        if v > max_val:
            max_val = v
            max_i = i
    return max_val, max_i, row_max_j[max_i]

# --- user-facing class ---

class MatrixMaxHeap:
    def __init__(self, matrix: np.ndarray):
        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
            raise ValueError("matrix must be a 2D NumPy array")
        self.matrix = matrix
        self.row_max_val, self.row_max_j = _init_row_max(self.matrix)

    def update(self, i: int, j: int, val):
        _update_point(self.matrix, i, j, val, self.row_max_val, self.row_max_j)

    def get_max(self):
        return _get_global_max(self.row_max_val, self.row_max_j)