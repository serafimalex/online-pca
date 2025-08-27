import heapq
import numpy as np
from numba import njit, prange

# Numba-optimized functions for matrix operations
@njit
def find_row_max(matrix, row_idx):
    """Find maximum value and its index in a row"""
    max_val = -np.inf
    max_idx = -1
    
    for j in range(matrix.shape[1]):
        if matrix[row_idx, j] > max_val:
            max_val = matrix[row_idx, j]
            max_idx = j
            
    return max_val, max_idx

@njit(parallel=True)
def update_matrix_row(matrix, row_idx, new_values):
    """Update a matrix row"""
    for j in prange(matrix.shape[1]):
        matrix[row_idx, j] = new_values[j]
    
    return matrix

@njit(parallel=True)
def update_matrix_cell(matrix, row_idx, col_idx, new_value):
    """Update a single matrix cell"""
    matrix[row_idx, col_idx] = new_value
    return matrix

class MatrixHeap:
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=np.float64)
        self.rows, self.cols = self.matrix.shape
        
        # Precompute row maximums
        self.row_max_values = np.zeros(self.rows, dtype=np.float64)
        self.row_max_indices = np.zeros(self.rows, dtype=np.int64)
        
        # Calculate initial row maximums using Numba-optimized function
        for i in range(self.rows):
            self.row_max_values[i], self.row_max_indices[i] = find_row_max(self.matrix, i)
        
        # Create a heap for global maximum tracking
        self.global_heap = []
        for i in range(self.rows):
            heapq.heappush(self.global_heap, (-self.row_max_values[i], i))

    def get_max(self):
        """Get the global maximum value and its position"""
        if not self.global_heap:
            return -np.inf, -1, -1
            
        # Find the row with the current maximum
        while self.global_heap:
            neg_val, row_idx = self.global_heap[0]
            current_max = -neg_val
            
            # Check if this row's max is still valid
            if current_max == self.row_max_values[row_idx]:
                break
                
            heapq.heappop(self.global_heap)
        else:
            return -np.inf, -1, -1
            
        col_idx = self.row_max_indices[row_idx]
        return current_max, row_idx, col_idx

    def update_row(self, row_idx, new_values):
        """Update an entire row"""
        # Update the matrix using Numba-optimized function
        self.matrix = update_matrix_row(self.matrix, row_idx, new_values)
        
        # Update row maximum using Numba-optimized function
        self.row_max_values[row_idx], self.row_max_indices[row_idx] = find_row_max(self.matrix, row_idx)
        
        # Push updated row to global heap
        heapq.heappush(self.global_heap, (-self.row_max_values[row_idx], row_idx))

    def update_cell_wrapper(self, row_idx, col_idx, new_value):
        """Update a single cell"""
        # Update the matrix using Numba-optimized function
        self.matrix = update_matrix_cell(self.matrix, row_idx, col_idx, new_value)
        
        # Check if we need to update the row maximum
        current_max = self.row_max_values[row_idx]
        if new_value > current_max:
            self.row_max_values[row_idx] = new_value
            self.row_max_indices[row_idx] = col_idx
            heapq.heappush(self.global_heap, (-new_value, row_idx))
        elif col_idx == self.row_max_indices[row_idx] and new_value < current_max:
            # The previous maximum was decreased, need to recalc row maximum
            self.row_max_values[row_idx], self.row_max_indices[row_idx] = find_row_max(self.matrix, row_idx)
            heapq.heappush(self.global_heap, (-self.row_max_values[row_idx], row_idx))
    
    def update_col(self, new_vals, rows, col_idx):
        vals = new_vals.shape[0]
        for idx in range(vals):
            self.update_cell_wrapper(rows[idx], col_idx, new_vals[idx])


    def get_score_matrix(self):
        """Get the entire matrix"""
        return self.matrix.copy()

    def get_row(self, row_idx):
        """Get a specific row"""
        return self.matrix[row_idx].copy()