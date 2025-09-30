import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.sparse.linalg import svds
import logging
import time
from time import perf_counter
from numba import njit
import numpy as np
from math import sqrt
from joblib import Parallel, delayed
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import shared_memory
import multiprocessing as mp
from numba import njit, prange, set_num_threads
from line_profiler import profile

np.set_printoptions(suppress=True, precision = 4, linewidth = 200)

GLOBAL_ROW1 = None
GLOBAL_COLUMN1 = None
GLOBAL_COLUMN_ZEROS1 = None
GLOBAL_ROW_ZEROS1 = None

GLOBAL_ROW2 = None
GLOBAL_COLUMN2 = None
GLOBAL_COLUMN_ZEROS2 = None
GLOBAL_ROW_ZEROS2 = None

NEG_INF32 = np.float32(-1e30)
NUMBA_THREADS = 8

@njit(parallel=True)
def compute_and_assign_numba_cf(p, x, scores, row_max_vals, row_max_idx):
    d, n = x.shape

    for i in prange(p):
        for j in range(i + 1, n):
            xji = x[j, i] if j < d else 0.0
            xjj = x[j, j] if j < d else 0.0

            xii = x[i, i]
            xij = x[i, j]

            if xii * xjj - xij * xji >= 0:
                diff = xij - xji
                val = np.sqrt((xii + xjj)**2 + diff**2) - xii - xjj
            else:
                diff = xij + xji
                val = np.sqrt((xii - xjj)**2 + diff**2) - xii - xjj
            if val > row_max_vals[i]:
                row_max_vals[i] = val
                row_max_idx[i] = j
            scores[i, j] = val

@njit(parallel=True)
def compute_score_cf_numba(i, j, x, d):
    xji = x[j, i] if j < d else 0.0
    xjj = x[j, j] if j < d else 0.0

    xii = x[i, i]
    xij = x[i, j]

    if xii * xjj - xij * xji >= 0:
        diff = xij - xji
        val = np.sqrt((xii + xjj)**2 + diff**2) - xii - xjj
    else:
        diff = xij + xji
        val = np.sqrt((xii - xjj)**2 + diff**2) - xii - xjj

    return i, j, val

@njit(parallel=True, fastmath=True)
def rightMatmul_numba(x, i, j, m):
    n_rows, n_cols = x.shape

    # Precompute matrix elements
    m00 = m[0, 0]
    m01 = m[0, 1]
    m10 = m[1, 0]
    m11 = m[1, 1]

    # Handle all cases without temporary copies
    if i < n_cols and j < n_cols:
        # Both columns are valid - process together
        for r in prange(n_rows):
            temp_i = x[r, i]
            temp_j = x[r, j]
            x[r, i] = m00 * temp_i + m10 * temp_j
            x[r, j] = m01 * temp_i + m11 * temp_j
    elif i < n_cols:
        # Only column i is valid
        for r in prange(n_rows):
            x[r, i] = m00 * x[r, i]
    elif j < n_cols:
        # Only column j is valid
        for r in prange(n_rows):
            x[r, j] = m11 * x[r, j]
    # If both are invalid, do nothing

@njit(parallel=True, fastmath=True)
def rightMatmulTranspose_numba(x, i, j, m):
    n_rows, n_cols = x.shape

    # Precompute matrix elements
    m00 = m[0, 0]
    m01 = m[0, 1]
    m10 = m[1, 0]
    m11 = m[1, 1]

    # Handle all cases without temporary copies
    if i < n_cols and j < n_cols:
        # Both columns are valid - process together
        for r in prange(n_rows):
            temp_i = x[r, i]
            temp_j = x[r, j]
            x[r, i] = m00 * temp_i + m01 * temp_j
            x[r, j] = m10 * temp_i + m11 * temp_j
    elif i < n_cols:
        # Only column i is valid
        for r in prange(n_rows):
            temp_i = x[r, i]
            x[r, i] = m00 * temp_i  # m01 * 0 (j column is treated as zeros)
    elif j < n_cols:
        # Only column j is valid
        for r in prange(n_rows):
            temp_j = x[r, j]
            x[r, j] = m11 * temp_j  # m10 * 0 (i column is treated as zeros)

# @njit(parallel=True)
# def rightMatmul_numba(x, i, j, m, TEMP_COL_2):
#     n_rows, n_cols = x.shape
 
#     if i < n_cols and j < n_cols:
#         # backup original column i
#         TEMP_COL_2[:] = x[:, i]
 
#         # compute new column i
#         x[:, i] = m[0, 0] * x[:, i] + m[1, 0] * x[:, j]
 
#         # compute new column j
#         x[:, j] = m[0, 1] * TEMP_COL_2 + m[1, 1] * x[:, j]
 
#     elif i < n_cols:
#         x[:, i] *= m[0, 0]
 
#     # Only column j exists
#     elif j < n_cols:
#         x[:, j] *= m[1, 1]


@njit(parallel=True, fastmath=True)
def leftMatmul_numba(x, i, j, m,  GLOBAL_ROW1, GLOBAL_ROW2, GLOBAL_ROW_ZEROS1, GLOBAL_ROW_ZEROS2):
    n_cols = x.shape[1]

    for c in prange(n_cols):
        GLOBAL_ROW1[c] = x[i, c]
        GLOBAL_ROW2[c] = x[j, c]
    
    row_i = GLOBAL_ROW1
    row_j = GLOBAL_ROW2

    for c in prange(n_cols):
        x[i, c] = m[0, 0] * row_i[c] + m[0, 1] * row_j[c]
        x[j, c] = m[1, 0] * row_i[c] + m[1, 1] * row_j[c]



@njit(parallel=True, fastmath=True)
def leftMatmulTranspose_numba(x, i, j, m):
    n_rows, n_cols = x.shape

    # Precompute matrix elements
    m00 = m[0, 0]
    m01 = m[0, 1]
    m10 = m[1, 0]
    m11 = m[1, 1]

    if i < n_rows and j < n_rows:
        # Both rows are valid
        for c in prange(n_cols):
            temp_i = x[i, c]
            temp_j = x[j, c]
            x[i, c] = m00 * temp_i + m01 * temp_j
            x[j, c] = m10 * temp_i + m11 * temp_j
    elif i < n_rows:
        # Only row i is valid
        for c in prange(n_cols):
            x[i, c] = m00 * x[i, c]
    elif j < n_rows:
        # Only row j is valid
        for c in prange(n_cols):
            x[j, c] = m11 * x[j, c]
    # If both are invalid → do nothing


@njit
def recompute_row_max(scores, row_max_vals, row_max_idx, r):
    """Fast row maximum recomputation (SIMD-friendly)."""
    max_val = NEG_INF32
    max_idx = -1
    row_sz = scores[r].shape[0]
    for j in range(row_sz):
        v = scores[r][j]
        if v > max_val:
            max_val = v
            max_idx = j
    row_max_vals[r] = max_val
    row_max_idx[r] = max_idx
 
@njit(parallel=True)
def get_new_vals_numba(scores, iq, x, d, row_max_vals, row_max_idx):
    for s in prange(iq + 1, d):
        val = compute_score_cf_numba(iq, s, x, d)[2]
        scores[iq][s] = val
        if val > row_max_vals[iq]:
            row_max_vals[iq] = val
            row_max_idx[iq] = s

@njit(parallel=True)
def get_new_vals_numba2(scores, jq, x, d, n, row_max_vals, row_max_idx):
    for s in prange(jq + 1, n):
        val = compute_score_cf_numba(jq, s, x, d)[2]
        scores[jq][s] = val
        if val > row_max_vals[jq]:
            row_max_vals[jq] = val
            row_max_idx[jq] = s

@njit(parallel=True)
def get_new_vals_col_numba(scores, iq, x, d, row_max_vals, row_max_idx):
    for r in prange(iq):
        val = compute_score_cf_numba(r, iq, x, d)[2]
        scores[r][iq] = val
        if val > row_max_vals[r]:
            row_max_vals[r] = val
            row_max_idx[r] = iq
        elif row_max_idx[r] == iq:
            recompute_row_max(scores, row_max_vals, row_max_idx, r)

@njit(parallel=True)
def get_new_vals_col_numba2(scores, jq, x, d, p, row_max_vals, row_max_idx):
    min_val = min(jq, p)
    for r in prange(min_val):
        val = compute_score_cf_numba(r, jq, x, d)[2]
        scores[r][jq] = val
        if val > row_max_vals[r]:
            row_max_vals[r] = val
            row_max_idx[r] = jq
        elif row_max_idx[r] == jq:
            recompute_row_max(scores, row_max_vals, row_max_idx, r)

@njit()
def get_max(row_max_vals, row_max_idx):
    r = np.argmax(row_max_vals)
    return r, row_max_idx[r]
                            
#@njit()
@njit()
def mul_update_numba(x, iq, jq, H, G, u, d, n, p, scores, row_max_vals, row_max_idx, TEMP_COL):
    rightMatmulTranspose_numba(x, iq, jq, H)
    rightMatmul_numba(u, iq, jq, G)

    if jq < d:
        leftMatmulTranspose_numba(x, iq, jq, G)
    get_new_vals_numba(scores, iq, x, d, row_max_vals, row_max_idx)   
    if jq < p:
        get_new_vals_numba2(scores, jq, x, d, n, row_max_vals, row_max_idx)   
    get_new_vals_col_numba(scores, iq, x, d, row_max_vals, row_max_idx) 
    get_new_vals_col_numba2(scores, jq, x, d, p, row_max_vals, row_max_idx)

 
@njit
def fit(x, p, n_iter, u,scores, row_max_vals, row_max_idx, traces, batch_i, TEMP_COL):
    d = x.shape[0]
    n = x.shape[1]
    compute_and_assign_numba_cf(p, x, scores, row_max_vals, row_max_idx)
    for q in range(n_iter):
        # get max score from matrix
        iq, jq = get_max(row_max_vals, row_max_idx)
        if jq >= d:
            xji = 0
            xjj = 0
        else:
            xji = x[jq, iq]
            xjj = x[jq, jq]
        t = np.zeros((2,2), dtype = np.float32)
        t[0, 0] = x[iq, iq]
        t[0, 1] = x[iq, jq]
        t[1, 0] = xji
        t[1, 1] = xjj
        G, _, H = np.linalg.svd(t)
        # update intermediate x and u
        mul_update_numba(x, iq, jq, H, G, u, d, n, p, scores, row_max_vals, row_max_idx, TEMP_COL)
        traces[n_iter*batch_i + q]= np.trace(x[:p, :p])


        #traces = np.append(traces,  np.trace(x[:self.p, :self.p]))
    #print(catchtime.get_stats())
    #catchtime.reset()
    return u, x
 
BUF = None
TEMP_COL = None
def _init_global_buf(n_rows, dtype):
    global BUF
    global TEMP_COL
    if BUF is None:
        BUF = np.zeros((n_rows, 2), dtype=dtype)
        TEMP_COL = np.zeros(n_rows, dtype=dtype)
        
def fit_batched(trueX, p, n_iter, batch_size=300):
    set_num_threads(NUMBA_THREADS)
    d = trueX.shape[0]
    n = trueX.shape[1]
    _init_global_buf(d, np.float32)
    TEMP_COL = np.zeros(d, dtype=np.float32)

    if batch_size < d:
        print(f"Batch size too small! Setting to {d}")
        batch_size = d

    total_batches = n // batch_size
    if n % batch_size != 0:
        total_batches += 1
    print("Total batches:", total_batches)

    start_index = 0
    end_index = min(batch_size, n)

    # Note: original code used end_index+1; keeping that behavior
    x = trueX[:, start_index:end_index+1]
    x_batch = np.array(x, copy=True)

    u = np.identity(d)
    # Allocate once, reuse
    scores = np.empty((p, n), dtype=np.float32); scores.fill(0.0)
    row_max_vals = np.empty(p, dtype=np.float32); row_max_vals.fill(NEG_INF32)
    row_max_idx = np.empty(p, dtype=np.int64); row_max_idx.fill(-1)

    traces = np.zeros(n_iter * total_batches, dtype=np.float32)
    i = 0
    u, x = fit(x_batch, p, n_iter, u, scores, row_max_vals, row_max_idx, traces, i, TEMP_COL)

    while True:
        i += 1
        if end_index == n:
            break

        start_index = start_index + batch_size
        end_index = min(end_index + batch_size, n)

        x_batch = np.hstack((
            x[:, :p],
            u.T @ trueX[:, start_index:end_index+1]
        ))

        # Reset buffers in-place instead of reallocating
        scores.fill(0.0)
        row_max_vals.fill(NEG_INF32)
        row_max_idx.fill(-1)

        u, x = fit(x_batch, p, n_iter, u, scores, row_max_vals, row_max_idx, traces, i, TEMP_COL)
        print(f"Done batch {i}")

    return traces, u, x
