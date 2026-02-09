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
import math
from joblib import Parallel, delayed
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import shared_memory
import multiprocessing as mp
from numba import njit, prange, set_num_threads
from line_profiler import profile

np.set_printoptions(suppress=True, precision = 4, linewidth = 200)

TEMP_COL = None
TEMP_ROW = None

TOP_K_SCORES = 5

NEG_INF32 = np.float64(-1e30)
NUMBA_THREADS = 1

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
def compute_and_assign_topk_cf(p, x, scores,
                               row_topk_vals, row_topk_idx):
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

            scores[i, j] = val

            if val > row_topk_vals[i, TOP_K_SCORES - 1]:
                k = TOP_K_SCORES - 1
                while k > 0 and val > row_topk_vals[i, k - 1]:
                    row_topk_vals[i, k] = row_topk_vals[i, k - 1]
                    row_topk_idx[i, k] = row_topk_idx[i, k - 1]
                    k -= 1
                row_topk_vals[i, k] = val
                row_topk_idx[i, k] = j


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

@njit(parallel=True)
def compute_score_nf_numba(i, j, x, d):
    c = x[j, i] if j < d else 0.0
    z = x[j, j] if j < d else 0.0

    a = x[i, i]
    b = x[i, j]
    
    t = a*a + c*c
    d1 = a*z - b*c
    d2 = d1*d1
    
    discriminant = t*t - 4*d2
    
    if discriminant < 0:
        discriminant = 0.0
    
    lambda1 = (t + np.sqrt(discriminant)) / 2.0
    lambda2 = (t - np.sqrt(discriminant)) / 2.0
    
    return i, j, np.sqrt(max(lambda1, lambda2)) - a

def rightMatmul_fast(x, i, j, m):
    n_rows, n_cols = x.shape

    m00 = m[0, 0]
    m01 = m[0, 1]
    m10 = m[1, 0]
    m11 = m[1, 1]

    i_in = i < n_cols
    j_in = j < n_cols

    if i_in and j_in:
        TEMP_COL[:n_rows] = x[:, i]
        x[:, i] = m00 * TEMP_COL + m10 * x[:, j]
        x[:, j] = m01 * TEMP_COL + m11 * x[:, j]

    elif i_in:
        x[:, i] *= m00

    elif j_in:
        x[:, j] *= m11


def rightMatmulTranspose_fast(x, i, j, m):
    n_rows, n_cols = x.shape

    m00 = m[0, 0]
    m01 = m[0, 1]
    m10 = m[1, 0]
    m11 = m[1, 1]

    i_in = i < n_cols
    j_in = j < n_cols

    if i_in and j_in:
        TEMP_COL[:n_rows] = x[:, i] 
        x[:, i] = m00 * TEMP_COL + m01 * x[:, j]
        x[:, j] = m10 * TEMP_COL + m11 * x[:, j]

    elif i_in:
        x[:, i] *= m00

    elif j_in:
        x[:, j] *= m11


def leftMatmulTranspose_fast(x, i, j, m):   
    n_rows, n_cols = x.shape

    m00 = m[0, 0]
    m01 = m[0, 1]
    m10 = m[1, 0]
    m11 = m[1, 1]

    i_in = i < n_rows
    j_in = j < n_rows

    if i_in and j_in:
        TEMP_ROW[:n_cols] = x[i, :] 
        x[i, :] = m00 * TEMP_ROW[:n_cols] + m01 * x[j, :]
        x[j, :] = m10 * TEMP_ROW[:n_cols] + m11 * x[j, :]

    elif i_in:
        x[i, :] *= m00

    elif j_in:
        x[j, :] *= m11


@njit
def recompute_row_max(scores, row_max_vals, row_max_idx, r):
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


@njit
def recompute_row_topk(scores, topk_vals, topk_idxs, r):
    row_sz = scores[r].shape[0]

    top_vals = np.full(TOP_K_SCORES, NEG_INF32, dtype=np.float64)
    top_idxs = np.full(TOP_K_SCORES, -1, dtype=np.int32)

    for j in range(row_sz):
        v = scores[r, j]

        if v > top_vals[TOP_K_SCORES - 1]:
            k = TOP_K_SCORES - 1
            while k > 0 and v > top_vals[k - 1]:
                top_vals[k] = top_vals[k - 1]
                top_idxs[k] = top_idxs[k - 1]
                k -= 1
            top_vals[k] = v
            top_idxs[k] = j

    for k in range(TOP_K_SCORES):
        topk_vals[r, k] = top_vals[k]
        topk_idxs[r, k] = top_idxs[k]
 
 
@njit(parallel=True)
def get_new_vals_numba(scores, iq, x, d, row_max_vals, row_max_idx):
    for s in prange(iq + 1, d):
        val = compute_score_nf_numba(iq, s, x, d)[2]
        scores[iq][s] = val
        if val > row_max_vals[iq]:
            row_max_vals[iq] = val
            row_max_idx[iq] = s

@njit(parallel=True)
def get_new_topk_numba(scores, iq, x, d, row_topk_vals, row_topk_idx):
    top_vals = np.full(TOP_K_SCORES, NEG_INF32, dtype=np.float64)
    top_idxs = np.full(TOP_K_SCORES, -1, dtype=np.int32)

    for s in prange(iq + 1, d):
        val = compute_score_nf_numba(iq, s, x, d)[2]
        scores[iq, s] = val

        if val > top_vals[TOP_K_SCORES - 1]:
            k = TOP_K_SCORES - 1
            while k > 0 and val > top_vals[k - 1]:
                top_vals[k] = top_vals[k - 1]
                top_idxs[k] = top_idxs[k - 1]
                k -= 1
            top_vals[k] = val
            top_idxs[k] = s

    for k in range(TOP_K_SCORES):
        row_topk_vals[iq, k] = top_vals[k]
        row_topk_idx[iq, k] = top_idxs[k]


@njit(parallel=True)
def get_new_vals_numba2(scores, jq, x, d, n, row_max_vals, row_max_idx):
    for s in prange(jq + 1, n):
        val = compute_score_nf_numba(jq, s, x, d)[2]
        scores[jq][s] = val
        if val > row_max_vals[jq]:
            row_max_vals[jq] = val
            row_max_idx[jq] = s


@njit(parallel=True)
def get_new_topk_numba2(scores, jq, x, d, n, row_topk_vals, row_topk_idx):
    top_vals = np.full(TOP_K_SCORES, NEG_INF32, dtype=np.float64)
    top_idxs = np.full(TOP_K_SCORES, -1, dtype=np.int32)

    for s in prange(jq + 1, n):
        val = compute_score_nf_numba(jq, s, x, d)[2]
        scores[jq, s] = val

        if val > top_vals[TOP_K_SCORES - 1]:
            k = TOP_K_SCORES - 1
            while k > 0 and val > top_vals[k - 1]:
                top_vals[k] = top_vals[k - 1]
                top_idxs[k] = top_idxs[k - 1]
                k -= 1
            top_vals[k] = val
            top_idxs[k] = s

    for k in range(TOP_K_SCORES):
        row_topk_vals[jq, k] = top_vals[k]
        row_topk_idx[jq, k] = top_idxs[k]


@njit(parallel=True)
def get_new_vals_col_numba(scores, iq, x, d, row_max_vals, row_max_idx):
    for r in prange(iq):
        val = compute_score_nf_numba(r, iq, x, d)[2]
        scores[r][iq] = val
        if val > row_max_vals[r]:
            row_max_vals[r] = val
            row_max_idx[r] = iq
        elif row_max_idx[r] == iq:
            recompute_row_max(scores, row_max_vals, row_max_idx, r)


@njit(parallel=True)
def get_new_topk_col_numba(scores, iq, x, d,
                           row_topk_vals, row_topk_idx):
    for r in prange(iq):
        val = compute_score_nf_numba(r, iq, x, d)[2]
        scores[r, iq] = val

        existing_pos = -1
        for k in range(TOP_K_SCORES):
            if row_topk_idx[r, k] == iq:
                existing_pos = k
                break

        if existing_pos != -1:
            for k in range(existing_pos, TOP_K_SCORES - 1):
                row_topk_vals[r, k] = row_topk_vals[r, k + 1]
                row_topk_idx[r, k] = row_topk_idx[r, k + 1]
            row_topk_vals[r, TOP_K_SCORES - 1] = NEG_INF32
            row_topk_idx[r, TOP_K_SCORES - 1] = -1

        if val > row_topk_vals[r, TOP_K_SCORES - 1]:
            k = TOP_K_SCORES - 1
            while k > 0 and val > row_topk_vals[r, k - 1]:
                row_topk_vals[r, k] = row_topk_vals[r, k - 1]
                row_topk_idx[r, k] = row_topk_idx[r, k - 1]
                k -= 1
            row_topk_vals[r, k] = val
            row_topk_idx[r, k] = iq

        if row_topk_idx[r, 0] == -1: 
            recompute_row_topk(scores, row_topk_vals, row_topk_idx, r)


@njit(parallel=True)
def get_new_vals_col_numba2(scores, jq, x, d, p, row_max_vals, row_max_idx):
    min_val = min(jq, p)
    for r in prange(min_val):
        val = compute_score_nf_numba(r, jq, x, d)[2]
        scores[r][jq] = val
        if val > row_max_vals[r]:
            row_max_vals[r] = val
            row_max_idx[r] = jq
        elif row_max_idx[r] == jq:
            recompute_row_max(scores, row_max_vals, row_max_idx, r)

@njit(parallel=True)
def get_new_topk_col_numba2(scores, jq, x, d, p,
                            row_topk_vals, row_topk_idx):
    min_val = min(jq, p)
    for r in prange(min_val):
        val = compute_score_nf_numba(r, jq, x, d)[2]
        scores[r, jq] = val

        existing_pos = -1
        for k in range(TOP_K_SCORES):
            if row_topk_idx[r, k] == jq:
                existing_pos = k
                break

        if existing_pos != -1:
            for k in range(existing_pos, TOP_K_SCORES - 1):
                row_topk_vals[r, k] = row_topk_vals[r, k + 1]
                row_topk_idx[r, k] = row_topk_idx[r, k + 1]
            row_topk_vals[r, TOP_K_SCORES - 1] = NEG_INF32
            row_topk_idx[r, TOP_K_SCORES - 1] = -1

        if val > row_topk_vals[r, TOP_K_SCORES - 1]:
            k = TOP_K_SCORES - 1
            while k > 0 and val > row_topk_vals[r, k - 1]:
                row_topk_vals[r, k] = row_topk_vals[r, k - 1]
                row_topk_idx[r, k] = row_topk_idx[r, k - 1]
                k -= 1
            row_topk_vals[r, k] = val
            row_topk_idx[r, k] = jq

        if row_topk_idx[r, 0] == -1:
            recompute_row_topk(scores, row_topk_vals, row_topk_idx, r)

@njit()
def get_max(row_max_vals, row_max_idx):
    r = np.argmax(row_max_vals)
    return r, row_max_idx[r]

@njit()
def get_max_topk(row_topk_vals, row_topk_idx):
    max_val = -np.inf
    max_r = -1
    for r in range(row_topk_vals.shape[0]):
        if row_topk_vals[r, 0] > max_val:
            max_val = row_topk_vals[r, 0]
            max_r = r

    # If no valid value, return -1
    if max_r == -1:
        return -1, -1

    return max_r, row_topk_idx[max_r, 0]
                            
# @njit()
def mul_update_numba(x, iq, jq, H, G, u, d, n, p, scores, row_max_vals, row_max_idx, TEMP_COL):
    rightMatmulTranspose_fast(x, iq, jq, H)
    rightMatmul_fast(u, iq, jq, G)

    if jq < d:
        leftMatmulTranspose_fast(x, iq, jq, G) 

    get_new_topk_numba(scores, iq, x, d, row_max_vals, row_max_idx) 

    if jq < p:   
        get_new_topk_numba2(scores, jq, x, d, n, row_max_vals, row_max_idx)   

    get_new_topk_col_numba(scores, iq, x, d, row_max_vals, row_max_idx) 
    get_new_topk_col_numba2(scores, jq, x, d, p, row_max_vals, row_max_idx)

 
# @njit
def fit(x, p, n_iter, u,scores, row_max_vals, row_max_idx, traces, batch_i, TEMP_COL):
    d = x.shape[0]
    n = x.shape[1]
    # compute_and_assign_numba_cf(p, x, scores, row_max_vals, row_max_idx)
    compute_and_assign_topk_cf(p, x, scores, row_max_vals, row_max_idx)
    for q in range(n_iter):
        # iq, jq = get_max(row_max_vals, row_max_idx)
        iq, jq = get_max_topk(row_max_vals, row_max_idx)
        if jq >= d:
            xji = 0
            xjj = 0
        else:
            xji = x[jq, iq]
            xjj = x[jq, jq]
        t = np.zeros((2,2), dtype = np.float64)
        t[0, 0] = x[iq, iq]
        t[0, 1] = x[iq, jq]
        t[1, 0] = xji
        t[1, 1] = xjj
        G, _, H = np.linalg.svd(t)
        # update intermediate x and u
        mul_update_numba(x, iq, jq, H, G, u, d, n, p, scores, row_max_vals, row_max_idx, TEMP_COL)

    return u, x
 
def _init_global_buf(n_rows, n_cols, dtype):
    global TEMP_COL
    global TEMP_ROW

    TEMP_COL = np.zeros(n_rows, dtype=dtype)
    TEMP_ROW = np.zeros(n_cols, dtype=dtype)

def get_evr(x, u, p):
    x_reduced = u.T[:p, :] @ x
    x_recon = u[:, :p] @ x_reduced
    error = np.linalg.norm(x - x_recon, 'fro') ** 2
    total = np.linalg.norm(x, 'fro') ** 2
    return 1 - error / total
        
def fit_batched(trueX, p, n_iter, batch_size=300):
    set_num_threads(NUMBA_THREADS)
    d = trueX.shape[0]
    n = trueX.shape[1]

    if batch_size < d:
        print(f"Batch size too small! Setting to {d}")
        batch_size = d

    _init_global_buf(d, batch_size + p, np.float64)

    total_batches = n // batch_size
    if n % batch_size != 0:
        total_batches += 1
    print("Total batches:", total_batches)

    start_index = 0
    end_index = min(batch_size, n)

    x = trueX[:, start_index:end_index]
    x_batch = np.array(x, copy=True)

    u = np.identity(d)

    scores = np.empty((p, n), dtype=np.float64)
    scores.fill(0.0)
    row_max_vals = np.empty((p, TOP_K_SCORES), dtype=np.float64)
    row_max_vals.fill(NEG_INF32)
    row_max_idx = np.empty((p, TOP_K_SCORES), dtype=np.int64)
    row_max_idx.fill(-1)

    traces = np.zeros(total_batches, dtype=np.float64)
    i = 0
    u, x = fit(x_batch, p, n_iter, u, scores, row_max_vals, row_max_idx, traces, i, TEMP_COL)
   
    traces[i] = 0 #get_evr(trueX, u, p)
    print(f"Done batch {i}")
    while True:
        i += 1
        if end_index == n:
            break

        start_index = start_index + batch_size
        end_index = min(end_index + batch_size, n)

        x_batch = np.hstack((
            x[:, :p],
            u.T @ trueX[:, start_index:end_index]
        ))

        scores.fill(0.0)
        row_max_vals.fill(NEG_INF32)
        row_max_idx.fill(-1)

        u, x = fit(x_batch, p, n_iter, u, scores, row_max_vals, row_max_idx, traces, i, TEMP_COL)
        traces[i] = 0 #get_evr(trueX, u, p)
        print(f"Done batch {i}")

    return traces, u, x
