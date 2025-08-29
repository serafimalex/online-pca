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
from heap import TwoLevelHeap
from numba import njit, prange
from simple_heap import MatrixMaxHeap
from numba_heap import MatrixHeap
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

EPS = 1e-8

class catchtime:
    _totals = defaultdict(lambda: {"time": 0.0, "count": 0})  # stores per-label stats
    
    def __init__(self, debug_mode, log, label=""):
        self.label = label
        self.debug_mode = debug_mode
        self.log = log
 
    def __enter__(self):
        if not self.debug_mode:
            return
        self.start = perf_counter()
        return self
 
    def __exit__(self, type, value, traceback):
        if not self.debug_mode:
            return
        elapsed = perf_counter() - self.start
        self.readout = f'Time: {elapsed:.3f} seconds for {self.label}'
        #self.log.debug(self.readout)

        # accumulate total + count for this label
        catchtime._totals[self.label]["time"] += elapsed
        catchtime._totals[self.label]["count"] += 1

    @classmethod
    def get_stats(cls, label=None):
        """Get stats for a label or all labels.
        Returns dict with total time, count, and average time.
        """
        if label is not None:
            stats = cls._totals.get(label, {"time": 0.0, "count": 0})
            avg = stats["time"] / stats["count"] if stats["count"] else 0.0
            return {"total": stats["time"], "count": stats["count"], "avg": avg}
        result = {}
        for lbl, stats in cls._totals.items():
            avg = stats["time"] / stats["count"] if stats["count"] else 0.0
            result[lbl] = {"total": stats["time"], "count": stats["count"], "avg": avg}
        return result

    @classmethod
    def reset(cls, label=None):
        """Reset stats for a label or all labels."""
        if label is not None:
            cls._totals.pop(label, None)
        else:
            cls._totals.clear()


@njit(parallel=True)
def compute_and_assign_numba_cf(p, x, scores):
    d, n = x.shape

    idx = 0
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

            scores[i, j] = np.float64(val)

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

    return i, j, np.float64(val)

@njit(parallel=True)
def compute_and_assign_numba_fro(p, x, scores):
    d, n = x.shape

    for i in prange(p):
        for j in range(i + 1, n):
            if j >= d:
                xji = 0.0
                xjj = 0.0
            else:
                xji = x[j, i]
                xjj = x[j, j]

            xii = x[i, i]
            xij = x[i, j]

            # Frobenius-based formula
            c1 = (xii * xii + xij * xij + xji * xji + xjj * xjj) / 2.0
            c1_squared = c1 * c1
            det_t = xii * xjj - xij * xji
            c2_squared = det_t * det_t

            value = np.sqrt(c1 + np.sqrt(abs(c1_squared - c2_squared))) - xii

            scores[i, j] = np.float64(value)


@njit(fastmath=True)
def compute_score_fro_numba(i, j, x, d):
    if j >= d:
        xji = 0.0
        xjj = 0.0
    else:
        xji = x[j, i]
        xjj = x[j, j]

    xii = x[i, i]
    xij = x[i, j]

    # Precompute squares once
    xii2 = xii * xii
    xij2 = xij * xij
    xji2 = xji * xji
    xjj2 = xjj * xjj

    # Frobenius norm squared / 2
    c1 = (xii2 + xij2 + xji2 + xjj2) * 0.5
    c1_squared = c1 * c1

    # Determinant squared
    det_t = xii * xjj - xij * xji
    c2_squared = det_t * det_t

    # Guard against negatives (branchless)
    inner = c1_squared - c2_squared
    if inner < 0.0:
        inner = 0.0

    value = sqrt(c1 + sqrt(inner)) - xii
    return i, j, value


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
    # If both are invalid, do nothing
 
@njit(parallel=True)
def get_new_vals_numba(new_vals, iq, x, d):
    for s in prange(iq + 1, d):
        new_vals[s] = compute_score_cf_numba(iq, s, x, d)[2]
    return new_vals

@njit(parallel=True)
def get_new_vals_numba2(new_vals, jq, x, d, n):
    for s in prange(jq + 1, n):
        new_vals[s] = compute_score_cf_numba(jq, s, x, d)[2]
    return new_vals

@njit(parallel=True)
def get_new_vals_col_numba(iq, x, d):
    col_idx = np.arange(iq)
    new_vals = np.empty(iq, dtype = np.float32)
    for r in prange(iq):
        new_vals[r] = compute_score_cf_numba(r, iq, x, d)[2]
    return new_vals, col_idx

@njit(parallel=True)
def get_new_vals_col_numba2(jq, x, d, p):
    min_val = min(jq, p)
    col_idx = np.arange(min_val)
    new_vals = np.empty(min_val, dtype = np.float32)
    for r in prange(min_val):
        new_vals[r] = compute_score_cf_numba(r, jq, x, d)[2]
    return new_vals, col_idx
                            
class ApproxSVD():
 
    def __init__(self, n_iter, p, score_method = 'cf', debug_mode = False, jobs = -1, stored_g = False, use_shared_memory = True, use_heap = False):
        self.n_iter = n_iter
        self.p = p
        self.debug_mode = debug_mode
        self.jobs = jobs
        self.use_shared_memory = use_shared_memory
        self.use_heap = use_heap
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stored_g = stored_g
        if self.stored_g:
            self.g_transforms = []
        if self.debug_mode:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
 
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
 
        match score_method:
            case "fro":
                self.score_fn = compute_score_fro_numba
                self.assign_fn = compute_and_assign_numba_fro
            case "cf":
                self.score_fn = compute_score_cf_numba
                self.assign_fn = compute_and_assign_numba_cf
            case "svd":
                self.score_fn = self.compute_score_svd
            case _:
                raise ValueError("Unknown scoring function")
 
    # we get a d x N matrix x and a 2x2 matrix m
    # columns i and j of x are multiplied with columns 1 and 2 of m
    # equivalent to applying a givens rotation on the matrix
 
    def rightMatmul(self, x, i, j, m):
        # extract columns i and j
 
        if i < x.shape[1]:
            column_i = copy.deepcopy(x[:,i])
        else:
            column_i = np.zeros(x.shape[0])
 
        if j < x.shape[1]:
            column_j = x[:,j]
        else:
            column_j = np.zeros(x.shape[0])
 
        if i < x.shape[1]:
            x[:, i] = m[0][0] * column_i + m[1][0] * column_j
 
        if j < x.shape[1]:    
            x[:, j] = m[0][1] * column_i + m[1][1] * column_j
 
 
    def leftMatmul(self, x, i, j, m):
        row_i = copy.deepcopy(x[i, :])
        row_j = x[j, :]
        x[i, :] = m[0][0] * row_i + m[0][1] * row_j
        x[j, :] = m[1][0] * row_i + m[1][1] * row_j
 
    def rightMatmulTranspose(self, x, i, j, m):
        # extract columns i and j
 
        if i < x.shape[1]:
            column_i = copy.deepcopy(x[:,i])
        else:
            column_i = np.zeros(x.shape[0])
 
        if j < x.shape[1]:
            column_j = x[:,j]
        else:
            column_j = np.zeros(x.shape[0])
 
        if i < x.shape[1]:
            x[:, i] = m[0][0] * column_i + m[0][1] * column_j
 
        if j < x.shape[1]:    
            x[:, j] = m[1][0] * column_i + m[1][1] * column_j
 
 
    def leftMatmulTranspose(self, x, i, j, m):
        row_i = copy.deepcopy(x[i, :])
        row_j = x[j, :]
        x[i, :] = m[0][0] * row_i + m[1][0] * row_j
        x[j, :] = m[0][1] * row_i + m[1][1] * row_j
 
    # score using frobenius norm
    def compute_score_fro(self, i, j, x, d):
        if j >= d:
            xji = 0
            xjj = 0
        else:
            xji = x[j][i]
            xjj = x[j][j]
 
        t = np.array([
                        [x[i][i], x[i][j]],
                        [xji    , xjj]
                    ])
 
        c1 = (np.linalg.norm(t, 'fro') ** 2)/2
        c1_squared = c1 ** 2
        c2_squared = np.linalg.det(t) ** 2
        # due to rounding, use abs below issues
        value = sqrt(c1 + sqrt(abs(c1_squared - c2_squared))) - t[0][0]
 
        return (i, j, value)
 
    # score using closed form of svd
    def compute_score_cf(self, i, j, x, d):
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
    
    # score using explicit svd
    def compute_score_svd(self, i, j, x , d):
        if j >= d:
            xji = 0
            xjj = 0
        else:
            xji = x[j][i]
            xjj = x[j][j]
 
        t = np.array([[x[i][i], x[i][j]],
                      [xji,     xjj]])
        _, s, _ = np.linalg.svd(t)
 
        value = s[0] - x[i][i]
        return (i, j, value)
 
    def build_ubar(self, u):
        for g, i, j in self.g_transforms:
            self.rightMatmul(u, i, j, g)
 
    @staticmethod
    def _compute_score_shared(args):
        i, j, shm_name, shape, dtype, d, score_method = args
 
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        x_shared = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
 
        if score_method == 'cf':
            result = ApproxSVD._compute_score_cf_static(i, j, x_shared, d)
        elif score_method == 'fro':
            result = ApproxSVD._compute_score_fro_static(i, j, x_shared, d)
        elif score_method == 'svd':
            result = ApproxSVD._compute_score_svd_static(i, j, x_shared, d)
        else:
            raise ValueError(f"Unknown scoring method: {score_method}")
 
        # Clean up
        existing_shm.close()
        return result
 
    @staticmethod
    def _compute_score_cf_static(i, j, x, d):
        # Conditional logic for xji, xjj
        if j < d:
            xji = x[j, i]
            xjj = x[j, j]
        else:
            xji = 0.0
            xjj = 0.0

        xii = x[i, i]
        xij = x[i, j]

        if xii * xjj - xij * xji >= 0:
            diff = xij - xji
            value = np.sqrt((xii + xjj)**2 + diff**2) - xii - xjj
        else:
            diff = xij + xji
            value = np.sqrt((xii - xjj)**2 + diff**2) - xii - xjj

        return i, j, value
 
    @staticmethod
    def _compute_score_fro_static(i, j, x, d):
        if j >= d:
            xji = 0
            xjj = 0
        else:
            xji = x[j][i]
            xjj = x[j][j]
 
        t = np.array([
                        [x[i][i], x[i][j]],
                        [xji    , xjj]
                    ])
 
        c1 = (np.linalg.norm(t, 'fro') ** 2)/2
        c1_squared = c1 ** 2
        c2_squared = np.linalg.det(t) ** 2
        value = sqrt(c1 + sqrt(c1_squared - c2_squared)) - t[0][0]
 
        return (i, j, value)
 
    @staticmethod
    def _compute_score_svd_static(i, j, x, d):
        if j >= d:
            xji = 0
            xjj = 0
        else:
            xji = x[j][i]
            xjj = x[j][j]
 
        t = np.array([[x[i][i], x[i][j]],
                      [xji,     xjj]])
        _, s, _ = np.linalg.svd(t)
 
        value = s[0] - x[i][i]
        return (i, j, value)
 
    @profile
    def fit(self, trueX, u = None):
        d = trueX.shape[0]
        n = trueX.shape[1]

        global GLOBAL_ROW1
        global GLOBAL_COLUMN1
        global GLOBAL_COLUMN_ZEROS1
        global GLOBAL_ROW_ZEROS1
        GLOBAL_ROW1 = np.empty(n, dtype=np.float64)
        GLOBAL_COLUMN1 = np.empty(d, dtype=np.float64)
        GLOBAL_ROW_ZEROS1 = np.zeros(n, dtype=np.float64)
        GLOBAL_COLUMN_ZEROS1 = np.zeros(d, dtype=np.float64)

        global GLOBAL_ROW2
        global GLOBAL_COLUMN2
        global GLOBAL_COLUMN_ZEROS2
        global GLOBAL_ROW_ZEROS2
        GLOBAL_ROW2 = np.empty(n, dtype=np.float64)
        GLOBAL_COLUMN2 = np.empty(d, dtype=np.float32)
        GLOBAL_ROW_ZEROS2 = np.zeros(n, dtype=np.float64)
        GLOBAL_COLUMN_ZEROS2 = np.zeros(d, dtype=np.float64)

        if u is None:
            u = np.identity(d)
        x = np.array(trueX, copy=True)
 
        traces = np.array([])
        scores = np.zeros((self.p, n))
        #with catchtime(self.debug_mode, self.logger, "total time"):
 
        #with catchtime(self.debug_mode, self.logger, "initial scores"):
        self.assign_fn(self.p, x, scores)
            

        if self.use_heap == "optimized_heap":
            #with catchtime(self.debug_mode, self.logger, "build heap"):
            self.heap = MatrixHeap(scores)
        elif self.use_heap == "basic_heap":
            #with catchtime(self.debug_mode, self.logger, "build heap"):
            self.heap = MatrixMaxHeap(scores)

        for q in range(self.n_iter):
            # get max score from matrix
            #with catchtime(self.debug_mode, self.logger, "find max score"):
            if self.use_heap == "optimized_heap" or self.use_heap == "basic_heap":
                _, iq, jq = self.heap.get_max()
            else:
                iq, jq = np.unravel_index(np.argmax(scores), scores.shape)

            if jq >= d:
                xji = 0
                xjj = 0
            else:
                xji = x[jq][iq]
                xjj = x[jq][jq]
            #with catchtime(self.debug_mode, self.logger, "perform small svd"):
            t = np.array([
                            [x[iq][iq], x[iq][jq]], 
                            [xji,       xjj]
                        ])
            G, _, H = np.linalg.svd(t)

            # update intermediate x and u
            #with catchtime(self.debug_mode, self.logger, "perform matrix mul"):
            rightMatmulTranspose_numba(x, iq, jq, H)#, GLOBAL_COLUMN1, GLOBAL_COLUMN2, GLOBAL_COLUMN_ZEROS1, GLOBAL_COLUMN_ZEROS2) # equivalent to x @ H.transpose()
            if self.stored_g == False:
                rightMatmul_numba(u, iq, jq, G)#, GLOBAL_COLUMN1, GLOBAL_COLUMN2, GLOBAL_COLUMN_ZEROS1, GLOBAL_COLUMN_ZEROS2) # equivalent to u @ G
            else:
                self.g_transforms.append((G, iq, jq))

            if jq < d:
                leftMatmulTranspose_numba(x, iq, jq, G)#, GLOBAL_ROW1, GLOBAL_ROW2) # equivalent to G.transpose() @ x

            #with catchtime(self.debug_mode, self.logger, "update scores"):
            # update scores
            # Assuming self.use_heap determines whether to use MatrixHeap or plain scores array

            if self.use_heap == "optimized_heap":
                #with catchtime(self.debug_mode, self.logger, "calculate row score"):
                # Update row iq for all columns from iq+1 to d-1
                
                new_values_iq = get_new_vals_numba(self.heap.get_row(iq), iq, x, d)
                # self.heap.get_row(iq)  # convert back from negative
                # for s in range(iq + 1, d):
                #     new_values_iq[s] = self.score_fn(iq, s, x, d)[2]
                #with catchtime(self.debug_mode, self.logger, "rebuild heap"):
                self.heap.update_row_fast(iq, new_values_iq)

                if jq < self.p:
                    # new_values_jq = self.heap.get_row(jq)
                    # for s in range(jq + 1, n):
                    #     new_values_jq[s] = self.score_fn(jq, s, x, d)[2]
                    #with catchtime(self.debug_mode, self.logger, "calculate row score"):
                    new_values_jq = get_new_vals_numba2(self.heap.get_row(jq), jq, x, d, n)
                    #with catchtime(self.debug_mode, self.logger, "rebuild heap"):
                    self.heap.update_row_fast(jq, new_values_jq)

                #with catchtime(self.debug_mode, self.logger, "update cols"):
                new_vals, col_idx = get_new_vals_col_numba(iq, x, d)
                self.heap.update_col_fast(new_vals, col_idx, iq)

                new_vals, col_idx = get_new_vals_col_numba2(jq, x, d, self.p)
                self.heap.update_col_fast(new_vals, col_idx, jq)

            elif self.use_heap == "basic_heap":
                for s in range(iq + 1, d):
                    self.heap.update(iq, s, self.score_fn(iq, s, x, d)[2])

                if jq < self.p:
                    for s in range(jq + 1, n):
                        self.heap.update(jq, s, self.score_fn(jq, s, x, d)[2])

                for r in range(iq):
                    self.heap.update(r, iq, self.score_fn(r, iq, x, d)[2])

                for r in range(min(jq, self.p)):
                    self.heap.update(r, jq, self.score_fn(r, jq, x, d)[2])
            
            else:
                for s in range(iq + 1, d):
                    scores[iq][s] = self.score_fn(iq, s, x, d)[2]
                if jq < self.p:
                    for s in range(jq + 1, n):
                        scores[jq][s] = self.score_fn(jq, s, x, d)[2]
                for r in range(iq):
                    scores[r][iq] = self.score_fn(r, iq, x, d)[2]
                for r in range(min(jq, self.p)):
                    scores[r][jq] = self.score_fn(r, jq, x, d)[2]

            #traces = np.append(traces,  np.trace(x[:self.p, :self.p]))
        if self.stored_g == True:
            self.build_ubar(u)
        #print(catchtime.get_stats())
        #catchtime.reset()
        return traces, u, x
 
    def _compute_initial_scores_shared(self, x, d, scores):
        shm = shared_memory.SharedMemory(create=True, size=x.nbytes)
        x_shared = np.ndarray(x.shape, dtype=x.dtype, buffer=shm.buf)
        x_shared[:] = x[:]  # Copy data to shared memory
 
        try:
            score_method_map = {
                self.compute_score_cf: 'cf',
                self.compute_score_fro: 'fro', 
                self.compute_score_svd: 'svd'
            }
            score_method_str = score_method_map.get(self.score_fn, 'cf')
 
            args_list = [
                (i, j, shm.name, x.shape, x.dtype, d, score_method_str)
                for i in range(self.p)
                for j in range(i + 1, x.shape[1])
            ]
 
            with mp.Pool(processes=self.jobs if self.jobs > 0 else None) as pool:
                results = pool.map(self._compute_score_shared, args_list)
 
            for i, j, value in results:
                scores[i][j] = value
 
        finally:
            shm.close()
            shm.unlink() 

    def fit_transform(self, trueX):
        traces, u, x = self.fit(trueX)
        reducedX = u[:self.p, :] @ trueX

        return traces, reducedX, x

    @profile
    def fit_batched(self, trueX, batch_size = 300):
        d = trueX.shape[0]
        n = trueX.shape[1]
        if batch_size < d:
            self.logger.info("Batch size too small! Setting to %d", d)
            batch_size = d
        start_index = 0
        end_index = min(batch_size, n)
        x_batch = trueX[:, start_index:end_index+1]
        traces = []
        sub_traces, u, x = self.fit(x_batch)
        i = 1
        while True:
            traces.extend(sub_traces)
            if end_index == n:
                break
            start_index = start_index + batch_size
            end_index = min(end_index + batch_size, n)
            x_batch = np.hstack((
                x[:, :self.p],                          
                u @ trueX[:, start_index:end_index+1]  # Matrix multiplication, note +1 because Python slicing is exclusive
            ))
            sub_traces, u, x = self.fit(x_batch, u)
            print(f"Done batch {i}")
            i += 1
        
        return traces, u, x
