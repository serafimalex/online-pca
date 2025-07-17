import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.sparse.linalg import svds
import logging
import time
from time import perf_counter
import math
from joblib import Parallel, delayed
from tqdm import tqdm
from multiprocessing import shared_memory
import multiprocessing as mp
 
np.set_printoptions(suppress=True, precision = 4, linewidth = 200)
 
class catchtime:
    def __init__(self, debug_mode, log, label = ""):
        self.label = label
        self.debug_mode = debug_mode
        self.log = log
 
    def __enter__(self):
        if self.debug_mode == False:
            return
        self.start = perf_counter()
        return self
 
    def __exit__(self, type, value, traceback):
        if self.debug_mode == False:
            return
        self.time = perf_counter() - self.start
        self.readout = f'Time: {self.time:.3f} seconds for {self.label}'
        self.log.debug(self.readout)
 
class ApproxSVD():
 
    def __init__(self, n_iter, p, score_method = 'cf', debug_mode = False, jobs = -1, stored_g = False, use_shared_memory = True):
        self.n_iter = n_iter
        self.p = p
        self.debug_mode = debug_mode
        self.jobs = jobs
        self.use_shared_memory = use_shared_memory
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
                self.score_fn = self.compute_score_fro
            case "cf":
                self.score_fn = self.compute_score_cf
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
        value = math.sqrt(c1 + math.sqrt(c1_squared - c2_squared)) - t[0][0]
 
        return (i, j, value)
 
    # score using closed form of svd
    def compute_score_cf(self, i, j, x, d):
        if j >= d:
            xji = 0
            xjj = 0
        else:
            xji = x[j][i]
            xjj = x[j][j]
 
        if x[i][i] * xjj - x[i][j] * xji >= 0:
            value = math.sqrt((x[i][i] + xjj)**2 + (x[i][j] - xji)**2) - x[i][i] - xjj
        else:
            value = math.sqrt((x[i][i] - xjj)**2 + (x[i][j] + xji)**2) - x[i][i] - xjj
 
        return (i, j, value)
 
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
        if j >= d:
            xji = 0
            xjj = 0
        else:
            xji = x[j][i]
            xjj = x[j][j]
 
        if x[i][i] * xjj - x[i][j] * xji >= 0:
            value = math.sqrt((x[i][i] + xjj)**2 + (x[i][j] - xji)**2) - x[i][i] - xjj
        else:
            value = math.sqrt((x[i][i] - xjj)**2 + (x[i][j] + xji)**2) - x[i][i] - xjj
 
        return (i, j, value)
 
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
        value = math.sqrt(c1 + math.sqrt(c1_squared - c2_squared)) - t[0][0]
 
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
 
 
    def fit(self, trueX):
        d = trueX.shape[0]
        n = trueX.shape[1]
        u = np.identity(d)
        x = copy.deepcopy(trueX)
 
        traces = np.array([])
        scores = np.zeros((self.p, n))
        with catchtime(self.debug_mode, self.logger, "total time"):
 
            with catchtime(self.debug_mode, self.logger, "initial scores"):
                if self.use_shared_memory and self.jobs != 1:
                    self._compute_initial_scores_shared(x, d, scores)
                else:
                    results = Parallel(n_jobs=self.jobs, prefer="processes")(
                        delayed(self.score_fn)(i, j, x, d)
                        for i in range(self.p)
                        for j in range(i + 1, n)
                    )
                    for i, j, value in results:
                        scores[i][j] = value
 
            for q in tqdm(range(self.n_iter)):
                # get max score from matrix
                #with catchtime(self.debug_mode, self.logger, "find max score"):
                iq, jq = np.unravel_index(np.argmax(scores), scores.shape)
 
                if jq >= d:
                    xji = 0
                    xjj = 0
                else:
                    xji = x[jq][iq]
                    xjj = x[jq][jq]
                #with catchtime(self.debug_mode, self.logger, "perform matrix mul"):
                t = np.array([
                                [x[iq][iq], x[iq][jq]], 
                                [xji,       xjj]
                            ])
                G, _, H = np.linalg.svd(t)
 
                # update intermediate x and u
                self.rightMatmulTranspose(x, iq, jq, H) # equivalent to x @ H.transpose()
                if self.stored_g == False:
                    self.rightMatmul(u, iq, jq, G) # equivalent to u @ G
                else:
                    self.g_transforms.append((G, iq, jq))
 
                if jq < d:
                    self.leftMatmulTranspose(x, iq, jq, G) # equivalent to G.transpose() @ x
 
                #with catchtime(self.debug_mode, self.logger, "update scores"):
                # update scores
                for s in range(iq + 1, d):
                    scores[iq][s] = self.score_fn(iq, s, x, d)[2]
 
                if jq < self.p:
                    for s in range(jq + 1, n):
                        scores[jq][s] = self.score_fn(jq, s, x, d)[2]
 
                for r in range(iq):
                    scores[r][iq] = self.score_fn(r, iq, x, d)[2]
 
                for r in range(min(jq, self.p)):
                    scores[r][jq] = self.score_fn(r, jq, x, d)[2]
 
                traces = np.append(traces,  np.trace(x[:self.p, :self.p]))
            if self.stored_g == True:
                self.build_ubar(u)
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