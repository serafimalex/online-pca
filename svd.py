import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.sparse.linalg import svds
import logging
import time
from time import perf_counter
import math
from joblib import Parallel, delayed

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

    def __init__(self, n_iter, p, debug_mode = False):
        self.n_iter = n_iter
        self.p = p
        self.debug_mode = debug_mode

        self.logger = logging.getLogger(self.__class__.__name__)
        if self.debug_mode:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)


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


    def compute_score(self, i, j, x, d):
        if j >= d:
            xji = 0
            xjj = 0
        else:
            xji = x[j][i]
            xjj = x[j][j]

        if x[i][j] * xjj - x[i][j] * xji >= 0:
            value = math.sqrt((x[i][i] + xjj)**2 + (x[i][j] - xji)**2) - x[i][i] - xjj
        else:
            value = math.sqrt((x[i][i] - xjj)**2 + (x[i][j] + xji)**2) - x[i][i] - xjj

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
                results = Parallel(n_jobs=1, prefer="processes")(
                    delayed(self.compute_score)(i, j, x, d)
                    for i in range(self.p)
                    for j in range(i + 1, n)
                )

            for i, j, value in results:
                scores[i][j] = value
            
            for q in range(self.n_iter):
                # get max score from matrix
                with catchtime(self.debug_mode, self.logger, "find max score"):
                    iq, jq = np.unravel_index(np.argmax(scores), scores.shape)
                
                if jq >= d:
                    xji = 0
                    xjj = 0
                else:
                    xji = x[jq][iq]
                    xjj = x[jq][jq]
                with catchtime(self.debug_mode, self.logger, "perform matrix mul"):
                    t = np.array([
                                    [x[iq][iq], x[iq][jq]], 
                                    [xji,       xjj]
                                ])
                    G, _, H = np.linalg.svd(t)

                    # update intermediate x and u
                    self.rightMatmul(x, iq, jq, H.transpose()) # equivalent to x @ H.transpose()
                    self.rightMatmul(u, iq, jq, G) # equivalent to u @ G
                    
                    if jq < d:
                        self.leftMatmul(x, iq, jq, G.transpose()) # equivalent to G.transpose() @ x
                
                with catchtime(self.debug_mode, self.logger, "update scores"):
                    # update scores
                    for s in range(iq + 1, d):
                        t = np.array([
                                        [x[iq][iq], x[iq][s]],
                                        [x[s][iq], x[s][s]]
                                    ])
                        _, sig, _ = np.linalg.svd(t)
                        scores[iq][s] = sig[0] - x[iq][iq]
                        # if x[iq][iq] * x[s][s] - x[iq][s] *x[s][iq] >= 0:
                        #     scores[iq][s] = math.sqrt((x[iq][iq] + x[s][s])**2 + (x[iq][s] - x[s][iq])**2) - x[iq][iq] - x[s][s]
                        # else:
                        #     scores[iq][s] = math.sqrt((x[iq][iq] - x[s][s])**2 + (x[iq][s] + x[s][iq])**2) - x[iq][iq] - x[s][s]


                    if jq < self.p:
                        for s in range(jq + 1, n):
                            if s >= d:
                                xsjq = 0
                                xss = 0
                            else:
                                xsjq = x[s][jq]
                                xss = x[s][s]

                            t = np.array([
                                            [x[jq][jq], x[jq][s]],
                                            [xsjq, xss]
                                        ])
                            _, sig, _ = np.linalg.svd(t)
                            scores[jq][s] = sig[0] - x[jq][jq]
                            # if x[jq][jq] * xss - x[jq][s] *xsjq >= 0:
                            #     scores[jq][s] = math.sqrt((x[jq][jq] + xss)**2 + (x[jq][s] - xsjq)**2) - x[jq][jq] - xss
                            # else:
                            #     scores[jq][s] = math.sqrt((x[jq][jq] - xss)**2 + (x[jq][s] + xsjq)**2) - x[jq][jq] - xss
                        
                    for r in range(iq):
                        t = np.array([[x[r][r], x[r][iq]],
                                    [x[iq][r], x[iq][iq]]])
                        _, sig, _ = np.linalg.svd(t)
                        scores[r][iq] = sig[0] - x[r][r]
                        # if x[r][iq] * x[iq][iq] - x[r][iq] *x[iq][r] >= 0:
                        #     scores[r][iq] = math.sqrt((x[r][r] + x[iq][iq])**2 + (x[r][iq] - x[iq][r])**2) - x[r][r] - x[iq][iq]
                        # else:
                        #     scores[r][iq] = math.sqrt((x[r][r] - x[iq][iq])**2 + (x[r][iq] + x[iq][r])**2) - x[r][r] - x[iq][iq]
                    
                    for r in range(min(jq, self.p)):
                        if jq >= d:
                            x_jq_r = 0
                            x_jq_jq = 0
                        else:
                            x_jq_r = x[jq][r]
                            x_jq_jq = x[jq][jq]

                        t = np.array([
                                        [x[r][r], x[r][jq]],
                                        [x_jq_r, x_jq_jq]
                                    ])
                        _, sig, _ = np.linalg.svd(t)

                        scores[r][jq] = sig[0] - x[r][r]
                        # if x[r][jq] * x_jq_jq - x[r][jq] * x_jq_r >= 0:
                        #     scores[r][jq] = math.sqrt((x[r][r] + x_jq_jq)**2 + (x[r][jq] - x_jq_r)**2) - x[r][r] - x_jq_jq
                        # else:
                        #     scores[r][jq] = math.sqrt((x[r][r] - x_jq_jq)**2 + (x[r][jq] + x_jq_r)**2) - x[r][r] - x_jq_jq
                    
                traces = np.append(traces,  np.trace(x[:self.p, :self.p]))
        
        return traces, u, x

        
    def fit_transform(self, trueX):
        traces, u, x = self.fit(trueX)
        reducedX = u[:self.p, :] @ trueX
        return traces, reducedX, x

            

    