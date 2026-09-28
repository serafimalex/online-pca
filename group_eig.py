"""
group_eig.py

Group / Block Online EIG.

This is the symmetric (covariance-style) analog of group_svd.py, and the
group/block extension of pairwise_eig.py.

Algorithm (per group iteration):
    1. For each i in [0, p), scan all j > i and keep top_m best j-candidates
       by score C_ij = lambda_max(S_2x2) - S_ii.
    2. Greedily assign one unused j to each row i. Build index set
           idx = [0, 1, ..., p-1,  j_0, j_1, ..., j_{p-1}]   (size 2p)
    3. Extract block B = S[idx, idx] (size 2p x 2p, symmetric).
    4. Eigendecompose: B = G_local @ Lambda @ G_local^T.
    5. Apply similarity transform on the chosen slice of S and right-multiply U:
           S[idx, :]   = G_local^T @ S[idx, :]
           S[:, idx]   = S[:, idx]   @ G_local
           U[:, idx]   = U[:, idx]   @ G_local
"""
import time
from contextlib import nullcontext

import numpy as np

from validation import check_batch, check_init

try:
    from threadpoolctl import ThreadpoolController
except ImportError:
    ThreadpoolController = None

TOP_CANDIDATES_PER_ROW = 32
NEG_INF = np.float64(-1e30)


def _gram(X, S=None):
    """
    Form/add X X.T using the same NumPy BLAS runtime as the other matmuls.
    """
    if not (X.flags.c_contiguous or X.flags.f_contiguous):
        X = np.ascontiguousarray(X)
    C = X @ X.T
    if S is None:
        return C
    S += C
    return S


def compute_row_top_candidates_eig(S, p, top_vals, top_idxs):
    n = S.shape[0]
    top_m = top_vals.shape[1]

    diag = np.diag(S)
    di = diag[:p, None]
    half_dif = 0.5 * (di - diag[None, :])
    vals = 0.5 * (di + diag[None, :]) + np.sqrt(half_dif * half_dif + S[:p, :] ** 2) - di
    # j <= i is not a candidate
    vals[np.arange(n)[None, :] <= np.arange(p)[:, None]] = NEG_INF

    m = min(top_m, n)
    part = np.argpartition(-vals, m - 1, axis=1)[:, :m]
    rows = np.arange(p)[:, None]
    part = part[rows, np.argsort(-vals[rows, part], axis=1)]

    top_vals[:] = NEG_INF
    top_idxs[:] = -1
    picked = vals[rows, part]
    valid = picked > NEG_INF
    top_vals[:, :m] = np.where(valid, picked, NEG_INF)
    top_idxs[:, :m] = np.where(valid, part, -1)


def choose_group_from_top_candidates(top_idxs, p, n_active):
    used = set(range(p))
    indices = list(range(p))

    for row in top_idxs[:p]:
        for value in row:
            j = int(value)
            if p <= j < n_active and j not in used:
                indices.append(j)
                used.add(j)
                break
    return np.asarray(indices, dtype=np.int64)


class _Workspace:
    def __init__(self, n, p, top_m):
        self.n, self.p, self.top_m = n, p, top_m
        self.scores = np.empty((p, n - p), dtype=np.float64)
        self.scratch = np.empty_like(self.scores)
        self.indices = np.empty(min(n, 2 * p), dtype=np.int64)
        self.indices[:p] = np.arange(p)
        self.top_vals = self.top_idxs = None
        if top_m < p:
            self.top_vals = np.empty((p, min(top_m, n)), dtype=np.float64)
            self.top_idxs = np.empty_like(self.top_vals, dtype=np.int64)

    def select(self, S):
        p, n = self.p, self.n
        if self.top_m < p:
            compute_row_top_candidates_eig(S, p, self.top_vals, self.top_idxs)
            return choose_group_from_top_candidates(self.top_idxs, p, n)

        d = np.diag(S)
        di, dj = d[:p, None], d[None, p:]
        h, scores = self.scratch, self.scores
        np.subtract(di, dj, out=h)
        h *= 0.5
        np.square(h, out=h)
        np.square(S[:p, p:], out=scores)
        scores += h
        np.sqrt(scores, out=scores)
        np.add(di, dj, out=h)
        h *= 0.5
        scores += h
        scores -= di
        count = p
        for i in range(p):
            j = int(np.argmax(scores[i]))
            if scores[i, j] <= NEG_INF:
                continue
            self.indices[count] = p + j
            count += 1
            scores[i + 1:, j] = NEG_INF
            if count == n:
                break
        return self.indices[:count]


def _block_update(S, U, idx):
    rows = S[idx, :]
    B = rows[:, idx]
    B = np.asfortranarray(0.5 * (B + B.T))
    values, vectors = np.linalg.eigh(B)
    GT = np.ascontiguousarray(vectors[:, ::-1].T)
    new_rows = GT @ rows
    S[idx, :] = new_rows
    S[:, idx] = new_rows.T

    S[np.ix_(idx, idx)] = 0.0
    S[idx, idx] = values[::-1]
    UT = U.T
    UT[idx, :] = GT @ UT[idx, :]


def block_eig_update(S, U, indices):
    n = S.shape[0]
    idx = np.asarray(indices, dtype=np.int64).ravel()
    idx = idx[(idx >= 0) & (idx < n)]

    if idx.size:
        _, first = np.unique(idx, return_index=True)
        idx = idx[np.sort(first)]
    if idx.size > 1:
        _block_update(S, U, idx)
    return S, U


def _fit(S, p, n_iter, U, workspace):
    if p == S.shape[0] or n_iter == 0:
        return 0
    steps = 0
    for _ in range(n_iter):
        idx = workspace.select(S)
        if idx.size <= p:
            break
        _block_update(S, U, idx)
        steps += 1
    return steps


def fit_group_eig(S, p, n_iter, U, top_m=TOP_CANDIDATES_PER_ROW):
    S, U = np.asarray(S, dtype=np.float64), np.asarray(U, dtype=np.float64)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or U.shape != S.shape:
        raise ValueError("S and U must be equally sized square matrices")
    n = S.shape[0]
    n, p, _ = check_init(n, p, 1)
    n_iter = int(n_iter)
    if n_iter < 0:
        raise ValueError(f"n_iter must be >= 0, got {n_iter}")
    top_m = int(max(1, top_m))
    if not S.flags.writeable or not U.flags.writeable:
        raise ValueError("S and U must be writable")
    _fit(S, p, n_iter, U, _Workspace(n, p, top_m))
    return U, S


def fit_group_full_recompute(S, p, n_iter, u, top_m=TOP_CANDIDATES_PER_ROW):
    return fit_group_eig(S, p, n_iter, u, top_m=top_m)


def fit_group(S, p, n_iter, u):
    return fit_group_eig(S, p, n_iter, u)


def fit(S, p, n_iter, u):
    return fit_group_eig(S, p, n_iter, u)


class OnlineGroupEIG:
    """Streaming Group EIG.

    Lifecycle:
        eig = OnlineGroupEIG(n=d, p=P, k_per_batch=GROUP_G)
        for batch in stream:           # batch is (n, m)
            eig.partial_fit(batch)
        U = eig.U_

    State (bounded regardless of stream length):
        self.S_ : (n, n) running second-moment matrix in U's frame, C order
        self.U_ : (n, n) accumulated rotation, F order; first p columns =
                  approx top-p basis

    Additional keyword-only options
    --------------------------------
    covariance_mode : 'auto', 'project', or 'covariance'
        'project': Y = U.T @ X, followed by S += Y @ Y.T.
        'covariance': form X @ X.T first, then rotate that covariance.
        'auto': choose by multiplication counts. This changes evaluation order,
        not the covariance being accumulated. For an untouched identity basis,
        every mode skips multiplication by identity.
    exploit_identity : bool
        Detect the exact identity part of U and avoid multiplying that part.
        Nonzero entries are tested exactly, with NO numerical threshold. This
        remains safe after valid external changes to U or a warmstart.
    blas_threads : int or None
        Optional BLAS thread count during accumulation. None leaves it unchanged.
    rotation_threads : int or None
        BLAS thread count during the whole group loop; default 1. Small 2p-by-2p
        problems often do not benefit from many threads. Restored on exit.
        Requires threadpoolctl when a limit is requested. Thread limits affect
        native libraries process-wide; do not fit concurrently in Python threads.

    Diagnostics
    -----------
    last_timings_ : dict of validation, covariance, rotations, total seconds.
    last_update_ : chosen mode, effective active size, columns, group steps.
    timings_ : cumulative timings. n_samples_seen_ counts columns passed in,
        including a RunningMean correction column.
    """
    def __init__(self, n, p, k_per_batch=33, top_m=TOP_CANDIDATES_PER_ROW,
                 dtype=np.float64, check_finite=True, *, covariance_mode="auto",
                 exploit_identity=True, blas_threads=None, rotation_threads=1):
        n, p, k_per_batch = check_init(n, p, k_per_batch)
        self.n, self.p, self.k_per_batch = n, p, k_per_batch
        self.top_m = int(max(1, top_m))
        if np.dtype(dtype) != np.dtype(np.float64):
            raise TypeError("this implementation uses float64; set dtype=np.float64")
        self.dtype = np.dtype(np.float64)
        self.check_finite = bool(check_finite)
        if covariance_mode not in ("auto", "project", "covariance"):
            raise ValueError("covariance_mode must be auto, project, or covariance")
        self.covariance_mode = covariance_mode
        self.exploit_identity = bool(exploit_identity)
        self.blas_threads = blas_threads
        self.rotation_threads = rotation_threads
        if ThreadpoolController is None and (blas_threads is not None or rotation_threads is not None):
            raise ImportError(
                "threadpoolctl is required for blas_threads / rotation_threads "
                "(pip install threadpoolctl), or pass rotation_threads=None")

        self._controller = ThreadpoolController() if ThreadpoolController is not None else None
        self.S_ = np.zeros((self.n, self.n), dtype=np.float64, order="C")
        self.U_ = np.eye(self.n, dtype=np.float64, order="F")
        self.n_samples_seen_ = 0
        self._first_batch = True
        self._workspace = _Workspace(self.n, self.p, self.top_m)
        self.last_timings_ = dict(validation=0.0, covariance=0.0, rotations=0.0, total=0.0)
        self.timings_ = self.last_timings_.copy()
        self.last_update_ = {}

    def _limit(self, threads):
        if threads is None:
            return nullcontext()
        return self._controller.limit(limits=threads, user_api="blas")

    def _active_indices(self):
        n = self.n
        if not self.exploit_identity:
            return np.arange(n)

        nz = self.U_ != 0.0
        active = (np.count_nonzero(nz, axis=0) != 1)
        active |= (np.count_nonzero(nz, axis=1) != 1)
        active |= (np.diag(self.U_) != 1.0)
        idx = np.flatnonzero(active)

        # if near full, return everything
        return np.arange(n) if idx.size > 0.85 * n else idx

    def _accumulate(self, X):
        n, m = X.shape
        idx = self._active_indices()
        a = idx.size
        if a == 0:
            _gram(X, self.S_)
            return "identity", a
        mode = self.covariance_mode
        if mode == "auto":
            # Projection costs 2*a*a*m. Covariance rotation costs approximately
            # 2*a*a*(n+a), in addition to the same symmetric rank-m update.
            mode = "covariance" if m > n + a else "project"
        U = self.U_ if a == n else self.U_[np.ix_(idx, idx)]
        if mode == "project":
            if a == n:
                Y = U.T @ X
            else:
                Y = np.array(X, dtype=np.float64, order="C", copy=True)
                Y[idx, :] = U.T @ X[idx, :]
            _gram(Y, self.S_)
        else:
            C = _gram(X)
            if a == n:
                rotated = (U.T @ C) @ U
                # Restore symmetry before the group loop.
                self.S_ += 0.5 * (rotated + rotated.T)
            else:
                rows = U.T @ C[idx, :]
                B = rows[:, idx] @ U
                rows[:, idx] = 0.5 * (B + B.T)
                C[idx, :] = rows
                C[:, idx] = rows.T
                self.S_ += C
        return mode, a

    def partial_fit(self, X_batch, warmstart=None):
        """
        Fold a new batch into S and run k_per_batch group iterations.

        Parameters
        ----------
        X_batch : (n, m) float
        warmstart : callable(S, U, p) or None
            Applied only on the first batch, after covariance accumulation
            but before the group iterations. It must mutate S and U in place
            and preserve U @ S @ U.T.
        """
        t0 = time.perf_counter()
        X = check_batch(X_batch, self.n, self.dtype, self.check_finite)

        # Keep supported user replacements of state arrays BLAS-friendly.
        self.S_ = np.require(self.S_, dtype=np.float64, requirements=["C", "W"])
        self.U_ = np.require(self.U_, dtype=np.float64, requirements=["F", "W"])
        if self.S_.shape != (self.n, self.n) or self.U_.shape != self.S_.shape:
            raise ValueError("S_ and U_ must have shape (n, n)")
        if self._workspace.top_m != self.top_m:
            self._workspace = _Workspace(self.n, self.p, self.top_m)
        t1 = time.perf_counter()
        with self._limit(self.blas_threads):
            mode, active = self._accumulate(X)
        if self._first_batch and warmstart is not None:
            warmstart(self.S_, self.U_, self.p)
        self._first_batch = False
        t2 = time.perf_counter()
        with self._limit(self.rotation_threads):
            steps = _fit(self.S_, self.p, self.k_per_batch, self.U_, self._workspace)
        t3 = time.perf_counter()
        self.n_samples_seen_ += X.shape[1]
        self.last_timings_ = dict(validation=t1-t0, covariance=t2-t1, rotations=t3-t2, total=t3-t0)
        for key, value in self.last_timings_.items():
            self.timings_[key] += value
        self.last_update_ = dict(mode=mode, active_size=active, columns=X.shape[1], group_steps=steps)
        return self

    @property
    def components_(self):
        """First p eigenvectors as rows (sklearn convention)."""
        return self.U_[:, :self.p].T

    def transform(self, X):
        return self.components_ @ np.asarray(X, dtype=self.dtype)

    def inverse_transform(self, codes):
        return self.U_[:, :self.p] @ np.asarray(codes, dtype=self.dtype)


def run_group_eig(X, P, G, batch_size, monitor, evr_fn,
                  top_m=TOP_CANDIDATES_PER_ROW, warmstart=None, **kwargs):
    """
    Benchmark-shape streaming driver.

    Returns
    -------
    tr, time_axis, samples, eig
    """
    X = np.asarray(X, dtype=np.float64)

    eig = OnlineGroupEIG(n=X.shape[0], p=P, k_per_batch=G, top_m=top_m, **kwargs)
    traces, times, samples = [], [], []
    elapsed = 0.0
    for start in range(0, X.shape[1], batch_size):
        end = min(start + batch_size, X.shape[1])
        t0 = time.perf_counter()
        eig.partial_fit(X[:, start:end], warmstart=warmstart)
        elapsed += time.perf_counter() - t0
        traces.append(evr_fn(monitor, eig.U_, P))
        times.append(elapsed)
        samples.append(end)
    return np.asarray(traces), np.asarray(times), np.asarray(samples), eig
