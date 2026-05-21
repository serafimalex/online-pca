"""
group_eig_buffer.py

Group / Block Online EIG.

This is the symmetric (covariance-style) analog of group_svd.py, and the
group/block extension of online_eig_buffer.py.

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

Streaming wrapper:
    OnlineGroupEIG maintains (S, U). Each batch:
        Y = U^T @ X_new
        S += Y @ Y^T
        run k_per_batch group iterations on S

Import:
    import group_eig_buffer as geb

Drop-in for the benchmark notebook:
    geb.set_num_threads(geb.NUMBA_THREADS)
    geb.warmup()
    tr, t, s, eig = geb.run_group_eig(X, P=P, G=GROUP_G,
                                       batch_size=ONLINE_BATCH,
                                       monitor=monitor.astype(np.float64),
                                       evr_fn=evr_from_U)
"""

import numpy as np

try:
    from numba import njit, prange, set_num_threads as _set_num_threads
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco

    def prange(*args):
        return range(*args)

    def _set_num_threads(n):
        return None


# ============================================================
# MODULE-LEVEL CONSTANTS (mirror online_svd_buffer.py / online_eig_buffer.py)
# ============================================================

TEMP_COL = None
TEMP_ROW = None
TOP_K_SCORES = 5                    # unused here, kept for API parity
TOP_CANDIDATES_PER_ROW = 32         # like group_svd.py
NEG_INF32 = np.float64(-1e30)
NEG_INF = np.float64(-1e30)
NUMBA_THREADS = 1


def set_num_threads(k):
    global NUMBA_THREADS
    NUMBA_THREADS = int(k)
    if NUMBA_AVAILABLE:
        _set_num_threads(NUMBA_THREADS)


def _init_global_buf(n_rows, n_cols, dtype):
    """Allocate scratch buffers. Kept for API parity with online_svd_buffer."""
    global TEMP_COL, TEMP_ROW
    TEMP_COL = np.zeros(n_rows, dtype=dtype)
    TEMP_ROW = np.zeros(n_cols, dtype=dtype)


# ============================================================
# CANDIDATE LIST COMPUTATION (the EIG-flavored score)
# ============================================================
# For symmetric S, the 2x2 block at (i,j) is [[S_ii, S_ij], [S_ij, S_jj]].
# Its largest eigenvalue is:
#   lambda_max = (S_ii + S_jj)/2 + sqrt(((S_ii - S_jj)/2)^2 + S_ij^2)
# Score: C_ij = lambda_max - S_ii.

@njit(parallel=True, cache=True)
def compute_row_top_candidates_eig(S, p, top_vals, top_idxs):
    """
    For each i in [0, p), scan j > i in [0, n) and keep top_m best candidates.

    S is symmetric n x n. Writes top_vals (p, top_m) sorted descending
    and top_idxs (p, top_m) with -1 for empty slots.
    """
    n = S.shape[0]
    top_m = top_vals.shape[1]

    for i in prange(p):
        # reset
        for k in range(top_m):
            top_vals[i, k] = NEG_INF
            top_idxs[i, k] = -1

        sii = S[i, i]

        for j in range(i + 1, n):
            sjj = S[j, j]
            sij = S[i, j]

            half_sum = 0.5 * (sii + sjj)
            half_dif = 0.5 * (sii - sjj)
            radius = np.sqrt(half_dif * half_dif + sij * sij)
            lmax = half_sum + radius

            val = lmax - sii

            # inline top-m insertion
            if val > top_vals[i, top_m - 1]:
                kk = top_m - 1
                while kk > 0 and val > top_vals[i, kk - 1]:
                    top_vals[i, kk] = top_vals[i, kk - 1]
                    top_idxs[i, kk] = top_idxs[i, kk - 1]
                    kk -= 1
                top_vals[i, kk] = val
                top_idxs[i, kk] = j


def choose_group_from_top_candidates(top_idxs, p, n_active):
    """
    Greedy: for each i in [0, p), walk its top-m list and claim the first j
    that isn't already used. Returns indices = [0..p-1, j_0, ..., j_{p-1}].

    Same logic as group_svd.py but with n_active = n for EIG (S is square).
    """
    used = set(range(p))
    indices = list(range(p))

    for i in range(p):
        chosen = -1
        for k in range(top_idxs.shape[1]):
            j = int(top_idxs[i, k])
            if j < 0:
                continue
            if j >= n_active:
                continue
            if j in used:
                continue
            chosen = j
            break

        if chosen != -1:
            indices.append(chosen)
            used.add(chosen)

    return np.asarray(indices, dtype=np.int64)


# ============================================================
# BLOCK EIG UPDATE
# ============================================================

def block_eig_update(S, U, indices):
    """
    Apply one block eigen-update.

    Let idx = indices (size up to 2p). Then:
        B = S[idx, idx]                          (square, symmetric)
        B = G_local @ Lambda @ G_local^T         (eigh)

        S[idx, :]   = G_local^T @ S[idx, :]      (left mult on the slice)
        S[:, idx]   = S[:, idx]   @ G_local      (right mult on the slice)
        U[:, idx]   = U[:, idx]   @ G_local      (right mult on basis)

    Eigh returns eigenvalues in ascending order; we reorder so that the largest
    eigenvalue sits at position 0 of the block, which puts the dominant
    direction at the smallest index in idx (i.e. at index 0 in the kept block).
    This convention matches the spirit of the pairwise algorithm where the
    larger eigenvalue lands at S[i,i].
    """
    n = S.shape[0]

    # dedupe (defensive: choose_group_from_top_candidates already dedupes,
    # but indices may include i values that overlap if p > n which is invalid).
    idx = np.asarray(indices, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < n)]
    if idx.size <= 1:
        return S, U

    seen = set()
    ordered = []
    for v in idx:
        ii = int(v)
        if ii not in seen:
            ordered.append(ii)
            seen.add(ii)
    idx = np.asarray(ordered, dtype=np.int64)

    # Extract the symmetric block
    B = S[np.ix_(idx, idx)]
    # Symmetrize defensively to suppress floating-point drift
    B = 0.5 * (B + B.T)

    # Eigendecomposition (ascending eigenvalues). Reverse to descending.
    eigvals, eigvecs = np.linalg.eigh(B)
    # eigvecs columns are eigenvectors. Reverse so column 0 = largest.
    G_local = eigvecs[:, ::-1]

    # Apply similarity transform on the slice.
    # S[idx, :] = G_local^T @ S[idx, :]
    S_rows = np.ascontiguousarray(S[idx, :])
    S[idx, :] = G_local.T @ S_rows

    # S[:, idx] = S[:, idx] @ G_local
    S_cols = np.ascontiguousarray(S[:, idx])
    S[:, idx] = S_cols @ G_local

    # Re-symmetrize after the slice update (floating-point drift).
    S = 0.5 * (S + S.T)

    # U[:, idx] = U[:, idx] @ G_local
    U_cols = np.ascontiguousarray(U[:, idx])
    U[:, idx] = U_cols @ G_local

    return S, U


# ============================================================
# CORE FIT
# ============================================================

def fit_group_eig(S, p, n_iter, U, top_m=TOP_CANDIDATES_PER_ROW):
    """
    Run n_iter group-EIG iterations on S, accumulating rotations into U.

    Parameters
    ----------
    S : (n, n) float64, symmetric, modified in place
    p : int
    n_iter : int
    U : (n, n) float64, modified in place
    top_m : int, candidate-list width per row

    Returns
    -------
    U, S (the modified arrays; returning them keeps API symmetry with the SVD class)
    """
    if S.dtype != np.float64:
        S = S.astype(np.float64, copy=False)
    if U.dtype != np.float64:
        U = U.astype(np.float64, copy=False)

    n = S.shape[0]
    if p > n:
        raise ValueError(f"p must be <= n. Got p={p}, n={n}.")
    if p >= n:
        # No j > i in [0, n) for any i < p when p == n, nothing to do.
        return U, S

    top_m = int(max(1, top_m))

    top_vals = np.empty((p, top_m), dtype=np.float64)
    top_idxs = np.empty((p, top_m), dtype=np.int64)

    for _ in range(int(n_iter)):
        compute_row_top_candidates_eig(S, p, top_vals, top_idxs)

        indices = choose_group_from_top_candidates(top_idxs, p, n)

        if indices.size <= p:
            # No usable partners — all candidates exhausted. Bail.
            break

        S, U = block_eig_update(S, U, indices)

    return U, S


# Aliases for API parity with group_svd.py
def fit_group_full_recompute(S, p, n_iter, u, top_m=TOP_CANDIDATES_PER_ROW):
    """Alias matching group_svd.fit_group_full_recompute signature."""
    return fit_group_eig(S, p, n_iter, u, top_m=top_m)


def fit_group(S, p, n_iter, u):
    return fit_group_eig(S, p, n_iter, u)


def fit(S, p, n_iter, u):
    return fit_group_eig(S, p, n_iter, u)


# ============================================================
# STREAMING WRAPPER
# ============================================================

class OnlineGroupEIG:
    """
    Streaming group/block EIG.

    Lifecycle:
        eig = OnlineGroupEIG(n=d, p=P, k_per_batch=GROUP_G)
        for batch in stream:           # batch is (n, m)
            eig.partial_fit(batch)
        U = eig.U_

    State (bounded regardless of stream length):
        self.S_ : (n, n) running second-moment matrix in U's frame
        self.U_ : (n, n) accumulated rotation; first p columns = approx top-p basis
    """

    def __init__(self, n, p, k_per_batch=33, top_m=TOP_CANDIDATES_PER_ROW, dtype=np.float64):
        self.n = n
        self.p = p
        self.k_per_batch = k_per_batch
        self.top_m = top_m
        self.dtype = dtype

        self.S_ = np.zeros((n, n), dtype=dtype)
        self.U_ = np.eye(n, dtype=dtype)

        _init_global_buf(n, n, dtype)

        self.n_samples_seen_ = 0
        self._first_batch = True

    def partial_fit(self, X_batch, warmstart=None):
        """
        Fold a new batch into S and run k_per_batch group iterations.

        Parameters
        ----------
        X_batch : (n, m) float
        warmstart : callable(S, U, p) -> (S, U) or None
            Applied only on the first batch, after covariance accumulation
            but before the group iterations.
        """
        Xb = np.asarray(X_batch, dtype=self.dtype)
        # Project into U's frame
        Y = self.U_.T @ Xb               # (n, m)
        # Rank-m symmetric update
        self.S_ += Y @ Y.T
        # Symmetrize defensively
        self.S_ = 0.5 * (self.S_ + self.S_.T)

        if self._first_batch and warmstart is not None:
            warmstart(self.S_, self.U_, self.p)
        if self._first_batch:
            self._first_batch = False

        # Run group iterations
        self.U_, self.S_ = fit_group_eig(
            self.S_, self.p, self.k_per_batch, self.U_, top_m=self.top_m
        )

        self.n_samples_seen_ += Xb.shape[1]
        return self

    @property
    def components_(self):
        """First p eigenvectors as rows (sklearn convention)."""
        return self.U_[:, :self.p].T

    def transform(self, X):
        return self.U_[:, :self.p].T @ np.asarray(X, dtype=self.dtype)

    def inverse_transform(self, codes):
        return self.U_[:, :self.p] @ np.asarray(codes, dtype=self.dtype)


# ============================================================
# CONVENIENCE NOTEBOOK DRIVER
# ============================================================

def run_group_eig(X, P, G, batch_size, monitor, evr_fn,
                  top_m=TOP_CANDIDATES_PER_ROW, warmstart=None):
    """
    Benchmark-shape streaming driver.

    Parameters
    ----------
    X : (d, n_total) float
    P : int — components
    G : int — group iterations per batch (e.g. GROUP_G = G_pairwise // P)
    batch_size : int
    monitor : (d, m)
    evr_fn : callable(monitor, U, P) -> float
    top_m : int — width of per-row candidate lists
    warmstart : callable(S, U, p) -> (S, U) or None.
        Applied ONLY on the first batch, between covariance accumulation
        and the first group iterations.

    Returns
    -------
    tr, time_axis, samples, eig
    """
    import time
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kwargs):
            return it

    d = X.shape[0]
    n_total = X.shape[1]
    batch_size = max(batch_size, d)

    eig = OnlineGroupEIG(n=d, p=P, k_per_batch=G, top_m=top_m)

    tr, time_axis, samples = [], [], []
    samples_seen = 0
    fit_time_accum = 0.0

    total_batches = n_total // batch_size + (1 if n_total % batch_size else 0)

    start = 0
    for _ in tqdm(range(total_batches), desc="GroupEIG batches"):
        end = min(start + batch_size, n_total)
        batch = X[:, start:end]

        t0 = time.perf_counter()
        eig.partial_fit(batch, warmstart=warmstart)
        t1 = time.perf_counter()

        fit_time_accum += t1 - t0
        samples_seen += end - start

        time_axis.append(fit_time_accum)
        samples.append(samples_seen)
        tr.append(evr_fn(monitor, eig.U_, P))

        start = end
        if start >= n_total:
            break

    return np.array(tr), np.array(time_axis), np.array(samples), eig


# ============================================================
# WARMUP
# ============================================================

def warmup(n=8, p=3, n_iter=4, dtype=np.float64):
    """
    Run one tiny end-to-end fit to compile every @njit function.

    Call once before timing, mirroring the SVD/EIG pattern in the notebook.
    """
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)).astype(dtype)
    S = (A + A.T) / 2.0

    U = np.eye(n, dtype=dtype)
    _init_global_buf(n, n, dtype)

    # Run the full path: compute_row_top_candidates_eig, choose_group, block_eig_update
    fit_group_eig(S, p, n_iter, U)

    return None