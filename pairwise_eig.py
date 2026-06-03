import numpy as np
from numba import njit, prange, set_num_threads as _set_num_threads


# ============================================================
# MODULE-LEVEL CONSTANTS
# ============================================================

TEMP_COL = None         
TEMP_ROW = None         
TOP_K_SCORES = 5
NEG_INF32 = np.float64(-1e30)
NUMBA_THREADS = 1


def set_num_threads(k):
    global NUMBA_THREADS
    NUMBA_THREADS = k
    _set_num_threads(k)


def _init_global_buf(n_rows, n_cols, dtype):
    global TEMP_COL, TEMP_ROW
    TEMP_COL = np.zeros(n_rows, dtype=dtype)
    TEMP_ROW = np.zeros(n_cols, dtype=dtype)


# ============================================================
# SCORE COMPUTATION
# ============================================================
# For symmetric S, the 2x2 block at (i,j) is [[S_ii, S_ij], [S_ij, S_jj]].
# Its eigenvalues are:
#   lambda_max,min = (S_ii + S_jj) / 2  +/-  sqrt( ((S_ii - S_jj)/2)^2 + S_ij^2 )
# The score is lambda_max - S_ii.

@njit(parallel=True)
def compute_and_assign_topk_eig(p, S, scores, row_topk_vals, row_topk_idx):
    n = S.shape[0]

    for i in prange(p):
        for k in range(TOP_K_SCORES):
            row_topk_vals[i, k] = NEG_INF32
            row_topk_idx[i, k] = -1

        sii = S[i, i]
        for j in range(i + 1, n):
            sjj = S[j, j]
            sij = S[i, j]

            half_sum = 0.5 * (sii + sjj)
            half_dif = 0.5 * (sii - sjj)
            radius = np.sqrt(half_dif * half_dif + sij * sij)
            lmax = half_sum + radius

            val = lmax - sii
            scores[i, j] = val

            if val > row_topk_vals[i, TOP_K_SCORES - 1]:
                k = TOP_K_SCORES - 1
                while k > 0 and val > row_topk_vals[i, k - 1]:
                    row_topk_vals[i, k] = row_topk_vals[i, k - 1]
                    row_topk_idx[i, k] = row_topk_idx[i, k - 1]
                    k -= 1
                row_topk_vals[i, k] = val
                row_topk_idx[i, k] = j


@njit
def _score_one(i, j, S):
    sii = S[i, i]
    sjj = S[j, j]
    sij = S[i, j]
    half_sum = 0.5 * (sii + sjj)
    half_dif = 0.5 * (sii - sjj)
    radius = np.sqrt(half_dif * half_dif + sij * sij)
    
    return half_sum + radius - sii


@njit
def recompute_row_topk(scores, topk_vals, topk_idxs, r):
    row_sz = scores.shape[1]

    top_vals = np.full(TOP_K_SCORES, NEG_INF32, dtype=np.float64)
    top_idxs = np.full(TOP_K_SCORES, -1, dtype=np.int64)

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


@njit
def get_max_topk(row_topk_vals, row_topk_idx):
    max_val = -np.inf
    max_r = -1
    for r in range(row_topk_vals.shape[0]):
        if row_topk_vals[r, 0] > max_val:
            max_val = row_topk_vals[r, 0]
            max_r = r

    if max_r == -1:
        return -1, -1

    return max_r, int(row_topk_idx[max_r, 0])


# ============================================================
# AFFECTED-SCORE REFRESH
# ============================================================

@njit(parallel=True)
def refresh_row_topk(scores, r, S, p, row_topk_vals, row_topk_idx):
    """
    Recompute scores[r, r+1 .. n-1] for row r (must have r < p), rebuild topk.
    """
    n = S.shape[0]

    # reset
    top_vals = np.full(TOP_K_SCORES, NEG_INF32, dtype=np.float64)
    top_idxs = np.full(TOP_K_SCORES, -1, dtype=np.int64)

    for s in prange(r + 1, n):
        val = _score_one(r, s, S)
        scores[r, s] = val
        # NB: prange + shared top arrays — but each iteration only reads/writes
        # via the merge below, which we do serially after the loop.

    # serial merge to top-k (small, n iterations)
    for s in range(r + 1, n):
        val = scores[r, s]
        if val > top_vals[TOP_K_SCORES - 1]:
            k = TOP_K_SCORES - 1
            while k > 0 and val > top_vals[k - 1]:
                top_vals[k] = top_vals[k - 1]
                top_idxs[k] = top_idxs[k - 1]
                k -= 1
            top_vals[k] = val
            top_idxs[k] = s

    for k in range(TOP_K_SCORES):
        row_topk_vals[r, k] = top_vals[k]
        row_topk_idx[r, k] = top_idxs[k]


@njit(parallel=True)
def refresh_col_topk(scores, c, S, p, row_topk_vals, row_topk_idx):
    """
    Recompute scores[r, c] for r < min(c, p) and update those rows' topk.

    When a column changes, every row r < c that had c as a candidate must be
    updated. We surgically update scores[r, c]; if c was in row r's top-k, we
    evict it and re-insert (possibly demoting). If c isn't in top-k, we just
    try to insert.
    """
    limit = min(c, p)

    for r in prange(limit):
        val = _score_one(r, c, S)
        scores[r, c] = val

        # remove existing occurrence of c from row r's topk
        existing = -1
        for k in range(TOP_K_SCORES):
            if row_topk_idx[r, k] == c:
                existing = k
                break
        if existing != -1:
            for k in range(existing, TOP_K_SCORES - 1):
                row_topk_vals[r, k] = row_topk_vals[r, k + 1]
                row_topk_idx[r, k] = row_topk_idx[r, k + 1]
            row_topk_vals[r, TOP_K_SCORES - 1] = NEG_INF32
            row_topk_idx[r, TOP_K_SCORES - 1] = -1

        # insert new val for column c
        if val > row_topk_vals[r, TOP_K_SCORES - 1]:
            k = TOP_K_SCORES - 1
            while k > 0 and val > row_topk_vals[r, k - 1]:
                row_topk_vals[r, k] = row_topk_vals[r, k - 1]
                row_topk_idx[r, k] = row_topk_idx[r, k - 1]
                k -= 1
            row_topk_vals[r, k] = val
            row_topk_idx[r, k] = c

        # if topk has been emptied (slot 0 invalid), rebuild from scratch
        if row_topk_idx[r, 0] == -1:
            recompute_row_topk(scores, row_topk_vals, row_topk_idx, r)


# ============================================================
# 2x2 EIGENDECOMPOSITION + ROTATION APPLICATION
# ============================================================

@njit
def jacobi_2x2_rotation(sii, sjj, sij):
    """
    Compute the 2x2 Jacobi rotation G such that
        G^T [[sii, sij], [sij, sjj]] G = diag(lambda_max, lambda_min)
    with lambda_max at (0,0). Returns the 2x2 matrix G.
    """
    G = np.zeros((2, 2), dtype=np.float64)
    if sij == 0.0:
        # already diagonal: identity, or swap if needed so larger is at (0,0)
        if sjj > sii:
            G[0, 0] = 0.0
            G[0, 1] = -1.0
            G[1, 0] = 1.0
            G[1, 1] = 0.0
        else:
            G[0, 0] = 1.0
            G[1, 1] = 1.0
        return G

    # standard Jacobi: tan(2 theta) = 2 S_ij / (S_ii - S_jj)
    diff = sii - sjj
    if diff == 0.0:
        t = 1.0 if sij > 0.0 else -1.0
        c = 1.0 / np.sqrt(2.0)
        s = t * c
    else:
        tau = diff / (2.0 * sij)
        # Guard against tau-squared overflow when |sij| is tiny.
        abs_tau = abs(tau)
        if abs_tau > 1e8:
            t = 0.5 / tau
        else:
            if tau >= 0.0:
                t = 1.0 / (tau + np.sqrt(1.0 + tau * tau))
            else:
                t = 1.0 / (tau - np.sqrt(1.0 + tau * tau))
        c = 1.0 / np.sqrt(1.0 + t * t)
        s = t * c

    # Orientation: place the LARGER eigenvalue at (0,0). Decide from the actual
    # post-rotation diagonal entries for G = [[c, -s], [s, c]], not a proxy.
    new_ii = c * c * sii + 2.0 * c * s * sij + s * s * sjj
    new_jj = s * s * sii - 2.0 * c * s * sij + c * c * sjj
    if new_ii >= new_jj:
        G[0, 0] = c
        G[0, 1] = -s
        G[1, 0] = s
        G[1, 1] = c
    else:
        # swap eigenvectors (column swap) so the larger eigenvalue lands at (0,0)
        G[0, 0] = -s
        G[0, 1] = c
        G[1, 0] = c
        G[1, 1] = s
    return G


@njit
def apply_similarity(S, i, j, G):
    """
    In-place similarity transform S <- G^T S G acting only on rows/cols i, j.

    G is 2x2 with rows indexed (i, j). This touches:
      - rows i and j  (left mult by G^T)
      - cols i and j  (right mult by G)
    cost: O(n)
    """
    n = S.shape[0]
    c = G[0, 0]
    s = G[1, 0]
    # G = [[c, -s], [s, c]]  =>  G^T = [[c, s], [-s, c]]
    # (we hard-code the swap case via G coefficients passed in)
    g00 = G[0, 0]
    g01 = G[0, 1]
    g10 = G[1, 0]
    g11 = G[1, 1]

    # --- Left multiply: S' = G^T S, affects rows i and j only ---
    # new_row_i = g00 * row_i + g10 * row_j
    # new_row_j = g01 * row_i + g11 * row_j
    for col in range(n):
        a = S[i, col]
        b = S[j, col]
        S[i, col] = g00 * a + g10 * b
        S[j, col] = g01 * a + g11 * b

    # --- Right multiply: S'' = (G^T S) G, affects cols i and j only ---
    # new_col_i = g00 * col_i + g10 * col_j
    # new_col_j = g01 * col_i + g11 * col_j
    for row in range(n):
        a = S[row, i]
        b = S[row, j]
        S[row, i] = g00 * a + g10 * b
        S[row, j] = g01 * a + g11 * b

    # Force the off-diagonals (i,j) and (j,i) to exact zero — cleaner than
    # relying on floating-point cancellation, and keeps the score grid honest.
    S[i, j] = 0.0
    S[j, i] = 0.0


@njit
def apply_right_to_U(U, i, j, G):
    """
    In-place U <- U G, only affects columns i, j of U.  Cost: O(n).
    """
    n_rows = U.shape[0]
    g00 = G[0, 0]
    g01 = G[0, 1]
    g10 = G[1, 0]
    g11 = G[1, 1]

    for row in range(n_rows):
        a = U[row, i]
        b = U[row, j]
        U[row, i] = g00 * a + g10 * b
        U[row, j] = g01 * a + g11 * b


# ============================================================
# CORE FIT LOOP
# ============================================================

def fit_safe(S, p, n_iter, u, scores, row_topk_vals, row_topk_idx, TEMP_COL=None):
    """
    Drop-in analog of online_svd_buffer.fit_safe, for the EIG algorithm.

    Parameters
    ----------
    S : (n, n) float64
        Current symmetric working matrix; modified in place.
    p : int
        Number of kept directions.
    n_iter : int
        Maximum number of Jacobi rotations to apply this call.
    u : (n, n) float64
        Running basis; modified in place.
    scores : (p, n) float64
        Score grid scratch, modified in place.
    row_topk_vals, row_topk_idx : per-row top-k caches.
    TEMP_COL : unused, kept for signature compatibility with the SVD class.

    Returns
    -------
    u, S : the (possibly-modified) basis and working matrix.
    """
    n = S.shape[0]

    compute_and_assign_topk_eig(p, S, scores, row_topk_vals, row_topk_idx)

    for _ in range(n_iter):
        iq, jq = get_max_topk(row_topk_vals, row_topk_idx)

        if iq == -1 or jq == -1:
            break

        # No j >= n edge case for EIG — S is square, all (i,j) with i<j<=n are valid.
        # But still guard the SVD-style "stale pivot" case.
        guard = 0
        while jq >= n:
            row_topk_vals[iq, 0] = NEG_INF32
            row_topk_idx[iq, 0] = -1
            iq, jq = get_max_topk(row_topk_vals, row_topk_idx)
            if iq == -1 or jq == -1:
                return u, S
            guard += 1
            if guard > 10000:
                raise RuntimeError("Too many invalid pivots in EIG fit.")

        # Build 2x2 Jacobi rotation
        sii = S[iq, iq]
        sjj = S[jq, jq]
        sij = S[iq, jq]
        G = jacobi_2x2_rotation(sii, sjj, sij)

        # Apply similarity transform to S (in place) and right-mult U
        apply_similarity(S, iq, jq, G)
        apply_right_to_U(u, iq, jq, G)

        # Refresh affected scores:
        #   row iq: scores[iq, s] for s > iq
        #   row jq: scores[jq, s] for s > jq  (only if jq < p)
        #   col iq: scores[r, iq] for r < iq
        #   col jq: scores[r, jq] for r < min(jq, p)
        if iq < p:
            refresh_row_topk(scores, iq, S, p, row_topk_vals, row_topk_idx)
        if jq < p:
            refresh_row_topk(scores, jq, S, p, row_topk_vals, row_topk_idx)
        refresh_col_topk(scores, iq, S, p, row_topk_vals, row_topk_idx)
        refresh_col_topk(scores, jq, S, p, row_topk_vals, row_topk_idx)

    return u, S


# ============================================================
# STREAMING WRAPPER
# ============================================================

class OnlineEIG:
    """
    Streaming Jacobi-style top-p eigen-decomposition.

    Lifecycle:
        eig = OnlineEIG(n=d, p=P, k_per_batch=G)
        for batch in stream:  # batch is (d, m)
            eig.partial_fit(batch)
        U = eig.U_           # (d, d), first p columns are the top-p basis

    State (bounded regardless of stream length):
        self.S  : (n, n) running second-moment matrix in U's frame
        self.U  : (n, n) accumulated rotation, columns = approx eigenvectors
    """

    def __init__(self, n, p, k_per_batch=500, dtype=np.float64):
        self.n = n
        self.p = p
        self.k_per_batch = k_per_batch
        self.dtype = dtype

        self.S_ = np.zeros((n, n), dtype=dtype)
        self.U_ = np.eye(n, dtype=dtype)

        self.scores = np.empty((p, n), dtype=dtype)
        self.scores.fill(NEG_INF32)
        self.row_topk_vals = np.empty((p, TOP_K_SCORES), dtype=dtype)
        self.row_topk_vals.fill(NEG_INF32)
        self.row_topk_idx = np.empty((p, TOP_K_SCORES), dtype=np.int64)
        self.row_topk_idx.fill(-1)

        _init_global_buf(n, n, dtype)

        self.n_samples_seen_ = 0
        self._first_batch = True

    def partial_fit(self, X_batch, warmstart=None):
        """
        Fold a new batch of samples into S and run k_per_batch Jacobi iterations.

        Parameters
        ----------
        X_batch : (n, m) float
            New raw samples, columns are samples.
        warmstart : callable(S, U, p) -> (S, U) or None
            Applied only on the first batch, after the covariance update but
            before the Jacobi iterations. Typically a partial tridiagonalizer.
        """
        Xb = np.asarray(X_batch, dtype=self.dtype)
        # Project new data into U's frame, then accumulate covariance
        Y = self.U_.T @ Xb                 # (n, m)
        # In-frame rank-m symmetric update: S += Y Y^T
        # Done with a single gemm rather than an outer-product loop.
        self.S_ += Y @ Y.T
        # Force symmetry (defensive: floating-point drift on big updates).
        self.S_ = 0.5 * (self.S_ + self.S_.T)

        # Apply warmstart preconditioning on first batch only.
        # The warmstart should mutate S and U so that U @ S_new @ U^T equals
        # the original observed covariance — i.e. S <- Q^T S Q, U <- U @ Q.
        if self._first_batch and warmstart is not None:
            warmstart(self.S_, self.U_, self.p)
            self._first_batch = False
        elif self._first_batch:
            self._first_batch = False

        # Run k Jacobi iterations
        self.U_, self.S_ = fit_safe(
            self.S_, self.p, self.k_per_batch,
            self.U_,
            self.scores, self.row_topk_vals, self.row_topk_idx,
            TEMP_COL,
        )

        self.n_samples_seen_ += Xb.shape[1]
        return self

    @property
    def components_(self):
        """First p eigenvectors as rows (sklearn convention)."""
        return self.U_[:, :self.p].T

    def transform(self, X):
        """Project (n, m) samples down to p-dim codes (p, m)."""
        return self.U_[:, :self.p].T @ np.asarray(X, dtype=self.dtype)

    def inverse_transform(self, codes):
        """Reconstruct (p, m) codes back to (n, m) samples."""
        return self.U_[:, :self.p] @ np.asarray(codes, dtype=self.dtype)


# ============================================================
# CONVENIENCE FUNCTION FOR NOTEBOOK COMPATIBILITY
# ============================================================

def run_online_eig(X, P, G, batch_size, monitor, evr_fn, warmstart=None):
    """
    Streaming benchmark driver, matches the shape of the SVD benchmark cell.

    Parameters
    ----------
    X : (d, n_total) float — full stream, columns are samples.
    P : int — number of components.
    G : int — Jacobi iterations per batch.
    batch_size : int.
    monitor : (d, m) — evaluation set.
    evr_fn : callable(monitor, U, P) -> float — typically `evr_from_U`.
    warmstart : callable(S, U, p) -> (S, U) or None.
        Applied ONLY on the first batch, between covariance accumulation and
        the first Jacobi iterations.

    Returns
    -------
    tr, time_axis, samples : numpy arrays.
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

    eig = OnlineEIG(n=d, p=P, k_per_batch=G)

    tr, time_axis, samples = [], [], []
    samples_seen = 0
    fit_time_accum = 0.0

    total_batches = n_total // batch_size + (1 if n_total % batch_size else 0)

    start = 0
    for _ in tqdm(range(total_batches), desc="OnlineEIG batches"):
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
# WARMUP — trigger numba JIT compilation on all hot paths
# ============================================================

def warmup(n=8, p=3, n_iter=4, dtype=np.float64):
    """
    Run one tiny end-to-end fit to compile every @njit function in this module.

    Call this once before timing — e.g. in the same cell as the SVD class's
    warmup, mirroring the pattern in benchmarking_3.ipynb:

        op.set_num_threads(op.NUMBA_THREADS)
        op._init_global_buf(...)
        _ = fit_safe(...)              # warms online_svd_buffer

        oeb.set_num_threads(oeb.NUMBA_THREADS)
        oeb.warmup()                   # warms online_eig_buffer

    The defaults (n=8, p=3, n_iter=4) are deliberately tiny — the goal is to
    compile, not to compute.
    """
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)).astype(dtype)
    S = (A + A.T) / 2.0   # symmetric

    U = np.eye(n, dtype=dtype)
    scores = np.empty((p, n), dtype=dtype)
    scores.fill(NEG_INF32)
    row_topk_vals = np.empty((p, TOP_K_SCORES), dtype=dtype)
    row_topk_vals.fill(NEG_INF32)
    row_topk_idx = np.empty((p, TOP_K_SCORES), dtype=np.int64)
    row_topk_idx.fill(-1)

    _init_global_buf(n, n, dtype)

    # End-to-end fit_safe call exercises: compute_and_assign_topk_eig,
    # get_max_topk, jacobi_2x2_rotation, apply_similarity, apply_right_to_U,
    # refresh_row_topk, refresh_col_topk, _score_one, and recompute_row_topk
    # (the last one fires when topk gets exhausted; we force it below).
    fit_safe(S, p, n_iter, U, scores, row_topk_vals, row_topk_idx, TEMP_COL)

    # Force the recompute_row_topk path: blank out a row's topk and refresh.
    # This is otherwise only hit when refresh_col_topk evicts everything.
    row_topk_vals[0, :] = NEG_INF32
    row_topk_idx[0, :] = -1
    recompute_row_topk(scores, row_topk_vals, row_topk_idx, 0)

    return None