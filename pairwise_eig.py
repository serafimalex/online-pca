import numpy as np

from validation import check_batch, check_init


# ============================================================
# MODULE-LEVEL CONSTANTS
# ============================================================

NEG_INF = np.float64(-1e30)


# ============================================================
# SCORE COMPUTATION
# ============================================================
# For symmetric S, the 2x2 block at (i,j) is [[S_ii, S_ij], [S_ij, S_jj]].
# Its eigenvalues are:
#   lambda_max,min = (S_ii + S_jj) / 2  +/-  sqrt( ((S_ii - S_jj)/2)^2 + S_ij^2 )
# The score is lambda_max - S_ii.

def score_grid(S, p, diag):
    """
    Full (p, n) score grid C_ij, with j <= i masked to NEG_INF.

    Vectorized over both axes: this is the whole scan the pivot search needs.
    """
    n = S.shape[0]
    di = diag[:p, None]
    dj = diag[None, :]
    half_dif = 0.5 * (di - dj)
    radius = np.sqrt(half_dif * half_dif + S[:p, :] ** 2)
    vals = 0.5 * (di + dj) + radius - di
    vals[np.arange(n)[None, :] <= np.arange(p)[:, None]] = NEG_INF
    return vals


def row_scores(S, r, diag):
    """Scores C_rs for s > r (entries s <= r masked). Shape (n,)."""
    dr = diag[r]
    half_dif = 0.5 * (dr - diag)
    vals = 0.5 * (dr + diag) + np.sqrt(half_dif * half_dif + S[r, :] ** 2) - dr
    vals[:r + 1] = NEG_INF
    return vals


def col_scores(S, c, diag, limit):
    """Scores C_rc for r < limit. Shape (limit,)."""
    dr = diag[:limit]
    half_dif = 0.5 * (dr - diag[c])
    return 0.5 * (dr + diag[c]) + np.sqrt(half_dif * half_dif + S[:limit, c] ** 2) - dr


# ============================================================
# 2x2 EIGENDECOMPOSITION + ROTATION APPLICATION
# ============================================================

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
        if abs(tau) > 1e8:
            t = 0.5 / tau
        elif tau >= 0.0:
            t = 1.0 / (tau + np.sqrt(1.0 + tau * tau))
        else:
            t = 1.0 / (tau - np.sqrt(1.0 + tau * tau))
        c = 1.0 / np.sqrt(1.0 + t * t)
        s = t * c

    # Orientation: place the LARGER eigenvalue at (0,0), computing the actual
    # post-rotation diagonal entries for G = [[c, -s], [s, c]].
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


def apply_similarity(S, i, j, G):
    """
    In-place similarity transform S <- G^T S G acting only on rows/cols i, j.

    G is 2x2 with rows indexed (i, j). This touches rows i and j (left mult by
    G^T) and cols i and j (right mult by G).  cost: O(n)
    """
    g00, g01, g10, g11 = G[0, 0], G[0, 1], G[1, 0], G[1, 1]

    # --- Left multiply: S' = G^T S, affects rows i and j only ---
    row_i = S[i, :].copy()
    row_j = S[j, :].copy()
    S[i, :] = g00 * row_i + g10 * row_j
    S[j, :] = g01 * row_i + g11 * row_j

    # --- Right multiply: S'' = (G^T S) G, affects cols i and j only ---
    col_i = S[:, i].copy()
    col_j = S[:, j].copy()
    S[:, i] = g00 * col_i + g10 * col_j
    S[:, j] = g01 * col_i + g11 * col_j

    # Force the off-diagonals (i,j) and (j,i) to exact zero - cleaner than
    # relying on floating-point cancellation, and keeps the score grid honest.
    S[i, j] = 0.0
    S[j, i] = 0.0


def apply_right_to_U(U, i, j, G):
    """In-place U <- U G, only affects columns i, j of U.  Cost: O(n)."""
    g00, g01, g10, g11 = G[0, 0], G[0, 1], G[1, 0], G[1, 1]
    col_i = U[:, i].copy()
    col_j = U[:, j].copy()
    U[:, i] = g00 * col_i + g10 * col_j
    U[:, j] = g01 * col_i + g11 * col_j


# ============================================================
# CORE FIT LOOP
# ============================================================

def fit_safe(S, p, n_iter, u):
    """
    Run up to n_iter Jacobi rotations on S, accumulating them into u.

    Pivot selection is the argmax over the full (p, n) score grid, as in
    Algorithm 1. Only the rows and columns a rotation actually invalidates are
    recomputed afterwards: rows i_q and j_q, columns i_q and j_q.

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

    Returns
    -------
    u, S : the (possibly-modified) basis and working matrix.
    """
    n = S.shape[0]
    if not 1 <= p <= n:
        raise ValueError(f"p must satisfy 1 <= p <= n; got p={p}, n={n}")

    diag = np.diag(S).copy()
    scores = score_grid(S, p, diag)

    for _ in range(n_iter):
        iq, jq = divmod(int(np.argmax(scores)), n)
        if scores[iq, jq] <= NEG_INF / 2:
            break

        G = jacobi_2x2_rotation(S[iq, iq], S[jq, jq], S[iq, jq])

        apply_similarity(S, iq, jq, G)
        apply_right_to_U(u, iq, jq, G)

        diag[iq] = S[iq, iq]
        diag[jq] = S[jq, jq]

        # Refresh only what the rotation invalidated:
        #   rows iq, jq  (their S_ii changed, so the whole row moves)
        #   cols iq, jq  (their S_jj changed, for every row above them)
        for r in (iq, jq):
            if r < p:
                scores[r, :] = row_scores(S, r, diag)
        for c in (iq, jq):
            limit = min(c, p)
            if limit > 0:
                scores[:limit, c] = col_scores(S, c, diag, limit)

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
        self.S_ : (n, n) running second-moment matrix in U's frame
        self.U_ : (n, n) accumulated rotation, columns = approx eigenvectors
    """

    def __init__(self, n, p, k_per_batch=500, dtype=np.float64, check_finite=True):
        n, p, k_per_batch = check_init(n, p, k_per_batch)
        self.n = n
        self.p = p
        self.k_per_batch = k_per_batch
        self.dtype = dtype
        self.check_finite = check_finite

        self.S_ = np.zeros((n, n), dtype=dtype)
        self.U_ = np.eye(n, dtype=dtype)

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
        Xb = check_batch(X_batch, self.n, self.dtype, self.check_finite)
        # Project new data into U's frame, then accumulate covariance
        Y = self.U_.T @ Xb                 # (n, m)
        # In-frame rank-m symmetric update: S += Y Y^T
        self.S_ += Y @ Y.T
        # Force symmetry (defensive: floating-point drift on big updates).
        self.S_ = 0.5 * (self.S_ + self.S_.T)

        # Apply warmstart preconditioning on first batch only.
        # The warmstart should mutate S and U so that U @ S_new @ U^T equals
        # the original observed covariance - i.e. S <- Q^T S Q, U <- U @ Q.
        if self._first_batch and warmstart is not None:
            warmstart(self.S_, self.U_, self.p)
        if self._first_batch:
            self._first_batch = False

        # Run k Jacobi iterations
        self.U_, self.S_ = fit_safe(self.S_, self.p, self.k_per_batch, self.U_)

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
    X : (d, n_total) float - full stream, columns are samples.
    P : int - number of components.
    G : int - Jacobi iterations per batch.
    batch_size : int.
    monitor : (d, m) - evaluation set.
    evr_fn : callable(monitor, U, P) -> float - typically `evr_from_U`.
    warmstart : callable(S, U, p) -> (S, U) or None.
        Applied ONLY on the first batch, between covariance accumulation and
        the first Jacobi iterations.

    Returns
    -------
    tr, time_axis, samples, eig : numpy arrays plus the fitted estimator.
    """
    import time
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kwargs):
            return it

    d = X.shape[0]
    n_total = X.shape[1]

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
