"""
group_eig.py

Fast Group / Block Online symmetric eigendecomposition (PCA) implementation.

Import:
    import group_eig as group_op

Main API:
    group_op.fit_group_full_recompute(s, p, n_iter, u)
    group_op.fit_group(s, p, n_iter, u)
    group_op.fit(s, p, n_iter, u)
    group_op.fit_batched(...)
    group_op.fit_batched_traced(...)


Key optimizations:
    1. Does NOT build the full scores matrix.
    2. Computes only the top candidate list per retained row.
    3. Uses Numba for score scanning.
    4. Reuses candidate work arrays inside the fit loop.
    5. Uses BLAS-backed NumPy block updates via np.linalg.eigh (symmetry-aware).
    6. Avoids np.argsort over full score rows.
    7. Resymmetrizes S after each iteration to control roundoff drift.

Algorithm (block-parallel variant of Algorithm 1 for symmetric eigendecomposition):
    For each group iteration:
        - For every i = 0,...,p-1, scan all j > i and keep top candidates
          scored by lambda_max(S_tilde_ij) - S_ii, where S_tilde_ij is the
          2x2 symmetric block [[S_ii, S_ij], [S_ji, S_jj]].
        - Build group [0,...,p-1, j_0,...,j_{p-1}] using one unused
          candidate per row (greedy conflict resolution).
        - Compute eigendecomposition of the selected symmetric block.
        - Apply the congruence transform G^T S_block G on the selected
          rows AND columns (same G on both sides, since S is symmetric).

Notes vs. group_svd.py:
    - S is n x n and symmetric, so there is no rectangular / overcomplete case
      and no separate left/right transforms. One orthogonal G per pivot.
    - Score is unified: lambda_max(S_tilde_ij) - S_ii for all (i, j).
    - tr_p(S) = sum_{t<p} S_tt is a free, natural progress monitor
      (the running approximation to the sum of top-p eigenvalues, i.e.
      explained variance for uncentered PCA).
"""

import numpy as np

try:
    from numba import njit, prange, set_num_threads
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

    def prange(*args):
        return range(*args)

    def set_num_threads(n):
        return None


np.set_printoptions(suppress=True, precision=4, linewidth=200)

NEG_INF = np.float64(-1e30)


# 16 is usually enough, 32 if many rows choose same candidates
TOP_CANDIDATES_PER_ROW = 32

NUMBA_THREADS = 1


def set_group_num_threads(n):
    global NUMBA_THREADS
    NUMBA_THREADS = int(n)
    if NUMBA_AVAILABLE:
        set_num_threads(NUMBA_THREADS)


def evr_from_recon_sym(S_true, u, p):
    """
    Explained variance ratio for symmetric S given an orthonormal basis u.

    EVR = trace(U_p^T S U_p) / trace(S)

    where U_p = u[:, :p]. For uncentered PCA on data X (S = X X^T),
    this equals the fraction of total variance captured by the top-p
    components.
    """
    U_p = u[:, :p]
    num = np.trace(U_p.T @ S_true @ U_p)
    den = np.trace(S_true)
    if den == 0.0:
        return 0.0
    return float(num / den)


def get_evr_on_matrix(S, u, p):
    return evr_from_recon_sym(S, u, p)


def tr_p(s, p):
    """Sum of the first p diagonal entries of s (spec's progress monitor)."""
    return float(np.sum(np.diag(s)[:p]))


@njit(cache=True)
def _insert_top_candidate(val, idx, vals, idxs):
    kmax = vals.shape[0]
    if val <= vals[kmax - 1]:
        return

    k = kmax - 1
    while k > 0 and val > vals[k - 1]:
        vals[k] = vals[k - 1]
        idxs[k] = idxs[k - 1]
        k -= 1

    vals[k] = val
    idxs[k] = idx


@njit(parallel=True, cache=True)
def compute_row_top_candidates_numba(s, p, top_vals, top_idxs):
    """
    For each i in 0..p-1, scan j in i+1..n-1 and keep the top_m largest
    gain scores

        C_ij = lambda_max(S_tilde_ij) - S_ii

    where S_tilde_ij = [[S_ii, S_ij], [S_ji, S_jj]] (symmetric 2x2).

    The two eigenvalues of a symmetric 2x2 [[a, b], [b, z]] are

        lam = 0.5 * (a + z) +/- 0.5 * sqrt((a - z)^2 + 4 b^2)

    so the discriminant is exactly non-negative; no clamp needed beyond
    a defensive floor for roundoff.
    """
    n_local = s.shape[0]
    top_m = top_vals.shape[1]

    for i in prange(p):
        for k in range(top_m):
            top_vals[i, k] = NEG_INF
            top_idxs[i, k] = -1

        a = s[i, i]

        for j in range(i + 1, n_local):
            b = s[i, j]
            z = s[j, j]

            tr = a + z
            diff = a - z
            disc = diff * diff + 4.0 * b * b

            # Defensive: disc is mathematically >= 0 for symmetric 2x2,
            # but floor anyway in case of denormal weirdness.
            if disc < 0.0:
                disc = 0.0

            lam1 = 0.5 * (tr + np.sqrt(disc))
            val = lam1 - a

            # inline top insertion for speed
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
    Greedy conflict resolution: for each i in 0..p-1, pick the first
    candidate in its top-m list that is not yet used. The group always
    starts with [0, 1, ..., p-1] and is extended by up to p more indices.
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


def block_eig_update(s, u, indices):
    """
    Apply one block congruence update.

    Let C be the selected index set (rows AND columns, since S is symmetric).

    Sbar = S[C, C]   (a symmetric submatrix)
    Sbar = G Lambda G^T

    Then:
        S[C, :] = G^T @ S[C, :]
        S[:, C] = S[:, C] @ G
        u[:, C] = u[:, C] @ G

    The columns of G are reordered so that the largest eigenvalues come
    first; this is what pulls mass onto the leading diagonal of S, which
    is what the algorithm is optimizing (tr_p).

    After the two-sided update, S is symmetric in exact arithmetic; we
    resymmetrize in the outer loop to suppress floating-point drift.
    """
    n_local = s.shape[0]

    col_indices = np.asarray(indices, dtype=np.int64)
    col_indices = col_indices[(col_indices >= 0) & (col_indices < n_local)]

    if col_indices.size <= 1:
        return u, s

    # dedup preserving order
    seen = set()
    ordered_cols = []
    for idx in col_indices:
        ii = int(idx)
        if ii not in seen:
            ordered_cols.append(ii)
            seen.add(ii)

    col_indices = np.asarray(ordered_cols, dtype=np.int64)

    Sbar = s[np.ix_(col_indices, col_indices)]

    # Symmetrize Sbar before decomposing to clean up any prior drift.
    Sbar = 0.5 * (Sbar + Sbar.T)

    # eigh returns eigenvalues in ascending order; reverse so the largest
    # land on the lowest indices of the selected block.
    _, G = np.linalg.eigh(Sbar)
    G = np.ascontiguousarray(G[:, ::-1])

    # Congruence update: same G acts on the selected rows (from the left
    # via G^T) and the selected columns (from the right via G).
    s_rows = np.ascontiguousarray(s[col_indices, :])
    s[col_indices, :] = G.T @ s_rows

    s_cols = np.ascontiguousarray(s[:, col_indices])
    s[:, col_indices] = s_cols @ G

    u_cols = np.ascontiguousarray(u[:, col_indices])
    u[:, col_indices] = u_cols @ G

    return u, s


def fit_group_full_recompute(s, p, n_iter, u, top_m=TOP_CANDIDATES_PER_ROW):
    """
    Block-parallel one-sided Jacobi-style iteration for symmetric
    eigendecomposition. Mutates s and u in place (after the dtype cast)
    and also returns them.

    Parameters
    ----------
    s : (n, n) array, symmetric
        Working matrix; will be rotated toward block-diagonal form with
        the leading p-by-p block carrying the top eigenvalues.
    p : int
        Target eigenspace dimension; must satisfy p <= n.
    n_iter : int
        Number of outer iterations. Each outer iteration applies one
        block congruence on up to 2p indices.
    u : (n, n) array
        Running orthonormal basis; the first p columns approximate the
        top-p eigenvectors on return.
    top_m : int
        Candidates retained per pivot row.
    """
    if s.dtype != np.float64:
        s = s.astype(np.float64, copy=False)
    if u.dtype != np.float64:
        u = u.astype(np.float64, copy=False)

    n_local = s.shape[0]
    if s.shape[1] != n_local:
        raise ValueError(f"s must be square. Got shape {s.shape}.")

    if p > n_local:
        raise ValueError(f"p must be <= n. Got p={p}, n={n_local}.")
    if p >= n_local:
        return u, s

    top_m = int(max(1, top_m))

    top_vals = np.empty((p, top_m), dtype=np.float64)
    top_idxs = np.empty((p, top_m), dtype=np.int64)

    for _ in range(int(n_iter)):
        compute_row_top_candidates_numba(s, p, top_vals, top_idxs)

        indices = choose_group_from_top_candidates(top_idxs, p, n_local)

        if indices.size <= p:
            break

        u, s = block_eig_update(s, u, indices)

        # Resymmetrize to control roundoff drift from the two-sided update.
        s = 0.5 * (s + s.T)

    return u, s


def fit_group(s, p, n_iter, u):
    return fit_group_full_recompute(s, p, n_iter, u)


def fit(s, p, n_iter, u):
    return fit_group_full_recompute(s, p, n_iter, u)


def _accumulate_covariance(X_batch):
    """
    Form S_batch = X_batch @ X_batch.T for an (n, b) data slice.
    No centering is applied; this is uncentered PCA / eigendecomposition
    of the scatter matrix. If the caller wants centered PCA they should
    center X before passing it in.
    """
    return X_batch @ X_batch.T


def fit_batched(trueX, p, n_iter, batch_size=300, monitor=None, eval_every=1):
    """
    Streaming/online eigendecomposition over a data matrix trueX of shape
    (n, N). Internally accumulates uncentered covariance contributions
    per batch and applies block-eig iterations.

    Deflation between batches: after processing a batch, the running
    S in the current rotated basis has its top-p-by-p block carrying
    the approximate leading eigenvalues. We keep that top block, drop
    the trailing (n-p)-by-(n-p) corner (treated as residual), and fold
    in the next batch's contribution computed in the current basis:

        Y_new = u.T @ X_new
        S_next[:p, :p] = S[:p, :p] + Y_new[:p, :] @ Y_new[:p, :].T
        S_next[:p, p:] = Y_new[:p, :] @ Y_new[p:, :].T
        S_next[p:, :p] = S_next[:p, p:].T
        S_next[p:, p:] = Y_new[p:, :] @ Y_new[p:, :].T

    The top-p rows/cols therefore retain accumulated information from
    all prior batches, while the orthogonal complement is reset each
    batch from the latest contribution only. This is the symmetric
    analog of the SVD code's `hstack((x[:, :p], u.T @ X_new))` deflation.

    Parameters
    ----------
    trueX : (n, N) array
        Full data stream.
    p : int
        Target eigenspace dimension.
    n_iter : int
        Iterations per batch.
    batch_size : int
        Number of data columns per batch. Forced up to n if smaller.
    monitor : (n, n) array or None
        If provided, an external symmetric reference matrix used to
        compute EVR after each batch (for tracking convergence). If
        None, falls back to tr_p of the running matrix.
    eval_every : int
        How often (in batches) to recompute the EVR/tr_p trace.

    Returns
    -------
    traces : (used_batches,) array
        Progress metric after each batch (EVR if monitor given, else tr_p).
    u : (n, n) array
        Approximate eigenbasis; first p columns are the leading components.
    s : (n, n) array
        Final working symmetric matrix in the rotated basis.
    """
    trueX = trueX.astype(np.float64, copy=False)
    n_local, n_total = trueX.shape

    if p > n_local:
        raise ValueError(f"p must be <= n. Got p={p}, n={n_local}.")

    if batch_size < n_local:
        print(f"Batch size too small! Setting to {n_local}")
        batch_size = n_local

    total_batches = n_total // batch_size + (1 if n_total % batch_size else 0)

    u = np.identity(n_local, dtype=np.float64)
    s = np.zeros((n_local, n_local), dtype=np.float64)
    traces = np.zeros(total_batches, dtype=np.float64)

    start_index = 0
    end_index = min(batch_size, n_total)

    # First batch: form S from scratch in the original (identity) basis.
    X_batch = trueX[:, start_index:end_index]
    s = _accumulate_covariance(X_batch)
    s = 0.5 * (s + s.T)

    last_trace = 0.0
    used_batches = 0

    for bi in range(total_batches):
        u, s = fit_group_full_recompute(s, p, n_iter, u)

        if monitor is not None:
            if bi % eval_every == 0:
                last_trace = get_evr_on_matrix(
                    monitor.astype(np.float64, copy=False), u, p
                )
            traces[bi] = last_trace
        else:
            traces[bi] = tr_p(s, p)

        used_batches = bi + 1

        if end_index == n_total:
            break

        start_index += batch_size
        end_index = min(end_index + batch_size, n_total)

        # Deflate and fold in the next batch in the current basis.
        X_next = trueX[:, start_index:end_index]
        Y_new = u.T @ X_next                       # (n, b_next)
        Y_top = Y_new[:p, :]
        Y_bot = Y_new[p:, :]

        s_next = np.empty_like(s)
        s_next[:p, :p] = s[:p, :p] + Y_top @ Y_top.T
        s_next[:p, p:] = Y_top @ Y_bot.T
        s_next[p:, :p] = s_next[:p, p:].T
        s_next[p:, p:] = Y_bot @ Y_bot.T

        s = 0.5 * (s_next + s_next.T)

    return traces[:used_batches], u, s


def fit_batched_traced(trueX, p, n_iter, batch_size=300, monitor=None, eval_every=1):
    """
    Same as fit_batched but also returns the cumulative sample count
    at the end of each batch (handy for plotting EVR vs. samples-seen).
    """
    trueX = trueX.astype(np.float64, copy=False)
    n_local, n_total = trueX.shape

    if p > n_local:
        raise ValueError(f"p must be <= n. Got p={p}, n={n_local}.")

    if batch_size < n_local:
        print(f"Batch size too small! Setting to {n_local}")
        batch_size = n_local

    total_batches = n_total // batch_size + (1 if n_total % batch_size else 0)

    u = np.identity(n_local, dtype=np.float64)
    s = np.zeros((n_local, n_local), dtype=np.float64)
    traces = np.zeros(total_batches, dtype=np.float64)
    samples_seen_arr = np.zeros(total_batches, dtype=np.int64)

    start_index = 0
    end_index = min(batch_size, n_total)

    X_batch = trueX[:, start_index:end_index]
    s = _accumulate_covariance(X_batch)
    s = 0.5 * (s + s.T)

    last_trace = 0.0
    used_batches = 0

    for bi in range(total_batches):
        u, s = fit_group_full_recompute(s, p, n_iter, u)

        if monitor is not None:
            if bi % eval_every == 0:
                last_trace = get_evr_on_matrix(
                    monitor.astype(np.float64, copy=False), u, p
                )
            traces[bi] = last_trace
        else:
            traces[bi] = tr_p(s, p)

        samples_seen_arr[bi] = end_index
        used_batches = bi + 1

        if end_index == n_total:
            break

        start_index += batch_size
        end_index = min(end_index + batch_size, n_total)

        X_next = trueX[:, start_index:end_index]
        Y_new = u.T @ X_next
        Y_top = Y_new[:p, :]
        Y_bot = Y_new[p:, :]

        s_next = np.empty_like(s)
        s_next[:p, :p] = s[:p, :p] + Y_top @ Y_top.T
        s_next[:p, p:] = Y_top @ Y_bot.T
        s_next[p:, :p] = s_next[:p, p:].T
        s_next[p:, p:] = Y_bot @ Y_bot.T

        s = 0.5 * (s_next + s_next.T)

    return traces[:used_batches], samples_seen_arr[:used_batches], u