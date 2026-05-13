"""
group_svd.py

Fast Group / Block Online SVD implementation.

Import:
    import group_svd as group_op

Main API:
    group_op.fit_group_full_recompute(x, p, n_iter, u)
    group_op.fit_group(x, p, n_iter, u)
    group_op.fit(x, p, n_iter, u)
    group_op.fit_batched(...)
    group_op.fit_batched_traced(...)

This version is designed to be faster than the earlier prototype.

Key optimizations:
    1. Does NOT build the full scores matrix.
    2. Computes only the top candidate list per retained row.
    3. Uses Numba for score scanning.
    4. Reuses candidate work arrays inside the fit loop.
    5. Uses BLAS-backed NumPy block updates.
    6. Uses reduced SVD for rectangular blocks.
    7. Avoids np.argsort over full score rows.

Algorithm:
    For each group iteration:
        - For every i = 0,...,p-1, scan all j > i and keep top candidates.
        - Build group [0,...,p-1, j_0,...,j_{p-1}] using one unused candidate per row.
        - Compute SVD of the selected block.
        - Apply the block left/right transforms.
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

NEG_INF32 = np.float64(-1e30)
NEG_INF = np.float64(-1e30)

# Keep this modest. It is only for resolving duplicate candidates between rows.
# For p=15, 16 is usually enough; increase to 32 if many rows choose same candidates.
TOP_CANDIDATES_PER_ROW = 16

NUMBA_THREADS = 1


def set_group_num_threads(n):
    """
    Optional:
        group_op.set_group_num_threads(4)

    The block matrix multiplications may also use BLAS threads controlled by
    environment variables like OMP_NUM_THREADS / MKL_NUM_THREADS.
    """
    global NUMBA_THREADS
    NUMBA_THREADS = int(n)
    if NUMBA_AVAILABLE:
        set_num_threads(NUMBA_THREADS)


def evr_from_recon(X_true, X_recon):
    err = np.linalg.norm(X_true - X_recon, "fro") ** 2
    tot = np.linalg.norm(X_true, "fro") ** 2
    if tot == 0.0:
        return 0.0
    return 1.0 - err / tot


def get_evr_on_matrix(X, u, p):
    U_p = u[:, :p]
    Z = U_p.T @ X
    X_rec = U_p @ Z
    return evr_from_recon(X, X_rec)


@njit(cache=True)
def _insert_top_candidate(val, idx, vals, idxs):
    """
    Insert val/idx into descending top list if it qualifies.
    """
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
def compute_row_top_candidates_numba(x, p, top_vals, top_idxs):
    """
    For each retained row i, scan all valid j > i and keep the top candidates.

    This replaces full score matrix construction.

    top_vals shape: (p, top_m)
    top_idxs shape: (p, top_m)
    """
    d_local, n_active = x.shape
    top_m = top_vals.shape[1]

    for i in prange(p):
        for k in range(top_m):
            top_vals[i, k] = NEG_INF
            top_idxs[i, k] = -1

        a = x[i, i]
        aa = a * a

        for j in range(i + 1, n_active):
            b = x[i, j]

            if j < d_local:
                c = x[j, i]
                z = x[j, j]
            else:
                c = 0.0
                z = 0.0

            # sigma_1([[a,b],[c,z]]) from eigenvalues of T T^T
            row_sq = aa + b * b           # = a*a + b*b
            cc = c * c + z * z

            # --- Pre-sqrt pruning ---
            # We want val = sqrt(lam1) - a > threshold.
            # Since lam1 <= tr = row_sq + cc, we have
            # sqrt(lam1) <= sqrt(row_sq + cc), so
            # val <= sqrt(row_sq + cc) - a.
            # Skip if even this upper bound cannot beat the current worst kept value.
            threshold = top_vals[i, top_m - 1]
            # If a + threshold <= 0, the bound below is automatically satisfied
            # (any non-negative sqrt beats it), so we cannot prune; fall through.
            ap = a + threshold
            if ap > 0.0:
                # Equivalent test without sqrt: row_sq + cc <= ap*ap  =>  prune
                if row_sq + cc <= ap * ap:
                    continue

            ac = a * c + b * z

            tr = row_sq + cc
            det = row_sq * cc - ac * ac
            disc = tr * tr - 4.0 * det

            if disc < 0.0:
                disc = 0.0

            lam1 = 0.5 * (tr + np.sqrt(disc))
            if lam1 < 0.0:
                lam1 = 0.0

            val = np.sqrt(lam1) - a

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
    Build [0,...,p-1, j_0,...] using the first unused candidate from each row.

    This is intentionally simple because p is small.
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


def block_svd_update(x, u, indices):
    """
    Apply one block/group SVD update.

    Let C be selected column indices.
    Let R = C ∩ {0,...,d-1} be selected row indices.

    Xbar = X[R, C]
    Xbar = U S Vh

    Then:
        X[R, :] = U.T @ X[R, :]
        X[:, C] = X[:, C] @ V

    and:
        u[:, R] = u[:, R] @ U

    Notes:
        - For square blocks, R and C have the same size.
        - For overcomplete blocks, C can contain column-only indices >= d.
        - We use full_matrices=True to make V square with shape len(C) x len(C),
          so the right update is dimensionally valid for rectangular Xbar.
    """
    d_local, n_active = x.shape

    col_indices = np.asarray(indices, dtype=np.int64)
    col_indices = col_indices[(col_indices >= 0) & (col_indices < n_active)]

    if col_indices.size <= 1:
        return u, x

    # Preserve order and remove duplicates.
    seen = set()
    ordered_cols = []
    for idx in col_indices:
        ii = int(idx)
        if ii not in seen:
            ordered_cols.append(ii)
            seen.add(ii)

    col_indices = np.asarray(ordered_cols, dtype=np.int64)
    row_indices = col_indices[col_indices < d_local]

    if row_indices.size == 0:
        return u, x

    # Small block extraction.
    Xbar = x[np.ix_(row_indices, col_indices)]

    # Need full_matrices=True so V is len(C) x len(C) for rectangular blocks.
    U_local, _, Vh_local = np.linalg.svd(Xbar, full_matrices=True)
    V_local = Vh_local.T

    # BLAS-backed updates.
    # Fast path: by construction, choose_group_from_top_candidates always puts
    # [0, 1, ..., p-1] as the first p entries of the group, so row_indices is
    # exactly arange(p) and we can use a contiguous slice view instead of
    # fancy-indexing + ascontiguousarray (which would allocate + copy).
    p_rows = row_indices.size
    is_canonical_rows = (
        p_rows <= d_local
        and row_indices[0] == 0
        and row_indices[p_rows - 1] == p_rows - 1
    )

    if is_canonical_rows:
        # x[:p_rows, :] is already a contiguous C-order view.
        x[:p_rows, :] = U_local.T @ x[:p_rows, :]
    else:
        x_rows = np.ascontiguousarray(x[row_indices, :])
        x[row_indices, :] = U_local.T @ x_rows

    # Columns are arbitrary, so we still need to gather them.
    x_cols = np.ascontiguousarray(x[:, col_indices])
    x[:, col_indices] = x_cols @ V_local

    if is_canonical_rows:
        u[:, :p_rows] = u[:, :p_rows] @ U_local
    else:
        u_cols = np.ascontiguousarray(u[:, row_indices])
        u[:, row_indices] = u_cols @ U_local

    return u, x


def fit_group_full_recompute(x, p, n_iter, u, top_m=TOP_CANDIDATES_PER_ROW):
    """
    Fast group/block inner loop.

    Despite the name, this does not materialize the full score matrix.
    It fully rescans all candidates after every block update, but only stores
    top candidates per retained row.
    """
    if x.dtype != np.float64:
        x = x.astype(np.float64, copy=False)
    if u.dtype != np.float64:
        u = u.astype(np.float64, copy=False)

    d_local, n_active = x.shape

    if p > d_local:
        raise ValueError(f"p must be <= d. Got p={p}, d={d_local}.")
    if p >= n_active:
        return u, x

    top_m = int(max(1, top_m))

    top_vals = np.empty((p, top_m), dtype=np.float64)
    top_idxs = np.empty((p, top_m), dtype=np.int64)

    for _ in range(int(n_iter)):
        compute_row_top_candidates_numba(x, p, top_vals, top_idxs)

        indices = choose_group_from_top_candidates(top_idxs, p, n_active)

        # No candidate beyond [0,...,p-1]
        if indices.size <= p:
            break

        u, x = block_svd_update(x, u, indices)

    return u, x


def fit_group(x, p, n_iter, u):
    return fit_group_full_recompute(x, p, n_iter, u)


def fit(x, p, n_iter, u):
    return fit_group_full_recompute(x, p, n_iter, u)


def fit_batched(trueX, p, n_iter, batch_size=300, monitor=None, eval_every=1):
    """
    Batched / streaming wrapper.

    Returns:
        traces, u, x
    """
    trueX = trueX.astype(np.float64, copy=False)
    d_local, n_total = trueX.shape

    if p > d_local:
        raise ValueError(f"p must be <= d. Got p={p}, d={d_local}.")

    if batch_size < d_local:
        print(f"Batch size too small! Setting to {d_local}")
        batch_size = d_local

    total_batches = n_total // batch_size + (1 if n_total % batch_size else 0)

    u = np.identity(d_local, dtype=np.float64)
    traces = np.zeros(total_batches, dtype=np.float64)

    start_index = 0
    end_index = min(batch_size, n_total)

    x_batch = trueX[:, start_index:end_index].copy()

    last_trace = 0.0
    used_batches = 0

    for bi in range(total_batches):
        u, x = fit_group_full_recompute(x_batch, p, n_iter, u)

        if monitor is not None:
            if bi % eval_every == 0:
                last_trace = get_evr_on_matrix(monitor.astype(np.float64, copy=False), u, p)
            traces[bi] = last_trace
        else:
            traces[bi] = 0.0

        used_batches = bi + 1

        if end_index == n_total:
            break

        start_index += batch_size
        end_index = min(end_index + batch_size, n_total)

        x_batch = np.hstack((
            x[:, :p],
            u.T @ trueX[:, start_index:end_index]
        )).astype(np.float64, copy=False)

    return traces[:used_batches], u, x


def fit_batched_traced(trueX, p, n_iter, batch_size=300, monitor=None, eval_every=1):
    """
    Batched / streaming wrapper.

    Returns:
        traces, samples_seen, u
    """
    trueX = trueX.astype(np.float64, copy=False)
    d_local, n_total = trueX.shape

    if p > d_local:
        raise ValueError(f"p must be <= d. Got p={p}, d={d_local}.")

    if batch_size < d_local:
        print(f"Batch size too small! Setting to {d_local}")
        batch_size = d_local

    total_batches = n_total // batch_size + (1 if n_total % batch_size else 0)

    u = np.identity(d_local, dtype=np.float64)
    traces = np.zeros(total_batches, dtype=np.float64)
    samples_seen_arr = np.zeros(total_batches, dtype=np.int64)

    start_index = 0
    end_index = min(batch_size, n_total)

    x_batch = trueX[:, start_index:end_index].copy()

    last_trace = 0.0
    used_batches = 0

    for bi in range(total_batches):
        u, x = fit_group_full_recompute(x_batch, p, n_iter, u)

        if monitor is not None:
            if bi % eval_every == 0:
                last_trace = get_evr_on_matrix(monitor.astype(np.float64, copy=False), u, p)
            traces[bi] = last_trace
        else:
            traces[bi] = 0.0

        samples_seen_arr[bi] = end_index
        used_batches = bi + 1

        if end_index == n_total:
            break

        start_index += batch_size
        end_index = min(end_index + batch_size, n_total)

        x_batch = np.hstack((
            x[:, :p],
            u.T @ trueX[:, start_index:end_index]
        )).astype(np.float64, copy=False)

    return traces[:used_batches], samples_seen_arr[:used_batches], u
