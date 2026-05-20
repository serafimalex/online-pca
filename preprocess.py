import numpy as np
from scipy.linalg.blas import get_blas_funcs


def partial_hessenberg_householder_inplace(X, p, u=None):
    """
    Partial QR via Householder reflections on the first p columns of X.

    Mutates X in place so that X[k+1:, k] == 0 for k = 0..p-1.
    If u is provided, mutates it in place as u <- u @ Q so that
    (u_new) @ (X_new) == (u_old) @ (X_old).

    Parameters
    ----------
    X : (d, n) float64 array, modified in place.
    p : int, number of columns to triangularize.
    u : (d, d) float64 array or None. If given, accumulated into.

    Returns
    -------
    X, u  (both possibly mutated)
    """
    n, N = X.shape
    dtype = X.dtype

    wX = np.empty(max(N - 1, 1), dtype=dtype)
    wU = np.empty(n, dtype=dtype) if u is not None else None

    for k in range(p):
        x = X[k:, k]
        norm_x = np.linalg.norm(x)
        if norm_x == 0:
            continue

        alpha = x[0]
        phase = 1.0 if alpha == 0 else alpha / abs(alpha)
        beta = -phase * norm_x

        x[0] -= beta
        v = x
        tau = 2.0 / np.vdot(v, v)

        # Apply to trailing columns of X from the left
        if k + 1 < N:
            X_trail = X[k:, k + 1:]
            m = N - k - 1
            w = wX[:m]
            np.dot(v.conj(), X_trail, out=w)
            w *= tau
            X_trail -= np.multiply.outer(v, w)

        # Accumulate into u: u <- u @ H_k, where H_k acts on rows k:
        if u is not None:
            U_block = u[:, k:]
            z = wU
            np.dot(U_block, v, out=z)
            z *= tau
            U_block -= np.multiply.outer(z, v.conj())

        X[k, k] = beta
        X[k + 1:, k] = 0

    return X, u


def partial_symmetric_householder_inplace(S, p, Q=None):
    """
    Partial symmetric tridiagonalization of S = S.T via two-sided
    Householder reflections. Mutates S in place; if Q is given, mutates
    it as Q <- Q @ H so that Q_new @ S_new @ Q_new.T == Q_old @ S_old @ Q_old.T.

    Parameters
    ----------
    S : (d, d) symmetric float64 array, Fortran-ordered, modified in place.
    p : int, number of columns to tridiagonalize.
    Q : (d, d) float64 array or None.

    Returns
    -------
    S, Q
    """
    n = S.shape[0]
    ger = get_blas_funcs("ger", arrays=(S,))

    u = np.zeros(n, dtype=S.dtype)
    w = np.empty(n, dtype=S.dtype)
    z = np.empty(n, dtype=S.dtype) if Q is not None else None

    num_steps = min(p, max(n - 2, 0))

    for k in range(num_steps):
        x = S[k + 1:, k]
        if np.linalg.norm(x[1:]) <= 1e-8:
            S[k + 2:, k] = 0.0
            S[k, k + 2:] = 0.0
            continue

        norm_x = np.linalg.norm(x)
        if norm_x == 0.0:
            continue

        alpha = x[0]
        beta = -np.copysign(norm_x, alpha if alpha != 0.0 else 1.0)

        x[0] -= beta
        v = x
        tau = 2.0 / np.dot(v, v)

        u.fill(0.0)
        u[k + 1:] = v

        np.dot(S, u, out=w)
        w *= tau
        if k > 0:
            w[:k] = 0.0

        gamma = -0.5 * tau * np.dot(u, w)
        w += gamma * u
        if k > 0:
            w[:k] = 0.0

        ger(alpha=-1.0, x=u, y=w, a=S, overwrite_a=1)
        ger(alpha=-1.0, x=w, y=u, a=S, overwrite_a=1)

        if Q is not None:
            np.dot(Q, u, out=z)
            z *= tau
            ger(alpha=-1.0, x=z, y=u, a=Q, overwrite_a=1)

        S[k + 1, k] = beta
        S[k, k + 1] = beta
        if k + 2 < n:
            S[k + 2:, k] = 0.0
            S[k, k + 2:] = 0.0

    return S, Q


def hessenberg_warmstart(x_batch, u, p):
    """
    Apply partial Hessenberg preprocessing to a streaming batch.

    Mutates x_batch and u in place. After this call, x_batch[k+1:, k] == 0
    for k = 0..p-1, and u @ x_batch (new) == u @ x_batch (old).
    """
    partial_hessenberg_householder_inplace(x_batch, p, u)
    return x_batch, u


def symmetric_warmstart(x_batch, u, p):
    """
    Apply partial symmetric tridiagonalization preprocessing to a streaming
    batch via S = x_batch @ x_batch.T.

    Mutates x_batch and u in place: x_batch <- Q.T @ x_batch, u <- u @ Q,
    where Q comes from partial-tridiagonalizing S.
    """
    d = x_batch.shape[0]
    S = np.asfortranarray(x_batch @ x_batch.T)
    Q = np.eye(d, dtype=x_batch.dtype, order="F")
    partial_symmetric_householder_inplace(S, p, Q)
    # Apply Q to x and u
    x_batch[:] = Q.T @ x_batch
    u[:] = u @ Q
    return x_batch, u