import numpy as np
from scipy.linalg.blas import get_blas_funcs


def partial_hessenberg_householder_inplace(X, p, u=None):
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

        if k + 1 < N:
            X_trail = X[k:, k + 1:]
            m = N - k - 1
            w = wX[:m]
            np.dot(v.conj(), X_trail, out=w)
            w *= tau
            X_trail -= np.multiply.outer(v, w)

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
    partial_hessenberg_householder_inplace(x_batch, p, u)
    return x_batch, u


def symmetric_warmstart(x_batch, u, p):
    d = x_batch.shape[0]
    S = np.asfortranarray(x_batch @ x_batch.T)
    Q = np.eye(d, dtype=x_batch.dtype, order="F")
    partial_symmetric_householder_inplace(S, p, Q)
    x_batch[:] = Q.T @ x_batch
    u[:] = u @ Q
    return x_batch, u


def symmetric_warmstart_eig(S, u, p):
    """
    EIG-flavor warmstart: partially tridiagonalize S (n x n symmetric) in place
    via Householder, accumulate the rotation Q into u (u <- u @ Q), and apply
    Q^T S Q similarity transform in place.

    The EIG-side analog of `symmetric_warmstart`. Differs in that the input
    is already the symmetric matrix S (no x_batch @ x_batch.T step), so this
    is strictly cheaper.

    Mutates S and u in place. Returns (S, u).
    """
    n = S.shape[0]
    if not S.flags["F_CONTIGUOUS"]:
        S_F = np.asfortranarray(S)
    else:
        S_F = S
    Q = np.eye(n, dtype=S.dtype, order="F")
    partial_symmetric_householder_inplace(S_F, p, Q)
    if S_F is not S:
        S[:] = S_F
    u[:] = u @ Q
    return S, u