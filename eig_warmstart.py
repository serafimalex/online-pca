"""Partial Householder warm starts for OnlineEIG and OnlineGroupEIG.

Usage (exactly the p-step partial tridiagonalization):
    warm = HouseholderWarmStart()
    estimator.partial_fit(Z, warmstart=warm)

Optional, different initializer, aimed at the leading p-dimensional subspace:
    warm = HouseholderWarmStart(krylov_dim=2*P, ritz=True, start='random', seed=0)

The estimator calls the callback only after its FIRST covariance update.
Both S and U are mutated IN PLACE. Their invariant U @ S @ U.T is preserved
up to floating-point roundoff. No full n-by-n eigendecomposition is performed
unless krylov_dim >= n and ritz=True. No data centering.

Dependencies: NumPy, SciPy; threadpoolctl when blas_threads is not None.
"""
from __future__ import annotations

import operator
import time
from contextlib import nullcontext

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.linalg.blas import get_blas_funcs
from scipy.linalg.lapack import get_lapack_funcs

try:
    from threadpoolctl import ThreadpoolController
except ImportError:
    ThreadpoolController = None

__all__ = ['HouseholderWarmStart', 'householder_warmstart']

_SYMV, _SYR2, _GEMM = get_blas_funcs(('symv', 'syr2', 'gemm'), dtype=np.float64)
_LARFG = get_lapack_funcs('larfg', dtype=np.float64)

def _similarity_lower(A, v, tau):
    """A <- H A H, H = I - tau*v*v.T; only A's lower triangle is valid.

    The input is Fortran-contiguous so SciPy can overwrite it without a
    hidden n-by-n copy at every reflector. Never form the dense matrix H.
    """
    if tau == 0.0:
        return A
    w = _SYMV(tau, A, v, lower=1)
    w -= (0.5 * tau * float(v @ w)) * v
    return _SYR2(-1.0, v, w, a=A, lower=1, overwrite_a=1)


def _apply_product(Ubase, vectors, taus, identity, permutation):
    """Accumulate H1...Hh = I - V R V.T using compact WY, then update U.

    R here is the small triangular WY factor, NOT the tridiagonal block.
    Applying the product with a BLAS-3 operation avoids h full U updates.
    """
    h = len(vectors)
    if h == 0:
        return Ubase
    V = np.asfortranarray(np.column_stack(vectors))
    R = np.zeros((h, h))
    for j, tau in enumerate(taus):
        R[j, j] = tau
        if j:
            R[:j, j] = -tau * (R[:j, :j] @ (V[:, :j].T @ V[:, j]))
    if identity:
        # Ubase = I[:, permutation], where permutation is identity or one swap.
        UV = V if permutation is None else V[permutation, :]
    else:
        UV = Ubase @ V
    left = np.asfortranarray(UV @ R)
    return _GEMM(-1.0, left, V, trans_b=1, beta=1.0,
                 c=Ubase, overwrite_c=1)


class HouseholderWarmStart:
    """Warm start wrappers.

    Parameters
    ----------
    krylov_dim : int or None, default None
        Size m of the leading tridiagonal block. None means m=p, exactly the
        requested construction. Values greater than n are capped at n; m<p
        is rejected. At most min(m, n-2) nontrivial elimination stages are
        needed; some stages may use the identity reflector.
    ritz : bool, default False
        False leaves the partial tridiagonal form unchanged. True diagonalizes
        its m-by-m leading block in descending eigenvalue order, rotating S
        and the first m columns of U consistently. For m=p this changes only
        the basis, NOT the retained subspace or its explained variance.
    start : {'first', 'maxdiag', 'random'}, default 'first'
        'first' uses the unmodified coordinate ordering, exactly as in the
        manuscript. 'maxdiag' first swaps the largest diagonal to position 0.
        'random' first applies one extra Householder reflector with a random
        first basis vector (up to sign). These last two options MODIFY the
        proposed initialization. Seeds live in the incoming U coordinate frame.
    seed : nonnegative int, default 0
        Local random seed for start='random'. Reused deterministically at each
        call; the global NumPy RNG state is not changed.
    blas_threads : int or None, default 1
        Thread limit within the initializer; the previous limits are restored.
        None leaves BLAS settings unchanged. Native thread limits are process-
        wide, so do not run concurrent thread-based fits with differing limits.
    check_finite : bool, default True
        Validate S and U for NaN/infinity. No O(n^3) orthogonality test is done.

    Attributes
    ----------
    calls_ : number of successful invocations.
    last_info_ : dimensions, actual reflector count, exact breakdown locations,
        beta at the m/(n-m) boundary before Ritz rotation, initial/final captured
        trace, final subspace residual, and total elapsed seconds.

    This callback itself is NOT first-call-only: the estimator owns that guard.
    """
    def __init__(self, krylov_dim=None, ritz=False, start='first', seed=0,
                 blas_threads=1, check_finite=True):
        self.krylov_dim = (None if krylov_dim is None else krylov_dim)

        self.start = start
        self.ritz = bool(ritz)
        self.seed = seed
        self.blas_threads = (None if blas_threads is None else blas_threads)
        self.check_finite = bool(check_finite)

        self._controller = (ThreadpoolController()
                            if ThreadpoolController is not None else None)
        self.calls_ = 0
        self.last_info_ = {}

    def __call__(self, S, U, p):
        t0 = time.perf_counter()
        n = S.shape[0]
        m = p if self.krylov_dim is None else min(n, self.krylov_dim)

        limit = (nullcontext() if self.blas_threads is None else
                 self._controller.limit(limits=self.blas_threads, user_api='blas'))

        with limit:
            info = self._initialize(S, U, p, m)

        info['seconds'] = time.perf_counter() - t0
        self.last_info_ = info
        self.calls_ += 1

        # The existing callbacks ignore this return value; mutation is essential.
        return S, U

    def _initialize(self, S, U, p, m):
        n = S.shape[0]
        initial_trace = float(np.trace(S[:p, :p]))

        # BLAS rank-two updates need Fortran layout. Copy once, not per stage.
        A = np.array(S, dtype=np.float64, order='F', copy=True)
        A *= 0.5
        A += 0.5 * S.T

        Ubase = np.array(U, dtype=np.float64, order='F', copy=True)
        identity = bool(np.all(np.diag(U) == 1.0) and np.count_nonzero(U) == n)
        permutation = None
        start_index = 0 if self.start == 'first' else None

        if self.start == 'maxdiag':
            start_index = int(np.argmax(np.diag(A)))
            if start_index != 0:
                permutation = np.arange(n)
                permutation[[0, start_index]] = permutation[[start_index, 0]]
                A = np.array(A[np.ix_(permutation, permutation)], order='F')
                Ubase = np.array(Ubase[:, permutation], order='F')

        vectors, taus = [], []
        if self.start == 'random' and n > 1:
            q = np.random.default_rng(self.seed).standard_normal(n)
            q /= np.linalg.norm(q)
            _, tail, tau = _LARFG(n, float(q[0]), q[1:].copy(), overwrite_x=1)
            v = np.empty(n)
            v[0], v[1:] = 1.0, tail
            if tau != 0.0:
                A = _similarity_lower(A, v, tau)
                vectors.append(v)
                taus.append(float(tau))

        stages = min(m, max(n - 2, 0))
        breakdowns = []
        reduction_reflectors = 0
        for k in range(stages):
            first = k + 1
            alpha, tail, tau = _LARFG(n - first, float(A[first, k]), A[first + 1:, k].copy(), overwrite_x=1)
            if tau != 0.0:
                v = np.zeros(n)
                v[first] = 1.0
                v[first + 1:] = tail
                A = _similarity_lower(A, v, tau)
                vectors.append(v)
                taus.append(float(tau))
                reduction_reflectors += 1
            if alpha == 0.0:
                breakdowns.append(k + 1)  # one-based column numbers

            # Explicit structural zeros, just as in standard tridiagonalization.
            A[first, k] = alpha
            A[first + 1:, k] = 0.0

        # BLAS has updated only the lower triangle. Mirror it ONCE.
        for j in range(n - 1):
            A[j, j + 1:] = A[j + 1:, j]
        Ubase = _apply_product(Ubase, vectors, taus, identity, permutation)
        beta = float(A[m, m - 1]) if m < n else 0.0
        ritz_residuals = None
        if self.ritz:
            if m == 1:
                eigenvalues = np.array([A[0, 0]])
                Z = np.ones((1, 1))
            else:
                eigenvalues, Z = eigh_tridiagonal(np.diag(A)[:m].copy(), np.diag(A, k=-1)[:m-1].copy(), check_finite=False, lapack_driver='stev')
                eigenvalues = eigenvalues[::-1]
                Z = np.asfortranarray(Z[:, ::-1])

            # Update the FULL basis and transformed matrix, not only components_.
            Ubase[:, :m] = Ubase[:, :m] @ Z
            A[:m, :m] = 0.0
            A[np.arange(m), np.arange(m)] = eigenvalues
            if m < n:
                A[:m, m:] = 0.0
                A[m:, :m] = 0.0
                coupling = beta * Z[-1, :]
                A[:m, m] = coupling
                A[m, :m] = coupling
            ritz_residuals = (np.abs(beta * Z[-1, :p])).tolist()

        S[...] = A
        U[...] = Ubase
        return dict(p=p, krylov_dim=m, ritz=self.ritz, start=self.start,
                    start_index=start_index, reduction_stages=stages,
                    reduction_reflectors=reduction_reflectors,
                    total_reflectors=len(vectors), exact_breakdowns=breakdowns,
                    beta=beta, initial_trace=initial_trace,
                    initialized_trace=float(np.trace(S[:p, :p])),
                    subspace_residual_fro=float(np.linalg.norm(S[p:, :p])),
                    ritz_residual_norms=ritz_residuals)


def householder_warmstart(S, U, p):
    """Direct callback for the exact p-step, unseeded, no-Ritz construction.

    For configurable settings and recorded timings, use HouseholderWarmStart.
    """
    return HouseholderWarmStart()(S, U, p)
