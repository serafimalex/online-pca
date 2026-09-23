"""
Input validation shared by the EIG estimators.

These checks exist because the failure modes underneath them are silent rather
than loud. Three in particular:

  * ``p > n`` makes the numba score kernels index past the end of ``S``. They are
    ``@njit(parallel=True)``, and numba's bounds checking does not apply to
    parallel loops, so the out-of-bounds read is never reported -- it just
    returns whatever happens to be in adjacent memory.

  * A single NaN or inf in a batch poisons every entry of ``S`` within one
    update. Pivot selection compares with ``>``, which is always False against
    NaN, so no rotation is ever applied again: the estimator silently stops
    learning while still returning an orthonormal-looking ``U_``.

  * A 1-D batch is a legal matmul against ``U_.T``, so ``Y @ Y.T`` collapses to a
    scalar that broadcasts over all of ``S``. The state is corrupted before the
    shape error surfaces further down.
"""

import numpy as np


def check_init(n, p, k_per_batch):
    """Validate constructor arguments. Raises ValueError on bad input."""
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if not isinstance(p, (int, np.integer)) or p < 1:
        raise ValueError(f"p must be a positive integer, got {p!r}")
    if p > n:
        raise ValueError(f"p must be <= n; got p={p}, n={n}")
    if not isinstance(k_per_batch, (int, np.integer)) or k_per_batch < 1:
        raise ValueError(f"k_per_batch must be a positive integer, got {k_per_batch!r}")
    return int(n), int(p), int(k_per_batch)


def check_batch(X, n, dtype, check_finite=True):
    """Validate and convert one batch. Returns an (n, m) array of ``dtype``.

    Samples are COLUMNS: X is (n_features, n_samples).
    """
    X = np.asarray(X, dtype=dtype)

    if X.ndim != 2:
        raise ValueError(
            f"X_batch must be 2-D (n_features, n_samples); got shape {X.shape}. "
            "A single sample must be passed as a column, e.g. x[:, None]."
        )
    if X.shape[0] != n:
        raise ValueError(
            f"X_batch has {X.shape[0]} features, but this estimator was built "
            f"for n={n}. Samples are columns: expected shape ({n}, n_samples)."
        )
    if check_finite and X.size and not np.isfinite(X).all():
        bad = "NaN" if np.isnan(X).any() else "inf"
        raise ValueError(
            f"X_batch contains {bad}. A single non-finite value poisons the whole "
            "second-moment matrix and silently halts learning; clean the batch "
            "first, or pass check_finite=False if you have handled this yourself."
        )
    return X
