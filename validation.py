"""
Input validation shared by the EIG estimators.

These checks exist because the failure modes underneath them are quiet, or else
report themselves in a way that points at the wrong place. Three in particular:

  * ``p > n`` does not fail where you would expect. Numpy slicing clamps, so
    ``S[:p, :]`` silently yields n rows, and the mistake only surfaces several
    lines later as an ``IndexError`` about a boolean mask shape -- which says
    nothing about p. Checking up front names the actual error.

  * A single NaN or inf in a batch poisons essentially every entry of ``S``
    within one update, and nothing raises. What happens next differs by variant:
    pairwise selects the NaN as a pivot (``np.argmax`` returns the index of a
    NaN) and carries it into ``U_``; group leaves ``U_`` looking clean and
    orthonormal on top of an ``S`` that is entirely NaN. Neither reports a
    problem, and neither is learning anything after that point.

  * A 1-D batch is a legal matmul against ``U_.T``, so ``Y @ Y.T`` collapses to a
    0-d scalar that broadcasts over all of ``S``. The state is corrupted before
    any shape error surfaces further down.
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
            "second-moment matrix within one update, and nothing downstream "
            "raises; clean the batch first, or pass check_finite=False if you "
            "have handled this yourself."
        )
    return X
