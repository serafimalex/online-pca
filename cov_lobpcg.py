import warnings

import numpy as np
from scipy.sparse.linalg import lobpcg

from validation import check_batch, check_init


class OnlineCovLOBPCG:
    """
    Baseline with the same memory as OnlineEIG / OnlineGroupEIG.

    Keeps the running second-moment matrix C = sum x x^T in the original
    coordinates. After every batch the top-p eigenvectors are refined with a
    few LOBPCG iterations, starting from the previous batch's eigenvectors.

        est = OnlineCovLOBPCG(n=d, p=P, n_iter=1)
        for batch in stream:  # batch is (d, m)
            est.partial_fit(batch)
        Q = est.Q_            # (d, p), approx top-p eigenvectors

    State:
        self.C_ : (n, n) running second-moment matrix
        self.Q_ : (n, p) current eigenvector estimates
    """

    def __init__(self, n, p, n_iter=1, seed=0, dtype=np.float64, check_finite=True):
        n, p, n_iter = check_init(n, p, n_iter)
        self.n = n
        self.p = p
        self.n_iter = n_iter
        self.dtype = dtype
        self.check_finite = check_finite

        self.C_ = np.zeros((n, n), dtype=dtype)
        rng = np.random.default_rng(seed)
        self.Q_ = np.linalg.qr(rng.standard_normal((n, p)))[0].astype(dtype)

        self.n_samples_seen_ = 0

    def partial_fit(self, X_batch):
        Xb = check_batch(X_batch, self.n, self.dtype, self.check_finite)
        self.C_ += Xb @ Xb.T

        # With a small maxiter lobpcg always warns that it did not reach its
        # tolerance. That is expected here: we only want a few refinement steps.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, self.Q_ = lobpcg(self.C_, self.Q_, largest=True, maxiter=self.n_iter)

        self.n_samples_seen_ += Xb.shape[1]
        return self

    @property
    def components_(self):
        """Top-p eigenvectors as rows (sklearn convention)."""
        return self.Q_.T

    def transform(self, X):
        return self.Q_.T @ np.asarray(X, dtype=self.dtype)

    def inverse_transform(self, codes):
        return self.Q_ @ np.asarray(codes, dtype=self.dtype)
