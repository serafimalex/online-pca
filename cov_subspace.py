import numpy as np
import scipy.linalg as sla

from validation import check_batch, check_init


class OnlineCovSubspace:
    """
    Baseline with the same memory as OnlineEIG / OnlineGroupEIG.

    Keeps the running second-moment matrix C = sum x x^T in the original
    coordinates. On the first batch the top-p eigenvectors are computed
    exactly; after that, every `every` batches one step of subspace
    (orthogonal) iteration refines them:

        Q <- qr(C @ Q)

    starting from the previous batch's Q. This is the block power method
    (Golub & Van Loan, "orthogonal iteration"; Saad 2016 for evolving matrices).

        est = OnlineCovSubspace(n=d, p=P)
        for batch in stream:  # batch is (d, m)
            est.partial_fit(batch)
        Q = est.Q_            # (d, p), approx top-p eigenvectors, sorted

    With rayleigh_ritz (the default) each step ends with a Rayleigh-Ritz rotation,
    so the columns of Q are eigenvector estimates sorted by eigenvalue, at the cost
    of one more C @ Q per step. rayleigh_ritz=False only keeps a basis of the
    top-p subspace, whose columns are not individual eigenvectors.

    State:
        self.C_ : (n, n) running second-moment matrix
        self.Q_ : (n, p) current basis
    """

    def __init__(self, n, p, every=1, rayleigh_ritz=True, dtype=np.float64, check_finite=True):
        n, p, every = check_init(n, p, every)
        self.n = n
        self.p = p
        self.every = every
        self.rayleigh_ritz = rayleigh_ritz
        self.dtype = dtype
        self.check_finite = check_finite

        self.C_ = np.zeros((n, n), dtype=dtype)
        self.Q_ = None

        self.n_samples_seen_ = 0
        self._n_batches = 0

    def partial_fit(self, X_batch):
        Xb = check_batch(X_batch, self.n, self.dtype, self.check_finite)
        self.C_ += Xb @ np.ascontiguousarray(Xb.T)

        if self.Q_ is None:
            _, V = sla.eigh(self.C_, subset_by_index=[self.n - self.p, self.n - 1])
            self.Q_ = np.ascontiguousarray(V[:, ::-1])
        elif self._n_batches % self.every == 0:
            self.Q_ = np.linalg.qr(self.C_ @ self.Q_)[0]
            if self.rayleigh_ritz:
                w, V = np.linalg.eigh(self.Q_.T @ self.C_ @ self.Q_)
                self.Q_ = self.Q_ @ V[:, ::-1]

        self._n_batches += 1
        self.n_samples_seen_ += Xb.shape[1]
        return self

    @property
    def components_(self):
        """Top-p basis as rows (sklearn convention)."""
        return self.Q_.T

    def transform(self, X):
        return self.Q_.T @ np.asarray(X, dtype=self.dtype)

    def inverse_transform(self, codes):
        return self.Q_ @ np.asarray(codes, dtype=self.dtype)
