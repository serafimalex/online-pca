"""
:Author: Zachary T. Varley
:Year: 2025
:License: MIT

Oja batched update with a fixed learning rate.

This module implements the batched version of Oja's update rule for PCA. Each
batch is used to update the estimates of the top-k eigenvectors of the
covariance matrix. Then a QR decomposition is performed to re-orthogonalize the
estimates. QR is O(mn^2) compute via Schwarz-Rutishauser for m x n matrices.

References:

Allen-Zhu, Zeyuan, and Yuanzhi Li. “First Efficient Convergence for Streaming
K-PCA: A Global, Gap-Free, and Near-Optimal Rate.” 2017 IEEE 58th Annual
Symposium on Foundations of Computer Science (FOCS), IEEE, 2017, pp. 487–92.
DOI.org (Crossref), https://doi.org/10.1109/FOCS.2017.51.

Tang, Cheng. “Exponentially Convergent Stochastic K-PCA without Variance
Reduction.” Advances in Neural Information Processing Systems, vol. 32, 2019.

"""

import torch
from torch import Tensor


class OjaPCA(torch.nn.Module):
    def __init__(
        self,
        n_features: int,
        n_components: int,
        eta: float = 0.005,
        dtype: torch.dtype = torch.float32,
        use_oja_plus: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_components = n_components
        self.eta = eta
        self.use_oja_plus = use_oja_plus

        # Initialize parameters
        self.register_buffer("Q", torch.randn(n_features, n_components, dtype=dtype))
        self.register_buffer("step", torch.zeros(1, dtype=torch.int64))

        # For Oja++, we initialize columns gradually
        if self.use_oja_plus:
            self.register_buffer(
                "initialized_cols", torch.zeros(n_components, dtype=torch.bool)
            )
            self.register_buffer("next_col_to_init", torch.tensor(0, dtype=torch.int64))

    def forward(self, x: Tensor) -> None:
        """Update PCA with new batch of data using Oja's algorithm"""
        # Update then Orthonormalize Q_t using QR decomposition
        self.Q.copy_(torch.linalg.qr(self.Q + self.eta * (x.T @ (x @ self.Q)))[0])

        # Update step
        self.step.add_(1)

        # For Oja++, gradually initialize columns
        if self.use_oja_plus and self.next_col_to_init < self.n_components:
            if self.step % (self.n_components // 2) == 0:
                self.Q[:, self.next_col_to_init] = torch.randn(
                    self.n_features, dtype=self.Q.dtype
                )
                self.initialized_cols[self.next_col_to_init] = True
                self.next_col_to_init.add_(1)

    def get_components(self) -> Tensor:
        return self.Q.T

    def transform(self, x: Tensor) -> Tensor:
        return x @ self.Q

    def inverse_transform(self, x: Tensor) -> Tensor:
        return x @ self.Q.T
