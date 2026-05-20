import numpy as np
from scipy.linalg.blas import get_blas_funcs


def partial_hessenberg_householder_inplace(X, p):
    n, N = X.shape

    dtype = X.dtype
    Q = np.eye(n, dtype=dtype)

    wX = np.empty(max(N - 1, 1), dtype=dtype)
    wQ = np.empty(n, dtype=dtype)

    for k in range(p):
        x = X[k:, k]

        norm_x = np.linalg.norm(x)

        if norm_x == 0:
            continue

        alpha = x[0]

        if alpha == 0:
            phase = dtype.type(1)
        else:
            phase = alpha / abs(alpha)

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

        Q_block = Q[:, k:]

        z = wQ[:n]

        np.dot(Q_block, v, out=z)
        z *= tau

        Q_block -= np.multiply.outer(z, v.conj())

        X[k, k] = beta
        X[k + 1:, k] = 0

    return X, Q


def partial_symmetric_householder_inplace(S, p):
    n = S.shape[0]

    ger = get_blas_funcs("ger", arrays=(S,))

    Q = np.eye(n, dtype=S.dtype, order="F")

    u = np.zeros(n, dtype=S.dtype)
    w = np.empty(n, dtype=S.dtype)
    z = np.empty(n, dtype=S.dtype)

    num_steps = min(p, max(n - 2, 0))

    for k in range(num_steps):
        x = S[k + 1:, k]

        if np.linalg.norm(x[1:]) <= 10e-8:
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

        np.dot(Q, u, out=z)
        z *= tau

        ger(alpha=-1.0, x=z, y=u, a=Q, overwrite_a=1)

        S[k + 1, k] = beta
        S[k, k + 1] = beta

        if k + 2 < n:
            S[k + 2:, k] = 0.0
            S[k, k + 2:] = 0.0

    return S, Q


np.random.seed(0)

n = 6
N = 8
p = 3


### first test with svd

X = np.random.randn(n, N)
X_original = X.copy()

X, U = partial_hessenberg_householder_inplace(X, p)

print(X)

print("Reconstruction error:", np.linalg.norm(X_original - U @ X))
print("Q orthogonality error:", np.linalg.norm(U.T @ U - np.eye(n)))



### second test for eig

A = np.random.randn(n, n)
S = 0.5 * (A @ A.T)

S = np.asfortranarray(S)
S0 = S.copy(order="F")

S, Q = partial_symmetric_householder_inplace(S, p)

zero_error = 0.0
for k in range(p):
    if k + 2 < n:
        zero_error += np.linalg.norm(S[k + 2:, k]) ** 2
        zero_error += np.linalg.norm(S[k, k + 2:]) ** 2

zero_error = np.sqrt(zero_error)
print("Zero-pattern error:", zero_error)

print("Symmetry error:", np.linalg.norm(S - S.T))
print("Orthogonality error:", np.linalg.norm(Q.T @ Q - np.eye(n)))
print("Reconstruction error:", np.linalg.norm(S0 - Q @ S @ Q.T))

print(S)
