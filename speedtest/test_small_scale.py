import os
import scipy
import time
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

N_ROWS = 28*28
N_COLS = 2589
DTYPE = np.float32
RUNS = 60
I, J = 25, 260

NUMBA_THREADS = None


@njit(parallel=True, fastmath=True, cache=True)
def rightMatmul_numba(x, i, j, m):
    n_rows, n_cols = x.shape

    m00 = m[0, 0]
    m01 = m[0, 1]
    m10 = m[1, 0]
    m11 = m[1, 1]

    if i < n_cols and j < n_cols:
        for r in prange(n_rows):
            temp_i = x[r, i]
            temp_j = x[r, j]
            x[r, i] = m00 * temp_i + m10 * temp_j
            x[r, j] = m01 * temp_i + m11 * temp_j
    elif i < n_cols:
        for r in prange(n_rows):
            x[r, i] = m00 * x[r, i]
    elif j < n_cols:
        for r in prange(n_rows):
            x[r, j] = m11 * x[r, j]


BUF = None
TEMP_COL = None
def _init_global_buf(n_rows, dtype):
    global BUF
    global TEMP_COL
    if BUF is None:
        BUF = np.zeros((n_rows, 2), dtype=dtype)
        TEMP_COL = np.zeros(n_rows, dtype=dtype)


def rightMatmul_numpy_globalbuf(x, i, j, m):
    BUF[:, 0] = x[:, i]
    BUF[:, 1] = x[:, j]

    np.dot(BUF, m, out=BUF)

    x[:, i] = BUF[:, 0]
    x[:, j] = BUF[:, 1]


def rightMatmul_fast(x, i, j, m):
    n_rows, n_cols = x.shape

    TEMP_COL[:] = x[:, i]

    x[:, i] *= m[0, 0]
    x[:, i] += m[1, 0] * x[:, j]

    x[:, j] *= m[1, 1]
    x[:, j] += m[0, 1] * TEMP_COL





# ------------------------------ UTILS ------------------------------
def time_func(func, x0, i, j, m, runs):
    times = []
    for _ in range(runs):
        x = x0.copy()
        t0 = time.perf_counter()
        func(x, i, j, m)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.array(times)


def main():
    if NUMBA_THREADS is not None:
        set_num_threads(NUMBA_THREADS)

    print(f"Config: N_ROWS={N_ROWS:,}, N_COLS={N_COLS}, dtype={DTYPE}, runs={RUNS}")
    print(f"Numba threads: {get_num_threads()}")
    print(f"I={I}, J={J} (must be distinct and < N_COLS)")

    assert 0 <= I < N_COLS and 0 <= J < N_COLS and I != J

    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((N_ROWS, N_COLS), dtype=DTYPE)
    m = rng.standard_normal((2, 2), dtype=DTYPE)

    _init_global_buf(N_ROWS, DTYPE)

    x_warm = x0.copy()
    rightMatmul_numba(x_warm, I, J, m)
    x_warm = x0.copy()
    rightMatmul_numpy_globalbuf(x_warm, I, J, m)

    x_a = x0.copy()
    x_b = x0.copy()
    x_c = x0.copy()
    rightMatmul_numba(x_a, I, J, m)
    rightMatmul_numpy_globalbuf(x_b, I, J, m)
    rightMatmul_fast(x_c, I, J, m)
    max_abs_diff = np.max(np.abs(x_a - x_b))
    print(f"Max abs diff (after one transform): {max_abs_diff:e}")
    if not np.allclose(x_a, x_b, rtol=1e-5 if DTYPE==np.float32 else 1e-12,
                                  atol=1e-6 if DTYPE==np.float32 else 1e-12):
        print("WARNING: results differ more than tolerance.")

    max_abs_diff = np.max(np.abs(x_a - x_c))
    print(f"Max abs diff (after one transform): {max_abs_diff:e}")
    if not np.allclose(x_a, x_c, rtol=1e-5 if DTYPE == np.float32 else 1e-12,
                       atol=1e-6 if DTYPE == np.float32 else 1e-12):
        print("WARNING: results differ more than tolerance.")

    t_numba = time_func(rightMatmul_numba, x0, I, J, m, RUNS)
    t_numpy = time_func(rightMatmul_numpy_globalbuf, x0, I, J, m, RUNS)
    t_1temp = time_func(rightMatmul_fast, x0, I, J, m, RUNS)

    def fmt(arr):
        return f"min {arr.min():.6f}s | mean {arr.mean():.6f}s | std {arr.std(ddof=1):.6f}s"

    print("\nResults:")
    print(f"Numba (prange):     {fmt(t_numba)}")
    print(f"NumPy global buffer:{fmt(t_numpy)}")
    print(f"NumPy 1temp:        {fmt(t_1temp)}")

    speedup = t_numba.mean() / t_numpy.mean()
    print(f"Speedup (NumPy / Numba): {speedup:.2f}×")
    speedup = t_numba.mean() / t_1temp.mean()
    print(f"Speedup (1temp / Numba): {speedup:.2f}×")


if __name__ == "__main__":
    main()
