import time
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

# ----------------------- CONFIG -----------------------
N_ROWS = 28*28
N_COLS = 2589
DTYPE = np.float32
RUNS = 30
I, J = 2, 150
NUMBA_THREADS = None


@njit(parallel=True, fastmath=True, cache=True)
def leftMatmulTranspose_numba(x, i, j, m):
    n_rows, n_cols = x.shape
    m00 = m[0, 0]; m01 = m[0, 1]
    m10 = m[1, 0]; m11 = m[1, 1]

    if i < n_rows and j < n_rows:
        for c in prange(n_cols):
            ti = x[i, c]; tj = x[j, c]
            x[i, c] = m00 * ti + m01 * tj
            x[j, c] = m10 * ti + m11 * tj
    elif i < n_rows:
        for c in prange(n_cols):
            x[i, c] = m00 * x[i, c]
    elif j < n_rows:
        for c in prange(n_cols):
            x[j, c] = m11 * x[j, c]


ROWBUF = None
def _init_global_rowbuf(n_cols, dtype):
    global ROWBUF
    if ROWBUF is None:
        ROWBUF = np.zeros((2, n_cols), dtype=dtype)


def leftMatmul_numpy_globalbuf(x, i, j, m):
    ROWBUF[0, :] = x[i, :]
    ROWBUF[1, :] = x[j, :]

    np.dot(m, ROWBUF, out=ROWBUF)

    x[i, :] = ROWBUF[0, :]
    x[j, :] = ROWBUF[1, :]


TEMP_ROW2 = None
def _ensure_temp2(n_cols, dtype):
    global TEMP_ROW2
    if TEMP_ROW2 is None:
        TEMP_ROW2 = np.zeros(n_cols, dtype=dtype)


def leftMatmul_numpy_1temp(x, i, j, m):
    n_rows, n_cols = x.shape

    if j <= n_rows:
        xi = TEMP_ROW2
        xi[:] = x[i, :]
        x[i, :] = x[j, :]
        x[i, :] *= m[0, 1]
        x[j, :] *= m[1, 1]
        x[j, :] += m[1, 0] * xi
        x[i, :] += m[0, 0] * xi
    else:
        x[i, :] *= m[0, 0]



# ------------------------------ UTILS -------------------
def time_func(func, x0, i, j, m, runs):
    ts = []
    for _ in range(runs):
        x = x0.copy()
        t0 = time.perf_counter()
        func(x, i, j, m)
        ts.append(time.perf_counter() - t0)
    return np.array(ts)


def main():
    if NUMBA_THREADS is not None:
        set_num_threads(NUMBA_THREADS)

    print(f"Config: N_ROWS={N_ROWS:,}, N_COLS={N_COLS}, dtype={DTYPE}, runs={RUNS}")
    print(f"Numba threads: {get_num_threads()}  |  I={I}  J={J}")
    assert 0 <= I < N_ROWS and 0 <= J < N_ROWS and I != J

    rng = np.random.default_rng(42)
    x0 = rng.standard_normal((N_ROWS, N_COLS), dtype=DTYPE)
    m  = rng.standard_normal((2, 2), dtype=DTYPE)

    _init_global_rowbuf(N_COLS, DTYPE)
    _ensure_temp2(N_COLS, DTYPE)

    # Warm-up (JIT + touch BLAS)
    tmp = x0.copy(); leftMatmulTranspose_numba(tmp, I, J, m)
    tmp = x0.copy(); leftMatmul_numpy_globalbuf(tmp, I, J, m)

    # Correctness check
    xa = x0.copy(); xb = x0.copy()
    leftMatmulTranspose_numba(xa, I, J, m)
    leftMatmul_numpy_globalbuf(xb, I, J, m)
    max_abs_diff = np.max(np.abs(xa - xb))
    ok = np.allclose(
        xa, xb,
        rtol=1e-5 if DTYPE == np.float32 else 1e-12,
        atol=1e-6 if DTYPE == np.float32 else 1e-12
    )
    print(f"Max abs diff: {max_abs_diff:e}  |  Allclose: {ok}")

    # Timings
    t_numba = time_func(leftMatmulTranspose_numba, x0, I, J, m, RUNS)
    t_numpy = time_func(leftMatmul_numpy_globalbuf, x0, I, J, m, RUNS)
    t_1temp = time_func(leftMatmul_numpy_1temp, x0, I, J, m, RUNS)

    def fmt(a): return f"min {a.min():.6f}s | mean {a.mean():.6f}s | std {a.std(ddof=1):.6f}s"
    print("\nResults:")
    print(f"Numba (rows, prange):   {fmt(t_numba)}")
    print(f"NumPy global rowbuf:    {fmt(t_numpy)}")
    print(f"NumPy global 1temp:     {fmt(t_1temp)}")
    print(f"Speedup (NumPy/Numba): {t_numba.mean()/t_numpy.mean():.2f}×")
    print(f"Speedup (1temp/Numba): {t_numba.mean() / t_1temp.mean():.2f}×")

if __name__ == "__main__":
    main()
