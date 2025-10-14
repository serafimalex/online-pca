import numpy as np
from time import perf_counter
from numba import njit, prange, set_num_threads, get_num_threads

CC = (1.0 / np.sqrt(2.0))


@njit(parallel=True, fastmath=True, cache=True)
def compute_scores(score, row_max_vals, row_max_idx, X, p):
    # X = np.asarray(X, dtype=float)
    n, m = X.shape

    # score.fill(-1)
    row_max_vals.fill(-1)
    row_max_idx.fill(-1)

    for i in prange(p):
        for j in range(i + 1, n):
            a = X[i, i]
            b = X[i, j]
            c = X[j, i]
            d = X[j, j]

            T = a*a + b*b + c*c + d*d
            det = a * d - b * c
            rad = T*T - 4.0*(det*det)

            val = CC * np.sqrt(T + np.sqrt(rad)) - a

            if val > row_max_vals[i]:
                row_max_vals[i] = val
                row_max_idx[i] = j

            score[i, j] = val

        for j in range(n + 1, m):
            a = X[i, i]
            b = X[i, j]

            T = a * a + b * b
            val = np.sqrt(T) - a

            if val > row_max_vals[i]:
                row_max_vals[i] = val
                row_max_idx[i] = j

            score[i, j] = val

    return score


@njit(parallel=True, fastmath=True, cache=True)
def compute_scores_vec(score, row_max_vals, row_max_idx, X, p, CC=np.sqrt(0.5)):
    n, m = X.shape

    # score.fill(-1.0)
    row_max_vals.fill(-1.0)
    row_max_idx.fill(-1)

    diag = np.diag(X)

    for i in prange(p):
        j_idx = np.arange(i + 1, n)

        a = diag[i]
        b = X[i, j_idx]
        c = X[j_idx, i]
        d = diag[j_idx]

        T = a*a + b*b + c*c + d*d
        det = a*d - b*c
        rad = T*T - 4.0*(det*det)
        score[i, j_idx] = CC * np.sqrt(T + np.sqrt(rad)) - a

        j_idx = np.arange(n + 1, m)

        # a = diag[i]
        b = X[i, j_idx]

        T = a * a + b * b
        score[i, j_idx] = np.sqrt(T) - a

        k = int(np.argmax(score[i, :]))
        row_max_vals[i] = (score[i, :])[k]
        row_max_idx[i] = k

    return score


# -----------------------------
# Testing + benchmarking helpers
# -----------------------------

def _time_call(fn, *args, repeats=3, **kwargs):
    """
    Run fn(*args, **kwargs) 'repeats' times and return (best_time, last_result).
    """
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = perf_counter()
        out = fn(*args, **kwargs)
        dt = perf_counter() - t0
        if dt < best:
            best = dt
    return best, out


def benchmark(score, row_max_vals, row_max_idx, n=1200, m=1200, p=600, repeats=30, block_size=None, seed=0):
    """
    Benchmark both implementations on one random case and print results.

    Note: the loop version can be slow if p and min(n, m) are large.
    Adjust n, m, p as needed for your machine.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m))
    X = np.ascontiguousarray(X, dtype=np.float64)

    # Time baseline
    t_loop, S_loop = _time_call(compute_scores, score, row_max_vals, row_max_idx, X, p, repeats=repeats)

    # Time vectorized (optionally blocked)
    t_vec, S_vec = _time_call(compute_scores_vec, score, row_max_vals, row_max_idx, X, p, repeats=repeats)

    # Verify equality
    same = np.allclose(S_loop, S_vec, equal_nan=True, rtol=1e-5, atol=1e-5)
    max_diff = np.nanmean(np.abs(S_loop - S_vec))

    print(f"\nBenchmark (n={n}, m={m}, p={p}, repeats={repeats}, block_size={block_size})")
    print(f"Looped version:      {t_loop:8.7f} s (avg of {repeats})")
    print(f"Vectorized version:  {t_vec:8.7f} s (avg of {repeats})")
    if t_vec > 0:
        print(f"Speedup (loop/vec):  {t_loop / t_vec:8.2f}×")
    print(f"Results equal:       {same} (max |diff| = {max_diff:.7e})")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # run_accuracy_tests()
    # Unblocked vectorized benchmark
    score = np.full((15, 1589), np.nan, dtype=float)
    row_max_vals = np.full(15, -1.0, dtype=np.float32)
    row_max_idx = np.full(15, -1, dtype=np.int64)

    benchmark(score, row_max_vals, row_max_idx, n=738, m=1589, p=15, repeats=5, seed=0)
    # Blocked vectorized benchmark (useful for very large matrices)
    # benchmark(score, row_max_vals, row_max_idx, n=8000, m=8000, p=100, repeats=3, block_size=512, seed=1)
