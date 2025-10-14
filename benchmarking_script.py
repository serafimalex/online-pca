import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.datasets import fetch_openml
from tools.oja_pca import OjaPCA
import torch
import sys
import os
sys.path.append(os.path.abspath(".."))
from online_svd_buffer import fit_batched

from scipy.sparse.linalg import svds
from tqdm import tqdm
sys.path.append(os.path.abspath(".."))
from matplotlib.ticker import FuncFormatter
import os, sys
from pathlib import Path

def load_mnist_subset(n_samples=5000):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype(np.float32).T[:, :n_samples] / 255.0

    return X  # shape: (784, n_samples)

def load_fashion_mnist_subset(n_samples=5000):
    fmnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
    X = fmnist.data.astype(np.float32).T[:, :n_samples] / 255.0
    return X  # shape: (784, n_samples)

def load_usps_subset(n_samples=5000):
    usps = fetch_openml('USPS', version=1, as_frame=False)
    X = usps.data.astype(np.float32).T[:, :n_samples] / 255.0

    return X  # shape: (256, n_samples) since USPS has 16x16 images

def load_isolet_subset(n_samples=5000):
    isolet = fetch_openml('isolet', version=1, as_frame=False)
    # Features are already real-valued, just normalize by max
    X = isolet.data.astype(np.float32).T[:, :n_samples]
    print(X.shape)
    X /= np.max(X)  # scale to [0,1]
    return X  # shape: (617, n_samples), 617 audio features

def load_mnist_and_fashion(n_samples=5000):
    # Load MNIST
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X_mnist = mnist.data.astype(np.float32)[:n_samples] / 255.0
    y_mnist = mnist.target.astype(int)[:n_samples]

    # Load Fashion-MNIST
    fmnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
    X_fmnist = fmnist.data.astype(np.float32)[:n_samples] / 255.0
    y_fmnist = fmnist.target.astype(int)[:n_samples]

    # Stack them together
    X = np.vstack([X_mnist, X_fmnist]).T   # shape: (784, 2*n_samples)
    return X

def load_mnist_isolet_combined(n_samples=5000):
    X_mnist = load_mnist_subset(n_samples).astype(np.float32, copy=False)   # (784, n)
    X_iso   = load_isolet_subset(n_samples).astype(np.float32, copy=False)  # (617, n)

    d_mnist, _ = X_mnist.shape
    d_iso,   _ = X_iso.shape
    d_common = min(d_mnist, d_iso)

    if d_mnist > d_common:
        X_mnist = X_mnist[:d_common, :]
    if d_iso > d_common:
        X_iso = X_iso[:d_common, :]

    X_combined = np.concatenate([X_mnist, X_iso], axis=1).astype(np.float32, copy=False)

    mu = X_combined.mean(axis=1, keepdims=True).astype(X_combined.dtype)
    X_combined = X_combined - mu

    print(X_combined.shape)  # (d_common, 2*n_samples)
    return X_combined

def explained_variance_ratio(X, X_recon):
    error = np.linalg.norm(X - X_recon, 'fro') ** 2
    total = np.linalg.norm(X, 'fro') ** 2
    return 1 - error / total

def plot_traces(traces, X, p):

    traces = np.array(traces)
    u, s, vt = np.linalg.svd(X)
    aux = u.transpose() @ X @ vt.transpose()
    true_energy = np.trace(aux[:p, :p])


    true_energy = np.sum(np.linalg.svd(X, compute_uv=False)[:p])

    print(true_energy)
    print(traces[:-20])
    
    plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 11,
    "axes.titlepad": 8,
    })

    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    # Data
    ax.plot(traces, label='OnlinePCA', linewidth=2)
    ax.axhline(true_energy, linestyle=':', linewidth=2,
            label='Sum of first p singular values')

    # Y-limit: 5% above the red line (and safe if traces ever exceed it)
    y_top = 1.10 * max(true_energy, np.nanmax(traces))
    y_bottom = min(0, float(np.nanmin(traces)))
    ax.set_ylim(y_bottom, y_top)

    # Clean up axes
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.4)
    ax.margins(x=0)  # no extra horizontal padding

    # Tick formatting: show iterations as “k”
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}')
    )

    # Labels & legend
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Trace')
    ax.set_title('Trace progression vs. True energy (random)')
    ax.legend(frameon=False, loc='lower right')

    plt.tight_layout()
    plt.savefig('trace_mnist.pdf', bbox_inches='tight')
    plt.show()

def benchmark(method_name, fit_fn):
    start = time.time()
    U, S, Vt, X_recon, traces = fit_fn()
    elapsed = time.time() - start
    # if traces is not None:
    #     plot_traces(traces, X, p)
    evr = explained_variance_ratio(X, X_recon)
    return {
        "method": method_name,
        "time": elapsed,
        "explained_variance": evr,
    }

def run_benchmarks(X, p=50, g=200):
    results = []

    #sklearn PCA (full SVD)
    def run_pca():
        model = PCA(n_components=p, svd_solver="full")
        model.fit(X.T)
        X_recon = model.inverse_transform(model.transform(X.T)).T
        print("Done PCA")
        return model.components_.T, model.singular_values_, None, X_recon, None
    results.append(benchmark("PCA", run_pca))

    def run_online_pca():
        batch_sz = X.shape[0] * 2
        traces, U, X_approx = fit_batched(X, p, g, batch_sz)
        X_reduced = U.T[:p, :] @ X
        X_recon = U[:, :p] @ X_reduced
        print("Done OnlinePCA")
        return U, None, None, X_recon, traces
    results.append(benchmark("OnlinePCA", run_online_pca))

    # Incremental PCA
    def run_incpca():
        batch_sz = X.shape[0] * 2
        model = IncrementalPCA(n_components=p, batch_size=batch_sz)
        model.fit(X.T)
        X_recon = model.inverse_transform(model.transform(X.T)).T
        print("Done IncrementalPCA")
        return model.components_.T, None, None, X_recon, None
    results.append(benchmark("IncrementalPCA", run_incpca))

    def run_oja():
        model = OjaPCA(
            n_features=X.shape[0],
            n_components=p,
            eta=0.001,
        )
        X_tensor = torch.tensor(X.T)
        b_size = X.shape[0] * 2
        for i in range(0, len(X_tensor) - b_size, b_size):
            batch = X_tensor[i : i + b_size]
            if len(batch) < b_size:
                # This line means we use up to an extra partial batch over 1 pass
                batch = torch.cat([batch, X_tensor[: b_size - len(batch)]], dim=0)
            error = model(batch) if hasattr(model, "forward") else None
        recon = model.inverse_transform(model.transform(X_tensor))
        print("Done oja")
        return np.array(model.get_components()), None, None, np.array(recon).T, None
    results.append(benchmark("OjaPCA", run_oja))

    return results

if __name__ == "__main__":
    print("Loading dataset...")
    X = load_mnist_subset(n_samples=50000)

    # in case we want to enlarge the dataset
    n_samples = 50000
    repeats = int(np.ceil(n_samples / X.shape[1]))
    X = np.tile(X, (1, repeats))[:, :n_samples]
    
    print("Loaded dataset. Shape:")
    print(X.shape)

    print("Warming up Numba...")
    # numba warmup
    temp = X.copy()
    _, _, _ = fit_batched(temp, 15, 100, 1576)
    print("Numba warmed up. Running benchmarks...")

    p = 15
    g = 450

    results = run_benchmarks(X, p=p, g=g)
    results.sort(key=lambda d: d["time"]) 
    print("\nBenchmark Results:")
    for r in results:
        print(f"{r['method']:15s} | Time: {r['time']:.2f}s | Explained Var: {r['explained_variance']*100:.2f}%")

    methods = [r["method"] for r in results]
    times = [r["time"] for r in results]
    evrs = [r["explained_variance"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8,4))
    ax2 = ax1.twinx()

    ax1.bar(methods, times, alpha=0.6, label="Time (s)")
    ax2.plot(methods, evrs, "o-", color="red", label="Explained Var")

    ax1.set_ylabel("Time (s)")
    ax2.set_ylabel("Explained Variance")
    plt.title(f"PCA Benchmark (p={p}, g={g}) on MNIST subset")
    plt.show()

