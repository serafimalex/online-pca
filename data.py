"""
data.py - download and preprocess the benchmark datasets.

Each dataset is stored as a single dense ``datasets/<name>.npy`` of shape
(N, d), float32, and read back with ``mmap_mode='r'`` so nothing holds the
full matrix unless it asks to.

    import data as ds

    ds.build("glove300")              # download + preprocess + save (once)
    X = ds.load("glove300")           # memmapped (N, d) float32
    monitor, stream = ds.split(X)     # holdout rows, then the rest

Heavy third-party imports (torchvision, pandas, sklearn, unlzw3) are done
inside the builders that need them, so building one dataset never requires
the dependencies of another.
"""

import os
import shutil
import urllib.request
import zipfile

import numpy as np

DATA_DIR = "datasets"
RAW_DIR = "raw_data"

#: Rows held out of the stream and used as the evaluation/monitor set.
MONITOR_ROWS = 2000

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


# ============================================================
# DOWNLOAD HELPERS
# ============================================================

def _download(url, dest):
    """Fetch ``url`` to ``dest`` unless it already exists. Prints progress."""
    if os.path.exists(dest):
        print(f"  cached: {dest}")
        return dest

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    print(f"  downloading {url}")

    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total:
                print(f"\r    {done / 2**20:.0f} / {total / 2**20:.0f} MB", end="")
            else:
                print(f"\r    {done / 2**20:.0f} MB", end="")
    print()
    os.replace(tmp, dest)
    return dest


def _unzip(zip_path, dest_dir, marker=None):
    """Extract ``zip_path`` into ``dest_dir``.

    ``marker`` is the path that proves the extraction already happened. It
    defaults to ``dest_dir``, which is only meaningful when ``dest_dir`` is
    created *by* the extraction - pass the extracted file explicitly when
    unpacking into a directory that already exists.
    """
    if os.path.exists(marker or dest_dir):
        print(f"  cached: {marker or dest_dir}")
        return dest_dir
    print(f"  extracting {os.path.basename(zip_path)}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    return dest_dir


def _raw(name):
    d = os.path.join(RAW_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


# ============================================================
# BUILDERS - each returns a dense (N, d) float32 array
# ============================================================

def _build_glove300():
    """GloVe 6B 300d word vectors -> (400000, 300).

    Replaces the old arxiv-embeddings dataset. Pulled from Stanford directly
    (the Kaggle mirror needs an API token); the 822 MB zip also carries the
    50d/100d/200d files if you ever want a d-scaling sweep.
    """
    import pandas as pd

    raw = _raw("glove")
    zip_path = _download("https://nlp.stanford.edu/data/glove.6B.zip",
                         os.path.join(raw, "glove.6B.zip"))
    txt_path = os.path.join(raw, "glove.6B.300d.txt")
    if not os.path.exists(txt_path):
        print("  extracting glove.6B.300d.txt")
        with zipfile.ZipFile(zip_path) as z, open(txt_path, "wb") as out:
            with z.open("glove.6B.300d.txt") as src:
                shutil.copyfileobj(src, out)

    # Column 0 is the token, columns 1.. are the vector. quoting=3 is
    # csv.QUOTE_NONE: GloVe's vocabulary contains bare " and ' tokens that
    # would otherwise swallow the rest of the file.
    print("  parsing 300d vectors")
    blocks = []
    for chunk in pd.read_csv(txt_path, sep=" ", header=None, quoting=3,
                             engine="c", chunksize=50_000, na_filter=False):
        blocks.append(chunk.iloc[:, 1:].to_numpy(dtype=np.float32))
    return np.vstack(blocks)


def _build_cifar10():
    """CIFAR-10 train+test, flattened -> (60000, 3072). Raw 0-255 values."""
    import torchvision

    train = torchvision.datasets.CIFAR10(root=RAW_DIR, train=True, download=True)
    test = torchvision.datasets.CIFAR10(root=RAW_DIR, train=False, download=True)
    return np.vstack([
        np.asarray(train.data).reshape(len(train.data), -1),
        np.asarray(test.data).reshape(len(test.data), -1),
    ]).astype(np.float32)


def _build_fashion_mnist():
    """Fashion-MNIST train+test, flattened and scaled to [0, 1] -> (70000, 784)."""
    import torchvision

    train = torchvision.datasets.FashionMNIST(root=RAW_DIR, train=True, download=True)
    test = torchvision.datasets.FashionMNIST(root=RAW_DIR, train=False, download=True)
    X = np.vstack([
        train.data.numpy().reshape(len(train), -1),
        test.data.numpy().reshape(len(test), -1),
    ]).astype(np.float32)
    X /= 255.0
    return X


def _build_isolet():
    """ISOLET 1-5 -> (7797, 617). Drops the trailing class-label column."""
    import pandas as pd
    from unlzw3 import unlzw

    raw = _raw("isolet")
    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/"
    frames = []
    for fname in ("isolet1+2+3+4.data.Z", "isolet5.data.Z"):
        z_path = _download(base + fname, os.path.join(raw, fname))
        dat_path = z_path[:-2]
        if not os.path.exists(dat_path):
            print(f"  decompressing {fname}")
            with open(dat_path, "wb") as f:
                f.write(unlzw(open(z_path, "rb").read()))
        frames.append(pd.read_csv(dat_path, header=None))

    # Last column is the 1-26 letter label, not a feature.
    X = pd.concat(frames, ignore_index=True).to_numpy(dtype=np.float32)
    return X[:, :-1]


def _build_yearprediction(max_samples=100_000):
    """YearPredictionMSD -> (100000, 90). Drops column 0 (the release year)."""
    import pandas as pd

    raw = _raw("yearprediction")
    zip_path = _download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/"
        "YearPredictionMSD.txt.zip",
        os.path.join(raw, "YearPredictionMSD.txt.zip"))
    txt_path = os.path.join(raw, "YearPredictionMSD.txt")
    _unzip(zip_path, raw, marker=txt_path)

    # read_csv rather than np.loadtxt: same result, ~50x faster on 500k rows.
    df = pd.read_csv(txt_path, header=None, nrows=max_samples)
    return df.iloc[:, 1:].to_numpy(dtype=np.float32)


def _build_gas_sensor(shuffle=False):
    """Gas Sensor Array Drift -> (13910, 128) from the LIBSVM-format batches.

    ``shuffle=False`` keeps the batches in acquisition order, so the stream is
    genuinely non-stationary (sensor drift across batch 1..10). Pass
    ``shuffle=True`` for an i.i.d. stream.
    """
    import glob

    raw = _raw("gas_sensor")
    zip_path = _download(
        "https://archive.ics.uci.edu/static/public/224/"
        "gas+sensor+array+drift+dataset.zip",
        os.path.join(raw, "gas_sensor.zip"))
    extract_dir = _unzip(zip_path, os.path.join(raw, "extracted"))

    files = sorted(glob.glob(os.path.join(extract_dir, "**", "*.dat"), recursive=True))
    print(f"  parsing {len(files)} batch files")
    X = np.vstack([_parse_libsvm(f, n_features=128) for f in files])

    if shuffle:
        X = X[np.random.default_rng(0).permutation(X.shape[0])]
    return X


def _build_har():
    """UCI HAR train+test -> (10299, 561)."""
    import pandas as pd

    raw = _raw("har")
    zip_path = _download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/"
        "UCI%20HAR%20Dataset.zip",
        os.path.join(raw, "har.zip"))
    extract_dir = _unzip(zip_path, os.path.join(raw, "extracted"))

    base = os.path.join(extract_dir, "UCI HAR Dataset")
    parts = [
        pd.read_csv(os.path.join(base, split, f"X_{split}.txt"),
                    sep=r"\s+", header=None).to_numpy(dtype=np.float32)
        for split in ("train", "test")
    ]
    return np.vstack(parts)


def _build_news20():
    """20 Newsgroups TF-IDF, densified -> (18846, 1000)."""
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer

    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    vec = TfidfVectorizer(max_features=1000, stop_words="english",
                          sublinear_tf=True, norm=None, dtype=np.float32)
    return vec.fit_transform(data.data).toarray().astype(np.float32)


def _build_synthetic(n_samples=100_000, n_features=256, rank=30, noise_std=0.1, seed=42):
    """Low-rank Gaussian data plus isotropic noise -> (100000, 256)."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((rank, n_features), dtype=np.float32)
    X = np.empty((n_samples, n_features), dtype=np.float32)
    for start in range(0, n_samples, 10_000):      # chunked to cap peak RAM
        stop = min(start + 10_000, n_samples)
        block = rng.standard_normal((stop - start, rank), dtype=np.float32) @ V
        if noise_std > 0:
            block += rng.normal(0, noise_std, block.shape).astype(np.float32)
        X[start:stop] = block
    return X


def _parse_libsvm(path, n_features):
    """Dense (rows, n_features) float32 from a LIBSVM-format file, labels dropped."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            vec = np.zeros(n_features, dtype=np.float32)
            for item in parts[1:]:                 # parts[0] is the label
                idx, _, val = item.partition(":")
                i = int(idx) - 1                   # LIBSVM indices are 1-based
                if 0 <= i < n_features:
                    vec[i] = float(val)
            rows.append(vec)
    return np.asarray(rows, dtype=np.float32)


BUILDERS = {
    "glove300":       _build_glove300,
    "cifar10":        _build_cifar10,
    "fashion_mnist":  _build_fashion_mnist,
    "isolet":         _build_isolet,
    "yearprediction": _build_yearprediction,
    "gas_sensor":     _build_gas_sensor,
    "har":            _build_har,
    "news20":         _build_news20,
    "synthetic":      _build_synthetic,
}


# ============================================================
# PUBLIC API
# ============================================================

def available():
    """Names accepted by :func:`build`."""
    return sorted(BUILDERS)


def path(name):
    """Where :func:`build` writes, and :func:`load` reads."""
    return os.path.join(DATA_DIR, f"{name}.npy")


def build(name, force=False, **kwargs):
    """Download + preprocess ``name`` into ``datasets/<name>.npy``.

    Returns the output path. Skips the work if the file exists and
    ``force`` is False. Extra kwargs go to the builder (e.g.
    ``build("gas_sensor", shuffle=True)``).
    """
    if name not in BUILDERS:
        raise KeyError(f"unknown dataset {name!r}; available: {available()}")

    out = path(name)
    if os.path.exists(out) and not force:
        X = np.load(out, mmap_mode="r")
        print(f"{name}: already built, shape {X.shape} -> {out}")
        return out

    print(f"{name}: building")
    X = np.ascontiguousarray(BUILDERS[name](**kwargs), dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"{name} builder returned shape {X.shape}, expected 2-D")

    os.makedirs(DATA_DIR, exist_ok=True)
    # Write via a handle: np.save(path, ...) would append a second ".npy" to
    # the ".part" name. Writing to a temp file first keeps a crashed build
    # from leaving behind a truncated dataset that looks valid.
    tmp = out + ".part"
    with open(tmp, "wb") as f:
        np.save(f, X)
    os.replace(tmp, out)
    print(f"{name}: shape {X.shape}, {X.nbytes / 2**20:.0f} MB -> {out}")
    return out


def load(name, mmap=True):
    """Return the (N, d) float32 matrix, memory-mapped by default."""
    p = path(name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found - run data.build({name!r}) first")
    return np.load(p, mmap_mode="r" if mmap else None)


def split(X, monitor_rows=MONITOR_ROWS):
    """Split rows into (monitor, stream).

    The first ``monitor_rows`` rows become the held-out evaluation set and are
    excluded from the stream, mirroring the old ``HOLDOUT_FILE = 0`` chunk.
    """
    if X.shape[0] <= monitor_rows:
        raise ValueError(
            f"need more than {monitor_rows} rows to hold out a monitor set; got {X.shape[0]}")
    return np.asarray(X[:monitor_rows]), X[monitor_rows:]


def info(name):
    """Shape/dtype/size of a built dataset, without reading it into RAM."""
    X = load(name)
    return {"name": name, "shape": X.shape, "dtype": str(X.dtype),
            "MB": round(X.nbytes / 2**20, 1), "path": path(name)}
