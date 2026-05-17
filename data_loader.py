import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split


def _find_dataset_dir():
    env_path = os.environ.get("DATASET_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if glob.glob(os.path.join(parent, "black_*.php")):
        return parent
    cwd = os.getcwd()
    if glob.glob(os.path.join(cwd, "black_*.php")):
        return cwd
    raise FileNotFoundError(
        "Dataset not found. Set DATASET_PATH env var to the directory containing black_*.php and white_*.php files."
    )


def load_raw_files(data_dir=None):
    if data_dir is None:
        data_dir = _find_dataset_dir()
    black_files = sorted(glob.glob(os.path.join(data_dir, "black_*.php")))
    white_files = sorted(glob.glob(os.path.join(data_dir, "white_*.php")))
    return black_files, white_files


def read_file(path, max_chars=200000):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(max_chars)
    except Exception:
        return ""


def load_dataset(data_dir=None, random_state=42):
    black_files, white_files = load_raw_files(data_dir)
    rng = np.random.RandomState(random_state)

    n = min(len(black_files), len(white_files))
    black_idx = rng.choice(len(black_files), n, replace=False)
    white_idx = rng.choice(len(white_files), n, replace=False)
    black_files = [black_files[i] for i in sorted(black_idx)]
    white_files = [white_files[i] for i in sorted(white_idx)]

    texts, labels = [], []
    for path in black_files:
        t = read_file(path)
        if t:
            texts.append(t)
            labels.append(1)
    for path in white_files:
        t = read_file(path)
        if t:
            texts.append(t)
            labels.append(0)

    texts = np.array(texts, dtype=object)
    labels = np.array(labels, dtype=np.int32)

    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.15, stratify=labels, random_state=random_state
    )
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def print_dataset_stats(data_dir=None):
    black_files, white_files = load_raw_files(data_dir)
    n_black = len(black_files)
    n_white = len(white_files)
    n_total = n_black + n_white

    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Raw class distribution:")
    print(f"  Black (webshell): {n_black}")
    print(f"  White (benign):   {n_white}")
    print(f"  Total:            {n_total}")

    all_files = black_files + white_files
    sizes = []
    for path in all_files:
        try:
            sizes.append(os.path.getsize(path))
        except Exception:
            pass
    sizes = np.array(sizes)

    print(f"\nFile size statistics (bytes):")
    print(f"  Min:    {sizes.min():,}")
    print(f"  Max:    {sizes.max():,}")
    print(f"  Mean:   {sizes.mean():,.1f}")
    print(f"  Median: {np.median(sizes):,.1f}")
    print(f"  Std:    {sizes.std():,.1f}")

    n_balanced = min(n_black, n_white)
    n_total_balanced = n_balanced * 2
    n_train = int(round(n_total_balanced * 0.70))
    n_val = int(round(n_total_balanced * 0.15))
    n_test = n_total_balanced - n_train - n_val

    print(f"\nAfter balancing (50/50):")
    print(f"  Per class:      {n_balanced}")
    print(f"  Total balanced: {n_total_balanced}")
    print(f"  Train (70%):    {n_train}")
    print(f"  Val   (15%):    {n_val}")
    print(f"  Test  (15%):    {n_test}")
    print("=" * 60)
