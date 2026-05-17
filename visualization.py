import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = "results"


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _save(fig, name, dpi=150):
    _ensure_dir()
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_loss_curves_grid(all_histories, run_labels, title="Loss Curves (3x3 Grid)"):
    """Plot 1: 3x3 subplot grid of train/val loss curves for all 9 Phase 3 runs."""
    n = len(all_histories)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, (hist, label) in enumerate(zip(all_histories, run_labels)):
        ax = axes[i]
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], label="Train", linewidth=1.2)
        ax.plot(epochs, hist["val_loss"], label="Val", linewidth=1.2)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "01_loss_curves_grid.png")


def plot_accuracy_curves_grid(all_histories, run_labels, title="Accuracy Curves (3x3 Grid)"):
    """Plot 2: 3x3 subplot grid of train/val accuracy."""
    n = len(all_histories)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, (hist, label) in enumerate(zip(all_histories, run_labels)):
        ax = axes[i]
        epochs = range(1, len(hist["train_acc"]) + 1)
        ax.plot(epochs, hist["train_acc"], label="Train", linewidth=1.2)
        ax.plot(epochs, hist["val_acc"], label="Val", linewidth=1.2)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.5, 1.01)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "02_accuracy_curves_grid.png")


def plot_vectorization_comparison(histories_by_vec, arch_label, title="Vectorization Comparison"):
    """Plot 3: Overlay val loss curves for OH vs BoW vs TF-IDF on best architecture."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = {"tfidf": "#2196F3", "bow": "#FF9800", "onehot": "#4CAF50"}
    names = {"tfidf": "TF-IDF", "bow": "Bag-of-Words", "onehot": "One-Hot"}

    for vec, hist in histories_by_vec.items():
        epochs = range(1, len(hist["val_loss"]) + 1)
        c = colors.get(vec, "gray")
        label = names.get(vec, vec)
        axes[0].plot(epochs, hist["val_loss"], label=label, color=c, linewidth=1.5)
        axes[1].plot(epochs, hist["val_acc"], label=label, color=c, linewidth=1.5)

    axes[0].set_title(f"Val Loss — {arch_label}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(f"Val Accuracy — {arch_label}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.5, 1.01)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "03_vectorization_comparison.png")


def plot_architecture_comparison(histories_by_arch, vec_label, title="Architecture Comparison"):
    """Plot 4: Overlay val loss for A vs B vs C on best vectorization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = {"A": "#E91E63", "B": "#2196F3", "C": "#FF9800"}

    for arch, hist in histories_by_arch.items():
        epochs = range(1, len(hist["val_loss"]) + 1)
        c = colors.get(arch, "gray")
        axes[0].plot(epochs, hist["val_loss"], label=f"Arch {arch}", color=c, linewidth=1.5)
        axes[1].plot(epochs, hist["val_acc"], label=f"Arch {arch}", color=c, linewidth=1.5)

    for ax, ylabel in zip(axes, ["BCE Loss", "Accuracy"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[1].set_ylim(0.5, 1.01)
    axes[0].set_title(f"Val Loss — {vec_label}")
    axes[1].set_title(f"Val Accuracy — {vec_label}")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "04_architecture_comparison.png")


def plot_depth_sensitivity(depths, val_accs, test_accs=None, title="Depth Sensitivity Analysis"):
    """Plot 5: Accuracy vs num_blocks line plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, val_accs, "o-", label="Val Accuracy", linewidth=2, markersize=8, color="#2196F3")
    if test_accs is not None:
        ax.plot(depths, test_accs, "s--", label="Test Accuracy", linewidth=2, markersize=8, color="#E91E63")
    ax.set_xlabel("Number of Residual Blocks")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)
    ax.set_ylim(0.5, 1.01)
    fig.tight_layout()
    return _save(fig, "05_depth_sensitivity.png")


def plot_gradient_flow(grad_norms_A, grad_norms_B, epochs_list, title="Gradient Flow Analysis"):
    """Plot 6: Mean |grad| per layer at key epochs for Arch A vs B."""
    fig, axes = plt.subplots(2, len(epochs_list), figsize=(5 * len(epochs_list), 8))
    if len(epochs_list) == 1:
        axes = axes.reshape(2, 1)

    arch_data = [("Arch A (Plain)", grad_norms_A), ("Arch B (Residual)", grad_norms_B)]

    for row, (arch_name, grad_norms) in enumerate(arch_data):
        for col, ep in enumerate(epochs_list):
            ax = axes[row][col]
            norms = grad_norms.get(ep, [])
            if norms:
                ax.bar(range(len(norms)), norms, color="#2196F3" if row == 1 else "#E91E63", alpha=0.7)
            ax.set_title(f"{arch_name}\nEpoch {ep}", fontsize=9)
            ax.set_xlabel("Layer index")
            ax.set_ylabel("Mean |grad|")
            ax.grid(True, alpha=0.3)
            if norms:
                ax.set_yscale("log")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "06_gradient_flow.png")


def plot_roc_curves(all_results, title="ROC Curves — All Configurations"):
    """Plot 7: ROC curves for all 9 configurations overlaid with AUC in legend."""
    from metrics import roc_curve as my_roc

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.get_cmap("tab10")

    for i, (label, y_true, y_proba, auc) in enumerate(all_results):
        fprs, tprs, _ = my_roc(y_true, y_proba)
        ax.plot(fprs, tprs, label=f"{label} (AUC={auc:.3f})", color=cmap(i % 10), linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "07_roc_curves.png")


def plot_confusion_matrix(cm, class_names=("Benign", "Webshell"), title="Confusion Matrix — Best Model"):
    """Plot 8: Confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    thresh = cm.max() / 2.0
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                    color="white" if cm[r, c] > thresh else "black", fontsize=14)

    fig.tight_layout()
    return _save(fig, "08_confusion_matrix.png")


def _pca_2d(X, n_components=2):
    """PCA from scratch using SVD for dimensionality reduction to 2D."""
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:n_components].T


def _tsne_2d(X, perplexity=30.0, n_iter=1000, lr=200.0, seed=42):
    """
    t-SNE dimensionality reduction from scratch.
    van der Maaten & Hinton 2008 - Visualizing Data using t-SNE.
    JMLR 9:2579-2605. https://www.jmlr.org/papers/v9/vandermaaten08a.html
    Uses PCA initialization and early exaggeration.
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    X_init = _pca_2d(X)

    def _pairwise_sq_dists(A):
        sum_sq = np.sum(A ** 2, axis=1, keepdims=True)
        return sum_sq + sum_sq.T - 2.0 * A @ A.T

    def _compute_P(D_sq, target_entropy):
        n = D_sq.shape[0]
        P = np.zeros((n, n))
        for i in range(n):
            di = D_sq[i].copy()
            di[i] = np.inf
            lo, hi = 1e-10, 1e10
            beta = 1.0
            for _ in range(50):
                pi = np.exp(-di * beta)
                pi_sum = pi.sum()
                if pi_sum == 0:
                    pi_sum = 1e-12
                H = np.log(pi_sum) + beta * np.sum(pi * di) / pi_sum
                if abs(H - target_entropy) < 1e-5:
                    break
                if H < target_entropy:
                    hi = beta
                    beta = (lo + hi) / 2.0
                else:
                    lo = beta
                    beta = (lo + hi) / 2.0 if hi < 1e9 else beta * 2.0
            pi = np.exp(-di * beta)
            pi[i] = 0.0
            P[i] = pi / (pi.sum() + 1e-12)
        return (P + P.T) / (2.0 * n)

    target_entropy = np.log(perplexity)
    D_sq = _pairwise_sq_dists(X_init)
    P = _compute_P(D_sq, target_entropy)

    Y = rng.randn(n, 2) * 0.01
    dY = np.zeros_like(Y)
    gains = np.ones_like(Y)
    momentum = 0.5
    exaggeration = 4.0
    P_exag = P * exaggeration

    for step in range(n_iter):
        if step == 100:
            momentum = 0.8
            P_exag = P

        sum_sq_Y = np.sum(Y ** 2, axis=1, keepdims=True)
        dist_sq_Y = sum_sq_Y + sum_sq_Y.T - 2.0 * Y @ Y.T
        Q_unnorm = 1.0 / (1.0 + dist_sq_Y)
        np.fill_diagonal(Q_unnorm, 0.0)
        Q = Q_unnorm / (Q_unnorm.sum() + 1e-12)

        PQ = P_exag - Q
        grad = 4.0 * np.einsum("ij,ij,ik->ik", PQ, Q_unnorm, Y - Y[:, None, :].reshape(n, 1, 2).mean(0))

        for d in range(2):
            g = np.zeros(n)
            for i in range(n):
                g[i] = np.sum(PQ[i] * Q_unnorm[i] * (Y[i, d] - Y[:, d]))
            g *= 4.0
            gains[:, d] = np.where(
                np.sign(g) != np.sign(dY[:, d]),
                gains[:, d] * 0.2 + 0.8,
                gains[:, d] * 0.8 + 0.2
            )
            gains[:, d] = np.maximum(gains[:, d], 0.01)
            dY[:, d] = momentum * dY[:, d] - lr * gains[:, d] * g
            Y[:, d] += dY[:, d]

        Y -= Y.mean(axis=0)

    return Y


def plot_tsne(X_train, y_train, X_test, y_test, penultimate_fn, title="t-SNE Penultimate Layer"):
    """Plot 9: t-SNE of penultimate layer activations colored by class."""
    max_samples = 400
    rng = np.random.RandomState(42)

    n_train = min(max_samples // 2, len(X_train))
    n_test = min(max_samples // 2, len(X_test))
    idx_tr = rng.choice(len(X_train), n_train, replace=False)
    idx_te = rng.choice(len(X_test), n_test, replace=False)

    X_all = np.vstack([X_train[idx_tr], X_test[idx_te]])
    y_all = np.concatenate([y_train[idx_tr], y_test[idx_te]])
    split_flag = np.concatenate([np.zeros(n_train), np.ones(n_test)])

    features = penultimate_fn(X_all)
    if features.shape[1] > 50:
        features = _pca_2d(features.astype(np.float64))
        embedded = _tsne_2d(features, perplexity=30, n_iter=500)
    else:
        embedded = _tsne_2d(features.astype(np.float64), perplexity=30, n_iter=500)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors_class = {0: "#4CAF50", 1: "#F44336"}
    labels_class = {0: "Benign", 1: "Webshell"}

    for ax, (flag_val, split_name) in zip(axes, [(0, "Train"), (1, "Test")]):
        mask = split_flag == flag_val
        for cls in [0, 1]:
            cmask = mask & (y_all == cls)
            ax.scatter(
                embedded[cmask, 0], embedded[cmask, 1],
                c=colors_class[cls], label=labels_class[cls],
                alpha=0.6, s=20, edgecolors="none"
            )
        ax.set_title(f"t-SNE {split_name} Set")
        ax.legend()
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "09_tsne.png")


def plot_activation_benchmark(activation_names, val_losses, title="Activation Benchmark"):
    """Plot 10: Bar chart of final val loss per activation."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(activation_names)))
    bars = ax.bar(activation_names, val_losses, color=colors, edgecolor="black", linewidth=0.5)
    best_idx = int(np.argmin(val_losses))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)
    ax.set_xlabel("Activation Function")
    ax.set_ylabel("Val BCE Loss (10 epochs)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, val_losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return _save(fig, "10_activation_benchmark.png")


def plot_optimizer_benchmark(optimizer_names, val_losses, val_accs, title="Optimizer Benchmark"):
    """Plot 11: Bar chart of val loss and val accuracy per optimizer after 100 epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.get_cmap("Set2")(np.linspace(0, 1, len(optimizer_names)))

    for ax, values, ylabel in zip(axes, [val_losses, val_accs], ["Val BCE Loss", "Val Accuracy"]):
        bars = ax.bar(optimizer_names, values, color=colors, edgecolor="black", linewidth=0.5)
        best_idx = int(np.argmin(values) if "Loss" in ylabel else np.argmax(values))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)
        ax.set_xlabel("Optimizer")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "11_optimizer_benchmark.png")
