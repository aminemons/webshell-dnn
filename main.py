import os
import sys
import json
import time
import argparse
import numpy as np

from data_loader import load_dataset, print_dataset_stats
from vectorization import build_vectorizers, token_dropout
from model import build_and_train, build_model, _sigmoid
from metrics import evaluate, print_metrics, roc_curve, auc_roc
from optimizers import make_optimizer, make_scheduler
import visualization as viz

try:
    import cupy as cp
    _GPU_AVAILABLE = True
except ImportError:
    cp = None
    _GPU_AVAILABLE = False

RESULTS_DIR = "results"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")


def _ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _save_json(obj, path):
    _ensure_dirs()
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


def _to_cpu(X):
    if _GPU_AVAILABLE and cp is not None and isinstance(X, cp.ndarray):
        return cp.asnumpy(X)
    return X


def _fit_vectorizers(X_train_texts, X_val_texts, X_test_texts, tokenizer_type="bpe"):
    print(f"\nFitting vectorizers (tokenizer={tokenizer_type}) ...")
    t0 = time.time()
    oh_vec, bow_vec, tfidf_vec = build_vectorizers(tokenizer_type)

    print("  [1/9] OneHot fit+transform train ...", flush=True)
    X_train_oh = oh_vec.fit_transform(X_train_texts)
    print("  [2/9] OneHot transform val ...", flush=True)
    X_val_oh = oh_vec.transform(X_val_texts)
    print("  [3/9] OneHot transform test ...", flush=True)
    X_test_oh = oh_vec.transform(X_test_texts)

    print("  [4/9] BoW fit+transform train ...", flush=True)
    X_train_bow = bow_vec.fit_transform(X_train_texts)
    print("  [5/9] BoW transform val ...", flush=True)
    X_val_bow = bow_vec.transform(X_val_texts)
    print("  [6/9] BoW transform test ...", flush=True)
    X_test_bow = bow_vec.transform(X_test_texts)

    print("  [7/9] TF-IDF fit+transform train ...", flush=True)
    X_train_tfidf = tfidf_vec.fit_transform(X_train_texts)
    print("  [8/9] TF-IDF transform val ...", flush=True)
    X_val_tfidf = tfidf_vec.transform(X_val_texts)
    print("  [9/9] TF-IDF transform test ...", flush=True)
    X_test_tfidf = tfidf_vec.transform(X_test_texts)

    print(f"  Vectorization done in {time.time()-t0:.1f}s. Feature dims: {X_train_oh.shape[1]}", flush=True)
    return {
        "onehot": (X_train_oh, X_val_oh, X_test_oh),
        "bow": (X_train_bow, X_val_bow, X_test_bow),
        "tfidf": (X_train_tfidf, X_val_tfidf, X_test_tfidf),
    }


def phase0(data_dir=None):
    print("\n" + "=" * 60)
    print("PHASE 0: Data Exploration")
    print("=" * 60)
    print_dataset_stats(data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(data_dir)
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Train class balance: {y_train.mean():.3f} (0.5 = perfect)")
    return X_train, X_val, X_test, y_train, y_val, y_test


def phase1(X_train, X_val, y_train, y_val, vec_data):
    """Activation benchmark: Architecture B, TF-IDF, AdamW, 10 epochs each."""
    print("\n" + "=" * 60)
    print("PHASE 1: Activation Benchmark")
    print("=" * 60)

    from activations import BENCHMARK_ACTIVATIONS
    activations_to_test = BENCHMARK_ACTIVATIONS

    X_tr_tf, X_vl_tf, _ = vec_data["tfidf"]
    results = {}

    for act_name in activations_to_test:
        print(f"\n  Testing activation: {act_name}")
        trainer, hist = build_and_train(
            X_tr_tf, y_train, X_vl_tf, y_val,
            arch="B", hidden_dim=256, num_blocks=20,
            activation=act_name, dropout_rate=0.3, normalization="batch",
            optimizer_name="adamw", lr=1e-3, weight_decay=0.01,
            scheduler_type="cosine", max_epochs=10, patience=10,
            batch_size=64,
            checkpoint_path=os.path.join(CHECKPOINT_DIR, f"phase1_{act_name}"),
            verbose=False, log_every=5,
        )
        final_val_loss = hist["val_loss"][-1]
        final_val_acc = hist["val_acc"][-1]
        results[act_name] = {"val_loss": final_val_loss, "val_acc": final_val_acc, "history": hist}
        print(f"    val_loss={final_val_loss:.4f}  val_acc={final_val_acc:.4f}")

    best_act = min(results, key=lambda k: results[k]["val_loss"])
    print(f"\nBest activation: {best_act} (val_loss={results[best_act]['val_loss']:.4f})")

    viz.plot_activation_benchmark(
        list(results.keys()),
        [results[a]["val_loss"] for a in results]
    )

    _save_json(
        {k: {kk: v for kk, v in vv.items() if kk != "history"} for k, vv in results.items()},
        os.path.join(RESULTS_DIR, "phase1_activation_benchmark.json")
    )
    return best_act, results


def phase2(X_train, X_val, y_train, y_val, vec_data, best_activation):
    """Optimizer benchmark: Architecture B, TF-IDF, best activation, 100 epochs each."""
    print("\n" + "=" * 60)
    print("PHASE 2: Optimizer Benchmark")
    print("=" * 60)

    optimizers_to_test = ["adam", "adamw", "radam", "ranger"]
    X_tr_tf, X_vl_tf, _ = vec_data["tfidf"]
    results = {}

    for opt_name in optimizers_to_test:
        print(f"\n  Testing optimizer: {opt_name}")
        trainer, hist = build_and_train(
            X_tr_tf, y_train, X_vl_tf, y_val,
            arch="B", hidden_dim=256, num_blocks=20,
            activation=best_activation, dropout_rate=0.3, normalization="batch",
            optimizer_name=opt_name, lr=1e-3, weight_decay=0.01,
            scheduler_type="cosine", max_epochs=100, patience=50,
            batch_size=64,
            checkpoint_path=os.path.join(CHECKPOINT_DIR, f"phase2_{opt_name}"),
            verbose=False, log_every=25,
        )
        final_val_loss = hist["val_loss"][-1]
        final_val_acc = hist["val_acc"][-1]
        best_val_acc = max(hist["val_acc"])
        results[opt_name] = {
            "val_loss": final_val_loss, "val_acc": final_val_acc,
            "best_val_acc": best_val_acc, "history": hist
        }
        print(f"    val_loss={final_val_loss:.4f}  val_acc={final_val_acc:.4f}  best_val_acc={best_val_acc:.4f}")

    best_opt = min(results, key=lambda k: results[k]["val_loss"])
    print(f"\nBest optimizer: {best_opt} (val_loss={results[best_opt]['val_loss']:.4f})")

    viz.plot_optimizer_benchmark(
        list(results.keys()),
        [results[o]["val_loss"] for o in results],
        [results[o]["val_acc"] for o in results],
    )

    _save_json(
        {k: {kk: v for kk, v in vv.items() if kk != "history"} for k, vv in results.items()},
        os.path.join(RESULTS_DIR, "phase2_optimizer_benchmark.json")
    )
    return best_opt, results


def phase3(X_train, X_val, y_train, y_val, vec_data, y_test, test_vec_data,
           best_activation, best_optimizer):
    """Full 3x3 training matrix: 3 vectorizations x 3 architectures x best optimizer."""
    print("\n" + "=" * 60)
    print("PHASE 3: Full Training Matrix (3 vec x 3 arch)")
    print("=" * 60)

    vec_names = ["tfidf", "bow", "onehot"]
    arch_names = ["B", "A", "C"]
    all_histories = []
    all_labels = []
    all_results = []
    matrix = {}

    for vec_name in vec_names:
        matrix[vec_name] = {}
        X_tr, X_vl, X_te = vec_data[vec_name]
        X_te_final = test_vec_data[vec_name][2]

        for arch in arch_names:
            label = f"{vec_name.upper()}-Arch{arch}"
            print(f"\n  Run: {label}")
            num_blocks = 20 if arch in ("B", "C") else 40
            trainer, hist = build_and_train(
                X_tr, y_train, X_vl, y_val,
                arch=arch, hidden_dim=256, num_blocks=num_blocks,
                activation=best_activation, dropout_rate=0.3, normalization="batch",
                optimizer_name=best_optimizer, lr=1e-3, weight_decay=0.01,
                scheduler_type="cosine", max_epochs=1500, patience=150,
                batch_size=64,
                checkpoint_path=os.path.join(CHECKPOINT_DIR, f"phase3_{vec_name}_{arch}"),
                verbose=True, log_every=100,
            )
            y_proba = trainer.predict_proba(X_te_final)
            y_pred = (y_proba >= 0.5).astype(np.int32)
            met = evaluate(y_test, y_pred, y_proba)
            all_histories.append(hist)
            all_labels.append(label)
            all_results.append((label, y_test, y_proba, met["auc_roc"]))
            matrix[vec_name][arch] = {"metrics": {k: v for k, v in met.items() if k != "confusion_matrix"}, "label": label}
            print(f"  Test metrics for {label}:")
            print_metrics(met, prefix="    ")

    viz.plot_loss_curves_grid(all_histories, all_labels)
    viz.plot_accuracy_curves_grid(all_histories, all_labels)

    histories_by_vec_on_archB = {
        vn: [h for h, l in zip(all_histories, all_labels) if vn.upper() in l and "ArchB" in l][0]
        for vn in vec_names
    }
    viz.plot_vectorization_comparison(histories_by_vec_on_archB, "Arch B")

    histories_by_arch_on_tfidf = {
        arch: [h for h, l in zip(all_histories, all_labels) if "TFIDF" in l and f"Arch{arch}" in l][0]
        for arch in arch_names
    }
    viz.plot_architecture_comparison(histories_by_arch_on_tfidf, "TF-IDF")

    viz.plot_roc_curves(all_results)

    _save_json(
        {vn: {arch: v for arch, v in vv.items()} for vn, vv in matrix.items()},
        os.path.join(RESULTS_DIR, "phase3_matrix.json")
    )
    return matrix, all_histories, all_labels, all_results


def phase4(X_train, X_val, X_test, y_train, y_val, y_test, vec_data,
           best_activation, best_optimizer):
    """Depth sensitivity: vary number of residual blocks in Arch B with TF-IDF."""
    print("\n" + "=" * 60)
    print("PHASE 4: Depth Sensitivity Analysis")
    print("=" * 60)

    depths = [5, 10, 20, 30, 40, 60]
    X_tr, X_vl, X_te = vec_data["tfidf"]
    depth_results = {}

    val_accs = []
    test_accs = []

    for n_blocks in depths:
        print(f"\n  Arch B, num_blocks={n_blocks}")
        trainer, hist = build_and_train(
            X_tr, y_train, X_vl, y_val,
            arch="B", hidden_dim=256, num_blocks=n_blocks,
            activation=best_activation, dropout_rate=0.3, normalization="batch",
            optimizer_name=best_optimizer, lr=1e-3, weight_decay=0.01,
            scheduler_type="cosine", max_epochs=1500, patience=150,
            batch_size=64,
            checkpoint_path=os.path.join(CHECKPOINT_DIR, f"phase4_depth{n_blocks}"),
            verbose=True, log_every=150,
        )
        best_val_acc = max(hist["val_acc"])
        y_proba = trainer.predict_proba(X_te)
        y_pred = (y_proba >= 0.5).astype(np.int32)
        met = evaluate(y_test, y_pred, y_proba)
        val_accs.append(best_val_acc)
        test_accs.append(met["accuracy"])
        depth_results[n_blocks] = {"best_val_acc": best_val_acc, "test_acc": met["accuracy"], "test_auc": met["auc_roc"]}
        print(f"    best_val_acc={best_val_acc:.4f}  test_acc={met['accuracy']:.4f}")

    viz.plot_depth_sensitivity(depths, val_accs, test_accs)
    _save_json(depth_results, os.path.join(RESULTS_DIR, "phase4_depth.json"))
    return depth_results


def phase5(X_train, X_val, X_test, y_train, y_val, y_test, vec_data,
           best_activation, best_optimizer, matrix_results, all_histories, all_labels, all_roc_results):
    """Final evaluation: best configuration on held-out test set. Gradient flow, t-SNE."""
    print("\n" + "=" * 60)
    print("PHASE 5: Final Evaluation")
    print("=" * 60)

    best_config = None
    best_acc = 0.0
    for vec_name, arch_dict in matrix_results.items():
        for arch, res in arch_dict.items():
            acc = res["metrics"]["accuracy"]
            if acc > best_acc:
                best_acc = acc
                best_config = (vec_name, arch, res)

    vec_name, arch, best_res = best_config
    best_label = best_res["label"]
    print(f"\nBest configuration: {best_label}")
    print(f"  Test accuracy: {best_acc:.4f}")

    X_tr, X_vl, X_te = vec_data[vec_name]
    num_blocks = 20 if arch in ("B", "C") else 40

    print("\nRetraining best configuration for gradient flow capture ...")
    trainer_A, hist_A = build_and_train(
        vec_data["tfidf"][0], y_train, vec_data["tfidf"][1], y_val,
        arch="A", hidden_dim=256, num_blocks=40,
        activation=best_activation, dropout_rate=0.3, normalization="batch",
        optimizer_name=best_optimizer, lr=1e-3, weight_decay=0.01,
        scheduler_type="cosine", max_epochs=1500, patience=150,
        batch_size=64,
        checkpoint_path=os.path.join(CHECKPOINT_DIR, "phase5_archA_gradflow"),
        verbose=False, log_every=500,
    )
    trainer_B, hist_B = build_and_train(
        vec_data["tfidf"][0], y_train, vec_data["tfidf"][1], y_val,
        arch="B", hidden_dim=256, num_blocks=20,
        activation=best_activation, dropout_rate=0.3, normalization="batch",
        optimizer_name=best_optimizer, lr=1e-3, weight_decay=0.01,
        scheduler_type="cosine", max_epochs=1500, patience=150,
        batch_size=64,
        checkpoint_path=os.path.join(CHECKPOINT_DIR, "phase5_archB_gradflow"),
        verbose=False, log_every=500,
    )

    recorded_epochs = [ep for ep in [1, 100, 500, 1000] if ep in hist_A.get("grad_norms", {})]
    viz.plot_gradient_flow(
        hist_A.get("grad_norms", {}),
        hist_B.get("grad_norms", {}),
        recorded_epochs if recorded_epochs else [1],
    )

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"phase3_{vec_name}_{arch}.npz")
    final_model = build_model(
        input_dim=X_tr.shape[1], hidden_dim=256, num_blocks=num_blocks,
        activation=best_activation, dropout_rate=0.3, use_residual=True,
        normalization="batch", arch=arch,
    )
    if _GPU_AVAILABLE and cp is not None:
        final_model.to_gpu()
    if os.path.exists(checkpoint_path):
        final_model.load(checkpoint_path)

    def penultimate_fn(X):
        if _GPU_AVAILABLE and cp is not None:
            X_gpu = cp.asarray(X)
        else:
            X_gpu = X
        layers = final_model._layers
        out = X_gpu
        for layer in layers[:-1]:
            out = layer(out, training=False)
        if _GPU_AVAILABLE and cp is not None and isinstance(out, cp.ndarray):
            return cp.asnumpy(out)
        return out

    viz.plot_tsne(X_tr, y_train, X_te, y_test, penultimate_fn)

    y_proba = trainer_B.predict_proba(X_te)
    y_pred = (y_proba >= 0.5).astype(np.int32)
    final_metrics = evaluate(y_test, y_pred, y_proba)

    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS (Best Configuration)")
    print("=" * 60)
    print_metrics(final_metrics, prefix="  ")

    viz.plot_confusion_matrix(final_metrics["confusion_matrix"])

    target = 0.97
    acc = final_metrics["accuracy"]
    if acc < 0.95:
        print(f"\nWARNING: test accuracy {acc:.4f} < 0.95. Revisit hyperparameters.")
    elif acc >= target:
        print(f"\nTarget achieved: test accuracy {acc:.4f} >= {target:.2f}")
    else:
        print(f"\nTest accuracy {acc:.4f} (target={target:.2f}, close)")

    _save_json(
        {k: v.tolist() if hasattr(v, "tolist") else v for k, v in final_metrics.items()},
        os.path.join(RESULTS_DIR, "phase5_final_metrics.json")
    )
    return final_metrics, best_label


def generate_results_md(activation_results, optimizer_results, matrix_results,
                        depth_results, final_metrics, best_config_label,
                        best_activation, best_optimizer):
    lines = []
    lines.append("# Results Summary")
    lines.append("")
    lines.append("Auto-generated by `main.py`.")
    lines.append("")

    lines.append("## Best Configuration")
    lines.append(f"- **Config**: {best_config_label}")
    lines.append(f"- **Activation**: {best_activation}")
    lines.append(f"- **Optimizer**: {best_optimizer}")
    lines.append(f"- **Test Accuracy**: {final_metrics['accuracy']:.4f}")
    lines.append(f"- **Test AUC-ROC**: {final_metrics['auc_roc']:.4f}")
    lines.append(f"- **Test F1**: {final_metrics['f1']:.4f}")
    lines.append("")

    lines.append("## Activation Benchmark (10 epochs, Arch B, TF-IDF, AdamW)")
    lines.append("")
    lines.append("| Activation | Val Loss | Val Acc |")
    lines.append("|------------|----------|---------|")
    for act, res in activation_results.items():
        marker = " **best**" if act == best_activation else ""
        lines.append(f"| {act}{marker} | {res['val_loss']:.4f} | {res['val_acc']:.4f} |")
    lines.append("")

    lines.append("## Optimizer Benchmark (100 epochs, Arch B, TF-IDF)")
    lines.append("")
    lines.append("| Optimizer | Val Loss | Val Acc | Best Val Acc |")
    lines.append("|-----------|----------|---------|--------------|")
    for opt, res in optimizer_results.items():
        marker = " **best**" if opt == best_optimizer else ""
        lines.append(f"| {opt}{marker} | {res['val_loss']:.4f} | {res['val_acc']:.4f} | {res['best_val_acc']:.4f} |")
    lines.append("")

    lines.append("## Phase 3: Full Training Matrix")
    lines.append("")
    lines.append("| Config | Accuracy | Precision | Recall | F1 | AUC-ROC |")
    lines.append("|--------|----------|-----------|--------|----|---------|")
    for vec_name, arch_dict in matrix_results.items():
        for arch, res in arch_dict.items():
            m = res["metrics"]
            label = res["label"]
            lines.append(
                f"| {label} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
                f"{m['recall']:.4f} | {m['f1']:.4f} | {m['auc_roc']:.4f} |"
            )
    lines.append("")

    lines.append("## Phase 4: Depth Sensitivity (Arch B, TF-IDF)")
    lines.append("")
    lines.append("| Num Blocks | Best Val Acc | Test Acc | Test AUC |")
    lines.append("|------------|-------------|----------|----------|")
    for n_blocks, res in depth_results.items():
        lines.append(f"| {n_blocks} | {res['best_val_acc']:.4f} | {res['test_acc']:.4f} | {res['test_auc']:.4f} |")
    lines.append("")

    lines.append("## Phase 5: Final Evaluation")
    lines.append("")
    cm = final_metrics["confusion_matrix"]
    lines.append(f"- Accuracy:  {final_metrics['accuracy']:.4f}")
    lines.append(f"- Precision: {final_metrics['precision']:.4f}")
    lines.append(f"- Recall:    {final_metrics['recall']:.4f}")
    lines.append(f"- F1:        {final_metrics['f1']:.4f}")
    lines.append(f"- AUC-ROC:   {final_metrics['auc_roc']:.4f}")
    lines.append(f"- Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    lines.append("")

    with open("results.md", "w") as f:
        f.write("\n".join(lines))
    print("Generated results.md")


def main():
    parser = argparse.ArgumentParser(description="PHP Webshell Detection DNN")
    parser.add_argument("--data-dir", default=None, help="Path to dataset directory")
    parser.add_argument("--phases", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5],
                        help="Which phases to run (default: all)")
    parser.add_argument("--tokenizer", default="whitespace", choices=["bpe", "whitespace"])
    args = parser.parse_args()

    _ensure_dirs()
    np.random.seed(42)

    print(f"GPU available: {_GPU_AVAILABLE}")
    if _GPU_AVAILABLE and cp is not None:
        print(f"GPU device: {cp.cuda.Device().attributes}")

    if 0 in args.phases:
        X_train, X_val, X_test, y_train, y_val, y_test = phase0(args.data_dir)
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(args.data_dir)

    vec_data = _fit_vectorizers(X_train, X_val, X_test, tokenizer_type=args.tokenizer)
    test_vec_data = vec_data

    best_activation = "gelu"
    best_optimizer = "adamw"
    activation_results = {}
    optimizer_results = {}

    if 1 in args.phases:
        best_activation, activation_results = phase1(X_train, X_val, y_train, y_val, vec_data)
    if 2 in args.phases:
        best_optimizer, optimizer_results = phase2(X_train, X_val, y_train, y_val, vec_data, best_activation)

    matrix_results = {}
    all_histories = []
    all_labels = []
    all_roc = []

    if 3 in args.phases:
        matrix_results, all_histories, all_labels, all_roc = phase3(
            X_train, X_val, y_train, y_val, vec_data, y_test, test_vec_data,
            best_activation, best_optimizer
        )

    depth_results = {}
    if 4 in args.phases:
        depth_results = phase4(
            X_train, X_val, X_test, y_train, y_val, y_test, vec_data,
            best_activation, best_optimizer
        )

    final_metrics = None
    best_config_label = "unknown"
    if 5 in args.phases:
        final_metrics, best_config_label = phase5(
            X_train, X_val, X_test, y_train, y_val, y_test, vec_data,
            best_activation, best_optimizer, matrix_results, all_histories, all_labels, all_roc
        )

    if final_metrics is not None and matrix_results and depth_results:
        generate_results_md(
            activation_results, optimizer_results, matrix_results,
            depth_results, final_metrics, best_config_label,
            best_activation, best_optimizer
        )

    print("\nAll phases complete. Results in ./results/")


if __name__ == "__main__":
    main()
