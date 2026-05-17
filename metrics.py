import numpy as np
from sklearn.metrics import roc_auc_score


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def precision(y_true, y_pred):
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    return tp / (tp + fp + 1e-12)


def recall(y_true, y_pred):
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fn + 1e-12)


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2.0 * p * r / (p + r + 1e-12)


def confusion_matrix(y_true, y_pred):
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def auc_roc(y_true, y_proba):
    return float(roc_auc_score(y_true, y_proba))


def roc_curve(y_true, y_proba, n_thresholds=200):
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    fprs, tprs = [], []
    for thr in thresholds:
        pred = (y_proba >= thr).astype(np.int32)
        tp = float(((pred == 1) & (y_true == 1)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        fn = float(((pred == 0) & (y_true == 1)).sum())
        tn = float(((pred == 0) & (y_true == 0)).sum())
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        tprs.append(tpr)
        fprs.append(fpr)
    return np.array(fprs), np.array(tprs), thresholds


def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc_roc": auc_roc(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def print_metrics(metrics_dict, prefix=""):
    cm = metrics_dict["confusion_matrix"]
    print(f"{prefix}Accuracy:  {metrics_dict['accuracy']:.4f}")
    print(f"{prefix}Precision: {metrics_dict['precision']:.4f}")
    print(f"{prefix}Recall:    {metrics_dict['recall']:.4f}")
    print(f"{prefix}F1:        {metrics_dict['f1']:.4f}")
    print(f"{prefix}AUC-ROC:   {metrics_dict['auc_roc']:.4f}")
    print(f"{prefix}Confusion matrix (TN FP / FN TP):")
    print(f"{prefix}  [[{cm[0,0]:5d}  {cm[0,1]:5d}]")
    print(f"{prefix}   [{cm[1,0]:5d}  {cm[1,1]:5d}]]")
