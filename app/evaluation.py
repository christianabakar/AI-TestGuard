import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(y_true, y_pred, y_proba) -> dict:
    """Calcule toutes les métriques exigées par l'énoncé (section 3.1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def bootstrap_ci(y_true, y_pred, metric, n_iter: int = 1000, seed: int = 42):
    """Intervalle de confiance à 95 % d'une métrique, par rééchantillonnage."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_iter):
        idx = rng.integers(0, len(y_true), len(y_true))  # tirage avec remise
        scores.append(metric(y_true[idx], y_pred[idx]))
    return tuple(np.percentile(scores, [2.5, 97.5]))


# ---------- Non-régression (Partie VI) ----------

CRITERES_ACCEPTATION = {"recall_min": 0.80, "precision_min": 0.10, "fpr_max": 0.005}
TOLERANCES = {"pr_auc": 0.02, "recall": 0.03, "precision": 0.05}


def compare_to_baseline(baseline: dict, nouveau: dict):
    """Compare une nouvelle version à la référence. Renvoie (décision, motifs)."""
    # 1. Critères d'acceptation du plan de test : bloquants
    motifs = []
    if nouveau["recall"] < CRITERES_ACCEPTATION["recall_min"]:
        motifs.append(f"rappel {nouveau['recall']:.3f} < {CRITERES_ACCEPTATION['recall_min']}")
    if nouveau["precision"] < CRITERES_ACCEPTATION["precision_min"]:
        motifs.append(f"précision {nouveau['precision']:.3f} < "
                      f"{CRITERES_ACCEPTATION['precision_min']}")
    if nouveau["fpr"] > CRITERES_ACCEPTATION["fpr_max"]:
        motifs.append(f"FPR {nouveau['fpr']:.4f} > {CRITERES_ACCEPTATION['fpr_max']}")
    if motifs:
        return "REJETER", motifs

    # 2. Tolérances de non-régression par rapport à la baseline
    for metrique, tolerance in TOLERANCES.items():
        delta = nouveau[metrique] - baseline[metrique]
        if delta < -tolerance:
            motifs.append(f"{metrique} : {delta:+.3f} (tolérance -{tolerance})")
    if motifs:
        return "INVESTIGUER", motifs

    return "ACCEPTER", []