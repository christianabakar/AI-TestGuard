import json

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    recall_score,
)

from app.evaluation import bootstrap_ci, evaluate_model
from app.prediction import predict_fraud
from app.training import load_data, select_threshold, split_data, train_model

df = load_data("data/creditcard.csv")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

resultats = {}
fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
fig_pr, ax_pr = plt.subplots(figsize=(6, 5))

for nom in ["logistic", "tree"]:
    modele = train_model(X_train, y_train, nom)

    # 1. Le seuil est choisi sur la VALIDATION (le test n'intervient pas)
    try:
        seuil = select_threshold(y_val, modele.predict_proba(X_val)[:, 1])
        statut = "seuil métier (rappel >= 0,80 sur validation)"
    except ValueError as erreur:
        seuil = 0.5
        statut = f"REJETÉ ({erreur}) - évalué à 0,5 pour comparaison uniquement"

    # 2. Évaluation UNIQUE sur le TEST
    proba, classes = predict_fraud(modele, X_test, seuil)
    m = evaluate_model(y_test, classes, proba)
    m["recall_ic95"] = bootstrap_ci(y_test, classes, recall_score)
    m["seuil"], m["statut"] = seuil, statut
    m["pr_auc_train"] = evaluate_model(
        y_train, modele.predict(X_train), modele.predict_proba(X_train)[:, 1]
    )["pr_auc"]
    resultats[nom] = m

    # 3. Graphiques
    RocCurveDisplay.from_predictions(y_test, proba, name=nom, ax=ax_roc)
    PrecisionRecallDisplay.from_predictions(y_test, proba, name=nom, ax=ax_pr)
    cm = ConfusionMatrixDisplay.from_predictions(
        y_test, classes, display_labels=["Normale", "Fraude"], cmap="Blues"
    )
    cm.ax_.set_title(f"Matrice de confusion - {nom} (seuil = {seuil:.5f})")
    cm.figure_.savefig(f"reports/matrice_confusion_{nom}.png", dpi=150, bbox_inches="tight")

    print(f"\n===== {nom} : {statut}")
    print(f"Seuil {seuil:.5f} | TP {m['TP']}  FN {m['FN']}  FP {m['FP']}  TN {m['TN']}")
    print(f"Rappel {m['recall']:.3f}  IC95 [{m['recall_ic95'][0]:.3f} ; {m['recall_ic95'][1]:.3f}]")
    print(f"Précision {m['precision']:.3f} | F1 {m['f1']:.3f} | FPR {100 * m['fpr']:.3f} % "
          f"| FNR {100 * m['fnr']:.1f} %")
    print(f"ROC-AUC {m['roc_auc']:.4f} | PR-AUC test {m['pr_auc']:.4f} "
          f"| PR-AUC train {m['pr_auc_train']:.4f} | Accuracy {m['accuracy']:.5f}")

ax_roc.plot([0, 1], [0, 1], "k--", label="Hasard")
ax_roc.set_title("Courbes ROC - jeu de test")
ax_roc.legend()
fig_roc.savefig("reports/courbes_roc.png", dpi=150, bbox_inches="tight")

ax_pr.axhline(y_test.mean(), color="k", linestyle="--", label="Hasard (prévalence)")
ax_pr.set_title("Courbes Précision-Rappel - jeu de test")
ax_pr.legend()
fig_pr.savefig("reports/courbes_precision_rappel.png", dpi=150, bbox_inches="tight")

with open("reports/metriques_test.json", "w", encoding="utf-8") as f:
    json.dump(resultats, f, indent=2, ensure_ascii=False, default=float)

print("\nFichiers écrits dans reports/")