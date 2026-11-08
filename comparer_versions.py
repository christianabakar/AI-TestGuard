import json

from app.evaluation import compare_to_baseline, evaluate_model
from app.prediction import predict_fraud
from app.training import build_pipeline, load_data, select_threshold, split_data

with open("reports/baseline.json", encoding="utf-8") as f:
    baseline = json.load(f)
ref = baseline["metriques_validation"]

df = load_data("data/creditcard.csv")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
# Le jeu de test n'est pas utilisé : les versions se comparent sur la validation fixe.

CANDIDATS = {
    "v2a : logistique, régularisation plus forte (C = 0,1)":
        build_pipeline("logistic").set_params(model__C=0.1),
    "v2b : logistique, classes pondérées (class_weight = balanced)":
        build_pipeline("logistic").set_params(model__class_weight="balanced"),
    "v2c : arbre de décision (max_depth = 6)":
        build_pipeline("tree"),
}

print(f"BASELINE (commit {baseline['version_code']}) : rappel {ref['recall']:.3f} | "
      f"précision {ref['precision']:.3f} | FPR {100 * ref['fpr']:.3f} % | "
      f"PR-AUC {ref['pr_auc']:.4f}")

rapport = {"baseline": baseline["version_code"], "candidats": {}}
for nom, modele in CANDIDATS.items():
    modele.fit(X_train, y_train)
    proba_val = modele.predict_proba(X_val)[:, 1]
    try:
        seuil = select_threshold(y_val, proba_val)
    except ValueError as erreur:
        print(f"\n{nom}\n  -> REJETER : {erreur}")
        rapport["candidats"][nom] = {"decision": "REJETER", "motifs": [str(erreur)]}
        continue

    proba, classes = predict_fraud(modele, X_val, seuil)
    m = evaluate_model(y_val, classes, proba)
    decision, motifs = compare_to_baseline(ref, m)

    print(f"\n{nom}")
    print(f"  seuil {seuil:.5f} | rappel {m['recall']:.3f} ({m['recall'] - ref['recall']:+.3f}) | "
          f"précision {m['precision']:.3f} ({m['precision'] - ref['precision']:+.3f}) | "
          f"FPR {100 * m['fpr']:.3f} % | PR-AUC {m['pr_auc']:.4f} "
          f"({m['pr_auc'] - ref['pr_auc']:+.4f}) | FP {m['FP']}")
    print(f"  -> {decision}" + (f" : {motifs}" if motifs else ""))
    rapport["candidats"][nom] = {"decision": decision, "motifs": motifs, "seuil": seuil,
                                 "metriques_validation": {k: float(v) for k, v in m.items()}}

with open("reports/rapport_non_regression.json", "w", encoding="utf-8") as f:
    json.dump(rapport, f, indent=2, ensure_ascii=False)
print("\nRapport écrit dans reports/rapport_non_regression.json")