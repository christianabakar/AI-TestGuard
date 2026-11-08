import json

import numpy as np

from app.evaluation import evaluate_model
from app.prediction import load_pipeline, predict_fraud, save_pipeline
from app.training import load_data, select_threshold, split_data, train_model

CHEMIN_MODELE = "reports/modele_logistic.joblib"

print("1. Chargement des données")
print("2. Vérification du schéma + dédoublonnage")
df = load_data("data/creditcard.csv")

print("3. Découpage train / validation / test")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

print("4. Ajustement du prétraitement et du modèle sur l'entraînement")
modele = train_model(X_train, y_train, "logistic")

print("5. Choix du seuil sur la validation")
seuil = select_threshold(y_val, modele.predict_proba(X_val)[:, 1])

print("6. Sauvegarde du pipeline")
save_pipeline(modele, seuil, CHEMIN_MODELE)

print("7. Rechargement du pipeline sauvegardé")
modele_recharge, seuil_recharge = load_pipeline(CHEMIN_MODELE)

print("8. Prédictions sur le test (modèle d'origine ET modèle rechargé)")
p_origine, c_origine = predict_fraud(modele, X_test, seuil)
p_recharge, c_recharge = predict_fraud(modele_recharge, X_test, seuil_recharge)

print("9. Rapport de métriques\n")
m = evaluate_model(y_test, c_recharge, p_recharge)

ecart_max = float(np.max(np.abs(p_origine - p_recharge)))
classes_differentes = int((c_origine != c_recharge).sum())

print("===== PREUVE DE REPRODUCTIBILITÉ =====")
print(f"Seuil sauvegardé = seuil rechargé : {seuil == seuil_recharge} ({seuil:.5f})")
print(f"Écart maximal des probabilités    : {ecart_max:.2e}  (tolérance 1e-12)")
print(f"Classes différentes               : {classes_differentes} sur {len(c_origine)}")
print(f"Résultat : {'CONFORME' if ecart_max <= 1e-12 and classes_differentes == 0 else 'NON CONFORME'}")
print(f"\nModèle rechargé -> rappel {m['recall']:.3f} | précision {m['precision']:.3f} "
      f"| FPR {100 * m['fpr']:.3f} %")

with open("reports/seuil.json", "w", encoding="utf-8") as f:
    json.dump({"modele": "logistic", "seuil": seuil,
               "regle": "fraude si probabilité >= seuil",
               "choisi_sur": "validation", "rappel_minimum": 0.80,
               "fpr_maximum": 0.005}, f, indent=2, ensure_ascii=False)