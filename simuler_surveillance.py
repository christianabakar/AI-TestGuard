import time

import numpy as np
import pandas as pd

from app.monitoring import out_of_range, psi, training_ranges
from app.prediction import predict_fraud
from app.training import load_data, select_threshold, split_data, train_model

SEUIL_PSI, DELAI_LABELS = 0.25, 2  # dérive si PSI > 0,25 ; labels connus à J+2
rng = np.random.default_rng(42)

df = load_data("data/creditcard.csv")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
modele = train_model(X_train, y_train, "logistic")
seuil = select_threshold(y_val, modele.predict_proba(X_val)[:, 1])
plages = training_ranges(X_train)
_, c_val = predict_fraud(modele, X_val, seuil)
taux_alertes_ref = c_val.mean()

# --- 7 « jours » de production simulés à partir du jeu de test ---
jours = np.array_split(rng.permutation(len(X_test)), 7)
lots = []
for j, idx in enumerate(jours, start=1):
    X, y = X_test.iloc[idx].copy(), y_test.iloc[idx].copy()
    if j >= 4:   # DÉRIVE DES DONNÉES : hausse générale des montants (x 3)
        X["Amount"] *= 3
    if j >= 6:   # DÉRIVE DE CONCEPT : les fraudeurs imitent les transactions normales
        fraudes = y == 1
        X.loc[fraudes, [f"V{i}" for i in range(1, 29)]] *= 0.2
    lots.append((j, X, y))

journal, alertes = [], []
for j, X, y in lots:
    debut = time.perf_counter()
    _, classes = predict_fraud(modele, X, seuil)
    duree_ms = 1000 * (time.perf_counter() - debut)

    ligne = {
        "jour": f"J{j}", "transactions": len(X),
        "taux_alertes_%": round(100 * classes.mean(), 3),
        "psi_amount": round(psi(X_train["Amount"], X["Amount"]), 3),
        "hors_plage": int(out_of_range(X, plages).sum()),
        "duree_lot_ms": round(duree_ms, 1),
    }
    # Métriques IMMÉDIATES : pas besoin des labels
    if ligne["psi_amount"] > SEUIL_PSI:
        alertes.append(f"J{j} [immédiat] DÉRIVE DES DONNÉES : PSI Amount = {ligne['psi_amount']}")
    if classes.mean() > 2 * taux_alertes_ref:
        alertes.append(f"J{j} [immédiat] TAUX D'ALERTES x{classes.mean() / taux_alertes_ref:.1f} "
                       f"par rapport à la référence")
    journal.append((ligne, y.to_numpy(), classes))

# Métriques DIFFÉRÉES : labels (rétrofacturations) connus DELAI_LABELS jours plus tard
lignes = []
for k, (ligne, y, c) in enumerate(journal):
    j = k + 1
    tp, fn = int(((c == 1) & (y == 1)).sum()), int(((c == 0) & (y == 1)).sum())
    fp, tn = int(((c == 1) & (y == 0)).sum()), int(((c == 0) & (y == 0)).sum())
    ligne["fraudes"] = tp + fn
    ligne["rappel"] = round(tp / (tp + fn), 3) if tp + fn else np.nan
    ligne["fpr_%"] = round(100 * fp / (fp + tn), 3)
    ligne["labels_connus_le"] = f"J{j + DELAI_LABELS}"
    if j + DELAI_LABELS <= 9:
        if ligne["rappel"] < 0.80:
            alertes.append(f"J{j + DELAI_LABELS} [différé, labels de J{j}] RAPPEL {ligne['rappel']} < 0,80")
        if ligne["fpr_%"] > 0.5:
            alertes.append(f"J{j + DELAI_LABELS} [différé, labels de J{j}] FPR {ligne['fpr_%']} % > 0,5 %")
    lignes.append(ligne)

resultats = pd.DataFrame(lignes)
resultats.to_csv("reports/surveillance_simulee.csv", index=False)
pd.set_option("display.width", 220)
print("Scénario : J1-J3 normal | J4+ montants x3 (dérive des données) | "
      "J6+ fraudeurs imitant les clients (dérive de concept)\n")
print(resultats.to_string(index=False))
print(f"\nTaux d'alertes de référence (validation) : {100 * taux_alertes_ref:.3f} %\n")
print("ALERTES DÉCLENCHÉES (triées par jour de détection) :")
for a in sorted(alertes, key=lambda s: int(s.split()[0][1:])):
    print("  " + a)