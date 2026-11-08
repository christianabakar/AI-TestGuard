import numpy as np
import pandas as pd

from app.evaluation import evaluate_model
from app.monitoring import out_of_range, psi, training_ranges
from app.prediction import predict_fraud
from app.preprocessing import SchemaError
from app.training import load_data, select_threshold, split_data, train_model

rng = np.random.default_rng(42)
V_COLS = [f"V{i}" for i in range(1, 29)]

df = load_data("data/creditcard.csv")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
modele = train_model(X_train, y_train, "logistic")
seuil = select_threshold(y_val, modele.predict_proba(X_val)[:, 1])
plages = training_ranges(X_train)
ecarts_types = X_train[V_COLS].std()

p_ref, c_ref = predict_fraud(modele, X_test, seuil)
m_ref = evaluate_model(y_test, c_ref, p_ref)


def bruit_v(k):
    """Bruit gaussien sur V1-V28, proportionnel à l'écart-type de chaque colonne."""
    X = X_test.copy()
    X[V_COLS] += rng.normal(size=X[V_COLS].shape) * (k * ecarts_types.to_numpy())
    return X


def montant_relatif(r):
    """Montant multiplié par (1 + u), u uniforme dans [-r, r] (arrondis, conversions)."""
    X = X_test.copy()
    X["Amount"] = X["Amount"] * (1 + rng.uniform(-r, r, len(X)))
    return X


def decalage_time(secondes):
    """Horloge décalée (fuseau horaire, horloge serveur)."""
    X = X_test.copy()
    X["Time"] = X["Time"] + secondes
    return X


def montant_extreme(facteur):
    """Montant multiplié par un facteur (valeurs extrêmes)."""
    X = X_test.copy()
    X["Amount"] = X["Amount"] * facteur
    return X


scenarios = {
    "Bruit V1-V28 : 1 % écart-type": bruit_v(0.01),
    "Bruit V1-V28 : 5 % écart-type": bruit_v(0.05),
    "Bruit V1-V28 : 10 % écart-type": bruit_v(0.10),
    "Amount ± 5 %": montant_relatif(0.05),
    "Amount ± 20 %": montant_relatif(0.20),
    "Time + 1 heure": decalage_time(3600),
    "Amount x 10 (extrême)": montant_extreme(10),
    "Amount x 100 (extrême)": montant_extreme(100),
}

lignes = []
for nom, X in scenarios.items():
    try:
        p, c = predict_fraud(modele, X, seuil)
    except SchemaError as erreur:
        lignes.append({"scenario": nom, "rejet": str(erreur)})
        continue
    m = evaluate_model(y_test, c, p)
    lignes.append({
        "scenario": nom,
        "delta_proba_moy": float(np.mean(np.abs(p - p_ref))),
        "delta_proba_max": float(np.max(np.abs(p - p_ref))),
        "changements_classe_pct": 100 * float(np.mean(c != c_ref)),
        "rappel": m["recall"],
        "delta_rappel_pts": 100 * (m["recall"] - m_ref["recall"]),
        "precision": m["precision"],
        "fpr_pct": 100 * m["fpr"],
        "hors_plage": int(out_of_range(X, plages).sum()),
        "psi_amount": psi(X_train["Amount"], X["Amount"]),
    })

resultats = pd.DataFrame(lignes).round(4)
resultats.to_csv("reports/robustesse.csv", index=False)

pd.set_option("display.width", 200)
print(f"Référence (test non perturbé) : rappel {m_ref['recall']:.3f} | "
      f"précision {m_ref['precision']:.3f} | FPR {100 * m_ref['fpr']:.3f} % | "
      f"hors plage {int(out_of_range(X_test, plages).sum())} | "
      f"PSI Amount {psi(X_train['Amount'], X_test['Amount']):.4f}\n")
print(resultats[["scenario", "delta_proba_moy", "changements_classe_pct", "rappel",
                 "delta_rappel_pts", "precision", "fpr_pct", "hors_plage",
                 "psi_amount"]].to_string(index=False))
print("\nRésultats écrits dans reports/robustesse.csv")