import numpy as np
import pandas as pd

from app.prediction import predict_fraud
from app.training import load_data, select_threshold, split_data, train_model

df = load_data("data/creditcard.csv")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
modele = train_model(X_train, y_train, "logistic")
seuil = select_threshold(y_val, modele.predict_proba(X_val)[:, 1])
_, classes = predict_fraud(modele, X_test, seuil)

segments = pd.cut(X_test["Amount"], [-0.01, 10, 100, 500, np.inf],
                  labels=["0-10 €", "10-100 €", "100-500 €", "> 500 €"])
rng = np.random.default_rng(42)


def ic_rappel(y, c, n_iter=1000):
    """IC 95 % du rappel par bootstrap (rééchantillonnage des fraudes du segment)."""
    fraudes = c[y == 1]
    if len(fraudes) == 0:
        return (np.nan, np.nan)
    tirages = [rng.choice(fraudes, len(fraudes)).mean() for _ in range(n_iter)]
    return tuple(np.percentile(tirages, [2.5, 97.5]))


lignes = []
for seg in segments.cat.categories:
    masque = (segments == seg).to_numpy()
    y, c = y_test.to_numpy()[masque], classes[masque]
    tp, fn = int(((c == 1) & (y == 1)).sum()), int(((c == 0) & (y == 1)).sum())
    fp, tn = int(((c == 1) & (y == 0)).sum()), int(((c == 0) & (y == 0)).sum())
    bas, haut = ic_rappel(y, c)
    lignes.append({
        "segment": seg, "transactions": int(masque.sum()), "fraudes": tp + fn,
        "rappel": round(tp / (tp + fn), 3) if tp + fn else np.nan,
        "IC95_rappel": f"[{bas:.2f} ; {haut:.2f}]",
        "precision": round(tp / (tp + fp), 3) if tp + fp else np.nan,
        "FPR_%": round(100 * fp / (fp + tn), 3), "FP": fp,
    })

resultats = pd.DataFrame(lignes)
resultats.to_csv("reports/analyse_segments.csv", index=False)
pd.set_option("display.width", 200)
print(resultats.to_string(index=False))