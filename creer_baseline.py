import hashlib
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from app.evaluation import evaluate_model
from app.prediction import predict_fraud
from app.training import RANDOM_STATE, load_data, select_threshold, split_data, train_model

CHEMIN_BASELINE = Path("reports/baseline.json")
CHEMIN_DONNEES = "data/creditcard.csv"

# Garde-fou : la baseline ne se régénère JAMAIS automatiquement
if CHEMIN_BASELINE.exists() and "--force" not in sys.argv:
    sys.exit(f"ARRÊT : {CHEMIN_BASELINE} existe déjà. Une baseline ne se remplace que "
             "par une décision explicite (python creer_baseline.py --force).")

df = load_data(CHEMIN_DONNEES)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
modele = train_model(X_train, y_train, "logistic")
seuil = select_threshold(y_val, modele.predict_proba(X_val)[:, 1])
proba, classes = predict_fraud(modele, X_val, seuil)
metriques = evaluate_model(y_val, classes, proba)

try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
except (subprocess.CalledProcessError, FileNotFoundError):
    commit = "inconnu"

baseline = {
    "date_creation": date.today().isoformat(),
    "version_code": commit,
    "modele": "logistic",
    "parametres_modele": {k: str(v) for k, v in
                          modele.named_steps["model"].get_params().items()},
    "versions_bibliotheques": {"python": platform.python_version(),
                               "scikit-learn": sklearn.__version__,
                               "pandas": pd.__version__, "numpy": np.__version__},
    "graine_aleatoire": RANDOM_STATE,
    "seuil": seuil,
    "jeu_evaluation": "validation (fixe, graine 42)",
    "metriques_validation": {k: float(v) for k, v in metriques.items()},
    "donnees": {
        "sha256": hashlib.sha256(Path(CHEMIN_DONNEES).read_bytes()).hexdigest(),
        "lignes_apres_dedoublonnage": len(df),
        "fraudes_apres_dedoublonnage": int(df["Class"].sum()),
        "taille_validation": len(X_val),
        "fraudes_validation": int(y_val.sum()),
    },
}

CHEMIN_BASELINE.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(baseline, indent=2, ensure_ascii=False))