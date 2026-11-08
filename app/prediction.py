from pathlib import Path

import joblib
import pandas as pd
import sklearn

from app.preprocessing import validate_data


def predict_fraud(model, X: pd.DataFrame, threshold: float):
    """Renvoie (probabilités, classes). Règle : fraude si probabilité >= seuil."""
    X_valide = validate_data(X, require_target=False)
    proba = model.predict_proba(X_valide)[:, 1]
    classes = (proba >= threshold).astype(int)
    return proba, classes


def save_pipeline(model, threshold: float, path: str) -> None:
    """Sauvegarde le pipeline AVEC son seuil et la version de scikit-learn."""
    paquet = {
        "pipeline": model,
        "threshold": float(threshold),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(paquet, path)


def load_pipeline(path: str):
    """Recharge (pipeline, seuil). Lève une erreur explicite en cas de problème."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")

    try:
        paquet = joblib.load(path)
    except Exception as erreur:
        raise ValueError(f"Fichier modèle illisible ou corrompu : {path}") from erreur

    if not isinstance(paquet, dict) or not {"pipeline", "threshold"} <= paquet.keys():
        raise ValueError(f"Format de modèle invalide : {path}")

    if paquet.get("sklearn_version") != sklearn.__version__:
        raise ValueError(
            f"Version incompatible : modèle créé avec scikit-learn "
            f"{paquet.get('sklearn_version')}, version installée {sklearn.__version__}"
        )

    return paquet["pipeline"], paquet["threshold"]