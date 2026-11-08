import pandas as pd

from app.preprocessing import validate_data


def predict_fraud(model, X: pd.DataFrame, threshold: float):
    """Renvoie (probabilités, classes). Règle : fraude si probabilité >= seuil."""
    X_valide = validate_data(X, require_target=False)
    proba = model.predict_proba(X_valide)[:, 1]
    classes = (proba >= threshold).astype(int)
    return proba, classes