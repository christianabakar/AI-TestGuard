import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.preprocessing import (
    FEATURE_COLUMNS,
    SCALED_COLUMNS,
    TARGET_COLUMN,
    validate_data,
)

RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Charge le CSV, vérifie le contrat, puis retire les doublons exacts (T08)."""
    df = validate_data(pd.read_csv(path))
    return df.drop_duplicates().reset_index(drop=True)


def split_data(df: pd.DataFrame):
    """Découpage stratifié 60 / 20 / 20 (T09). Renvoie 6 objets."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # 1er découpage : 60 % entraînement, 40 % mis de côté
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    # 2e découpage : les 40 % sont coupés en deux -> 20 % validation, 20 % test
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline(model_name: str) -> Pipeline:
    """Assemble prétraitement + modèle dans une seule chaîne."""
    scaler = ColumnTransformer(
        [("scale", StandardScaler(), SCALED_COLUMNS)],
        remainder="passthrough",  # V1 à V28 passent sans modification
    )
    models = {
        "logistic": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    }
    if model_name not in models:
        raise ValueError(f"Modèle inconnu : {model_name}. Choix : {list(models)}")
    return Pipeline([("preprocessing", scaler), ("model", models[model_name])])


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_name: str) -> Pipeline:
    """Entraîne la chaîne complète, sur l'entraînement uniquement."""
    pipeline = build_pipeline(model_name)
    pipeline.fit(X_train, y_train)
    return pipeline


def select_threshold(y_val, proba_val, min_recall: float = 0.80,
                     max_fpr: float = 0.005) -> float:
    """T11 : seuil le plus haut atteignant le rappel minimum, sur la VALIDATION.

    Lève ValueError si ce seuil dépasse le taux de fausses alertes autorisé (ANO-02).
    """
    _precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
    valid = np.where(recall[:-1] >= min_recall)[0]
    seuil = float(thresholds[valid[-1]])

    y = np.asarray(y_val)
    alertes = np.asarray(proba_val) >= seuil
    fpr = (alertes & (y == 0)).sum() / (y == 0).sum()
    if fpr > max_fpr:
        raise ValueError(
            f"Aucun seuil acceptable : rappel >= {min_recall} impose "
            f"un FPR de {fpr:.2%} (maximum autorisé : {max_fpr:.2%})"
        )
    return seuil