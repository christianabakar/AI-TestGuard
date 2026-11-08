import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Contrat de données : les colonnes attendues, dans l'ordre
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


class SchemaError(ValueError):
    """Erreur levée quand les données ne respectent pas le contrat."""

def validate_data(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Vérifie que df respecte le contrat. Lève SchemaError sinon."""

    # Contrôle 1 : le tableau ne doit pas être vide
    if df.empty:
        raise SchemaError("Jeu de données vide")

    # Contrôle 2 : aucune colonne ne doit apparaître deux fois
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise SchemaError(f"Colonnes dupliquées : {duplicated}")

    # Contrôle 3 : toutes les colonnes attendues doivent être présentes
    expected = FEATURE_COLUMNS + ([TARGET_COLUMN] if require_target else [])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SchemaError(f"Colonnes manquantes : {missing}")

    # Contrôle 4 : toutes les variables doivent être numériques
    non_numeric = [c for c in FEATURE_COLUMNS
                   if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise SchemaError(f"Type non numérique pour : {non_numeric}")

    # Contrôle 5 : aucune valeur manquante (NaN) ni infinie (inf)
    values = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad_cols = [c for c in FEATURE_COLUMNS
                    if not np.isfinite(df[c].to_numpy(dtype=float)).all()]
        raise SchemaError(f"Valeurs NaN ou infinies dans : {bad_cols}")

    # Contrôle 6 : un montant ne peut pas être négatif
    if (df["Amount"] < -1).any():
        raise SchemaError("Amount négatif détecté (domaine attendu : Amount >= 0)")

    
    # Contrôle 7 : la cible ne peut valoir que 0 (normale) ou 1 (fraude)
    if require_target:
        invalid = set(df[TARGET_COLUMN].dropna().unique()) - {0, 1}
        if invalid or df[TARGET_COLUMN].isna().any():
            raise SchemaError(f"Valeurs de Class invalides : {sorted(int(v) for v in invalid)}")

    # Colonnes en trop : on ne bloque pas, mais on prévient et on les écarte
    extra = [c for c in df.columns if c not in expected]
    if extra:
        warnings.warn(f"Colonnes ignorées : {extra}", UserWarning)

    return df[expected].copy()


SCALED_COLUMNS = ["Time", "Amount"]


def fit_preprocessor(X_train: pd.DataFrame) -> StandardScaler:
    """Apprend les paramètres de mise à l'échelle, UNIQUEMENT sur l'entraînement."""
    scaler = StandardScaler()
    scaler.fit(X_train[SCALED_COLUMNS])
    return scaler


def preprocess_data(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Applique une mise à l'échelle DÉJÀ apprise. Ne réapprend jamais."""
    result = X[FEATURE_COLUMNS].copy()
    result[SCALED_COLUMNS] = scaler.transform(X[SCALED_COLUMNS])

    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise SchemaError("Le prétraitement a produit des valeurs non finies")

    return result