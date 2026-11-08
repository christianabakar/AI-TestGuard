import numpy as np
import pandas as pd

from app.preprocessing import FEATURE_COLUMNS


def training_ranges(X_train: pd.DataFrame) -> dict:
    """Mémorise, pour chaque variable, la plage [min, max] vue à l'entraînement."""
    return {c: (float(X_train[c].min()), float(X_train[c].max())) for c in FEATURE_COLUMNS}


def out_of_range(X: pd.DataFrame, ranges: dict) -> np.ndarray:
    """Scénario 3 : True pour chaque transaction hors de la plage d'entraînement."""
    hors_plage = np.zeros(len(X), dtype=bool)
    for c, (mini, maxi) in ranges.items():
        hors_plage |= ((X[c] < mini) | (X[c] > maxi)).to_numpy()
    return hors_plage


def psi(reference, actuel, n_bins: int = 10) -> float:
    """Scénario 5 : Population Stability Index, mesure de dérive d'une variable.

    Repères usuels : < 0,10 stable | 0,10 à 0,25 à surveiller | > 0,25 dérive.
    """
    reference, actuel = np.asarray(reference, float), np.asarray(actuel, float)
    bornes = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    bornes[0], bornes[-1] = -np.inf, np.inf

    p_ref = np.histogram(reference, bornes)[0] / len(reference)
    p_act = np.histogram(actuel, bornes)[0] / len(actuel)
    p_ref, p_act = np.clip(p_ref, 1e-6, None), np.clip(p_act, 1e-6, None)

    return float(np.sum((p_act - p_ref) * np.log(p_act / p_ref)))