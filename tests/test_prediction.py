import numpy as np
import pandas as pd
import pytest

from app.prediction import predict_fraud
from app.preprocessing import FEATURE_COLUMNS, SchemaError
from app.training import select_threshold


class FauxModele:
    """Stub : renvoie des probabilités choisies à l'avance."""

    def __init__(self, probas):
        self.probas = np.asarray(probas, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.probas, self.probas])


@pytest.fixture
def transactions():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(4, 30)), columns=FEATURE_COLUMNS)
    df["Amount"] = [10.0, 20.0, 30.0, 40.0]
    return df


# ---------- predict_fraud ----------

def test_probabilites_valides(transactions):
    """T04 : une probabilité par transaction, finie, comprise entre 0 et 1."""
    proba, classes = predict_fraud(FauxModele([0.0, 0.2, 0.7, 1.0]), transactions, 0.5)

    assert proba.shape == (4,)
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()
    assert set(classes) <= {0, 1}


def test_probabilite_egale_au_seuil_est_une_fraude(transactions):
    """Non-régression ANO-01 : règle « fraude si probabilité >= seuil »."""
    _, classes = predict_fraud(FauxModele([0.49, 0.5, 0.51, 0.1]), transactions, 0.5)

    assert list(classes) == [0, 1, 1, 0]


def test_transaction_invalide_refusee_avant_le_modele(transactions):
    """Une transaction invalide n'atteint jamais le modèle."""
    transactions.loc[0, "Amount"] = np.nan

    with pytest.raises(SchemaError):
        predict_fraud(FauxModele([0.1] * 4), transactions, 0.5)


# ---------- select_threshold ----------

def test_seuil_atteint_le_rappel_minimum():
    """T11 : le seuil retenu détecte au moins 80 % des fraudes."""
    y = np.array([0] * 90 + [1] * 10)
    proba = np.concatenate([np.linspace(0, 0.3, 90), np.linspace(0.2, 0.9, 10)])

    seuil = select_threshold(y, proba, min_recall=0.80, max_fpr=0.5)
    rappel = ((proba >= seuil) & (y == 1)).sum() / 10

    assert rappel >= 0.80


def test_seuil_impossible_leve_une_erreur():
    """Non-régression ANO-02 : un seuil inacceptable est refusé, jamais renvoyé."""
    y = np.array([0] * 90 + [1] * 10)
    proba = np.full(100, 0.5)  # le modèle ne distingue rien : il faut tout signaler

    with pytest.raises(ValueError, match="Aucun seuil acceptable"):
        select_threshold(y, proba, min_recall=0.80, max_fpr=0.005)