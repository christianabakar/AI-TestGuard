import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.monitoring import out_of_range, psi, training_ranges
from app.prediction import predict_fraud
from app.preprocessing import FEATURE_COLUMNS, SchemaError
from app.training import train_model

V_COLS = [f"V{i}" for i in range(1, 29)]


@pytest.fixture(scope="module")
def donnees():
    """2 000 transactions synthétiques, 5 % de fraudes plus « éloignées »."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(2000, 30)), columns=FEATURE_COLUMNS)
    X["Time"] = rng.uniform(0, 172800, 2000)
    X["Amount"] = rng.exponential(80, 2000)
    y = pd.Series(np.zeros(2000, dtype=int))
    y[:100] = 1
    X.loc[:99, V_COLS[:5]] += 3  # les fraudes se distinguent sur V1-V5
    return X, y


@pytest.fixture(scope="module")
def modele(donnees):
    X, y = donnees
    return train_model(X, y, "logistic")


# ---------- Tests basés sur les propriétés (Hypothesis) ----------

transaction_valide = st.fixed_dictionaries({
    "Time": st.floats(0, 200_000),
    "Amount": st.floats(0, 10_000_000),
    **{c: st.floats(-100, 100) for c in V_COLS},
})


@settings(max_examples=200, deadline=None)
@given(t=transaction_valide)
def test_toute_transaction_valide_donne_une_probabilite_valide(modele, t):
    """Propriété : quelle que soit la transaction valide, la sortie respecte le contrat."""
    X = pd.DataFrame([t])[FEATURE_COLUMNS]
    proba, classe = predict_fraud(modele, X, 0.5)

    assert np.isfinite(proba).all()
    assert 0 <= proba[0] <= 1
    assert classe[0] in (0, 1)


@settings(max_examples=100, deadline=None)
@given(colonne=st.sampled_from(FEATURE_COLUMNS),
       valeur=st.sampled_from([np.nan, np.inf, -np.inf]))
def test_toute_valeur_non_finie_est_rejetee(modele, donnees, colonne, valeur):
    """Propriété : où qu'elle soit, une valeur non finie est rejetée avant le modèle."""
    X = donnees[0].head(3).copy()
    X.loc[1, colonne] = valeur

    with pytest.raises(SchemaError):
        predict_fraud(modele, X, 0.5)


# ---------- Test métamorphique ----------

def test_ordre_des_lignes_sans_effet(modele, donnees):
    """Mélanger les transactions ne change la prédiction d'aucune d'entre elles."""
    X = donnees[0].head(200)
    melange = X.sample(frac=1, random_state=0)

    p_origine, _ = predict_fraud(modele, X, 0.5)
    p_melange, _ = predict_fraud(modele, melange, 0.5)

    np.testing.assert_allclose(pd.Series(p_melange, index=melange.index)[X.index], p_origine)


# ---------- Sensibilité (T17) ----------

def test_sensibilite_bruit_faible(modele, donnees):
    """T17 : un bruit de 1 % de l'écart-type change au plus 1 % des classes."""
    X, _ = donnees
    rng = np.random.default_rng(0)
    bruit = X.copy()
    bruit[V_COLS] += rng.normal(size=(len(X), 28)) * 0.01 * X[V_COLS].std().to_numpy()

    _, c_avant = predict_fraud(modele, X, 0.5)
    _, c_apres = predict_fraud(modele, bruit, 0.5)

    assert np.mean(c_avant != c_apres) <= 0.01


# ---------- Détecteur hors plage (scénario 3) ----------

def test_hors_plage_detecte(donnees):
    X, _ = donnees
    plages = training_ranges(X)
    nouveau = X.head(3).copy()
    nouveau.loc[0, "Amount"] = X["Amount"].max() * 10  # extrême
    nouveau.loc[1, "Amount"] = X["Amount"].max()       # valeur limite : pile au max

    assert list(out_of_range(nouveau, plages)) == [True, False, False]


# ---------- Dérive PSI (T18) ----------

def test_psi_stable_sur_meme_distribution(donnees):
    rng = np.random.default_rng(1)
    reference = donnees[0]["Amount"]
    meme_loi = rng.exponential(80, 2000)

    assert psi(reference, meme_loi) < 0.10


def test_psi_detecte_une_derive(donnees):
    """T18 : des montants multipliés par 3 déclenchent l'alerte (PSI > 0,25)."""
    reference = donnees[0]["Amount"]

    assert psi(reference, reference * 3) > 0.25


def test_psi_tranche_vide_reste_fini(donnees):
    """Cas limite : toutes les nouvelles valeurs dans une seule tranche, pas de log(0)."""
    reference = donnees[0]["Amount"]

    assert np.isfinite(psi(reference, np.full(500, reference.max() * 100)))