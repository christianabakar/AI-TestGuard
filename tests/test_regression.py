import json
from pathlib import Path

import pytest

from app.evaluation import compare_to_baseline, evaluate_model
from app.prediction import predict_fraud
from app.training import load_data, select_threshold, split_data, train_model

RACINE = Path(__file__).resolve().parents[1]
CHEMIN_BASELINE = RACINE / "reports" / "baseline.json"
CHEMIN_DONNEES = RACINE / "data" / "creditcard.csv"

REFERENCE = {"recall": 0.81, "precision": 0.59, "fpr": 0.0009, "pr_auc": 0.70}


# ---------- Logique de décision (tourne partout, y compris dans la CI) ----------

def test_version_identique_acceptee():
    decision, motifs = compare_to_baseline(REFERENCE, dict(REFERENCE))

    assert decision == "ACCEPTER"
    assert motifs == []


def test_baisse_dans_la_tolerance_acceptee():
    """Une baisse de PR-AUC de 0,01 (tolérance 0,02) est un bruit normal."""
    nouveau = {**REFERENCE, "pr_auc": 0.69}

    assert compare_to_baseline(REFERENCE, nouveau)[0] == "ACCEPTER"


def test_baisse_hors_tolerance_a_investiguer():
    """T19 : PR-AUC en baisse de 0,05 (> 0,02) mais critères respectés -> investiguer."""
    nouveau = {**REFERENCE, "pr_auc": 0.65}
    decision, motifs = compare_to_baseline(REFERENCE, nouveau)

    assert decision == "INVESTIGUER"
    assert "pr_auc" in motifs[0]


@pytest.mark.parametrize("modif, motif", [
    ({"recall": 0.75}, "rappel"),
    ({"precision": 0.05}, "précision"),
    ({"fpr": 0.01}, "FPR"),
])
def test_critere_d_acceptation_viole_rejete(modif, motif):
    """Un critère bloquant du plan de test non respecté -> rejet, quelle que soit la baseline."""
    decision, motifs = compare_to_baseline(REFERENCE, {**REFERENCE, **modif})

    assert decision == "REJETER"
    assert motif in motifs[0]


# ---------- La baseline elle-même ----------

def test_baseline_presente_et_complete():
    """Les 7 éléments exigés par l'énoncé (section 6.2) sont enregistrés."""
    baseline = json.loads(CHEMIN_BASELINE.read_text(encoding="utf-8"))

    for cle in ["version_code", "parametres_modele", "versions_bibliotheques",
                "graine_aleatoire", "metriques_validation", "seuil", "donnees"]:
        assert cle in baseline, f"Élément manquant dans la baseline : {cle}"


# ---------- Non-régression sur les vraies données (si disponibles) ----------

@pytest.mark.skipif(not CHEMIN_DONNEES.exists(), reason="Jeu de données réel absent (CI)")
def test_version_actuelle_contre_baseline():
    """T19 : le code actuel, réentraîné, ne régresse pas par rapport à la baseline figée."""
    baseline = json.loads(CHEMIN_BASELINE.read_text(encoding="utf-8"))

    df = load_data(str(CHEMIN_DONNEES))
    X_train, X_val, _, y_train, y_val, _ = split_data(df)
    modele = train_model(X_train, y_train, "logistic")
    seuil = select_threshold(y_val, modele.predict_proba(X_val)[:, 1])
    proba, classes = predict_fraud(modele, X_val, seuil)
    nouveau = evaluate_model(y_val, classes, proba)

    decision, motifs = compare_to_baseline(baseline["metriques_validation"], nouveau)
    assert decision == "ACCEPTER", f"{decision} : {motifs}"