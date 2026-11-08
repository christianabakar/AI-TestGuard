import numpy as np
import pytest
from sklearn.metrics import recall_score

from app.evaluation import bootstrap_ci, evaluate_model

Y_TRUE = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
Y_PRED = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
Y_PROBA = [0.9, 0.8, 0.3, 0.6, 0.1, 0.2, 0.1, 0.05, 0.3, 0.2]


def test_matrice_de_confusion_calculee_a_la_main():
    """La matrice correspond au calcul fait à la main."""
    m = evaluate_model(Y_TRUE, Y_PRED, Y_PROBA)

    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (2, 1, 1, 6)


def test_metriques_calculees_a_la_main():
    """Chaque métrique correspond à l'oracle calculé de tête."""
    m = evaluate_model(Y_TRUE, Y_PRED, Y_PROBA)

    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 7)
    assert m["fnr"] == pytest.approx(1 / 3)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["recall"] + m["fnr"] == pytest.approx(1)  # propriété : rappel + FNR = 1


def test_aucune_alerte_ne_plante_pas():
    """Cas limite : un modèle qui ne signale rien ne provoque pas de division par zéro."""
    m = evaluate_model(Y_TRUE, [0] * 10, Y_PROBA)

    assert m["precision"] == 0
    assert m["recall"] == 0
    assert m["fpr"] == 0


def test_piege_de_l_accuracy():
    """R04 : un modèle qui répond toujours « normale » a 99,8 % d'accuracy et ne détecte rien."""
    y_true = [1] * 2 + [0] * 998  # 0,2 % de fraudes, comme le vrai jeu
    m = evaluate_model(y_true, [0] * 1000, [0.0] * 1000)

    assert m["accuracy"] == pytest.approx(0.998)
    assert m["recall"] == 0
    assert m["pr_auc"] < 0.01


def test_intervalle_de_confiance():
    """L'IC contient la valeur mesurée, et il est reproductible à graine fixe."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = np.where(rng.random(200) < 0.8, y_true, 1 - y_true)  # 80 % de bonnes réponses

    bas, haut = bootstrap_ci(y_true, y_pred, recall_score, n_iter=300)

    assert bas < recall_score(y_true, y_pred) < haut
    assert (bas, haut) == bootstrap_ci(y_true, y_pred, recall_score, n_iter=300)