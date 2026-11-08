import numpy as np
import pandas as pd
import pytest

from app.prediction import load_pipeline, predict_fraud, save_pipeline
from app.preprocessing import FEATURE_COLUMNS
from app.training import build_pipeline, load_data, split_data, train_model


@pytest.fixture
def dataset():
    """1 000 transactions synthétiques, 5 % de fraudes."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(1000, 30)), columns=FEATURE_COLUMNS)
    df["Amount"] = rng.uniform(0, 500, 1000)
    df["Class"] = 0
    df.loc[:49, "Class"] = 1
    return df


def test_doublons_supprimes_au_chargement(dataset, tmp_path):
    """T08 : les doublons exacts sont retirés avant tout découpage."""
    avec_doublons = pd.concat([dataset, dataset.iloc[:30]])  # 30 doublons ajoutés
    chemin = tmp_path / "data.csv"
    avec_doublons.to_csv(chemin, index=False)

    df = load_data(str(chemin))

    assert len(df) == 1000
    assert not df.duplicated().any()


def test_jeux_disjoints(dataset):
    """T09 / T05 : aucune transaction n'appartient à deux jeux à la fois."""
    X_train, X_val, X_test, *_ = split_data(dataset)

    train, val, test = set(X_train.index), set(X_val.index), set(X_test.index)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train | val | test) == len(dataset)  # rien n'est perdu


def test_proportions_60_20_20(dataset):
    """Les tailles respectent 60 / 20 / 20."""
    X_train, X_val, X_test, *_ = split_data(dataset)

    assert len(X_train) == 600
    assert len(X_val) == 200
    assert len(X_test) == 200


def test_stratification(dataset):
    """T09 : chaque jeu garde le taux de fraude global, à 0,02 point près."""
    *_, y_train, y_val, y_test = split_data(dataset)
    taux_global = dataset["Class"].mean()

    for y in (y_train, y_val, y_test):
        assert abs(y.mean() - taux_global) <= 0.0002


def test_decoupage_reproductible(dataset):
    """T06 : même graine, même découpage."""
    premier = split_data(dataset)
    second = split_data(dataset)

    assert list(premier[0].index) == list(second[0].index)



# ---------- Entraînement et Pipeline ----------

@pytest.mark.parametrize("nom", ["logistic", "tree"])
def test_entrainement_produit_un_modele_utilisable(dataset, nom):
    """Chaque modèle s'entraîne et prédit une probabilité par transaction."""
    X_train, X_val, _, y_train, *_ = split_data(dataset)
    modele = train_model(X_train, y_train, nom)
    proba = modele.predict_proba(X_val)[:, 1]

    assert proba.shape == (len(X_val),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_modele_inconnu_refuse():
    """Un nom de modèle inconnu lève une erreur explicite qui liste les choix."""
    with pytest.raises(ValueError, match="logistic"):
        build_pipeline("random_forest")


def test_scaler_du_pipeline_appris_sur_train_uniquement(dataset):
    """T10 au niveau du Pipeline réel : le scaler interne n'a vu que l'entraînement."""
    X_train, _, _, y_train, *_ = split_data(dataset)
    modele = train_model(X_train, y_train, "logistic")
    scaler = modele.named_steps["preprocessing"].named_transformers_["scale"]

    assert scaler.mean_[1] == pytest.approx(X_train["Amount"].mean())
    assert scaler.mean_[1] != pytest.approx(dataset["Amount"].mean())


def test_entrainement_reproductible(dataset):
    """T06 : deux entraînements identiques donnent exactement les mêmes probabilités."""
    X_train, X_val, _, y_train, *_ = split_data(dataset)
    p1 = train_model(X_train, y_train, "tree").predict_proba(X_val)
    p2 = train_model(X_train, y_train, "tree").predict_proba(X_val)

    np.testing.assert_array_equal(p1, p2)


# ---------- Sauvegarde et rechargement (R09) ----------

@pytest.fixture
def modele_sauvegarde(dataset, tmp_path):
    """Entraîne, choisit un seuil, sauvegarde. Renvoie tout ce qu'il faut pour comparer."""
    X_train, _, X_test, y_train, *_ = split_data(dataset)
    modele = train_model(X_train, y_train, "logistic")
    chemin = tmp_path / "modele.joblib"
    save_pipeline(modele, 0.3, chemin)
    return modele, X_test, chemin


def test_predictions_identiques_apres_rechargement(modele_sauvegarde):
    """T15 : le modèle rechargé prédit exactement comme le modèle d'origine."""
    modele, X_test, chemin = modele_sauvegarde
    modele_recharge, seuil = load_pipeline(chemin)

    p_avant, c_avant = predict_fraud(modele, X_test, 0.3)
    p_apres, c_apres = predict_fraud(modele_recharge, X_test, seuil)

    np.testing.assert_allclose(p_avant, p_apres, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(c_avant, c_apres)


def test_seuil_conserve(modele_sauvegarde):
    """Le seuil voyage avec le modèle : un modèle sans son seuil est inutilisable."""
    *_, chemin = modele_sauvegarde
    _, seuil = load_pipeline(chemin)

    assert seuil == 0.3


def test_fichier_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="introuvable"):
        load_pipeline(tmp_path / "inexistant.joblib")


def test_fichier_corrompu(tmp_path):
    chemin = tmp_path / "corrompu.joblib"
    chemin.write_bytes(b"ceci n'est pas un modele")

    with pytest.raises(ValueError, match="corrompu"):
        load_pipeline(chemin)


def test_format_invalide(tmp_path):
    """Un fichier joblib valide mais sans seuil est refusé."""
    import joblib
    chemin = tmp_path / "sans_seuil.joblib"
    joblib.dump({"pipeline": "x"}, chemin)

    with pytest.raises(ValueError, match="Format"):
        load_pipeline(chemin)


def test_version_incompatible(modele_sauvegarde, monkeypatch):
    """Un modèle créé avec une autre version de scikit-learn est refusé."""
    *_, chemin = modele_sauvegarde
    monkeypatch.setattr("app.prediction.sklearn.__version__", "0.0.1")

    with pytest.raises(ValueError, match="Version incompatible"):
        load_pipeline(chemin)