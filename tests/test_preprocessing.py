import numpy as np
import pandas as pd
import pytest

from app.preprocessing import FEATURE_COLUMNS, SchemaError, validate_data


@pytest.fixture
def valid_df():
    """Petit jeu synthétique valide : 5 transactions, 31 colonnes."""
    rng = np.random.default_rng(42)  # graine fixe -> toujours les mêmes données
    df = pd.DataFrame(rng.normal(size=(5, 30)), columns=FEATURE_COLUMNS)
    df["Time"] = [0.0, 10.0, 20.0, 30.0, 40.0]
    df["Amount"] = [1.0, 25.5, 100.0, 0.0, 999.9]
    df["Class"] = [0, 0, 1, 0, 1]
    return df


def test_donnees_valides_acceptees(valid_df):
    """CT-01 : un jeu conforme est accepté et rien n'est perdu."""
    result = validate_data(valid_df)

    assert result.shape == (5, 31)
    assert list(result.columns) == FEATURE_COLUMNS + ["Class"]
    

def test_colonne_manquante_rejetee(valid_df):
    """CT-02 : une colonne absente est rejetée, et le message la nomme."""
    df = valid_df.drop(columns=["V14"])

    with pytest.raises(SchemaError, match="V14"):
        validate_data(df)


def test_colonne_dupliquee_rejetee(valid_df):
    """CT-03 : une colonne en double est rejetée."""
    df = pd.concat([valid_df, valid_df[["Amount"]]], axis=1)

    with pytest.raises(SchemaError, match="dupliquées"):
        validate_data(df)


def test_donnees_vides_rejetees(valid_df):
    """CT-07 : un tableau sans aucune ligne est rejeté."""
    df = valid_df.iloc[0:0]  # mêmes colonnes, zéro ligne

    with pytest.raises(SchemaError, match="vide"):
        validate_data(df)


def test_type_texte_rejete(valid_df):
    """CT-04 : un montant écrit en texte ("12,50") est rejeté, pas converti."""
    df = valid_df.astype({"Amount": object})
    df.loc[0, "Amount"] = "12,50"

    with pytest.raises(SchemaError, match="non numérique"):
        validate_data(df)


@pytest.mark.parametrize("valeur", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("colonne", ["V3", "Amount"])
def test_valeurs_non_finies_rejetees(valid_df, colonne, valeur):
    """CT-05 / CT-06 : NaN, +inf et -inf sont rejetés, la colonne est nommée."""
    df = valid_df.copy()
    df.loc[2, colonne] = valeur

    with pytest.raises(SchemaError, match=colonne):
        validate_data(df)


def test_montant_negatif_rejete(valid_df):
    """CT-11 : un montant négatif est rejeté (contrôle du domaine métier)."""
    df = valid_df.copy()
    df.loc[0, "Amount"] = -0.01  # valeur limite : juste sous zéro

    with pytest.raises(SchemaError, match="négatif"):
        validate_data(df)


@pytest.mark.parametrize("cible_invalide", [2, -1])
def test_cible_hors_domaine_rejetee(valid_df, cible_invalide):
    """CT-08 : Class ne peut valoir que 0 ou 1."""
    df = valid_df.copy()
    df.loc[0, "Class"] = cible_invalide

    with pytest.raises(SchemaError, match=str(cible_invalide)):
        validate_data(df)


def test_colonne_en_trop_ignoree_avec_alerte(valid_df):
    """CT-12 : une colonne inconnue déclenche un avertissement et n'atteint pas le modèle."""
    df = valid_df.assign(customer_id=123)

    with pytest.warns(UserWarning, match="customer_id"):
        result = validate_data(df)

    assert "customer_id" not in result.columns
    assert result.shape == (5, 31)


def test_une_seule_transaction_acceptee(valid_df):
    """CT-09 : valeur limite, un tableau d'une seule ligne est valide."""
    result = validate_data(valid_df.iloc[[0]])

    assert result.shape == (1, 31)


def test_mode_prediction_sans_cible(valid_df):
    """En production, la transaction arrive sans Class : elle doit être acceptée."""
    df = valid_df.drop(columns=["Class"])
    result = validate_data(df, require_target=False)

    assert list(result.columns) == FEATURE_COLUMNS


def test_donnees_originales_non_modifiees(valid_df):
    """La validation ne doit jamais altérer le tableau fourni par l'appelant."""
    df = valid_df.assign(customer_id=123)
    copie_avant = df.copy()

    with pytest.warns(UserWarning):
        validate_data(df)

    pd.testing.assert_frame_equal(df, copie_avant)