import numpy as np
import pandas as pd
import pytest

from app.preprocessing import (
    FEATURE_COLUMNS,
    SchemaError,
    fit_preprocessor,
    preprocess_data,
    validate_data,
)


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


# ---------- Prétraitement et prévention des fuites ----------

@pytest.fixture
def train_test():
    """Entraînement et test aux distributions VOLONTAIREMENT différentes."""
    rng = np.random.default_rng(42)

    def make(n, montant_moyen):
        df = pd.DataFrame(rng.normal(size=(n, 30)), columns=FEATURE_COLUMNS)
        df["Time"] = rng.uniform(0, 1000, n)
        df["Amount"] = rng.normal(montant_moyen, 10, n).clip(min=0)
        return df

    return make(200, montant_moyen=50), make(50, montant_moyen=5000)


def test_scaler_appris_sur_entrainement_uniquement(train_test):
    """T10 : la moyenne apprise est celle de l'entraînement, pas celle du test."""
    X_train, X_test = train_test
    scaler = fit_preprocessor(X_train)

    assert scaler.mean_[1] == pytest.approx(X_train["Amount"].mean())
    assert abs(scaler.mean_[1] - X_test["Amount"].mean()) > 1000


def test_appliquer_ne_reapprend_pas(train_test):
    """T10 : transformer le test ne modifie pas les paramètres appris."""
    X_train, X_test = train_test
    scaler = fit_preprocessor(X_train)
    moyenne_avant = scaler.mean_.copy()

    preprocess_data(X_test, scaler)

    np.testing.assert_array_equal(scaler.mean_, moyenne_avant)


def test_entrainement_centre_reduit(train_test):
    """Sur l'entraînement, Time et Amount sont ramenés à moyenne 0 et écart-type 1."""
    X_train, _ = train_test
    result = preprocess_data(X_train, fit_preprocessor(X_train))

    assert result["Amount"].mean() == pytest.approx(0, abs=1e-10)
    assert result["Amount"].std(ddof=0) == pytest.approx(1)


def test_dimensions_et_valeurs_finies(train_test):
    """T03 + CT-10 : N lignes en entrée, N x 30 en sortie, aucune valeur non finie."""
    X_train, X_test = train_test
    result = preprocess_data(X_test, fit_preprocessor(X_train))

    assert result.shape == (len(X_test), 30)
    assert np.isfinite(result.to_numpy()).all()


def test_variables_pca_inchangees(train_test):
    """V1 à V28 ne doivent pas être modifiées par le prétraitement."""
    X_train, X_test = train_test
    result = preprocess_data(X_test, fit_preprocessor(X_train))

    pd.testing.assert_frame_equal(result[FEATURE_COLUMNS[1:-1]],
                                  X_test[FEATURE_COLUMNS[1:-1]])


def test_garde_fou_valeurs_non_finies(train_test):
    """T03 : défense en profondeur, une valeur non finie est stoppée au prétraitement."""
    X_train, X_test = train_test
    scaler = fit_preprocessor(X_train)
    X_corrompu = X_test.copy()
    X_corrompu.loc[0, "Amount"] = np.nan  # on contourne volontairement validate_data

    with pytest.raises(SchemaError, match="non finies"):
        preprocess_data(X_corrompu, scaler)