import logging
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.prediction import predict_fraud
from app.preprocessing import FEATURE_COLUMNS, SchemaError, validate_data
from app.training import train_model

RACINE = Path(__file__).resolve().parents[1]
MONTANT_TEMOIN = 98765.43  # valeur « marquée », facile à retrouver dans un texte


@pytest.fixture(scope="module")
def modele():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(300, 30)), columns=FEATURE_COLUMNS)
    X["Amount"] = rng.uniform(0, 500, 300)
    y = pd.Series([1] * 30 + [0] * 270)
    return train_model(X, y, "logistic")


@pytest.fixture
def transaction():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(1, 30)), columns=FEATURE_COLUMNS)
    X["Amount"] = MONTANT_TEMOIN
    return X


# ---------- Journaux (logs) ----------

def test_journal_sans_donnees_de_transaction(modele, transaction, caplog):
    """T21 : le journal trace l'activité, jamais le contenu d'une transaction."""
    with caplog.at_level(logging.INFO, logger="app.prediction"):
        predict_fraud(modele, transaction, 0.5)

    assert "1 transaction(s)" in caplog.text            # l'activité est bien tracée...
    assert "98765" not in caplog.text                   # ...mais pas le montant
    assert str(round(transaction.loc[0, "V1"], 3))[:5] not in caplog.text  # ni V1


def test_message_d_erreur_sans_valeur_sensible(transaction):
    """Un rejet nomme la colonne fautive, sans recopier la valeur reçue."""
    transaction.loc[0, "Amount"] = -MONTANT_TEMOIN

    with pytest.raises(SchemaError) as erreur:
        validate_data(transaction, require_target=False)

    assert "Amount" in str(erreur.value)
    assert "98765" not in str(erreur.value)


# ---------- Contenu du dépôt Git ----------

def fichiers_suivis():
    if shutil.which("git") is None or not (RACINE / ".git").exists():
        pytest.skip("Dépôt Git indisponible")
    sortie = subprocess.run(["git", "ls-files"], cwd=RACINE, capture_output=True,
                            text=True, check=True).stdout
    return [ligne for ligne in sortie.splitlines() if ligne]


def test_aucune_donnee_ni_modele_dans_le_depot():
    """T21 : ni données brutes, ni modèle, ni fichier d'environnement versionnés.

    Exception justifiée : reports/*.csv contient des résultats AGRÉGÉS (livrables),
    aucune transaction individuelle.
    """
    fichiers = fichiers_suivis()
    interdits = [f for f in fichiers if f.endswith((".joblib", ".pkl", ".env"))]
    interdits += [f for f in fichiers if f.endswith(".csv") and not f.startswith("reports/")]

    assert interdits == [], f"Fichiers interdits dans Git : {interdits}"


MOTIFS_SECRETS = {
    "clé AWS": r"AKIA[0-9A-Z]{16}",
    "jeton GitHub": r"gh[pousr]_[A-Za-z0-9]{36}",
    "clé privée": r"-----BEGIN [A-Z ]*PRIVATE" + r" KEY-----",
    "mot de passe en clair": r"(?i)(password|passwd|mot_de_passe)\s*=\s*['\"][^'\"]+['\"]",
    "clé d'API en clair": r"(?i)api[_-]?key\s*=\s*['\"][^'\"]+['\"]",
}


def test_aucun_secret_dans_le_depot():
    """T21 : aucun fichier versionné ne contient de secret reconnaissable."""
    trouves = []
    for fichier in fichiers_suivis():
        chemin = RACINE / fichier
        if chemin.suffix not in {".py", ".md", ".txt", ".yml", ".yaml", ".json", ".toml",
                                 ".cfg", ".ini"} or chemin.name == "test_security.py":
            continue
        texte = chemin.read_text(encoding="utf-8", errors="ignore")
        trouves += [f"{fichier} : {nom}" for nom, motif in MOTIFS_SECRETS.items()
                    if re.search(motif, texte)]

    assert trouves == [], f"Secrets potentiels : {trouves}"