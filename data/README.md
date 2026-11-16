ÉVALUATION FORMATIVE N° 1 Test logiciel et assurance qualité d'un modèle d'intelligence artificielle
# AI-TestGuard

Qualification d'un modèle de détection de fraude bancaire : stratégie de test, tests
automatisés, robustesse, non-régression, performance et intégration continue.

## Données

Le jeu Credit Card Fraud Detection (Kaggle) n'est pas versionné : voir `data/README.md`.

## Installation (Python 3.14)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Commandes

| Action | Commande |
|---|---|
| Style | `ruff check .` |
| Tests + couverture | `pytest -v --cov=app --cov-report=term-missing` |
| Évaluation finale (jeu de test) | `python evaluation_finale.py` |
| Reproductibilité du pipeline | `python pipeline_complet.py` |
| Robustesse | `python mesurer_robustesse.py` |
| Baseline (une seule fois) | `python creer_baseline.py` |
| Rapport de non-régression | `python comparer_versions.py` |
| Performance | `python mesurer_performance.py` |
| Simulation de surveillance | `python simuler_surveillance.py` |

## Résultats

70 tests, couverture 100 %. Modèle retenu : régression logistique (seuil 0,01772),
rappel 0,853 [IC 95 % : 0,780 ; 0,921], FPR 0,095 %. Rapports dans `reports/`.











# Données

Le jeu de données n'est pas versionné dans ce dépôt (taille : 150 Mo, licence de la source).

**Source :** Credit Card Fraud Detection — Kaggle
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Emplacement attendu :** `data/creditcard.csv`

**Caractéristiques :** 284 807 transactions, 31 colonnes, 492 fraudes (0,173 %)

**Empreinte SHA-256 de référence :**
`76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`

Pour vérifier votre copie (PowerShell) :
`Get-FileHash data\creditcard.csv -Algorithm SHA256`