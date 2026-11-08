import json
import os
import platform
import time

import numpy as np
import psutil

from app.prediction import load_pipeline, predict_fraud
from app.training import load_data, split_data

CHEMIN_MODELE = "reports/modele_logistic.joblib"
REPETITIONS = 1000
ECHAUFFEMENT = 50
processus = psutil.Process()


def percentiles_ms(durees):
    d = np.array(durees) * 1000
    return {"moyenne_ms": float(d.mean()), "p50_ms": float(np.percentile(d, 50)),
            "p95_ms": float(np.percentile(d, 95)), "p99_ms": float(np.percentile(d, 99))}


def chronometrer(fonction, repetitions):
    for _ in range(ECHAUFFEMENT):  # échauffement : caches, chargement paresseux
        fonction()
    durees = []
    for _ in range(repetitions):
        debut = time.perf_counter()
        fonction()
        durees.append(time.perf_counter() - debut)
    return durees


# --- Données : transactions de test (jamais utilisées pour régler quoi que ce soit) ---
df = load_data("data/creditcard.csv")
*_, X_test, _, _, _ = split_data(df)

# --- Mémoire et taille ---
memoire_avant = processus.memory_info().rss
modele, seuil = load_pipeline(CHEMIN_MODELE)
memoire_modele = processus.memory_info().rss - memoire_avant
taille_fichier = os.path.getsize(CHEMIN_MODELE)

une_transaction = X_test.iloc[[0]]

# --- 1. Latence d'UNE prédiction : modèle seul vs chaîne complète ---
modele_seul = percentiles_ms(chronometrer(lambda: modele.predict_proba(une_transaction),
                                          REPETITIONS))
chaine_complete = percentiles_ms(chronometrer(
    lambda: predict_fraud(modele, une_transaction, seuil), REPETITIONS))

# --- 2. Débit par taille de lot ---
debits = {}
for taille in [1, 100, 1_000, 10_000]:
    lot = X_test.iloc[:taille]
    durees = chronometrer(lambda: predict_fraud(modele, lot, seuil), 20)
    debits[taille] = float(taille / np.median(durees))

memoire_pic = processus.memory_info().rss

resultats = {
    "protocole": {
        "machine": f"{platform.system()} {platform.release()} | {platform.processor()}",
        "coeurs_cpu": psutil.cpu_count(logical=True),
        "ram_totale_go": round(psutil.virtual_memory().total / 1e9, 1),
        "python": platform.python_version(),
        "methode": "time.perf_counter, 50 itérations d'échauffement exclues",
        "repetitions_latence": REPETITIONS,
        "repetitions_debit": 20,
        "tailles_de_lot": [1, 100, 1_000, 10_000],
    },
    "latence_1_transaction_modele_seul": modele_seul,
    "latence_1_transaction_chaine_complete": chaine_complete,
    "debit_tx_par_seconde": debits,
    "taille_modele_ko": round(taille_fichier / 1024, 1),
    "memoire_chargement_modele_mo": round(memoire_modele / 1e6, 1),
    "memoire_processus_pic_mo": round(memoire_pic / 1e6, 1),
}

with open("reports/performance.json", "w", encoding="utf-8") as f:
    json.dump(resultats, f, indent=2, ensure_ascii=False)

p = resultats["protocole"]
print(f"Machine : {p['machine']} | {p['coeurs_cpu']} coeurs | {p['ram_totale_go']} Go RAM\n")
print("LATENCE D'UNE TRANSACTION (ms)     moyenne    P50     P95     P99")
for nom, r in [("Modèle seul", modele_seul), ("Chaîne complète (validation)", chaine_complete)]:
    print(f"  {nom:30} {r['moyenne_ms']:7.3f} {r['p50_ms']:7.3f} {r['p95_ms']:7.3f} "
          f"{r['p99_ms']:7.3f}")
print("\nDÉBIT (chaîne complète)")
for taille, debit in debits.items():
    print(f"  lot de {taille:>6} transactions : {debit:>12,.0f} tx/s")
print(f"\nTaille du modèle sauvegardé : {resultats['taille_modele_ko']} Ko")
print(f"Mémoire pour charger le modèle : {resultats['memoire_chargement_modele_mo']} Mo")
print(f"Mémoire du processus (pic)     : {resultats['memoire_processus_pic_mo']} Mo")
print("\nRésultats écrits dans reports/performance.json")