from sklearn.metrics import confusion_matrix

from app.prediction import predict_fraud
from app.training import load_data, select_threshold, split_data, train_model

df = load_data("data/creditcard.csv")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
# Le jeu de test reste scellé : il n'apparaît nulle part ci-dessous.

for nom in ["logistic", "tree"]:
    modele = train_model(X_train, y_train, nom)
    proba_val = modele.predict_proba(X_val)[:, 1]

    try:
        seuil = select_threshold(y_val, proba_val, min_recall=0.80)
    except ValueError as erreur:
        print(f"\n===== {nom} =====\nREJETÉ : {erreur}")
        continue

    _, classes = predict_fraud(modele, X_val, seuil)
    tn, fp, fn, tp = confusion_matrix(y_val, classes, labels=[0, 1]).ravel()

    print(f"\n===== {nom} =====")
    print(f"Seuil retenu     : {seuil:.5f}")
    print(f"Fraudes détectées (TP) : {tp}   ratées (FN) : {fn}")
    print(f"Fausses alertes  (FP) : {fp}   normales OK (TN) : {tn}")
    print(f"Rappel    : {tp / (tp + fn):.3f}   (exigence >= 0,80)")
    print(f"Précision : {tp / (tp + fp):.3f}   (exigence >= 0,10)")
    print(f"FPR       : {100 * fp / (fp + tn):.3f} %  (exigence <= 0,5 %)")