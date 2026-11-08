from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.training import load_data, split_data, train_model

df = load_data("data/creditcard.csv")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
# X_test et y_test ne sont PAS utilisés ici : le jeu de test reste scellé.

for nom in ["logistic", "tree"]:
    modele = train_model(X_train, y_train, nom)

    proba_train = modele.predict_proba(X_train)[:, 1]
    proba_val = modele.predict_proba(X_val)[:, 1]
    pred_val = modele.predict(X_val)  # seuil par défaut : 0,5

    print(f"\n===== {nom} =====")
    print(f"PR-AUC  train : {average_precision_score(y_train, proba_train):.4f}")
    print(f"PR-AUC  val   : {average_precision_score(y_val, proba_val):.4f}")
    print(f"ROC-AUC val   : {roc_auc_score(y_val, proba_val):.4f}")
    print("--- au seuil par défaut 0,5 ---")
    print(f"Accuracy  : {accuracy_score(y_val, pred_val):.5f}")
    print(f"Précision : {precision_score(y_val, pred_val):.3f}")
    print(f"Rappel    : {recall_score(y_val, pred_val):.3f}")