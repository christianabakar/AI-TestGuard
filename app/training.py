import pandas as pd
from sklearn.model_selection import train_test_split

from app.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, validate_data

RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Charge le CSV, vérifie le contrat, puis retire les doublons exacts (T08)."""
    df = validate_data(pd.read_csv(path))
    return df.drop_duplicates().reset_index(drop=True)


def split_data(df: pd.DataFrame):
    """Découpage stratifié 60 / 20 / 20 (T09). Renvoie 6 objets."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # 1er découpage : 60 % entraînement, 40 % mis de côté
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    # 2e découpage : les 40 % sont coupés en deux -> 20 % validation, 20 % test
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test