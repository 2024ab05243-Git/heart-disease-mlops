import pandas as pd
from pathlib import Path

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

RAW_DATA_PATH = Path("data/raw/heart_disease.csv")


def load_raw_data():
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_URL, header=None, names=COLUMN_NAMES, na_values="?")

    df.to_csv(RAW_DATA_PATH, index=False)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
    print(df.shape)
