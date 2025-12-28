import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/heart_disease.csv")
PROCESSED_DATA_PATH = Path("data/processed/heart_disease_processed.csv")


def preprocess_data():
    df = pd.read_csv(RAW_DATA_PATH)

    # Binary classification: presence or absence
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    X = df.drop("target", axis=1)
    y = df["target"]

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    X_processed = pipeline.fit_transform(X)

    processed_df = pd.DataFrame(X_processed, columns=X.columns)
    processed_df["target"] = y.values

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)

    return processed_df, pipeline


if __name__ == "__main__":
    df, _ = preprocess_data()
    print(df.head())
