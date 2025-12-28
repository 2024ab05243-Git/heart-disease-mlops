from src.features.build_features import preprocess_data

def test_preprocess_data():
    df, pipeline = preprocess_data()
    assert "target" in df.columns
    assert df.isnull().sum().sum() == 0
