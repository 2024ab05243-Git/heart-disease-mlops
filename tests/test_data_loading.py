from src.data.load_data import load_raw_data

def test_load_raw_data():
    df = load_raw_data()
    assert df.shape[0] > 0
    assert "target" in df.columns
