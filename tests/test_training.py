from src.models.train import load_data
from sklearn.linear_model import LogisticRegression

def test_model_training():
    X, y = load_data()
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    assert model.coef_ is not None
