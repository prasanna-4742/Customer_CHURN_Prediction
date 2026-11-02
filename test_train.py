import joblib

from scripts.generate_sample_data import generate
from src.train import train


def test_training_pipeline_end_to_end(tmp_path):
    data = tmp_path / "data.csv"
    generate(600, seed=1).to_csv(data, index=False)
    best, results = train(data, tmp_path / "m.joblib", tmp_path / "metrics.json")
    assert len(results) >= 2
    assert best in results
    assert results[best]["roc_auc"] > 0.6
    model = joblib.load(tmp_path / "m.joblib")
    assert hasattr(model, "predict_proba")
