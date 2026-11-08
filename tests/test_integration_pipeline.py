from pathlib import Path
import joblib
import pytest

from src.inference import classify_tweet

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_MODEL_PATH = (
    ROOT / "models" / "linear_svm_submission.joblib"
)

def test_real_pipeline_inference():
    if not SUBMISSION_MODEL_PATH.exists():
        pytest.skip("Modelo nao encontrado!")

    pipeline = joblib.load(SUBMISSION_MODEL_PATH)

    assert "tfidf" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    tweet1 = classify_tweet("Wildfire spreads near downtown. Residents evacuated.", pipeline)
    assert tweet1 == "Disaster (1)"

    tweet2 = classify_tweet("Happy birthday! Let's celebrate with some music.", pipeline)
    assert tweet2 == "Not Disaster (0)"

