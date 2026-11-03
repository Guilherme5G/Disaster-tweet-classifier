import pandas as pd
from src.select_model import evaluate_candidates

def test_evaluate_candidates():
    train_df = pd.DataFrame({
        "text": [
            "fire evacuation alert",
            "music concert party",
            "fire smoke downtown",
            "music dance festival",
            "fire rescue emergency",
            "music guitar song",
        ],
        "target": [1, 0, 1, 0, 1, 0],
    })
    results = evaluate_candidates(train_df)

    assert set(results["model"]) == {"NaiveBayes_alpha1", "LogisticRegression_C1", "LogisticRegression_C10", "LinearSVC_C1"}

    assert (results["f1_mean"] >= 0.0).all() and (results["f1_mean"] <= 1.0).all()

    assert (results["recall_mean"] >= 0.0).all() and (results["recall_mean"] <= 1.0).all()

    assert not results.isna().any().any()

    assert list(results["f1_mean"]) == sorted(results["f1_mean"], reverse=True)