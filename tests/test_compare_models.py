from src.compare_models import compare_models

def test_compare_models():
    X_train = [
        [1.0, 0.0],
        [0.8, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ]

    y_train = [1, 1, 0, 0]

    X_val = [
        [0.9, 0.1],
        [0.1, 0.8],
    ]

    y_val = [1, 0]

    results = compare_models(X_train, y_train, X_val, y_val)

    expected_models = {
        "MultinomialNB",
        "LogisticRegression",
        "LinearSVC",
    }
    assert set(results.keys()) == expected_models

    expected_keys = {"accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"}

    for model_name in expected_models:
        metrics = results[model_name]

        assert expected_keys.issubset(metrics.keys())
        assert 0.0 <= metrics["f1"] <= 1.0

        total = metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"]
        assert total == len(y_val)