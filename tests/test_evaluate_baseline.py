import pytest

from src.evaluate_baseline import evaluate_predictions

y_true = [0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 1, 1, 1, 0, 0, 1]

def test_evaluate_predictions():
    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(0.625)
    assert metrics["precision"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(0.60)
    assert metrics["f1"] == pytest.approx(2 / 3, abs=1e-4)

    assert metrics["tn"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 2
    assert metrics["tp"] == 3