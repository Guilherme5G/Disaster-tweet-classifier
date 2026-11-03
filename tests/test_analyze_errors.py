import pandas as pd
import pytest

from src.analyze_errors import extract_errors

def test_extract_errors():
    val_df = pd.DataFrame({
        "id": [10, 20, 30, 40],
        "text": [
            "fire downtown",
            "music concert",
            "earthquake reported",
            "sunny afternoon"
        ],
        "target": [1, 0, 1, 0]
    })

    predictions = [1, 1, 0, 0]

    fp, fn = extract_errors(val_df, predictions)

    assert len(fp) == 1
    assert len(fn) == 1

    assert list(fp["id"]) == [20]
    assert list(fn["id"]) == [30]

    assert (fp["target"] == 0).all()
    assert (fp["prediction"] == 1).all()
    assert (fn["target"] == 1).all()
    assert (fn["prediction"] == 0).all()

def test_extract_errors_invalid_length():
    val_df = pd.DataFrame({
        "id": [1, 2, 3],
        "text": ["fire", "music", "flood"],
        "target": [1, 0, 1]
    })

    predictions = [1, 0]

    with pytest.raises(ValueError):
        extract_errors(val_df, predictions)
