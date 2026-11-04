import pandas as pd

from src.train_submission import train_submission_model


def test_train_submission_model():
    train_df = pd.DataFrame({
        "text": [
            "fire evacuation",
            "music concert",
            "flood emergency",
            "happy birthday"
        ],
        "target": [1, 0, 1, 0]
    })

    pipeline = train_submission_model(train_df)

    assert "tfidf" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    new_texts = ["fire downtown", "birthday music"]
    preds = pipeline.predict(new_texts)

    assert len(preds) == len(new_texts)
    assert set(preds).issubset({0, 1})
