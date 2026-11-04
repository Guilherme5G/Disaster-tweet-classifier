import pandas as pd

from src.generate_submission import generate_submission

class FakePipeline:
    def predict(self, texts):
        return [
            1 if "fire" in text else 0
            for text in texts
        ]

def test_generate_submission():
    test_df = pd.DataFrame({
        "id": [30, 10, 20],
        "text": [
            "fire downtown",
            "happy birthday",
            "fire evacuation"
        ]
    })
    submission = generate_submission(test_df, FakePipeline())

    assert submission.columns.tolist() == ["id", "target"]
    assert submission["id"].tolist() == [30, 10, 20]
    assert submission["target"].tolist() == [1, 0, 1]
