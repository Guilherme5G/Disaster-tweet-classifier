import pandas as pd

from src.final_pipeline import ROOT, load_pipeline
from src.train_submission import SUBMISSION_MODEL_PATH

TEST_PATH = ROOT / "data" / "raw" / "test.csv"
SAMPLE_PATH = ROOT / "data" / "raw" / "sample_submission.csv"
SUBMISSION_PATH = ROOT / "submissions" / "submission.csv"

def generate_submission(test_df, pipeline):
    preds = pipeline.predict(test_df["text"])
    submission = pd.DataFrame({
        "id": test_df["id"].to_numpy(),
        "target": preds
    })
    return submission

if __name__ == "__main__":
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_PATH)

    pipeline = load_pipeline(SUBMISSION_MODEL_PATH)
    submission = generate_submission(test_df, pipeline)

    assert submission.columns.tolist() == ["id", "target"]
    assert len(submission) == len(sample_submission)
    assert submission["id"].equals(sample_submission["id"])
    assert set(submission["target"]).issubset({0, 1})

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(submission.shape)
    print(submission.head(5).to_string(index=False))
