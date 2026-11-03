from pathlib import Path
import pandas as pd

from src.data_split import split_by_text
from src.evaluate_baseline import evaluate_predictions

from src.final_pipeline import MODEL_PATH, load_pipeline, DATA_PATH

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data"/ "raw" / "train.csv"

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    train_df, val_df = split_by_text(df)

    pipeline = load_pipeline(MODEL_PATH)
    predictions = pipeline.predict(val_df["text"])
    metrics = evaluate_predictions(val_df["target"], predictions)

    print("Saved Pipeline Verification — Validation Results")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"\nTN: {metrics['tn']}")
    print(f"FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}")
    print(f"TP: {metrics['tp']}")