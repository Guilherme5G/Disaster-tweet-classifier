from pathlib import Path
import pandas as pd

from sklearn.linear_model import LogisticRegression

from src.data_split import split_by_text
from src.text_pipeline import vectorize_text
from src.evaluate_baseline import evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

def train_logistic_model(X_train, y_train):
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)
    return  model

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    train_df, val_df = split_by_text(df)

    X_train, X_val, vectorizer = vectorize_text(train_df["text"], val_df["text"])

    y_train = train_df["target"]
    y_val = val_df["target"]

    model = train_logistic_model(X_train, y_train)

    y_pred = model.predict(X_val)

    metrics = evaluate_predictions(y_val, y_pred)

    assert len(y_pred) == len(y_val)
    assert set(y_pred).issubset({0, 1})
    assert set(train_df["text"]).isdisjoint(set(val_df["text"]))

    print("\nLogistic Regression — Validation Results")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")

    print("\nConfusion Matrix:")
    print([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]]
    ])

    print(f"\nTrue Negatives: {metrics['tn']}")
    print(f"False Positives: {metrics['fp']}")
    print(f"False Negatives: {metrics['fn']}")
    print(f"True Positives: {metrics['tp']}")

    print(f"\nOptimizer iterations: {model.n_iter_}")
