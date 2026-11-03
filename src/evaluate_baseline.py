from pathlib import Path
import pandas as pd
import math

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.data_split import split_by_text
from src.text_pipeline import vectorize_text

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"


def evaluate_predictions(y_true, y_pred):
    assert len(y_pred) == len(y_true)
    assert set(y_pred).issubset({0, 1})

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    assert cm.sum() == len(y_true)
    assert tn + fp + fn + tp == len(y_true)

    f1_manual = (
        2 * prec * rec / (prec + rec)
        if prec + rec > 0
        else 0.0
    )
    assert math.isclose(f1, f1_manual, rel_tol=1e-9)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    train_df, val_df = split_by_text(df)

    X_train, X_val, vectorizer = vectorize_text(train_df["text"], val_df["text"])

    y_train = train_df["target"]
    y_val = val_df["target"]

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    assert X_train.shape[0] == len(y_train)
    assert len(y_pred) == len(y_val)
    assert set(train_df["text"]).isdisjoint(set(val_df["text"]))
    assert set(y_pred).issubset({0, 1})

    metrics = evaluate_predictions(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")

    print("\nMatriz de confusão:")
    print(cm)

    print(f"\nTrue Negatives: {metrics['tn']}")
    print(f"False Positives: {metrics['fp']}")
    print(f"False Negatives: {metrics['fn']}")
    print(f"True Positives: {metrics['tp']}")

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))