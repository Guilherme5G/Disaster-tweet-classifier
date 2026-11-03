from pathlib import Path

import pandas as pd

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.data_split import split_by_text

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

def build_candidates():
    return {
        "NaiveBayes_alpha1": MultinomialNB(alpha=1.0),

        "LogisticRegression_C1": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        ),

        "LogisticRegression_C10": LogisticRegression(
            C=10.0,
            max_iter=1000,
            random_state=42
        ),

        "LinearSVC_C1": LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=42
        )
    }

def evaluate_candidates(train_df):
    texts = train_df["text"]
    labels = train_df["target"]
    groups = train_df["text"]

    cv = GroupKFold(n_splits=3)
    folds = list(cv.split(texts, labels, groups=groups))

    candidates = build_candidates()
    rows = []

    for name, classifier in candidates.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", classifier)
        ])

        scores = cross_validate(
            estimator=pipeline,
            X=texts,
            y=labels,
            cv=folds,
            scoring={
                "f1": "f1",
                "precision": "precision",
                "recall": "recall"
            },
            n_jobs=1
        )
        rows.append({
            "model": name,
            "f1_mean": scores["test_f1"].mean(),
            "f1_std": scores["test_f1"].std(),
            "precision_mean": scores["test_precision"].mean(),
            "recall_mean": scores["test_recall"].mean(),
        })

    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    train_df, val_df = split_by_text(df)

    results = evaluate_candidates(train_df)

    print("\n=== GROUP-AWARE MODEL SELECTION ===")
    print(results.to_string(index=False))