from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV

from src.data_split import split_by_text

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

def tune_logistic(texts, labels, groups):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    param_grid = {
        "classifier__C": [0.1, 1.0, 10.0]
    }

    cv = GroupKFold(n_splits=3)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        refit=True,
        n_jobs=1
    )

    search.fit(texts, labels, groups=groups)

    return search

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    train_df, val_df = split_by_text(df)

    search = tune_logistic(
        texts=train_df["text"],
        labels=train_df["target"],
        groups=train_df["text"]
    )

    print("Best parameters:", search.best_params_)
    print(f"Best mean CV F1: {search.best_score_:.4f}")

    for params, mean, std in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"]
    ):
        print(
            f"C={params['classifier__C']}: "
            f"Mean F1={mean:.4f}, "
            f"Std={std:.4f}"
        )