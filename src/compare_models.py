from pathlib import Path
import pandas as pd

from sklearn.naive_bayes import MultinomialNB

from src.data_split import split_by_text
from src.text_pipeline import vectorize_text
from src.evaluate_baseline import evaluate_predictions
from src.train_logistic import train_logistic_model

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

def compare_models(X_train, y_train, X_val, y_val):
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_val)

    lr_model = train_logistic_model(X_train, y_train)
    lr_pred = lr_model.predict(X_val)

    return {
        "MultinomialNB": evaluate_predictions(y_val, nb_pred),
        "LogisticRegression": evaluate_predictions(y_val, lr_pred),
    }

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    train_df, val_df = split_by_text(df)

    assert set(train_df["text"]).isdisjoint(set(val_df["text"]))

    X_train, X_val, vectorizer = vectorize_text(train_df["text"], val_df["text"])
    y_train = train_df["target"]
    y_val = val_df["target"]

    results = compare_models(X_train, y_train, X_val, y_val)

    for model_name, metrics in results.items():
        print(f"\n=== {model_name} ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-score:  {metrics['f1']:.4f}")
        print(f"TN: {metrics['tn']} | FP: {metrics['fp']} | FN: {metrics['fn']} | TP: {metrics['tp']}")

    comparison_df = pd.DataFrame(results).T
    print("\n=== Tabela Comparativa ===")
    print(comparison_df[["accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]].round(4).to_string())

    nb = results["MultinomialNB"]
    lr = results["LogisticRegression"]
    print("\n=== Diferenças (LogisticRegression - MultinomialNB) ===")
    print(f"Diferença F1-score: {lr['f1'] - nb['f1']:+.4f}")
    print(f"Diferença Recall:   {lr['recall'] - nb['recall']:+.4f}")
    print(f"Diferença TP:       {lr['tp'] - nb['tp']:+d}")
    print(f"Diferença FN:       {lr['fn'] - nb['fn']:+d}")
    print(f"Diferença FP:       {lr['fp'] - nb['fp']:+d}")

    print("Model Comparison — Validation Set\n")
    print(f"{'Model':<22}{'Accuracy':<12}{'Precision':<12}{'Recall':<10}{'F1':<10}")
    for model_name, m in results.items():
        print(
            f"{model_name:<22}"
            f"{m['accuracy']:<12.4f}"
            f"{m['precision']:<12.4f}"
            f"{m['recall']:<10.4f}"
            f"{m['f1']:<10.4f}"
        )
    print("\nConfusion Matrix Summary\n")
    print(f"{'Model':<22}{'TN':<8}{'FP':<8}{'FN':<8}{'TP':<8}")
    for model_name, m in results.items():
        print(
            f"{model_name:<22}"
            f"{m['tn']:<8}"
            f"{m['fp']:<8}"
            f"{m['fn']:<8}"
            f"{m['tp']:<8}"
        )



