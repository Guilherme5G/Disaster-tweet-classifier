from pathlib import Path
import textwrap
import pandas as pd
from sklearn.svm import LinearSVC

from src.data_split import split_by_text
from src.text_pipeline import vectorize_text

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

def extract_errors(val_df, predictions):
    if len(val_df) != len(predictions):
        raise ValueError(
            "Number of predictions must match validation samples"
        )

    errors_df = val_df[["id", "text", "target"]].copy()
    errors_df["prediction"] = predictions

    fp = errors_df[(errors_df["target"] == 0) & (errors_df["prediction"] == 1)]
    fn = errors_df[(errors_df["target"] == 1) & (errors_df["prediction"] == 0)]

    return fp, fn

def display_errors(title, errors_df, limit=5):
    print("\n" + "=" * 75)
    print(f"{title.upper()} — Total: {len(errors_df)}")
    print("=" * 75)

    for index, (_, row) in enumerate(errors_df.head(limit).iterrows(), start=1):
        # Normaliza quebras de linha e espaços repetidos
        text = " ".join(str(row["text"]).split())

        print(f"\n[{index:02d}] Tweet ID: {row['id']}")
        print(f"     Actual: {row['target']} | Predicted: {row['prediction']}")
        print()

        formatted_text = textwrap.fill(
            text,
            width=90,
            initial_indent="     ",
            subsequent_indent="     ",
            break_long_words=False
        )

        print(formatted_text)

    print("\n" + "-" * 75)

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    train_df, val_df = split_by_text(df)

    X_train, X_val, vectorizer = vectorize_text(
        train_df["text"],
        val_df["text"]
    )

    model = LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=42
    )

    model.fit(X_train, train_df["target"])

    predictions = model.predict(X_val)

    fp, fn = extract_errors(val_df, predictions)

    print("\nLINEAR SVM — ERROR ANALYSIS")
    print(f"Validation samples: {len(val_df)}")

    display_errors(
        title="False Positives",
        errors_df=fp,
        limit=5
    )

    display_errors(
        title="False Negatives",
        errors_df=fn,
        limit=5
    )