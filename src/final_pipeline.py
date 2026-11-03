from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.data_split import split_by_text

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "train.csv"

MODEL_PATH = ROOT / "models" / "linear_svm_pipeline.joblib"

def build_final_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=42
        ))
    ])

def save_pipeline(pipeline, path):
    path.parent.mkdir(parents=True, exist_ok=True) #garante q exista no pc
    joblib.dump(pipeline, path) # salva o pipeline dentro do path

def load_pipeline(path):
    return joblib.load(path) # cvarrega o path de volta da memoria

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    train_df, val_df = split_by_text(df)

    pipeline = build_final_pipeline()
    pipeline.fit(train_df["text"], train_df["target"])
    save_pipeline(pipeline, MODEL_PATH)

    loaded = load_pipeline(MODEL_PATH)
    preds = loaded.predict(val_df["text"])
    assert (pipeline.predict(val_df["text"]) == preds).all()

    print(f"Modelo salvo e recarregado com sucesso em: {MODEL_PATH}")
    print(f"Quantidade de previsões geradas: {len(preds)}")
