from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_split import split_by_text

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

def vectorize_text(train_texts, val_texts):
    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    return X_train, X_val, vectorizer

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    train_df, val_df = split_by_text(df)

    X_train, X_val, vectorizer = vectorize_text(train_df["text"], val_df["text"])

    print(X_train.shape)
    print(X_val.shape)

    qtd_words = vectorizer.get_feature_names_out()
    print(len(qtd_words))

    assert X_train.shape[1] == X_val.shape[1], \
        "Erro: a quantidade de colunas é diferente!"

    assert set(train_df["text"]).isdisjoint(set(val_df["text"]))

    print("Mesma quantidade de colunas:", X_train.shape[1] == X_val.shape[1])