from pathlib import Path
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from src.data_split import split_by_text
from src.text_pipeline import vectorize_text

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

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

print(f"Formato X_train: {X_train.shape}")
print(f"Formato X_val: {X_val.shape}")
print(f"Quantidade de labels no treino: {len(y_train)}")
print(f"Quantidade de previsões: {len(y_pred)}")
print(f"Primeiras 10 previsões: {y_pred[:10]}")