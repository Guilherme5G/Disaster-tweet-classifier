from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

df = pd.read_csv(DATA_PATH)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_idx, val_idx = next(splitter.split(df, groups=df["text"]))

train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(train_df["text"])
X_val = vectorizer.transform(val_df["text"])

print(X_train.shape)
print(X_val.shape)

qtd_words = vectorizer.get_feature_names_out()
print(len(qtd_words))

assert X_train.shape[1] == X_val.shape[1], \
    "Erro: a quantidade de colunas é diferente!"

assert set(train_df["text"]).isdisjoint(set(val_df["text"]))

print("Mesma quantidade de colunas:", X_train.shape[1] == X_val.shape[1])


