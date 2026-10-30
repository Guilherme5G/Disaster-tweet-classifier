from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

df = pd.read_csv(DATA_PATH)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_idx, val_idx = next(splitter.split(df, groups=df["text"]))

train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]

print(f"Quantidade de registros de treino: {len(train_df)}")
print(f"Quantidade de registros de validação: {len(val_df)}\n")

print("Distribuição das classes no treino:")
print(train_df["target"].value_counts(normalize=True))
print("\nDistribuição das classes na validação:")
print(val_df["target"].value_counts(normalize=True))

textos_treino = set(train_df["text"])
textos_val = set(val_df["text"])
textos_comuns = textos_treino.intersection(textos_val)

# Verificações de integridade da divisão
assert len(train_df) + len(val_df) == len(df), \
    "Erro: a quantidade total de registros mudou!" #verifica se a soma dos registros dos dois conjuntos corresponde a quantidade original

assert textos_treino.isdisjoint(textos_val), \
    "Erro: existem textos idênticos em treino e validação!" # o metodo isdisjoint() retorna True quandos os conjuntos nao possuem elementos em comum
#assert nao imprime nada

print(f"\nTextos presentes nos dois conjuntos: {len(textos_comuns)}")
