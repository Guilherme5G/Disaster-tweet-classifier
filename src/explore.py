from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

print("Interpretador:", sys.executable)

df = pd.read_csv(DATA_PATH)
print("Dataset carregado com sucesso!")

'''print(df.head(5))
print(df.columns)
print(df.shape)
print(df.dtypes)
'''
'''print("Arquivo atual:", Path(__file__).resolve())
print("Pasta src:", Path(__file__).resolve().parents[0])
print("Raiz do projeto:", ROOT)
print("Caminho do dataset:", DATA_PATH)
print("Arquivo existe?", DATA_PATH.is_file())'''


ausentes = df.isna().sum()
print("Quantidade de ausentes por coluna:\n", ausentes)

percentual = (df.isna().sum() / df.shape[0]) * 100
print("\nPercentual de ausentes por coluna:\n", percentual)

coluna_maior_percentual = percentual.idxmax()
print(f"\nA coluna com o maior percentual é: {coluna_maior_percentual}")

print("=====================================")

duplicadas_totais = df.duplicated().sum()
print(f"Linhas completamente duplicadas: {duplicadas_totais}")

textos_repetidos_qtd = df.duplicated(subset=["text"]).sum()
print(f"Linhas com text repetido: {textos_repetidos_qtd}")

mascara_repetidos = df.duplicated(subset=["text"], keep=False)
exemplos = df[mascara_repetidos]

print("\nExemplos de mensagens repetidas (completas):")
print(exemplos[["id", "text", "target"]]
      .sort_values("text")
      .head(10)
      .to_string(index=False))


alvo_por_texto = df.groupby("text")["target"].nunique()
conflitos_qtd = (alvo_por_texto > 1).sum()
print(f"Quantidade de textos com targets diferentes (conflitos): {conflitos_qtd}")



contagem = df["target"].value_counts()
print("quantidade por classe:\n", contagem)

percentual = df["target"].value_counts(normalize=True) * 100
print("\nPorcentagem por classe (%):\n", percentual)

textos_vazios_qtd = (df["text"].str.strip() == "").sum()
print(f"Textos vazios ou só com espaços: {textos_vazios_qtd}")

ids_duplicados_qtd = df.duplicated(subset=["id"]).sum()
print(f"IDs duplicados: {ids_duplicados_qtd}")

print(df.shape)