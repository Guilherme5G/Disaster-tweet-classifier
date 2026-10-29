from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

print("Interpretador:", sys.executable)

df = pd.read_csv(DATA_PATH)
print("Dataset carregado com sucesso!")

print(df.head(5))
print(df.columns)
print(df.shape)
print(df.dtypes)

'''print("Arquivo atual:", Path(__file__).resolve())
print("Pasta src:", Path(__file__).resolve().parents[0])
print("Raiz do projeto:", ROOT)
print("Caminho do dataset:", DATA_PATH)
print("Arquivo existe?", DATA_PATH.is_file())'''