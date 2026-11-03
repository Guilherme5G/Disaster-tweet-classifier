from pathlib import Path
import pandas as pd

from src.data_split import split_by_text

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"


def analyze_conflicts(train_df):
    targets_por_texto = train_df.groupby("text")["target"].nunique()
    textos_conflitantes = targets_por_texto[targets_por_texto > 1].index

    linhas_conflitantes = train_df[train_df["text"].isin(textos_conflitantes)]

    resumo = (
        linhas_conflitantes
        .groupby(["text", "target"])
        .size()
        .unstack(fill_value=0)
    )

    resumo = resumo.reindex(columns=[0, 1], fill_value=0)
    resumo["total"] = resumo[0] + resumo[1]
    resumo["empate"] = resumo[0] == resumo[1]

    return linhas_conflitantes, resumo


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    train_df, val_df = split_by_text(df)

    linhas_conflitantes, resumo = analyze_conflicts(train_df)

    print(f"Textos distintos com conflito: {len(resumo)}")
    print(f"Linhas associadas a esses textos: {len(linhas_conflitantes)}")

    for texto in resumo.index[:3]:
        print(f"\nTexto: {texto}")
        print(train_df[train_df["text"] == texto]["target"].value_counts())

    print("\nResumo dos conflitos:")
    print(resumo[[0, 1, "total", "empate"]].to_string())

    print("\nQuantidade de empates:")
    print(resumo["empate"].sum())