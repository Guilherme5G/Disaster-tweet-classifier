import pandas as pd

from src.data_split import split_by_text

#DataSet sintetico
df = pd.DataFrame({
    "text": [
        "fire near city",
        "fire near city",
        "flood downtown",
        "earthquake today",
        "music tonight",
        "heavy rain"
    ],
    "target": [1, 1, 1, 1, 0, 1]
})

def test_split_keeps_identical_texts_together():
    train_df, val_df = split_by_text(df)

    assert len(train_df) + len(val_df) == len(df), \
        "Erro: a quantidade total de registros mudou!"

    assert set(train_df.index).isdisjoint(set(val_df.index))

    assert set(train_df["text"]).isdisjoint(set(val_df["text"]))
