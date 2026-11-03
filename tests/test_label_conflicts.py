import pandas as pd
from src.label_conflicts import analyze_conflicts

df = pd.DataFrame({
    "text": [
        "fire", "fire", "fire",
        "rain", "rain",
        "sun", "sun",
        "wind"
    ],
    "target": [
        1, 1, 0,
        0, 1,
        0, 0,
        1
    ]
})

def test_analyze_conflicts():
    linhas_conflitantes, resumo = analyze_conflicts(df)

    assert len(resumo) == 2
    assert len(linhas_conflitantes) == 5
    assert resumo["empate"].sum() == 1

    assert set(resumo.index) == {"fire", "rain"}

    assert resumo.loc["fire", 0] == 1
    assert resumo.loc["fire", 1] == 2

    assert resumo.loc["rain", "empate"]