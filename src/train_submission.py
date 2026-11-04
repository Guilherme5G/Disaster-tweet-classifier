import pandas as pd

from src.final_pipeline import (
    ROOT,
    DATA_PATH,
    build_final_pipeline,
    save_pipeline,
    load_pipeline,
)

SUBMISSION_MODEL_PATH = (
    ROOT / "models" / "linear_svm_submission.joblib"
)

def train_submission_model(train_df):
    pipeline = build_final_pipeline()
    pipeline.fit(train_df["text"], train_df["target"])
    return pipeline


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    pipeline = train_submission_model(df)

    save_pipeline(pipeline, SUBMISSION_MODEL_PATH)

    loaded = load_pipeline(SUBMISSION_MODEL_PATH)
    preds = loaded.predict(["fire evacuation downtown", "happy birthday party"])

    print(f"Modelo salvo em: {SUBMISSION_MODEL_PATH}")
    print(f"Registros usados no treinamento: {len(df)}")
    print(f"Previsões para dois textos novos: {preds}")