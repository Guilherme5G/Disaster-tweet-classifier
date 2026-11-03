import numpy as np
from src.final_pipeline import build_final_pipeline, save_pipeline, load_pipeline

def test_pipeline_persistence(tmp_path):
    model_path = tmp_path / "test_model.joblib"
    train_texts = [
        "fire evacuation alert",
        "music concert party",
        "flood rescue emergency",
        "happy sunny morning"
    ]
    train_labels = [1, 0, 1, 0]
    test_texts = ["fire emergency", "music party"]

    pipeline = build_final_pipeline()
    pipeline.fit(train_texts, train_labels)

    predictions_before = pipeline.predict(test_texts)

    save_pipeline(pipeline, model_path)
    loaded_pipeline = load_pipeline(model_path)

    predictions_after = loaded_pipeline.predict(test_texts)

    np.testing.assert_array_equal(predictions_before, predictions_after)

def test_loaded_pipeline_inference(tmp_path):
    model_path = tmp_path / "test_model.joblib"
    train_texts = [
        "fire evacuation alert",
        "music concert party",
        "flood rescue emergency",
        "happy sunny morning"
    ]
    train_labels = [1, 0, 1, 0]
    test_texts = ["fire emergency", "music party"]

    pipeline = build_final_pipeline()
    pipeline.fit(train_texts, train_labels)

    predictions_before = pipeline.predict(test_texts)

    save_pipeline(pipeline, model_path)
    loaded_pipeline = load_pipeline(model_path)

    assert "tfidf" in loaded_pipeline.named_steps
    assert "classifier" in loaded_pipeline.named_steps

    new_texts = ["fire smoke downtown", "sunny music festival", "flood warning"]
    preds = loaded_pipeline.predict(new_texts)

    assert len(preds) == len(new_texts)
    assert set(preds).issubset({0, 1})
