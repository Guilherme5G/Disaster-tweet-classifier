import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "linear_svm_submission.joblib"
OUTPUT_PATH = ROOT / "web" / "model.json"


def export_model():
    pipeline = joblib.load(MODEL_PATH)

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    assert isinstance(vectorizer, TfidfVectorizer)
    assert isinstance(classifier, LinearSVC)

    # This exporter supports our standard unigram TF-IDF.
    assert vectorizer.analyzer == "word"
    assert vectorizer.lowercase is True
    assert vectorizer.token_pattern == r"(?u)\b\w\w+\b"
    assert vectorizer.ngram_range == (1, 1)
    assert vectorizer.strip_accents is None
    assert vectorizer.stop_words is None
    assert vectorizer.preprocessor is None
    assert vectorizer.tokenizer is None
    assert vectorizer.norm == "l2"
    assert vectorizer.use_idf is True
    assert vectorizer.binary is False
    assert vectorizer.sublinear_tf is False
    assert classifier.classes_.tolist() == [0, 1]

    terms = {
        term: [
            float(vectorizer.idf_[idx]),
            float(classifier.coef_[0, idx]),
        ]
        for term, idx in vectorizer.vocabulary_.items()
    }

    examples = [
        "Wildfire spreads near downtown. Residents evacuated.",
        "Happy birthday! Let's celebrate with some music.",
        "Fire fire fire near the school!",
        "Heavy rainfall floods city streets.",
        "Enjoying a beautiful sunny afternoon.",
        "Earthquake reported near the coast.",
    ]

    predictions = pipeline.predict(examples)

    payload = {
        "intercept": float(classifier.intercept_[0]),
        "terms": terms,
        "self_tests": [
            {"text": text, "expected": int(pred)}
            for text, pred in zip(examples, predictions)
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    print("Export completed:", OUTPUT_PATH)
    print("Vocabulary size:", len(terms))
    print("File size (MB):", round(OUTPUT_PATH.stat().st_size / 1024**2, 2))
    print("Reference tests:", len(payload["self_tests"]))


if __name__ == "__main__":
    export_model()