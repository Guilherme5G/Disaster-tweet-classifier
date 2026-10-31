from src.text_pipeline import vectorize_text

train_texts = [
    "fire near school",
    "flood near town",
    "fire downtown"
]

val_texts = [
    "earthquake near town"
]

def test_dimensions_match():
    X_train, X_val, vectorizer = vectorize_text(train_texts, val_texts)

    assert X_train.shape[1] == X_val.shape[1], \
        "Erro: a quantidade de colunas é diferente!"


def test_vocabulary_restricted_to_train():
    X_train, X_val, vectorizer = vectorize_text(train_texts, val_texts)

    assert "earthquake" not in vectorizer.vocabulary_

def test_vocabulary_unchanged_after_validation():
    X_train, X_val, vectorizer = vectorize_text(train_texts, val_texts)
    vocab_antes = dict(vectorizer.vocabulary_)
    vectorizer.transform(val_texts)
    assert vectorizer.vocabulary_ == vocab_antes
    assert X_val.nnz == 2
    assert set(vectorizer.vocabulary_.keys()) == {"fire", "near", "school", "flood", "town", "downtown"}