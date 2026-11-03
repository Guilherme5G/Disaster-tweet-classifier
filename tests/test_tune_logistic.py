from sklearn.model_selection import GroupKFold
from src.tune_logistic import tune_logistic


def test_group_kfold_integrity():
    texts = [
        "fire near city",
        "fire near city",
        "flood downtown",
        "flood downtown",
        "music tonight",
        "sunny day",
    ]
    labels = [1, 1, 1, 1, 0, 0]
    groups = texts

    cv = GroupKFold(n_splits=3)

    for train_idx, val_idx in cv.split(texts, labels, groups=groups):
        train_groups = {groups[i] for i in train_idx}
        val_groups = {groups[i] for i in val_idx}
        assert train_groups.isdisjoint(val_groups)


def test_tune_logistic_search():
    texts = [
        "fire near school",
        "fire near school",
        "music festival tonight",
        "flood in downtown",
        "sunny calm morning",
        "earthquake alert today",
        "happy birthday party",
    ]
    labels = [1, 1, 0, 1, 0, 1, 0]
    groups = texts

    search = tune_logistic(texts=texts, labels=labels, groups=groups)

    # 1. search.best_params_ contém "classifier__C"
    assert "classifier__C" in search.best_params_

    # 2. O C selecionado pertence a [0.1, 1.0, 10.0]
    assert search.best_params_["classifier__C"] in [0.1, 1.0, 10.0]

    # 3. search.best_score_ está entre 0 e 1
    assert 0.0 <= search.best_score_ <= 1.0

    # 4. search.best_estimator_ possui as etapas tfidf e classifier
    assert "tfidf" in search.best_estimator_.named_steps
    assert "classifier" in search.best_estimator_.named_steps