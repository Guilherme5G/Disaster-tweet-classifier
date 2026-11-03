from src.train_logistic import train_logistic_model

def test_train_logistic():
    X_train = [
        [1.0, 0.0],
        [0.8, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ]
    y_train = [1, 1, 0, 0]
    X_val = [
        [0.9, 0.1],
        [0.1, 0.8],
    ]
    model = train_logistic_model(X_train, y_train)
    y_pred = model.predict(X_val)

    assert hasattr(model, "coef_")
    assert len(y_pred) == len(X_val)
    assert set(y_pred).issubset({0, 1})
