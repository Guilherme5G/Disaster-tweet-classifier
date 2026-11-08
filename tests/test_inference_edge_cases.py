import pytest
from src.inference import classify_tweet

class RecordingPipeline:
    """Records inference calls without loading a real ML model."""

    def __init__(self, prediction=0):
        self.prediction = prediction
        self.calls = []

    def predict(self, texts):
        self.calls.append(texts)
        return [self.prediction]


@pytest.mark.parametrize(
    "invalid_text",
    [
        "",
        " ",
        "\n\t",
        None,
        123,
        [],
    ],
)
def test_invalid_input_does_not_call_model(invalid_text):
    pipeline = RecordingPipeline()

    result = classify_tweet(invalid_text, pipeline)

    assert result == "Please enter a tweet."
    assert pipeline.calls == []


def test_long_text_is_processed_once():
    pipeline = RecordingPipeline(prediction=1)
    long_text = "Wildfire spreads near downtown." * 500

    result = classify_tweet(long_text, pipeline)

    assert result == "Disaster (1)"
    assert pipeline.calls == [[long_text.strip()]]


def test_surrounding_whitespace_is_removed():
    pipeline = RecordingPipeline(prediction=0)

    result = classify_tweet(
        "  Happy birthday!  ",
        pipeline,
    )

    assert result == "Not Disaster (0)"
    assert pipeline.calls == [["Happy birthday!"]]


def test_unicode_text_is_preserved():
    pipeline = RecordingPipeline(prediction=0)
    text = "Earthquake near São Paulo 🌎"

    result = classify_tweet(text, pipeline)

    assert result == "Not Disaster (0)"
    assert pipeline.calls == [[text]]
