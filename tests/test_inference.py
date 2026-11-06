from src.inference import classify_tweet

class FakePipeline:
    def predict(self, texts):
        return [
            1 if "fire" in text.lower() else 0
            for text in texts
        ]

def test_classify_tweet_positive():
    pipeline = FakePipeline()
    result = classify_tweet(" Wildfire near downtown ", pipeline)
    assert result == "Disaster (1)"

def test_classify_tweet_negative():
    pipeline = FakePipeline()
    result = classify_tweet("Happy birthday party with music", pipeline)
    assert result == "Not Disaster (0)"

def test_classify_tweet_empty():
    pipeline = FakePipeline()
    assert classify_tweet("", pipeline) == "Please enter a tweet."
    assert classify_tweet(" ", pipeline) == "Please enter a tweet."