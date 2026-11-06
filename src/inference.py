def classify_tweet(text, pipeline):
    if not isinstance(text, str) or not text.strip():
        return "Please enter a tweet."

    cleaned_text = text.strip()
    prediction = int(pipeline.predict([cleaned_text])[0])

    return "Disaster (1)" if prediction == 1 else "Not Disaster (0)"