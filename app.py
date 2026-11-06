import gradio as gr

from src.final_pipeline import load_pipeline
from src.inference import classify_tweet
from src.train_submission import SUBMISSION_MODEL_PATH

# Load the trained model once when the app starts.
pipeline = load_pipeline(SUBMISSION_MODEL_PATH)


def predict(text):
    return classify_tweet(text, pipeline)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Enter a tweet",
        placeholder="Type a tweet to classify...",
        lines=4,
    ),
    outputs=gr.Textbox(label="Prediction"),
    title="Disaster Tweet Classifier",
    description=(
        "Classify tweets using TF-IDF and Linear SVM. "
        "Educational demonstration — not an emergency detection service."
    ),
    examples=[
        ["Wildfire spreads near downtown. Residents evacuated."],
        ["Happy birthday! Let's celebrate with some music."],
    ],
)

if __name__ == "__main__":
    demo.launch()