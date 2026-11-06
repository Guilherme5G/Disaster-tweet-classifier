from pathlib import Path
import gradio as gr
import joblib
from src.inference import classify_tweet

# Locate the trained model relative to app.py.
ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "linear_svm_submission.joblib"

# Load the trained Pipeline once when the app starts.
pipeline = joblib.load(MODEL_PATH)

def predict(text):
    return classify_tweet(text, pipeline)

demo = gr.Interface(
    fn=predict,

    inputs=gr.Textbox(
        label="Enter a tweet",
        placeholder="Type a tweet to classify...",
        lines=4,
    ),

    outputs=gr.Textbox(
        label="Prediction"
    ),

    title="Disaster Tweet Classifier",

    description=(
        "Classify tweets using TF-IDF and Linear SVM. "
        "Educational demonstration — not an emergency detection service."
    ),

    examples=[
        ["Wildfire spreads near downtown. Residents evacuated."],
        ["Happy birthday! Let's celebrate with some music."],
    ],

    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()