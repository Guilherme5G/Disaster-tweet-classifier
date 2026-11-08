
from pathlib import Path

import joblib
import streamlit as st

from src.inference import classify_tweet


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "linear_svm_submission.joblib"


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="Disaster Tweet Classifier",
    layout="centered",
)

# Small, deliberate styling adjustments.
st.markdown(
    """
    <style>
        .block-container {
            max-width: 760px;
            padding-top: 3.5rem;
            padding-bottom: 3rem;
        }

        h1 {
            font-size: 2.1rem !important;
            font-weight: 650 !important;
            letter-spacing: -0.035em;
            line-height: 1.2;
        }

        .result-panel {
            margin-top: 1.5rem;
            padding: 20px 22px;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 10px;
            background: rgba(128, 128, 128, 0.045);
        }

        .result-label {
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            opacity: 0.65;
            margin-bottom: 8px;
        }

        .result-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .result-description {
            font-size: 0.875rem;
            opacity: 0.7;
            line-height: 1.6;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("Disaster Tweet Classifier")

st.write(
    "Analyze tweet text using a TF-IDF and Linear SVM classifier."
)

st.divider()


try:
    pipeline = load_model()
except (FileNotFoundError, OSError) as error:
    st.error(f"Unable to load model: {error}")
    st.stop()


with st.form("classifier_form"):
    tweet = st.text_area(
        "Tweet text",
        placeholder="Enter a tweet to classify...",
        height=130,
    )

    submitted = st.form_submit_button(
        "Classify tweet",
        type="primary",
    )


if submitted:
    prediction = classify_tweet(tweet, pipeline)

    if prediction == "Please enter a tweet.":
        st.warning("Please enter a tweet before classifying.")

    else:
        is_disaster = prediction == "Disaster (1)"

        title = (
            "Disaster-related"
            if is_disaster
            else "Not disaster-related"
        )

        description = (
            "The model identified language associated with a disaster."
            if is_disaster
            else "The model did not identify disaster-related language."
        )

        class_id = 1 if is_disaster else 0

        st.markdown(
            f"""
            <div class="result-panel">
                <div class="result-label">CLASSIFICATION RESULT</div>
                <div class="result-title">{title}</div>
                <div class="result-description">{description}</div>
                <div class="result-description">
                    Predicted class: {class_id}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.divider()

st.caption(
    "Model: TF-IDF + LinearSVC  ·  "
    "Educational use only. Not an emergency detection service."
)
