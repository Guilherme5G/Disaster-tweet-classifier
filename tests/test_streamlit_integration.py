from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = (
    ROOT / "streamlit_app.py"
)

def test_streamlit_app_loads():
    at = AppTest.from_file(str(APP_PATH))
    at.run()
    assert len(at.exception) == 0
    assert len(at.text_area) == 1
    assert len(at.form_submit_button) == 1


def test_streamlit_disaster_prediction():
    at = AppTest.from_file(str(APP_PATH))
    at.run()

    at.text_area[0].set_value("Wildfire spreads near downtown. Residents evacuated.")
    at.form_submit_button[0].click().run()

    assert len(at.exception) == 0

    mkdown_values = [m.value for m in at.markdown]
    assert any("Disaster-related" in text for text in mkdown_values)