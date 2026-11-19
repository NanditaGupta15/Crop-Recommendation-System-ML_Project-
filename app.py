"""
Crop Recommendation System - Streamlit app.

Run locally:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

MODEL_PATH = Path(__file__).parent / "crop_pipeline.joblib"

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="centered")

# Hard limits only stop impossible entries (e.g. humidity > 100%).
# Values outside the TRAINING range are still allowed, but trigger a warning.
FIELDS = {
    "N":           dict(label="Nitrogen (N)",    lo=0.0,   hi=300.0, step=1.0, help="Nitrogen content in soil"),
    "P":           dict(label="Phosphorus (P)",  lo=0.0,   hi=300.0, step=1.0, help="Phosphorus content in soil"),
    "K":           dict(label="Potassium (K)",   lo=0.0,   hi=300.0, step=1.0, help="Potassium content in soil"),
    "temperature": dict(label="Temperature (°C)", lo=-10.0, hi=60.0,  step=0.5, help="Average temperature in °C"),
    "humidity":    dict(label="Humidity (%)",    lo=0.0,   hi=100.0, step=0.5, help="Relative humidity in %"),
    "ph":          dict(label="Soil pH",         lo=0.0,   hi=14.0,  step=0.1, help="Soil pH value (0-14)"),
    "rainfall":    dict(label="Rainfall (mm)",   lo=0.0,   hi=500.0, step=1.0, help="Rainfall in mm"),
}


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Model file 'crop_pipeline.joblib' not found. Run `python build_pipeline.py` first.")
    st.stop()

bundle = load_bundle()
pipeline = bundle["pipeline"]
FEATURES = bundle["features"]
ranges = bundle["ranges"]
defaults = bundle["defaults"]

st.title("🌾 Crop Recommendation System")
st.write(
    "Enter your soil nutrient levels and local climate conditions to get the "
    "top 3 most suitable crops."
)

if bundle.get("sklearn_version") != sklearn.__version__:
    st.warning(
        f"Model was trained with scikit-learn {bundle.get('sklearn_version')} but this "
        f"environment uses {sklearn.__version__}. Retrain with `python build_pipeline.py` "
        f"or pin the same version in requirements.txt."
    )

# ---- Input form (the model only runs when the button is pressed) -------
with st.form("crop_form"):
    values = {}
    left, right = st.columns(2)
    for i, feat in enumerate(FEATURES):
        cfg = FIELDS[feat]
        col = left if i % 2 == 0 else right
        start = min(max(float(defaults[feat]), cfg["lo"]), cfg["hi"])
        values[feat] = col.number_input(
            cfg["label"],
            min_value=cfg["lo"],
            max_value=cfg["hi"],
            value=float(start),
            step=cfg["step"],
            help=cfg["help"],
        )
    submitted = st.form_submit_button("Recommend crop", use_container_width=True)

# ---- Prediction ---------------------------------------------------------
if submitted:
    sample = pd.DataFrame([values], columns=FEATURES)

    # Warn when an input is outside what the model saw during training
    out_of_range = [
        f"**{FIELDS[f]['label']}** = {values[f]:g} (training range: {lo:g} to {hi:g})"
        for f, (lo, hi) in ranges.items()
        if not (lo <= values[f] <= hi)
    ]
    if out_of_range:
        st.warning(
            "Some inputs are outside the range the model was trained on, "
            "so the recommendation may be unreliable:\n\n- " + "\n- ".join(out_of_range)
        )

    model = pipeline.named_steps["model"]
    proba = pipeline.predict_proba(sample)[0]
    top_idx = proba.argsort()[::-1][:3]

    result = pd.DataFrame(
        {
            "Crop": [str(model.classes_[i]).title() for i in top_idx],
            "Confidence (%)": [round(float(proba[i]) * 100, 1) for i in top_idx],
        }
    )

    st.success(f"Recommended crop: **{result.loc[0, 'Crop']}**")
    st.subheader("Top 3 recommendations")
    st.dataframe(result, hide_index=True, use_container_width=True)
    st.bar_chart(result.set_index("Crop")["Confidence (%)"])
    st.caption(
        "Confidence is the share of Random Forest trees voting for each crop. It is useful "
        "for ranking crops but is not a calibrated probability."
    )

st.divider()
st.caption(
    "Trained on a clean, idealized dataset (22 crops, 7 features), so real-world accuracy "
    "may be lower. Treat results as decision support, not a guarantee."
)
