from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_preparation import engineer_features

MODEL_DIR = Path(__file__).resolve().parent / "model"

st.set_page_config(
    page_title="Ankylosing Spondylitis Classification",
    layout="wide",
)
st.title("Ankylosing Spondylitis Classification Model Comparison")
st.caption(
    "Educational model comparison using the supplied assignment dataset. "
    "This application is not a clinical diagnostic tool."
)

files = json.loads((MODEL_DIR / "model_files.json").read_text(encoding="utf-8"))
with st.form("evaluation_form"):
    name = st.selectbox("Select model", list(files))
    upload = st.file_uploader("Upload test CSV", type=["csv"])
    submitted = st.form_submit_button("Evaluate model")

if not submitted:
    st.info("Select a model and upload a test CSV, then click Evaluate model.")
elif upload is None:
    st.info(
        "Upload test_data.csv from the repository. Raw source CSVs are also "
        "accepted; engineered features are created automatically."
    )
else:
    try:
        uploaded_df = pd.read_csv(upload)
        df = engineer_features(uploaded_df)
        excluded_rows = len(uploaded_df) - len(df)

        if excluded_rows:
            st.warning(
                f"Excluded {excluded_rows} row(s) with ages outside the "
                "training range of 18-90."
            )
        if df.empty:
            st.error("No rows remain after applying the training age range of 18–90.")
            st.stop()

        features = joblib.load(MODEL_DIR / "feature_names.joblib")
        missing = [column for column in features + ["Diagnosis"] if column not in df]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            model = joblib.load(MODEL_DIR / files[name])
            X = df[features]
            y = df["Diagnosis"]
            pred = model.predict(X)
            prob = model.predict_proba(X)[
                :, list(model.classes_).index("Positive")
            ]

            vals = {
                "Accuracy": accuracy_score(y, pred),
                "AUC": roc_auc_score((y == "Positive").astype(int), prob),
                "Precision": precision_score(
                    y, pred, pos_label="Positive", zero_division=0
                ),
                "Recall": recall_score(y, pred, pos_label="Positive", zero_division=0),
                "F1": f1_score(y, pred, pos_label="Positive", zero_division=0),
                "MCC": matthews_corrcoef(y, pred),
            }
            for column, (metric, value) in zip(st.columns(6), vals.items()):
                column.metric(metric, f"{value:.4f}")

            st.subheader("Confusion Matrix")
            st.dataframe(
                pd.DataFrame(
                    confusion_matrix(y, pred, labels=["Negative", "Positive"]),
                    index=["Actual Negative", "Actual Positive"],
                    columns=["Predicted Negative", "Predicted Positive"],
                ),
                width="stretch",
            )
            st.subheader("Classification Report")
            st.dataframe(
                pd.DataFrame(
                    classification_report(
                        y, pred, output_dict=True, zero_division=0
                    )
                ).T.round(4),
                width="stretch",
            )
    except Exception as error:
        st.error(f"Unable to evaluate file: {error}")
