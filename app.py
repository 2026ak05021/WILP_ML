import json
from pathlib import Path

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

from data_preparation import TARGET_COLUMN, TARGET_LABELS, prepare_labeled_data


MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_FILES = json.loads(
    (MODEL_DIR / "model_files.json").read_text(encoding="utf-8")
)
FEATURES = joblib.load(MODEL_DIR / "feature_names.joblib")

st.set_page_config(
    page_title="Student Depression Model Comparison",
    layout="wide",
)
st.title("Student Depression Classification Model Comparison")
st.caption(
    "Academic comparison on a labeled dataset. This model is not a mental-health "
    "assessment or diagnostic tool."
)

with st.form("evaluation_form"):
    model_name = st.selectbox("Select classifier", list(MODEL_FILES))
    uploaded_file = st.file_uploader("Upload labeled test data (CSV)", type=["csv"])
    submitted = st.form_submit_button("Evaluate model")

if not submitted:
    st.info("Upload test_data.csv, select a model, and click Evaluate model.")
elif uploaded_file is None:
    st.error("Upload a labeled CSV containing the predictor columns and Depression target.")
else:
    try:
        uploaded_data = pd.read_csv(uploaded_file)
        evaluation_data = prepare_labeled_data(uploaded_data, FEATURES)
        X = evaluation_data[FEATURES]
        y = evaluation_data[TARGET_COLUMN]

        if set(y.unique()) != set(TARGET_LABELS):
            st.error("The uploaded test CSV must contain both Depression labels: Yes and No.")
        else:
            model = joblib.load(MODEL_DIR / MODEL_FILES[model_name])
            predictions = model.predict(X)
            positive_probabilities = model.predict_proba(X)[
                :, list(model.classes_).index("Yes")
            ]

            metrics = {
                "Accuracy": accuracy_score(y, predictions),
                "AUC": roc_auc_score(
                    (y == "Yes").astype(int), positive_probabilities
                ),
                "Precision": precision_score(
                    y, predictions, pos_label="Yes", zero_division=0
                ),
                "Recall": recall_score(
                    y, predictions, pos_label="Yes", zero_division=0
                ),
                "F1": f1_score(
                    y, predictions, pos_label="Yes", zero_division=0
                ),
                "MCC": matthews_corrcoef(y, predictions),
            }
            for column, (metric, value) in zip(st.columns(6), metrics.items()):
                column.metric(metric, f"{value:.4f}")

            st.subheader("Confusion Matrix")
            st.dataframe(
                pd.DataFrame(
                    confusion_matrix(y, predictions, labels=list(TARGET_LABELS)),
                    index=["Actual No", "Actual Yes"],
                    columns=["Predicted No", "Predicted Yes"],
                ),
                width="stretch",
            )
            st.subheader("Classification Report")
            st.dataframe(
                pd.DataFrame(
                    classification_report(
                        y,
                        predictions,
                        labels=list(TARGET_LABELS),
                        output_dict=True,
                        zero_division=0,
                    )
                ).T.round(4),
                width="stretch",
            )
    except (ValueError, KeyError, pd.errors.ParserError) as error:
        st.error(f"Unable to evaluate this CSV: {error}")
