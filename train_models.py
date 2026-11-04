import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_preparation import TARGET_COLUMN, TARGET_LABELS, prepare_labeled_data


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "student_depression_dataset.csv"
MODEL_DIR = ROOT / "model"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def main() -> None:
    data = prepare_labeled_data(pd.read_csv(DATA_PATH))
    features = [column for column in data.columns if column != TARGET_COLUMN]
    X = data[features]
    y = data[TARGET_COLUMN]

    if set(y.unique()) != set(TARGET_LABELS):
        raise ValueError("Training data must include both Depression labels: Yes and No.")

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [
        column for column in features if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=4, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE
        ),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    metric_rows = []
    model_files = {}

    for name, classifier in models.items():
        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", classifier)]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        positive_probabilities = pipeline.predict_proba(X_test)[
            :, list(pipeline.classes_).index("Yes")
        ]

        metric_rows.append(
            {
                "ML Model Name": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "AUC": roc_auc_score(
                    (y_test == "Yes").astype(int), positive_probabilities
                ),
                "Precision": precision_score(
                    y_test, predictions, pos_label="Yes", zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predictions, pos_label="Yes", zero_division=0
                ),
                "F1": f1_score(y_test, predictions, pos_label="Yes", zero_division=0),
                "MCC": matthews_corrcoef(y_test, predictions),
            }
        )

        filename = name.lower().replace(" ", "_").replace("-", "") + ".joblib"
        model_files[name] = filename
        joblib.dump(pipeline, MODEL_DIR / filename)

    pd.DataFrame(metric_rows).to_csv(ROOT / "model_metrics.csv", index=False)
    data.loc[X_test.index].to_csv(ROOT / "test_data.csv", index=False)
    joblib.dump(features, MODEL_DIR / "feature_names.joblib")
    (MODEL_DIR / "model_files.json").write_text(
        json.dumps(model_files, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(metric_rows).round(4).to_string(index=False))
    print(f"\nTraining records: {len(X_train)}; test records: {len(X_test)}")
    print(f"Predictor columns: {len(features)}")


if __name__ == "__main__":
    main()
