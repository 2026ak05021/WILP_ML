# Student Depression Classification Model Comparison

**Author:** Sachin Saraswat | **Student ID:** 2026ak05021

## a. Problem statement

This project compares five supervised classification models on a student
depression survey dataset. The Streamlit app lets a user upload labeled test
data, select a saved model, and review its evaluation metrics. The app evaluates
the uploaded labels against model predictions; it does not retrain models or
provide a clinical assessment.

## b. Dataset description

The dataset is the Kaggle [Student Depression Detection
Dataset](https://www.kaggle.com/datasets/mkhoirulmustaqim/student-depression-detection-dataset)
by `mkhoirulmustaqim`. Kaggle describes it as survey data about demographic,
academic, social, economic, and health-related factors associated with student
depression. The provided CSV contains 4,464 records, 12 predictor columns, and
the binary `Depression` target (`Yes`/`No`). There are no missing values in the
supplied file. The CSV is used in English as provided; it was not translated or
relabelled.

The predictors are `Gender`, `Age`, `Family_background`,
`Relationship_status`, `Marital_status`, `Living_style`, `Current_semester`,
`CGPA`, `friend_circle`, `Health_condition`, `Cost_without_study`, and
`Home_division`. These meet the assignment's minimum of 12 features without
adding engineered columns. The target distribution is 3,263 `No` and 1,201
`Yes`.

The dataset is used only for an academic comparison. Its associations and
predictions must not be used to diagnose depression or make decisions about
individual students.

## c. GitHub Repository Link

- GitHub repository: [2026ak05021/WILP_ML](https://github.com/2026ak05021/WILP_ML/tree/master)
- Streamlit Community Cloud app: [Open the app](https://2026ak05021-wilp-ml-app-igbbfe.streamlit.app/)

The live Streamlit URL was verified in BITS Virtual Lab while evaluating the
held-out test data.

## d. Models used

The data was split once into a stratified 80% training set and 20% test set
using `random_state=42` (3,571 training records and 893 test records). The same
split and preprocessing pipeline were used for all five models. Numeric
predictors are median-imputed and standardized; categorical predictors are
most-frequent-imputed and one-hot encoded. Preprocessing is fitted within each
model pipeline using training data only.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9709 | 0.9931 | 0.9612 | 0.9292 | 0.9449 | 0.9254 |
| Decision Tree | 0.9798 | 0.9928 | 1.0000 | 0.9250 | 0.9610 | 0.9488 |
| K-Nearest Neighbors | 0.9485 | 0.9778 | 0.9292 | 0.8750 | 0.9013 | 0.8672 |
| Gaussian Naive Bayes | 0.9496 | 0.9875 | 0.8884 | 0.9292 | 0.9084 | 0.8740 |
| Support Vector Machine | 0.9877 | 0.9965 | 0.9872 | 0.9667 | 0.9768 | 0.9686 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Provides a linear baseline and achieved 0.9709 accuracy on this held-out split. |
| Decision Tree | Achieved perfect precision (1.0000) with 0.9250 recall; this does not establish that it will generalize equally well to new data. |
| K-Nearest Neighbors | Had the lowest accuracy (0.9485) among the five on this split, while still achieving 0.8750 recall. |
| Gaussian Naive Bayes | Achieved 0.9496 accuracy and 0.9292 recall; its conditional-independence assumption differs from the other models. |
| Support Vector Machine | Achieved the highest accuracy (0.9877), AUC (0.9965), F1 (0.9768), and MCC (0.9686) on this split. |
| Overall winner | Support Vector Machine had the strongest results across these metrics on this single held-out split. Further validation on independent data would be needed before making broader claims. |

## Repository contents

- `app.py`: Streamlit CSV evaluation application
- `data_preparation.py`: target-label validation and input-column checks
- `train_models.py`: train/test split, preprocessing, model training, and metric calculation
- `requirements.txt`: pinned application dependencies
- `student_depression_dataset.csv`: Kaggle dataset used for this assignment
- `test_data.csv`: held-out test rows with all predictors and the `Depression` target
- `model/`: five saved pipelines and model/feature metadata
- `model_metrics.csv`: metrics measured on the held-out test split

## Streamlit app features

The app has a labeled CSV uploader and model-selection dropdown. After the user
uploads `test_data.csv` and clicks **Evaluate model**, it displays Accuracy,
AUC, Precision, Recall, F1, MCC, a confusion matrix, and a classification
report. It loads the selected saved pipeline and does not retrain it on
uploaded data. The CSV must contain all 12 predictor columns and the
`Depression` target with both `Yes` and `No` labels.

## Run locally

1. Create a virtual environment with `python -m venv .venv`.
2. Activate it: on Linux/macOS, run `source .venv/bin/activate`; on Windows, run `.venv\Scripts\activate`.
3. Install dependencies with `python -m pip install -r requirements.txt`.
4. Regenerate models, metrics, and test data with `python train_models.py`.
5. Start the app with `python -m streamlit run app.py`.
6. Upload `test_data.csv`, select a model, and click **Evaluate model**.

## Acknowledgment

AI assistance was used to review and revise parts of the data-preparation
checks, model-training pipeline, Streamlit app, and README. The student should
review and understand the submitted work and follow the course rules for
acknowledging AI assistance.

## Submission PDF checklist

Prepare one PDF in the assignment's required order:

1. Clickable GitHub repository link.
2. Clickable Streamlit Community Cloud app link, after the updated app has
   been deployed and verified.
3. Screenshot evidence of the assignment running in BITS Virtual Lab. The
   screenshots must be captured in the lab. This PDF includes a metrics view
   and a confusion-matrix/classification-report view for each of the five
   models.
4. This complete README content.
