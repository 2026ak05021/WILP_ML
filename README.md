# Ankylosing Spondylitis Classification Model Comparison

## a. Problem statement
Build and compare five supervised classification models using the supplied ankylosing spondylitis dataset. The Streamlit application evaluates previously saved models on uploaded test data without retraining.

## b. Dataset description
The supplied spreadsheet contains 921 records and 10 original predictor columns plus the Diagnosis target. During cleaning, two implausible ages were removed, diagnostic labels were standardized to Positive and Negative, and Functional_Restriction values were standardized. Three transparent derived predictors, Age_Group, Family_Risk_Count, and Symptom_Burden, were added. The modelling dataset therefore contains 919 records and 13 predictor columns. Categorical predictors are one-hot encoded inside each model pipeline.

This dataset is used only for an academic machine-learning exercise and must not be interpreted as a clinical diagnostic system.

## c. GitHub Repository Link
GitHub repository: [https://github.com/2026ak05021/WILP_ML/tree/master](https://github.com/2026ak05021/WILP_ML/tree/master)

Live Streamlit app: [https://2026ak05021-wilp-ml-app-igbbfe.streamlit.app/](https://2026ak05021-wilp-ml-app-igbbfe.streamlit.app/)

## d. Models used

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.7935 | 0.8421 |      0.7803 |   0.9196 | 0.8443 | 0.5602 |
| Decision Tree          |     0.788  | 0.8879 |      0.7664 |   0.9375 | 0.8434 | 0.5518 |
| KNN                    |     0.8424 | 0.8938 |      0.8029 |   0.9821 | 0.8835 | 0.6794 |
| Naive Bayes            |     0.5    | 0.8362 |      1      |   0.1786 | 0.303  | 0.28   |
| Support Vector Machine |     0.8533 | 0.8951 |      0.8571 |   0.9107 | 0.8831 | 0.6888 |

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Provides a clear linear baseline after numeric scaling and one-hot encoding of categorical fields. |
| Decision Tree | Captures non-linear rules and is easy to inspect, but performance depends on depth and leaf-size controls. |
| KNN | Uses distances between encoded and scaled records. Its performance is sensitive to the selected number of neighbours. |
| Naive Bayes | Trains quickly and provides a useful probabilistic baseline, although its independence assumption is restrictive. |
| Support Vector Machine | Models a non-linear decision boundary after preprocessing and produces probability estimates for AUC calculation. |
| Overall winner | Support Vector Machine produced the strongest combined MCC, AUC and F1 result on the fixed test split. |

## Repository contents
- `app.py`: Streamlit application
- `data_preparation.py`: shared label normalization and feature engineering used by training and the app
- `train_models.py`: cleaning, feature engineering, training and evaluation
- `requirements.txt`: deployment dependencies
- `deneme_cleaned_english.csv`: supplied source spreadsheet in CSV form
- `cleaned_dataset.csv`: cleaned modelling dataset
- `test_data.csv`: held-out test data with target
- `model/`: five saved pipelines and metadata
- `model_metrics.csv`: comparison metrics

## Run locally
1. Create a virtual environment with `python -m venv .venv`.
2. Install dependencies with `.venv\Scripts\python.exe -m pip install -r requirements.txt`.
3. Run `.venv\Scripts\python.exe train_models.py` to regenerate the saved models and test data.
4. Start the app with `.venv\Scripts\python.exe -m streamlit run app.py`.
5. Upload `test_data.csv` and select a model, then click **Evaluate model** to display the results. The model and upload controls are in a form, so changing either one requires clicking the button to submit the new selection.

The app displays Accuracy, AUC, Precision, Recall, F1, MCC, a confusion matrix, and a classification report. It loads a saved model and evaluates the uploaded labeled test data; it does not retrain on uploaded data. Raw source CSVs with the original input columns are also accepted; the app generates the engineered features used during training and excludes rows outside the training age range of 18-90.

## Before submission
- Include the required screenshot of the assignment running in BITS Virtual Lab in the submission PDF. This screenshot must be captured in the lab; it cannot be produced by running the app locally.
- Include this complete README content in the submission PDF as required by the assignment.
- Add the original public dataset source citation if it is available from the assignment materials; the supplied project files do not identify that source.
