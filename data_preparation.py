import pandas as pd


DERIVED_FEATURES = ("Age_Group", "Family_Risk_Count", "Symptom_Burden")
FEATURE_SOURCE_COLUMNS = (
    "Age",
    "Family_History_AS",
    "Family_History_Autoimmune",
    "Morning_Stiffness",
    "Fever",
    "HLA_B27",
    "Functional_Restriction",
)


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the same age filtering, label normalization, and features used for training."""
    df = data.copy()
    missing = [column for column in FEATURE_SOURCE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Missing required input columns: " + ", ".join(missing))

    df = df[df["Age"].between(18, 90)].copy()
    if "Diagnosis" in df.columns:
        df["Diagnosis"] = df["Diagnosis"].replace(
            {"Ankylosing Spondylitis": "Positive", "No": "Negative"}
        )
    df["Functional_Restriction"] = df["Functional_Restriction"].replace(
        {"Yes": "Restricted", "No": "Normal"}
    )

    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[17, 29, 39, 49, 59, 120],
        labels=["18-29", "30-39", "40-49", "50-59", "60+"],
        include_lowest=True,
    ).astype(str)
    df["Family_Risk_Count"] = (
        df["Family_History_AS"].eq("Yes").astype(int)
        + df["Family_History_Autoimmune"].eq("Yes").astype(int)
    )
    df["Symptom_Burden"] = (
        df["Morning_Stiffness"].eq("Yes").astype(int)
        + df["Fever"].eq("Yes").astype(int)
        + df["HLA_B27"].eq("Yes").astype(int)
        + df["Functional_Restriction"].isin(["Limited", "Restricted"]).astype(int)
    )
    return df
