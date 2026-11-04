import pandas as pd


TARGET_COLUMN = "Depression"
TARGET_LABELS = ("No", "Yes")


def prepare_labeled_data(
    data: pd.DataFrame, feature_columns: list[str] | None = None
) -> pd.DataFrame:
    """Validate labels and return the target with the requested input columns."""
    df = data.copy()
    df.columns = df.columns.str.strip()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    labels = df[TARGET_COLUMN].astype("string").str.strip().str.title()
    invalid_labels = sorted(set(labels.dropna()) - set(TARGET_LABELS))
    if labels.isna().any() or invalid_labels:
        raise ValueError(
            f"{TARGET_COLUMN} must contain only 'Yes' or 'No'; "
            f"invalid labels: {invalid_labels}"
        )
    df[TARGET_COLUMN] = labels

    if feature_columns is not None:
        missing = [column for column in feature_columns if column not in df.columns]
        if missing:
            raise ValueError("Missing required input columns: " + ", ".join(missing))
        df = df[feature_columns + [TARGET_COLUMN]]

    if df.empty:
        raise ValueError("The uploaded CSV contains no data rows.")
    return df
