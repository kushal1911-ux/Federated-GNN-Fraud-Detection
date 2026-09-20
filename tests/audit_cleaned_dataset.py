from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# RFGN — CLEANED DATASET AUDIT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ieee_cleaned.csv"
)


# ============================================================
# EXPECTED DATASET VALUES
# ============================================================

EXPECTED_ROWS = 590_540
EXPECTED_COLUMNS = 848

ID_COLUMN = "TransactionID"
TARGET_COLUMN = "isFraud"
TIME_COLUMN = "TransactionDT"


# ============================================================
# HELPERS
# ============================================================

def header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# FILE CHECK
# ============================================================

def check_file():

    header("RFGN CLEANED DATASET AUDIT")

    print()
    print("This script is READ-ONLY.")
    print("The cleaned dataset will NOT be modified.")

    print()
    print("Project root :", PROJECT_ROOT)
    print("Dataset      :", DATA_FILE)

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Cleaned dataset not found:\n{DATA_FILE}"
        )

    size_gb = (
        DATA_FILE.stat().st_size
        / (1024 ** 3)
    )

    print()
    print(
        f"File size   : {size_gb:.3f} GB"
    )

    print(
        "File status : PRESENT"
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    header("LOADING CLEANED DATASET")

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False
    )

    print()
    print("Rows    :", f"{len(df):,}")
    print("Columns :", f"{len(df.columns):,}")

    memory_gb = (
        df.memory_usage(
            deep=True
        ).sum()
        / (1024 ** 3)
    )

    print(
        f"Memory  : {memory_gb:.3f} GB"
    )

    if len(df) != EXPECTED_ROWS:

        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"got {len(df):,}"
        )

    if len(df.columns) != EXPECTED_COLUMNS:

        raise ValueError(
            f"Expected {EXPECTED_COLUMNS} columns, "
            f"got {len(df.columns):,}"
        )

    print()
    print("Dataset dimensions: PASS")

    return df


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

def check_required_columns(df):

    header("REQUIRED COLUMN AUDIT")

    required_columns = [
        ID_COLUMN,
        TARGET_COLUMN,
        TIME_COLUMN
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    for column in required_columns:

        print(
            f"{column:<20} PASS"
        )


# ============================================================
# TRANSACTION ID AUDIT
# ============================================================

def audit_transaction_ids(df):

    header("TRANSACTION ID AUDIT")

    missing_ids = int(
        df[ID_COLUMN].isna().sum()
    )

    duplicate_ids = int(
        df[ID_COLUMN].duplicated().sum()
    )

    unique_ids = int(
        df[ID_COLUMN].nunique()
    )

    print(
        "Missing TransactionIDs :",
        missing_ids
    )

    print(
        "Duplicate TransactionIDs:",
        duplicate_ids
    )

    print(
        "Unique TransactionIDs  :",
        f"{unique_ids:,}"
    )

    if missing_ids != 0:

        raise ValueError(
            "Missing TransactionIDs detected."
        )

    if duplicate_ids != 0:

        raise ValueError(
            "Duplicate TransactionIDs detected."
        )

    if unique_ids != len(df):

        raise ValueError(
            "TransactionID uniqueness failed."
        )

    print()
    print(
        "TransactionID uniqueness: PASS"
    )


# ============================================================
# TARGET AUDIT
# ============================================================

def audit_target(df):

    header("FRAUD TARGET AUDIT")

    if df[TARGET_COLUMN].isna().any():

        raise ValueError(
            "Missing fraud labels detected."
        )

    unique_values = sorted(
        df[TARGET_COLUMN]
        .unique()
        .tolist()
    )

    print(
        "Target values:",
        unique_values
    )

    if unique_values != [0, 1]:

        raise ValueError(
            "Target is not binary."
        )

    legitimate = int(
        (df[TARGET_COLUMN] == 0).sum()
    )

    fraud = int(
        (df[TARGET_COLUMN] == 1).sum()
    )

    print()
    print(
        f"LEGITIMATE : {legitimate:>10,} "
        f"({legitimate / len(df) * 100:.4f}%)"
    )

    print(
        f"FRAUD      : {fraud:>10,} "
        f"({fraud / len(df) * 100:.4f}%)"
    )

    print()
    print(
        "Binary target: PASS"
    )


# ============================================================
# TEMPORAL AUDIT
# ============================================================

def audit_transaction_time(df):

    header("TEMPORAL FIELD AUDIT")

    missing_time = int(
        df[TIME_COLUMN].isna().sum()
    )

    print(
        "Missing TransactionDT:",
        missing_time
    )

    if missing_time != 0:

        raise ValueError(
            "TransactionDT contains missing values."
        )

    min_time = df[TIME_COLUMN].min()
    max_time = df[TIME_COLUMN].max()

    print(
        "Minimum TransactionDT:",
        min_time
    )

    print(
        "Maximum TransactionDT:",
        max_time
    )

    print(
        "Time span:",
        max_time - min_time
    )

    print()
    print(
        "TransactionDT: PASS"
    )


# ============================================================
# TEMPORAL ORDER CHECK
# ============================================================

def audit_temporal_order(df):

    header("CHRONOLOGICAL ORDER ANALYSIS")

    time_values = (
        df[TIME_COLUMN]
        .to_numpy()
    )

    differences = np.diff(
        time_values
    )

    backwards = int(
        (differences < 0).sum()
    )

    equal_timestamps = int(
        (differences == 0).sum()
    )

    forward = int(
        (differences > 0).sum()
    )

    print(
        "Forward time transitions :",
        f"{forward:,}"
    )

    print(
        "Equal timestamps         :",
        f"{equal_timestamps:,}"
    )

    print(
        "Backward transitions     :",
        f"{backwards:,}"
    )

    print()

    if backwards == 0:

        print(
            "Dataset is already chronologically ordered: PASS"
        )

    else:

        print(
            "Dataset is not chronologically ordered."
        )

        print(
            "This is acceptable because Stage 3 "
            "will explicitly sort by TransactionDT."
        )


# ============================================================
# MISSING VALUE AUDIT
# ============================================================

def audit_missing_values(df):

    header("MISSING VALUE AUDIT")

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            ID_COLUMN,
            TARGET_COLUMN,
            TIME_COLUMN
        ]
    ]

    missing_cells = int(
        df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    print(
        "Remaining feature missing cells:",
        f"{missing_cells:,}"
    )

    if missing_cells != 0:

        raise ValueError(
            "Missing feature values remain."
        )

    print()
    print(
        "Feature missing values: PASS"
    )


# ============================================================
# INFINITE VALUE AUDIT
# ============================================================

def audit_infinite_values(df):

    header("INFINITE VALUE AUDIT")

    numeric_columns = (
        df.select_dtypes(
            include=["number"]
        )
        .columns
    )

    infinite_count = 0

    for column in numeric_columns:

        values = df[column].to_numpy()

        infinite_count += int(
            np.isinf(values).sum()
        )

    print(
        "Infinite numeric values:",
        infinite_count
    )

    if infinite_count != 0:

        raise ValueError(
            "Infinite numeric values detected."
        )

    print()
    print(
        "Infinite values: PASS"
    )


# ============================================================
# TARGET / FEATURE SEPARATION
# ============================================================

def audit_feature_separation(df):

    header("FEATURE / TARGET SEPARATION")

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            ID_COLUMN,
            TARGET_COLUMN
        ]
    ]

    if TARGET_COLUMN in feature_columns:

        raise ValueError(
            "isFraud is incorrectly included "
            "in feature columns."
        )

    print(
        "Target excluded from model features: PASS"
    )

    print(
        "Identifier excluded from model features: PASS"
    )

    print(
        "Potential model input columns:",
        len(feature_columns)
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_summary(df):

    header("CLEANED DATASET AUDIT SUMMARY")

    print(
        "Rows              :",
        f"{len(df):,}"
    )

    print(
        "Columns           :",
        f"{len(df.columns):,}"
    )

    print(
        "Fraud             :",
        f"{int((df[TARGET_COLUMN] == 1).sum()):,}"
    )

    print(
        "Legitimate        :",
        f"{int((df[TARGET_COLUMN] == 0).sum()):,}"
    )

    print(
        "Minimum time      :",
        df[TIME_COLUMN].min()
    )

    print(
        "Maximum time      :",
        df[TIME_COLUMN].max()
    )

    print()
    print(
        "No data was modified."
    )

    print(
        "No split was performed."
    )

    print(
        "No client partition was performed."
    )

    print(
        "No graph was constructed."
    )

    print(
        "No model was trained."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    check_file()

    df = load_dataset()

    check_required_columns(df)

    audit_transaction_ids(df)

    audit_target(df)

    audit_transaction_time(df)

    audit_temporal_order(df)

    audit_missing_values(df)

    audit_infinite_values(df)

    audit_feature_separation(df)

    print_summary(df)

    print()
    print("=" * 70)
    print("CLEANED DATASET AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()