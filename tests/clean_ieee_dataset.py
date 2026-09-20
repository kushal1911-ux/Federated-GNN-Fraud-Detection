import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# RFGN IEEE-CIS DATASET CLEANING
# ============================================================
#
# Purpose:
#   Clean the merged IEEE-CIS dataset for the next stages
#   of the RFGN project.
#
# Input:
#   data/processed/ieee_merged.csv
#
# Outputs:
#   data/processed/ieee_cleaned.csv
#   data/processed/feature_manifest.json
#
# Important:
#   - Raw datasets are never modified.
#   - Target isFraud is never used as a feature.
#   - TransactionID is preserved.
#   - TransactionDT is preserved.
#   - Missingness indicators are created for feature columns.
#   - Numeric feature missing values are filled with medians.
#   - Categorical feature missing values are filled with "MISSING".
#   - Infinite numeric values are replaced safely.
#
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INPUT_FILE = (
    PROCESSED_DIR
    / "ieee_merged.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "ieee_cleaned.csv"
)

MANIFEST_FILE = (
    PROCESSED_DIR
    / "feature_manifest.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# CORE COLUMNS
# ============================================================

ID_COLUMN = "TransactionID"
TARGET_COLUMN = "isFraud"
TIME_COLUMN = "TransactionDT"


# ============================================================
# LOGGING HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_section(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    print_header("RFGN IEEE-CIS DATASET CLEANING")

    print()
    print("This script creates a cleaned copy.")
    print("The merged dataset will NOT be modified.")
    print()
    print("Project root :", PROJECT_ROOT)
    print("Input file   :", INPUT_FILE)
    print("Output file  :", OUTPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_FILE}"
        )

    print_section("LOADING MERGED DATASET")

    df = pd.read_csv(INPUT_FILE)

    print("Dataset loaded successfully.")
    print("Rows    :", len(df))
    print("Columns :", len(df))
    print(
        "Memory  :",
        f"{df.memory_usage(deep=True).sum() / (1024 ** 3):.3f} GB"
    )

    return df


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

def validate_required_columns(df):

    print_section("VALIDATING REQUIRED COLUMNS")

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
            f"Required columns missing: {missing}"
        )

    print("TransactionID : PASS")
    print("isFraud       : PASS")
    print("TransactionDT : PASS")

    return True


# ============================================================
# VALIDATE TARGET
# ============================================================

def validate_target(df):

    print_section("VALIDATING FRAUD TARGET")

    unique_values = sorted(
        df[TARGET_COLUMN].dropna().unique().tolist()
    )

    print("Unique target values:", unique_values)

    if set(unique_values) != {0, 1}:
        raise ValueError(
            "isFraud must contain exactly binary labels 0 and 1."
        )

    fraud_count = int(
        (df[TARGET_COLUMN] == 1).sum()
    )

    legitimate_count = int(
        (df[TARGET_COLUMN] == 0).sum()
    )

    print()
    print(
        f"LEGITIMATE : {legitimate_count:>10,} "
        f"({legitimate_count / len(df) * 100:.4f}%)"
    )

    print(
        f"FRAUD      : {fraud_count:>10,} "
        f"({fraud_count / len(df) * 100:.4f}%)"
    )

    print()
    print("Binary fraud target: PASS")

    return True


# ============================================================
# PRESERVE ORIGINAL VALUES
# ============================================================

def capture_integrity_reference(df):

    print_section("CAPTURING DATA INTEGRITY REFERENCE")

    reference = {
        ID_COLUMN: df[ID_COLUMN].copy(),
        TARGET_COLUMN: df[TARGET_COLUMN].copy(),
        TIME_COLUMN: df[TIME_COLUMN].copy()
    }

    print("TransactionID reference: CAPTURED")
    print("isFraud reference      : CAPTURED")
    print("TransactionDT reference: CAPTURED")

    return reference


# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

def identify_feature_columns(df):

    print_section("IDENTIFYING FEATURE COLUMNS")

    excluded_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
        TIME_COLUMN
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    print("Total dataset columns :", len(df.columns))
    print("Excluded columns      :", len(excluded_columns))
    print("Feature columns       :", len(feature_columns))

    print()
    print("Excluded from model features:")

    for column in sorted(excluded_columns):
        print(" -", column)

    return feature_columns


# ============================================================
# REMOVE COMPLETELY EMPTY FEATURE COLUMNS
# ============================================================

def remove_empty_feature_columns(df, feature_columns):

    print_section("CHECKING COMPLETELY EMPTY FEATURES")

    empty_columns = [
        column
        for column in feature_columns
        if df[column].isna().all()
    ]

    if empty_columns:

        print(
            "Completely empty feature columns:",
            len(empty_columns)
        )

        for column in empty_columns:
            print(" -", column)

        df = df.drop(
            columns=empty_columns
        )

        feature_columns = [
            column
            for column in feature_columns
            if column not in empty_columns
        ]

    else:

        print("Completely empty feature columns: 0")

    print("Feature columns remaining:", len(feature_columns))

    return df, feature_columns, empty_columns


# ============================================================
# HANDLE INFINITE VALUES
# ============================================================

def replace_infinite_values(df, feature_columns):

    print_section("HANDLING INFINITE NUMERIC VALUES")

    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    infinite_count = 0

    for column in numeric_columns:

        values = df[column].to_numpy()

        count = np.isinf(values).sum()

        if count > 0:
            infinite_count += int(count)

            df[column] = df[column].replace(
                [np.inf, -np.inf],
                np.nan
            )

    print("Infinite values found :", infinite_count)

    if infinite_count == 0:
        print("Infinite values: PASS")
    else:
        print(
            "Infinite values converted to NaN:",
            infinite_count
        )

    return df


# ============================================================
# ADD MISSINGNESS INDICATORS
# ============================================================

def add_missingness_indicators(df, feature_columns):

    print_section("ADDING MISSINGNESS INDICATORS")

    indicator_data = {}

    for column in feature_columns:

        missing_mask = df[column].isna()

        if missing_mask.any():

            indicator_name = f"{column}__missing"

            indicator_data[indicator_name] = (
                missing_mask.astype("int8")
            )

    if indicator_data:

        indicators_df = pd.DataFrame(
            indicator_data,
            index=df.index
        )

        df = pd.concat(
            [df, indicators_df],
            axis=1
        )

    indicator_columns = list(
        indicator_data.keys()
    )

    print(
        "Missingness indicators added:",
        len(indicator_columns)
    )

    print(
        "Missingness indicators: PASS"
    )

    return df, indicator_columns


# ============================================================
# HANDLE CATEGORICAL FEATURES
# ============================================================

def handle_categorical_features(df, feature_columns):

    print_section("HANDLING CATEGORICAL FEATURES")

    categorical_columns = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    print(
        "Categorical feature columns:",
        len(categorical_columns)
    )

    filled_count = 0

    for column in categorical_columns:

        missing_before = int(
            df[column].isna().sum()
        )

        if missing_before > 0:

            df[column] = (
                df[column]
                .astype("object")
                .fillna("MISSING")
            )

            filled_count += missing_before

        else:

            df[column] = (
                df[column]
                .astype("object")
            )

    print(
        "Categorical missing cells filled:",
        filled_count
    )

    print(
        "Categorical missing-value handling: PASS"
    )

    return df, categorical_columns


# ============================================================
# HANDLE NUMERIC FEATURES
# ============================================================

def handle_numeric_features(df, feature_columns):

    print_section("HANDLING NUMERICAL MISSING VALUES")

    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    filled_columns = 0
    filled_cells = 0

    for column in numeric_columns:

        missing_before = int(
            df[column].isna().sum()
        )

        if missing_before == 0:
            continue

        median_value = df[column].median()

        if pd.isna(median_value):
            median_value = 0.0

        df[column] = (
            df[column]
            .fillna(median_value)
        )

        filled_columns += 1
        filled_cells += missing_before

    remaining_missing = int(
        df[numeric_columns].isna().sum().sum()
    )

    print(
        "Numeric feature columns filled:",
        filled_columns
    )

    print(
        "Numeric missing cells filled:",
        filled_cells
    )

    print(
        "Remaining numeric missing cells:",
        remaining_missing
    )

    if remaining_missing != 0:
        raise ValueError(
            "Numeric missing values remain after cleaning."
        )

    print(
        "Numerical missing-value handling: PASS"
    )

    return df, numeric_columns


# ============================================================
# VERIFY NUMERIC VALUES
# ============================================================

def verify_numeric_values(df, feature_columns):

    print_section("VERIFYING NUMERICAL VALUES")

    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    infinite_values = 0
    nan_values = 0

    for column in numeric_columns:

        values = df[column].to_numpy()

        infinite_values += int(
            np.isinf(values).sum()
        )

        nan_values += int(
            np.isnan(values).sum()
        )

    print(
        "Remaining numeric NaN values:",
        nan_values
    )

    print(
        "Infinite numeric values:",
        infinite_values
    )

    if nan_values != 0:
        raise ValueError(
            "Numeric NaN values remain."
        )

    if infinite_values != 0:
        raise ValueError(
            "Infinite numeric values remain."
        )

    print("Infinite values: PASS")

    return True


# ============================================================
# VERIFY TARGET / IDENTIFIERS
# ============================================================

def verify_integrity(df, reference):

    print_section("VERIFYING TARGET AND IDENTIFIERS")

    id_same = (
        df[ID_COLUMN].reset_index(drop=True)
        .equals(
            reference[ID_COLUMN]
            .reset_index(drop=True)
        )
    )

    target_same = (
        df[TARGET_COLUMN]
        .reset_index(drop=True)
        .equals(
            reference[TARGET_COLUMN]
            .reset_index(drop=True)
        )
    )

    time_same = (
        df[TIME_COLUMN]
        .reset_index(drop=True)
        .equals(
            reference[TIME_COLUMN]
            .reset_index(drop=True)
        )
    )

    print(
        "TransactionID preservation:",
        "PASS" if id_same else "FAIL"
    )

    print(
        "isFraud preservation:",
        "PASS" if target_same else "FAIL"
    )

    print(
        "TransactionDT preservation:",
        "PASS" if time_same else "FAIL"
    )

    if not id_same:
        raise ValueError(
            "TransactionID changed during cleaning."
        )

    if not target_same:
        raise ValueError(
            "isFraud changed during cleaning."
        )

    if not time_same:
        raise ValueError(
            "TransactionDT changed during cleaning."
        )

    return True


# ============================================================
# VERIFY DUPLICATE TRANSACTION IDS
# ============================================================

def verify_transaction_ids(df):

    print_section("VERIFYING TRANSACTION ID UNIQUENESS")

    duplicate_count = int(
        df[ID_COLUMN].duplicated().sum()
    )

    print(
        "Duplicate TransactionIDs:",
        duplicate_count
    )

    if duplicate_count != 0:
        raise ValueError(
            "Duplicate TransactionIDs detected."
        )

    print(
        "TransactionID uniqueness: PASS"
    )

    return True


# ============================================================
# VERIFY ROW COUNT
# ============================================================

def verify_row_count(df, original_row_count):

    print_section("VERIFYING ROW COUNT")

    print(
        "Original rows:",
        original_row_count
    )

    print(
        "Cleaned rows :",
        len(df)
    )

    if len(df) != original_row_count:
        raise ValueError(
            "Row count changed during cleaning."
        )

    print("Row preservation: PASS")

    return True


# ============================================================
# BUILD FEATURE MANIFEST
# ============================================================

def build_feature_manifest(
    df,
    feature_columns,
    indicator_columns,
    categorical_columns,
    numeric_columns,
    removed_empty_columns
):

    print_section("BUILDING FEATURE MANIFEST")

    manifest = {
        "project": "RFGN",
        "dataset": "IEEE-CIS Fraud Detection",
        "random_seed": RANDOM_SEED,

        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),

        "rows": int(len(df)),
        "columns": int(len(df.columns)),

        "identifier_column": ID_COLUMN,
        "target_column": TARGET_COLUMN,
        "temporal_column": TIME_COLUMN,

        "original_feature_count": int(
            len(feature_columns)
        ),

        "numeric_feature_count": int(
            len(numeric_columns)
        ),

        "categorical_feature_count": int(
            len(categorical_columns)
        ),

        "missingness_indicator_count": int(
            len(indicator_columns)
        ),

        "removed_empty_feature_count": int(
            len(removed_empty_columns)
        ),

        "feature_columns": feature_columns,

        "numeric_features": numeric_columns,

        "categorical_features": categorical_columns,

        "missingness_indicators": indicator_columns,

        "removed_empty_features": removed_empty_columns
    }

    return manifest


# ============================================================
# SAVE CLEANED DATASET
# ============================================================

def save_cleaned_dataset(df, manifest):

    print_section("SAVING CLEANED DATASET")

    # This is the variable that was missing in the
    # previous version.
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Saving cleaned dataset...")
    print("This may take some time.")

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Saved to:")
    print(OUTPUT_FILE)

    file_size_mb = (
        OUTPUT_FILE.stat().st_size
        / (1024 ** 2)
    )

    print(
        "File size:",
        f"{file_size_mb:.2f} MB"
    )

    print()
    print("Save operation: PASS")

    print()
    print("Saving feature manifest...")

    with open(
        MANIFEST_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4
        )

    print()
    print("Feature manifest saved to:")
    print(MANIFEST_FILE)

    print()
    print("Manifest save: PASS")


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary(
    df,
    feature_columns,
    indicator_columns,
    numeric_columns,
    categorical_columns,
    removed_empty_columns
):

    print_header("RFGN IEEE-CIS CLEANING SUMMARY")

    print("Rows                  :", len(df))
    print("Final columns         :", len(df.columns))
    print(
        "Original feature cols :",
        len(feature_columns)
    )
    print(
        "Numeric features      :",
        len(numeric_columns)
    )
    print(
        "Categorical features  :",
        len(categorical_columns)
    )
    print(
        "Missing indicators    :",
        len(indicator_columns)
    )
    print(
        "Removed empty columns :",
        len(removed_empty_columns)
    )

    fraud_count = int(
        (df[TARGET_COLUMN] == 1).sum()
    )

    legitimate_count = int(
        (df[TARGET_COLUMN] == 0).sum()
    )

    print()
    print("Fraud transactions     :", fraud_count)
    print(
        "Legitimate transactions:",
        legitimate_count
    )

    print()
    print("Output dataset:")
    print(OUTPUT_FILE)

    print()
    print("Feature manifest:")
    print(MANIFEST_FILE)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_dataset()

    original_row_count = len(df)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validate_required_columns(df)

    validate_target(df)

    reference = capture_integrity_reference(df)

    # --------------------------------------------------------
    # FEATURE IDENTIFICATION
    # --------------------------------------------------------

    feature_columns = identify_feature_columns(df)

    # --------------------------------------------------------
    # REMOVE COMPLETELY EMPTY FEATURES
    # --------------------------------------------------------

    (
        df,
        feature_columns,
        removed_empty_columns
    ) = remove_empty_feature_columns(
        df,
        feature_columns
    )

    # --------------------------------------------------------
    # INFINITE VALUES
    # --------------------------------------------------------

    df = replace_infinite_values(
        df,
        feature_columns
    )

    # --------------------------------------------------------
    # MISSINGNESS INDICATORS
    # --------------------------------------------------------

    (
        df,
        indicator_columns
    ) = add_missingness_indicators(
        df,
        feature_columns
    )

    # --------------------------------------------------------
    # CATEGORICAL FEATURES
    # --------------------------------------------------------

    (
        df,
        categorical_columns
    ) = handle_categorical_features(
        df,
        feature_columns
    )

    # --------------------------------------------------------
    # NUMERIC FEATURES
    # --------------------------------------------------------

    (
        df,
        numeric_columns
    ) = handle_numeric_features(
        df,
        feature_columns
    )

    # --------------------------------------------------------
    # NUMERIC VERIFICATION
    # --------------------------------------------------------

    verify_numeric_values(
        df,
        feature_columns
    )

    # --------------------------------------------------------
    # IDENTIFIER / TARGET VERIFICATION
    # --------------------------------------------------------

    verify_integrity(
        df,
        reference
    )

    # --------------------------------------------------------
    # ID VERIFICATION
    # --------------------------------------------------------

    verify_transaction_ids(df)

    # --------------------------------------------------------
    # ROW COUNT VERIFICATION
    # --------------------------------------------------------

    verify_row_count(
        df,
        original_row_count
    )

    # --------------------------------------------------------
    # FEATURE MANIFEST
    # --------------------------------------------------------

    manifest = build_feature_manifest(
        df=df,
        feature_columns=feature_columns,
        indicator_columns=indicator_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        removed_empty_columns=removed_empty_columns
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_cleaned_dataset(
        df,
        manifest
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_summary(
        df=df,
        feature_columns=feature_columns,
        indicator_columns=indicator_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        removed_empty_columns=removed_empty_columns
    )

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("IEEE-CIS CLEANING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print()
    print("Raw dataset modified: NO")
    print("Merged dataset modified: NO")
    print("Cleaned dataset created: YES")
    print("Feature manifest created: YES")

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()