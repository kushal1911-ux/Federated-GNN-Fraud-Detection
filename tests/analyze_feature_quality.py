from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# RFGN — FEATURE QUALITY ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ieee_merged.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

NEAR_CONSTANT_THRESHOLD = 0.995

TOP_N = 50


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("=" * 70)
    print("RFGN FEATURE QUALITY ANALYSIS")
    print("=" * 70)

    print()
    print("READ-ONLY ANALYSIS")
    print("No dataset modifications will be performed.")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    print()
    print(f"Dataset: {DATA_FILE}")

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False
    )

    print()
    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns):,}"
    )

    return df


# ============================================================
# BASIC COLUMN STATISTICS
# ============================================================

def build_quality_table(df):

    print()
    print("=" * 70)
    print("BUILDING FEATURE QUALITY TABLE")
    print("=" * 70)

    records = []

    total_rows = len(df)

    for column in df.columns:

        series = df[column]

        missing_count = int(
            series.isna().sum()
        )

        missing_rate = (
            missing_count
            / total_rows
        )

        non_missing = series.dropna()

        unique_count = int(
            series.nunique(
                dropna=False
            )
        )

        non_missing_unique = int(
            series.nunique(
                dropna=True
            )
        )

        if len(non_missing) > 0:

            top_frequency = (
                non_missing
                .value_counts(
                    normalize=True
                )
                .iloc[0]
            )

        else:

            top_frequency = 1.0

        constant = (
            unique_count <= 1
        )

        near_constant = (
            top_frequency
            >= NEAR_CONSTANT_THRESHOLD
        )

        records.append(
            {
                "column": column,
                "dtype": str(
                    series.dtype
                ),
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "unique_count": unique_count,
                "non_missing_unique": (
                    non_missing_unique
                ),
                "unique_rate": (
                    unique_count
                    / total_rows
                ),
                "top_value_frequency": (
                    top_frequency
                ),
                "constant": constant,
                "near_constant": (
                    near_constant
                ),
            }
        )

    quality = pd.DataFrame(
        records
    )

    return quality


# ============================================================
# TARGET ASSOCIATION
# ============================================================

def calculate_target_association(
    df,
    quality
):

    print()
    print("=" * 70)
    print("TARGET ASSOCIATION ANALYSIS")
    print("=" * 70)

    if "isFraud" not in df.columns:

        raise ValueError(
            "isFraud column not found."
        )

    overall_fraud_rate = (
        df["isFraud"].mean()
    )

    print()
    print(
        f"Overall fraud rate: "
        f"{overall_fraud_rate * 100:.4f}%"
    )

    missing_fraud_rates = []
    present_fraud_rates = []

    for column in df.columns:

        if column == "isFraud":
            continue

        missing_mask = (
            df[column].isna()
        )

        present_mask = (
            ~missing_mask
        )

        missing_count = int(
            missing_mask.sum()
        )

        present_count = int(
            present_mask.sum()
        )

        if missing_count > 0:

            missing_fraud_rate = (
                df.loc[
                    missing_mask,
                    "isFraud"
                ].mean()
            )

        else:

            missing_fraud_rate = np.nan

        if present_count > 0:

            present_fraud_rate = (
                df.loc[
                    present_mask,
                    "isFraud"
                ].mean()
            )

        else:

            present_fraud_rate = np.nan

        if (
            not np.isnan(
                missing_fraud_rate
            )
            and not np.isnan(
                present_fraud_rate
            )
        ):

            difference = (
                present_fraud_rate
                - missing_fraud_rate
            )

            absolute_difference = abs(
                difference
            )

        else:

            difference = np.nan
            absolute_difference = np.nan

        missing_fraud_rates.append(
            missing_fraud_rate
        )

        present_fraud_rates.append(
            present_fraud_rate
        )

        quality.loc[
            quality["column"] == column,
            "missing_fraud_rate"
        ] = missing_fraud_rate

        quality.loc[
            quality["column"] == column,
            "present_fraud_rate"
        ] = present_fraud_rate

        quality.loc[
            quality["column"] == column,
            "fraud_rate_difference"
        ] = difference

        quality.loc[
            quality["column"] == column,
            "absolute_fraud_rate_difference"
        ] = absolute_difference

    print()
    print(
        "Target association statistics calculated."
    )

    return quality


# ============================================================
# MISSINGNESS RANKING
# ============================================================

def show_missingness_ranking(
    quality
):

    print()
    print("=" * 70)
    print("MISSINGNESS RANKING")
    print("=" * 70)

    ranked = (
        quality[
            quality["column"] != "isFraud"
        ]
        .sort_values(
            "missing_rate",
            ascending=False
        )
        .head(TOP_N)
    )

    print()

    print(
        f"{'Column':<22}"
        f"{'Missing %':>12}"
        f"{'Unique':>12}"
    )

    print("-" * 50)

    for _, row in ranked.iterrows():

        print(
            f"{row['column']:<22}"
            f"{row['missing_rate'] * 100:>11.3f}%"
            f"{row['unique_count']:>12,}"
        )


# ============================================================
# TARGET-ASSOCIATION RANKING
# ============================================================

def show_target_association(
    quality
):

    print()
    print("=" * 70)
    print("MISSINGNESS VS FRAUD-RATE ANALYSIS")
    print("=" * 70)

    ranked = (
        quality[
            quality["column"] != "isFraud"
        ]
        .dropna(
            subset=[
                "absolute_fraud_rate_difference"
            ]
        )
        .sort_values(
            "absolute_fraud_rate_difference",
            ascending=False
        )
        .head(TOP_N)
    )

    print()

    print(
        f"{'Column':<22}"
        f"{'Missing %':>11}"
        f"{'Missing Fraud %':>17}"
        f"{'Present Fraud %':>17}"
    )

    print("-" * 70)

    for _, row in ranked.iterrows():

        print(
            f"{row['column']:<22}"
            f"{row['missing_rate'] * 100:>10.2f}%"
            f"{row['missing_fraud_rate'] * 100:>16.3f}%"
            f"{row['present_fraud_rate'] * 100:>16.3f}%"
        )


# ============================================================
# CATEGORICAL ANALYSIS
# ============================================================

def analyze_categorical_features(
    df,
    quality
):

    print()
    print("=" * 70)
    print("CATEGORICAL FEATURE ANALYSIS")
    print("=" * 70)

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category"
            ]
        )
        .columns
    )

    print()
    print(
        f"Categorical columns: "
        f"{len(categorical_columns)}"
    )

    print()

    for column in categorical_columns:

        series = df[column]

        unique = int(
            series.nunique(
                dropna=True
            )
        )

        missing = int(
            series.isna().sum()
        )

        print(
            f"{column:<22}"
            f"unique={unique:>6,} "
            f"missing={missing:>10,}"
        )


# ============================================================
# NEAR-CONSTANT ANALYSIS
# ============================================================

def show_near_constant(
    quality
):

    print()
    print("=" * 70)
    print("NEAR-CONSTANT FEATURE ANALYSIS")
    print("=" * 70)

    result = (
        quality[
            quality["near_constant"]
        ]
        .sort_values(
            "top_value_frequency",
            ascending=False
        )
    )

    print()
    print(
        f"Near-constant columns: "
        f"{len(result)}"
    )

    print()

    for _, row in result.iterrows():

        print(
            f"{row['column']:<22}"
            f"{row['top_value_frequency'] * 100:>10.3f}% "
            f"dominant"
        )


# ============================================================
# HIGH-MISSINGNESS ANALYSIS
# ============================================================

def show_high_missing_groups(
    quality
):

    print()
    print("=" * 70)
    print("HIGH-MISSINGNESS GROUPS")
    print("=" * 70)

    thresholds = [
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]

    print()

    for threshold in thresholds:

        count = int(
            (
                quality[
                    "missing_rate"
                ]
                >= threshold
            )
            .sum()
        )

        print(
            f">= {threshold * 100:.0f}% missing:"
            f" {count} columns"
        )


# ============================================================
# SPECIAL FEATURES
# ============================================================

def inspect_special_features(
    df
):

    print()
    print("=" * 70)
    print("SPECIAL FEATURE INSPECTION")
    print("=" * 70)

    special_columns = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
    ]

    print()

    for column in special_columns:

        if column not in df.columns:
            continue

        series = df[column]

        print(
            f"{column:<20}"
            f"dtype={str(series.dtype):<12}"
            f"missing={series.isna().sum():>8,}"
            f"unique={series.nunique(dropna=True):>10,}"
        )


# ============================================================
# BUILD RECOMMENDATION REPORT
# ============================================================

def print_recommendation_summary(
    quality
):

    print()
    print("=" * 70)
    print("FEATURE QUALITY SUMMARY")
    print("=" * 70)

    feature_quality = quality[
        ~quality["column"].isin(
            [
                "TransactionID",
                "isFraud",
            ]
        )
    ]

    constant_count = int(
        feature_quality[
            "constant"
        ].sum()
    )

    near_constant_count = int(
        feature_quality[
            "near_constant"
        ].sum()
    )

    high_missing_90 = int(
        (
            feature_quality[
                "missing_rate"
            ]
            >= 0.90
        )
        .sum()
    )

    high_missing_95 = int(
        (
            feature_quality[
                "missing_rate"
            ]
            >= 0.95
        )
        .sum()
    )

    print()
    print(
        f"Model candidate features: "
        f"{len(feature_quality):,}"
    )

    print(
        f"Constant features: "
        f"{constant_count:,}"
    )

    print(
        f"Near-constant features: "
        f"{near_constant_count:,}"
    )

    print(
        f">=90% missing: "
        f"{high_missing_90:,}"
    )

    print(
        f">=95% missing: "
        f"{high_missing_95:,}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "No features are removed by this analysis."
    )

    print(
        "No missing values are filled."
    )

    print(
        "No categorical values are encoded."
    )

    print(
        "No scaling is performed."
    )

    print(
        "No client splitting is performed."
    )

    print(
        "No graph construction is performed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    quality = build_quality_table(
        df
    )

    quality = calculate_target_association(
        df,
        quality
    )

    inspect_special_features(
        df
    )

    show_missingness_ranking(
        quality
    )

    show_target_association(
        quality
    )

    analyze_categorical_features(
        df,
        quality
    )

    show_near_constant(
        quality
    )

    show_high_missing_groups(
        quality
    )

    print_recommendation_summary(
        quality
    )

    print()
    print("=" * 70)
    print("FEATURE QUALITY ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()