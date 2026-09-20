from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# RFGN — MERGED IEEE-CIS DATASET AUDIT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MERGED_FILE = PROCESSED_DIR / "ieee_merged.csv"


# ============================================================
# EXPECTED VALUES FROM PREVIOUS MERGE
# ============================================================

EXPECTED_ROWS = 590_540
EXPECTED_COLUMNS = 434


# ============================================================
# IMPORTANT COLUMNS
# ============================================================

IDENTIFIER_COLUMNS = [
    "TransactionID",
]

TARGET_COLUMNS = [
    "isFraud",
]

TEMPORAL_COLUMNS = [
    "TransactionDT",
]


# ============================================================
# HELPER
# ============================================================

def check_file(path: Path) -> None:

    print("=" * 70)
    print("CHECKING MERGED DATASET")
    print("=" * 70)

    if not path.exists():
        raise FileNotFoundError(
            f"Merged dataset not found:\n{path}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"Expected a file:\n{path}"
        )

    size_mb = (
        path.stat().st_size
        / (1024 ** 2)
    )

    print()
    print(f"Path   : {path}")
    print(f"Size   : {size_mb:.2f} MB")
    print("Status : PRESENT")


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset() -> pd.DataFrame:

    check_file(MERGED_FILE)

    print()
    print("=" * 70)
    print("LOADING MERGED DATASET")
    print("=" * 70)

    df = pd.read_csv(
        MERGED_FILE,
        low_memory=False
    )

    print()
    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns):,}"
    )

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
            f"Unexpected row count. "
            f"Expected {EXPECTED_ROWS:,}, "
            f"got {len(df):,}"
        )

    if len(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected column count. "
            f"Expected {EXPECTED_COLUMNS}, "
            f"got {len(df.columns)}"
        )

    print()
    print("Merged dataset dimensions: PASS")

    return df


# ============================================================
# BASIC INTEGRITY
# ============================================================

def audit_basic_integrity(
    df: pd.DataFrame
) -> None:

    print()
    print("=" * 70)
    print("BASIC DATA INTEGRITY")
    print("=" * 70)

    required_columns = (
        IDENTIFIER_COLUMNS
        + TARGET_COLUMNS
        + TEMPORAL_COLUMNS
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns missing: {missing}"
        )

    print(
        "Required columns: PASS"
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    print(
        f"Duplicate complete rows: "
        f"{duplicate_rows:,}"
    )

    transaction_duplicates = int(
        df["TransactionID"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate TransactionID: "
        f"{transaction_duplicates:,}"
    )

    if transaction_duplicates != 0:
        raise ValueError(
            "Duplicate TransactionID detected."
        )

    print(
        "TransactionID uniqueness: PASS"
    )


# ============================================================
# TARGET AUDIT
# ============================================================

def audit_target(
    df: pd.DataFrame
) -> None:

    print()
    print("=" * 70)
    print("TARGET AUDIT")
    print("=" * 70)

    if df["isFraud"].isna().any():

        raise ValueError(
            "isFraud contains missing values."
        )

    unique_values = set(
        df["isFraud"].unique()
    )

    print(
        f"Unique target values: "
        f"{sorted(unique_values)}"
    )

    if not unique_values.issubset({0, 1}):

        raise ValueError(
            "Target contains values "
            "other than 0 and 1."
        )

    counts = (
        df["isFraud"]
        .value_counts()
        .sort_index()
    )

    total = len(df)

    print()

    for label, count in counts.items():

        percentage = (
            count
            / total
            * 100
        )

        name = (
            "LEGITIMATE"
            if label == 0
            else "FRAUD"
        )

        print(
            f"{name:<12} : "
            f"{count:>10,} "
            f"({percentage:>8.4f}%)"
        )

    print()
    print(
        "Binary target: PASS"
    )


# ============================================================
# TEMPORAL AUDIT
# ============================================================

def audit_temporal_field(
    df: pd.DataFrame
) -> None:

    print()
    print("=" * 70)
    print("TEMPORAL FIELD AUDIT")
    print("=" * 70)

    transaction_dt = (
        df["TransactionDT"]
    )

    missing = int(
        transaction_dt.isna().sum()
    )

    print(
        f"Missing TransactionDT: "
        f"{missing:,}"
    )

    if missing != 0:
        raise ValueError(
            "TransactionDT contains missing values."
        )

    minimum = transaction_dt.min()
    maximum = transaction_dt.max()

    print(
        f"Minimum TransactionDT: "
        f"{minimum:,}"
    )

    print(
        f"Maximum TransactionDT: "
        f"{maximum:,}"
    )

    print(
        f"Time span: "
        f"{maximum - minimum:,}"
    )

    print()
    print(
        "TransactionDT: PASS"
    )


# ============================================================
# MISSING VALUE AUDIT
# ============================================================

def audit_missing_values(
    df: pd.DataFrame
) -> None:

    print()
    print("=" * 70)
    print("MISSING VALUE AUDIT")
    print("=" * 70)

    missing_counts = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    total_cells = (
        df.shape[0]
        * df.shape[1]
    )

    missing_cells = int(
        missing_counts.sum()
    )

    missing_percentage = (
        missing_cells
        / total_cells
        * 100
    )

    print()
    print(
        f"Missing cells : "
        f"{missing_cells:,}"
    )

    print(
        f"Missing rate  : "
        f"{missing_percentage:.4f}%"
    )

    print()
    print(
        "Missing-value distribution:"
    )

    print(
        "Threshold              Columns"
    )
    print(
        "----------------------------------------"
    )

    thresholds = [
        0,
        0.01,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]

    for threshold in thresholds:

        if threshold == 0:

            count = int(
                (missing_counts == 0)
                .sum()
            )

            label = "0% missing"

        else:

            count = int(
                (
                    missing_counts
                    / len(df)
                    >= threshold
                ).sum()
            )

            label = (
                f">={threshold * 100:.0f}% missing"
            )

        print(
            f"{label:<24} {count:>5}"
        )

    print()
    print(
        "Top 30 columns by missing percentage:"
    )

    for column, count in (
        missing_counts
        .head(30)
        .items()
    ):

        percentage = (
            count
            / len(df)
            * 100
        )

        print(
            f"{column:<22} "
            f"{percentage:>8.3f}% "
            f"({count:,})"
        )


# ============================================================
# CONSTANT / LOW-VARIANCE AUDIT
# ============================================================

def audit_constant_columns(
    df: pd.DataFrame
) -> None:

    print()
    print("=" * 70)
    print("CONSTANT / LOW-VARIANCE COLUMN AUDIT")
    print("=" * 70)

    constant_columns = []

    near_constant_columns = []

    for column in df.columns:

        nunique = (
            df[column]
            .nunique(
                dropna=False
            )
        )

        if nunique <= 1:

            constant_columns.append(
                column
            )

        non_null = (
            df[column]
            .dropna()
        )

        if len(non_null) > 0:

            top_frequency = (
                non_null
                .value_counts(
                    normalize=True
                )
                .iloc[0]
            )

            if (
                top_frequency >= 0.995
                and nunique > 1
            ):
                near_constant_columns.append(
                    (
                        column,
                        top_frequency
                    )
                )

    print()
    print(
        f"Constant columns: "
        f"{len(constant_columns)}"
    )

    if constant_columns:

        for column in constant_columns:

            print(
                f"  CONSTANT: {column}"
            )

    print()
    print(
        f"Near-constant columns "
        f"(>=99.5% same value): "
        f"{len(near_constant_columns)}"
    )

    for column, frequency in (
        near_constant_columns[:30]
    ):

        print(
            f"  {column:<22} "
            f"{frequency * 100:.3f}%"
        )


# ============================================================
# DATA TYPE AUDIT
# ============================================================

def audit_data_types(
    df: pd.DataFrame
) -> None:

    print()
    print("=" * 70)
    print("DATA TYPE AUDIT")
    print("=" * 70)

    numeric = df.select_dtypes(
        include=["number"]
    ).columns

    categorical = df.select_dtypes(
        include=[
            "object",
            "category"
        ]
    ).columns

    boolean = df.select_dtypes(
        include=["bool"]
    ).columns

    print(
        f"Numeric columns     : "
        f"{len(numeric):,}"
    )

    print(
        f"Categorical columns : "
        f"{len(categorical):,}"
    )

    print(
        f"Boolean columns     : "
        f"{len(boolean):,}"
    )

    print()
    print(
        "Categorical columns:"
    )

    for column in categorical:

        unique_count = (
            df[column]
            .nunique(
                dropna=True
            )
        )

        print(
            f"  {column:<22} "
            f"{unique_count:>10,} unique"
        )


# ============================================================
# NUMERIC VALIDITY AUDIT
# ============================================================

def audit_numeric_values(
    df: pd.DataFrame
) -> None:

    print()
    print("=" * 70)
    print("NUMERIC VALUE AUDIT")
    print("=" * 70)

    numeric_columns = (
        df.select_dtypes(
            include=["number"]
        ).columns
    )

    total_nan = 0
    total_inf = 0

    for column in numeric_columns:

        values = df[column]

        nan_count = int(
            values.isna().sum()
        )

        inf_count = int(
            np.isinf(
                values.to_numpy(
                    dtype=np.float64
                )
            ).sum()
        )

        total_nan += nan_count
        total_inf += inf_count

    print(
        f"Numeric NaN values : "
        f"{total_nan:,}"
    )

    print(
        f"Infinite values    : "
        f"{total_inf:,}"
    )

    if total_inf == 0:

        print(
            "Infinite numeric values: PASS"
        )

    else:

        print(
            "Infinite numeric values: WARNING"
        )


# ============================================================
# IDENTIFIER / POTENTIAL LEAKAGE AUDIT
# ============================================================

def audit_special_columns(
    df: pd.DataFrame
) -> None:

    print()
    print("=" * 70)
    print("SPECIAL COLUMN AUDIT")
    print("=" * 70)

    print()
    print("Identifier columns:")

    for column in IDENTIFIER_COLUMNS:

        print(
            f"  {column:<20} "
            f"present"
        )

    print()
    print("Target columns:")

    for column in TARGET_COLUMNS:

        print(
            f"  {column:<20} "
            f"present"
        )

    print()
    print("Temporal columns:")

    for column in TEMPORAL_COLUMNS:

        print(
            f"  {column:<20} "
            f"present"
        )

    print()
    print(
        "Potential leakage review:"
    )

    leakage_keywords = [
        "fraud",
        "label",
        "target",
        "prediction",
        "score",
        "future",
    ]

    candidates = []

    for column in df.columns:

        column_lower = (
            column.lower()
        )

        if any(
            keyword in column_lower
            for keyword in leakage_keywords
        ):

            candidates.append(
                column
            )

    for column in candidates:

        print(
            f"  REVIEW: {column}"
        )

    print()
    print(
        "These are candidates for manual "
        "leakage review; none are automatically "
        "removed by this script."
    )


# ============================================================
# CARD / GRAPH FIELD AUDIT
# ============================================================

def audit_graph_fields(
    df: pd.DataFrame
) -> None:

    print()
    print("=" * 70)
    print("GRAPH RELATIONSHIP FIELD AUDIT")
    print("=" * 70)

    fields = [
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
    ]

    found = []

    for column in fields:

        if column in df.columns:

            found.append(column)

            unique_count = (
                df[column]
                .nunique(
                    dropna=True
                )
            )

            print(
                f"  PASS  "
                f"{column:<20} "
                f"{unique_count:>10,} unique"
            )

    print()
    print(
        f"Relationship fields found: "
        f"{len(found)}"
    )


# ============================================================
# SAMPLE STATISTICS
# ============================================================

def audit_transaction_amount(
    df: pd.DataFrame
) -> None:

    print()
    print("=" * 70)
    print("TRANSACTION AMOUNT AUDIT")
    print("=" * 70)

    if "TransactionAmt" not in df.columns:
        print(
            "TransactionAmt not found."
        )
        return

    amount = df["TransactionAmt"]

    print(
        f"Missing : "
        f"{amount.isna().sum():,}"
    )

    print(
        f"Minimum : "
        f"{amount.min():.4f}"
    )

    print(
        f"Median  : "
        f"{amount.median():.4f}"
    )

    print(
        f"Mean    : "
        f"{amount.mean():.4f}"
    )

    print(
        f"Maximum : "
        f"{amount.max():.4f}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    df: pd.DataFrame
) -> None:

    print()
    print("=" * 70)
    print("MERGED DATASET AUDIT SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Rows                  : "
        f"{len(df):,}"
    )

    print(
        f"Columns               : "
        f"{len(df.columns):,}"
    )

    print(
        f"Fraud transactions    : "
        f"{int(df['isFraud'].sum()):,}"
    )

    print(
        f"Legitimate transactions: "
        f"{int((df['isFraud'] == 0).sum()):,}"
    )

    print()
    print(
        "No columns were removed."
    )

    print(
        "No values were imputed."
    )

    print(
        "No categorical encoding was performed."
    )

    print(
        "No feature scaling was performed."
    )

    print(
        "No train/validation/test split was performed."
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

    print()
    print("=" * 70)
    print("RFGN MERGED IEEE-CIS DATASET AUDIT")
    print("=" * 70)

    print()
    print(
        "This script is READ-ONLY."
    )

    print(
        "The merged dataset will NOT be modified."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Audits
    # --------------------------------------------------------

    audit_basic_integrity(df)

    audit_target(df)

    audit_temporal_field(df)

    audit_missing_values(df)

    audit_constant_columns(df)

    audit_data_types(df)

    audit_numeric_values(df)

    audit_special_columns(df)

    audit_graph_fields(df)

    audit_transaction_amount(df)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_final_summary(df)

    print()
    print("=" * 70)
    print("MERGED DATASET AUDIT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()