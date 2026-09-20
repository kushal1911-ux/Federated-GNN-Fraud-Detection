from pathlib import Path

import pandas as pd


# ============================================================
# RFGN IEEE-CIS DATASET AUDIT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRANSACTION_FILE = RAW_DATA_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DATA_DIR / "train_identity.csv"


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_TRANSACTION_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
]

REQUIRED_IDENTITY_COLUMNS = [
    "TransactionID",
]


# ============================================================
# FILE CHECK
# ============================================================

def check_file(path: Path) -> None:
    """Verify that a required dataset file exists."""

    print()
    print("-" * 70)
    print(f"Checking: {path.name}")
    print("-" * 70)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found:\n{path}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset path is not a file:\n{path}"
        )

    size_mb = path.stat().st_size / (1024 ** 2)

    print(f"Path   : {path}")
    print(f"Size   : {size_mb:.2f} MB")
    print("Status : PRESENT")


# ============================================================
# LOAD TRANSACTION DATA
# ============================================================

def load_transaction_data() -> pd.DataFrame:

    print()
    print("=" * 70)
    print("TRANSACTION DATA")
    print("=" * 70)

    check_file(TRANSACTION_FILE)

    df = pd.read_csv(
        TRANSACTION_FILE,
        low_memory=False
    )

    print()
    print("Transaction dataset loaded successfully.")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    memory_gb = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 3)
    )

    print(f"Memory  : {memory_gb:.3f} GB")

    missing = [
        column
        for column in REQUIRED_TRANSACTION_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing transaction columns: {missing}"
        )

    print()
    print("Required transaction columns: PASS")

    return df


# ============================================================
# LOAD IDENTITY DATA
# ============================================================

def load_identity_data() -> pd.DataFrame:

    print()
    print("=" * 70)
    print("IDENTITY DATA")
    print("=" * 70)

    check_file(IDENTITY_FILE)

    df = pd.read_csv(
        IDENTITY_FILE,
        low_memory=False
    )

    print()
    print("Identity dataset loaded successfully.")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    memory_gb = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 3)
    )

    print(f"Memory  : {memory_gb:.3f} GB")

    missing = [
        column
        for column in REQUIRED_IDENTITY_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing identity columns: {missing}"
        )

    print()
    print("Required identity columns: PASS")

    return df


# ============================================================
# LABEL AUDIT
# ============================================================

def audit_labels(df: pd.DataFrame) -> None:

    print()
    print("=" * 70)
    print("FRAUD LABEL AUDIT")
    print("=" * 70)

    if df["isFraud"].isna().any():
        raise ValueError(
            "isFraud contains missing values."
        )

    counts = (
        df["isFraud"]
        .value_counts()
        .sort_index()
    )

    total = len(df)

    for label, count in counts.items():

        percentage = (
            count / total * 100
        )

        if label == 0:
            name = "LEGITIMATE"
        elif label == 1:
            name = "FRAUD"
        else:
            name = "UNKNOWN"

        print(
            f"{name:<12} "
            f"({label}) : "
            f"{count:>10,} "
            f"({percentage:>8.4f}%)"
        )

    unique_labels = set(
        df["isFraud"].unique()
    )

    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Unexpected labels: {unique_labels}"
        )

    print()
    print("Binary fraud labels: PASS")


# ============================================================
# TRANSACTION ID AUDIT
# ============================================================

def audit_transaction_ids(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame
) -> None:

    print()
    print("=" * 70)
    print("TRANSACTION ID AUDIT")
    print("=" * 70)

    transaction_ids = transaction_df[
        "TransactionID"
    ]

    identity_ids = identity_df[
        "TransactionID"
    ]

    transaction_duplicates = (
        transaction_ids.duplicated().sum()
    )

    identity_duplicates = (
        identity_ids.duplicated().sum()
    )

    print(
        "TransactionID duplicates "
        f"in transaction data : "
        f"{transaction_duplicates:,}"
    )

    print(
        "TransactionID duplicates "
        f"in identity data    : "
        f"{identity_duplicates:,}"
    )

    transaction_set = set(
        transaction_ids
    )

    identity_set = set(
        identity_ids
    )

    matched = len(
        transaction_set.intersection(
            identity_set
        )
    )

    print()
    print(
        "TransactionIDs shared with identity: "
        f"{matched:,}"
    )

    if transaction_duplicates == 0:
        print(
            "TransactionID uniqueness: PASS"
        )
    else:
        print(
            "TransactionID uniqueness: WARNING"
        )


# ============================================================
# TIME AUDIT
# ============================================================

def audit_transaction_time(
    df: pd.DataFrame
) -> None:

    print()
    print("=" * 70)
    print("TRANSACTION TIME AUDIT")
    print("=" * 70)

    if df["TransactionDT"].isna().any():
        raise ValueError(
            "TransactionDT contains missing values."
        )

    minimum = df["TransactionDT"].min()
    maximum = df["TransactionDT"].max()

    print(
        f"Minimum TransactionDT : {minimum:,}"
    )

    print(
        f"Maximum TransactionDT : {maximum:,}"
    )

    print(
        f"Time span             : "
        f"{maximum - minimum:,}"
    )

    print(
        "Temporal field: PASS"
    )


# ============================================================
# MISSING VALUE AUDIT
# ============================================================

def audit_missing_values(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame
) -> None:

    print()
    print("=" * 70)
    print("MISSING VALUE AUDIT")
    print("=" * 70)

    for name, df in [
        ("TRANSACTION", transaction_df),
        ("IDENTITY", identity_df),
    ]:

        missing_cells = int(
            df.isna().sum().sum()
        )

        total_cells = (
            df.shape[0]
            * df.shape[1]
        )

        percentage = (
            missing_cells
            / total_cells
            * 100
        )

        print()
        print(f"{name}")

        print(
            f"Missing cells : "
            f"{missing_cells:,}"
        )

        print(
            f"Missing rate  : "
            f"{percentage:.4f}%"
        )

        column_missing = (
            df.isna()
            .sum()
            .sort_values(
                ascending=False
            )
        )

        nonzero = column_missing[
            column_missing > 0
        ].head(20)

        print()
        print(
            "Top columns by missing values:"
        )

        for column, count in nonzero.items():

            column_percentage = (
                count
                / len(df)
                * 100
            )

            print(
                f"  {column:<20} "
                f"{count:>10,} "
                f"({column_percentage:>8.3f}%)"
            )


# ============================================================
# COLUMN TYPE AUDIT
# ============================================================

def audit_column_types(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame
) -> None:

    print()
    print("=" * 70)
    print("COLUMN TYPE AUDIT")
    print("=" * 70)

    for name, df in [
        ("TRANSACTION", transaction_df),
        ("IDENTITY", identity_df),
    ]:

        numeric = df.select_dtypes(
            include=["number"]
        ).columns

        categorical = df.select_dtypes(
            include=["object", "category"]
        ).columns

        print()
        print(name)

        print(
            f"Numeric columns     : "
            f"{len(numeric):,}"
        )

        print(
            f"Categorical columns : "
            f"{len(categorical):,}"
        )


# ============================================================
# RELATIONSHIP FIELD AUDIT
# ============================================================

def audit_relationship_fields(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame
) -> None:

    print()
    print("=" * 70)
    print("GRAPH RELATIONSHIP FIELD AUDIT")
    print("=" * 70)

    transaction_fields = [
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
    ]

    identity_fields = [
        "DeviceType",
        "DeviceInfo",
        "id_01",
        "id_02",
        "id_03",
        "id_04",
        "id_05",
        "id_06",
        "id_07",
        "id_08",
        "id_09",
        "id_10",
        "id_11",
        "id_12",
        "id_13",
        "id_14",
        "id_15",
        "id_16",
        "id_17",
        "id_18",
        "id_19",
        "id_20",
        "id_21",
        "id_22",
        "id_23",
        "id_24",
        "id_25",
        "id_26",
        "id_27",
        "id_28",
        "id_29",
        "id_30",
        "id_31",
        "id_32",
        "id_33",
        "id_34",
        "id_35",
        "id_36",
        "id_37",
        "id_38",
    ]

    found_transaction = [
        column
        for column in transaction_fields
        if column in transaction_df.columns
    ]

    found_identity = [
        column
        for column in identity_fields
        if column in identity_df.columns
    ]

    print()
    print("Transaction relationship fields:")

    for column in found_transaction:
        print(f"  PASS  {column}")

    print()
    print("Identity relationship fields:")

    for column in found_identity:
        print(f"  PASS  {column}")

    print()
    print(
        "Transaction relationship fields found: "
        f"{len(found_transaction)}"
    )

    print(
        "Identity relationship fields found: "
        f"{len(found_identity)}"
    )


# ============================================================
# DATASET OVERVIEW
# ============================================================

def print_dataset_overview(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame
) -> None:

    print()
    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)

    print(
        f"Transaction rows : "
        f"{len(transaction_df):,}"
    )

    print(
        f"Transaction cols : "
        f"{len(transaction_df.columns):,}"
    )

    print(
        f"Identity rows    : "
        f"{len(identity_df):,}"
    )

    print(
        f"Identity cols    : "
        f"{len(identity_df.columns):,}"
    )

    print(
        f"Transaction range: "
        f"{transaction_df['TransactionID'].min():,} "
        f"to "
        f"{transaction_df['TransactionID'].max():,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RFGN IEEE-CIS DATASET AUDIT")
    print("=" * 70)

    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Raw data dir: {RAW_DATA_DIR}")

    transaction_df = load_transaction_data()

    identity_df = load_identity_data()

    print_dataset_overview(
        transaction_df,
        identity_df
    )

    audit_labels(
        transaction_df
    )

    audit_transaction_ids(
        transaction_df,
        identity_df
    )

    audit_transaction_time(
        transaction_df
    )

    audit_missing_values(
        transaction_df,
        identity_df
    )

    audit_column_types(
        transaction_df,
        identity_df
    )

    audit_relationship_fields(
        transaction_df,
        identity_df
    )

    print()
    print("=" * 70)
    print("DATASET AUDIT COMPLETED")
    print("=" * 70)

    print()
    print("No preprocessing performed.")
    print("No graph construction performed.")
    print("No model training performed.")
    print("Raw dataset was not modified.")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()