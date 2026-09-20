from pathlib import Path

import pandas as pd


# ============================================================
# RFGN — IEEE-CIS DATASET MERGE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRANSACTION_FILE = RAW_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DIR / "train_identity.csv"

OUTPUT_FILE = PROCESSED_DIR / "ieee_merged.csv"


# ============================================================
# EXPECTED DATASET INFORMATION
# ============================================================

EXPECTED_TRANSACTION_ROWS = 590_540
EXPECTED_TRANSACTION_COLUMNS = 394

EXPECTED_IDENTITY_ROWS = 144_233
EXPECTED_IDENTITY_COLUMNS = 41


# ============================================================
# HELPER
# ============================================================

def check_file(path: Path) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"Expected a file but found something else:\n{path}"
        )


# ============================================================
# LOAD TRANSACTION DATA
# ============================================================

def load_transaction_data():

    print("=" * 70)
    print("LOADING TRANSACTION DATA")
    print("=" * 70)

    check_file(TRANSACTION_FILE)

    df = pd.read_csv(
        TRANSACTION_FILE,
        low_memory=False
    )

    print()
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    if len(df) != EXPECTED_TRANSACTION_ROWS:
        raise ValueError(
            "Unexpected transaction row count. "
            f"Expected {EXPECTED_TRANSACTION_ROWS:,}, "
            f"got {len(df):,}"
        )

    if len(df.columns) != EXPECTED_TRANSACTION_COLUMNS:
        raise ValueError(
            "Unexpected transaction column count. "
            f"Expected {EXPECTED_TRANSACTION_COLUMNS}, "
            f"got {len(df.columns)}"
        )

    required_columns = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required transaction columns: {missing}"
        )

    print()
    print("Transaction dataset: PASS")

    return df


# ============================================================
# LOAD IDENTITY DATA
# ============================================================

def load_identity_data():

    print()
    print("=" * 70)
    print("LOADING IDENTITY DATA")
    print("=" * 70)

    check_file(IDENTITY_FILE)

    df = pd.read_csv(
        IDENTITY_FILE,
        low_memory=False
    )

    print()
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    if len(df) != EXPECTED_IDENTITY_ROWS:
        raise ValueError(
            "Unexpected identity row count. "
            f"Expected {EXPECTED_IDENTITY_ROWS:,}, "
            f"got {len(df):,}"
        )

    if len(df.columns) != EXPECTED_IDENTITY_COLUMNS:
        raise ValueError(
            "Unexpected identity column count. "
            f"Expected {EXPECTED_IDENTITY_COLUMNS}, "
            f"got {len(df.columns)}"
        )

    if "TransactionID" not in df.columns:
        raise ValueError(
            "TransactionID is missing from identity dataset."
        )

    print()
    print("Identity dataset: PASS")

    return df


# ============================================================
# CHECK IDS BEFORE MERGE
# ============================================================

def check_transaction_ids(
    transaction_df,
    identity_df
):

    print()
    print("=" * 70)
    print("CHECKING TRANSACTION IDs")
    print("=" * 70)

    transaction_duplicates = (
        transaction_df["TransactionID"]
        .duplicated()
        .sum()
    )

    identity_duplicates = (
        identity_df["TransactionID"]
        .duplicated()
        .sum()
    )

    print(
        "Transaction duplicates : "
        f"{transaction_duplicates:,}"
    )

    print(
        "Identity duplicates    : "
        f"{identity_duplicates:,}"
    )

    if transaction_duplicates != 0:
        raise ValueError(
            "Duplicate TransactionIDs found "
            "in transaction data."
        )

    if identity_duplicates != 0:
        raise ValueError(
            "Duplicate TransactionIDs found "
            "in identity data."
        )

    transaction_ids = set(
        transaction_df["TransactionID"]
    )

    identity_ids = set(
        identity_df["TransactionID"]
    )

    matched_ids = (
        transaction_ids
        .intersection(identity_ids)
    )

    print()
    print(
        "Matching TransactionIDs : "
        f"{len(matched_ids):,}"
    )

    print("TransactionID integrity: PASS")

    return len(matched_ids)


# ============================================================
# MERGE DATASETS
# ============================================================

def merge_datasets(
    transaction_df,
    identity_df
):

    print()
    print("=" * 70)
    print("MERGING TRANSACTION + IDENTITY")
    print("=" * 70)

    print()
    print("Merge type: LEFT JOIN")
    print("Join key : TransactionID")

    merged_df = transaction_df.merge(
        identity_df,
        on="TransactionID",
        how="left",
        validate="one_to_one"
    )

    print()
    print(
        f"Merged rows    : "
        f"{len(merged_df):,}"
    )

    print(
        f"Merged columns : "
        f"{len(merged_df.columns):,}"
    )

    if len(merged_df) != len(transaction_df):
        raise ValueError(
            "LEFT JOIN changed transaction row count."
        )

    if (
        merged_df["TransactionID"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate TransactionIDs detected "
            "after merge."
        )

    print()
    print("Row preservation: PASS")
    print("Merged ID uniqueness: PASS")

    return merged_df


# ============================================================
# IDENTITY COVERAGE
# ============================================================

def calculate_identity_coverage(
    merged_df
):

    print()
    print("=" * 70)
    print("IDENTITY COVERAGE")
    print("=" * 70)

    identity_columns = [
        column
        for column in merged_df.columns
        if column.startswith("id_")
        or column in [
            "DeviceType",
            "DeviceInfo",
        ]
    ]

    if not identity_columns:
        print(
            "No identity/device columns detected."
        )
        return

    identity_present = (
        merged_df[identity_columns]
        .notna()
        .any(axis=1)
    )

    count = int(
        identity_present.sum()
    )

    total = len(merged_df)

    percentage = (
        count / total * 100
    )

    print(
        "Transactions with identity/device data: "
        f"{count:,} / {total:,}"
    )

    print(
        f"Identity coverage: {percentage:.4f}%"
    )


# ============================================================
# LABEL VERIFICATION
# ============================================================

def verify_labels(
    original_df,
    merged_df
):

    print()
    print("=" * 70)
    print("VERIFYING FRAUD LABELS")
    print("=" * 70)

    original_labels = (
        original_df["isFraud"]
        .reset_index(drop=True)
    )

    merged_labels = (
        merged_df["isFraud"]
        .reset_index(drop=True)
    )

    if not original_labels.equals(
        merged_labels
    ):
        raise ValueError(
            "Fraud labels changed during merge."
        )

    print(
        "Fraud labels preserved: PASS"
    )

    fraud_count = int(
        merged_df["isFraud"].sum()
    )

    legitimate_count = (
        len(merged_df)
        - fraud_count
    )

    print(
        f"Legitimate : {legitimate_count:,}"
    )

    print(
        f"Fraud      : {fraud_count:,}"
    )


# ============================================================
# SAVE MERGED DATASET
# ============================================================

def save_dataset(
    merged_df
):

    print()
    print("=" * 70)
    print("SAVING MERGED DATASET")
    print("=" * 70)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    merged_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    if not OUTPUT_FILE.exists():
        raise IOError(
            "Merged dataset was not created."
        )

    size_mb = (
        OUTPUT_FILE.stat().st_size
        / (1024 ** 2)
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print(
        f"File size: {size_mb:.2f} MB"
    )

    print()
    print("Save operation: PASS")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RFGN IEEE-CIS DATASET MERGE")
    print("=" * 70)

    print()
    print(
        "Raw datasets will NOT be modified."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    transaction_df = (
        load_transaction_data()
    )

    identity_df = (
        load_identity_data()
    )

    # --------------------------------------------------------
    # ID verification
    # --------------------------------------------------------

    matched_ids = check_transaction_ids(
        transaction_df,
        identity_df
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged_df = merge_datasets(
        transaction_df,
        identity_df
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    calculate_identity_coverage(
        merged_df
    )

    # --------------------------------------------------------
    # Label verification
    # --------------------------------------------------------

    verify_labels(
        transaction_df,
        merged_df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_dataset(
        merged_df
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MERGE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print()
    print(
        f"Transactions        : "
        f"{len(transaction_df):,}"
    )

    print(
        f"Identity records    : "
        f"{len(identity_df):,}"
    )

    print(
        f"Matched IDs         : "
        f"{matched_ids:,}"
    )

    print(
        f"Merged rows         : "
        f"{len(merged_df):,}"
    )

    print(
        f"Merged columns      : "
        f"{len(merged_df.columns):,}"
    )

    print()
    print(
        "Raw datasets modified: NO"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()