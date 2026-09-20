from pathlib import Path

import pandas as pd


# ============================================================
# RFGN — IEEE-CIS TEMPORAL DATA SPLITTING
# ============================================================
#
# Purpose:
#   Create chronological Train / Validation / Test datasets.
#
# Split:
#   70% Train
#   15% Validation
#   15% Test
#
# IMPORTANT:
#   - No random splitting
#   - No shuffling
#   - Original cleaned dataset is NOT modified
#   - Test data remains completely unseen by training stages
#
# Input:
#   data/processed/ieee_cleaned.csv
#
# Output:
#   data/processed/splits/train.csv
#   data/processed/splits/validation.csv
#   data/processed/splits/test.csv
#
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INPUT_FILE = (
    PROCESSED_DIR
    / "ieee_cleaned.csv"
)

SPLIT_DIR = (
    PROCESSED_DIR
    / "splits"
)

TRAIN_FILE = (
    SPLIT_DIR
    / "train.csv"
)

VALIDATION_FILE = (
    SPLIT_DIR
    / "validation.csv"
)

TEST_FILE = (
    SPLIT_DIR
    / "test.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

ID_COLUMN = "TransactionID"
TARGET_COLUMN = "isFraud"
TIME_COLUMN = "TransactionDT"


# ============================================================
# EXPECTED DATASET
# ============================================================

EXPECTED_ROWS = 590_540
EXPECTED_COLUMNS = 848


# ============================================================
# HELPERS
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
# VALIDATE CONFIGURATION
# ============================================================

def validate_configuration():

    print_header("RFGN IEEE-CIS TEMPORAL DATA SPLITTING")

    print()
    print("Split strategy:")
    print("  Train      :", f"{TRAIN_RATIO * 100:.0f}%")
    print("  Validation :", f"{VALIDATION_RATIO * 100:.0f}%")
    print("  Test       :", f"{TEST_RATIO * 100:.0f}%")

    total_ratio = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if abs(total_ratio - 1.0) > 1e-9:

        raise ValueError(
            "Train + Validation + Test ratios "
            "must equal 1.0."
        )

    print()
    print("Split ratio configuration: PASS")


# ============================================================
# CHECK INPUT FILE
# ============================================================

def check_input_file():

    print_section("CHECKING INPUT DATASET")

    print("Project root :", PROJECT_ROOT)
    print("Input file   :", INPUT_FILE)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Cleaned dataset not found:\n{INPUT_FILE}"
        )

    file_size_gb = (
        INPUT_FILE.stat().st_size
        / (1024 ** 3)
    )

    print(
        f"File size    : {file_size_gb:.3f} GB"
    )

    print("Input status : PRESENT")


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print_section("LOADING CLEANED DATASET")

    print("Loading...")
    print("This may take some time.")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print()
    print("Rows    :", f"{len(df):,}")
    print("Columns :", f"{len(df.columns):,}")

    if len(df) != EXPECTED_ROWS:

        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"found {len(df):,}."
        )

    if len(df.columns) != EXPECTED_COLUMNS:

        raise ValueError(
            f"Expected {EXPECTED_COLUMNS} columns, "
            f"found {len(df.columns)}."
        )

    print()
    print("Dataset dimensions: PASS")

    return df


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

def validate_columns(df):

    print_section("VALIDATING REQUIRED COLUMNS")

    required_columns = [
        ID_COLUMN,
        TARGET_COLUMN,
        TIME_COLUMN
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    for column in required_columns:

        print(
            f"{column:<20} PASS"
        )


# ============================================================
# VALIDATE ORIGINAL DATA
# ============================================================

def validate_original_dataset(df):

    print_section("VALIDATING ORIGINAL DATASET")

    # --------------------------------------------------------
    # TransactionID
    # --------------------------------------------------------

    duplicate_ids = int(
        df[ID_COLUMN].duplicated().sum()
    )

    missing_ids = int(
        df[ID_COLUMN].isna().sum()
    )

    if duplicate_ids != 0:

        raise ValueError(
            "Duplicate TransactionIDs detected."
        )

    if missing_ids != 0:

        raise ValueError(
            "Missing TransactionIDs detected."
        )

    print(
        "TransactionID uniqueness: PASS"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target_values = set(
        df[TARGET_COLUMN]
        .dropna()
        .unique()
        .tolist()
    )

    if target_values != {0, 1}:

        raise ValueError(
            "isFraud is not binary."
        )

    print(
        "Binary fraud target: PASS"
    )

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    missing_time = int(
        df[TIME_COLUMN].isna().sum()
    )

    if missing_time != 0:

        raise ValueError(
            "TransactionDT contains missing values."
        )

    print(
        "TransactionDT completeness: PASS"
    )

    # --------------------------------------------------------
    # Chronological order
    # --------------------------------------------------------

    backwards = int(
        (
            df[TIME_COLUMN]
            .diff()
            < 0
        ).sum()
    )

    if backwards != 0:

        print(
            "Dataset chronological order: "
            "REQUIRES SORTING"
        )

    else:

        print(
            "Dataset chronological order: PASS"
        )


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

def sort_chronologically(df):

    print_section("SORTING TRANSACTIONS BY TIME")

    print(
        "Sorting by TransactionDT..."
    )

    # Stable sorting is important when multiple
    # transactions have the same TransactionDT.
    df = df.sort_values(
        by=[
            TIME_COLUMN,
            ID_COLUMN
        ],
        kind="mergesort"
    ).reset_index(
        drop=True
    )

    print(
        "Chronological sorting: PASS"
    )

    # Verify no backward time movement remains.

    backwards = int(
        (
            df[TIME_COLUMN]
            .diff()
            < 0
        ).sum()
    )

    if backwards != 0:

        raise ValueError(
            "Chronological sorting failed."
        )

    print(
        "Final temporal order: PASS"
    )

    return df


# ============================================================
# CALCULATE SPLIT INDICES
# ============================================================

def calculate_split_indices(total_rows):

    train_end = int(
        total_rows * TRAIN_RATIO
    )

    validation_end = (
        train_end
        + int(total_rows * VALIDATION_RATIO)
    )

    # Test receives all remaining rows.
    # This guarantees that the three partitions
    # sum exactly to the original row count.

    test_end = total_rows

    return (
        train_end,
        validation_end,
        test_end
    )


# ============================================================
# CREATE SPLITS
# ============================================================

def create_splits(df):

    print_section("CREATING CHRONOLOGICAL SPLITS")

    total_rows = len(df)

    (
        train_end,
        validation_end,
        test_end
    ) = calculate_split_indices(
        total_rows
    )

    print()
    print("Total rows      :", f"{total_rows:,}")
    print("Train end index :", f"{train_end:,}")
    print(
        "Validation end  :",
        f"{validation_end:,}"
    )
    print("Test end index  :", f"{test_end:,}")

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_df = df.iloc[
        :train_end
    ].copy()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_df = df.iloc[
        validation_end:test_end
    ].copy()

    print()
    print(
        "Train rows      :",
        f"{len(train_df):,}"
    )

    print(
        "Validation rows :",
        f"{len(validation_df):,}"
    )

    print(
        "Test rows       :",
        f"{len(test_df):,}"
    )

    # --------------------------------------------------------
    # Row count verification
    # --------------------------------------------------------

    combined_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    if combined_rows != total_rows:

        raise ValueError(
            "Split row counts do not sum to "
            "original dataset."
        )

    print()
    print(
        "Row count preservation: PASS"
    )

    return (
        train_df,
        validation_df,
        test_df
    )


# ============================================================
# TEMPORAL BOUNDARY ANALYSIS
# ============================================================

def analyze_temporal_boundaries(
    train_df,
    validation_df,
    test_df
):

    print_section("TEMPORAL SPLIT BOUNDARY ANALYSIS")

    train_min = train_df[TIME_COLUMN].min()
    train_max = train_df[TIME_COLUMN].max()

    validation_min = (
        validation_df[TIME_COLUMN].min()
    )

    validation_max = (
        validation_df[TIME_COLUMN].max()
    )

    test_min = test_df[TIME_COLUMN].min()
    test_max = test_df[TIME_COLUMN].max()

    print()
    print("TRAIN")
    print("  Min TransactionDT:", train_min)
    print("  Max TransactionDT:", train_max)

    print()
    print("VALIDATION")
    print("  Min TransactionDT:", validation_min)
    print("  Max TransactionDT:", validation_max)

    print()
    print("TEST")
    print("  Min TransactionDT:", test_min)
    print("  Max TransactionDT:", test_max)

    # --------------------------------------------------------
    # Strict temporal separation
    # --------------------------------------------------------

    if train_max > validation_min:

        raise ValueError(
            "Training period overlaps validation period."
        )

    if validation_max > test_min:

        raise ValueError(
            "Validation period overlaps test period."
        )

    print()
    print(
        "Train -> Validation temporal separation: PASS"
    )

    print(
        "Validation -> Test temporal separation: PASS"
    )

    print(
        "Temporal isolation: PASS"
    )


# ============================================================
# FRAUD DISTRIBUTION
# ============================================================

def analyze_class_distribution(
    train_df,
    validation_df,
    test_df
):

    print_section("FRAUD DISTRIBUTION BY SPLIT")

    datasets = [
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_df)
    ]

    for name, split_df in datasets:

        legitimate = int(
            (
                split_df[TARGET_COLUMN] == 0
            ).sum()
        )

        fraud = int(
            (
                split_df[TARGET_COLUMN] == 1
            ).sum()
        )

        total = len(split_df)

        fraud_rate = (
            fraud / total * 100
        )

        print()
        print(name)

        print(
            "  Total       :",
            f"{total:,}"
        )

        print(
            "  Legitimate  :",
            f"{legitimate:,}"
        )

        print(
            "  Fraud       :",
            f"{fraud:,}"
        )

        print(
            "  Fraud rate  :",
            f"{fraud_rate:.4f}%"
        )


# ============================================================
# TRANSACTION ID OVERLAP CHECK
# ============================================================

def verify_no_id_overlap(
    train_df,
    validation_df,
    test_df
):

    print_section("TRANSACTION ID OVERLAP AUDIT")

    train_ids = set(
        train_df[ID_COLUMN]
    )

    validation_ids = set(
        validation_df[ID_COLUMN]
    )

    test_ids = set(
        test_df[ID_COLUMN]
    )

    train_validation_overlap = (
        train_ids
        & validation_ids
    )

    train_test_overlap = (
        train_ids
        & test_ids
    )

    validation_test_overlap = (
        validation_ids
        & test_ids
    )

    print(
        "Train / Validation overlap:",
        len(train_validation_overlap)
    )

    print(
        "Train / Test overlap:",
        len(train_test_overlap)
    )

    print(
        "Validation / Test overlap:",
        len(validation_test_overlap)
    )

    if train_validation_overlap:

        raise ValueError(
            "TransactionID overlap between "
            "Train and Validation."
        )

    if train_test_overlap:

        raise ValueError(
            "TransactionID overlap between "
            "Train and Test."
        )

    if validation_test_overlap:

        raise ValueError(
            "TransactionID overlap between "
            "Validation and Test."
        )

    print()
    print(
        "TransactionID isolation: PASS"
    )


# ============================================================
# SAVE SPLITS
# ============================================================

def save_splits(
    train_df,
    validation_df,
    test_df
):

    print_section("SAVING TEMPORAL SPLITS")

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("Saving training split...")
    train_df.to_csv(
        TRAIN_FILE,
        index=False
    )

    print(
        "Saved:",
        TRAIN_FILE
    )

    print()
    print("Saving validation split...")
    validation_df.to_csv(
        VALIDATION_FILE,
        index=False
    )

    print(
        "Saved:",
        VALIDATION_FILE
    )

    print()
    print("Saving test split...")
    test_df.to_csv(
        TEST_FILE,
        index=False
    )

    print(
        "Saved:",
        TEST_FILE
    )

    print()
    print("All split files saved: PASS")


# ============================================================
# VERIFY SAVED FILES
# ============================================================

def verify_saved_files():

    print_section("VERIFYING SAVED SPLIT FILES")

    files = [
        ("TRAIN", TRAIN_FILE),
        ("VALIDATION", VALIDATION_FILE),
        ("TEST", TEST_FILE)
    ]

    for name, path in files:

        if not path.exists():

            raise FileNotFoundError(
                f"{name} split was not created:\n{path}"
            )

        size_mb = (
            path.stat().st_size
            / (1024 ** 2)
        )

        print(
            f"{name:<12} PRESENT"
        )

        print(
            f"             {size_mb:.2f} MB"
        )

    print()
    print(
        "Saved split verification: PASS"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    train_df,
    validation_df,
    test_df
):

    print_header("RFGN TEMPORAL SPLIT SUMMARY")

    total = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    print()
    print(
        "TRAIN"
    )

    print(
        "  Rows:",
        f"{len(train_df):,}"
    )

    print(
        "  Time:",
        train_df[TIME_COLUMN].min(),
        "->",
        train_df[TIME_COLUMN].max()
    )

    print()
    print(
        "VALIDATION"
    )

    print(
        "  Rows:",
        f"{len(validation_df):,}"
    )

    print(
        "  Time:",
        validation_df[TIME_COLUMN].min(),
        "->",
        validation_df[TIME_COLUMN].max()
    )

    print()
    print(
        "TEST"
    )

    print(
        "  Rows:",
        f"{len(test_df):,}"
    )

    print(
        "  Time:",
        test_df[TIME_COLUMN].min(),
        "->",
        test_df[TIME_COLUMN].max()
    )

    print()
    print(
        "Total split rows:",
        f"{total:,}"
    )

    print(
        "Original rows:",
        f"{EXPECTED_ROWS:,}"
    )

    print()
    print(
        "Original cleaned dataset modified: NO"
    )

    print(
        "Random shuffle performed: NO"
    )

    print(
        "Chronological split: YES"
    )

    print(
        "Train / Validation / Test overlap: NO"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    validate_configuration()

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    check_input_file()

    df = load_dataset()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validate_columns(df)

    validate_original_dataset(df)

    # --------------------------------------------------------
    # CHRONOLOGICAL ORDER
    # --------------------------------------------------------

    df = sort_chronologically(df)

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df
    ) = create_splits(df)

    # --------------------------------------------------------
    # TEMPORAL ANALYSIS
    # --------------------------------------------------------

    analyze_temporal_boundaries(
        train_df,
        validation_df,
        test_df
    )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    analyze_class_distribution(
        train_df,
        validation_df,
        test_df
    )

    # --------------------------------------------------------
    # ID ISOLATION
    # --------------------------------------------------------

    verify_no_id_overlap(
        train_df,
        validation_df,
        test_df
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_splits(
        train_df,
        validation_df,
        test_df
    )

    # --------------------------------------------------------
    # VERIFY OUTPUT FILES
    # --------------------------------------------------------

    verify_saved_files()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_final_summary(
        train_df,
        validation_df,
        test_df
    )

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEMPORAL DATA SPLITTING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print()
    print("Next stage:")
    print("NON-IID FEDERATED CLIENT PARTITIONING")

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()