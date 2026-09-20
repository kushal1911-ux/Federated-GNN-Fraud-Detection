from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# RFGN
# LEVEL 3 — NON-IID FEDERATED CLIENT PARTITIONING
# STAGE 2 — CLIENT PARTITION GENERATION
# ============================================================
#
# PURPOSE
# -------
# Generate three complete federated training datasets from
# the original 848-column training dataset.
#
# IMPORTANT
# ---------
# 1. Only train.csv is partitioned.
# 2. validation.csv is NOT touched.
# 3. test.csv is NOT touched.
# 4. Original train.csv is NOT modified.
# 5. isFraud is NOT used to assign clients.
# 6. ProductCD is the primary heterogeneity feature.
# 7. Every client receives ALL original columns.
# 8. Every training row belongs to exactly one client.
# 9. TransactionID must remain unique across clients.
# 10. The partition is deterministic and reproducible.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "train.csv"
)

CLIENT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clients"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

CHUNK_SIZE = 50_000

ID_COLUMN = "TransactionID"
TARGET_COLUMN = "isFraud"
TIME_COLUMN = "TransactionDT"
PARTITION_COLUMN = "ProductCD"

CLIENT_NAMES = [
    "client_1",
    "client_2",
    "client_3",
]

NUM_CLIENTS = len(CLIENT_NAMES)


# ============================================================
# CONTROLLED NON-IID ALLOCATION
# ============================================================
#
# Each ProductCD category is distributed differently across
# the three clients.
#
# The proportions intentionally create heterogeneity while
# keeping total client sizes approximately balanced.
#
# IMPORTANT:
# These proportions are based ONLY on ProductCD.
# isFraud is NOT used for allocation.
#
# ============================================================

PRODUCT_ALLOCATION = {

    "C": [0.45, 0.40, 0.15],

    "H": [0.15, 0.50, 0.35],

    "R": [0.25, 0.20, 0.55],

    "S": [0.20, 0.10, 0.70],

    "W": [0.34, 0.33, 0.33],

}


# ============================================================
# PRINT HELPERS
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
# INPUT VALIDATION
# ============================================================

def validate_input():

    print_header(
        "RFGN NON-IID CLIENT PARTITION GENERATION"
    )

    print()
    print("LEVEL 3")
    print("STAGE 2 — CLIENT PARTITION GENERATION")

    print()
    print("Partition strategy:")
    print("  Primary feature :", PARTITION_COLUMN)
    print("  Method          : Controlled non-IID allocation")
    print("  Random seed     :", RANDOM_SEED)
    print("  Clients         :", NUM_CLIENTS)

    print()
    print("IMPORTANT:")
    print("  isFraud is NOT used for client assignment.")
    print("  Validation data will NOT be touched.")
    print("  Test data will NOT be touched.")
    print("  Original train.csv will NOT be modified.")
    print("  All 848 original columns will be preserved.")

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Input:")
    print(TRAIN_FILE)

    if not TRAIN_FILE.exists():

        raise FileNotFoundError(
            f"Training dataset not found:\n{TRAIN_FILE}"
        )

    size_gb = (
        TRAIN_FILE.stat().st_size
        / (1024 ** 3)
    )

    print()
    print(
        f"Input size: {size_gb:.3f} GB"
    )

    print(
        "Training dataset: PRESENT"
    )


# ============================================================
# HEADER VALIDATION
# ============================================================

def validate_header():

    print_section(
        "VALIDATING TRAINING DATASET HEADER"
    )

    header = pd.read_csv(
        TRAIN_FILE,
        nrows=0
    )

    required_columns = [
        ID_COLUMN,
        TARGET_COLUMN,
        TIME_COLUMN,
        PARTITION_COLUMN,
    ]

    missing = [
        column
        for column in required_columns
        if column not in header.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f" - {column}"
                for column in missing
            )
        )

    print(
        "TransactionID : PASS"
    )

    print(
        "isFraud       : PASS"
    )

    print(
        "TransactionDT : PASS"
    )

    print(
        "ProductCD     : PASS"
    )

    print()
    print(
        "Total input columns:",
        len(header.columns)
    )

    if len(header.columns) != 848:

        print()
        print(
            "WARNING:"
        )

        print(
            "Expected 848 columns based on the current"
        )

        print(
            "RFGN cleaned dataset, but found:",
            len(header.columns)
        )

    else:

        print(
            "Full 848-column dataset: PASS"
        )

    return list(header.columns)


# ============================================================
# LOAD ASSIGNMENT INFORMATION
# ============================================================

def load_assignment_information():

    print_section(
        "LOADING CLIENT ASSIGNMENT INFORMATION"
    )

    columns = [
        ID_COLUMN,
        TARGET_COLUMN,
        TIME_COLUMN,
        PARTITION_COLUMN,
    ]

    chunks = []

    chunk_number = 0

    print()
    print(
        "Loading only assignment columns..."
    )

    for chunk in pd.read_csv(
        TRAIN_FILE,
        usecols=columns,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk_number += 1

        chunks.append(chunk)

        print(
            f"\rProcessed chunks: {chunk_number:>4}",
            end=""
        )

    print()

    assignment_df = pd.concat(
        chunks,
        ignore_index=True
    )

    print()
    print(
        "Rows loaded:",
        f"{len(assignment_df):,}"
    )

    print(
        "Assignment information: PASS"
    )

    return assignment_df


# ============================================================
# VALIDATE ASSIGNMENT INFORMATION
# ============================================================

def validate_assignment_information(df):

    print_section(
        "VALIDATING ASSIGNMENT INFORMATION"
    )

    if df[ID_COLUMN].isna().any():

        raise ValueError(
            "Missing TransactionID detected."
        )

    if df[ID_COLUMN].duplicated().any():

        raise ValueError(
            "Duplicate TransactionID detected."
        )

    if df[TARGET_COLUMN].isna().any():

        raise ValueError(
            "Missing isFraud values detected."
        )

    if df[TIME_COLUMN].isna().any():

        raise ValueError(
            "Missing TransactionDT values detected."
        )

    if df[PARTITION_COLUMN].isna().any():

        print(
            "WARNING: Missing ProductCD detected."
        )

        df[PARTITION_COLUMN] = (
            df[PARTITION_COLUMN]
            .fillna("MISSING")
        )

    print(
        "TransactionID uniqueness: PASS"
    )

    print(
        "Fraud target completeness: PASS"
    )

    print(
        "TransactionDT completeness: PASS"
    )

    print(
        "ProductCD completeness: PASS"
    )

    return df


# ============================================================
# ORIGINAL PRODUCT DISTRIBUTION
# ============================================================

def show_original_distribution(df):

    print_section(
        "ORIGINAL PRODUCTCD DISTRIBUTION"
    )

    counts = (
        df[PARTITION_COLUMN]
        .value_counts()
        .sort_index()
    )

    total = len(df)

    print()

    print(
        f"{'ProductCD':<15}"
        f"{'Rows':>15}"
        f"{'Share %':>15}"
    )

    print("-" * 45)

    for category, count in counts.items():

        print(
            f"{str(category):<15}"
            f"{count:>15,}"
            f"{count / total * 100:>14.4f}"
        )

    print("-" * 45)

    print(
        f"{'TOTAL':<15}"
        f"{total:>15,}"
        f"{100:>14.4f}"
    )


# ============================================================
# VALIDATE ALLOCATION CONFIGURATION
# ============================================================

def validate_allocation_configuration(
    categories
):

    print_section(
        "VALIDATING NON-IID ALLOCATION CONFIGURATION"
    )

    missing_categories = [
        category
        for category in categories
        if category not in PRODUCT_ALLOCATION
    ]

    if missing_categories:

        raise ValueError(
            "No allocation defined for:\n"
            + "\n".join(
                f" - {category}"
                for category in missing_categories
            )
        )

    for category, proportions in (
        PRODUCT_ALLOCATION.items()
    ):

        if len(proportions) != NUM_CLIENTS:

            raise ValueError(
                f"Invalid allocation for {category}"
            )

        total = sum(proportions)

        if not np.isclose(
            total,
            1.0
        ):

            raise ValueError(
                f"Allocation for {category} "
                f"must sum to 1.0. "
                f"Found {total}"
            )

        if any(
            p < 0
            for p in proportions
        ):

            raise ValueError(
                f"Negative allocation for {category}"
            )

    print(
        "Allocation configuration: PASS"
    )

    print()

    print(
        "Configured ProductCD allocation:"
    )

    print()

    print(
        f"{'ProductCD':<15}"
        f"{'Client 1':>15}"
        f"{'Client 2':>15}"
        f"{'Client 3':>15}"
    )

    print("-" * 60)

    for category in categories:

        p = PRODUCT_ALLOCATION[
            category
        ]

        print(
            f"{category:<15}"
            f"{p[0] * 100:>14.2f}%"
            f"{p[1] * 100:>14.2f}%"
            f"{p[2] * 100:>14.2f}%"
        )


# ============================================================
# CREATE DETERMINISTIC ASSIGNMENT
# ============================================================

def create_client_assignment(df):

    print_section(
        "CREATING CONTROLLED NON-IID CLIENT ASSIGNMENT"
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    assignment = np.full(
        len(df),
        -1,
        dtype=np.int8
    )

    categories = sorted(
        df[PARTITION_COLUMN]
        .astype(str)
        .unique()
    )

    print()

    for category in categories:

        category_mask = (
            df[PARTITION_COLUMN]
            .astype(str)
            == category
        )

        indices = np.flatnonzero(
            category_mask.to_numpy()
        )

        count = len(indices)

        proportions = (
            PRODUCT_ALLOCATION[category]
        )

        raw_counts = (
            np.array(proportions)
            * count
        )

        client_counts = np.floor(
            raw_counts
        ).astype(int)

        remainder = (
            count
            - client_counts.sum()
        )

        if remainder > 0:

            fractions = (
                raw_counts
                - client_counts
            )

            order = np.argsort(
                -fractions
            )

            for i in range(remainder):

                client_counts[
                    order[i]
                ] += 1

        shuffled_indices = indices.copy()

        rng.shuffle(
            shuffled_indices
        )

        start = 0

        for client_id in range(
            NUM_CLIENTS
        ):

            end = (
                start
                + client_counts[
                    client_id
                ]
            )

            selected = (
                shuffled_indices[
                    start:end
                ]
            )

            assignment[
                selected
            ] = client_id

            start = end

        print(
            f"{category:<10}"
            f"rows={count:>8,} "
            f"allocation="
            f"{client_counts.tolist()}"
        )

    if np.any(
        assignment < 0
    ):

        unassigned = int(
            np.sum(
                assignment < 0
            )
        )

        raise RuntimeError(
            f"{unassigned} rows remain unassigned."
        )

    df = df.copy()

    df["_client"] = assignment

    print()
    print(
        "All rows assigned: PASS"
    )

    return df


# ============================================================
# ASSIGNMENT SUMMARY
# ============================================================

def show_assignment_summary(
    df
):

    print_section(
        "CLIENT ASSIGNMENT SUMMARY"
    )

    total = len(df)

    for client_id, client_name in enumerate(
        CLIENT_NAMES
    ):

        client = df[
            df["_client"] == client_id
        ]

        rows = len(client)

        fraud = int(
            client[TARGET_COLUMN].sum()
        )

        legitimate = (
            rows - fraud
        )

        fraud_rate = (
            fraud / rows * 100
            if rows
            else 0
        )

        print()
        print(
            client_name.upper()
        )

        print(
            "  Rows       :",
            f"{rows:,}"
        )

        print(
            "  Share      :",
            f"{rows / total * 100:.4f}%"
        )

        print(
            "  Legitimate :",
            f"{legitimate:,}"
        )

        print(
            "  Fraud      :",
            f"{fraud:,}"
        )

        print(
            "  Fraud rate :",
            f"{fraud_rate:.4f}%"
        )

        print(
            "  Time min   :",
            int(client[TIME_COLUMN].min())
        )

        print(
            "  Time max   :",
            int(client[TIME_COLUMN].max())
        )

        print()
        print(
            "  ProductCD:"
        )

        counts = (
            client[PARTITION_COLUMN]
            .value_counts()
            .sort_index()
        )

        for category, count in (
            counts.items()
        ):

            print(
                f"    {category}: "
                f"{count:,} "
                f"({count / rows * 100:.2f}%)"
            )


# ============================================================
# ASSIGNMENT INTEGRITY
# ============================================================

def validate_assignment_integrity(
    df
):

    print_section(
        "VALIDATING CLIENT ASSIGNMENT INTEGRITY"
    )

    total = len(df)

    assigned = int(
        (df["_client"] >= 0).sum()
    )

    print(
        "Original rows:",
        f"{total:,}"
    )

    print(
        "Assigned rows:",
        f"{assigned:,}"
    )

    if assigned != total:

        raise RuntimeError(
            "Assigned row count mismatch."
        )

    print(
        "Row assignment preservation: PASS"
    )

    # --------------------------------------------------------
    # TransactionID overlap
    # --------------------------------------------------------

    client_id_sets = []

    for client_id in range(
        NUM_CLIENTS
    ):

        ids = set(
            df.loc[
                df["_client"] == client_id,
                ID_COLUMN
            ]
        )

        client_id_sets.append(ids)

    total_unique_ids = len(
        set(
            df[ID_COLUMN]
        )
    )

    combined_unique_ids = len(
        set().union(
            *client_id_sets
        )
    )

    if combined_unique_ids != total_unique_ids:

        raise RuntimeError(
            "TransactionID uniqueness mismatch."
        )

    overlap_found = False

    for i in range(NUM_CLIENTS):

        for j in range(
            i + 1,
            NUM_CLIENTS
        ):

            overlap = (
                client_id_sets[i]
                & client_id_sets[j]
            )

            if overlap:

                overlap_found = True

                print(
                    f"Client {i+1} / "
                    f"Client {j+1} overlap:",
                    len(overlap)
                )

    if overlap_found:

        raise RuntimeError(
            "Cross-client TransactionID overlap found."
        )

    print(
        "Cross-client TransactionID overlap: 0"
    )

    print(
        "TransactionID isolation: PASS"
    )


# ============================================================
# PREPARE OUTPUT DIRECTORIES
# ============================================================

def prepare_output_directories():

    print_section(
        "PREPARING CLIENT DIRECTORIES"
    )

    CLIENT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    for client_name in CLIENT_NAMES:

        client_dir = (
            CLIENT_ROOT
            / client_name
        )

        client_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    print(
        "Client directories: PASS"
    )


# ============================================================
# SAVE COMPLETE CLIENT DATASETS
# ============================================================

def save_complete_clients(
    assignment_df,
    original_columns
):

    print_section(
        "CREATING COMPLETE 848-COLUMN CLIENT DATASETS"
    )

    # --------------------------------------------------------
    # Build lookup from TransactionID -> client
    # --------------------------------------------------------

    assignment_lookup = (
        assignment_df[
            [
                ID_COLUMN,
                "_client"
            ]
        ]
        .copy()
    )

    assignment_lookup = (
        assignment_lookup
        .set_index(ID_COLUMN)["_client"]
    )

    # --------------------------------------------------------
    # Prepare output files.
    # --------------------------------------------------------

    output_paths = {}

    for client_name in CLIENT_NAMES:

        output_path = (
            CLIENT_ROOT
            / client_name
            / "train.csv"
        )

        output_paths[client_name] = (
            output_path
        )

        if output_path.exists():

            print()
            print(
                "Removing previous incomplete file:"
            )

            print(
                output_path
            )

            output_path.unlink()

    # --------------------------------------------------------
    # Read the FULL training dataset in chunks.
    # --------------------------------------------------------

    print()
    print(
        "Reading the complete training dataset"
    )

    print(
        "and writing every original column to"
    )

    print(
        "the appropriate client."
    )

    print()

    header_written = {
        client_name: False
        for client_name in CLIENT_NAMES
    }

    total_written = {
        client_name: 0
        for client_name in CLIENT_NAMES
    }

    chunk_number = 0

    for chunk in pd.read_csv(
        TRAIN_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk_number += 1

        # ----------------------------------------------------
        # Ensure original columns are preserved.
        # ----------------------------------------------------

        if list(chunk.columns) != original_columns:

            raise RuntimeError(
                "Column structure changed while reading "
                "training data."
            )

        # ----------------------------------------------------
        # Determine client assignment.
        # ----------------------------------------------------

        clients = (
            chunk[ID_COLUMN]
            .map(assignment_lookup)
        )

        if clients.isna().any():

            raise RuntimeError(
                "Could not find client assignment "
                "for one or more TransactionIDs."
            )

        # ----------------------------------------------------
        # Write each client.
        # ----------------------------------------------------

        for client_id, client_name in enumerate(
            CLIENT_NAMES
        ):

            mask = (
                clients.to_numpy()
                == client_id
            )

            if not np.any(mask):

                continue

            client_chunk = (
                chunk.loc[mask]
                .copy()
            )

            # Preserve chronological order inside
            # each client chunk.
            client_chunk = (
                client_chunk
                .sort_values(
                    by=TIME_COLUMN
                )
            )

            output_path = (
                output_paths[
                    client_name
                ]
            )

            client_chunk.to_csv(
                output_path,
                mode="a",
                header=not header_written[
                    client_name
                ],
                index=False
            )

            header_written[
                client_name
            ] = True

            total_written[
                client_name
            ] += len(client_chunk)

        print(
            f"\rProcessed full-data chunks: "
            f"{chunk_number:>4}",
            end=""
        )

    print()
    print()

    # --------------------------------------------------------
    # Validate row counts.
    # --------------------------------------------------------

    expected_total = len(
        assignment_df
    )

    actual_total = sum(
        total_written.values()
    )

    print(
        "Rows expected:",
        f"{expected_total:,}"
    )

    print(
        "Rows written :",
        f"{actual_total:,}"
    )

    if actual_total != expected_total:

        raise RuntimeError(
            "Total client rows do not match "
            "training dataset."
        )

    print(
        "Full dataset row preservation: PASS"
    )

    return output_paths, total_written


# ============================================================
# VERIFY CLIENT FILES
# ============================================================

def verify_client_files(
    output_paths,
    expected_counts,
    original_columns
):

    print_section(
        "VERIFYING COMPLETE CLIENT DATASETS"
    )

    total_rows = 0

    for client_name in CLIENT_NAMES:

        path = output_paths[
            client_name
        ]

        print()
        print(
            client_name.upper()
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing client file:\n{path}"
            )

        size_mb = (
            path.stat().st_size
            / (1024 ** 2)
        )

        print(
            "  File status : PRESENT"
        )

        print(
            f"  File size   : {size_mb:.2f} MB"
        )

        # ----------------------------------------------------
        # Read header first.
        # ----------------------------------------------------

        client_header = pd.read_csv(
            path,
            nrows=0
        )

        if list(
            client_header.columns
        ) != original_columns:

            raise RuntimeError(
                f"{client_name} does not contain "
                "the complete original column set."
            )

        print(
            "  Column count:",
            len(client_header.columns)
        )

        print(
            "  Full feature set: PASS"
        )

        # ----------------------------------------------------
        # Verify row count without loading all columns.
        # ----------------------------------------------------

        row_count = 0

        for chunk in pd.read_csv(
            path,
            usecols=[ID_COLUMN],
            chunksize=CHUNK_SIZE
        ):

            row_count += len(chunk)

        expected = expected_counts[
            client_name
        ]

        print(
            "  Expected rows:",
            f"{expected:,}"
        )

        print(
            "  Actual rows  :",
            f"{row_count:,}"
        )

        if row_count != expected:

            raise RuntimeError(
                f"{client_name} row count mismatch."
            )

        print(
            "  Row count: PASS"
        )

        total_rows += row_count

    print()

    print(
        "Total client rows:",
        f"{total_rows:,}"
    )

    print(
        "All client files verified: PASS"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(
    assignment_df,
    total_written
):

    print_header(
        "RFGN NON-IID CLIENT GENERATION SUMMARY"
    )

    print()

    print(
        "Training transactions:",
        f"{len(assignment_df):,}"
    )

    print(
        "Original training columns:",
        848
    )

    print(
        "Number of clients:",
        NUM_CLIENTS
    )

    print(
        "Partition feature:",
        PARTITION_COLUMN
    )

    print(
        "Partition method:",
        "Controlled non-IID allocation"
    )

    print(
        "Random seed:",
        RANDOM_SEED
    )

    print()

    for client_name in CLIENT_NAMES:

        rows = total_written[
            client_name
        ]

        print(
            f"{client_name.upper():<12}"
            f"{rows:>12,} rows"
        )

    print()

    print(
        "Fraud labels used for assignment: NO"
    )

    print(
        "Validation modified: NO"
    )

    print(
        "Test modified: NO"
    )

    print(
        "Original train.csv modified: NO"
    )

    print(
        "TransactionID overlap: 0"
    )

    print(
        "All original columns preserved: YES"
    )

    print()

    print(
        "LEVEL 3 — STAGE 2: COMPLETE"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    validate_input()

    original_columns = validate_header()

    assignment_df = (
        load_assignment_information()
    )

    assignment_df = (
        validate_assignment_information(
            assignment_df
        )
    )

    show_original_distribution(
        assignment_df
    )

    categories = sorted(
        assignment_df[
            PARTITION_COLUMN
        ]
        .astype(str)
        .unique()
    )

    validate_allocation_configuration(
        categories
    )

    assignment_df = (
        create_client_assignment(
            assignment_df
        )
    )

    show_assignment_summary(
        assignment_df
    )

    validate_assignment_integrity(
        assignment_df
    )

    prepare_output_directories()

    output_paths, total_written = (
        save_complete_clients(
            assignment_df,
            original_columns
        )
    )

    verify_client_files(
        output_paths,
        total_written,
        original_columns
    )

    final_summary(
        assignment_df,
        total_written
    )

    print()
    print("=" * 70)
    print(
        "RFGN NON-IID CLIENT GENERATION COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print()
    print(
        "NEXT:"
    )

    print(
        "LEVEL 3 — STAGE 3"
    )

    print(
        "CLIENT DISTRIBUTION AUDIT"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()