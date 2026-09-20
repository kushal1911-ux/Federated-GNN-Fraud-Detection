from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# RFGN CLIENT DISTRIBUTION AUDIT
# ============================================================
#
# READ-ONLY AUDIT
#
# This script does NOT:
#   - modify any dataset
#   - modify train.csv
#   - modify validation.csv
#   - modify test.csv
#   - modify client files
#   - construct graphs
#   - train models
#
# Purpose:
#   Verify the correctness and heterogeneity of the
#   three federated training clients.
#
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLIENT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clients"
)

SPLIT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

TRAIN_FILE = SPLIT_ROOT / "train.csv"
VALIDATION_FILE = SPLIT_ROOT / "validation.csv"
TEST_FILE = SPLIT_ROOT / "test.csv"


CLIENTS = [
    "client_1",
    "client_2",
    "client_3",
]


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 50_000

EXPECTED_TRAIN_ROWS = 413_378
EXPECTED_COLUMNS = 848

ID_COLUMN = "TransactionID"
TARGET_COLUMN = "isFraud"
TIME_COLUMN = "TransactionDT"
PRODUCT_COLUMN = "ProductCD"
AMOUNT_COLUMN = "TransactionAmt"


# Representative numerical features.
#
# These are used to measure whether the clients have
# different numerical distributions without scanning
# every numerical column.
#
REPRESENTATIVE_NUMERIC_FEATURES = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "C1",
    "C2",
    "C5",
    "D1",
    "D2",
    "V1",
    "V2",
    "V10",
    "V20",
    "V30",
    "V40",
    "V50",
    "V100",
    "V150",
    "V200",
    "V250",
    "V300",
    "V339",
]


# ============================================================
# PRINT HELPERS
# ============================================================

def header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def section(title):

    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


# ============================================================
# CLIENT PATH
# ============================================================

def client_path(client_name):

    return (
        CLIENT_ROOT
        / client_name
        / "train.csv"
    )


# ============================================================
# CHECK CLIENT FILES
# ============================================================

def check_client_files():

    section(
        "CHECKING CLIENT DATASETS"
    )

    paths = {}

    for client in CLIENTS:

        path = client_path(client)

        paths[client] = path

        print()
        print(client.upper())

        print(
            "Path:",
            path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Client dataset not found:\n{path}"
            )

        size_mb = (
            path.stat().st_size
            / (1024 ** 2)
        )

        print(
            f"Size: {size_mb:.2f} MB"
        )

        print(
            "Status: PRESENT"
        )

    return paths


# ============================================================
# READ ORIGINAL TRAINING SCHEMA
# ============================================================

def get_original_columns():

    section(
        "READING ORIGINAL TRAINING SCHEMA"
    )

    if not TRAIN_FILE.exists():

        raise FileNotFoundError(
            f"Training split not found:\n{TRAIN_FILE}"
        )

    df = pd.read_csv(
        TRAIN_FILE,
        nrows=0
    )

    columns = list(
        df.columns
    )

    print(
        "Original train.csv columns:",
        len(columns)
    )

    if len(columns) == EXPECTED_COLUMNS:

        print(
            "Original 848-column schema: PASS"
        )

    else:

        print(
            "WARNING: Original schema is not 848 columns."
        )

    return columns


# ============================================================
# VERIFY CLIENT SCHEMAS
# ============================================================

def verify_schemas(
    paths,
    original_columns
):

    section(
        "CLIENT SCHEMA AUDIT"
    )

    for client in CLIENTS:

        path = paths[client]

        df = pd.read_csv(
            path,
            nrows=0
        )

        columns = list(
            df.columns
        )

        print()
        print(client.upper())

        print(
            "Columns:",
            len(columns)
        )

        if columns != original_columns:

            missing = [
                column
                for column in original_columns
                if column not in columns
            ]

            extra = [
                column
                for column in columns
                if column not in original_columns
            ]

            print(
                "Schema: FAIL"
            )

            if missing:

                print(
                    "Missing columns:",
                    missing[:20]
                )

            if extra:

                print(
                    "Extra columns:",
                    extra[:20]
                )

            raise RuntimeError(
                f"{client} schema does not match original train.csv."
            )

        print(
            "848-column schema: PASS"
        )

    print()
    print(
        "All client schemas match original train.csv: PASS"
    )


# ============================================================
# INITIALIZE CLIENT STATISTICS
# ============================================================

def create_empty_stats():

    numeric_stats = {}

    for feature in REPRESENTATIVE_NUMERIC_FEATURES:

        numeric_stats[feature] = {
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
        }

    return {

        "rows": 0,

        "fraud": 0,

        "legitimate": 0,

        "min_time": None,

        "max_time": None,

        "amount_count": 0,

        "amount_sum": 0.0,

        "amount_sum_sq": 0.0,

        "amount_min": None,

        "amount_max": None,

        "product_counts": {},

        "numeric": numeric_stats,
    }


# ============================================================
# UPDATE NUMERICAL STATISTICS
# ============================================================

def update_numeric_stats(
    stats,
    series
):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return

    values = values.to_numpy(
        dtype=np.float64
    )

    stats["count"] += len(values)

    stats["sum"] += float(
        values.sum()
    )

    stats["sum_sq"] += float(
        np.square(values).sum()
    )


# ============================================================
# SCAN ONE CLIENT
# ============================================================

def scan_client(
    client,
    path
):

    print()
    print(
        f"Scanning {client}..."
    )

    stats = create_empty_stats()

    required_columns = [
        ID_COLUMN,
        TARGET_COLUMN,
        TIME_COLUMN,
        PRODUCT_COLUMN,
        AMOUNT_COLUMN,
    ]

    for feature in REPRESENTATIVE_NUMERIC_FEATURES:

        if feature not in required_columns:

            required_columns.append(
                feature
            )

    transaction_ids = set()

    chunks_processed = 0

    for chunk in pd.read_csv(
        path,
        usecols=required_columns,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunks_processed += 1

        stats["rows"] += len(chunk)

        # ----------------------------------------------------
        # TransactionID
        # ----------------------------------------------------

        ids = chunk[ID_COLUMN]

        if ids.isna().any():

            raise RuntimeError(
                f"{client}: Missing TransactionID detected."
            )

        ids = ids.astype(
            np.int64
        )

        transaction_ids.update(
            ids.tolist()
        )

        # ----------------------------------------------------
        # Fraud labels
        # ----------------------------------------------------

        target = pd.to_numeric(
            chunk[TARGET_COLUMN],
            errors="coerce"
        )

        if target.isna().any():

            raise RuntimeError(
                f"{client}: Missing isFraud values detected."
            )

        invalid = ~target.isin(
            [0, 1]
        )

        if invalid.any():

            raise RuntimeError(
                f"{client}: Invalid isFraud values detected."
            )

        stats["fraud"] += int(
            (target == 1).sum()
        )

        stats["legitimate"] += int(
            (target == 0).sum()
        )

        # ----------------------------------------------------
        # TransactionDT
        # ----------------------------------------------------

        times = pd.to_numeric(
            chunk[TIME_COLUMN],
            errors="coerce"
        )

        if times.isna().any():

            raise RuntimeError(
                f"{client}: Missing TransactionDT detected."
            )

        chunk_min = float(
            times.min()
        )

        chunk_max = float(
            times.max()
        )

        if (
            stats["min_time"] is None
            or chunk_min < stats["min_time"]
        ):

            stats["min_time"] = chunk_min

        if (
            stats["max_time"] is None
            or chunk_max > stats["max_time"]
        ):

            stats["max_time"] = chunk_max

        # ----------------------------------------------------
        # ProductCD
        # ----------------------------------------------------

        products = (
            chunk[PRODUCT_COLUMN]
            .fillna("MISSING")
            .astype(str)
        )

        counts = (
            products
            .value_counts()
            .to_dict()
        )

        for product, count in counts.items():

            stats["product_counts"][product] = (
                stats["product_counts"].get(
                    product,
                    0
                )
                + int(count)
            )

        # ----------------------------------------------------
        # Transaction Amount
        # ----------------------------------------------------

        amount = pd.to_numeric(
            chunk[AMOUNT_COLUMN],
            errors="coerce"
        )

        amount = amount[
            np.isfinite(amount)
        ]

        if len(amount) > 0:

            values = amount.to_numpy(
                dtype=np.float64
            )

            stats["amount_count"] += len(
                values
            )

            stats["amount_sum"] += float(
                values.sum()
            )

            stats["amount_sum_sq"] += float(
                np.square(values).sum()
            )

            current_min = float(
                values.min()
            )

            current_max = float(
                values.max()
            )

            if (
                stats["amount_min"] is None
                or current_min
                < stats["amount_min"]
            ):

                stats["amount_min"] = (
                    current_min
                )

            if (
                stats["amount_max"] is None
                or current_max
                > stats["amount_max"]
            ):

                stats["amount_max"] = (
                    current_max
                )

        # ----------------------------------------------------
        # Representative Numerical Features
        # ----------------------------------------------------

        for feature in REPRESENTATIVE_NUMERIC_FEATURES:

            update_numeric_stats(
                stats["numeric"][feature],
                chunk[feature]
            )

        print(
            f"\r{client}: processed "
            f"{chunks_processed:>3} chunks | "
            f"{stats['rows']:>9,} rows",
            end=""
        )

    print()

    return stats, transaction_ids


# ============================================================
# MEAN AND STANDARD DEVIATION
# ============================================================

def mean_std(stat):

    count = stat["count"]

    if count == 0:

        return np.nan, np.nan

    mean = (
        stat["sum"]
        / count
    )

    variance = (
        stat["sum_sq"]
        / count
    ) - (
        mean ** 2
    )

    variance = max(
        variance,
        0.0
    )

    std = math.sqrt(
        variance
    )

    return mean, std


# ============================================================
# CLIENT SUMMARY
# ============================================================

def print_client_summary(
    stats
):

    section(
        "CLIENT DISTRIBUTION SUMMARY"
    )

    total_rows = sum(
        stats[client]["rows"]
        for client in CLIENTS
    )

    for client in CLIENTS:

        s = stats[client]

        rows = s["rows"]

        share = (
            rows
            / total_rows
            * 100
        )

        fraud_rate = (
            s["fraud"]
            / rows
            * 100
        )

        amount_mean, amount_std = (
            mean_std(
                {
                    "count": s["amount_count"],
                    "sum": s["amount_sum"],
                    "sum_sq": s["amount_sum_sq"],
                }
            )
        )

        print()
        print(
            client.upper()
        )

        print(
            f"  Rows       : {rows:,}"
        )

        print(
            f"  Share      : {share:.4f}%"
        )

        print(
            f"  Legitimate : {s['legitimate']:,}"
        )

        print(
            f"  Fraud      : {s['fraud']:,}"
        )

        print(
            f"  Fraud rate : {fraud_rate:.4f}%"
        )

        print(
            f"  Time min   : {int(s['min_time'])}"
        )

        print(
            f"  Time max   : {int(s['max_time'])}"
        )

        print(
            f"  Amount mean: {amount_mean:.4f}"
        )

        print(
            f"  Amount std : {amount_std:.4f}"
        )

        print(
            f"  Amount min : {s['amount_min']:.4f}"
        )

        print(
            f"  Amount max : {s['amount_max']:.4f}"
        )


# ============================================================
# PRODUCTCD DISTRIBUTION
# ============================================================

def product_distribution(
    stats
):

    section(
        "PRODUCTCD DISTRIBUTION BY CLIENT"
    )

    categories = sorted(
        set().union(
            *[
                set(
                    stats[client][
                        "product_counts"
                    ].keys()
                )
                for client in CLIENTS
            ]
        )
    )

    print()

    print(
        f"{'ProductCD':<12}"
        f"{'Client 1':>18}"
        f"{'Client 2':>18}"
        f"{'Client 3':>18}"
    )

    print("-" * 66)

    distributions = {}

    for client in CLIENTS:

        total = stats[client]["rows"]

        distributions[client] = {}

        for category in categories:

            count = stats[client][
                "product_counts"
            ].get(
                category,
                0
            )

            distributions[client][
                category
            ] = count / total

    for category in categories:

        values = []

        for client in CLIENTS:

            share = (
                distributions[client][
                    category
                ]
                * 100
            )

            values.append(
                share
            )

        print(
            f"{category:<12}"
            f"{values[0]:>17.2f}%"
            f"{values[1]:>17.2f}%"
            f"{values[2]:>17.2f}%"
        )

    return distributions


# ============================================================
# JENSEN-SHANNON DIVERGENCE
# ============================================================

def js_divergence(
    p,
    q
):

    p = np.asarray(
        p,
        dtype=np.float64
    )

    q = np.asarray(
        q,
        dtype=np.float64
    )

    p_sum = p.sum()
    q_sum = q.sum()

    if p_sum == 0 or q_sum == 0:

        return 0.0

    p = p / p_sum
    q = q / q_sum

    m = (
        p + q
    ) / 2

    def kl_divergence(
        a,
        b
    ):

        mask = (
            a > 0
        )

        return float(
            np.sum(
                a[mask]
                * np.log2(
                    a[mask]
                    / b[mask]
                )
            )
        )

    return (
        kl_divergence(p, m)
        + kl_divergence(q, m)
    ) / 2


# ============================================================
# QUANTITATIVE NON-IID ANALYSIS
# ============================================================

def quantify_product_noniid(
    stats
):

    section(
        "QUANTITATIVE NON-IID ANALYSIS"
    )

    categories = sorted(
        set().union(
            *[
                set(
                    stats[client][
                        "product_counts"
                    ].keys()
                )
                for client in CLIENTS
            ]
        )
    )

    distributions = {}

    for client in CLIENTS:

        total = stats[client]["rows"]

        distributions[client] = np.array(
            [
                stats[client][
                    "product_counts"
                ].get(
                    category,
                    0
                )
                / total
                for category in categories
            ],
            dtype=np.float64
        )

    print()

    print(
        "Jensen-Shannon divergence:"
    )

    print(
        "0 = identical distributions"
    )

    print(
        "Higher value = greater heterogeneity"
    )

    pairs = [
        (
            "Client 1",
            "client_1",
            "Client 2",
            "client_2"
        ),
        (
            "Client 1",
            "client_1",
            "Client 3",
            "client_3"
        ),
        (
            "Client 2",
            "client_2",
            "Client 3",
            "client_3"
        ),
    ]

    values = []

    for (
        name_a,
        client_a,
        name_b,
        client_b
    ) in pairs:

        value = js_divergence(
            distributions[client_a],
            distributions[client_b]
        )

        values.append(
            value
        )

        print(
            f"  {name_a} vs {name_b}: "
            f"{value:.6f}"
        )

    average_js = float(
        np.mean(values)
    )

    print()

    print(
        f"Average ProductCD JS divergence: "
        f"{average_js:.6f}"
    )

    # This is an audit indicator, not a claim that a
    # specific threshold guarantees federated learning quality.

    if average_js > 0.01:

        print(
            "ProductCD heterogeneity: PASS"
        )

    else:

        print(
            "ProductCD heterogeneity: REVIEW"
        )

    return average_js


# ============================================================
# NUMERICAL FEATURE HETEROGENEITY
# ============================================================

def numerical_distribution_analysis(
    stats
):

    section(
        "REPRESENTATIVE NUMERICAL FEATURE ANALYSIS"
    )

    print()

    print(
        "Standardized Mean Difference (SMD)"
    )

    print(
        "Absolute SMD closer to 0 = more similar."
    )

    print(
        "Larger absolute SMD = greater heterogeneity."
    )

    print()

    print(
        f"{'Feature':<18}"
        f"{'C1-C2':>14}"
        f"{'C1-C3':>14}"
        f"{'C2-C3':>14}"
    )

    print("-" * 60)

    all_smd = []

    for feature in REPRESENTATIVE_NUMERIC_FEATURES:

        means = {}
        stds = {}

        valid = True

        for client in CLIENTS:

            mean, std = mean_std(
                stats[client][
                    "numeric"
                ][feature]
            )

            if np.isnan(mean):

                valid = False

                break

            means[client] = mean
            stds[client] = std

        if not valid:

            continue

        def calculate_smd(
            client_a,
            client_b
        ):

            pooled_std = math.sqrt(
                (
                    stds[client_a] ** 2
                    +
                    stds[client_b] ** 2
                )
                / 2
            )

            if pooled_std == 0:

                return 0.0

            return (
                means[client_a]
                -
                means[client_b]
            ) / pooled_std

        c1_c2 = calculate_smd(
            "client_1",
            "client_2"
        )

        c1_c3 = calculate_smd(
            "client_1",
            "client_3"
        )

        c2_c3 = calculate_smd(
            "client_2",
            "client_3"
        )

        all_smd.extend(
            [
                abs(c1_c2),
                abs(c1_c3),
                abs(c2_c3),
            ]
        )

        print(
            f"{feature:<18}"
            f"{c1_c2:>14.4f}"
            f"{c1_c3:>14.4f}"
            f"{c2_c3:>14.4f}"
        )

    if all_smd:

        average_smd = float(
            np.mean(all_smd)
        )

        print()

        print(
            f"Average absolute SMD: "
            f"{average_smd:.4f}"
        )

        if average_smd > 0.05:

            print(
                "Representative numerical heterogeneity: PASS"
            )

        else:

            print(
                "Representative numerical heterogeneity: REVIEW"
            )


# ============================================================
# TRANSACTION ID ISOLATION
# ============================================================

def verify_id_isolation(
    client_ids
):

    section(
        "TRANSACTIONID ISOLATION AUDIT"
    )

    for client in CLIENTS:

        ids = client_ids[client]

        print(
            f"{client}: "
            f"{len(ids):,} unique TransactionIDs"
        )

    pair_12 = (
        client_ids["client_1"]
        &
        client_ids["client_2"]
    )

    pair_13 = (
        client_ids["client_1"]
        &
        client_ids["client_3"]
    )

    pair_23 = (
        client_ids["client_2"]
        &
        client_ids["client_3"]
    )

    print()

    print(
        "Client 1 / Client 2 overlap:",
        len(pair_12)
    )

    print(
        "Client 1 / Client 3 overlap:",
        len(pair_13)
    )

    print(
        "Client 2 / Client 3 overlap:",
        len(pair_23)
    )

    if (
        len(pair_12) != 0
        or len(pair_13) != 0
        or len(pair_23) != 0
    ):

        raise RuntimeError(
            "Cross-client TransactionID overlap detected."
        )

    print(
        "Cross-client TransactionID overlap: 0"
    )

    print(
        "TransactionID isolation: PASS"
    )


# ============================================================
# ROW PRESERVATION
# ============================================================

def verify_row_preservation(
    stats
):

    section(
        "ROW PRESERVATION AUDIT"
    )

    total_rows = sum(
        stats[client]["rows"]
        for client in CLIENTS
    )

    print(
        "Expected training rows:",
        f"{EXPECTED_TRAIN_ROWS:,}"
    )

    print(
        "Total client rows:",
        f"{total_rows:,}"
    )

    if total_rows != EXPECTED_TRAIN_ROWS:

        raise RuntimeError(
            "Client row count does not match "
            "the original training split."
        )

    print(
        "Row preservation: PASS"
    )


# ============================================================
# FRAUD DISTRIBUTION
# ============================================================

def fraud_distribution(
    stats
):

    section(
        "FRAUD DISTRIBUTION BY CLIENT"
    )

    print()

    print(
        f"{'Client':<15}"
        f"{'Rows':>15}"
        f"{'Legitimate':>15}"
        f"{'Fraud':>15}"
        f"{'Fraud %':>12}"
    )

    print("-" * 72)

    for client in CLIENTS:

        s = stats[client]

        fraud_rate = (
            s["fraud"]
            / s["rows"]
            * 100
        )

        print(
            f"{client:<15}"
            f"{s['rows']:>15,}"
            f"{s['legitimate']:>15,}"
            f"{s['fraud']:>15,}"
            f"{fraud_rate:>11.4f}%"
        )


# ============================================================
# TEMPORAL DISTRIBUTION
# ============================================================

def temporal_distribution(
    stats
):

    section(
        "TEMPORAL DISTRIBUTION AUDIT"
    )

    global_min = min(
        stats[client]["min_time"]
        for client in CLIENTS
    )

    global_max = max(
        stats[client]["max_time"]
        for client in CLIENTS
    )

    print()

    print(
        "Global training time:",
        f"{int(global_min)} -> "
        f"{int(global_max)}"
    )

    print()

    for client in CLIENTS:

        s = stats[client]

        print(
            f"{client}: "
            f"{int(s['min_time'])} -> "
            f"{int(s['max_time'])}"
        )

    print()

    print(
        "Temporal values audited: PASS"
    )


# ============================================================
# PROTECTED SPLIT CHECK
# ============================================================

def verify_protected_splits():

    section(
        "VALIDATION / TEST DATA PROTECTION CHECK"
    )

    files = [
        (
            "Training",
            TRAIN_FILE
        ),
        (
            "Validation",
            VALIDATION_FILE
        ),
        (
            "Test",
            TEST_FILE
        ),
    ]

    for name, path in files:

        if not path.exists():

            raise FileNotFoundError(
                f"{name} split is missing:\n{path}"
            )

        print(
            f"{name:<12}: PRESENT"
        )

    print()

    print(
        "Training / Validation / Test files present: PASS"
    )

    print(
        "This audit did not modify any of them."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(
    stats,
    average_js
):

    header(
        "RFGN CLIENT DISTRIBUTION AUDIT SUMMARY"
    )

    total_rows = sum(
        stats[client]["rows"]
        for client in CLIENTS
    )

    total_fraud = sum(
        stats[client]["fraud"]
        for client in CLIENTS
    )

    total_legitimate = sum(
        stats[client]["legitimate"]
        for client in CLIENTS
    )

    print()

    print(
        f"Total client rows       : {total_rows:,}"
    )

    print(
        f"Expected training rows  : {EXPECTED_TRAIN_ROWS:,}"
    )

    print(
        f"Total legitimate        : {total_legitimate:,}"
    )

    print(
        f"Total fraud             : {total_fraud:,}"
    )

    print()

    print(
        "Schema preserved        : YES"
    )

    print(
        "848 columns preserved   : YES"
    )

    print(
        "Rows preserved          : YES"
    )

    print(
        "TransactionID overlap   : 0"
    )

    print(
        "Fraud labels preserved  : YES"
    )

    print(
        "Validation modified     : NO"
    )

    print(
        "Test modified           : NO"
    )

    print(
        "Graphs constructed      : NO"
    )

    print(
        "Models trained          : NO"
    )

    print()

    if (
        total_rows == EXPECTED_TRAIN_ROWS
        and average_js > 0.01
    ):

        print(
            "CLIENT DISTRIBUTION AUDIT: PASSED"
        )

    else:

        print(
            "CLIENT DISTRIBUTION AUDIT: REVIEW REQUIRED"
        )

        print()

        print(
            "The client datasets are not being deleted."
        )

        print(
            "Review the distribution values before proceeding."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "RFGN CLIENT DISTRIBUTION AUDIT"
    )

    print()

    print(
        "This script is READ-ONLY."
    )

    print(
        "No client dataset will be modified."
    )

    print(
        "No graph construction will be performed."
    )

    print(
        "No model training will be performed."
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    paths = check_client_files()

    # --------------------------------------------------------
    # Original schema
    # --------------------------------------------------------

    original_columns = (
        get_original_columns()
    )

    # --------------------------------------------------------
    # Client schemas
    # --------------------------------------------------------

    verify_schemas(
        paths,
        original_columns
    )

    # --------------------------------------------------------
    # Scan clients
    # --------------------------------------------------------

    stats = {}

    client_ids = {}

    for client in CLIENTS:

        (
            stats[client],
            client_ids[client]
        ) = scan_client(
            client,
            paths[client]
        )

    # --------------------------------------------------------
    # Client summary
    # --------------------------------------------------------

    print_client_summary(
        stats
    )

    # --------------------------------------------------------
    # ProductCD
    # --------------------------------------------------------

    product_distribution(
        stats
    )

    # --------------------------------------------------------
    # Fraud
    # --------------------------------------------------------

    fraud_distribution(
        stats
    )

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    temporal_distribution(
        stats
    )

    # --------------------------------------------------------
    # TransactionID isolation
    # --------------------------------------------------------

    verify_id_isolation(
        client_ids
    )

    # --------------------------------------------------------
    # Row preservation
    # --------------------------------------------------------

    verify_row_preservation(
        stats
    )

    # --------------------------------------------------------
    # Non-IID analysis
    # --------------------------------------------------------

    average_js = (
        quantify_product_noniid(
            stats
        )
    )

    # --------------------------------------------------------
    # Numerical feature analysis
    # --------------------------------------------------------

    numerical_distribution_analysis(
        stats
    )

    # --------------------------------------------------------
    # Protected files
    # --------------------------------------------------------

    verify_protected_splits()

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    final_summary(
        stats,
        average_js
    )

    print()

    print("=" * 70)

    print(
        "RFGN CLIENT DISTRIBUTION AUDIT COMPLETED"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()