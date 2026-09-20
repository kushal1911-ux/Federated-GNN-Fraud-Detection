from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# RFGN — NON-IID CLIENT PARTITION ANALYSIS
# ============================================================
#
# LEVEL 3
# STAGE 1 — CLIENT PARTITION DESIGN
#
# IMPORTANT:
#   READ-ONLY ANALYSIS
#
# This script:
#   - Reads ONLY train.csv
#   - Does NOT modify train.csv
#   - Does NOT create client datasets
#   - Does NOT modify validation.csv
#   - Does NOT modify test.csv
#
# Purpose:
#   Analyze the training population and identify suitable
#   characteristics for creating reproducible non-IID
#   federated clients.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "train.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 50_000

ID_COLUMN = "TransactionID"
TARGET_COLUMN = "isFraud"
TIME_COLUMN = "TransactionDT"

# Features that can help us understand heterogeneity.
ANALYSIS_COLUMNS = [
    ID_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,

    "TransactionAmt",
    "ProductCD",

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
# CHECK INPUT
# ============================================================

def check_input():

    print_header(
        "RFGN NON-IID CLIENT PARTITION ANALYSIS"
    )

    print()
    print("LEVEL 3")
    print("STAGE 1 — CLIENT PARTITION DESIGN")

    print()
    print("IMPORTANT:")
    print("This is a READ-ONLY analysis.")
    print("No client datasets will be created.")
    print("The training dataset will NOT be modified.")
    print("Validation and test datasets will NOT be touched.")

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Training dataset:")
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
        f"File size: {size_gb:.3f} GB"
    )

    print(
        "Training dataset: PRESENT"
    )


# ============================================================
# VALIDATE HEADER
# ============================================================

def validate_header():

    print_section(
        "VALIDATING TRAINING DATASET HEADER"
    )

    header_df = pd.read_csv(
        TRAIN_FILE,
        nrows=0
    )

    available_columns = set(
        header_df.columns
    )

    missing_columns = [
        column
        for column in ANALYSIS_COLUMNS
        if column not in available_columns
    ]

    if missing_columns:

        raise ValueError(
            "Required analysis columns missing:\n"
            + "\n".join(
                f" - {column}"
                for column in missing_columns
            )
        )

    print(
        "Required analysis columns: PASS"
    )

    print(
        "Total columns in training file:",
        len(header_df.columns)
    )

    print(
        "Columns used for analysis:",
        len(ANALYSIS_COLUMNS)
    )


# ============================================================
# INITIALIZE ACCUMULATORS
# ============================================================

def initialize_accumulators():

    return {

        "rows": 0,

        "fraud": 0,

        "legitimate": 0,

        "missing_time": 0,

        "missing_amount": 0,

        "amount_sum": 0.0,

        "amount_count": 0,

        "amount_min": np.inf,

        "amount_max": -np.inf,

        "product_counts": {},

        "product_fraud": {},

        "card4_counts": {},

        "card4_fraud": {},

        "card6_counts": {},

        "card6_fraud": {},

        "device_type_counts": {},

        "device_type_fraud": {},

        "email_domain_counts": {},

        "email_domain_fraud": {},

        "time_min": np.inf,

        "time_max": -np.inf,

        "time_samples": [],

        "amount_samples": [],

    }


# ============================================================
# UPDATE CATEGORY COUNTS
# ============================================================

def update_category_counts(
    counts,
    fraud_counts,
    series,
    fraud
):

    temp = pd.DataFrame({
        "value": series,
        "fraud": fraud
    })

    temp["value"] = (
        temp["value"]
        .fillna("MISSING")
        .astype(str)
    )

    grouped = (
        temp
        .groupby("value", sort=False)["fraud"]
        .agg(["count", "sum"])
    )

    for value, row in grouped.iterrows():

        value = str(value)

        counts[value] = (
            counts.get(value, 0)
            + int(row["count"])
        )

        fraud_counts[value] = (
            fraud_counts.get(value, 0)
            + int(row["sum"])
        )


# ============================================================
# PROCESS CHUNK
# ============================================================

def process_chunk(
    chunk,
    stats
):

    rows = len(chunk)

    stats["rows"] += rows

    fraud = (
        chunk[TARGET_COLUMN]
        .fillna(0)
        .astype(np.int8)
    )

    fraud_count = int(
        fraud.sum()
    )

    stats["fraud"] += fraud_count

    stats["legitimate"] += (
        rows - fraud_count
    )

    # --------------------------------------------------------
    # Transaction time
    # --------------------------------------------------------

    time = pd.to_numeric(
        chunk[TIME_COLUMN],
        errors="coerce"
    )

    stats["missing_time"] += int(
        time.isna().sum()
    )

    valid_time = time.dropna()

    if len(valid_time) > 0:

        stats["time_min"] = min(
            stats["time_min"],
            float(valid_time.min())
        )

        stats["time_max"] = max(
            stats["time_max"],
            float(valid_time.max())
        )

        # Sample values for approximate distribution analysis.
        sample = valid_time.sample(
            n=min(5_000, len(valid_time)),
            random_state=42
        )

        stats["time_samples"].extend(
            sample.tolist()
        )

    # --------------------------------------------------------
    # Transaction amount
    # --------------------------------------------------------

    amount = pd.to_numeric(
        chunk["TransactionAmt"],
        errors="coerce"
    )

    stats["missing_amount"] += int(
        amount.isna().sum()
    )

    valid_amount = amount.dropna()

    if len(valid_amount) > 0:

        stats["amount_sum"] += float(
            valid_amount.sum()
        )

        stats["amount_count"] += len(
            valid_amount
        )

        stats["amount_min"] = min(
            stats["amount_min"],
            float(valid_amount.min())
        )

        stats["amount_max"] = max(
            stats["amount_max"],
            float(valid_amount.max())
        )

        sample = valid_amount.sample(
            n=min(5_000, len(valid_amount)),
            random_state=42
        )

        stats["amount_samples"].extend(
            sample.tolist()
        )

    # --------------------------------------------------------
    # ProductCD
    # --------------------------------------------------------

    update_category_counts(
        stats["product_counts"],
        stats["product_fraud"],
        chunk["ProductCD"],
        fraud
    )

    # --------------------------------------------------------
    # Card4
    # --------------------------------------------------------

    update_category_counts(
        stats["card4_counts"],
        stats["card4_fraud"],
        chunk["card4"],
        fraud
    )

    # --------------------------------------------------------
    # Card6
    # --------------------------------------------------------

    update_category_counts(
        stats["card6_counts"],
        stats["card6_fraud"],
        chunk["card6"],
        fraud
    )

    # --------------------------------------------------------
    # DeviceType
    # --------------------------------------------------------

    update_category_counts(
        stats["device_type_counts"],
        stats["device_type_fraud"],
        chunk["DeviceType"],
        fraud
    )

    # --------------------------------------------------------
    # Primary email domain
    # --------------------------------------------------------

    update_category_counts(
        stats["email_domain_counts"],
        stats["email_domain_fraud"],
        chunk["P_emaildomain"],
        fraud
    )


# ============================================================
# LOAD AND ANALYZE
# ============================================================

def analyze_training_dataset():

    print_section(
        "ANALYZING TRAINING DATASET"
    )

    print()
    print(
        f"Chunk size: {CHUNK_SIZE:,}"
    )

    print(
        "Only selected analysis columns will be loaded."
    )

    stats = initialize_accumulators()

    chunk_number = 0

    for chunk in pd.read_csv(
        TRAIN_FILE,
        usecols=ANALYSIS_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk_number += 1

        process_chunk(
            chunk,
            stats
        )

        print(
            f"\rProcessed chunks: {chunk_number:>4}",
            end=""
        )

    print()
    print()

    print(
        "Training rows analyzed:",
        f"{stats['rows']:,}"
    )

    print(
        "Training analysis: PASS"
    )

    return stats


# ============================================================
# BASIC DISTRIBUTION
# ============================================================

def report_basic_distribution(stats):

    print_section(
        "TRAINING DATA DISTRIBUTION"
    )

    total = stats["rows"]

    fraud = stats["fraud"]

    legitimate = stats["legitimate"]

    print(
        "Total transactions:",
        f"{total:,}"
    )

    print(
        "Legitimate:",
        f"{legitimate:,}",
        f"({legitimate / total * 100:.4f}%)"
    )

    print(
        "Fraud:",
        f"{fraud:,}",
        f"({fraud / total * 100:.4f}%)"
    )


# ============================================================
# TRANSACTION AMOUNT
# ============================================================

def report_transaction_amount(stats):

    print_section(
        "TRANSACTION AMOUNT ANALYSIS"
    )

    if stats["amount_count"] == 0:

        print(
            "No valid transaction amounts found."
        )

        return

    mean_amount = (
        stats["amount_sum"]
        / stats["amount_count"]
    )

    sample = np.array(
        stats["amount_samples"],
        dtype=np.float64
    )

    print(
        f"Minimum : {stats['amount_min']:.4f}"
    )

    print(
        f"Mean    : {mean_amount:.4f}"
    )

    print(
        f"Maximum : {stats['amount_max']:.4f}"
    )

    if len(sample) > 0:

        print(
            f"Approx. 25% : "
            f"{np.percentile(sample, 25):.4f}"
        )

        print(
            f"Approx. 50% : "
            f"{np.percentile(sample, 50):.4f}"
        )

        print(
            f"Approx. 75% : "
            f"{np.percentile(sample, 75):.4f}"
        )

        print(
            f"Approx. 90% : "
            f"{np.percentile(sample, 90):.4f}"
        )

        print(
            f"Approx. 95% : "
            f"{np.percentile(sample, 95):.4f}"
        )

        print(
            f"Approx. 99% : "
            f"{np.percentile(sample, 99):.4f}"
        )


# ============================================================
# CATEGORY REPORT
# ============================================================

def report_category(
    title,
    counts,
    fraud_counts,
    minimum_display=0
):

    print_section(title)

    total = sum(
        counts.values()
    )

    print(
        f"{'Category':<30}"
        f"{'Count':>12}"
        f"{'Share %':>12}"
        f"{'Fraud':>12}"
        f"{'Fraud %':>12}"
    )

    print("-" * 78)

    sorted_values = sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True
    )

    for value, count in sorted_values:

        if count < minimum_display:
            continue

        fraud = fraud_counts.get(
            value,
            0
        )

        share = (
            count / total * 100
            if total > 0
            else 0
        )

        fraud_rate = (
            fraud / count * 100
            if count > 0
            else 0
        )

        print(
            f"{value[:30]:<30}"
            f"{count:>12,}"
            f"{share:>11.4f}"
            f"{fraud:>12,}"
            f"{fraud_rate:>11.4f}"
        )


# ============================================================
# TIME ANALYSIS
# ============================================================

def report_time_distribution(stats):

    print_section(
        "TRANSACTION TIME ANALYSIS"
    )

    print(
        "Minimum TransactionDT:",
        int(stats["time_min"])
    )

    print(
        "Maximum TransactionDT:",
        int(stats["time_max"])
    )

    print(
        "Time span:",
        int(
            stats["time_max"]
            - stats["time_min"]
        )
    )

    sample = np.array(
        stats["time_samples"],
        dtype=np.float64
    )

    if len(sample) > 0:

        print()
        print(
            "Approximate temporal quantiles:"
        )

        for percentile in [
            25,
            33.33,
            50,
            66.67,
            75
        ]:

            value = np.percentile(
                sample,
                percentile
            )

            print(
                f"  P{percentile:<5}: {value:.0f}"
            )


# ============================================================
# CANDIDATE NON-IID FACTORS
# ============================================================

def analyze_candidate_factors(stats):

    print_section(
        "NON-IID PARTITION CANDIDATE ANALYSIS"
    )

    print()
    print(
        "The following characteristics can potentially"
    )

    print(
        "create heterogeneous federated clients:"
    )

    print()

    print(
        "1. ProductCD"
    )

    print(
        "   Categories:",
        len(stats["product_counts"])
    )

    print(
        "   Candidate: YES"
    )

    print()

    print(
        "2. Card4"
    )

    print(
        "   Categories:",
        len(stats["card4_counts"])
    )

    print(
        "   Candidate: YES"
    )

    print()

    print(
        "3. Card6"
    )

    print(
        "   Categories:",
        len(stats["card6_counts"])
    )

    print(
        "   Candidate: YES"
    )

    print()

    print(
        "4. DeviceType"
    )

    print(
        "   Categories:",
        len(stats["device_type_counts"])
    )

    print(
        "   Candidate: YES"
    )

    print()

    print(
        "5. Primary email domain"
    )

    print(
        "   Categories:",
        len(stats["email_domain_counts"])
    )

    print(
        "   Candidate: YES"
    )

    print()

    print(
        "6. Transaction amount"
    )

    print(
        "   Candidate: YES"
    )

    print()

    print(
        "7. Transaction time"
    )

    print(
        "   Candidate: YES"
    )

    print()

    print(
        "8. Fraud label"
    )

    print(
        "   Candidate for analysis: YES"
    )

    print(
        "   Artificial label-based partitioning: NO"
    )


# ============================================================
# HETEROGENEITY WARNINGS
# ============================================================

def report_warnings(stats):

    print_section(
        "PARTITION DESIGN WARNINGS"
    )

    print(
        "The following rules will apply when creating"
    )

    print(
        "the actual federated clients:"
    )

    print()

    print(
        "1. Validation data will NOT be partitioned."
    )

    print(
        "2. Test data will NOT be partitioned."
    )

    print(
        "3. Only train.csv will be partitioned."
    )

    print(
        "4. TransactionID must remain unique."
    )

    print(
        "5. Every training transaction must belong"
    )

    print(
        "   to exactly one client."
    )

    print(
        "6. Client partitions must be reproducible."
    )

    print(
        "7. We will NOT manufacture fraud labels"
    )

    print(
        "   to artificially improve performance."
    )

    print(
        "8. The final client distribution will be"
    )

    print(
        "   audited quantitatively."
    )


# ============================================================
# MISSING DATA CHECK
# ============================================================

def report_missingness(stats):

    print_section(
        "PARTITION-RELEVANT MISSINGNESS"
    )

    print(
        "Missing TransactionDT:",
        stats["missing_time"]
    )

    print(
        "Missing TransactionAmt:",
        stats["missing_amount"]
    )

    if stats["missing_time"] == 0:

        print(
            "TransactionDT completeness: PASS"
        )

    else:

        print(
            "TransactionDT completeness: REVIEW"
        )


# ============================================================
# DESIGN SUMMARY
# ============================================================

def print_design_summary(stats):

    print_header(
        "NON-IID CLIENT DESIGN SUMMARY"
    )

    print()
    print(
        "Training population:",
        f"{stats['rows']:,}"
    )

    print()
    print(
        "Potential heterogeneity dimensions:"
    )

    print(
        "  • ProductCD"
    )

    print(
        "  • Card characteristics"
    )

    print(
        "  • Device characteristics"
    )

    print(
        "  • Email-domain characteristics"
    )

    print(
        "  • Transaction amount"
    )

    print(
        "  • Transaction time"
    )

    print()
    print(
        "Client creation: NOT PERFORMED"
    )

    print(
        "Dataset modification: NO"
    )

    print(
        "Validation touched: NO"
    )

    print(
        "Test touched: NO"
    )

    print()
    print(
        "Stage 1 analysis: COMPLETE"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    check_input()

    validate_header()

    stats = analyze_training_dataset()

    report_basic_distribution(
        stats
    )

    report_transaction_amount(
        stats
    )

    report_category(
        "PRODUCTCD DISTRIBUTION",
        stats["product_counts"],
        stats["product_fraud"]
    )

    report_category(
        "CARD4 DISTRIBUTION",
        stats["card4_counts"],
        stats["card4_fraud"]
    )

    report_category(
        "CARD6 DISTRIBUTION",
        stats["card6_counts"],
        stats["card6_fraud"]
    )

    report_category(
        "DEVICE TYPE DISTRIBUTION",
        stats["device_type_counts"],
        stats["device_type_fraud"]
    )

    report_category(
        "PRIMARY EMAIL DOMAIN DISTRIBUTION",
        stats["email_domain_counts"],
        stats["email_domain_fraud"]
    )

    report_time_distribution(
        stats
    )

    report_missingness(
        stats
    )

    analyze_candidate_factors(
        stats
    )

    report_warnings(
        stats
    )

    print_design_summary(
        stats
    )

    print()
    print("=" * 70)
    print(
        "RFGN NON-IID CLIENT PARTITION ANALYSIS COMPLETED"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()