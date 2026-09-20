"""
RFGN DATABASE MODELS

Defines the DuckDB schema used by the real-time
fraud detection pipeline.
"""

from __future__ import annotations

from database.db import get_connection


# ============================================================
# TABLE NAMES
# ============================================================

TRANSACTIONS_TABLE = "transactions"

DETECTION_RESULTS_TABLE = "detection_results"

FEEDBACK_TABLE = "feedback"


# ============================================================
# CREATE TRANSACTIONS TABLE
# ============================================================

def create_transactions_table() -> None:

    connection = get_connection()

    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS
        {TRANSACTIONS_TABLE} (

            transaction_id VARCHAR PRIMARY KEY,

            transaction_dt DOUBLE NOT NULL,

            transaction_data JSON,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


# ============================================================
# CREATE DETECTION RESULTS TABLE
# ============================================================

def create_detection_results_table() -> None:

    connection = get_connection()

    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS
        {DETECTION_RESULTS_TABLE} (

            result_id VARCHAR PRIMARY KEY,

            transaction_id VARCHAR NOT NULL,

            fraud_probability DOUBLE,

            dqn_threshold DOUBLE,

            decision VARCHAR,

            model_version VARCHAR,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


# ============================================================
# CREATE FEEDBACK TABLE
# ============================================================

def create_feedback_table() -> None:

    connection = get_connection()

    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS
        {FEEDBACK_TABLE} (

            feedback_id VARCHAR PRIMARY KEY,

            transaction_id VARCHAR NOT NULL,

            actual_label INTEGER,

            feedback_source VARCHAR,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


# ============================================================
# CREATE ALL TABLES
# ============================================================

def create_all_tables() -> None:

    create_transactions_table()

    create_detection_results_table()

    create_feedback_table()


# ============================================================
# GET TABLE NAMES
# ============================================================

def get_table_names() -> list[str]:

    connection = get_connection()

    rows = connection.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
        """
    ).fetchall()

    return [
        row[0]
        for row in rows
    ]


# ============================================================
# SELF TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 70)
    print("RFGN DATABASE MODEL TEST")
    print("=" * 70)

    create_all_tables()

    tables = get_table_names()

    required_tables = {
        TRANSACTIONS_TABLE,
        DETECTION_RESULTS_TABLE,
        FEEDBACK_TABLE,
    }

    missing = (
        required_tables
        - set(tables)
    )

    if missing:

        raise RuntimeError(
            "Missing database tables: "
            f"{sorted(missing)}"
        )

    print()

    print(
        "Transactions table       : PASS"
    )

    print(
        "Detection results table  : PASS"
    )

    print(
        "Feedback table           : PASS"
    )

    print()

    print("Tables created:")

    for table in tables:

        print(
            f"  - {table}"
        )

    print()

    print(
        "Database schema: PASS"
    )

    print("=" * 70)

    return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_self_test()