"""
RFGN DATABASE QUERIES

Database operations for the real-time fraud detection
pipeline.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

from database.db import get_connection
from database.models import (
    TRANSACTIONS_TABLE,
    DETECTION_RESULTS_TABLE,
    FEEDBACK_TABLE,
)


# ============================================================
# TRANSACTION OPERATIONS
# ============================================================

def insert_transaction(
    transaction_id: str,
    transaction_dt: float,
    transaction_data: Dict[str, Any],
) -> str:

    connection = get_connection()

    payload = json.dumps(
        transaction_data,
        default=str
    )

    connection.execute(
        f"""
        INSERT INTO {TRANSACTIONS_TABLE} (
            transaction_id,
            transaction_dt,
            transaction_data
        )
        VALUES (?, ?, ?)
        """,
        [
            str(transaction_id),
            float(transaction_dt),
            payload,
        ],
    )

    return str(transaction_id)


# ============================================================
# CHECK TRANSACTION
# ============================================================

def transaction_exists(
    transaction_id: str,
) -> bool:

    connection = get_connection()

    result = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TRANSACTIONS_TABLE}
        WHERE transaction_id = ?
        """,
        [str(transaction_id)],
    ).fetchone()

    return int(result[0]) > 0


# ============================================================
# GET TRANSACTION
# ============================================================

def get_transaction(
    transaction_id: str,
) -> Optional[Dict[str, Any]]:

    connection = get_connection()

    row = connection.execute(
        f"""
        SELECT
            transaction_id,
            transaction_dt,
            transaction_data,
            created_at
        FROM {TRANSACTIONS_TABLE}
        WHERE transaction_id = ?
        """,
        [str(transaction_id)],
    ).fetchone()

    if row is None:

        return None

    transaction_data = row[2]

    if isinstance(transaction_data, str):

        try:
            transaction_data = json.loads(
                transaction_data
            )

        except json.JSONDecodeError:

            transaction_data = {
                "raw": transaction_data
            }

    return {
        "transaction_id": row[0],
        "transaction_dt": row[1],
        "transaction_data": transaction_data,
        "created_at": row[3],
    }


# ============================================================
# DETECTION RESULT
# ============================================================

def insert_detection_result(
    transaction_id: str,
    fraud_probability: float,
    dqn_threshold: float,
    decision: str,
    model_version: str,
) -> str:

    result_id = str(
        uuid.uuid4()
    )

    connection = get_connection()

    connection.execute(
        f"""
        INSERT INTO {DETECTION_RESULTS_TABLE} (
            result_id,
            transaction_id,
            fraud_probability,
            dqn_threshold,
            decision,
            model_version
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            result_id,
            str(transaction_id),
            float(fraud_probability),
            float(dqn_threshold),
            str(decision),
            str(model_version),
        ],
    )

    return result_id


# ============================================================
# GET DETECTION RESULT
# ============================================================

def get_detection_result(
    transaction_id: str,
) -> Optional[Dict[str, Any]]:

    connection = get_connection()

    row = connection.execute(
        f"""
        SELECT
            result_id,
            transaction_id,
            fraud_probability,
            dqn_threshold,
            decision,
            model_version,
            created_at
        FROM {DETECTION_RESULTS_TABLE}
        WHERE transaction_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [str(transaction_id)],
    ).fetchone()

    if row is None:

        return None

    return {
        "result_id": row[0],
        "transaction_id": row[1],
        "fraud_probability": row[2],
        "dqn_threshold": row[3],
        "decision": row[4],
        "model_version": row[5],
        "created_at": row[6],
    }


# ============================================================
# FEEDBACK
# ============================================================

def insert_feedback(
    transaction_id: str,
    actual_label: int,
    feedback_source: str = "unknown",
) -> str:

    if actual_label not in {0, 1}:

        raise ValueError(
            "actual_label must be either 0 or 1."
        )

    feedback_id = str(
        uuid.uuid4()
    )

    connection = get_connection()

    connection.execute(
        f"""
        INSERT INTO {FEEDBACK_TABLE} (
            feedback_id,
            transaction_id,
            actual_label,
            feedback_source
        )
        VALUES (?, ?, ?, ?)
        """,
        [
            feedback_id,
            str(transaction_id),
            int(actual_label),
            str(feedback_source),
        ],
    )

    return feedback_id


# ============================================================
# DATABASE SUMMARY
# ============================================================

def get_database_summary() -> Dict[str, int]:

    connection = get_connection()

    transaction_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TRANSACTIONS_TABLE}
        """
    ).fetchone()[0]

    result_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {DETECTION_RESULTS_TABLE}
        """
    ).fetchone()[0]

    feedback_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {FEEDBACK_TABLE}
        """
    ).fetchone()[0]

    return {
        "transactions": int(
            transaction_count
        ),
        "detection_results": int(
            result_count
        ),
        "feedback": int(
            feedback_count
        ),
    }


# ============================================================
# SELF TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 70)
    print("RFGN DATABASE QUERY TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Ensure schema exists
    # --------------------------------------------------------

    from database.models import create_all_tables

    create_all_tables()

    # --------------------------------------------------------
    # Create test transaction
    # --------------------------------------------------------

    transaction_id = (
        "DB-TEST-"
        + uuid.uuid4().hex.upper()
    )

    transaction_data = {
        "TransactionID": transaction_id,
        "TransactionDT": 123456,
        "TransactionAmt": 250.75,
        "card1": 1234,
        "card2": 111,
        "card3": 150,
        "card5": 222,
        "addr1": 100,
        "addr2": 200,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "DeviceInfo": "TestDevice",
    }

    # --------------------------------------------------------
    # Insert transaction
    # --------------------------------------------------------

    insert_transaction(
        transaction_id=transaction_id,
        transaction_dt=123456,
        transaction_data=transaction_data,
    )

    print()
    print(
        "Transaction insertion : PASS"
    )

    # --------------------------------------------------------
    # Existence check
    # --------------------------------------------------------

    if not transaction_exists(
        transaction_id
    ):

        raise RuntimeError(
            "Inserted transaction was not found."
        )

    print(
        "Transaction lookup    : PASS"
    )

    # --------------------------------------------------------
    # Retrieve transaction
    # --------------------------------------------------------

    transaction = get_transaction(
        transaction_id
    )

    if transaction is None:

        raise RuntimeError(
            "Transaction retrieval failed."
        )

    print(
        "Transaction retrieval : PASS"
    )

    # --------------------------------------------------------
    # Detection result
    # --------------------------------------------------------

    result_id = insert_detection_result(
        transaction_id=transaction_id,
        fraud_probability=0.87,
        dqn_threshold=0.80,
        decision="FRAUD",
        model_version="RFGN-Global-GraphSAGE",
    )

    if not result_id:

        raise RuntimeError(
            "Detection result ID was not created."
        )

    print(
        "Detection result      : PASS"
    )

    # --------------------------------------------------------
    # Retrieve detection
    # --------------------------------------------------------

    result = get_detection_result(
        transaction_id
    )

    if result is None:

        raise RuntimeError(
            "Detection result retrieval failed."
        )

    print(
        "Detection retrieval   : PASS"
    )

    # --------------------------------------------------------
    # Feedback
    # --------------------------------------------------------

    feedback_id = insert_feedback(
        transaction_id=transaction_id,
        actual_label=1,
        feedback_source="self_test",
    )

    if not feedback_id:

        raise RuntimeError(
            "Feedback ID was not created."
        )

    print(
        "Feedback insertion    : PASS"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = get_database_summary()

    print()
    print(
        "Database summary:"
    )

    print(
        "  Transactions       :",
        summary["transactions"]
    )

    print(
        "  Detection results  :",
        summary["detection_results"]
    )

    print(
        "  Feedback           :",
        summary["feedback"]
    )

    print()
    print(
        "Database query module: PASS"
    )

    print("=" * 70)

    return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_self_test()