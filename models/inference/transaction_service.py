"""
RFGN TRANSACTION SERVICE

Connects the Transaction Input Module with the
RFGN DuckDB database.

This module does NOT:
- construct the graph
- engineer model features
- run GraphSAGE
- run DQN
- make a fraud decision
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from database.queries import (
    get_transaction,
    insert_transaction,
    transaction_exists,
)

from models.inference.transaction_input import (
    TransactionInput,
    create_transaction,
)


# ============================================================
# SAVE TRANSACTION
# ============================================================

def save_transaction(
    transaction: TransactionInput,
) -> str:
    """
    Save a validated transaction to DuckDB.
    """

    transaction_id = str(
        transaction.data["TransactionID"]
    )

    if transaction_exists(
        transaction_id
    ):

        raise ValueError(
            f"Transaction already exists: "
            f"{transaction_id}"
        )

    transaction_dt = float(
        transaction.data["TransactionDT"]
    )

    insert_transaction(
        transaction_id=transaction_id,
        transaction_dt=transaction_dt,
        transaction_data=transaction.to_dict(),
    )

    return transaction_id


# ============================================================
# CREATE AND SAVE
# ============================================================

def create_and_save_transaction(
    data: Dict[str, Any],
    transaction_id: Optional[Any] = None,
) -> TransactionInput:
    """
    Validate a new transaction and save it to DuckDB.
    """

    transaction = create_transaction(
        data=data,
        transaction_id=transaction_id,
    )

    save_transaction(
        transaction
    )

    return transaction


# ============================================================
# GET STORED TRANSACTION
# ============================================================

def load_transaction(
    transaction_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a stored transaction.
    """

    return get_transaction(
        transaction_id
    )


# ============================================================
# HEALTH CHECK
# ============================================================

def health_check() -> Dict[str, Any]:

    return {
        "service": "transaction_service",
        "status": "ready",
    }


# ============================================================
# SELF TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 70)
    print("RFGN TRANSACTION SERVICE TEST")
    print("=" * 70)

    test_transaction = {
        "TransactionDT": 172800,
        "TransactionAmt": 499.99,

        "card1": 1234,
        "card2": 111,
        "card3": 150,
        "card5": 222,

        "addr1": 100,
        "addr2": 200,

        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",

        "DeviceInfo": "RFGN-Test-Device",
    }

    # --------------------------------------------------------
    # CREATE + SAVE
    # --------------------------------------------------------

    transaction = create_and_save_transaction(
        test_transaction
    )

    transaction_id = str(
        transaction.data["TransactionID"]
    )

    print()
    print(
        "Transaction validation : PASS"
    )

    print(
        "Transaction storage    : PASS"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    stored = load_transaction(
        transaction_id
    )

    if stored is None:

        raise RuntimeError(
            "Stored transaction could not be retrieved."
        )

    print(
        "Transaction retrieval  : PASS"
    )

    # --------------------------------------------------------
    # VERIFY CONTENT
    # --------------------------------------------------------

    stored_data = stored[
        "transaction_data"
    ]

    if stored_data.get(
        "TransactionAmt"
    ) != 499.99:

        raise RuntimeError(
            "Stored transaction data does not match."
        )

    if "isFraud" in stored_data:

        raise RuntimeError(
            "isFraud must not be stored for "
            "an incoming transaction."
        )

    print(
        "Data integrity         : PASS"
    )

    # --------------------------------------------------------
    # DUPLICATE PROTECTION
    # --------------------------------------------------------

    try:

        save_transaction(
            transaction
        )

    except ValueError:

        print(
            "Duplicate protection  : PASS"
        )

    else:

        raise RuntimeError(
            "Duplicate transaction was accepted."
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print(
        "Transaction ID:",
        transaction_id
    )

    print()
    print(
        "Transaction service: PASS"
    )

    print("=" * 70)

    return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_self_test()