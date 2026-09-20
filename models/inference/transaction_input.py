"""
RFGN TRANSACTION INPUT MODULE

Responsible for accepting and validating a new transaction
before it enters the real-time fraud detection pipeline.

This module does NOT:
- perform graph construction
- perform feature normalization
- run GraphSAGE
- run DQN
- determine fraud/legitimate status
- store results in the database

Those responsibilities belong to subsequent inference modules.
"""

from __future__ import annotations

import math
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# ============================================================
# CORE TRANSACTION COLUMNS
# ============================================================

TRANSACTION_ID_COLUMN = "TransactionID"
TEMPORAL_COLUMN = "TransactionDT"
TARGET_COLUMN = "isFraud"


# ============================================================
# GRAPH RELATIONSHIP COLUMNS
# ============================================================

RELATIONSHIP_COLUMNS = [
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
]


# ============================================================
# INPUT EXCEPTIONS
# ============================================================

class TransactionInputError(ValueError):
    """Raised when transaction input is invalid."""


# ============================================================
# TRANSACTION INPUT
# ============================================================

class TransactionInput:
    """
    Validated representation of one incoming transaction.

    The object intentionally keeps the raw transaction fields.
    Feature engineering and graph construction happen later.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        transaction_id: Optional[Any] = None,
    ):

        if not isinstance(data, dict):
            raise TransactionInputError(
                "Transaction input must be a dictionary."
            )

        self.data = dict(data)

        self._validate_target_absence()

        self.transaction_id = (
            transaction_id
            if transaction_id is not None
            else self.data.get(TRANSACTION_ID_COLUMN)
        )

        self._prepare_transaction_id()
        self._validate_temporal_field()
        self._normalize_empty_values()
        self._validate_numeric_values()
        self._validate_relationship_fields()

    # ========================================================
    # TARGET VALIDATION
    # ========================================================

    def _validate_target_absence(self) -> None:

        if TARGET_COLUMN in self.data:

            raise TransactionInputError(
                "isFraud must not be supplied with a new "
                "transaction. Fraud labels are only used "
                "during training/evaluation."
            )

    # ========================================================
    # TRANSACTION ID
    # ========================================================

    def _prepare_transaction_id(self) -> None:

        if self.transaction_id is None:

            self.transaction_id = (
                f"RT-{uuid.uuid4().hex.upper()}"
            )

        self.data[TRANSACTION_ID_COLUMN] = (
            self.transaction_id
        )

    # ========================================================
    # TEMPORAL FIELD
    # ========================================================

    def _validate_temporal_field(self) -> None:

        if TEMPORAL_COLUMN not in self.data:

            raise TransactionInputError(
                "TransactionDT is required for real-time "
                "graph construction."
            )

        value = self.data[TEMPORAL_COLUMN]

        if value is None:

            raise TransactionInputError(
                "TransactionDT cannot be None."
            )

        try:

            numeric_value = float(value)

        except (TypeError, ValueError):

            raise TransactionInputError(
                "TransactionDT must be numeric."
            )

        if not math.isfinite(numeric_value):

            raise TransactionInputError(
                "TransactionDT must be finite."
            )

        self.data[TEMPORAL_COLUMN] = numeric_value

    # ========================================================
    # EMPTY VALUE NORMALIZATION
    # ========================================================

    def _normalize_empty_values(self) -> None:

        for column, value in list(
            self.data.items()
        ):

            if isinstance(value, str):

                value = value.strip()

                if value == "":

                    self.data[column] = None

                else:

                    self.data[column] = value

    # ========================================================
    # NUMERIC VALUE VALIDATION
    # ========================================================

    def _validate_numeric_values(self) -> None:

        for column, value in self.data.items():

            if column in {
                TRANSACTION_ID_COLUMN,
                TEMPORAL_COLUMN,
            }:

                continue

            if value is None:

                continue

            if isinstance(value, bool):

                continue

            if isinstance(value, (int, float)):

                numeric_value = float(value)

                if not math.isfinite(
                    numeric_value
                ):

                    raise TransactionInputError(
                        f"Non-finite numeric value found "
                        f"in column '{column}'."
                    )

    # ========================================================
    # RELATIONSHIP FIELD VALIDATION
    # ========================================================

    def _validate_relationship_fields(self) -> None:

        missing = [
            column
            for column in RELATIONSHIP_COLUMNS
            if column not in self.data
        ]

        if missing:

            raise TransactionInputError(
                "Missing graph relationship fields: "
                f"{missing}"
            )

    # ========================================================
    # DICTIONARY REPRESENTATION
    # ========================================================

    def to_dict(self) -> Dict[str, Any]:

        return dict(self.data)

    # ========================================================
    # METADATA
    # ========================================================

    def metadata(self) -> Dict[str, Any]:

        return {
            "transaction_id": self.data[
                TRANSACTION_ID_COLUMN
            ],
            "transaction_dt": self.data[
                TEMPORAL_COLUMN
            ],
            "received_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }


# ============================================================
# PUBLIC INPUT FUNCTION
# ============================================================

def create_transaction(
    data: Dict[str, Any],
    transaction_id: Optional[Any] = None,
) -> TransactionInput:
    """
    Create and validate a real-time transaction.
    """

    return TransactionInput(
        data=data,
        transaction_id=transaction_id,
    )


# ============================================================
# SIMPLE HEALTH CHECK
# ============================================================

def health_check() -> Dict[str, Any]:

    return {
        "module": "transaction_input",
        "status": "ready",
        "transaction_id_required": False,
        "transaction_dt_required": True,
        "fraud_label_allowed": False,
        "relationship_fields_required": True,
        "graphsage_prediction": False,
        "dqn_decision": False,
    }


# ============================================================
# SELF TEST
# ============================================================

def run_self_test() -> bool:

    test_transaction = {
        "TransactionDT": 86400,
        "card1": 1234,
        "card2": 111,
        "card3": 150,
        "card5": 222,
        "addr1": 100,
        "addr2": 200,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "DeviceInfo": "TestDevice",
        "TransactionAmt": 100.50,
    }

    transaction = create_transaction(
        test_transaction
    )

    result = transaction.to_dict()

    if TRANSACTION_ID_COLUMN not in result:

        raise AssertionError(
            "TransactionID was not generated."
        )

    if TARGET_COLUMN in result:

        raise AssertionError(
            "isFraud was incorrectly included."
        )

    if TEMPORAL_COLUMN not in result:

        raise AssertionError(
            "TransactionDT was not preserved."
        )

    for column in RELATIONSHIP_COLUMNS:

        if column not in result:

            raise AssertionError(
                f"Relationship column missing: {column}"
            )

    print("=" * 70)
    print("RFGN TRANSACTION INPUT MODULE TEST")
    print("=" * 70)
    print()
    print("Transaction input       : PASS")
    print("TransactionID           : PASS")
    print("TransactionDT           : PASS")
    print("Relationship fields     : PASS")
    print("Fraud label exclusion   : PASS")
    print("Input validation        : PASS")
    print()
    print("Transaction Input Module: PASS")
    print("=" * 70)

    return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_self_test()