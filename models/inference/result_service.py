"""
RFGN - Result Storage & Feedback Service

Stores real-time transaction detection results in DuckDB.

Responsibilities:
    - Store transactions
    - Store GraphSAGE + DQN detection results
    - Retrieve stored transactions/results
    - Store confirmed feedback
    - Validate database integrity

This module does NOT:
    - train GraphSAGE
    - train DQN
    - perform feature engineering
    - make fraud decisions
"""

from __future__ import annotations

import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# DATABASE IMPORT
# ============================================================

try:

    from database.db import (
        execute,
        health_check as database_health_check,
    )

except ImportError as exc:

    raise ImportError(
        "Unable to import RFGN database layer."
    ) from exc


# ============================================================
# RESULT STORAGE SERVICE
# ============================================================

class ResultService:
    """
    Service responsible for storing and retrieving
    real-time RFGN fraud detection results.
    """

    MODEL_VERSION = (
        "RFGN-GraphSAGE-Flower-DQN-20000"
    )

    VALID_DECISIONS = {
        "FRAUD",
        "LEGITIMATE",
    }

    VALID_FEEDBACK_SOURCES = {
        "manual",
        "system",
        "bank",
        "investigation",
        "dataset",
        "unknown",
    }

    # ========================================================
    # VALIDATE TRANSACTION
    # ========================================================

    @staticmethod
    def _validate_transaction(
        transaction: Dict[str, Any],
    ) -> None:

        if not isinstance(
            transaction,
            dict,
        ):

            raise TypeError(
                "transaction must be a dictionary."
            )

        required_fields = [
            "transaction_id",
            "transaction_dt",
        ]

        for field in required_fields:

            if field not in transaction:

                raise ValueError(
                    f"Missing transaction field: {field}"
                )

        transaction_id = str(
            transaction[
                "transaction_id"
            ]
        ).strip()

        if not transaction_id:

            raise ValueError(
                "transaction_id cannot be empty."
            )

        try:

            transaction_dt = float(
                transaction[
                    "transaction_dt"
                ]
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "transaction_dt must be numeric."
            ) from exc

        if not math.isfinite(
            transaction_dt
        ):

            raise ValueError(
                "transaction_dt must be finite."
            )

    # ========================================================
    # VALIDATE DETECTION RESULT
    # ========================================================

    @classmethod
    def _validate_detection_result(
        cls,
        fraud_probability: float,
        dqn_threshold: float,
        decision: str,
        model_version: str,
    ) -> None:

        try:

            fraud_probability = float(
                fraud_probability
            )

            dqn_threshold = float(
                dqn_threshold
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "fraud_probability and dqn_threshold "
                "must be numeric."
            ) from exc

        if not math.isfinite(
            fraud_probability
        ):

            raise ValueError(
                "fraud_probability must be finite."
            )

        if not math.isfinite(
            dqn_threshold
        ):

            raise ValueError(
                "dqn_threshold must be finite."
            )

        if not (
            0.0
            <= fraud_probability
            <= 1.0
        ):

            raise ValueError(
                "fraud_probability must be between 0 and 1."
            )

        if not (
            0.0
            <= dqn_threshold
            <= 1.0
        ):

            raise ValueError(
                "dqn_threshold must be between 0 and 1."
            )

        decision = str(
            decision
        ).strip().upper()

        if decision not in cls.VALID_DECISIONS:

            raise ValueError(
                "decision must be either "
                "FRAUD or LEGITIMATE."
            )

        if not isinstance(
            model_version,
            str,
        ) or not model_version.strip():

            raise ValueError(
                "model_version cannot be empty."
            )

    # ========================================================
    # CHECK TRANSACTION EXISTS
    # ========================================================

    @staticmethod
    def transaction_exists(
        transaction_id: str,
    ) -> bool:

        result = execute(
            """
            SELECT COUNT(*)
            FROM transactions
            WHERE transaction_id = ?
            """,
            [
                transaction_id
            ],
        )

        try:

            count = result.fetchone()[0]

        except AttributeError:

            # Defensive fallback if execute() returns
            # a different DuckDB-compatible result object.

            rows = result.fetchall()

            if not rows:

                return False

            count = rows[0][0]

        return int(
            count
        ) > 0

    # ========================================================
    # SAVE TRANSACTION
    # ========================================================

    def save_transaction(
        self,
        transaction: Dict[str, Any],
    ) -> str:
        """
        Save a transaction.

        Duplicate transactions are not inserted again.
        """

        self._validate_transaction(
            transaction
        )

        transaction_id = str(
            transaction[
                "transaction_id"
            ]
        ).strip()

        transaction_dt = float(
            transaction[
                "transaction_dt"
            ]
        )

        # ----------------------------------------------------
        # Duplicate protection
        # ----------------------------------------------------

        if self.transaction_exists(
            transaction_id
        ):

            return transaction_id

        # ----------------------------------------------------
        # JSON serialization
        # ----------------------------------------------------

        transaction_json = json.dumps(
            transaction,
            default=str,
            allow_nan=False,
        )

        # ----------------------------------------------------
        # Insert
        # ----------------------------------------------------

        execute(
            """
            INSERT INTO transactions
            (
                transaction_id,
                transaction_dt,
                transaction_data
            )
            VALUES (?, ?, ?)
            """,
            [
                transaction_id,
                transaction_dt,
                transaction_json,
            ],
        )

        return transaction_id

    # ========================================================
    # GET TRANSACTION
    # ========================================================

    def get_transaction(
        self,
        transaction_id: str,
    ) -> Optional[Dict[str, Any]]:

        if not isinstance(
            transaction_id,
            str,
        ) or not transaction_id.strip():

            raise ValueError(
                "transaction_id cannot be empty."
            )

        result = execute(
            """
            SELECT
                transaction_id,
                transaction_dt,
                transaction_data,
                created_at
            FROM transactions
            WHERE transaction_id = ?
            """,
            [
                transaction_id
            ],
        )

        row = result.fetchone()

        if row is None:

            return None

        transaction_data = row[2]

        if isinstance(
            transaction_data,
            str,
        ):

            try:

                transaction_data = json.loads(
                    transaction_data
                )

            except json.JSONDecodeError:

                pass

        return {
            "transaction_id":
                row[0],

            "transaction_dt":
                row[1],

            "transaction_data":
                transaction_data,

            "created_at":
                row[3],
        }

    # ========================================================
    # SAVE DETECTION RESULT
    # ========================================================

    def save_detection_result(
        self,
        transaction_id: str,
        fraud_probability: float,
        dqn_threshold: float,
        decision: str,
        model_version: Optional[str] = None,
    ) -> str:
        """
        Store the GraphSAGE + DQN detection result.
        """

        if model_version is None:

            model_version = (
                self.MODEL_VERSION
            )

        decision = str(
            decision
        ).strip().upper()

        self._validate_detection_result(
            fraud_probability=fraud_probability,
            dqn_threshold=dqn_threshold,
            decision=decision,
            model_version=model_version,
        )

        if not self.transaction_exists(
            transaction_id
        ):

            raise ValueError(
                "Transaction does not exist. "
                "Save the transaction before "
                "saving its detection result."
            )

        # ----------------------------------------------------
        # Generate result ID
        # ----------------------------------------------------

        result_id = (
            "RES-"
            + str(
                uuid.uuid4()
            )
        )

        # ----------------------------------------------------
        # Insert detection result
        # ----------------------------------------------------

        execute(
            """
            INSERT INTO detection_results
            (
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
                transaction_id,
                float(
                    fraud_probability
                ),
                float(
                    dqn_threshold
                ),
                decision,
                model_version,
            ],
        )

        return result_id

    # ========================================================
    # GET DETECTION RESULT
    # ========================================================

    def get_result(
        self,
        result_id: str,
    ) -> Optional[Dict[str, Any]]:

        if not isinstance(
            result_id,
            str,
        ) or not result_id.strip():

            raise ValueError(
                "result_id cannot be empty."
            )

        result = execute(
            """
            SELECT
                result_id,
                transaction_id,
                fraud_probability,
                dqn_threshold,
                decision,
                model_version,
                created_at
            FROM detection_results
            WHERE result_id = ?
            """,
            [
                result_id
            ],
        )

        row = result.fetchone()

        if row is None:

            return None

        return {
            "result_id":
                row[0],

            "transaction_id":
                row[1],

            "fraud_probability":
                row[2],

            "dqn_threshold":
                row[3],

            "decision":
                row[4],

            "model_version":
                row[5],

            "created_at":
                row[6],
        }

    # ========================================================
    # STORE COMPLETE DETECTION
    # ========================================================

    def store_detection(
        self,
        transaction: Dict[str, Any],
        fraud_probability: float,
        dqn_threshold: float,
        decision: str,
        model_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Store a complete transaction + detection result.
        """

        transaction_id = (
            self.save_transaction(
                transaction
            )
        )

        result_id = (
            self.save_detection_result(
                transaction_id=transaction_id,
                fraud_probability=fraud_probability,
                dqn_threshold=dqn_threshold,
                decision=decision,
                model_version=model_version,
            )
        )

        stored_transaction = (
            self.get_transaction(
                transaction_id
            )
        )

        stored_result = (
            self.get_result(
                result_id
            )
        )

        return {
            "transaction_id":
                transaction_id,

            "result_id":
                result_id,

            "transaction":
                stored_transaction,

            "detection_result":
                stored_result,
        }

    # ========================================================
    # STORE FEEDBACK
    # ========================================================

    def store_feedback(
        self,
        transaction_id: str,
        actual_label: int,
        feedback_source: str = "unknown",
    ) -> str:
        """
        Store the confirmed actual transaction label.

        actual_label:
            0 = Legitimate
            1 = Fraud
        """

        if not isinstance(
            transaction_id,
            str,
        ) or not transaction_id.strip():

            raise ValueError(
                "transaction_id cannot be empty."
            )

        # ----------------------------------------------------
        # Validate label
        # ----------------------------------------------------

        if isinstance(
            actual_label,
            bool,
        ):

            raise ValueError(
                "actual_label must be 0 or 1."
            )

        try:

            actual_label = int(
                actual_label
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "actual_label must be 0 or 1."
            ) from exc

        if actual_label not in (
            0,
            1,
        ):

            raise ValueError(
                "actual_label must be either 0 or 1."
            )

        # ----------------------------------------------------
        # Validate feedback source
        # ----------------------------------------------------

        feedback_source = str(
            feedback_source
        ).strip().lower()

        if feedback_source not in (
            self.VALID_FEEDBACK_SOURCES
        ):

            raise ValueError(
                "Invalid feedback source.\n"
                f"Allowed: {self.VALID_FEEDBACK_SOURCES}"
            )

        # ----------------------------------------------------
        # Verify transaction
        # ----------------------------------------------------

        if not self.transaction_exists(
            transaction_id
        ):

            raise ValueError(
                "Transaction does not exist."
            )

        # ----------------------------------------------------
        # Generate feedback ID
        # ----------------------------------------------------

        feedback_id = (
            "FDB-"
            + str(
                uuid.uuid4()
            )
        )

        # ----------------------------------------------------
        # Insert feedback
        # ----------------------------------------------------

        execute(
            """
            INSERT INTO feedback
            (
                feedback_id,
                transaction_id,
                actual_label,
                feedback_source
            )
            VALUES (?, ?, ?, ?)
            """,
            [
                feedback_id,
                transaction_id,
                actual_label,
                feedback_source,
            ],
        )

        return feedback_id

    # ========================================================
    # DATABASE SUMMARY
    # ========================================================

    def get_summary(
        self,
    ) -> Dict[str, int]:

        transactions = execute(
            "SELECT COUNT(*) FROM transactions"
        ).fetchone()[0]

        detection_results = execute(
            "SELECT COUNT(*) FROM detection_results"
        ).fetchone()[0]

        feedback = execute(
            "SELECT COUNT(*) FROM feedback"
        ).fetchone()[0]

        return {
            "transactions":
                int(
                    transactions
                ),

            "detection_results":
                int(
                    detection_results
                ),

            "feedback":
                int(
                    feedback
                ),
        }

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:

        database_status = (
            database_health_check()
        )

        return {
            "service":
                "ResultService",

            "status":
                "healthy",

            "database":
                database_status,

            "summary":
                self.get_summary(),
        }


# ============================================================
# SELF TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 70)
    print(
        "RFGN RESULT STORAGE & FEEDBACK SELF-TEST"
    )
    print("=" * 70)

    print()

    service = ResultService()

    # --------------------------------------------------------
    # Database connection
    # --------------------------------------------------------

    try:

        service.health_check()

        print(
            "Database connection   : PASS"
        )

    except Exception as exc:

        print(
            "Database connection   : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Test IDs
    # --------------------------------------------------------

    transaction_id = (
        "TEST-"
        + str(
            uuid.uuid4()
        )
    )

    complete_transaction_id = (
        "TEST-COMPLETE-"
        + str(
            uuid.uuid4()
        )
    )

    # --------------------------------------------------------
    # Test transaction
    # --------------------------------------------------------

    transaction = {
        "transaction_id":
            transaction_id,

        "transaction_dt":
            123456.0,

        "TransactionAmt":
            250.75,

        "ProductCD":
            "W",

        "card1":
            12345,

        "card2":
            111,

        "card3":
            150,

        "card5":
            226,

        "addr1":
            100,

        "addr2":
            87,

        "P_emaildomain":
            "gmail.com",

        "R_emaildomain":
            "gmail.com",

        "DeviceInfo":
            "TEST_DEVICE",
    }

    # --------------------------------------------------------
    # Transaction storage
    # --------------------------------------------------------

    try:

        saved_id = service.save_transaction(
            transaction
        )

        if saved_id != transaction_id:

            raise RuntimeError(
                "Returned transaction ID does not match."
            )

        print(
            "Transaction storage   : PASS"
        )

    except Exception as exc:

        print(
            "Transaction storage   : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Transaction retrieval
    # --------------------------------------------------------

    try:

        stored_transaction = (
            service.get_transaction(
                transaction_id
            )
        )

        if stored_transaction is None:

            raise RuntimeError(
                "Stored transaction could not be retrieved."
            )

        print(
            "Transaction retrieval : PASS"
        )

    except Exception as exc:

        print(
            "Transaction retrieval : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Detection result storage
    # --------------------------------------------------------

    try:

        result_id = (
            service.save_detection_result(
                transaction_id=transaction_id,
                fraud_probability=0.92,
                dqn_threshold=0.80,
                decision="FRAUD",
            )
        )

        if not result_id.startswith(
            "RES-"
        ):

            raise RuntimeError(
                "Invalid result ID."
            )

        print(
            "Detection storage     : PASS"
        )

    except Exception as exc:

        print(
            "Detection storage     : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        # Continue to cleanup.
        result_id = None

    # --------------------------------------------------------
    # If detection storage failed, cleanup and stop
    # --------------------------------------------------------

    if result_id is None:

        try:

            execute(
                """
                DELETE FROM transactions
                WHERE transaction_id = ?
                """,
                [
                    transaction_id
                ],
            )

        except Exception:
            pass

        return False

    # --------------------------------------------------------
    # Detection retrieval
    # --------------------------------------------------------

    try:

        stored_result = service.get_result(
            result_id
        )

        if stored_result is None:

            raise RuntimeError(
                "Detection result could not be retrieved."
            )

        print(
            "Detection retrieval   : PASS"
        )

    except Exception as exc:

        print(
            "Detection retrieval   : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Detection integrity
    # --------------------------------------------------------

    try:

        probability = float(
            stored_result[
                "fraud_probability"
            ]
        )

        threshold = float(
            stored_result[
                "dqn_threshold"
            ]
        )

        decision = str(
            stored_result[
                "decision"
            ]
        )

        if abs(
            probability - 0.92
        ) > 1e-9:

            raise RuntimeError(
                "Fraud probability mismatch."
            )

        if abs(
            threshold - 0.80
        ) > 1e-9:

            raise RuntimeError(
                "DQN threshold mismatch."
            )

        if decision != "FRAUD":

            raise RuntimeError(
                "Decision mismatch."
            )

        print(
            "Detection integrity   : PASS"
        )

    except Exception as exc:

        print(
            "Detection integrity   : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Feedback storage
    # --------------------------------------------------------

    try:

        feedback_id = (
            service.store_feedback(
                transaction_id=transaction_id,
                actual_label=1,
                feedback_source="manual",
            )
        )

        if not feedback_id.startswith(
            "FDB-"
        ):

            raise RuntimeError(
                "Invalid feedback ID."
            )

        print(
            "Feedback storage      : PASS"
        )

    except Exception as exc:

        print(
            "Feedback storage      : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Complete workflow
    # --------------------------------------------------------

    try:

        complete_transaction = {
            "transaction_id":
                complete_transaction_id,

            "transaction_dt":
                654321.0,

            "TransactionAmt":
                75.25,

            "ProductCD":
                "C",
        }

        complete_result = (
            service.store_detection(
                transaction=complete_transaction,
                fraud_probability=0.18,
                dqn_threshold=0.80,
                decision="LEGITIMATE",
            )
        )

        if not complete_result[
            "result_id"
        ]:

            raise RuntimeError(
                "Complete workflow returned no result ID."
            )

        if not complete_result[
            "transaction"
        ]:

            raise RuntimeError(
                "Complete workflow returned no transaction."
            )

        if not complete_result[
            "detection_result"
        ]:

            raise RuntimeError(
                "Complete workflow returned no detection result."
            )

        print(
            "Complete workflow    : PASS"
        )

    except Exception as exc:

        print(
            "Complete workflow    : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Cleanup test records
    # --------------------------------------------------------

    try:

        # Feedback first because it references transactions.
        execute(
            """
            DELETE FROM feedback
            WHERE transaction_id = ?
            """,
            [
                transaction_id
            ],
        )

        # Detection results next.
        execute(
            """
            DELETE FROM detection_results
            WHERE transaction_id IN (?, ?)
            """,
            [
                transaction_id,
                complete_transaction_id,
            ],
        )

        # Transactions last.
        execute(
            """
            DELETE FROM transactions
            WHERE transaction_id IN (?, ?)
            """,
            [
                transaction_id,
                complete_transaction_id,
            ],
        )

        print(
            "Test data cleanup     : PASS"
        )

    except Exception as exc:

        print(
            "Test data cleanup     : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Verify database is clean
    # --------------------------------------------------------

    try:

        summary = service.get_summary()

        if (
            summary["transactions"] != 0
            or summary["detection_results"] != 0
            or summary["feedback"] != 0
        ):

            print(
                "Database cleanup     : FAIL"
            )

            print(
                summary
            )

            return False

        print(
            "Database cleanup     : PASS"
        )

    except Exception as exc:

        print(
            "Database cleanup     : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()

    print("=" * 70)
    print(
        "RESULT STORAGE & FEEDBACK: PASS"
    )
    print("=" * 70)

    return True


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    success = run_self_test()

    if not success:

        raise SystemExit(1)