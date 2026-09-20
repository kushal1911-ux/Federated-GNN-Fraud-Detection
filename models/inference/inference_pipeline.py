"""
RFGN - Complete Real-Time Inference Pipeline

Flow:

    Transaction Input
            ↓
    Database Integration
            ↓
    Real-Time Graph Construction
            ↓
    Feature Engineering & Alignment
            ↓
    Global GraphSAGE Inference
            ↓
    DQN Decision Layer
            ↓
    Result Storage & Feedback
"""

from __future__ import annotations

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
# IMPORTS
# ============================================================

from models.inference.transaction_input import (
    create_transaction,
)

from models.inference.transaction_service import (
    save_transaction,
    health_check as transaction_service_health,
)

from models.inference.real_time_graph import (
    RealTimeGraphBuilder,
)

from models.inference.feature_engineering import (
    FeatureEngineer,
)

from models.inference.global_inference import (
    GlobalGraphSAGEInference,
)

from models.inference.dqn_decision import (
    DQNDecisionLayer,
)

from models.inference.result_service import (
    ResultService,
)


# ============================================================
# PIPELINE
# ============================================================

class RFGNInferencePipeline:

    def __init__(self) -> None:

        print(
            "Initializing RFGN inference pipeline..."
        )

        # ----------------------------------------------------
        # Database
        # ----------------------------------------------------

        self.transaction_service_health = (
            transaction_service_health
        )

        # ----------------------------------------------------
        # Real-Time Graph
        # ----------------------------------------------------

        self.graph_builder = (
            RealTimeGraphBuilder()
        )

        # ----------------------------------------------------
        # Feature Engineering
        # ----------------------------------------------------

        self.feature_engineer = (
            FeatureEngineer()
        )

        # ----------------------------------------------------
        # Global GraphSAGE
        # ----------------------------------------------------

        self.graphsage = (
            GlobalGraphSAGEInference()
        )

        # ----------------------------------------------------
        # DQN
        # ----------------------------------------------------

        self.dqn = (
            DQNDecisionLayer()
        )

        # ----------------------------------------------------
        # Result Storage
        # ----------------------------------------------------

        self.result_service = (
            ResultService()
        )

        print(
            "RFGN inference pipeline initialized."
        )

    # ========================================================
    # NORMALIZE TRANSACTION DICTIONARY
    # ========================================================

    @staticmethod
    def _build_inference_transaction(
        transaction,
        transaction_id: str,
    ) -> Dict[str, Any]:
        """
        Convert the validated TransactionInput into the
        dictionary representation expected by the real-time
        inference modules.

        The original IEEE-CIS-style field names are preserved.
        Normalized lowercase database/inference fields are
        added explicitly.
        """

        data = dict(
            transaction.to_dict()
        )

        # ----------------------------------------------------
        # Canonical transaction ID
        # ----------------------------------------------------

        data[
            "transaction_id"
        ] = str(
            transaction_id
        )

        data[
            "TransactionID"
        ] = str(
            transaction_id
        )

        # ----------------------------------------------------
        # Canonical transaction timestamp
        # ----------------------------------------------------

        transaction_dt = float(
            transaction.data[
                "TransactionDT"
            ]
        )

        data[
            "transaction_dt"
        ] = transaction_dt

        data[
            "TransactionDT"
        ] = transaction_dt

        return data

    # ========================================================
    # PROCESS TRANSACTION
    # ========================================================

    def process_transaction(
        self,
        transaction_data: Dict[str, Any],
        transaction_id: Optional[Any] = None,
        store_result: bool = True,
    ) -> Dict[str, Any]:

        # ====================================================
        # 1. TRANSACTION INPUT
        # ====================================================

        transaction = create_transaction(
            data=transaction_data,
            transaction_id=transaction_id,
        )

        # ====================================================
        # 2. DATABASE INTEGRATION
        # ====================================================

        saved_transaction_id = (
            save_transaction(
                transaction
            )
        )

        # ====================================================
        # BUILD NORMALIZED INFERENCE DICTIONARY
        # ====================================================

        transaction_dict = (
            self._build_inference_transaction(
                transaction=transaction,
                transaction_id=saved_transaction_id,
            )
        )

        # ====================================================
        # 3. REAL-TIME GRAPH CONSTRUCTION
        # ====================================================

        graph = (
            self.graph_builder.build_graph(
                transaction_dict,
                feature_columns=(
                    self.feature_engineer
                    .graph_numeric_features
                ),
            )
        )

        # ====================================================
        # 4. FEATURE ENGINEERING & ALIGNMENT
        # ====================================================

        feature_tensor = (
            self.feature_engineer.transform_to_tensor(
                transaction_dict
            )
        )

        # ----------------------------------------------------
        # Authoritative model input
        # ----------------------------------------------------

        graph.x = feature_tensor

        # ----------------------------------------------------
        # Feature validation
        # ----------------------------------------------------

        if graph.x.ndim != 2:

            raise RuntimeError(
                "Graph feature tensor must be 2-dimensional."
            )

        if graph.x.shape[0] != 1:

            raise RuntimeError(
                "Real-time graph must contain exactly one node."
            )

        if graph.x.shape[1] != 769:

            raise RuntimeError(
                "Feature alignment mismatch. "
                "Expected 769 features, "
                f"received {graph.x.shape[1]}."
            )

        # ====================================================
        # 5. GLOBAL GRAPHSAGE INFERENCE
        # ====================================================

        graphsage_result = (
            self.graphsage.predict_single(
                graph
            )
        )

        fraud_probability = float(
            graphsage_result[
                "fraud_probability"
            ]
        )

        legitimate_probability = float(
            graphsage_result[
                "legitimate_probability"
            ]
        )

        graphsage_predicted_class = int(
            graphsage_result[
                "predicted_class"
            ]
        )

        # ====================================================
        # 6. DQN DECISION
        # ====================================================

        dqn_result = (
            self.dqn.decide(
                fraud_probability=(
                    fraud_probability
                ),
            )
        )

        dqn_threshold = float(
            dqn_result[
                "dqn_threshold"
            ]
        )

        dqn_action = int(
            dqn_result[
                "dqn_action"
            ]
        )

        decision = str(
            dqn_result[
                "decision"
            ]
        )

        final_predicted_class = int(
            dqn_result[
                "predicted_class"
            ]
        )

        # ====================================================
        # 7. RESULT STORAGE
        # ====================================================

        storage_result = None

        if store_result:

            storage_result = (
                self.result_service.store_detection(
                    transaction=transaction_dict,
                    fraud_probability=(
                        fraud_probability
                    ),
                    dqn_threshold=(
                        dqn_threshold
                    ),
                    decision=decision,
                )
            )

        # ====================================================
        # RETURN COMPLETE RESULT
        # ====================================================

        return {

            "pipeline":
                "RFGN Real-Time Inference",

            "transaction_id":
                saved_transaction_id,

            "graph":
                {
                    "nodes":
                        int(
                            graph.x.shape[0]
                        ),

                    "features":
                        int(
                            graph.x.shape[1]
                        ),

                    "edges":
                        int(
                            graph.edge_index.shape[1]
                        ),
                },

            "graphsage":
                {
                    "fraud_probability":
                        fraud_probability,

                    "legitimate_probability":
                        legitimate_probability,

                    "predicted_class":
                        graphsage_predicted_class,
                },

            "dqn":
                {
                    "action":
                        dqn_action,

                    "selected_threshold":
                        dqn_threshold,
                },

            "final_decision":
                {
                    "decision":
                        decision,

                    "predicted_class":
                        final_predicted_class,
                },

            "storage":
                storage_result,
        }

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:

        return {

            "pipeline":
                "RFGN Real-Time Inference",

            "status":
                "healthy",

            "transaction_service":
                self.transaction_service_health(),

            "graph_builder":
                self.graph_builder.health_check(),

            "feature_engineering":
                self.feature_engineer.health_check(),

            "global_graphsage":
                self.graphsage.health_check(),

            "dqn":
                self.dqn.health_check(),

            "result_service":
                self.result_service.health_check(),
        }


# ============================================================
# TEST TRANSACTION
# ============================================================

def create_test_transaction() -> Dict[str, Any]:

    return {

        "TransactionDT":
            1500000,

        "TransactionAmt":
            149.50,

        "ProductCD":
            "W",

        "card1":
            13926,

        "card2":
            555.0,

        "card3":
            150.0,

        "card5":
            226.0,

        "addr1":
            315.0,

        "addr2":
            87.0,

        "P_emaildomain":
            "gmail.com",

        "R_emaildomain":
            "gmail.com",

        "DeviceInfo":
            "Windows",
    }


# ============================================================
# SELF TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 70)
    print(
        "RFGN COMPLETE REAL-TIME INFERENCE PIPELINE SELF-TEST"
    )
    print("=" * 70)

    print()

    pipeline = None

    test_transaction_id = (
        "PIPELINE-TEST-"
        + str(uuid.uuid4())
    )

    # ========================================================
    # INITIALIZATION
    # ========================================================

    try:

        pipeline = (
            RFGNInferencePipeline()
        )

        print(
            "Pipeline initialization : PASS"
        )

    except Exception as exc:

        print(
            "Pipeline initialization : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    try:

        health = (
            pipeline.health_check()
        )

        if health.get(
            "status"
        ) != "healthy":

            raise RuntimeError(
                "Pipeline health status is not healthy."
            )

        print(
            "Pipeline health         : PASS"
        )

    except Exception as exc:

        print(
            "Pipeline health         : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # ========================================================
    # CREATE TEST TRANSACTION
    # ========================================================

    transaction_data = (
        create_test_transaction()
    )

    # ========================================================
    # RUN COMPLETE PIPELINE
    # ========================================================

    try:

        result = (
            pipeline.process_transaction(
                transaction_data=(
                    transaction_data
                ),
                transaction_id=(
                    test_transaction_id
                ),
                store_result=True,
            )
        )

        print(
            "Transaction processing  : PASS"
        )

    except Exception as exc:

        print(
            "Transaction processing  : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # ========================================================
    # TRANSACTION ID
    # ========================================================

    if (
        result["transaction_id"]
        != test_transaction_id
    ):

        print(
            "Transaction ID          : FAIL"
        )

        return False

    print(
        "Transaction ID          : PASS"
    )

    # ========================================================
    # GRAPH VALIDATION
    # ========================================================

    graph_info = result[
        "graph"
    ]

    if graph_info[
        "nodes"
    ] != 1:

        print(
            "Graph node count        : FAIL"
        )

        return False

    if graph_info[
        "features"
    ] != 769:

        print(
            "Graph feature count     : FAIL"
        )

        print(
            "Expected: 769"
        )

        print(
            "Actual: "
            f"{graph_info['features']}"
        )

        return False

    print(
        "Graph construction      : PASS"
    )

    # ========================================================
    # GRAPHSAGE VALIDATION
    # ========================================================

    graphsage = result[
        "graphsage"
    ]

    fraud_probability = float(
        graphsage[
            "fraud_probability"
        ]
    )

    legitimate_probability = float(
        graphsage[
            "legitimate_probability"
        ]
    )

    graphsage_class = int(
        graphsage[
            "predicted_class"
        ]
    )

    if not (
        0.0
        <= fraud_probability
        <= 1.0
    ):

        print(
            "Fraud probability       : FAIL"
        )

        return False

    if not (
        0.0
        <= legitimate_probability
        <= 1.0
    ):

        print(
            "Legitimate probability  : FAIL"
        )

        return False

    if abs(
        (
            fraud_probability
            + legitimate_probability
        ) - 1.0
    ) > 1e-5:

        print(
            "Probability consistency : FAIL"
        )

        return False

    if graphsage_class not in (
        0,
        1,
    ):

        print(
            "GraphSAGE class         : FAIL"
        )

        return False

    print(
        "GraphSAGE inference     : PASS"
    )

    # ========================================================
    # DQN VALIDATION
    # ========================================================

    dqn = result[
        "dqn"
    ]

    dqn_action = int(
        dqn[
            "action"
        ]
    )

    dqn_threshold = float(
        dqn[
            "selected_threshold"
        ]
    )

    if not (
        0
        <= dqn_action
        <= 18
    ):

        print(
            "DQN action              : FAIL"
        )

        return False

    if not (
        0.05
        <= dqn_threshold
        <= 0.95
    ):

        print(
            "DQN threshold           : FAIL"
        )

        return False

    print(
        "DQN decision            : PASS"
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    final_decision = result[
        "final_decision"
    ]

    decision = str(
        final_decision[
            "decision"
        ]
    )

    final_class = int(
        final_decision[
            "predicted_class"
        ]
    )

    if decision not in (
        "FRAUD",
        "LEGITIMATE",
    ):

        print(
            "Final decision          : FAIL"
        )

        return False

    if final_class not in (
        0,
        1,
    ):

        print(
            "Final class             : FAIL"
        )

        return False

    expected_decision = (
        "FRAUD"
        if fraud_probability
        >= dqn_threshold
        else "LEGITIMATE"
    )

    if decision != expected_decision:

        print(
            "Decision consistency    : FAIL"
        )

        return False

    print(
        "Final decision          : PASS"
    )

    # ========================================================
    # STORAGE VALIDATION
    # ========================================================

    storage = result[
        "storage"
    ]

    if storage is None:

        print(
            "Result storage          : FAIL"
        )

        return False

    result_id = storage.get(
        "result_id"
    )

    if not result_id:

        print(
            "Result ID               : FAIL"
        )

        return False

    print(
        "Result storage          : PASS"
    )

    # ========================================================
    # DATABASE VERIFICATION
    # ========================================================

    try:

        stored_transaction = (
            pipeline.result_service
            .get_transaction(
                test_transaction_id
            )
        )

        if stored_transaction is None:

            raise RuntimeError(
                "Stored transaction not found."
            )

        stored_result = (
            pipeline.result_service
            .get_result(
                result_id
            )
        )

        if stored_result is None:

            raise RuntimeError(
                "Stored detection result not found."
            )

        print(
            "Database verification   : PASS"
        )

    except Exception as exc:

        print(
            "Database verification   : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print()

    print(
        "---------------- COMPLETE RESULT ----------------"
    )

    print(
        f"Transaction ID        : "
        f"{test_transaction_id}"
    )

    print(
        f"Graph nodes           : "
        f"{graph_info['nodes']}"
    )

    print(
        f"Graph features        : "
        f"{graph_info['features']}"
    )

    print(
        f"Graph edges           : "
        f"{graph_info['edges']}"
    )

    print(
        f"Fraud probability     : "
        f"{fraud_probability:.6f}"
    )

    print(
        f"Legitimate probability: "
        f"{legitimate_probability:.6f}"
    )

    print(
        f"DQN action            : "
        f"{dqn_action}"
    )

    print(
        f"DQN threshold         : "
        f"{dqn_threshold:.2f}"
    )

    print(
        f"Final decision        : "
        f"{decision}"
    )

    print(
        f"Result ID             : "
        f"{result_id}"
    )

    print(
        "--------------------------------------------------"
    )

    # ========================================================
    # CLEANUP
    # ========================================================

    try:

        from database.db import execute

        execute(
            """
            DELETE FROM feedback
            WHERE transaction_id = ?
            """,
            [
                test_transaction_id
            ],
        )

        execute(
            """
            DELETE FROM detection_results
            WHERE transaction_id = ?
            """,
            [
                test_transaction_id
            ],
        )

        execute(
            """
            DELETE FROM transactions
            WHERE transaction_id = ?
            """,
            [
                test_transaction_id
            ],
        )

        summary = (
            pipeline.result_service
            .get_summary()
        )

        if (
            summary["transactions"] != 0
            or summary["detection_results"] != 0
            or summary["feedback"] != 0
        ):

            print(
                "Database cleanup       : FAIL"
            )

            print(
                summary
            )

            return False

        print(
            "Database cleanup       : PASS"
        )

    except Exception as exc:

        print(
            "Database cleanup       : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # ========================================================
    # FINAL
    # ========================================================

    print()

    print("=" * 70)
    print(
        "RFGN REAL-TIME INFERENCE PIPELINE: PASS"
    )
    print("=" * 70)

    return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    success = run_self_test()

    if not success:

        raise SystemExit(1)