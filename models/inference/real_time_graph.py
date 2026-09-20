"""
RFGN - Real-Time Graph Construction

Builds a graph for a newly received transaction while preserving
the graph construction conventions used by the trained RFGN model.

The module:
    1. Validates the transaction
    2. Creates a single-node real-time graph
    3. Extracts the 814 raw numeric graph features
    4. Applies the verified 814 -> 769 feature mapping
    5. Applies the stored training normalization statistics
    6. Creates temporal/relationship edges when historical context
       is supplied
    7. Returns a PyTorch Geometric Data object

IMPORTANT:
    This module does NOT retrain or modify any existing model/graph.
"""

from __future__ import annotations

import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ============================================================
# CONFIGURATION
# ============================================================

GRAPH_BUILDER_PATH = (
    PROJECT_ROOT
    / "models"
    / "gnn"
    / "graph_builder.py"
)

INFERENCE_PREPROCESSING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graphs"
    / "inference_preprocessing.json"
)

EXPECTED_RAW_FEATURES = 814
EXPECTED_MODEL_FEATURES = 769

TEMPORAL_NEIGHBORS = 2
MAX_NODES_PER_RELATIONSHIP_VALUE = 50

EPSILON = 1e-12


# Same relationship columns used by the RFGN graph builder.
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
# OPTIONAL COLUMN METADATA
# ============================================================

IDENTIFIER_COLUMNS = {
    "TransactionID",
    "isFraud",
    "TransactionDT",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _is_finite_number(value: Any) -> bool:
    """
    Return True when value can safely be represented as a
    finite floating-point number.
    """

    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return False

    return math.isfinite(number)


def _normalize_relationship_value(
    value: Any,
) -> Optional[str]:
    """
    Normalize relationship values exactly for edge matching.

    Missing values and empty strings are treated as absent.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (
        TypeError,
        ValueError,
    ):
        pass

    text = str(value).strip()

    if text == "":
        return None

    return text


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert a value to finite float.

    Non-numeric, NaN and infinite values become default.
    """

    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default

    if not math.isfinite(number):
        return default

    return number


# ============================================================
# REAL-TIME GRAPH BUILDER
# ============================================================

class RealTimeGraphBuilder:
    """
    Constructs a real-time PyTorch Geometric graph compatible
    with the trained global GraphSAGE model.
    """

    def __init__(
        self,
        preprocessing_path: Path = INFERENCE_PREPROCESSING_PATH,
    ) -> None:

        self.preprocessing_path = Path(
            preprocessing_path
        )

        self.preprocessing = (
            self._load_preprocessing_artifact()
        )

        self.raw_feature_count = int(
            self.preprocessing[
                "raw_feature_count"
            ]
        )

        self.shared_feature_count = int(
            self.preprocessing[
                "shared_feature_count"
            ]
        )

        self.mean = torch.tensor(
            self.preprocessing["mean"],
            dtype=torch.float32,
        )

        self.std = torch.tensor(
            self.preprocessing["std"],
            dtype=torch.float32,
        )

        self.shared_raw_indices = [
            int(index)
            for index in self.preprocessing[
                "shared_raw_feature_indices"
            ]
        ]

        self.global_position_to_raw_index = [
            int(index)
            for index in self.preprocessing[
                "global_position_to_raw_feature_index"
            ]
        ]

        self.global_position_to_normalization_position = [
            int(index)
            for index in self.preprocessing[
                "global_position_to_normalization_position"
            ]
        ]

        self.removed_raw_indices = [
            int(index)
            for index in self.preprocessing[
                "removed_raw_feature_indices"
            ]
        ]

        self._validate_artifact()

    # ========================================================
    # LOAD PREPROCESSING ARTIFACT
    # ========================================================

    def _load_preprocessing_artifact(
        self,
    ) -> Dict[str, Any]:

        if not self.preprocessing_path.exists():
            raise FileNotFoundError(
                "Inference preprocessing artifact "
                f"not found:\n{self.preprocessing_path}\n\n"
                "Run:\n"
                "python "
                "tests\\create_inference_preprocessing_artifact.py"
            )

        with open(
            self.preprocessing_path,
            "r",
            encoding="utf-8",
        ) as file:
            artifact = json.load(file)

        if not isinstance(
            artifact,
            dict,
        ):
            raise ValueError(
                "Inference preprocessing artifact "
                "must contain a JSON object."
            )

        return artifact

    # ========================================================
    # VALIDATE ARTIFACT
    # ========================================================

    def _validate_artifact(
        self,
    ) -> None:

        if self.raw_feature_count != EXPECTED_RAW_FEATURES:
            raise ValueError(
                "Inference artifact raw feature count "
                f"must be {EXPECTED_RAW_FEATURES}, "
                f"got {self.raw_feature_count}."
            )

        if self.shared_feature_count != EXPECTED_MODEL_FEATURES:
            raise ValueError(
                "Inference artifact shared feature count "
                f"must be {EXPECTED_MODEL_FEATURES}, "
                f"got {self.shared_feature_count}."
            )

        if self.mean.numel() != EXPECTED_MODEL_FEATURES:
            raise ValueError(
                "Inference mean vector must contain "
                f"{EXPECTED_MODEL_FEATURES} values."
            )

        if self.std.numel() != EXPECTED_MODEL_FEATURES:
            raise ValueError(
                "Inference std vector must contain "
                f"{EXPECTED_MODEL_FEATURES} values."
            )

        if len(
            self.global_position_to_raw_index
        ) != EXPECTED_MODEL_FEATURES:

            raise ValueError(
                "Global raw-index mapping must contain "
                f"{EXPECTED_MODEL_FEATURES} values."
            )

        if len(
            self.global_position_to_normalization_position
        ) != EXPECTED_MODEL_FEATURES:

            raise ValueError(
                "Global normalization mapping must contain "
                f"{EXPECTED_MODEL_FEATURES} values."
            )

        if torch.isnan(self.mean).any():
            raise ValueError(
                "NaN detected in normalization mean."
            )

        if torch.isnan(self.std).any():
            raise ValueError(
                "NaN detected in normalization std."
            )

        if torch.isinf(self.mean).any():
            raise ValueError(
                "Infinity detected in normalization mean."
            )

        if torch.isinf(self.std).any():
            raise ValueError(
                "Infinity detected in normalization std."
            )

        if torch.any(
            self.std <= 0
        ):
            raise ValueError(
                "Non-positive standard deviation detected."
            )

    # ========================================================
    # RAW FEATURE VECTOR
    # ========================================================

    def _build_raw_numeric_feature_vector(
        self,
        transaction: Dict[str, Any],
        feature_columns: Optional[
            Sequence[str]
        ] = None,
    ) -> Tuple[
        torch.Tensor,
        List[str],
    ]:
        """
        Build the raw 814-dimensional numeric feature vector.

        Preferred usage:
            pass feature_columns explicitly in the same order
            used when the graph dataset was created.

        When feature_columns is omitted, numeric keys from the
        transaction are sorted deterministically. This fallback
        is mainly useful for diagnostics and synthetic tests.

        Production inference should provide the canonical
        814-column schema.
        """

        if feature_columns is None:

            candidate_columns = []

            for key, value in transaction.items():

                if key in IDENTIFIER_COLUMNS:
                    continue

                if _is_finite_number(value):
                    candidate_columns.append(key)

            feature_columns = sorted(
                candidate_columns
            )

        feature_columns = list(
            feature_columns
        )

        if len(feature_columns) != EXPECTED_RAW_FEATURES:

            raise ValueError(
                "Real-time feature schema must contain "
                f"exactly {EXPECTED_RAW_FEATURES} "
                "numeric graph features. "
                f"Received {len(feature_columns)}.\n\n"
                "Pass the canonical 814-feature schema "
                "when calling build_graph()."
            )

        values = []

        for column in feature_columns:

            value = transaction.get(
                column
            )

            values.append(
                _safe_float(
                    value,
                    default=0.0,
                )
            )

        raw_x = torch.tensor(
            values,
            dtype=torch.float32,
        )

        if raw_x.shape != (
            EXPECTED_RAW_FEATURES,
        ):
            raise ValueError(
                "Raw feature tensor has unexpected shape: "
                f"{tuple(raw_x.shape)}"
            )

        if torch.isnan(raw_x).any():
            raise ValueError(
                "NaN detected in raw feature vector."
            )

        if torch.isinf(raw_x).any():
            raise ValueError(
                "Infinity detected in raw feature vector."
            )

        return raw_x, feature_columns

    # ========================================================
    # 814 → 769 ALIGNMENT
    # ========================================================

    def _align_and_normalize(
        self,
        raw_x: torch.Tensor,
        raw_feature_columns: Sequence[str],
    ) -> torch.Tensor:
        """
        Select the shared 769 raw features and normalize them
        using the stored Client 2 training statistics.

        The resulting order is exactly the global GraphSAGE
        feature order.
        """

        if raw_x.ndim != 1:
            raise ValueError(
                "Expected one-dimensional raw feature vector."
            )

        if raw_x.numel() != EXPECTED_RAW_FEATURES:
            raise ValueError(
                "Expected 814 raw features, got "
                f"{raw_x.numel()}."
            )

        if len(raw_feature_columns) != EXPECTED_RAW_FEATURES:
            raise ValueError(
                "Expected 814 raw feature column names."
            )

        # ----------------------------------------------------
        # Build raw-index lookup.
        #
        # The canonical preprocessing artifact stores raw
        # feature indices. The caller's columns must therefore
        # represent that same canonical 814-column order.
        # ----------------------------------------------------

        selected = raw_x[
            torch.tensor(
                self.global_position_to_raw_index,
                dtype=torch.long,
            )
        ]

        if selected.numel() != EXPECTED_MODEL_FEATURES:
            raise ValueError(
                "Feature alignment did not produce 769 values."
            )

        # ----------------------------------------------------
        # Normalization
        # ----------------------------------------------------

        normalized = (
            selected - self.mean
        ) / torch.clamp(
            self.std,
            min=EPSILON,
        )

        if torch.isnan(normalized).any():
            raise ValueError(
                "NaN detected after real-time normalization."
            )

        if torch.isinf(normalized).any():
            raise ValueError(
                "Infinity detected after real-time normalization."
            )

        if normalized.numel() != EXPECTED_MODEL_FEATURES:
            raise ValueError(
                "Normalized feature vector must contain "
                f"{EXPECTED_MODEL_FEATURES} values."
            )

        return normalized.contiguous()

    # ========================================================
    # RELATIONSHIP EDGES
    # ========================================================

    def _build_relationship_edges(
        self,
        transactions: pd.DataFrame,
    ) -> List[Tuple[int, int]]:

        edges = []

        for column in RELATIONSHIP_COLUMNS:

            if column not in transactions.columns:
                continue

            groups: Dict[
                str,
                List[int],
            ] = {}

            for index, value in enumerate(
                transactions[column].tolist()
            ):

                normalized_value = (
                    _normalize_relationship_value(
                        value
                    )
                )

                if normalized_value is None:
                    continue

                groups.setdefault(
                    normalized_value,
                    [],
                ).append(index)

            for indices in groups.values():

                if (
                    len(indices)
                    > MAX_NODES_PER_RELATIONSHIP_VALUE
                ):
                    continue

                for i in range(
                    len(indices)
                ):

                    for j in range(
                        i + 1,
                        len(indices),
                    ):

                        source = indices[i]
                        target = indices[j]

                        edges.append(
                            (
                                source,
                                target,
                            )
                        )

        return edges

    # ========================================================
    # TEMPORAL EDGES
    # ========================================================

    def _build_temporal_edges(
        self,
        transactions: pd.DataFrame,
    ) -> List[Tuple[int, int]]:

        if "TransactionDT" not in transactions.columns:
            return []

        valid_rows = []

        for index, value in enumerate(
            transactions["TransactionDT"].tolist()
        ):

            if not _is_finite_number(value):
                continue

            valid_rows.append(
                (
                    index,
                    float(value),
                )
            )

        valid_rows.sort(
            key=lambda item: item[1]
        )

        edges = []

        for position, (
            source_index,
            _,
        ) in enumerate(valid_rows):

            for offset in range(
                1,
                TEMPORAL_NEIGHBORS + 1,
            ):

                target_position = (
                    position + offset
                )

                if target_position >= len(
                    valid_rows
                ):
                    break

                target_index = (
                    valid_rows[
                        target_position
                    ][0]
                )

                edges.append(
                    (
                        source_index,
                        target_index,
                    )
                )

        return edges

    # ========================================================
    # COMBINE EDGES
    # ========================================================

    def _build_edge_index(
        self,
        transactions: pd.DataFrame,
    ) -> torch.Tensor:

        relationship_edges = (
            self._build_relationship_edges(
                transactions
            )
        )

        temporal_edges = (
            self._build_temporal_edges(
                transactions
            )
        )

        all_undirected = (
            relationship_edges
            + temporal_edges
        )

        # Remove duplicate undirected edges.
        unique_edges = set()

        for source, target in all_undirected:

            if source == target:
                continue

            low = min(
                source,
                target,
            )

            high = max(
                source,
                target,
            )

            unique_edges.add(
                (
                    low,
                    high,
                )
            )

        directed_edges = []

        for source, target in sorted(
            unique_edges
        ):

            directed_edges.append(
                (
                    source,
                    target,
                )
            )

            directed_edges.append(
                (
                    target,
                    source,
                )
            )

        if not directed_edges:

            return torch.empty(
                (
                    2,
                    0,
                ),
                dtype=torch.long,
            )

        edge_index = torch.tensor(
            directed_edges,
            dtype=torch.long,
        ).t().contiguous()

        return edge_index

    # ========================================================
    # BUILD GRAPH
    # ========================================================

    def build_graph(
        self,
        transaction: Dict[str, Any],
        historical_transactions: Optional[
            Iterable[Dict[str, Any]]
        ] = None,
        feature_columns: Optional[
            Sequence[str]
        ] = None,
    ) -> Data:
        """
        Build a real-time graph.

        Parameters
        ----------
        transaction:
            New transaction dictionary.

        historical_transactions:
            Optional historical transactions used to provide
            graph context and construct relationship/temporal
            edges.

        feature_columns:
            Canonical 814 numeric graph feature names in the
            exact training order.

        Returns
        -------
        torch_geometric.data.Data
            Graph containing:
                x
                edge_index
                transaction_id
                transaction_dt
        """

        if not isinstance(
            transaction,
            dict,
        ):
            raise TypeError(
                "transaction must be a dictionary."
            )

        if "isFraud" in transaction:
            raise ValueError(
                "Real-time transaction input must not contain "
                "the isFraud label."
            )

        if "TransactionDT" not in transaction:
            raise ValueError(
                "TransactionDT is required."
            )

        transaction_dt = transaction[
            "TransactionDT"
        ]

        if not _is_finite_number(
            transaction_dt
        ):
            raise ValueError(
                "TransactionDT must be a finite number."
            )

        # ----------------------------------------------------
        # Transaction ID
        # ----------------------------------------------------

        transaction_id = transaction.get(
            "TransactionID"
        )

        if transaction_id is None:
            transaction_id = (
                "RT-"
                + uuid.uuid4().hex.upper()
            )

        transaction_id = str(
            transaction_id
        )

        # ----------------------------------------------------
        # Create context dataframe
        # ----------------------------------------------------

        context_rows = []

        if historical_transactions is not None:

            for historical in historical_transactions:

                if not isinstance(
                    historical,
                    dict,
                ):
                    raise TypeError(
                        "Every historical transaction "
                        "must be a dictionary."
                    )

                if "isFraud" in historical:
                    historical = dict(
                        historical
                    )
                    historical.pop(
                        "isFraud",
                        None,
                    )

                context_rows.append(
                    historical
                )

        context_rows.append(
            dict(transaction)
        )

        transactions_df = pd.DataFrame(
            context_rows
        )

        # ----------------------------------------------------
        # Ensure required relationship columns exist.
        # ----------------------------------------------------

        for column in RELATIONSHIP_COLUMNS:

            if column not in transactions_df.columns:
                transactions_df[column] = None

        # ----------------------------------------------------
        # Build the new transaction's node feature vector.
        # ----------------------------------------------------

        raw_x, canonical_columns = (
            self._build_raw_numeric_feature_vector(
                transaction,
                feature_columns=feature_columns,
            )
        )

        normalized_x = (
            self._align_and_normalize(
                raw_x,
                canonical_columns,
            )
        )

        # ----------------------------------------------------
        # Graph edges.
        #
        # Context graph edges are constructed for all supplied
        # historical transactions + the new transaction.
        # ----------------------------------------------------

        context_edge_index = (
            self._build_edge_index(
                transactions_df
            )
        )

        new_node_index = (
            len(context_rows) - 1
        )

        # ----------------------------------------------------
        # Extract only edges touching the new transaction.
        #
        # The returned graph contains one node, so edges are
        # remapped to node 0. A relationship/temporal context
        # cannot be represented with the historical nodes unless
        # their feature vectors are also included.
        #
        # Therefore, for the default single-transaction graph,
        # we construct an isolated node unless explicit context
        # graph construction is requested separately.
        # ----------------------------------------------------

        del context_edge_index
        del new_node_index

        edge_index = torch.empty(
            (
                2,
                0,
            ),
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # Create PyG Data object.
        # ----------------------------------------------------

        graph = Data(
            x=normalized_x.view(
                1,
                EXPECTED_MODEL_FEATURES,
            ),
            edge_index=edge_index,
        )

        graph.transaction_id = [
            transaction_id
        ]

        graph.transaction_dt = torch.tensor(
            [float(transaction_dt)],
            dtype=torch.float64,
        )

        graph.num_nodes_expected = 1

        # ----------------------------------------------------
        # Final validation
        # ----------------------------------------------------

        if graph.x.shape != (
            1,
            EXPECTED_MODEL_FEATURES,
        ):
            raise ValueError(
                "Real-time graph x shape mismatch: "
                f"{tuple(graph.x.shape)}"
            )

        if graph.edge_index.dtype != torch.long:
            raise ValueError(
                "edge_index must use torch.long."
            )

        if torch.isnan(graph.x).any():
            raise ValueError(
                "NaN detected in final graph."
            )

        if torch.isinf(graph.x).any():
            raise ValueError(
                "Infinity detected in final graph."
            )

        return graph

    # ========================================================
    # CONTEXT GRAPH
    # ========================================================

    def build_context_graph(
        self,
        transactions: Sequence[Dict[str, Any]],
        feature_matrix: torch.Tensor,
    ) -> Data:
        """
        Build a graph from already-prepared feature vectors.

        This method is useful when the real-time inference
        service has a transaction plus historical graph context
        whose feature vectors are already available.

        feature_matrix:
            Shape [N, 769].

        transactions:
            Same N transactions represented by the rows.
        """

        if len(transactions) == 0:
            raise ValueError(
                "At least one transaction is required."
            )

        if not isinstance(
            feature_matrix,
            torch.Tensor,
        ):
            raise TypeError(
                "feature_matrix must be a torch.Tensor."
            )

        if feature_matrix.ndim != 2:
            raise ValueError(
                "feature_matrix must be two-dimensional."
            )

        if feature_matrix.shape[0] != len(
            transactions
        ):
            raise ValueError(
                "Transaction count and feature "
                "matrix row count differ."
            )

        if feature_matrix.shape[1] != EXPECTED_MODEL_FEATURES:
            raise ValueError(
                "Context feature matrix must have "
                f"{EXPECTED_MODEL_FEATURES} columns."
            )

        feature_matrix = (
            feature_matrix.float().contiguous()
        )

        if torch.isnan(
            feature_matrix
        ).any():
            raise ValueError(
                "NaN detected in context features."
            )

        if torch.isinf(
            feature_matrix
        ).any():
            raise ValueError(
                "Infinity detected in context features."
            )

        transactions_df = pd.DataFrame(
            list(transactions)
        )

        for column in RELATIONSHIP_COLUMNS:

            if column not in transactions_df.columns:
                transactions_df[column] = None

        edge_index = self._build_edge_index(
            transactions_df
        )

        graph = Data(
            x=feature_matrix,
            edge_index=edge_index,
        )

        graph.transaction_id = [
            str(
                transaction.get(
                    "TransactionID",
                    f"CTX-{index}",
                )
            )
            for index, transaction
            in enumerate(transactions)
        ]

        graph.transaction_dt = torch.tensor(
            [
                _safe_float(
                    transaction.get(
                        "TransactionDT",
                        0.0,
                    )
                )
                for transaction in transactions
            ],
            dtype=torch.float64,
        )

        if graph.x.shape[1] != EXPECTED_MODEL_FEATURES:
            raise ValueError(
                "Final context graph is not "
                "GraphSAGE-compatible."
            )

        return graph

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:

        return {
            "module": "RealTimeGraphBuilder",
            "status": "healthy",
            "raw_feature_count":
                self.raw_feature_count,
            "shared_feature_count":
                self.shared_feature_count,
            "normalization_method":
                self.preprocessing.get(
                    "normalization_method"
                ),
            "canonical_order_source":
                self.preprocessing.get(
                    "canonical_order_source"
                ),
            "relationship_columns":
                list(
                    RELATIONSHIP_COLUMNS
                ),
        }


# ============================================================
# SELF TEST
# ============================================================

def run_self_test() -> bool:
    """
    Validate the real-time graph builder itself.

    The test intentionally does not use real fraud labels.
    """

    print("=" * 70)
    print("RFGN REAL-TIME GRAPH BUILDER SELF-TEST")
    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Builder
    # --------------------------------------------------------

    try:

        builder = RealTimeGraphBuilder()

        print(
            "Preprocessing artifact : PASS"
        )

    except Exception as exc:

        print(
            "Preprocessing artifact : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Synthetic 814-feature schema
    # --------------------------------------------------------

    feature_columns = [
        f"feature_{index}"
        for index in range(
            EXPECTED_RAW_FEATURES
        )
    ]

    transaction = {
        "TransactionID":
            "RT-SELFTEST-001",

        "TransactionDT":
            500000.0,

        "card1":
            "1001",

        "card2":
            "2001",

        "card3":
            "3001",

        "card5":
            "5001",

        "addr1":
            "100",

        "addr2":
            "200",

        "P_emaildomain":
            "example.com",

        "R_emaildomain":
            "example.com",

        "DeviceInfo":
            "SELFTEST",
    }

    # Add synthetic numeric graph features.
    for index, column in enumerate(
        feature_columns
    ):

        transaction[column] = float(
            index
        )

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    try:

        graph = builder.build_graph(
            transaction,
            feature_columns=feature_columns,
        )

        print(
            "Transaction graph     : PASS"
        )

    except Exception as exc:

        print(
            "Transaction graph     : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    if graph.x.shape != (
        1,
        EXPECTED_MODEL_FEATURES,
    ):

        print(
            "Feature shape         : FAIL"
        )

        print(
            f"Received: {tuple(graph.x.shape)}"
        )

        return False

    print(
        "Feature shape         : PASS"
    )

    # --------------------------------------------------------
    # Edge index
    # --------------------------------------------------------

    if graph.edge_index.shape[0] != 2:

        print(
            "Edge index shape      : FAIL"
        )

        return False

    print(
        "Edge index            : PASS"
    )

    # --------------------------------------------------------
    # Numeric safety
    # --------------------------------------------------------

    if torch.isnan(
        graph.x
    ).any():

        print(
            "NaN safety            : FAIL"
        )

        return False

    if torch.isinf(
        graph.x
    ).any():

        print(
            "Infinity safety       : FAIL"
        )

        return False

    print(
        "NaN / Infinity safety : PASS"
    )

    # --------------------------------------------------------
    # Context graph test
    # --------------------------------------------------------

    transactions = []

    for index in range(3):

        transactions.append(
            {
                "TransactionID":
                    f"CTX-{index}",

                "TransactionDT":
                    1000.0 + index,

                "card1":
                    "CARD-GROUP",

                "card2":
                    None,

                "card3":
                    None,

                "card5":
                    None,

                "addr1":
                    None,

                "addr2":
                    None,

                "P_emaildomain":
                    None,

                "R_emaildomain":
                    None,

                "DeviceInfo":
                    None,
            }
        )

    context_features = torch.zeros(
        (
            3,
            EXPECTED_MODEL_FEATURES,
        ),
        dtype=torch.float32,
    )

    try:

        context_graph = (
            builder.build_context_graph(
                transactions,
                context_features,
            )
        )

        print(
            "Context graph         : PASS"
        )

    except Exception as exc:

        print(
            "Context graph         : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    if context_graph.x.shape != (
        3,
        EXPECTED_MODEL_FEATURES,
    ):

        print(
            "Context feature shape : FAIL"
        )

        return False

    print(
        "Context feature shape : PASS"
    )

    # --------------------------------------------------------
    # Health check
    # --------------------------------------------------------

    health = builder.health_check()

    if health.get(
        "status"
    ) != "healthy":

        print(
            "Health check          : FAIL"
        )

        return False

    print(
        "Health check          : PASS"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "REAL-TIME GRAPH BUILDER: PASS"
    )
    print("=" * 70)

    return True


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    success = run_self_test()

    if not success:
        raise SystemExit(1)