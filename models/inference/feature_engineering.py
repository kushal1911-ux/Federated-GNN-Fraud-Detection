"""
RFGN - Real-Time Feature Engineering & Alignment

Purpose
-------
Convert an incoming IEEE-CIS transaction into the exact
769-dimensional feature representation expected by the
trained global GraphSAGE model.

Pipeline
--------
Raw transaction
        ↓
Canonical 814-feature schema
        ↓
Missingness indicators
        ↓
Categorical missing-value handling
        ↓
Numeric missing-value handling
        ↓
814 raw numeric graph features
        ↓
Shared 769-feature selection
        ↓
Training normalization
        ↓
769-dimensional float32 tensor

IMPORTANT
---------
This module does not:
    - retrain models
    - modify graphs
    - modify datasets
    - modify model weights
    - use isFraud during inference
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ============================================================
# FILE PATHS
# ============================================================

GRAPHS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graphs"
)

INFERENCE_ARTIFACT = (
    GRAPHS_DIR
    / "inference_preprocessing.json"
)


# Search for the feature manifest in the known processed-data
# locations.
FEATURE_MANIFEST_CANDIDATES = [
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_manifest.json",

    PROJECT_ROOT
    / "data"
    / "processed"
    / "manifest"
    / "feature_manifest.json",

    PROJECT_ROOT
    / "data"
    / "processed"
    / "ieee"
    / "feature_manifest.json",

    PROJECT_ROOT
    / "data"
    / "processed"
    / "ieee"
    / "manifest"
    / "feature_manifest.json",
]


# ============================================================
# EXPECTED MODEL DIMENSIONS
# ============================================================

EXPECTED_RAW_FEATURES = 814
EXPECTED_MODEL_FEATURES = 769

EPSILON = 1e-12


# ============================================================
# IDENTIFIER / TARGET COLUMNS
# ============================================================

IDENTIFIER_COLUMNS = {
    "TransactionID",
}

TARGET_COLUMNS = {
    "isFraud",
}


# TransactionDT is required for graph construction but is not
# a node feature.
NON_FEATURE_COLUMNS = {
    "TransactionID",
    "isFraud",
    "TransactionDT",
}


# ============================================================
# RELATIONSHIP COLUMNS
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
# UTILITY FUNCTIONS
# ============================================================

def is_missing(value: Any) -> bool:
    """
    Safely determine whether a value is missing.
    """

    if value is None:
        return True

    try:
        result = pd.isna(value)

        if isinstance(
            result,
            (bool, np.bool_),
        ):
            return bool(result)

    except (
        TypeError,
        ValueError,
    ):
        pass

    return False


def is_finite_number(value: Any) -> bool:
    """
    Check whether a value can be represented as a finite float.
    """

    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return False

    return math.isfinite(
        number
    )


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert a value to a finite float.

    Missing or invalid values become default.
    """

    if is_missing(value):
        return default

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


def load_json(
    path: Path,
) -> Dict[str, Any]:
    """
    Load a JSON object.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            f"JSON file must contain an object:\n{path}"
        )

    return data


# ============================================================
# FEATURE ENGINEERING CLASS
# ============================================================

class FeatureEngineer:
    """
    Real-time IEEE-CIS feature engineering pipeline.
    """

    def __init__(
        self,
        inference_artifact_path: Path = INFERENCE_ARTIFACT,
        feature_manifest_path: Optional[
            Path
        ] = None,
    ) -> None:

        self.inference_artifact_path = Path(
            inference_artifact_path
        )

        # ----------------------------------------------------
        # Load inference artifact
        # ----------------------------------------------------

        self.inference_artifact = load_json(
            self.inference_artifact_path
        )

        self._load_inference_configuration()

        # ----------------------------------------------------
        # Load feature manifest
        # ----------------------------------------------------

        if feature_manifest_path is None:

            feature_manifest_path = (
                self._find_feature_manifest()
            )

        self.feature_manifest_path = Path(
            feature_manifest_path
        )

        self.feature_manifest = load_json(
            self.feature_manifest_path
        )

        self._load_feature_schema()

        # ----------------------------------------------------
        # Build median statistics
        # ----------------------------------------------------

        self.numeric_medians = (
            self._load_numeric_medians()
        )

        # ----------------------------------------------------
        # Final validation
        # ----------------------------------------------------

        self._validate_configuration()

    # ========================================================
    # LOAD INFERENCE CONFIGURATION
    # ========================================================

    def _load_inference_configuration(
        self,
    ) -> None:

        artifact = self.inference_artifact

        self.raw_feature_count = int(
            artifact[
                "raw_feature_count"
            ]
        )

        self.shared_feature_count = int(
            artifact[
                "shared_feature_count"
            ]
        )

        self.normalization_method = (
            artifact[
                "normalization_method"
            ]
        )

        self.normalization_formula = (
            artifact[
                "normalization_formula"
            ]
        )

        self.epsilon = float(
            artifact.get(
                "epsilon",
                EPSILON,
            )
        )

        self.removed_raw_indices = [
            int(index)
            for index in artifact[
                "removed_raw_feature_indices"
            ]
        ]

        self.shared_raw_indices = [
            int(index)
            for index in artifact[
                "shared_raw_feature_indices"
            ]
        ]

        self.global_raw_indices = [
            int(index)
            for index in artifact[
                "global_position_to_raw_feature_index"
            ]
        ]

        self.global_normalization_positions = [
            int(index)
            for index in artifact[
                "global_position_to_normalization_position"
            ]
        ]

        self.mean = torch.tensor(
            artifact["mean"],
            dtype=torch.float32,
        )

        self.std = torch.tensor(
            artifact["std"],
            dtype=torch.float32,
        )

    # ========================================================
    # FIND FEATURE MANIFEST
    # ========================================================

    def _find_feature_manifest(
        self,
    ) -> Path:

        for candidate in (
            FEATURE_MANIFEST_CANDIDATES
        ):

            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "Feature manifest could not be found.\n\n"
            "Expected one of:\n"
            + "\n".join(
                str(path)
                for path
                in FEATURE_MANIFEST_CANDIDATES
            )
        )

    # ========================================================
    # LOAD FEATURE SCHEMA
    # ========================================================

    def _load_feature_schema(
        self,
    ) -> None:

        manifest = self.feature_manifest

        self.feature_columns = list(
            manifest.get(
                "feature_columns",
                [],
            )
        )

        self.numeric_features = list(
            manifest.get(
                "numeric_features",
                [],
            )
        )

        self.categorical_features = list(
            manifest.get(
                "categorical_features",
                [],
            )
        )

        self.missingness_indicators = list(
            manifest.get(
                "missingness_indicators",
                [],
            )
        )

        self.removed_empty_features = list(
            manifest.get(
                "removed_empty_features",
                [],
            )
        )

        # The graph builder ultimately uses numeric columns
        # after preprocessing.
        #
        # Therefore the actual 814 graph features are derived
        # from numeric features + numeric missingness indicators.
        self.graph_numeric_features = [
            column
            for column in (
                self.numeric_features
            )
            if column not in NON_FEATURE_COLUMNS
        ]

        for indicator in (
            self.missingness_indicators
        ):

            if indicator not in (
                self.graph_numeric_features
            ):
                self.graph_numeric_features.append(
                    indicator
                )

    # ========================================================
    # LOAD NUMERIC MEDIANS
    # ========================================================

    def _load_numeric_medians(
        self,
    ) -> Dict[str, float]:
        """
        Load numeric median statistics.

        The original cleaning implementation uses the median
        of each numeric feature to fill missing numeric values.
        The cleaned training data contains those already-filled
        values.

        For real-time inference, medians are therefore loaded
        from the training split when available.

        This method first looks for a persisted median artifact.
        If unavailable, it calculates medians from the training
        CSV.
        """

        median_candidates = [

            PROJECT_ROOT
            / "data"
            / "processed"
            / "numeric_medians.json",

            PROJECT_ROOT
            / "data"
            / "processed"
            / "manifests"
            / "numeric_medians.json",

            PROJECT_ROOT
            / "data"
            / "processed"
            / "feature_medians.json",
        ]

        for path in median_candidates:

            if path.exists():

                data = load_json(
                    path
                )

                if (
                    "medians"
                    in data
                ):
                    data = data[
                        "medians"
                    ]

                if not isinstance(
                    data,
                    dict,
                ):
                    continue

                return {
                    str(column):
                        float(value)
                    for column, value
                    in data.items()
                }

        # ----------------------------------------------------
        # Training split fallback
        # ----------------------------------------------------

        train_candidates = [

            PROJECT_ROOT
            / "data"
            / "processed"
            / "splits"
            / "train.csv",

            PROJECT_ROOT
            / "data"
            / "processed"
            / "train.csv",
        ]

        train_path = None

        for candidate in (
            train_candidates
        ):

            if candidate.exists():

                train_path = candidate
                break

        if train_path is None:

            # If no training file is available, we don't silently
            # invent medians.
            raise FileNotFoundError(
                "No numeric median artifact or training CSV "
                "was found.\n\n"
                "Expected numeric_medians.json or:\n"
                "data\\processed\\splits\\train.csv"
            )

        print(
            "Calculating numeric medians from training split..."
        )

        required_columns = [
            column
            for column in self.numeric_features
            if column not in NON_FEATURE_COLUMNS
        ]

        medians = {}

        # Read only the numeric columns needed for medians.
        #
        # This keeps the operation smaller than loading the
        # complete 848-column dataset.
        dataframe = pd.read_csv(
            train_path,
            usecols=required_columns,
        )

        dataframe = dataframe.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        for column in required_columns:

            median = dataframe[
                column
            ].median()

            if pd.isna(
                median
            ):
                median = 0.0

            medians[
                column
            ] = float(median)

        return medians

    # ========================================================
    # VALIDATE CONFIGURATION
    # ========================================================

    def _validate_configuration(
        self,
    ) -> None:

        if self.raw_feature_count != EXPECTED_RAW_FEATURES:
            raise ValueError(
                "Inference artifact must contain "
                f"{EXPECTED_RAW_FEATURES} raw features."
            )

        if self.shared_feature_count != EXPECTED_MODEL_FEATURES:
            raise ValueError(
                "Inference artifact must contain "
                f"{EXPECTED_MODEL_FEATURES} shared features."
            )

        if len(
            self.global_raw_indices
        ) != EXPECTED_MODEL_FEATURES:

            raise ValueError(
                "Global raw feature mapping must contain 769 values."
            )

        if len(
            self.global_normalization_positions
        ) != EXPECTED_MODEL_FEATURES:

            raise ValueError(
                "Global normalization mapping must contain 769 values."
            )

        if self.mean.numel() != EXPECTED_MODEL_FEATURES:

            raise ValueError(
                "Mean vector must contain 769 values."
            )

        if self.std.numel() != EXPECTED_MODEL_FEATURES:

            raise ValueError(
                "Std vector must contain 769 values."
            )

        if torch.isnan(
            self.mean
        ).any():

            raise ValueError(
                "NaN found in normalization mean."
            )

        if torch.isnan(
            self.std
        ).any():

            raise ValueError(
                "NaN found in normalization std."
            )

        if torch.isinf(
            self.mean
        ).any():

            raise ValueError(
                "Infinity found in normalization mean."
            )

        if torch.isinf(
            self.std
        ).any():

            raise ValueError(
                "Infinity found in normalization std."
            )

        if torch.any(
            self.std <= 0
        ):

            raise ValueError(
                "Non-positive standard deviation detected."
            )

    # ========================================================
    # BUILD MISSINGNESS INDICATORS
    # ========================================================

    def _add_missingness_indicators(
        self,
        transaction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Reproduce the preprocessing rule:

            missing → indicator 1
            present → indicator 0

        The original preprocessing adds indicators only for
        features that have missing values in the training data.
        """

        processed = dict(
            transaction
        )

        for column in self.feature_columns:

            if column in NON_FEATURE_COLUMNS:
                continue

            indicator_name = (
                f"{column}__missing"
            )

            if indicator_name not in (
                self.missingness_indicators
            ):
                continue

            processed[
                indicator_name
            ] = int(
                is_missing(
                    transaction.get(
                        column
                    )
                )
            )

        return processed

    # ========================================================
    # HANDLE CATEGORICAL FEATURES
    # ========================================================

    def _handle_categorical_features(
        self,
        transaction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Reproduce the categorical preprocessing rule:

            missing categorical value → "MISSING"

        Categorical values themselves are not directly placed
        into graph.x. They remain available for graph relationship
        construction.
        """

        processed = dict(
            transaction
        )

        for column in (
            self.categorical_features
        ):

            if column not in processed:
                processed[
                    column
                ] = "MISSING"

                continue

            if is_missing(
                processed[column]
            ):

                processed[
                    column
                ] = "MISSING"

            else:

                processed[
                    column
                ] = str(
                    processed[column]
                ).strip()

        return processed

    # ========================================================
    # HANDLE NUMERIC FEATURES
    # ========================================================

    def _handle_numeric_features(
        self,
        transaction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Reproduce the numeric preprocessing rule:

            NaN → training median
            infinity → NaN → training median
        """

        processed = dict(
            transaction
        )

        for column in (
            self.numeric_features
        ):

            if column in NON_FEATURE_COLUMNS:
                continue

            value = processed.get(
                column
            )

            if is_missing(
                value
            ):

                processed[
                    column
                ] = self.numeric_medians.get(
                    column,
                    0.0,
                )

                continue

            try:

                number = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                processed[
                    column
                ] = self.numeric_medians.get(
                    column,
                    0.0,
                )

                continue

            if not math.isfinite(
                number
            ):

                processed[
                    column
                ] = self.numeric_medians.get(
                    column,
                    0.0,
                )

            else:

                processed[
                    column
                ] = number

        return processed

    # ========================================================
    # BUILD 814 RAW GRAPH FEATURES
    # ========================================================

    def _build_raw_graph_features(
        self,
        processed_transaction: Dict[str, Any],
    ) -> Tuple[
        torch.Tensor,
        List[str],
    ]:
        """
        Build the 814-dimensional raw numeric graph vector.

        The graph builder uses numeric columns as node features.
        """

        feature_names = list(
            self.graph_numeric_features
        )

        if len(feature_names) != EXPECTED_RAW_FEATURES:

            raise ValueError(
                "The preprocessing manifest produces "
                f"{len(feature_names)} numeric graph features, "
                f"but RFGN expects {EXPECTED_RAW_FEATURES}."
            )

        values = []

        for column in feature_names:

            value = processed_transaction.get(
                column,
                0.0,
            )

            values.append(
                safe_float(
                    value,
                    default=0.0,
                )
            )

        raw_tensor = torch.tensor(
            values,
            dtype=torch.float32,
        )

        if raw_tensor.numel() != EXPECTED_RAW_FEATURES:

            raise ValueError(
                "Raw feature tensor must contain 814 values."
            )

        if torch.isnan(
            raw_tensor
        ).any():

            raise ValueError(
                "NaN detected in raw feature tensor."
            )

        if torch.isinf(
            raw_tensor
        ).any():

            raise ValueError(
                "Infinity detected in raw feature tensor."
            )

        return (
            raw_tensor,
            feature_names,
        )

    # ========================================================
    # ALIGN TO GLOBAL 769 FEATURES
    # ========================================================

    def _align_to_global_features(
        self,
        raw_tensor: torch.Tensor,
        raw_feature_names: Sequence[str],
    ) -> torch.Tensor:
        """
        Select the exact 769 raw features in global model order.
        """

        if raw_tensor.numel() != EXPECTED_RAW_FEATURES:

            raise ValueError(
                "Expected 814 raw features."
            )

        if len(
            raw_feature_names
        ) != EXPECTED_RAW_FEATURES:

            raise ValueError(
                "Expected 814 raw feature names."
            )

        # ----------------------------------------------------
        # The preprocessing artifact stores raw feature indices.
        #
        # Verify that every requested global index exists.
        # ----------------------------------------------------

        for raw_index in (
            self.global_raw_indices
        ):

            if raw_index < 0:
                raise ValueError(
                    "Negative raw feature index."
                )

            if raw_index >= EXPECTED_RAW_FEATURES:
                raise ValueError(
                    "Raw feature index outside 814-feature range."
                )

        index_tensor = torch.tensor(
            self.global_raw_indices,
            dtype=torch.long,
        )

        aligned = raw_tensor[
            index_tensor
        ]

        if aligned.numel() != EXPECTED_MODEL_FEATURES:

            raise ValueError(
                "Global alignment did not produce 769 features."
            )

        return aligned

    # ========================================================
    # NORMALIZE
    # ========================================================

    def _normalize(
        self,
        aligned_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the exact stored training normalization.
        """

        if aligned_tensor.numel() != EXPECTED_MODEL_FEATURES:

            raise ValueError(
                "Expected 769 aligned features."
            )

        normalized = (
            aligned_tensor
            - self.mean
        ) / torch.clamp(
            self.std,
            min=self.epsilon,
        )

        normalized = normalized.float()

        if torch.isnan(
            normalized
        ).any():

            raise ValueError(
                "NaN detected after normalization."
            )

        if torch.isinf(
            normalized
        ).any():

            raise ValueError(
                "Infinity detected after normalization."
            )

        return normalized.contiguous()

    # ========================================================
    # MAIN TRANSFORM
    # ========================================================

    def transform(
        self,
        transaction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Transform one real-time transaction.

        Returns a dictionary containing:
            processed_transaction
            raw_features
            raw_feature_names
            aligned_features
            normalized_features
        """

        if not isinstance(
            transaction,
            dict,
        ):

            raise TypeError(
                "transaction must be a dictionary."
            )

        # Never allow a fraud label into inference.
        if "isFraud" in transaction:

            raise ValueError(
                "isFraud must not be supplied during real-time inference."
            )

        # ----------------------------------------------------
        # Processing order follows the training pipeline.
        # ----------------------------------------------------

        processed = (
            self._add_missingness_indicators(
                transaction
            )
        )

        processed = (
            self._handle_categorical_features(
                processed
            )
        )

        processed = (
            self._handle_numeric_features(
                processed
            )
        )

        # ----------------------------------------------------
        # Build raw features
        # ----------------------------------------------------

        raw_features, raw_feature_names = (
            self._build_raw_graph_features(
                processed
            )
        )

        # ----------------------------------------------------
        # 814 → 769
        # ----------------------------------------------------

        aligned_features = (
            self._align_to_global_features(
                raw_features,
                raw_feature_names,
            )
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        normalized_features = (
            self._normalize(
                aligned_features
            )
        )

        # ----------------------------------------------------
        # Final shape
        # ----------------------------------------------------

        final_features = (
            normalized_features
            .view(
                1,
                EXPECTED_MODEL_FEATURES,
            )
        )

        if final_features.shape != (
            1,
            EXPECTED_MODEL_FEATURES,
        ):

            raise ValueError(
                "Final feature tensor must have shape [1, 769]."
            )

        return {
            "processed_transaction":
                processed,

            "raw_features":
                raw_features,

            "raw_feature_names":
                raw_feature_names,

            "aligned_features":
                aligned_features,

            "normalized_features":
                final_features,
        }

    # ========================================================
    # SIMPLE TENSOR API
    # ========================================================

    def transform_to_tensor(
        self,
        transaction: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Return only the final [1, 769] tensor.
        """

        result = self.transform(
            transaction
        )

        return result[
            "normalized_features"
        ]

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:

        return {
            "module":
                "FeatureEngineer",

            "status":
                "healthy",

            "raw_feature_count":
                self.raw_feature_count,

            "shared_feature_count":
                self.shared_feature_count,

            "numeric_feature_count":
                len(
                    self.numeric_features
                ),

            "categorical_feature_count":
                len(
                    self.categorical_features
                ),

            "missingness_indicator_count":
                len(
                    self.missingness_indicators
                ),

            "graph_numeric_feature_count":
                len(
                    self.graph_numeric_features
                ),

            "normalization_method":
                self.normalization_method,

            "feature_manifest":
                str(
                    self.feature_manifest_path
                ),

            "inference_artifact":
                str(
                    self.inference_artifact_path
                ),
        }


# ============================================================
# SELF TEST
# ============================================================

def run_self_test() -> bool:
    """
    Validate the feature-engineering module.
    """

    print("=" * 70)
    print(
        "RFGN FEATURE ENGINEERING SELF-TEST"
    )
    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Load module
    # --------------------------------------------------------

    try:

        engineer = FeatureEngineer()

        print(
            "Configuration          : PASS"
        )

    except Exception as exc:

        print(
            "Configuration          : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Build a synthetic transaction.
    #
    # IMPORTANT:
    # The actual schema comes from the feature manifest.
    # --------------------------------------------------------

    transaction = {
        "TransactionID":
            "RT-FE-SELFTEST-001",

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

    # Provide a numeric value for every actual numeric
    # training feature.
    for position, column in enumerate(
        engineer.numeric_features
    ):

        if column in NON_FEATURE_COLUMNS:
            continue

        transaction[
            column
        ] = float(
            position + 1
        )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    try:

        result = engineer.transform(
            transaction
        )

        print(
            "Transaction transform : PASS"
        )

    except Exception as exc:

        print(
            "Transaction transform : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Raw feature count
    # --------------------------------------------------------

    raw_features = result[
        "raw_features"
    ]

    if raw_features.numel() != EXPECTED_RAW_FEATURES:

        print(
            "814 raw features     : FAIL"
        )

        print(
            f"Received: {raw_features.numel()}"
        )

        return False

    print(
        "814 raw features     : PASS"
    )

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    aligned = result[
        "aligned_features"
    ]

    if aligned.numel() != EXPECTED_MODEL_FEATURES:

        print(
            "769 aligned features : FAIL"
        )

        print(
            f"Received: {aligned.numel()}"
        )

        return False

    print(
        "769 aligned features : PASS"
    )

    # --------------------------------------------------------
    # Final tensor shape
    # --------------------------------------------------------

    normalized = result[
        "normalized_features"
    ]

    if normalized.shape != (
        1,
        EXPECTED_MODEL_FEATURES,
    ):

        print(
            "Final tensor shape    : FAIL"
        )

        print(
            f"Received: {tuple(normalized.shape)}"
        )

        return False

    print(
        "Final tensor shape    : PASS"
    )

    # --------------------------------------------------------
    # Numerical safety
    # --------------------------------------------------------

    if torch.isnan(
        normalized
    ).any():

        print(
            "NaN safety            : FAIL"
        )

        return False

    if torch.isinf(
        normalized
    ).any():

        print(
            "Infinity safety       : FAIL"
        )

        return False

    print(
        "NaN / Infinity safety : PASS"
    )

    # --------------------------------------------------------
    # Label exclusion
    # --------------------------------------------------------

    if "isFraud" in (
        result[
            "processed_transaction"
        ]
    ):

        print(
            "Fraud-label exclusion : FAIL"
        )

        return False

    print(
        "Fraud-label exclusion : PASS"
    )

    # --------------------------------------------------------
    # Missing-value test
    # --------------------------------------------------------

    missing_test = dict(
        transaction
    )

    # Pick the first actual numeric feature and make it missing.
    test_numeric_column = None

    for column in (
        engineer.numeric_features
    ):

        if column not in NON_FEATURE_COLUMNS:

            test_numeric_column = column
            break

    if test_numeric_column is not None:

        missing_test[
            test_numeric_column
        ] = None

        try:

            missing_result = engineer.transform(
                missing_test
            )

            missing_tensor = (
                missing_result[
                    "normalized_features"
                ]
            )

            if missing_tensor.shape != (
                1,
                EXPECTED_MODEL_FEATURES,
            ):

                print(
                    "Missing-value handling : FAIL"
                )

                return False

            if torch.isnan(
                missing_tensor
            ).any():

                print(
                    "Missing-value handling : FAIL"
                )

                return False

            print(
                "Missing-value handling : PASS"
            )

        except Exception as exc:

            print(
                "Missing-value handling : FAIL"
            )

            print(
                f"Error: {exc}"
            )

            return False

    # --------------------------------------------------------
    # Health check
    # --------------------------------------------------------

    health = engineer.health_check()

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
        "FEATURE ENGINEERING & ALIGNMENT: PASS"
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