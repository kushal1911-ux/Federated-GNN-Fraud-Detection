"""
RFGN - Create Inference Preprocessing Artifact

Creates a reusable preprocessing artifact containing the exact
feature-selection, feature-alignment, and normalization information
required for real-time inference.

Source:
    Client 2 normalization manifest
    Shared feature alignment manifest
    Client 2 original graph

This script does NOT modify:
    - training CSV files
    - normalized graphs
    - aligned graphs
    - trained models
    - labels
    - edges
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRAPHS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graphs"
)

CLIENT_2_MANIFEST = (
    GRAPHS_DIR
    / "client_2"
    / "normalization_manifest.json"
)

ALIGNMENT_MANIFEST = (
    GRAPHS_DIR
    / "shared_feature_alignment.json"
)

CLIENT_2_GRAPH = (
    GRAPHS_DIR
    / "client_2"
    / "graph.pt"
)

OUTPUT_PATH = (
    GRAPHS_DIR
    / "inference_preprocessing.json"
)


# ============================================================
# EXPECTED RFGN CONFIGURATION
# ============================================================

EXPECTED_RAW_FEATURES = 814
EXPECTED_SHARED_FEATURES = 769

EPSILON = 1e-12

# JSON statistics are compared against float32 graph data.
# A small difference can occur during float32 representation.
MEAN_TOLERANCE = 0.02
STD_TOLERANCE = 1e-5


# ============================================================
# HELPERS
# ============================================================

def print_separator():
    print("=" * 70)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_numeric_list(
    values,
    expected_length: int,
    name: str,
):
    if not isinstance(values, list):
        raise ValueError(
            f"{name} must be a list."
        )

    if len(values) != expected_length:
        raise ValueError(
            f"{name} length mismatch: "
            f"expected {expected_length}, "
            f"got {len(values)}"
        )

    tensor = torch.tensor(
        values,
        dtype=torch.float64,
    )

    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"{name} contains NaN or infinity."
        )


def validate_indices(
    values,
    name: str,
):
    if not isinstance(values, list):
        raise ValueError(
            f"{name} must be a list."
        )

    for position, value in enumerate(values):

        if not isinstance(
            value,
            int,
        ):
            raise ValueError(
                f"{name}[{position}] is not an integer."
            )

        if value < 0:
            raise ValueError(
                f"{name}[{position}] is negative."
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print_separator()
    print("RFGN INFERENCE PREPROCESSING ARTIFACT")
    print_separator()

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    # ========================================================
    # LOAD CLIENT 2 NORMALIZATION MANIFEST
    # ========================================================

    print()
    print_separator()
    print("LOADING CLIENT 2 NORMALIZATION MANIFEST")
    print_separator()

    client_2_manifest = load_json(
        CLIENT_2_MANIFEST
    )

    normalization = (
        client_2_manifest.get(
            "normalization"
        )
    )

    if normalization is None:
        raise ValueError(
            "Client 2 normalization section missing."
        )

    mean = normalization.get(
        "mean"
    )

    std = normalization.get(
        "std"
    )

    kept_indices = normalization.get(
        "kept_feature_indices"
    )

    removed_indices = normalization.get(
        "removed_feature_indices"
    )

    if mean is None:
        raise ValueError(
            "Client 2 mean statistics missing."
        )

    if std is None:
        raise ValueError(
            "Client 2 standard deviation statistics missing."
        )

    if kept_indices is None:
        raise ValueError(
            "Client 2 kept feature indices missing."
        )

    if removed_indices is None:
        raise ValueError(
            "Client 2 removed feature indices missing."
        )

    print(
        "Client 2 normalization manifest: PASS"
    )

    # ========================================================
    # VALIDATE CLIENT 2 NORMALIZATION
    # ========================================================

    print()
    print_separator()
    print("VALIDATING CLIENT 2 NORMALIZATION")
    print_separator()

    original_features = int(
        client_2_manifest[
            "original_features"
        ]
    )

    final_features = int(
        client_2_manifest[
            "final_features"
        ]
    )

    if original_features != EXPECTED_RAW_FEATURES:
        raise ValueError(
            "Unexpected Client 2 original feature count: "
            f"{original_features}"
        )

    if final_features != EXPECTED_SHARED_FEATURES:
        raise ValueError(
            "Unexpected Client 2 final feature count: "
            f"{final_features}"
        )

    validate_numeric_list(
        mean,
        EXPECTED_SHARED_FEATURES,
        "mean",
    )

    validate_numeric_list(
        std,
        EXPECTED_SHARED_FEATURES,
        "std",
    )

    validate_indices(
        kept_indices,
        "kept_feature_indices",
    )

    validate_indices(
        removed_indices,
        "removed_feature_indices",
    )

    if len(kept_indices) != EXPECTED_SHARED_FEATURES:
        raise ValueError(
            "Client 2 kept feature count is not 769."
        )

    if (
        len(kept_indices)
        + len(removed_indices)
        != EXPECTED_RAW_FEATURES
    ):
        raise ValueError(
            "Kept + removed feature count does not equal 814."
        )

    if len(
        set(kept_indices)
    ) != len(kept_indices):

        raise ValueError(
            "Duplicate kept feature indices detected."
        )

    if len(
        set(removed_indices)
    ) != len(removed_indices):

        raise ValueError(
            "Duplicate removed feature indices detected."
        )

    overlap = (
        set(kept_indices)
        & set(removed_indices)
    )

    if overlap:
        raise ValueError(
            "Kept and removed feature indices overlap."
        )

    if any(
        index >= EXPECTED_RAW_FEATURES
        for index in kept_indices
    ):
        raise ValueError(
            "A kept feature index is outside the 814-feature range."
        )

    if any(
        index >= EXPECTED_RAW_FEATURES
        for index in removed_indices
    ):
        raise ValueError(
            "A removed feature index is outside the 814-feature range."
        )

    for index, value in enumerate(std):

        if float(value) <= 0:
            raise ValueError(
                f"Non-positive standard deviation at position {index}."
            )

    print(
        f"Original features : {original_features}"
    )

    print(
        f"Kept features     : {final_features}"
    )

    print(
        f"Removed features  : {len(removed_indices)}"
    )

    print(
        "Mean statistics   : PASS"
    )

    print(
        "Std statistics    : PASS"
    )

    print(
        "Feature indices   : PASS"
    )

    # ========================================================
    # LOAD SHARED FEATURE ALIGNMENT
    # ========================================================

    print()
    print_separator()
    print("LOADING SHARED FEATURE ALIGNMENT")
    print_separator()

    alignment = load_json(
        ALIGNMENT_MANIFEST
    )

    shared_count = int(
        alignment[
            "shared_feature_count"
        ]
    )

    shared_indices = alignment[
        "shared_feature_indices"
    ]

    canonical_order_source = alignment[
        "canonical_order_source"
    ]

    if shared_count != EXPECTED_SHARED_FEATURES:
        raise ValueError(
            "Shared feature count is not 769."
        )

    if len(shared_indices) != EXPECTED_SHARED_FEATURES:
        raise ValueError(
            "Shared feature index list does not contain 769 entries."
        )

    validate_indices(
        shared_indices,
        "shared_feature_indices",
    )

    print(
        f"Shared feature count : {shared_count}"
    )

    print(
        f"Canonical source     : {canonical_order_source}"
    )

    print(
        "Shared alignment     : PASS"
    )

    # ========================================================
    # VERIFY CLIENT 2 FEATURE SET
    # ========================================================

    print()
    print_separator()
    print("VERIFYING CLIENT 2 → SHARED MAPPING")
    print_separator()

    client_2_set = set(
        int(index)
        for index in kept_indices
    )

    shared_set = set(
        int(index)
        for index in shared_indices
    )

    if client_2_set != shared_set:

        missing = sorted(
            shared_set
            - client_2_set
        )

        extra = sorted(
            client_2_set
            - shared_set
        )

        raise ValueError(
            "Client 2 retained feature set does not match "
            "the shared feature set.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    print(
        "Client 2 feature set == shared feature set: PASS"
    )

    # ========================================================
    # BUILD NORMALIZATION POSITION MAPPING
    # ========================================================

    print()
    print_separator()
    print("BUILDING FEATURE POSITION MAPPING")
    print_separator()

    normalization_position_by_raw_index = {
        int(raw_index): int(position)
        for position, raw_index
        in enumerate(kept_indices)
    }

    global_position_to_raw_index = []

    global_position_to_normalization_position = []

    for global_position, raw_index in enumerate(
        shared_indices
    ):

        raw_index = int(
            raw_index
        )

        if raw_index not in (
            normalization_position_by_raw_index
        ):
            raise ValueError(
                f"Shared raw feature {raw_index} "
                "does not exist in Client 2 normalization."
            )

        normalization_position = (
            normalization_position_by_raw_index[
                raw_index
            ]
        )

        global_position_to_raw_index.append(
            raw_index
        )

        global_position_to_normalization_position.append(
            normalization_position
        )

    if len(
        global_position_to_raw_index
    ) != EXPECTED_SHARED_FEATURES:

        raise ValueError(
            "Global raw feature mapping must contain 769 values."
        )

    if len(
        global_position_to_normalization_position
    ) != EXPECTED_SHARED_FEATURES:

        raise ValueError(
            "Normalization position mapping must contain 769 values."
        )

    print(
        "Raw → normalization mapping: PASS"
    )

    print(
        "Normalization → global mapping: PASS"
    )

    # ========================================================
    # LOAD CLIENT 2 ORIGINAL GRAPH
    # ========================================================

    print()
    print_separator()
    print("VERIFYING AGAINST CLIENT 2 GRAPH")
    print_separator()

    if not CLIENT_2_GRAPH.exists():
        raise FileNotFoundError(
            f"Client 2 graph not found:\n{CLIENT_2_GRAPH}"
        )

    graph = torch.load(
        CLIENT_2_GRAPH,
        map_location="cpu",
        weights_only=False,
    )

    if not hasattr(
        graph,
        "x",
    ):
        raise ValueError(
            "Client 2 graph does not contain x."
        )

    if graph.x.ndim != 2:
        raise ValueError(
            "Client 2 graph x tensor is not two-dimensional."
        )

    graph_features = int(
        graph.x.shape[1]
    )

    if graph_features != EXPECTED_RAW_FEATURES:
        raise ValueError(
            "Client 2 graph must contain "
            f"{EXPECTED_RAW_FEATURES} features, "
            f"got {graph_features}."
        )

    print(
        f"Graph input features : {graph_features}"
    )

    print(
        "Graph feature count   : PASS"
    )

    # ========================================================
    # VERIFY GRAPH NUMERICAL SAFETY
    # ========================================================

    print()
    print_separator()
    print("VERIFYING GRAPH NUMERICAL SAFETY")
    print_separator()

    graph_x = graph.x.float()

    if torch.isnan(
        graph_x
    ).any():

        raise ValueError(
            "NaN detected in Client 2 graph."
        )

    if torch.isinf(
        graph_x
    ).any():

        raise ValueError(
            "Infinity detected in Client 2 graph."
        )

    print(
        "Graph NaN check       : PASS"
    )

    print(
        "Graph infinity check  : PASS"
    )

    # ========================================================
    # VERIFY NORMALIZATION STATISTICS
    # ========================================================

    print()
    print_separator()
    print("VERIFYING NORMALIZATION STATISTICS")
    print_separator()

    kept_index_tensor = torch.tensor(
        kept_indices,
        dtype=torch.long,
    )

    selected_x = graph_x[
        :,
        kept_index_tensor,
    ]

    calculated_mean = selected_x.mean(
        dim=0
    )

    calculated_std = selected_x.std(
        dim=0,
        unbiased=False,
    )

    manifest_mean = torch.tensor(
        mean,
        dtype=torch.float32,
    )

    manifest_std = torch.tensor(
        std,
        dtype=torch.float32,
    )

    mean_difference = torch.max(
        torch.abs(
            calculated_mean
            - manifest_mean
        )
    ).item()

    std_difference = torch.max(
        torch.abs(
            calculated_std
            - manifest_std
        )
    ).item()

    print(
        f"Maximum mean difference : "
        f"{mean_difference:.10f}"
    )

    print(
        f"Maximum std difference  : "
        f"{std_difference:.10f}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The graph is represented in float32 and the normalization
    # manifest is serialized through JSON. The mean comparison
    # therefore allows a small representation difference.
    #
    # We are NOT replacing the stored training statistics.
    # The exact values from the normalization manifest remain
    # the values used by inference.
    # --------------------------------------------------------

    if mean_difference > MEAN_TOLERANCE:

        raise ValueError(
            "Stored mean statistics differ from the Client 2 "
            "graph beyond the accepted tolerance.\n"
            f"Observed difference: {mean_difference}\n"
            f"Allowed difference : {MEAN_TOLERANCE}"
        )

    if std_difference > STD_TOLERANCE:

        raise ValueError(
            "Stored std statistics differ from the Client 2 "
            "graph beyond the accepted tolerance.\n"
            f"Observed difference: {std_difference}\n"
            f"Allowed difference : {STD_TOLERANCE}"
        )

    print(
        f"Mean verification : PASS "
        f"(tolerance={MEAN_TOLERANCE})"
    )

    print(
        f"Std verification  : PASS "
        f"(tolerance={STD_TOLERANCE})"
    )

    # ========================================================
    # BUILD GLOBAL-ORDER STATISTICS
    # ========================================================

    print()
    print_separator()
    print("BUILDING GLOBAL FEATURE ORDER")
    print_separator()

    global_mean = []

    global_std = []

    for global_position in range(
        EXPECTED_SHARED_FEATURES
    ):

        normalization_position = (
            global_position_to_normalization_position[
                global_position
            ]
        )

        global_mean.append(
            float(
                mean[
                    normalization_position
                ]
            )
        )

        global_std.append(
            float(
                std[
                    normalization_position
                ]
            )
        )

    if len(global_mean) != EXPECTED_SHARED_FEATURES:
        raise ValueError(
            "Global mean vector length mismatch."
        )

    if len(global_std) != EXPECTED_SHARED_FEATURES:
        raise ValueError(
            "Global std vector length mismatch."
        )

    print(
        "Global mean vector : 769"
    )

    print(
        "Global std vector  : 769"
    )

    print(
        "Global feature order: PASS"
    )

    # ========================================================
    # CREATE ARTIFACT
    # ========================================================

    print()
    print_separator()
    print("CREATING INFERENCE PREPROCESSING ARTIFACT")
    print_separator()

    artifact = {

        "project":
            "RFGN",

        "purpose":
            "Exact preprocessing configuration for "
            "real-time GraphSAGE inference",

        "source_client":
            "client_2",

        "raw_feature_count":
            EXPECTED_RAW_FEATURES,

        "shared_feature_count":
            EXPECTED_SHARED_FEATURES,

        "normalization_method":
            "standard_score",

        "normalization_formula":
            "(x - mean) / std",

        "epsilon":
            EPSILON,

        "canonical_order_source":
            canonical_order_source,

        "raw_feature_indices":
            list(
                range(
                    EXPECTED_RAW_FEATURES
                )
            ),

        "removed_raw_feature_indices":
            [
                int(index)
                for index in removed_indices
            ],

        "shared_raw_feature_indices":
            [
                int(index)
                for index in shared_indices
            ],

        "global_position_to_raw_feature_index":
            global_position_to_raw_index,

        "global_position_to_normalization_position":
            global_position_to_normalization_position,

        "mean":
            global_mean,

        "std":
            global_std,

        "validation":
            {

                "client_2_feature_set_matches_shared":
                    True,

                "raw_feature_count_verified":
                    True,

                "shared_feature_count_verified":
                    True,

                "mean_verified":
                    True,

                "std_verified":
                    True,

                "feature_order_verified":
                    True,

                "source_graph_modified":
                    False,

                "trained_model_modified":
                    False,

                "training_statistics_replaced":
                    False,
            },
    }

    os.makedirs(
        OUTPUT_PATH.parent,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            artifact,
            file,
            indent=2,
        )

    print(
        "Artifact saved to:"
    )

    print(
        OUTPUT_PATH
    )

    # ========================================================
    # RELOAD ARTIFACT
    # ========================================================

    print()
    print_separator()
    print("VALIDATING SAVED ARTIFACT")
    print_separator()

    saved_artifact = load_json(
        OUTPUT_PATH
    )

    if (
        saved_artifact[
            "raw_feature_count"
        ]
        != EXPECTED_RAW_FEATURES
    ):
        raise ValueError(
            "Saved raw feature count is invalid."
        )

    if (
        saved_artifact[
            "shared_feature_count"
        ]
        != EXPECTED_SHARED_FEATURES
    ):
        raise ValueError(
            "Saved shared feature count is invalid."
        )

    if len(
        saved_artifact["mean"]
    ) != EXPECTED_SHARED_FEATURES:

        raise ValueError(
            "Saved mean vector length is invalid."
        )

    if len(
        saved_artifact["std"]
    ) != EXPECTED_SHARED_FEATURES:

        raise ValueError(
            "Saved std vector length is invalid."
        )

    if len(
        saved_artifact[
            "global_position_to_raw_feature_index"
        ]
    ) != EXPECTED_SHARED_FEATURES:

        raise ValueError(
            "Saved raw feature mapping is invalid."
        )

    if len(
        saved_artifact[
            "global_position_to_normalization_position"
        ]
    ) != EXPECTED_SHARED_FEATURES:

        raise ValueError(
            "Saved normalization mapping is invalid."
        )

    print(
        "Artifact reload       : PASS"
    )

    print(
        "814 → 769 mapping     : PASS"
    )

    print(
        "Mean/std preservation : PASS"
    )

    print(
        "Global order          : PASS"
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print_separator()
    print("INFERENCE PREPROCESSING ARTIFACT: PASS")
    print_separator()

    print()
    print(
        "Artifact:"
    )

    print(
        OUTPUT_PATH
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()