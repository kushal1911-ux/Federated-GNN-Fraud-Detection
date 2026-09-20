# ============================================================
# RFGN CLIENT 1 GRAPH NORMALIZATION
# ============================================================

import os
import sys
import json

import torch


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PATHS
# ============================================================

INPUT_GRAPH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs",
    "client_1",
    "graph.pt",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs",
    "client_1",
)

OUTPUT_GRAPH = os.path.join(
    OUTPUT_DIR,
    "graph_normalized.pt",
)

MANIFEST_PATH = os.path.join(
    OUTPUT_DIR,
    "normalization_manifest.json",
)


# ============================================================
# CONFIGURATION
# ============================================================

EPSILON = 1e-8


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RFGN CLIENT 1 GRAPH NORMALIZATION")
    print("=" * 70)

    print()
    print("This script creates a normalized copy.")
    print("Original graph.pt will NOT be modified.")
    print("No model training will be performed.")
    print("No validation/test data will be used.")

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CHECKING INPUT GRAPH")
    print("-" * 70)

    if not os.path.exists(INPUT_GRAPH):

        raise FileNotFoundError(
            f"Input graph not found:\n{INPUT_GRAPH}"
        )

    print(
        "Input graph: PRESENT"
    )

    print(
        "Path:",
        INPUT_GRAPH
    )

    # --------------------------------------------------------
    # LOAD GRAPH
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LOADING GRAPH")
    print("-" * 70)

    graph = torch.load(
        INPUT_GRAPH,
        map_location="cpu",
        weights_only=False,
    )

    print(
        "Graph loaded successfully."
    )

    print(
        f"Nodes    : {graph.num_nodes:,}"
    )

    print(
        f"Features : {graph.num_node_features}"
    )

    print(
        f"Edges    : {graph.num_edges:,}"
    )

    print(
        f"Labels   : {graph.y.shape[0]:,}"
    )

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("VALIDATING INPUT FEATURES")
    print("-" * 70)

    x = graph.x.float()

    if torch.isnan(x).any():

        raise ValueError(
            "Input feature tensor contains NaN values."
        )

    if torch.isinf(x).any():

        raise ValueError(
            "Input feature tensor contains infinity."
        )

    print(
        "Input NaN check: PASS"
    )

    print(
        "Input infinity check: PASS"
    )

    # --------------------------------------------------------
    # FEATURE STATISTICS
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CALCULATING FEATURE STATISTICS")
    print("-" * 70)

    feature_mean = torch.mean(
        x,
        dim=0
    )

    feature_std = torch.std(
        x,
        dim=0
    )

    feature_min = torch.min(
        x,
        dim=0
    ).values

    feature_max = torch.max(
        x,
        dim=0
    ).values

    print(
        "Statistics calculated: PASS"
    )

    # --------------------------------------------------------
    # ZERO-VARIANCE FEATURES
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("IDENTIFYING ZERO-VARIANCE FEATURES")
    print("-" * 70)

    zero_variance_mask = (
        feature_std <= EPSILON
    )

    zero_variance_indices = (
        torch.where(
            zero_variance_mask
        )[0]
    )

    zero_variance_count = (
        zero_variance_indices.numel()
    )

    print(
        f"Zero-variance features: "
        f"{zero_variance_count}"
    )

    if zero_variance_count > 0:

        print(
            "Zero-variance feature removal: REQUIRED"
        )

    else:

        print(
            "Zero-variance feature removal: NOT REQUIRED"
        )

    # --------------------------------------------------------
    # KEEP FEATURES
    # --------------------------------------------------------

    keep_mask = ~zero_variance_mask

    kept_indices = torch.where(
        keep_mask
    )[0]

    kept_feature_count = (
        kept_indices.numel()
    )

    print()
    print(
        f"Original features : "
        f"{x.shape[1]}"
    )

    print(
        f"Kept features     : "
        f"{kept_feature_count}"
    )

    print(
        f"Removed features  : "
        f"{zero_variance_count}"
    )

    # --------------------------------------------------------
    # SELECT NON-CONSTANT FEATURES
    # --------------------------------------------------------

    x_kept = x[
        :,
        keep_mask
    ]

    mean_kept = feature_mean[
        keep_mask
    ]

    std_kept = feature_std[
        keep_mask
    ]

    # --------------------------------------------------------
    # STANDARDIZATION
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("STANDARDIZING FEATURES")
    print("-" * 70)

    print(
        "Method: Standard score"
    )

    print(
        "Formula:"
    )

    print(
        "  z = (x - mean) / std"
    )

    print(
        "Applying normalization..."
    )

    x_normalized = (
        x_kept
        - mean_kept
    ) / (
        std_kept
        + EPSILON
    )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("VALIDATING NORMALIZED FEATURES")
    print("-" * 70)

    normalized_nan = int(
        torch.isnan(
            x_normalized
        )
        .sum()
        .item()
    )

    normalized_inf = int(
        torch.isinf(
            x_normalized
        )
        .sum()
        .item()
    )

    print(
        f"NaN values      : "
        f"{normalized_nan:,}"
    )

    print(
        f"Infinite values : "
        f"{normalized_inf:,}"
    )

    if normalized_nan != 0:

        raise ValueError(
            "Normalization produced NaN values."
        )

    if normalized_inf != 0:

        raise ValueError(
            "Normalization produced infinity values."
        )

    print(
        "Normalized NaN check: PASS"
    )

    print(
        "Normalized infinity check: PASS"
    )

    # --------------------------------------------------------
    # NORMALIZED STATISTICS
    # --------------------------------------------------------

    normalized_mean = torch.mean(
        x_normalized,
        dim=0
    )

    normalized_std = torch.std(
        x_normalized,
        dim=0
    )

    normalized_min = torch.min(
        x_normalized,
        dim=0
    ).values

    normalized_max = torch.max(
        x_normalized,
        dim=0
    ).values

    print()
    print(
        "Normalized global statistics:"
    )

    print(
        f"Minimum : "
        f"{x_normalized.min().item():.6f}"
    )

    print(
        f"Maximum : "
        f"{x_normalized.max().item():.6f}"
    )

    print(
        f"Mean    : "
        f"{x_normalized.mean().item():.6f}"
    )

    print(
        f"Std     : "
        f"{x_normalized.std().item():.6f}"
    )

    # --------------------------------------------------------
    # GRAPH COPY
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CREATING NORMALIZED GRAPH COPY")
    print("-" * 70)

    normalized_graph = graph.clone()

    normalized_graph.x = (
        x_normalized.contiguous()
    )

    # --------------------------------------------------------
    # PRESERVE ORIGINAL TARGET
    # --------------------------------------------------------

    if not torch.equal(
        graph.y,
        normalized_graph.y
    ):

        raise ValueError(
            "Target labels changed during normalization."
        )

    print(
        "Fraud label preservation: PASS"
    )

    # --------------------------------------------------------
    # PRESERVE EDGES
    # --------------------------------------------------------

    if not torch.equal(
        graph.edge_index,
        normalized_graph.edge_index
    ):

        raise ValueError(
            "Edge index changed during normalization."
        )

    print(
        "Edge preservation: PASS"
    )

    # --------------------------------------------------------
    # GRAPH DIMENSIONS
    # --------------------------------------------------------

    if (
        normalized_graph.x.shape[0]
        != graph.x.shape[0]
    ):

        raise ValueError(
            "Node count changed."
        )

    if (
        normalized_graph.edge_index.shape
        != graph.edge_index.shape
    ):

        raise ValueError(
            "Edge index shape changed."
        )

    print(
        "Node count preservation: PASS"
    )

    print(
        "Edge count preservation: PASS"
    )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SAVE NORMALIZED GRAPH
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("SAVING NORMALIZED GRAPH")
    print("-" * 70)

    print(
        "Saving..."
    )

    torch.save(
        normalized_graph,
        OUTPUT_GRAPH
    )

    if not os.path.exists(
        OUTPUT_GRAPH
    ):

        raise RuntimeError(
            "Normalized graph was not saved."
        )

    file_size_mb = (
        os.path.getsize(
            OUTPUT_GRAPH
        )
        / (1024 ** 2)
    )

    print(
        "Saved to:"
    )

    print(
        OUTPUT_GRAPH
    )

    print(
        f"File size: "
        f"{file_size_mb:.2f} MB"
    )

    print(
        "Graph save: PASS"
    )

    # --------------------------------------------------------
    # CREATE MANIFEST
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CREATING NORMALIZATION MANIFEST")
    print("-" * 70)

    manifest = {
        "client": "client_1",

        "input_graph": INPUT_GRAPH,

        "output_graph": OUTPUT_GRAPH,

        "original_feature_count": int(
            x.shape[1]
        ),

        "removed_zero_variance_features": int(
            zero_variance_count
        ),

        "final_feature_count": int(
            x_normalized.shape[1]
        ),

        "normalization_method": (
            "standardization"
        ),

        "formula": (
            "(x - mean) / std"
        ),

        "epsilon": EPSILON,

        "nodes": int(
            graph.num_nodes
        ),

        "edges": int(
            graph.num_edges
        ),

        "labels_preserved": True,

        "edges_preserved": True,

        "nan_after_normalization": (
            normalized_nan
        ),

        "infinity_after_normalization": (
            normalized_inf
        ),

        "removed_feature_indices": (
            zero_variance_indices
            .tolist()
        ),
    }

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4
        )

    print(
        "Manifest saved to:"
    )

    print(
        MANIFEST_PATH
    )

    print(
        "Manifest save: PASS"
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL NORMALIZED GRAPH VALIDATION")
    print("=" * 70)

    reloaded_graph = torch.load(
        OUTPUT_GRAPH,
        map_location="cpu",
        weights_only=False,
    )

    print(
        "Reloaded normalized graph: PASS"
    )

    assert (
        reloaded_graph.num_nodes
        == graph.num_nodes
    )

    print(
        "Node count: PASS"
    )

    assert (
        reloaded_graph.num_edges
        == graph.num_edges
    )

    print(
        "Edge count: PASS"
    )

    assert (
        reloaded_graph.y.shape
        == graph.y.shape
    )

    print(
        "Label count: PASS"
    )

    assert not torch.isnan(
        reloaded_graph.x
    ).any()

    print(
        "Final NaN check: PASS"
    )

    assert not torch.isinf(
        reloaded_graph.x
    ).any()

    print(
        "Final infinity check: PASS"
    )

    assert torch.equal(
        reloaded_graph.y,
        graph.y
    )

    print(
        "Final target preservation: PASS"
    )

    assert torch.equal(
        reloaded_graph.edge_index,
        graph.edge_index
    )

    print(
        "Final edge preservation: PASS"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RFGN CLIENT 1 GRAPH NORMALIZATION SUMMARY")
    print("=" * 70)

    print()

    print(
        f"Original nodes       : "
        f"{graph.num_nodes:,}"
    )

    print(
        f"Original features    : "
        f"{x.shape[1]:,}"
    )

    print(
        f"Removed zero-var     : "
        f"{zero_variance_count:,}"
    )

    print(
        f"Final features       : "
        f"{x_normalized.shape[1]:,}"
    )

    print(
        f"Edges preserved      : "
        f"{graph.num_edges:,}"
    )

    print(
        f"NaN after processing : "
        f"{normalized_nan:,}"
    )

    print(
        f"Inf after processing : "
        f"{normalized_inf:,}"
    )

    print()
    print(
        "Original graph modified: NO"
    )

    print(
        "Original graph preserved: YES"
    )

    print(
        "Normalized graph created: YES"
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "Run the GraphSAGE forward-pass test "
        "using graph_normalized.pt."
    )

    print()
    print("=" * 70)
    print(
        "RFGN CLIENT 1 GRAPH NORMALIZATION COMPLETED"
    )
    print("=" * 70)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()