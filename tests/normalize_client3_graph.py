# ============================================================
# RFGN CLIENT 3 GRAPH NORMALIZATION
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
    "client_3",
    "graph.pt",
)

OUTPUT_GRAPH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs",
    "client_3",
    "graph_normalized.pt",
)

MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs",
    "client_3",
    "normalization_manifest.json",
)


# ============================================================
# UTILITY
# ============================================================

def print_separator():
    print("-" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RFGN CLIENT 3 GRAPH NORMALIZATION")
    print("=" * 70)

    print()
    print("This script creates a normalized copy.")
    print("Original graph.pt will NOT be modified.")
    print("No model training will be performed.")
    print("No validation/test data will be used.")

    # ========================================================
    # CHECK INPUT
    # ========================================================

    print()
    print_separator()
    print("CHECKING INPUT GRAPH")
    print_separator()

    if not os.path.exists(INPUT_GRAPH):
        raise FileNotFoundError(
            f"Input graph not found:\n{INPUT_GRAPH}"
        )

    print("Input graph: PRESENT")
    print("Path:", INPUT_GRAPH)

    # ========================================================
    # LOAD GRAPH
    # ========================================================

    print()
    print_separator()
    print("LOADING GRAPH")
    print_separator()

    graph = torch.load(
        INPUT_GRAPH,
        map_location="cpu",
        weights_only=False,
    )

    print("Graph loaded successfully.")

    print(
        f"Nodes    : {graph.num_nodes:,}"
    )

    print(
        f"Features : {graph.num_node_features:,}"
    )

    print(
        f"Edges    : {graph.num_edges:,}"
    )

    print(
        f"Labels   : {graph.y.shape[0]:,}"
    )

    original_nodes = graph.num_nodes
    original_features = graph.num_node_features
    original_edges = graph.num_edges

    # ========================================================
    # VALIDATE INPUT FEATURES
    # ========================================================

    print()
    print_separator()
    print("VALIDATING INPUT FEATURES")
    print_separator()

    if torch.isnan(graph.x).any():
        raise ValueError(
            "Input graph contains NaN values."
        )

    print("Input NaN check: PASS")

    if torch.isinf(graph.x).any():
        raise ValueError(
            "Input graph contains infinite values."
        )

    print("Input infinity check: PASS")

    if graph.x.dtype != torch.float32:

        print(
            f"Converting feature dtype "
            f"from {graph.x.dtype} to float32."
        )

        graph.x = graph.x.float()

    # ========================================================
    # FEATURE STATISTICS
    # ========================================================

    print()
    print_separator()
    print("CALCULATING FEATURE STATISTICS")
    print_separator()

    x = graph.x

    feature_mean = x.mean(
        dim=0
    )

    feature_std = x.std(
        dim=0,
        unbiased=False,
    )

    print("Statistics calculated: PASS")

    # ========================================================
    # ZERO VARIANCE FEATURES
    # ========================================================

    print()
    print_separator()
    print("IDENTIFYING ZERO-VARIANCE FEATURES")
    print_separator()

    zero_variance_mask = (
        feature_std <= 1e-12
    )

    zero_variance_indices = torch.where(
        zero_variance_mask
    )[0]

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
            "Zero-variance feature removal: NONE"
        )

    keep_mask = ~zero_variance_mask

    kept_indices = torch.where(
        keep_mask
    )[0]

    kept_features = kept_indices.numel()

    print()
    print(
        f"Original features : "
        f"{original_features}"
    )

    print(
        f"Kept features     : "
        f"{kept_features}"
    )

    print(
        f"Removed features  : "
        f"{zero_variance_count}"
    )

    # ========================================================
    # STANDARDIZATION
    # ========================================================

    print()
    print_separator()
    print("STANDARDIZING FEATURES")
    print_separator()

    print("Method: Standard score")

    print("Formula:")

    print("  z = (x - mean) / std")

    print("Applying normalization...")

    x_kept = x[
        :,
        kept_indices
    ]

    mean_kept = feature_mean[
        kept_indices
    ]

    std_kept = feature_std[
        kept_indices
    ]

    std_kept = torch.clamp(
        std_kept,
        min=1e-12,
    )

    normalized_x = (
        x_kept - mean_kept
    ) / std_kept

    # ========================================================
    # VALIDATE NORMALIZED FEATURES
    # ========================================================

    print()
    print_separator()
    print("VALIDATING NORMALIZED FEATURES")
    print_separator()

    normalized_nan = int(
        torch.isnan(
            normalized_x
        ).sum().item()
    )

    normalized_inf = int(
        torch.isinf(
            normalized_x
        ).sum().item()
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
            "NaN values detected after normalization."
        )

    print("Normalized NaN check: PASS")

    if normalized_inf != 0:
        raise ValueError(
            "Infinite values detected after normalization."
        )

    print("Normalized infinity check: PASS")

    # ========================================================
    # GLOBAL STATISTICS
    # ========================================================

    global_min = normalized_x.min().item()
    global_max = normalized_x.max().item()
    global_mean = normalized_x.mean().item()
    global_std = normalized_x.std(
        unbiased=False
    ).item()

    print()
    print("Normalized global statistics:")

    print(
        f"Minimum : {global_min:.6f}"
    )

    print(
        f"Maximum : {global_max:.6f}"
    )

    print(
        f"Mean    : {global_mean:.6f}"
    )

    print(
        f"Std     : {global_std:.6f}"
    )

    # ========================================================
    # CREATE NORMALIZED GRAPH
    # ========================================================

    print()
    print_separator()
    print("CREATING NORMALIZED GRAPH COPY")
    print_separator()

    normalized_graph = graph.clone()

    normalized_graph.x = (
        normalized_x.contiguous()
    )

    # ========================================================
    # PRESERVATION CHECKS
    # ========================================================

    if not torch.equal(
        normalized_graph.y,
        graph.y,
    ):
        raise ValueError(
            "Fraud labels changed."
        )

    print(
        "Fraud label preservation: PASS"
    )

    if not torch.equal(
        normalized_graph.edge_index,
        graph.edge_index,
    ):
        raise ValueError(
            "Edge index changed."
        )

    print(
        "Edge preservation: PASS"
    )

    if (
        normalized_graph.num_nodes
        != original_nodes
    ):
        raise ValueError(
            "Node count changed."
        )

    print(
        "Node count preservation: PASS"
    )

    if (
        normalized_graph.num_edges
        != original_edges
    ):
        raise ValueError(
            "Edge count changed."
        )

    print(
        "Edge count preservation: PASS"
    )

    # ========================================================
    # SAVE GRAPH
    # ========================================================

    print()
    print_separator()
    print("SAVING NORMALIZED GRAPH")
    print_separator()

    os.makedirs(
        os.path.dirname(
            OUTPUT_GRAPH
        ),
        exist_ok=True,
    )

    print("Saving...")

    torch.save(
        normalized_graph,
        OUTPUT_GRAPH,
    )

    file_size_mb = (
        os.path.getsize(
            OUTPUT_GRAPH
        )
        / (1024 ** 2)
    )

    print("Saved to:")

    print(OUTPUT_GRAPH)

    print(
        f"File size: "
        f"{file_size_mb:.2f} MB"
    )

    print("Graph save: PASS")

    # ========================================================
    # NORMALIZATION MANIFEST
    # ========================================================

    print()
    print_separator()
    print("CREATING NORMALIZATION MANIFEST")
    print_separator()

    manifest = {

        "client": "client_3",

        "input_graph": INPUT_GRAPH,

        "output_graph": OUTPUT_GRAPH,

        "method": "standard_score",

        "formula": "(x - mean) / std",

        "original_nodes":
            int(original_nodes),

        "original_features":
            int(original_features),

        "removed_zero_variance_features":
            int(zero_variance_count),

        "final_features":
            int(kept_features),

        "original_edges":
            int(original_edges),

        "final_edges":
            int(normalized_graph.num_edges),

        "normalization": {

            "mean": [
                float(v)
                for v in mean_kept.tolist()
            ],

            "std": [
                float(v)
                for v in std_kept.tolist()
            ],

            "kept_feature_indices": [
                int(v)
                for v in kept_indices.tolist()
            ],

            "removed_feature_indices": [
                int(v)
                for v in zero_variance_indices.tolist()
            ],
        },

        "validation": {

            "input_nan": 0,

            "input_infinity": 0,

            "output_nan":
                int(normalized_nan),

            "output_infinity":
                int(normalized_inf),

            "labels_preserved": True,

            "edges_preserved": True,

            "nodes_preserved": True,
        },
    }

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
        )

    print("Manifest saved to:")

    print(MANIFEST_PATH)

    print("Manifest save: PASS")

    # ========================================================
    # FINAL RELOAD VALIDATION
    # ========================================================

    print()
    print_separator()
    print("FINAL NORMALIZED GRAPH VALIDATION")
    print_separator()

    reloaded_graph = torch.load(
        OUTPUT_GRAPH,
        map_location="cpu",
        weights_only=False,
    )

    print(
        "Reloaded normalized graph: PASS"
    )

    if (
        reloaded_graph.num_nodes
        != original_nodes
    ):
        raise ValueError(
            "Final node count mismatch."
        )

    print("Node count: PASS")

    if (
        reloaded_graph.num_edges
        != original_edges
    ):
        raise ValueError(
            "Final edge count mismatch."
        )

    print("Edge count: PASS")

    if (
        reloaded_graph.y.shape[0]
        != original_nodes
    ):
        raise ValueError(
            "Final label count mismatch."
        )

    print("Label count: PASS")

    final_nan = int(
        torch.isnan(
            reloaded_graph.x
        ).sum().item()
    )

    if final_nan != 0:
        raise ValueError(
            "Final normalized graph contains NaN."
        )

    print("Final NaN check: PASS")

    final_inf = int(
        torch.isinf(
            reloaded_graph.x
        ).sum().item()
    )

    if final_inf != 0:
        raise ValueError(
            "Final normalized graph contains infinity."
        )

    print("Final infinity check: PASS")

    if not torch.equal(
        reloaded_graph.y,
        graph.y,
    ):
        raise ValueError(
            "Final target preservation failed."
        )

    print(
        "Final target preservation: PASS"
    )

    if not torch.equal(
        reloaded_graph.edge_index,
        graph.edge_index,
    ):
        raise ValueError(
            "Final edge preservation failed."
        )

    print(
        "Final edge preservation: PASS"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("RFGN CLIENT 3 GRAPH NORMALIZATION SUMMARY")
    print("=" * 70)

    print()

    print(
        f"Original nodes       : "
        f"{original_nodes:,}"
    )

    print(
        f"Original features    : "
        f"{original_features:,}"
    )

    print(
        f"Removed zero-var     : "
        f"{zero_variance_count:,}"
    )

    print(
        f"Final features       : "
        f"{kept_features:,}"
    )

    print(
        f"Edges preserved      : "
        f"{original_edges:,}"
    )

    print(
        f"NaN after processing : "
        f"{final_nan:,}"
    )

    print(
        f"Inf after processing : "
        f"{final_inf:,}"
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

    print("NEXT:")

    print(
        "Run the GraphSAGE forward-pass test "
        "using graph_normalized.pt."
    )

    print()

    print("=" * 70)
    print(
        "RFGN CLIENT 3 GRAPH NORMALIZATION COMPLETED"
    )
    print("=" * 70)

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()