# ============================================================
# RFGN CLIENT 1 GRAPH DIAGNOSTIC
# ============================================================

import os
import sys

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
# GRAPH PATH
# ============================================================

GRAPH_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs",
    "client_1",
    "graph.pt",
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_FEATURES = 814


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RFGN CLIENT 1 GRAPH DIAGNOSTIC")
    print("=" * 70)

    print()
    print("READ-ONLY DIAGNOSTIC")
    print("No graph will be modified.")
    print("No model will be trained.")

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CHECKING GRAPH")
    print("-" * 70)

    if not os.path.exists(GRAPH_PATH):

        raise FileNotFoundError(
            f"Graph not found:\n{GRAPH_PATH}"
        )

    print("Graph file: PRESENT")
    print("Path:", GRAPH_PATH)

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LOADING GRAPH")
    print("-" * 70)

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )

    print("Graph loaded successfully.")

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
    # Basic validation
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("BASIC GRAPH VALIDATION")
    print("-" * 70)

    assert (
        graph.num_node_features
        == EXPECTED_FEATURES
    )

    print(
        "Feature dimension: PASS"
    )

    assert (
        graph.x.shape[0]
        == graph.num_nodes
    )

    print(
        "Feature row count: PASS"
    )

    # --------------------------------------------------------
    # Tensor information
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("FEATURE TENSOR INFORMATION")
    print("-" * 70)

    print(
        "Shape:",
        tuple(graph.x.shape)
    )

    print(
        "dtype:",
        graph.x.dtype
    )

    print(
        "Device:",
        graph.x.device
    )

    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("NAN / INFINITY ANALYSIS")
    print("-" * 70)

    nan_mask = torch.isnan(
        graph.x
    )

    inf_mask = torch.isinf(
        graph.x
    )

    nan_count = int(
        nan_mask.sum().item()
    )

    inf_count = int(
        inf_mask.sum().item()
    )

    print(
        f"NaN values      : {nan_count:,}"
    )

    print(
        f"Infinite values : {inf_count:,}"
    )

    if nan_count == 0:

        print(
            "NaN check: PASS"
        )

    else:

        print(
            "NaN check: FAIL"
        )

    if inf_count == 0:

        print(
            "Infinity check: PASS"
        )

    else:

        print(
            "Infinity check: FAIL"
        )

    # --------------------------------------------------------
    # Finite check
    # --------------------------------------------------------

    finite_mask = torch.isfinite(
        graph.x
    )

    finite_count = int(
        finite_mask.sum().item()
    )

    total_values = graph.x.numel()

    print()
    print(
        f"Finite values   : "
        f"{finite_count:,} / {total_values:,}"
    )

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("GLOBAL FEATURE STATISTICS")
    print("-" * 70)

    finite_values = graph.x[
        finite_mask
    ]

    if finite_values.numel() > 0:

        print(
            "Minimum:",
            finite_values.min().item()
        )

        print(
            "Maximum:",
            finite_values.max().item()
        )

        print(
            "Mean:",
            finite_values.mean().item()
        )

        print(
            "Std:",
            finite_values.std().item()
        )

    # --------------------------------------------------------
    # Per-feature statistics
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("PER-FEATURE STATISTICS")
    print("-" * 70)

    x = graph.x.float()

    feature_min = torch.min(
        x,
        dim=0
    ).values

    feature_max = torch.max(
        x,
        dim=0
    ).values

    feature_mean = torch.mean(
        x,
        dim=0
    )

    feature_std = torch.std(
        x,
        dim=0
    )

    # --------------------------------------------------------
    # Extreme features
    # --------------------------------------------------------

    print()
    print("Top 20 features by maximum absolute value:")

    max_abs = torch.max(
        torch.abs(x),
        dim=0
    ).values

    top_indices = torch.argsort(
        max_abs,
        descending=True
    )[:20]

    print()

    for rank, index in enumerate(
        top_indices.tolist(),
        start=1
    ):

        print(
            f"{rank:02d}. "
            f"Feature {index:03d} | "
            f"min={feature_min[index].item():.6g} | "
            f"max={feature_max[index].item():.6g} | "
            f"mean={feature_mean[index].item():.6g} | "
            f"std={feature_std[index].item():.6g}"
        )

    # --------------------------------------------------------
    # High variance features
    # --------------------------------------------------------

    print()
    print(
        "Top 20 features by standard deviation:"
    )

    std_indices = torch.argsort(
        feature_std,
        descending=True
    )[:20]

    print()

    for rank, index in enumerate(
        std_indices.tolist(),
        start=1
    ):

        print(
            f"{rank:02d}. "
            f"Feature {index:03d} | "
            f"std={feature_std[index].item():.6g} | "
            f"min={feature_min[index].item():.6g} | "
            f"max={feature_max[index].item():.6g}"
        )

    # --------------------------------------------------------
    # Zero variance features
    # --------------------------------------------------------

    zero_std_count = int(
        (
            feature_std == 0
        )
        .sum()
        .item()
    )

    print()
    print(
        f"Zero-variance features: "
        f"{zero_std_count}"
    )

    # --------------------------------------------------------
    # Very large values
    # --------------------------------------------------------

    thresholds = [
        1e2,
        1e3,
        1e4,
        1e5,
        1e6,
        1e9,
    ]

    print()
    print("-" * 70)
    print("EXTREME VALUE COUNTS")
    print("-" * 70)

    for threshold in thresholds:

        count = int(
            (
                torch.abs(x)
                > threshold
            )
            .sum()
            .item()
        )

        print(
            f"|x| > {threshold:<10.0e}: "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # Graph edge validation
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("EDGE INDEX VALIDATION")
    print("-" * 70)

    edge_index = (
        graph.edge_index
    )

    print(
        "Edge index shape:",
        tuple(edge_index.shape)
    )

    min_edge = int(
        edge_index.min().item()
    )

    max_edge = int(
        edge_index.max().item()
    )

    print(
        "Minimum node index:",
        min_edge
    )

    print(
        "Maximum node index:",
        max_edge
    )

    if (
        min_edge >= 0
        and max_edge < graph.num_nodes
    ):

        print(
            "Edge bounds: PASS"
        )

    else:

        print(
            "Edge bounds: FAIL"
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LABEL VALIDATION")
    print("-" * 70)

    labels = graph.y

    unique_labels, counts = (
        torch.unique(
            labels,
            return_counts=True
        )
    )

    for label, count in zip(
        unique_labels.tolist(),
        counts.tolist()
    ):

        print(
            f"Class {label}: "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # Check for non-finite model input
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("MODEL INPUT SAFETY")
    print("-" * 70)

    if (
        nan_count == 0
        and inf_count == 0
    ):

        print(
            "Finite feature tensor: PASS"
        )

    else:

        print(
            "Finite feature tensor: FAIL"
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RFGN CLIENT 1 GRAPH DIAGNOSTIC COMPLETED")
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This diagnostic did not modify the graph."
    )

    print(
        "No model training was performed."
    )

    print(
        "No validation/test data was used."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()