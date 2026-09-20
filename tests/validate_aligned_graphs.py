# ============================================================
# RFGN ALIGNED GRAPH VALIDATION
# ============================================================
#
# Purpose:
#   Validate the three feature-aligned graphs before local
#   GraphSAGE training and federated FedAvg aggregation.
#
# IMPORTANT:
#   READ-ONLY VALIDATION
#
#   No graph will be modified.
#   No CSV dataset will be modified.
#   No model will be trained.
#   No model parameters will be changed.
#
# ============================================================

import os
import json
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
# CLIENT CONFIGURATION
# ============================================================

CLIENTS = [
    "client_1",
    "client_2",
    "client_3",
]


# ============================================================
# EXPECTED SHARED FEATURE COUNT
# ============================================================

EXPECTED_FEATURES = 769


# ============================================================
# PATH HELPERS
# ============================================================

def graph_path(client_id):

    return os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "graphs",
        client_id,
        "graph_aligned.pt",
    )


def alignment_manifest_path():

    return os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "graphs",
        "shared_feature_alignment.json",
    )


# ============================================================
# PRINT SECTION
# ============================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD ALIGNMENT MANIFEST
# ============================================================

def load_alignment_manifest():

    path = alignment_manifest_path()

    print(
        f"Manifest: {path}"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nShared feature alignment manifest "
            f"not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        manifest = json.load(file)

    return manifest


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph(client_id):

    path = graph_path(
        client_id
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nAligned graph not found:\n{path}"
        )

    graph = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    return graph


# ============================================================
# BASIC GRAPH VALIDATION
# ============================================================

def validate_basic_graph(
    graph,
    client_id,
):

    required_attributes = [
        "x",
        "edge_index",
        "y",
    ]

    for attribute in required_attributes:

        if not hasattr(
            graph,
            attribute,
        ):

            raise RuntimeError(
                f"{client_id}: "
                f"missing graph attribute "
                f"'{attribute}'."
            )


    # --------------------------------------------------------
    # Feature tensor
    # --------------------------------------------------------

    if not isinstance(
        graph.x,
        torch.Tensor,
    ):

        raise RuntimeError(
            f"{client_id}: x is not a tensor."
        )


    if graph.x.dim() != 2:

        raise RuntimeError(
            f"{client_id}: "
            "x must be a 2-dimensional tensor."
        )


    # --------------------------------------------------------
    # Edge index
    # --------------------------------------------------------

    if not isinstance(
        graph.edge_index,
        torch.Tensor,
    ):

        raise RuntimeError(
            f"{client_id}: "
            "edge_index is not a tensor."
        )


    if graph.edge_index.dim() != 2:

        raise RuntimeError(
            f"{client_id}: "
            "edge_index must be 2-dimensional."
        )


    if graph.edge_index.shape[0] != 2:

        raise RuntimeError(
            f"{client_id}: "
            "edge_index must have shape (2, E)."
        )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    if not isinstance(
        graph.y,
        torch.Tensor,
    ):

        raise RuntimeError(
            f"{client_id}: "
            "y is not a tensor."
        )


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_features(
    graph,
    client_id,
):

    # Expected shared dimension
    if graph.num_node_features != (
        EXPECTED_FEATURES
    ):

        raise RuntimeError(
            f"{client_id}: "
            f"expected {EXPECTED_FEATURES} "
            f"features but found "
            f"{graph.num_node_features}."
        )


    # Node / feature rows
    if graph.x.shape[0] != (
        graph.num_nodes
    ):

        raise RuntimeError(
            f"{client_id}: "
            "feature row count does not "
            "match node count."
        )


    # NaN
    nan_count = int(
        torch.isnan(
            graph.x
        ).sum().item()
    )

    if nan_count != 0:

        raise RuntimeError(
            f"{client_id}: "
            f"{nan_count} NaN values found."
        )


    # Infinity
    infinity_count = int(
        torch.isinf(
            graph.x
        ).sum().item()
    )

    if infinity_count != 0:

        raise RuntimeError(
            f"{client_id}: "
            f"{infinity_count} infinite values found."
        )


    # Finite
    if not torch.isfinite(
        graph.x
    ).all():

        raise RuntimeError(
            f"{client_id}: "
            "feature tensor contains non-finite values."
        )


# ============================================================
# EDGE VALIDATION
# ============================================================

def validate_edges(
    graph,
    client_id,
):

    edge_index = graph.edge_index


    # Shape
    if edge_index.shape[0] != 2:

        raise RuntimeError(
            f"{client_id}: "
            "invalid edge index shape."
        )


    # Empty edge check
    if edge_index.numel() == 0:

        raise RuntimeError(
            f"{client_id}: "
            "graph contains no edges."
        )


    minimum_index = int(
        edge_index.min().item()
    )

    maximum_index = int(
        edge_index.max().item()
    )


    if minimum_index < 0:

        raise RuntimeError(
            f"{client_id}: "
            "negative node index detected."
        )


    if maximum_index >= (
        graph.num_nodes
    ):

        raise RuntimeError(
            f"{client_id}: "
            "edge index exceeds node count."
        )


    # Edge dtype
    if not edge_index.dtype in (
        torch.int64,
        torch.long,
    ):

        raise RuntimeError(
            f"{client_id}: "
            "edge_index must use integer "
            "index dtype."
        )


# ============================================================
# LABEL VALIDATION
# ============================================================

def validate_labels(
    graph,
    client_id,
):

    if graph.y.shape[0] != (
        graph.num_nodes
    ):

        raise RuntimeError(
            f"{client_id}: "
            "label count does not match "
            "node count."
        )


    unique_labels = torch.unique(
        graph.y
    ).cpu().tolist()


    if not set(
        unique_labels
    ).issubset({0, 1}):

        raise RuntimeError(
            f"{client_id}: "
            f"invalid labels detected: "
            f"{unique_labels}"
        )


    fraud_count = int(
        (graph.y == 1).sum().item()
    )

    legitimate_count = int(
        (graph.y == 0).sum().item()
    )


    if fraud_count == 0:

        raise RuntimeError(
            f"{client_id}: "
            "no fraud labels found."
        )


    if legitimate_count == 0:

        raise RuntimeError(
            f"{client_id}: "
            "no legitimate labels found."
        )


# ============================================================
# GRAPH STATISTICS
# ============================================================

def calculate_statistics(
    graph,
):

    nodes = graph.num_nodes

    directed_edges = graph.num_edges

    undirected_edges = (
        directed_edges // 2
    )


    average_degree = (
        directed_edges / nodes
        if nodes > 0
        else 0.0
    )


    density = (
        directed_edges
        /
        (nodes * (nodes - 1))
        if nodes > 1
        else 0.0
    )


    isolated_nodes = int(
        (
            torch.bincount(
                graph.edge_index[0],
                minlength=nodes,
            ) == 0
        ).sum().item()
    )


    return {
        "nodes": nodes,
        "directed_edges": directed_edges,
        "undirected_edges": undirected_edges,
        "average_degree": average_degree,
        "density": density,
        "isolated_nodes": isolated_nodes,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "RFGN ALIGNED GRAPH VALIDATION"
    )

    print("=" * 70)

    print()

    print(
        "This script is READ-ONLY."
    )

    print(
        "No aligned graph will be modified."
    )

    print(
        "No CSV dataset will be modified."
    )

    print(
        "No model training will be performed."
    )

    print(
        "No model parameters will be updated."
    )


    # ========================================================
    # LOAD ALIGNMENT MANIFEST
    # ========================================================

    print_section(
        "CHECKING SHARED FEATURE ALIGNMENT MANIFEST"
    )


    manifest = (
        load_alignment_manifest()
    )


    if manifest.get(
        "shared_feature_count"
    ) != EXPECTED_FEATURES:

        raise RuntimeError(
            "\nShared feature count in manifest "
            "does not match expected value.\n"
            f"Expected: {EXPECTED_FEATURES}\n"
            f"Found: "
            f"{manifest.get('shared_feature_count')}"
        )


    print()

    print(
        "Shared feature count: "
        f"{manifest['shared_feature_count']}"
    )

    print(
        "Manifest feature count: PASS"
    )


    if manifest.get(
        "fedavg_parameter_shape_compatible"
    ) is not True:

        raise RuntimeError(
            "Manifest does not confirm "
            "FedAvg parameter compatibility."
        )


    print(
        "FedAvg compatibility flag: PASS"
    )


    # ========================================================
    # LOAD ALL GRAPHS
    # ========================================================

    print_section(
        "CHECKING ALIGNED CLIENT GRAPHS"
    )


    graphs = {}

    statistics = {}


    for client_id in CLIENTS:

        print()

        print(
            client_id.upper()
        )


        path = graph_path(
            client_id
        )


        print(
            f"Path: {path}"
        )


        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                f"{client_id}: "
                "aligned graph not found."
            )


        size_mb = (
            os.path.getsize(path)
            /
            (1024 ** 2)
        )


        print(
            f"Size: {size_mb:.2f} MB"
        )

        print(
            "Status: PRESENT"
        )


        graph = load_graph(
            client_id
        )


        graphs[
            client_id
        ] = graph


        print(
            "Graph loading: PASS"
        )


    # ========================================================
    # VALIDATE EACH GRAPH
    # ========================================================

    print_section(
        "VALIDATING ALIGNED GRAPH STRUCTURE"
    )


    for client_id in CLIENTS:

        graph = graphs[
            client_id
        ]


        print()

        print(
            client_id.upper()
        )


        # ----------------------------------------------------
        # Basic
        # ----------------------------------------------------

        validate_basic_graph(
            graph,
            client_id,
        )

        print(
            "Basic graph object: PASS"
        )


        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        validate_features(
            graph,
            client_id,
        )

        print(
            "Feature dimension: PASS"
        )

        print(
            "Feature row count: PASS"
        )

        print(
            "Feature NaN check: PASS"
        )

        print(
            "Feature infinity check: PASS"
        )


        # ----------------------------------------------------
        # Edges
        # ----------------------------------------------------

        validate_edges(
            graph,
            client_id,
        )

        print(
            "Edge index shape: PASS"
        )

        print(
            "Edge index bounds: PASS"
        )


        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        validate_labels(
            graph,
            client_id,
        )

        print(
            "Label count: PASS"
        )

        print(
            "Binary fraud labels: PASS"
        )


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        stats = calculate_statistics(
            graph
        )


        statistics[
            client_id
        ] = stats


    # ========================================================
    # CLIENT SUMMARY
    # ========================================================

    print_section(
        "ALIGNED CLIENT GRAPH SUMMARY"
    )


    for client_id in CLIENTS:

        graph = graphs[
            client_id
        ]

        stats = statistics[
            client_id
        ]


        fraud_count = int(
            (
                graph.y == 1
            ).sum().item()
        )


        legitimate_count = int(
            (
                graph.y == 0
            ).sum().item()
        )


        fraud_rate = (
            fraud_count
            /
            graph.num_nodes
            *
            100.0
        )


        print()

        print(
            client_id.upper()
        )

        print(
            f"  Nodes              : "
            f"{graph.num_nodes:,}"
        )

        print(
            f"  Features           : "
            f"{graph.num_node_features:,}"
        )

        print(
            f"  Directed edges     : "
            f"{stats['directed_edges']:,}"
        )

        print(
            f"  Undirected edges   : "
            f"{stats['undirected_edges']:,}"
        )

        print(
            f"  Legitimate         : "
            f"{legitimate_count:,}"
        )

        print(
            f"  Fraud              : "
            f"{fraud_count:,}"
        )

        print(
            f"  Fraud rate         : "
            f"{fraud_rate:.4f}%"
        )

        print(
            f"  Average degree     : "
            f"{stats['average_degree']:.4f}"
        )

        print(
            f"  Isolated nodes     : "
            f"{stats['isolated_nodes']:,}"
        )


    # ========================================================
    # CROSS CLIENT FEATURE COMPATIBILITY
    # ========================================================

    print_section(
        "CROSS-CLIENT FEATURE COMPATIBILITY"
    )


    dimensions = [
        graphs[
            client_id
        ].num_node_features
        for client_id in CLIENTS
    ]


    print()


    for client_id in CLIENTS:

        print(
            f"{client_id}: "
            f"{graphs[client_id].num_node_features}"
            " features"
        )


    if len(
        set(dimensions)
    ) != 1:

        raise RuntimeError(
            "Feature dimensions are not identical."
        )


    if dimensions[0] != (
        EXPECTED_FEATURES
    ):

        raise RuntimeError(
            "Shared feature dimension is incorrect."
        )


    print()

    print(
        "Identical feature dimensions: PASS"
    )

    print(
        "FedAvg input compatibility: PASS"
    )


    # ========================================================
    # CROSS CLIENT EDGE / LABEL INDEPENDENCE
    # ========================================================

    print_section(
        "VALIDATING CLIENT INDEPENDENCE"
    )


    for client_id in CLIENTS:

        graph = graphs[
            client_id
        ]


        # ----------------------------------------------------
        # Client-specific node count
        # ----------------------------------------------------

        if graph.num_nodes <= 0:

            raise RuntimeError(
                f"{client_id}: "
                "invalid node count."
            )


        # ----------------------------------------------------
        # Node indices must be local
        # ----------------------------------------------------

        maximum_index = int(
            graph.edge_index.max().item()
        )


        if maximum_index >= (
            graph.num_nodes
        ):

            raise RuntimeError(
                f"{client_id}: "
                "cross-client/global node index detected."
            )


        print(
            f"{client_id}: local node indexing: PASS"
        )


    print()

    print(
        "Client graphs remain independently indexed: PASS"
    )


    # ========================================================
    # FEATURE TENSOR COMPATIBILITY
    # ========================================================

    print_section(
        "VALIDATING FEATURE TENSOR COMPATIBILITY"
    )


    dtypes = []

    shapes = []


    for client_id in CLIENTS:

        graph = graphs[
            client_id
        ]


        dtypes.append(
            str(graph.x.dtype)
        )


        shapes.append(
            graph.x.shape[1]
        )


        print(
            f"{client_id}: "
            f"dtype={graph.x.dtype}, "
            f"features={graph.x.shape[1]}"
        )


    if len(
        set(dtypes)
    ) != 1:

        raise RuntimeError(
            "Feature tensor dtypes differ."
        )


    if len(
        set(shapes)
    ) != 1:

        raise RuntimeError(
            "Feature tensor dimensions differ."
        )


    print()

    print(
        "Feature tensor dtype compatibility: PASS"
    )

    print(
        "Feature tensor shape compatibility: PASS"
    )


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print_section(
        "FINAL VALIDATION"
    )


    all_passed = True


    for client_id in CLIENTS:

        graph = graphs[
            client_id
        ]


        if (
            graph.num_node_features
            != EXPECTED_FEATURES
        ):

            all_passed = False


        if (
            graph.x.shape[0]
            != graph.num_nodes
        ):

            all_passed = False


        if (
            graph.y.shape[0]
            != graph.num_nodes
        ):

            all_passed = False


        if torch.isnan(
            graph.x
        ).any():

            all_passed = False


        if torch.isinf(
            graph.x
        ).any():

            all_passed = False


        if graph.edge_index.shape[0] != 2:

            all_passed = False


    if not all_passed:

        raise RuntimeError(
            "FINAL VALIDATION FAILED."
        )


    print(
        "All aligned graphs: PASS"
    )

    print(
        "All feature tensors: PASS"
    )

    print(
        "All edge indices: PASS"
    )

    print(
        "All labels: PASS"
    )

    print(
        "All clients feature-compatible: PASS"
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print("=" * 70)

    print(
        "RFGN ALIGNED GRAPH VALIDATION SUMMARY"
    )

    print("=" * 70)

    print()

    print(
        f"Clients validated       : {len(CLIENTS)}"
    )

    print(
        f"Shared features         : {EXPECTED_FEATURES}"
    )


    total_nodes = sum(
        graphs[
            client_id
        ].num_nodes
        for client_id in CLIENTS
    )


    total_edges = sum(
        graphs[
            client_id
        ].num_edges
        for client_id in CLIENTS
    )


    total_fraud = sum(
        int(
            (
                graphs[
                    client_id
                ].y == 1
            ).sum().item()
        )
        for client_id in CLIENTS
    )


    print(
        f"Total client nodes      : "
        f"{total_nodes:,}"
    )

    print(
        f"Total directed edges   : "
        f"{total_edges:,}"
    )

    print(
        f"Total fraud labels      : "
        f"{total_fraud:,}"
    )

    print()

    print(
        "Feature dimensions      : 769 / 769 / 769"
    )

    print(
        "NaN values              : 0"
    )

    print(
        "Infinite values         : 0"
    )

    print(
        "Feature compatibility   : YES"
    )

    print(
        "FedAvg compatibility    : YES"
    )

    print(
        "Graph modification      : NO"
    )

    print(
        "Training performed      : NO"
    )


    print()

    print("=" * 70)

    print(
        "ALIGNED GRAPH VALIDATION: PASSED"
    )

    print("=" * 70)

    print()

    print(
        "The three aligned graphs are ready "
        "for shared-dimension GraphSAGE training."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Local GraphSAGE training using "
        "graph_aligned.pt for Clients 1, 2 and 3."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()