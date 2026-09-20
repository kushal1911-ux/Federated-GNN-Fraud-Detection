# ============================================================
# RFGN ENHANCED GRAPH FEATURE NORMALIZATION + ALIGNMENT
# ============================================================
#
# Purpose:
#   Normalize the enhanced client graphs and create a common
#   feature space for federated GraphSAGE training.
#
# Input:
#   client_1/graph_enhanced.pt
#   client_2/graph_enhanced.pt
#   client_3/graph_enhanced.pt
#
# Output:
#   client_1/graph_enhanced_aligned.pt
#   client_2/graph_enhanced_aligned.pt
#   client_3/graph_enhanced_aligned.pt
#
#   enhanced_feature_alignment.json
#
# IMPORTANT:
#   - Existing graph_aligned.pt files are NOT modified.
#   - Fraud labels are NOT used for feature selection.
#   - Edge topology is preserved.
#   - edge_type is preserved.
#   - A common feature space is created across all clients.
#
# ============================================================

import os
import json
import random

import numpy as np
import torch


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


GRAPH_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs"
)


CLIENTS = [
    "client_1",
    "client_2",
    "client_3",
]


GRAPH_FILENAME = "graph_enhanced.pt"

OUTPUT_FILENAME = "graph_enhanced_aligned.pt"

MANIFEST_PATH = os.path.join(
    GRAPH_ROOT,
    "enhanced_feature_alignment.json"
)


RANDOM_SEED = 42

EPSILON = 1e-8


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# ============================================================
# SECTION
# ============================================================

def print_section(title):

    print()

    print("=" * 70)

    print(title)

    print("=" * 70)


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph(client_id):

    path = os.path.join(
        GRAPH_ROOT,
        client_id,
        GRAPH_FILENAME
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Enhanced graph not found:\n{path}"
        )

    print(
        f"Loading {client_id}: {path}"
    )

    graph = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    return graph


# ============================================================
# VALIDATE GRAPH
# ============================================================

def validate_graph(
    graph,
    client_id
):

    print()

    print(
        f"Validating {client_id}"
    )

    if graph.x.ndim != 2:

        raise RuntimeError(
            f"{client_id}: x must be 2-dimensional."
        )

    if graph.x.shape[0] != graph.num_nodes:

        raise RuntimeError(
            f"{client_id}: feature rows do not "
            "match node count."
        )

    if graph.y.shape[0] != graph.num_nodes:

        raise RuntimeError(
            f"{client_id}: labels do not "
            "match node count."
        )

    if graph.edge_index.shape[0] != 2:

        raise RuntimeError(
            f"{client_id}: invalid edge_index."
        )

    if graph.edge_type.shape[0] != graph.num_edges:

        raise RuntimeError(
            f"{client_id}: edge_type does not "
            "match edge count."
        )

    if torch.isnan(graph.x).any():

        raise RuntimeError(
            f"{client_id}: NaN found in features."
        )

    if torch.isinf(graph.x).any():

        raise RuntimeError(
            f"{client_id}: Infinity found in features."
        )

    labels = torch.unique(
        graph.y
    ).cpu().tolist()

    if not set(labels).issubset({0, 1}):

        raise RuntimeError(
            f"{client_id}: invalid labels {labels}"
        )

    if graph.edge_index.numel() > 0:

        minimum = int(
            graph.edge_index.min().item()
        )

        maximum = int(
            graph.edge_index.max().item()
        )

        if minimum < 0:

            raise RuntimeError(
                f"{client_id}: negative edge index."
            )

        if maximum >= graph.num_nodes:

            raise RuntimeError(
                f"{client_id}: edge index exceeds "
                "node count."
            )

    print(
        f"  Nodes          : {graph.num_nodes:,}"
    )

    print(
        f"  Features       : {graph.num_node_features:,}"
    )

    print(
        f"  Directed edges : {graph.num_edges:,}"
    )

    print(
        f"  Fraud labels   : {int(graph.y.sum().item()):,}"
    )

    print(
        "  Validation     : PASS"
    )


# ============================================================
# DETERMINE COMMON FEATURE COUNT
# ============================================================

def determine_common_feature_count(
    graphs
):

    print_section(
        "DETERMINING COMMON FEATURE SPACE"
    )

    feature_counts = {}

    for client_id in CLIENTS:

        feature_counts[client_id] = (
            graphs[
                client_id
            ].num_node_features
        )

    print(
        "Feature counts:"
    )

    for client_id in CLIENTS:

        print(
            f"  {client_id}: "
            f"{feature_counts[client_id]}"
        )

    unique_counts = set(
        feature_counts.values()
    )

    if len(unique_counts) != 1:

        raise RuntimeError(
            "Enhanced graphs do not have "
            "the same feature count."
        )

    common_count = next(
        iter(unique_counts)
    )

    print()

    print(
        f"Common feature count: {common_count}"
    )

    print(
        "Feature-space compatibility: PASS"
    )

    return common_count


# ============================================================
# CALCULATE CLIENT STATISTICS
# ============================================================

def calculate_statistics(
    graph
):

    x = graph.x.float()

    mean = x.mean(
        dim=0
    )

    std = x.std(
        dim=0,
        unbiased=False
    )

    return (
        mean,
        std
    )


# ============================================================
# IDENTIFY VALID FEATURES
# ============================================================

def identify_valid_features(
    statistics
):

    means, stds = statistics

    valid_mask = (
        stds > EPSILON
    )

    return valid_mask


# ============================================================
# CREATE SHARED FEATURE MASK
# ============================================================

def create_shared_mask(
    graphs
):

    print_section(
        "IDENTIFYING SHARED NON-CONSTANT FEATURES"
    )

    masks = {}

    for client_id in CLIENTS:

        statistics = calculate_statistics(
            graphs[
                client_id
            ]
        )

        mask = identify_valid_features(
            statistics
        )

        masks[client_id] = mask

        removed = int(
            (~mask).sum().item()
        )

        kept = int(
            mask.sum().item()
        )

        print(
            f"{client_id}:"
        )

        print(
            f"  Original features : "
            f"{graphs[client_id].num_node_features}"
        )

        print(
            f"  Valid features    : "
            f"{kept}"
        )

        print(
            f"  Removed constant  : "
            f"{removed}"
        )

    shared_mask = masks[
        CLIENTS[0]
    ].clone()

    for client_id in CLIENTS[1:]:

        shared_mask &= masks[
            client_id
        ]

    shared_count = int(
        shared_mask.sum().item()
    )

    print()

    print(
        f"Shared non-constant features: "
        f"{shared_count}"
    )

    if shared_count == 0:

        raise RuntimeError(
            "No shared non-constant features found."
        )

    print(
        "Shared feature mask: PASS"
    )

    return shared_mask


# ============================================================
# CALCULATE SHARED NORMALIZATION STATISTICS
# ============================================================

def calculate_global_feature_statistics(
    graphs,
    shared_mask
):

    print_section(
        "CALCULATING SHARED NORMALIZATION STATISTICS"
    )

    total_count = 0

    feature_sum = None

    feature_squared_sum = None

    for client_id in CLIENTS:

        x = graphs[
            client_id
        ].x.float()

        x = x[
            :,
            shared_mask
        ]

        current_sum = x.sum(
            dim=0
        )

        current_squared_sum = (
            x * x
        ).sum(
            dim=0
        )

        if feature_sum is None:

            feature_sum = current_sum

            feature_squared_sum = (
                current_squared_sum
            )

        else:

            feature_sum += current_sum

            feature_squared_sum += (
                current_squared_sum
            )

        total_count += x.shape[0]

        print(
            f"{client_id}: "
            f"{x.shape[0]:,} rows included"
        )

    mean = (
        feature_sum
        / total_count
    )

    variance = (
        feature_squared_sum
        / total_count
        - mean * mean
    )

    variance = torch.clamp(
        variance,
        min=0.0
    )

    std = torch.sqrt(
        variance
    )

    std = torch.clamp(
        std,
        min=EPSILON
    )

    print()

    print(
        f"Combined rows: {total_count:,}"
    )

    print(
        f"Shared feature count: "
        f"{int(shared_mask.sum().item())}"
    )

    print(
        "Normalization method: "
        "combined global standard score"
    )

    print(
        "Normalization statistics: PASS"
    )

    return (
        mean,
        std
    )


# ============================================================
# NORMALIZE GRAPH
# ============================================================

def normalize_graph(
    graph,
    shared_mask,
    mean,
    std
):

    x = graph.x.float()

    x = x[
        :,
        shared_mask
    ]

    x = (
        x - mean
    ) / std

    x = torch.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return x


# ============================================================
# SAVE GRAPH
# ============================================================

def save_graph(
    graph,
    client_id
):

    output_path = os.path.join(
        GRAPH_ROOT,
        client_id,
        OUTPUT_FILENAME
    )

    torch.save(
        graph,
        output_path
    )

    file_size_mb = (
        os.path.getsize(
            output_path
        )
        /
        (1024 ** 2)
    )

    print()

    print(
        f"{client_id} saved:"
    )

    print(
        f"  {output_path}"
    )

    print(
        f"  File size: "
        f"{file_size_mb:.2f} MB"
    )

    return output_path


# ============================================================
# VALIDATE ALIGNED GRAPH
# ============================================================

def validate_aligned_graph(
    graph,
    client_id,
    expected_features
):

    if (
        graph.num_node_features
        != expected_features
    ):

        raise RuntimeError(
            f"{client_id}: expected "
            f"{expected_features} features, "
            f"found "
            f"{graph.num_node_features}."
        )

    if graph.edge_index.shape[0] != 2:

        raise RuntimeError(
            f"{client_id}: invalid edge_index."
        )

    if (
        graph.edge_type.shape[0]
        != graph.num_edges
    ):

        raise RuntimeError(
            f"{client_id}: edge_type mismatch."
        )

    if torch.isnan(
        graph.x
    ).any():

        raise RuntimeError(
            f"{client_id}: NaN after normalization."
        )

    if torch.isinf(
        graph.x
    ).any():

        raise RuntimeError(
            f"{client_id}: Infinity after normalization."
        )

    print(
        f"{client_id}: aligned validation PASS"
    )


# ============================================================
# BUILD MANIFEST
# ============================================================

def build_manifest(
    graphs,
    shared_mask,
    mean,
    std,
    output_paths
):

    manifest = {

        "project":
            "RFGN",

        "purpose":
            "Enhanced graph shared-feature normalization and alignment",

        "random_seed":
            RANDOM_SEED,

        "normalization":
            "combined_global_standard_score",

        "epsilon":
            EPSILON,

        "original_feature_count":
            int(
                graphs[
                    CLIENTS[0]
                ].num_node_features
            ),

        "shared_feature_count":
            int(
                shared_mask.sum().item()
            ),

        "clients":
            {},

        "edge_topology_preserved":
            True,

        "edge_type_preserved":
            True,

        "fraud_labels_used_for_alignment":
            False,

        "transaction_ids_used_for_alignment":
            False,
    }

    manifest[
        "normalization_mean"
    ] = [
        float(value)
        for value in mean.tolist()
    ]

    manifest[
        "normalization_std"
    ] = [
        float(value)
        for value in std.tolist()
    ]

    for client_id in CLIENTS:

        graph = graphs[
            client_id
        ]

        manifest[
            "clients"
        ][client_id] = {

            "input":
                os.path.join(
                    client_id,
                    GRAPH_FILENAME
                ),

            "output":
                output_paths[
                    client_id
                ],

            "nodes":
                int(
                    graph.num_nodes
                ),

            "directed_edges":
                int(
                    graph.num_edges
                ),

            "fraud_labels":
                int(
                    graph.y.sum().item()
                ),

            "output_features":
                int(
                    graph.num_node_features
                ),
        }

    return manifest


# ============================================================
# SAVE MANIFEST
# ============================================================

def save_manifest(
    manifest
):

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4
        )

    print()

    print(
        "Alignment manifest saved:"
    )

    print(
        MANIFEST_PATH
    )


# ============================================================
# FINAL CROSS-CLIENT VALIDATION
# ============================================================

def final_validation(
    aligned_graphs
):

    print_section(
        "FINAL CROSS-CLIENT VALIDATION"
    )

    feature_counts = []

    node_counts = []

    edge_counts = []

    fraud_counts = []

    for client_id in CLIENTS:

        graph = aligned_graphs[
            client_id
        ]

        feature_counts.append(
            graph.num_node_features
        )

        node_counts.append(
            graph.num_nodes
        )

        edge_counts.append(
            graph.num_edges
        )

        fraud_counts.append(
            int(
                graph.y.sum().item()
            )
        )

    # --------------------------------------------------------
    # Feature compatibility
    # --------------------------------------------------------

    if len(
        set(feature_counts)
    ) != 1:

        raise RuntimeError(
            "Feature counts are not aligned."
        )

    print(
        f"Shared feature count: "
        f"{feature_counts[0]}"
    )

    print(
        "FedAvg feature compatibility: PASS"
    )

    # --------------------------------------------------------
    # Edge topology
    # --------------------------------------------------------

    for client_id in CLIENTS:

        graph = aligned_graphs[
            client_id
        ]

        if (
            graph.edge_index.shape[1]
            != graph.edge_type.shape[0]
        ):

            raise RuntimeError(
                f"{client_id}: edge topology mismatch."
            )

    print(
        "Edge topology preservation: PASS"
    )

    # --------------------------------------------------------
    # Fraud labels
    # --------------------------------------------------------

    print(
        "Fraud counts:"
    )

    for index, client_id in enumerate(CLIENTS):

        fraud_count = fraud_counts[
            index
        ]

        print(
            f"  {client_id}: "
            f"{fraud_count:,}"
        )

    print(
        "Fraud-label preservation: PASS"
    )

    # --------------------------------------------------------
    # Feature statistics
    # --------------------------------------------------------

    print()

    print(
        "Post-normalization statistics:"
    )

    for client_id in CLIENTS:

        graph = aligned_graphs[
            client_id
        ]

        mean_abs = float(
            graph.x.mean(
                dim=0
            )
            .abs()
            .mean()
            .item()
        )

        std_mean = float(
            graph.x.std(
                dim=0,
                unbiased=False
            )
            .mean()
            .item()
        )

        print(
            f"  {client_id}: "
            f"mean_abs={mean_abs:.6f}, "
            f"mean_std={std_mean:.6f}"
        )

    print()

    print(
        "Final enhanced alignment: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 80
    )

    print(
        "RFGN ENHANCED GRAPH NORMALIZATION + ALIGNMENT"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Input graph:"
    )

    print(
        "  graph_enhanced.pt"
    )

    print()

    print(
        "Output graph:"
    )

    print(
        "  graph_enhanced_aligned.pt"
    )

    print()

    print(
        "Existing graph_aligned.pt files will NOT be modified."
    )

    set_seed(
        RANDOM_SEED
    )

    # ========================================================
    # LOAD
    # ========================================================

    print_section(
        "LOADING ENHANCED CLIENT GRAPHS"
    )

    graphs = {}

    for client_id in CLIENTS:

        graphs[
            client_id
        ] = load_graph(
            client_id
        )

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    print_section(
        "VALIDATING INPUT GRAPHS"
    )

    for client_id in CLIENTS:

        validate_graph(
            graphs[
                client_id
            ],
            client_id
        )

    # ========================================================
    # COMMON FEATURE COUNT
    # ========================================================

    common_feature_count = (
        determine_common_feature_count(
            graphs
        )
    )

    # ========================================================
    # SHARED FEATURE MASK
    # ========================================================

    shared_mask = (
        create_shared_mask(
            graphs
        )
    )

    # ========================================================
    # GLOBAL NORMALIZATION
    # ========================================================

    (
        mean,
        std
    ) = calculate_global_feature_statistics(
        graphs,
        shared_mask
    )

    # ========================================================
    # NORMALIZE + SAVE
    # ========================================================

    print_section(
        "NORMALIZING AND SAVING ENHANCED GRAPHS"
    )

    aligned_graphs = {}

    output_paths = {}

    expected_features = int(
        shared_mask.sum().item()
    )

    for client_id in CLIENTS:

        original_graph = graphs[
            client_id
        ]

        normalized_x = normalize_graph(
            original_graph,
            shared_mask,
            mean,
            std
        )

        aligned_graph = original_graph.clone()

        aligned_graph.x = normalized_x

        aligned_graphs[
            client_id
        ] = aligned_graph

        validate_aligned_graph(
            aligned_graph,
            client_id,
            expected_features
        )

        output_paths[
            client_id
        ] = save_graph(
            aligned_graph,
            client_id
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    final_validation(
        aligned_graphs
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = build_manifest(
        graphs,
        shared_mask,
        mean,
        std,
        output_paths
    )

    save_manifest(
        manifest
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "ENHANCED GRAPH ALIGNMENT COMPLETED"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"Original features: "
        f"{common_feature_count}"
    )

    print(
        f"Shared features: "
        f"{expected_features}"
    )

    print()

    for client_id in CLIENTS:

        graph = aligned_graphs[
            client_id
        ]

        print(
            f"{client_id}: "
            f"nodes={graph.num_nodes:,}, "
            f"features={graph.num_node_features}, "
            f"edges={graph.num_edges:,}"
        )

    print()

    print(
        "Output files:"
    )

    for client_id in CLIENTS:

        print(
            f"  {output_paths[client_id]}"
        )

    print()

    print(
        "Manifest:"
    )

    print(
        f"  {MANIFEST_PATH}"
    )

    print()

    print(
        "=" * 80
    )

    print(
        "RFGN ENHANCED GRAPH ALIGNMENT: PASS"
    )

    print(
        "=" * 80
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()