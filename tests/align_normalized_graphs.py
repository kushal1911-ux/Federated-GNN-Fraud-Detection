# ============================================================
# RFGN SHARED FEATURE ALIGNMENT
# ============================================================
#
# Purpose:
#   Create feature-compatible graph copies for federated
#   GraphSAGE / FedAvg training.
#
# IMPORTANT:
#   Original graph.pt files are NOT modified.
#   Original graph_normalized.pt files are NOT modified.
#   CSV datasets are NOT modified.
#   Labels are NOT modified.
#   Edge indices are NOT modified.
#
# Client 1 manifest:
#   Contains removed_feature_indices only.
#   Kept indices are reconstructed as:
#
#       all original indices - removed indices
#
# Client 2 / Client 3:
#   Contains normalization.kept_feature_indices.
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
# CLIENTS
# ============================================================

CLIENTS = [
    "client_1",
    "client_2",
    "client_3",
]


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
        "graph_normalized.pt",
    )


def manifest_path(client_id):

    return os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "graphs",
        client_id,
        "normalization_manifest.json",
    )


def output_path(client_id):

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
# LOAD MANIFEST
# ============================================================

def load_manifest(client_id):

    path = manifest_path(
        client_id
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nManifest not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# EXTRACT KEPT INDICES
# ============================================================

def extract_kept_indices(
    manifest,
    client_id,
):
    """
    Extract the original feature indices that remain
    after normalization.

    Client 1:
        kept_feature_indices does not exist.
        Reconstruct from removed_feature_indices.

    Client 2 / Client 3:
        Use normalization.kept_feature_indices.
    """

    # --------------------------------------------------------
    # CLIENT 1
    # --------------------------------------------------------

    if client_id == "client_1":

        if (
            "removed_feature_indices"
            not in manifest
        ):

            raise KeyError(
                "\nClient 1 manifest does not contain "
                "'removed_feature_indices'."
            )

        removed = manifest[
            "removed_feature_indices"
        ]

        if not isinstance(
            removed,
            list,
        ):

            raise TypeError(
                "Client 1 removed_feature_indices "
                "must be a list."
            )

        removed = {
            int(index)
            for index in removed
        }

        original_count = int(
            manifest.get(
                "original_feature_count",
                814,
            )
        )

        all_indices = set(
            range(
                original_count
            )
        )

        kept = [
            index
            for index in range(
                original_count
            )
            if index not in removed
        ]

        if len(kept) != (
            original_count
            - len(removed)
        ):

            raise RuntimeError(
                "Client 1 feature reconstruction failed."
            )

        return kept, (
            "derived_from_removed_feature_indices"
        )


    # --------------------------------------------------------
    # CLIENT 2 / CLIENT 3
    # --------------------------------------------------------

    normalization = manifest.get(
        "normalization"
    )

    if not isinstance(
        normalization,
        dict,
    ):

        raise KeyError(
            f"\nClient {client_id} manifest does not "
            "contain a valid 'normalization' dictionary."
        )


    if (
        "kept_feature_indices"
        not in normalization
    ):

        raise KeyError(
            f"\nClient {client_id} normalization "
            "does not contain 'kept_feature_indices'."
        )


    kept = normalization[
        "kept_feature_indices"
    ]


    if not isinstance(
        kept,
        list,
    ):

        raise TypeError(
            f"Client {client_id} "
            "kept_feature_indices must be a list."
        )


    kept = [
        int(index)
        for index in kept
    ]


    return kept, (
        "normalization.kept_feature_indices"
    )


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph(client_id):

    path = graph_path(
        client_id
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nNormalized graph not found:\n{path}"
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


# ============================================================
# VALIDATE BASIC GRAPH
# ============================================================

def validate_graph(
    graph,
    client_id,
):

    if not hasattr(
        graph,
        "x",
    ):

        raise RuntimeError(
            f"{client_id}: missing x."
        )

    if not hasattr(
        graph,
        "edge_index",
    ):

        raise RuntimeError(
            f"{client_id}: missing edge_index."
        )

    if not hasattr(
        graph,
        "y",
    ):

        raise RuntimeError(
            f"{client_id}: missing y."
        )


    if torch.isnan(
        graph.x
    ).any():

        raise RuntimeError(
            f"{client_id}: NaN detected."
        )


    if torch.isinf(
        graph.x
    ).any():

        raise RuntimeError(
            f"{client_id}: infinity detected."
        )


    if graph.edge_index.dim() != 2:

        raise RuntimeError(
            f"{client_id}: invalid edge_index."
        )


    if graph.edge_index.shape[0] != 2:

        raise RuntimeError(
            f"{client_id}: edge_index must be (2,E)."
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
                f"{client_id}: edge index out of bounds."
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "RFGN SHARED FEATURE ALIGNMENT"
    )
    print("=" * 70)

    print()
    print(
        "Purpose:"
    )

    print(
        "Create feature-compatible graph copies "
        "for federated GraphSAGE training."
    )

    print()
    print(
        "Original graph.pt files will NOT be modified."
    )

    print(
        "Original graph_normalized.pt files will NOT be modified."
    )

    print(
        "CSV datasets will NOT be modified."
    )

    print(
        "Labels will NOT be modified."
    )

    print(
        "Edges will NOT be modified."
    )


    # ========================================================
    # LOAD MANIFESTS
    # ========================================================

    print_section(
        "LOADING NORMALIZATION MANIFESTS"
    )


    manifests = {}

    kept_indices = {}

    manifest_sources = {}


    for client_id in CLIENTS:

        print()

        print(
            client_id.upper()
        )

        path = manifest_path(
            client_id
        )

        print(
            f"Manifest: {path}"
        )


        manifest = load_manifest(
            client_id
        )


        indices, source = (
            extract_kept_indices(
                manifest,
                client_id,
            )
        )


        manifests[
            client_id
        ] = manifest


        kept_indices[
            client_id
        ] = indices


        manifest_sources[
            client_id
        ] = source


        print(
            f"Feature index source: "
            f"{source}"
        )


        print(
            f"Kept features: "
            f"{len(indices)}"
        )


        print(
            "Manifest: PASS"
        )


    # ========================================================
    # FEATURE INDEX VALIDATION
    # ========================================================

    print_section(
        "VALIDATING FEATURE INDICES"
    )


    for client_id in CLIENTS:

        indices = kept_indices[
            client_id
        ]


        if len(indices) != len(
            set(indices)
        ):

            raise RuntimeError(
                f"{client_id}: duplicate feature indices."
            )


        if min(indices) < 0:

            raise RuntimeError(
                f"{client_id}: negative feature index."
            )


        print(
            f"{client_id}: PASS"
        )


    # ========================================================
    # LOAD NORMALIZED GRAPHS
    # ========================================================

    print_section(
        "LOADING NORMALIZED GRAPHS"
    )


    graphs = {}


    for client_id in CLIENTS:

        print()

        print(
            f"Loading {client_id}..."
        )


        graph = load_graph(
            client_id
        )


        validate_graph(
            graph,
            client_id,
        )


        graphs[
            client_id
        ] = graph


        print(
            f"Nodes    : "
            f"{graph.num_nodes:,}"
        )


        print(
            f"Features : "
            f"{graph.num_node_features:,}"
        )


        print(
            f"Edges    : "
            f"{graph.num_edges:,}"
        )


        print(
            "Graph loading: PASS"
        )


    # ========================================================
    # MANIFEST / GRAPH CONSISTENCY
    # ========================================================

    print_section(
        "VALIDATING MANIFEST / GRAPH CONSISTENCY"
    )


    for client_id in CLIENTS:

        graph = graphs[
            client_id
        ]


        manifest_count = len(
            kept_indices[
                client_id
            ]
        )


        if graph.num_node_features != (
            manifest_count
        ):

            raise RuntimeError(
                f"\n{client_id}: "
                "manifest/graph mismatch.\n"
                f"Graph features: {graph.num_node_features}\n"
                f"Manifest kept: {manifest_count}"
            )


        print(
            f"{client_id}: PASS"
        )


    # ========================================================
    # FIND COMMON FEATURES
    # ========================================================

    print_section(
        "CALCULATING COMMON FEATURE SET"
    )


    common_features = set(
        kept_indices[
            CLIENTS[0]
        ]
    )


    for client_id in CLIENTS[1:]:

        common_features &= set(
            kept_indices[
                client_id
            ]
        )


    print()

    print(
        f"Common features: "
        f"{len(common_features)}"
    )


    if len(common_features) == 0:

        raise RuntimeError(
            "No common feature set exists."
        )


    # ========================================================
    # CANONICAL ORDER
    # ========================================================

    print_section(
        "CREATING CANONICAL FEATURE ORDER"
    )


    canonical_features = [
        index
        for index in kept_indices[
            "client_1"
        ]
        if index in common_features
    ]


    shared_dimension = len(
        canonical_features
    )


    print()

    print(
        "Canonical ordering source: CLIENT_1"
    )


    print(
        f"Shared feature count: "
        f"{shared_dimension}"
    )


    if shared_dimension != (
        len(common_features)
    ):

        raise RuntimeError(
            "Canonical feature list incomplete."
        )


    print(
        "Canonical feature schema: PASS"
    )


    # ========================================================
    # FEATURE DIFFERENCE
    # ========================================================

    print_section(
        "FEATURE DIFFERENCE ANALYSIS"
    )


    for client_id in CLIENTS:

        client_set = set(
            kept_indices[
                client_id
            ]
        )


        client_only = (
            client_set
            - common_features
        )


        print()

        print(
            client_id.upper()
        )


        print(
            f"Normalized features : "
            f"{len(client_set)}"
        )


        print(
            f"Shared features     : "
            f"{shared_dimension}"
        )


        print(
            f"Alignment removals  : "
            f"{len(client_only)}"
        )


    # ========================================================
    # POSITION MAPS
    # ========================================================

    print_section(
        "BUILDING FEATURE POSITION MAPS"
    )


    position_maps = {}


    for client_id in CLIENTS:

        position_maps[
            client_id
        ] = {
            feature_index: position
            for position, feature_index
            in enumerate(
                kept_indices[
                    client_id
                ]
            )
        }


        print(
            f"{client_id}: "
            f"{len(position_maps[client_id])} positions"
        )


    # ========================================================
    # CREATE ALIGNED GRAPHS
    # ========================================================

    print_section(
        "CREATING ALIGNED GRAPH COPIES"
    )


    aligned_graphs = {}


    for client_id in CLIENTS:

        print()

        print(
            client_id.upper()
        )


        graph = graphs[
            client_id
        ]


        positions = [
            position_maps[
                client_id
            ][feature_index]
            for feature_index
            in canonical_features
        ]


        if len(positions) != (
            shared_dimension
        ):

            raise RuntimeError(
                f"{client_id}: "
                "feature position mapping failed."
            )


        position_tensor = torch.tensor(
            positions,
            dtype=torch.long,
        )


        aligned_x = graph.x[
            :,
            position_tensor,
        ].contiguous()


        aligned_graph = graph.clone()


        aligned_graph.x = aligned_x


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        aligned_graph.feature_alignment = True

        aligned_graph.shared_feature_count = (
            shared_dimension
        )

        aligned_graph.alignment_method = (
            "common_feature_intersection"
        )

        aligned_graph.alignment_reference = (
            "client_1_canonical_order"
        )

        aligned_graph.original_feature_count = (
            graph.num_node_features
        )


        aligned_graphs[
            client_id
        ] = aligned_graph


        print(
            f"Original features : "
            f"{graph.num_node_features}"
        )


        print(
            f"Aligned features  : "
            f"{aligned_graph.num_node_features}"
        )


        print(
            "Feature alignment: PASS"
        )


    # ========================================================
    # VALIDATE ALIGNED GRAPHS
    # ========================================================

    print_section(
        "VALIDATING ALIGNED GRAPHS"
    )


    for client_id in CLIENTS:

        original = graphs[
            client_id
        ]


        aligned = aligned_graphs[
            client_id
        ]


        # ----------------------------------------------------
        # Feature count
        # ----------------------------------------------------

        if aligned.num_node_features != (
            shared_dimension
        ):

            raise RuntimeError(
                f"{client_id}: "
                "feature count mismatch."
            )


        print(
            f"{client_id} feature dimension: PASS"
        )


        # ----------------------------------------------------
        # NaN
        # ----------------------------------------------------

        if torch.isnan(
            aligned.x
        ).any():

            raise RuntimeError(
                f"{client_id}: NaN detected."
            )


        print(
            f"{client_id} NaN check: PASS"
        )


        # ----------------------------------------------------
        # Infinity
        # ----------------------------------------------------

        if torch.isinf(
            aligned.x
        ).any():

            raise RuntimeError(
                f"{client_id}: infinity detected."
            )


        print(
            f"{client_id} infinity check: PASS"
        )


        # ----------------------------------------------------
        # Node count
        # ----------------------------------------------------

        if aligned.num_nodes != (
            original.num_nodes
        ):

            raise RuntimeError(
                f"{client_id}: node count changed."
            )


        print(
            f"{client_id} node count: PASS"
        )


        # ----------------------------------------------------
        # Edge preservation
        # ----------------------------------------------------

        if aligned.num_edges != (
            original.num_edges
        ):

            raise RuntimeError(
                f"{client_id}: edge count changed."
            )


        if not torch.equal(
            aligned.edge_index,
            original.edge_index,
        ):

            raise RuntimeError(
                f"{client_id}: edge index changed."
            )


        print(
            f"{client_id} edge preservation: PASS"
        )


        # ----------------------------------------------------
        # Label preservation
        # ----------------------------------------------------

        if not torch.equal(
            aligned.y,
            original.y,
        ):

            raise RuntimeError(
                f"{client_id}: labels changed."
            )


        print(
            f"{client_id} label preservation: PASS"
        )


    # ========================================================
    # CROSS CLIENT COMPATIBILITY
    # ========================================================

    print_section(
        "CROSS-CLIENT FEDERATED COMPATIBILITY"
    )


    dimensions = [
        aligned_graphs[
            client_id
        ].num_node_features
        for client_id in CLIENTS
    ]


    for client_id in CLIENTS:

        print(
            f"{client_id}: "
            f"{aligned_graphs[client_id].num_node_features}"
            " features"
        )


    if len(
        set(dimensions)
    ) != 1:

        raise RuntimeError(
            "Feature dimensions are not identical."
        )


    print()

    print(
        "Identical feature dimensions: PASS"
    )


    # ========================================================
    # SAVE ALIGNED GRAPHS
    # ========================================================

    print_section(
        "SAVING ALIGNED GRAPH COPIES"
    )


    for client_id in CLIENTS:

        path = output_path(
            client_id
        )


        torch.save(
            aligned_graphs[
                client_id
            ],
            path,
        )


        if not os.path.exists(
            path
        ):

            raise RuntimeError(
                f"{client_id}: save failed."
            )


        size_mb = (
            os.path.getsize(path)
            / (1024 ** 2)
        )


        print()

        print(
            client_id.upper()
        )


        print(
            f"Saved to: {path}"
        )


        print(
            f"File size: {size_mb:.2f} MB"
        )


        print(
            "Graph save: PASS"
        )


    # ========================================================
    # RELOAD
    # ========================================================

    print_section(
        "RELOADING ALIGNED GRAPHS"
    )


    reloaded = {}


    for client_id in CLIENTS:

        path = output_path(
            client_id
        )


        graph = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )


        reloaded[
            client_id
        ] = graph


        original = graphs[
            client_id
        ]


        if graph.num_node_features != (
            shared_dimension
        ):

            raise RuntimeError(
                f"{client_id}: "
                "reloaded feature dimension mismatch."
            )


        if graph.num_nodes != (
            original.num_nodes
        ):

            raise RuntimeError(
                f"{client_id}: "
                "reloaded node count mismatch."
            )


        if graph.num_edges != (
            original.num_edges
        ):

            raise RuntimeError(
                f"{client_id}: "
                "reloaded edge count mismatch."
            )


        if not torch.equal(
            graph.edge_index,
            original.edge_index,
        ):

            raise RuntimeError(
                f"{client_id}: "
                "reloaded edges changed."
            )


        if not torch.equal(
            graph.y,
            original.y,
        ):

            raise RuntimeError(
                f"{client_id}: "
                "reloaded labels changed."
            )


        if torch.isnan(
            graph.x
        ).any():

            raise RuntimeError(
                f"{client_id}: "
                "reloaded graph contains NaN."
            )


        if torch.isinf(
            graph.x
        ).any():

            raise RuntimeError(
                f"{client_id}: "
                "reloaded graph contains infinity."
            )


        print(
            f"{client_id}: "
            "Reload validation: PASS"
        )


    # ========================================================
    # FINAL COMPATIBILITY
    # ========================================================

    print_section(
        "FINAL FEDERATED COMPATIBILITY CHECK"
    )


    final_dimensions = [
        reloaded[
            client_id
        ].num_node_features
        for client_id in CLIENTS
    ]


    print()


    for client_id in CLIENTS:

        print(
            f"{client_id}: "
            f"{reloaded[client_id].num_node_features}"
            " features"
        )


    if len(
        set(final_dimensions)
    ) != 1:

        raise RuntimeError(
            "FINAL CHECK FAILED."
        )


    print()

    print(
        "All clients have identical "
        "feature dimensions: PASS"
    )


    # ========================================================
    # SAVE ALIGNMENT MANIFEST
    # ========================================================

    print_section(
        "CREATING SHARED FEATURE ALIGNMENT MANIFEST"
    )


    alignment_manifest = {

        "project":
            "RFGN",

        "description":
            "Shared feature schema for federated GraphSAGE",

        "clients":
            CLIENTS,

        "method":
            "Intersection of normalized retained features",

        "canonical_order_source":
            "client_1",

        "shared_feature_count":
            shared_dimension,

        "shared_feature_indices":
            canonical_features,

        "feature_index_sources":
            manifest_sources,

        "normalized_feature_counts":
            {
                client_id:
                    len(
                        kept_indices[
                            client_id
                        ]
                    )
                for client_id in CLIENTS
            },

        "aligned_feature_counts":
            {
                client_id:
                    reloaded[
                        client_id
                    ].num_node_features
                for client_id in CLIENTS
            },

        "original_graphs_modified":
            False,

        "normalized_graphs_modified":
            False,

        "csv_datasets_modified":
            False,

        "labels_modified":
            False,

        "edges_modified":
            False,

        "fedavg_parameter_shape_compatible":
            True,
    }


    output_manifest = (
        alignment_manifest_path()
    )


    with open(
        output_manifest,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            alignment_manifest,
            file,
            indent=4,
        )


    print()

    print(
        f"Manifest saved to:"
    )

    print(
        output_manifest
    )

    print(
        "Manifest save: PASS"
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print("=" * 70)

    print(
        "RFGN SHARED FEATURE ALIGNMENT SUMMARY"
    )

    print("=" * 70)

    print()

    print(
        f"Common feature count: "
        f"{shared_dimension}"
    )

    print()

    for client_id in CLIENTS:

        print(
            f"{client_id.upper():<15}"
            f"{reloaded[client_id].num_node_features}"
            " features"
        )


    print()

    print(
        "Original graph.pt modified       : NO"
    )

    print(
        "Normalized graph modified        : NO"
    )

    print(
        "CSV datasets modified            : NO"
    )

    print(
        "Labels modified                  : NO"
    )

    print(
        "Edges modified                   : NO"
    )

    print(
        "FedAvg feature compatibility     : YES"
    )

    print()

    print("=" * 70)

    print(
        "RFGN SHARED FEATURE ALIGNMENT "
        "COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)

    print()

    print(
        "Next:"
    )

    print(
        "Validate the three aligned graphs."
    )

    print(
        "Then update the local GraphSAGE "
        "training pipeline to use the shared "
        "feature dimension."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()