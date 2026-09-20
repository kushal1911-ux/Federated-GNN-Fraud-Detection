import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# ============================================================
# RFGN ENHANCED GRAPH BUILDER
# LEVEL 11
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

CLIENT_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "clients"
)

GRAPH_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs"
)

os.makedirs(
    GRAPH_ROOT,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
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


# Prevent extremely large relationship cliques.
MAX_NODES_PER_VALUE = 50


# Maximum relationship edges.
MAX_UNDIRECTED_RELATIONSHIP_EDGES = 1_500_000


# Enhanced temporal configuration.
TEMPORAL_WINDOW_SECONDS = 60 * 60

TEMPORAL_NEIGHBORS = 5

MAX_UNDIRECTED_TEMPORAL_EDGES = 750_000


# Feature similarity configuration.
#
# Similarity is NOT calculated between every pair of
# transactions. Instead, only nearby chronological candidates
# are compared.
SIMILARITY_CANDIDATES = 10

SIMILARITY_THRESHOLD = 0.90

MAX_UNDIRECTED_SIMILARITY_EDGES = 750_000

SIMILARITY_FEATURE_LIMIT = 64


# ============================================================
# VALUE NORMALIZATION
# ============================================================

def normalize_value(value):

    if pd.isna(value):

        return None

    value = str(value).strip()

    if value == "":

        return None

    return value


# ============================================================
# LOAD CLIENT
# ============================================================

def load_client(client_id):

    path = os.path.join(
        CLIENT_ROOT,
        client_id,
        "train.csv"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Client dataset not found:\n{path}"
        )

    print()
    print("Loading:", path)

    df = pd.read_csv(
        path,
        low_memory=False
    )

    print(
        "Rows   :",
        f"{len(df):,}"
    )

    print(
        "Columns:",
        len(df.columns)
    )

    return df


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(df):

    print()
    print("=" * 70)
    print("VALIDATING GRAPH INPUT")
    print("=" * 70)

    required = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
    ]

    required.extend(
        RELATIONSHIP_COLUMNS
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(
        "Required columns: PASS"
    )

    if df["TransactionID"].isna().any():

        raise ValueError(
            "Missing TransactionID values."
        )

    if df["TransactionID"].duplicated().any():

        raise ValueError(
            "Duplicate TransactionID values."
        )

    print(
        "TransactionID uniqueness: PASS"
    )

    labels = set(
        df["isFraud"]
        .dropna()
        .unique()
        .tolist()
    )

    if not labels.issubset({0, 1}):

        raise ValueError(
            f"Invalid fraud labels: {labels}"
        )

    print(
        "Fraud labels: PASS"
    )

    if df["TransactionDT"].isna().any():

        raise ValueError(
            "Missing TransactionDT values."
        )

    transaction_dt = pd.to_numeric(
        df["TransactionDT"],
        errors="coerce"
    )

    if transaction_dt.isna().any():

        raise ValueError(
            "TransactionDT contains non-numeric values."
        )

    if not np.isfinite(
        transaction_dt.to_numpy()
    ).all():

        raise ValueError(
            "TransactionDT contains NaN or infinity."
        )

    print(
        "TransactionDT completeness: PASS"
    )

    print(
        "Fraud labels used for graph construction: NO"
    )


# ============================================================
# NODE FEATURES
# ============================================================

def select_features(df):

    print()
    print("=" * 70)
    print("PREPARING NODE FEATURES")
    print("=" * 70)

    excluded = {
        "TransactionID",
        "isFraud",
        "TransactionDT",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    print(
        "Original columns :",
        len(df.columns)
    )

    print(
        "Feature columns  :",
        len(feature_columns)
    )

    print(
        "Numeric features :",
        len(numeric_columns)
    )

    features = (
        df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
        .astype(np.float32)
    )

    x = torch.tensor(
        features.to_numpy(),
        dtype=torch.float32
    )

    print(
        "Node feature tensor:",
        tuple(x.shape)
    )

    print(
        "Feature preparation: PASS"
    )

    return (
        x,
        numeric_columns
    )


# ============================================================
# SELECT SIMILARITY FEATURES
# ============================================================

def select_similarity_features(
    df,
    numeric_columns
):

    print()
    print("=" * 70)
    print("SELECTING SIMILARITY FEATURES")
    print("=" * 70)

    if not numeric_columns:

        raise ValueError(
            "No numeric features available for similarity."
        )

    limit = min(
        SIMILARITY_FEATURE_LIMIT,
        len(numeric_columns)
    )

    numeric_frame = (
        df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
        .astype(np.float32)
    )

    variances = (
        numeric_frame
        .var(
            axis=0,
            ddof=0
        )
    )

    selected_columns = (
        variances
        .sort_values(
            ascending=False
        )
        .head(limit)
        .index
        .tolist()
    )

    similarity_features = (
        numeric_frame[
            selected_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    print(
        "Available numeric features:",
        len(numeric_columns)
    )

    print(
        "Similarity features selected:",
        len(selected_columns)
    )

    print(
        "Similarity feature selection: PASS"
    )

    return (
        similarity_features,
        selected_columns
    )


# ============================================================
# NORMALIZE SIMILARITY FEATURES
# ============================================================

def normalize_similarity_features(
    features
):

    tensor = torch.from_numpy(
        features
    )

    norms = torch.linalg.norm(
        tensor,
        dim=1,
        keepdim=True
    )

    norms = torch.clamp(
        norms,
        min=1e-12
    )

    normalized = (
        tensor / norms
    )

    normalized = torch.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return normalized


# ============================================================
# RELATIONSHIP INDEX
# ============================================================

def build_relationship_index(df):

    print()
    print("=" * 70)
    print("BUILDING RELATIONSHIP INDEX")
    print("=" * 70)

    relationship_maps = {}

    for column in RELATIONSHIP_COLUMNS:

        value_map = defaultdict(list)

        values = df[column].tolist()

        for node_index, value in enumerate(values):

            normalized = normalize_value(
                value
            )

            if normalized is None:

                continue

            value_map[
                normalized
            ].append(
                node_index
            )

        filtered_map = {}

        skipped = 0

        for value, nodes in value_map.items():

            if (
                len(nodes)
                <= MAX_NODES_PER_VALUE
            ):

                filtered_map[
                    value
                ] = nodes

            else:

                skipped += 1

        relationship_maps[column] = (
            filtered_map
        )

        print(
            f"{column:<18}"
            f"values={len(value_map):>8,} "
            f"usable={len(filtered_map):>8,} "
            f"skipped={skipped:>7,}"
        )

    print(
        "Relationship index: PASS"
    )

    return relationship_maps


# ============================================================
# RELATIONSHIP EDGES
# ============================================================

def build_relationship_edges(
    relationship_maps
):

    print()
    print("=" * 70)
    print("BUILDING RELATIONSHIP EDGES")
    print("=" * 70)

    edge_set = set()

    for column, value_map in (
        relationship_maps.items()
    ):

        before = len(edge_set)

        for nodes in value_map.values():

            count = len(nodes)

            if count < 2:

                continue

            for i in range(count):

                source = nodes[i]

                for j in range(
                    i + 1,
                    count
                ):

                    target = nodes[j]

                    if source == target:

                        continue

                    if source < target:

                        edge = (
                            source,
                            target
                        )

                    else:

                        edge = (
                            target,
                            source
                        )

                    edge_set.add(
                        edge
                    )

                    if (
                        len(edge_set)
                        >= MAX_UNDIRECTED_RELATIONSHIP_EDGES
                    ):

                        break

                if (
                    len(edge_set)
                    >= MAX_UNDIRECTED_RELATIONSHIP_EDGES
                ):

                    break

            if (
                len(edge_set)
                >= MAX_UNDIRECTED_RELATIONSHIP_EDGES
            ):

                break

        added = (
            len(edge_set)
            - before
        )

        print(
            f"{column:<18}"
            f"new edges={added:>12,}"
        )

        if (
            len(edge_set)
            >= MAX_UNDIRECTED_RELATIONSHIP_EDGES
        ):

            print(
                "Relationship edge safety limit reached."
            )

            break

    print()

    print(
        "Relationship edges:",
        f"{len(edge_set):,}"
    )

    return edge_set


# ============================================================
# TEMPORAL EDGES
# ============================================================

def build_temporal_edges(
    df
):

    print()
    print("=" * 70)
    print("BUILDING ENHANCED TEMPORAL EDGES")
    print("=" * 70)

    transaction_times = pd.to_numeric(
        df["TransactionDT"],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    sorted_indices = np.argsort(
        transaction_times,
        kind="mergesort"
    )

    edge_set = set()

    n = len(
        sorted_indices
    )

    for position in range(n):

        source = int(
            sorted_indices[position]
        )

        source_time = (
            transaction_times[source]
        )

        end = min(
            position
            + TEMPORAL_NEIGHBORS
            + 1,
            n
        )

        for next_position in range(
            position + 1,
            end
        ):

            target = int(
                sorted_indices[
                    next_position
                ]
            )

            target_time = (
                transaction_times[target]
            )

            time_difference = (
                target_time
                - source_time
            )

            if (
                time_difference
                > TEMPORAL_WINDOW_SECONDS
            ):

                break

            if source == target:

                continue

            if source < target:

                edge = (
                    source,
                    target
                )

            else:

                edge = (
                    target,
                    source
                )

            edge_set.add(
                edge
            )

            if (
                len(edge_set)
                >= MAX_UNDIRECTED_TEMPORAL_EDGES
            ):

                break

        if (
            len(edge_set)
            >= MAX_UNDIRECTED_TEMPORAL_EDGES
        ):

            break

    print(
        "Temporal window:",
        f"{TEMPORAL_WINDOW_SECONDS:,} seconds"
    )

    print(
        "Temporal neighbors:",
        TEMPORAL_NEIGHBORS
    )

    print(
        "Temporal edges:",
        f"{len(edge_set):,}"
    )

    return edge_set


# ============================================================
# FEATURE SIMILARITY EDGES
# ============================================================

def build_similarity_edges(
    df,
    similarity_features
):

    print()
    print("=" * 70)
    print("BUILDING FEATURE-SIMILARITY EDGES")
    print("=" * 70)

    n = len(df)

    if n == 0:

        return set()

    normalized_features = (
        normalize_similarity_features(
            similarity_features
        )
    )

    transaction_times = pd.to_numeric(
        df["TransactionDT"],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    sorted_indices = np.argsort(
        transaction_times,
        kind="mergesort"
    )

    edge_set = set()

    total_candidates = 0

    total_similar = 0

    print(
        "Similarity candidate window:",
        SIMILARITY_CANDIDATES
    )

    print(
        "Similarity threshold:",
        SIMILARITY_THRESHOLD
    )

    print(
        "Similarity features:",
        similarity_features.shape[1]
    )

    for position in range(n):

        source = int(
            sorted_indices[position]
        )

        source_time = (
            transaction_times[source]
        )

        candidate_start = max(
            0,
            position
            - SIMILARITY_CANDIDATES
        )

        candidate_positions = np.arange(
            candidate_start,
            position
        )

        if (
            candidate_positions.size
            == 0
        ):

            continue

        candidate_indices = (
            sorted_indices[
                candidate_positions
            ]
        )

        candidate_times = (
            transaction_times[
                candidate_indices
            ]
        )

        valid_mask = (
            (
                source_time
                - candidate_times
            )
            <= TEMPORAL_WINDOW_SECONDS
        )

        candidate_indices = (
            candidate_indices[
                valid_mask
            ]
        )

        if (
            candidate_indices.size
            == 0
        ):

            continue

        total_candidates += (
            candidate_indices.size
        )

        source_vector = (
            normalized_features[
                source
            ]
            .unsqueeze(0)
        )

        candidate_tensor_indices = (
            torch.from_numpy(
                candidate_indices
                .astype(np.int64)
            )
        )

        candidate_vectors = (
            normalized_features[
                candidate_tensor_indices
            ]
        )

        similarities = (
            torch.matmul(
                candidate_vectors,
                source_vector.t()
            )
            .squeeze(1)
        )

        matching_positions = (
            torch.where(
                similarities
                >= SIMILARITY_THRESHOLD
            )[0]
        )

        for match_position in (
            matching_positions.tolist()
        ):

            target = int(
                candidate_indices[
                    match_position
                ]
            )

            if source == target:

                continue

            if source < target:

                edge = (
                    source,
                    target
                )

            else:

                edge = (
                    target,
                    source
                )

            if edge not in edge_set:

                edge_set.add(
                    edge
                )

                total_similar += 1

            if (
                len(edge_set)
                >= MAX_UNDIRECTED_SIMILARITY_EDGES
            ):

                break

        if (
            len(edge_set)
            >= MAX_UNDIRECTED_SIMILARITY_EDGES
        ):

            print(
                "Similarity edge safety limit reached."
            )

            break

        if (
            position == 0
            or position % 10000 == 0
        ):

            print(
                "Processed:",
                f"{position:,}/{n:,}",
                "| similarity edges:",
                f"{len(edge_set):,}"
            )

    print()

    print(
        "Candidate comparisons:",
        f"{total_candidates:,}"
    )

    print(
        "Similarity matches:",
        f"{total_similar:,}"
    )

    print(
        "Similarity edges:",
        f"{len(edge_set):,}"
    )

    print(
        "Feature-similarity graph: PASS"
    )

    return edge_set


# ============================================================
# CREATE EDGE INDEX AND EDGE TYPES
# ============================================================

def create_edge_index_and_types(
    relationship_edges,
    temporal_edges,
    similarity_edges
):

    print()
    print("=" * 70)
    print("CREATING PYTORCH GEOMETRIC EDGE INDEX")
    print("=" * 70)

    combined = set(
        relationship_edges
    )

    combined.update(
        temporal_edges
    )

    combined.update(
        similarity_edges
    )

    if len(combined) == 0:

        raise ValueError(
            "No graph edges generated."
        )

    directed_edges = []

    edge_types = []

    for source, target in combined:

        if (
            (source, target)
            in relationship_edges
        ):

            edge_type = 0

        elif (
            (source, target)
            in temporal_edges
        ):

            edge_type = 1

        elif (
            (source, target)
            in similarity_edges
        ):

            edge_type = 2

        else:

            edge_type = 0

        directed_edges.append(
            [source, target]
        )

        edge_types.append(
            edge_type
        )

        directed_edges.append(
            [target, source]
        )

        edge_types.append(
            edge_type
        )

    edge_index = torch.tensor(
        directed_edges,
        dtype=torch.long
    ).t().contiguous()

    edge_type = torch.tensor(
        edge_types,
        dtype=torch.long
    )

    print(
        "Undirected edges:",
        f"{len(combined):,}"
    )

    print(
        "Directed PyG edges:",
        f"{edge_index.shape[1]:,}"
    )

    print(
        "Edge index shape:",
        tuple(edge_index.shape)
    )

    print(
        "Edge type shape:",
        tuple(edge_type.shape)
    )

    print(
        "Edge types:",
        "0=relationship, 1=temporal, 2=similarity"
    )

    print(
        "Edge index creation: PASS"
    )

    return (
        edge_index,
        edge_type
    )


# ============================================================
# CREATE GRAPH
# ============================================================

def create_graph(
    x,
    edge_index,
    edge_type,
    df
):

    print()
    print("=" * 70)
    print("CREATING GRAPH OBJECT")
    print("=" * 70)

    y = torch.tensor(
        df["isFraud"]
        .astype(np.int64)
        .to_numpy(),
        dtype=torch.long
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=y
    )

    print(
        "Nodes         :",
        f"{data.num_nodes:,}"
    )

    print(
        "Node features :",
        data.num_node_features
    )

    print(
        "Edges         :",
        f"{data.num_edges:,}"
    )

    print(
        "Labels        :",
        data.y.shape[0]
    )

    print(
        "Edge types    :",
        data.edge_type.shape[0]
    )

    return data


# ============================================================
# VALIDATE GRAPH
# ============================================================

def validate_graph(
    data,
    df
):

    print()
    print("=" * 70)
    print("VALIDATING ENHANCED GRAPH")
    print("=" * 70)

    assert (
        data.num_nodes
        == len(df)
    )

    print(
        "Node count: PASS"
    )

    assert (
        data.y.shape[0]
        == data.num_nodes
    )

    print(
        "Label count: PASS"
    )

    assert (
        data.edge_index.shape[0]
        == 2
    )

    print(
        "Edge index shape: PASS"
    )

    assert (
        data.edge_type.shape[0]
        == data.num_edges
    )

    print(
        "Edge type alignment: PASS"
    )

    assert (
        int(data.edge_index.min())
        >= 0
    )

    assert (
        int(data.edge_index.max())
        < data.num_nodes
    )

    print(
        "Edge bounds: PASS"
    )

    assert not torch.isnan(
        data.x
    ).any()

    print(
        "Feature NaN check: PASS"
    )

    assert not torch.isinf(
        data.x
    ).any()

    print(
        "Feature infinity check: PASS"
    )

    assert not torch.isnan(
        data.edge_type.float()
    ).any()

    print(
        "Edge type NaN check: PASS"
    )

    original_fraud = int(
        df["isFraud"].sum()
    )

    graph_fraud = int(
        data.y.sum().item()
    )

    assert (
        original_fraud
        == graph_fraud
    )

    print(
        "Fraud label preservation: PASS"
    )

    print()
    print(
        "Enhanced graph validation: PASS"
    )


# ============================================================
# GRAPH STATISTICS
# ============================================================

def graph_statistics(
    data
):

    print()
    print("=" * 70)
    print("ENHANCED GRAPH STATISTICS")
    print("=" * 70)

    nodes = data.num_nodes

    directed_edges = data.num_edges

    undirected_edges = (
        directed_edges // 2
    )

    possible_edges = (
        nodes
        * (nodes - 1)
        // 2
    )

    density = (
        undirected_edges
        / possible_edges
        if possible_edges > 0
        else 0
    )

    degree = torch.bincount(
        data.edge_index[0],
        minlength=nodes
    )

    isolated = int(
        (degree == 0)
        .sum()
        .item()
    )

    connected = (
        nodes - isolated
    )

    avg_degree = float(
        degree.float()
        .mean()
        .item()
    )

    max_degree = int(
        degree.max()
        .item()
    )

    print(
        "Nodes:",
        f"{nodes:,}"
    )

    print(
        "Undirected edges:",
        f"{undirected_edges:,}"
    )

    print(
        "Directed edges:",
        f"{directed_edges:,}"
    )

    print(
        "Density:",
        f"{density:.10f}"
    )

    print(
        "Average degree:",
        f"{avg_degree:.4f}"
    )

    print(
        "Maximum degree:",
        max_degree
    )

    print(
        "Isolated nodes:",
        f"{isolated:,}"
    )

    print(
        "Connected nodes:",
        f"{connected:,}"
    )

    print(
        "Connected percentage:",
        f"{connected / nodes * 100:.4f}%"
    )

    relationship_count = int(
        (
            data.edge_type
            == 0
        ).sum().item()
        // 2
    )

    temporal_count = int(
        (
            data.edge_type
            == 1
        ).sum().item()
        // 2
    )

    similarity_count = int(
        (
            data.edge_type
            == 2
        ).sum().item()
        // 2
    )

    print()
    print(
        "Relationship edges:",
        f"{relationship_count:,}"
    )

    print(
        "Temporal edges:",
        f"{temporal_count:,}"
    )

    print(
        "Similarity edges:",
        f"{similarity_count:,}"
    )


# ============================================================
# SAVE
# ============================================================

def save_graph(
    data,
    client_id
):

    print()
    print("=" * 70)
    print("SAVING ENHANCED GRAPH")
    print("=" * 70)

    output_dir = os.path.join(
        GRAPH_ROOT,
        client_id
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        output_dir,
        "graph_enhanced.pt"
    )

    torch.save(
        data,
        output_path
    )

    size_mb = (
        os.path.getsize(
            output_path
        )
        / (1024 ** 2)
    )

    print(
        "Saved to:",
        output_path
    )

    print(
        f"File size: {size_mb:.2f} MB"
    )

    print(
        "Enhanced graph save: PASS"
    )

    return output_path


# ============================================================
# MAIN GRAPH CONSTRUCTION
# ============================================================

def build_client_graph(
    client_id
):

    print()
    print("=" * 80)
    print(
        f"RFGN ENHANCED GRAPH CONSTRUCTION — "
        f"{client_id.upper()}"
    )
    print("=" * 80)

    print()
    print(
        "Full client dataset used: YES"
    )

    print(
        "Fraud labels used for edges: NO"
    )

    print(
        "TransactionID used as feature: NO"
    )

    print(
        "TransactionDT used as feature: NO"
    )

    print(
        "Relationship edges: YES"
    )

    print(
        "Enhanced temporal edges: YES"
    )

    print(
        "Feature-similarity edges: YES"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_client(
        client_id
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_input(
        df
    )

    # --------------------------------------------------------
    # Node features
    # --------------------------------------------------------

    (
        x,
        numeric_columns
    ) = select_features(
        df
    )

    # --------------------------------------------------------
    # Similarity features
    # --------------------------------------------------------

    (
        similarity_features,
        similarity_columns
    ) = select_similarity_features(
        df,
        numeric_columns
    )

    # --------------------------------------------------------
    # Relationship graph
    # --------------------------------------------------------

    relationship_maps = (
        build_relationship_index(
            df
        )
    )

    relationship_edges = (
        build_relationship_edges(
            relationship_maps
        )
    )

    # --------------------------------------------------------
    # Enhanced temporal graph
    # --------------------------------------------------------

    temporal_edges = (
        build_temporal_edges(
            df
        )
    )

    # --------------------------------------------------------
    # Feature similarity graph
    # --------------------------------------------------------

    similarity_edges = (
        build_similarity_edges(
            df,
            similarity_features
        )
    )

    # --------------------------------------------------------
    # Create combined graph
    # --------------------------------------------------------

    (
        edge_index,
        edge_type
    ) = create_edge_index_and_types(
        relationship_edges,
        temporal_edges,
        similarity_edges
    )

    # --------------------------------------------------------
    # Create PyG Data object
    # --------------------------------------------------------

    data = create_graph(
        x,
        edge_index,
        edge_type,
        df
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_graph(
        data,
        df
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    graph_statistics(
        data
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = save_graph(
        data,
        client_id
    )

    print()
    print("=" * 80)
    print(
        f"{client_id.upper()} ENHANCED GRAPH CONSTRUCTION COMPLETED"
    )
    print("=" * 80)

    print()

    print(
        "Nodes:",
        f"{data.num_nodes:,}"
    )

    print(
        "Features:",
        data.num_node_features
    )

    print(
        "Directed edges:",
        f"{data.num_edges:,}"
    )

    print(
        "Fraud nodes:",
        f"{int(data.y.sum().item()):,}"
    )

    print(
        "Similarity features:",
        len(similarity_columns)
    )

    print(
        "Saved:",
        output_path
    )

    return data


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    build_client_graph(
        "client_1"
    )