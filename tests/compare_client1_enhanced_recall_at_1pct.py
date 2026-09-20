# ============================================================
# RFGN - CLIENT 1 ENHANCED RECALL@1% COMPARISON
# ============================================================
#
# Compares:
#
#   1. Original GraphSAGE
#   2. Enhanced GraphSAGE
#
# Using:
#   - Same Client 1 enhanced aligned graph
#   - Same 15% validation ratio
#   - Same seed = 42
#   - Same Recall@1% methodology
#
# IMPORTANT:
#   The validation indices are generated deterministically.
#
# ============================================================

import os
import sys
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

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# IMPORT MODELS
# ============================================================

from models.gnn.graphsage_model import (
    GraphSAGE
)

from models.gnn.enhanced_graphsage_model import (
    EnhancedGraphSAGE
)


# ============================================================
# PATHS
# ============================================================

GRAPH_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs",
    "client_1",
    "graph_enhanced_aligned.pt",
)

ORIGINAL_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "local",
    "client_1_graphsage_aligned_best.pt",
)

ENHANCED_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "local",
    "client_1_enhanced_graphsage_best.pt",
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "local",
    "client_1_enhanced_recall_at_1pct_comparison.json",
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

VALIDATION_RATIO = 0.15

TOP_PERCENT = 0.01

INPUT_DIM = 769

HIDDEN_DIM = 128

OUTPUT_DIM = 2

DROPOUT = 0.30


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph():

    if not os.path.exists(
        GRAPH_PATH
    ):

        raise FileNotFoundError(
            f"Graph not found:\n{GRAPH_PATH}"
        )

    print(
        "Loading Client 1 enhanced aligned graph..."
    )

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False
    )

    required = [
        "x",
        "edge_index",
        "y",
    ]

    for attribute in required:

        if not hasattr(
            graph,
            attribute
        ):

            raise ValueError(
                f"Missing graph attribute: "
                f"{attribute}"
            )

    if graph.x.shape[1] != INPUT_DIM:

        raise ValueError(
            f"Expected {INPUT_DIM} features, "
            f"received {graph.x.shape[1]}."
        )

    if graph.x.shape[0] != graph.y.shape[0]:

        raise ValueError(
            "Node count and label count do not match."
        )

    if not torch.isfinite(
        graph.x
    ).all():

        raise ValueError(
            "Graph contains NaN or Inf values."
        )

    return graph


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def create_stratified_split(
    labels,
    validation_ratio,
    seed
):

    labels_np = labels.cpu().numpy()

    rng = np.random.default_rng(
        seed
    )

    train_indices = []

    validation_indices = []

    unique_classes = np.unique(
        labels_np
    )

    for class_id in unique_classes:

        class_indices = np.where(
            labels_np == class_id
        )[0]

        rng.shuffle(
            class_indices
        )

        validation_count = int(
            len(class_indices)
            * validation_ratio
        )

        validation_count = max(
            1,
            validation_count
        )

        validation_indices.extend(
            class_indices[
                :validation_count
            ].tolist()
        )

        train_indices.extend(
            class_indices[
                validation_count:
            ].tolist()
        )

    rng.shuffle(
        train_indices
    )

    rng.shuffle(
        validation_indices
    )

    train_mask = torch.zeros(
        len(labels),
        dtype=torch.bool
    )

    validation_mask = torch.zeros(
        len(labels),
        dtype=torch.bool
    )

    train_mask[
        torch.tensor(
            train_indices,
            dtype=torch.long
        )
    ] = True

    validation_mask[
        torch.tensor(
            validation_indices,
            dtype=torch.long
        )
    ] = True

    return (
        train_mask,
        validation_mask
    )


# ============================================================
# LOAD ORIGINAL MODEL
# ============================================================

def load_original_model(
    device
):

    if not os.path.exists(
        ORIGINAL_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Original GraphSAGE checkpoint "
            f"not found:\n{ORIGINAL_MODEL_PATH}"
        )

    checkpoint = torch.load(
        ORIGINAL_MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    # --------------------------------------------------------
    # Support checkpoint containing model_state_dict
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Original model
    # --------------------------------------------------------

    model = GraphSAGE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    ).to(
        device
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# LOAD ENHANCED MODEL
# ============================================================

def load_enhanced_model(
    device
):

    if not os.path.exists(
        ENHANCED_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Enhanced GraphSAGE checkpoint "
            f"not found:\n{ENHANCED_MODEL_PATH}"
        )

    checkpoint = torch.load(
        ENHANCED_MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model = EnhancedGraphSAGE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    ).to(
        device
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# RECALL@1%
# ============================================================

def calculate_recall_at_1_percent(
    probabilities,
    labels
):

    probabilities = probabilities.detach().cpu()

    labels = labels.detach().cpu()

    total_samples = len(
        labels
    )

    top_k_count = max(
        1,
        int(
            total_samples
            * TOP_PERCENT
        )
    )

    # --------------------------------------------------------
    # Rank transactions by fraud probability
    # --------------------------------------------------------

    sorted_indices = torch.argsort(
        probabilities,
        descending=True
    )

    top_indices = sorted_indices[
        :top_k_count
    ]

    top_labels = labels[
        top_indices
    ]

    captured_fraud = int(
        (top_labels == 1).sum().item()
    )

    total_fraud = int(
        (labels == 1).sum().item()
    )

    missed_fraud = (
        total_fraud
        - captured_fraud
    )

    recall = (
        captured_fraud
        / total_fraud
        if total_fraud > 0
        else 0.0
    )

    precision = (
        captured_fraud
        / top_k_count
        if top_k_count > 0
        else 0.0
    )

    cutoff_probability = float(
        probabilities[
            top_indices[-1]
        ].item()
    )

    return {
        "total_samples":
            total_samples,

        "total_fraud":
            total_fraud,

        "top_1_percent_count":
            top_k_count,

        "fraud_captured":
            captured_fraud,

        "fraud_missed":
            missed_fraud,

        "recall_at_1_percent":
            float(recall),

        "recall_at_1_percent_percentage":
            float(
                recall * 100.0
            ),

        "precision_at_top_1_percent":
            float(precision),

        "precision_at_top_1_percent_percentage":
            float(
                precision * 100.0
            ),

        "probability_cutoff":
            cutoff_probability,
    }


# ============================================================
# GET FRAUD PROBABILITIES
# ============================================================

def get_fraud_probabilities(
    model,
    x,
    edge_index,
    validation_mask
):

    model.eval()

    with torch.no_grad():

        logits = model(
            x,
            edge_index
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[:, 1]

    return probabilities[
        validation_mask
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )

    device = get_device()

    print()

    print(
        "=" * 70
    )

    print(
        "RFGN CLIENT 1 RECALL@1% COMPARISON"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Device: {device}"
    )

    print(
        f"Validation ratio: "
        f"{VALIDATION_RATIO}"
    )

    print(
        f"Ranking target: "
        f"Top {TOP_PERCENT * 100:.0f}%"
    )

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    graph = load_graph()

    x = graph.x.float().to(
        device
    )

    edge_index = graph.edge_index.long().to(
        device
    )

    y = graph.y.long().to(
        device
    )

    print()

    print(
        "Graph:"
    )

    print(
        f"  Nodes          : "
        f"{x.shape[0]:,}"
    )

    print(
        f"  Features       : "
        f"{x.shape[1]}"
    )

    print(
        f"  Directed edges : "
        f"{edge_index.shape[1]:,}"
    )

    # --------------------------------------------------------
    # Recreate exact validation split
    # --------------------------------------------------------

    (
        train_mask,
        validation_mask
    ) = create_stratified_split(
        y,
        VALIDATION_RATIO,
        SEED
    )

    validation_labels = y[
        validation_mask
    ]

    validation_count = int(
        validation_mask.sum().item()
    )

    validation_fraud = int(
        (
            validation_labels == 1
        ).sum().item()
    )

    print()

    print(
        "Validation set:"
    )

    print(
        f"  Samples : "
        f"{validation_count:,}"
    )

    print(
        f"  Fraud   : "
        f"{validation_fraud:,}"
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print()

    print(
        "Loading original GraphSAGE..."
    )

    original_model = load_original_model(
        device
    )

    print(
        "Original GraphSAGE: PASS"
    )

    print()

    print(
        "Loading enhanced GraphSAGE..."
    )

    enhanced_model = load_enhanced_model(
        device
    )

    print(
        "Enhanced GraphSAGE: PASS"
    )

    # --------------------------------------------------------
    # Original predictions
    # --------------------------------------------------------

    print()

    print(
        "Calculating original GraphSAGE "
        "fraud probabilities..."
    )

    original_probabilities = (
        get_fraud_probabilities(
            original_model,
            x,
            edge_index,
            validation_mask
        )
    )

    # --------------------------------------------------------
    # Enhanced predictions
    # --------------------------------------------------------

    print(
        "Calculating enhanced GraphSAGE "
        "fraud probabilities..."
    )

    enhanced_probabilities = (
        get_fraud_probabilities(
            enhanced_model,
            x,
            edge_index,
            validation_mask
        )
    )

    # --------------------------------------------------------
    # Validate predictions
    # --------------------------------------------------------

    if not torch.isfinite(
        original_probabilities
    ).all():

        raise RuntimeError(
            "Original model produced NaN/Inf "
            "probabilities."
        )

    if not torch.isfinite(
        enhanced_probabilities
    ).all():

        raise RuntimeError(
            "Enhanced model produced NaN/Inf "
            "probabilities."
        )

    # --------------------------------------------------------
    # Calculate Recall@1%
    # --------------------------------------------------------

    original_result = (
        calculate_recall_at_1_percent(
            original_probabilities,
            validation_labels
        )
    )

    enhanced_result = (
        calculate_recall_at_1_percent(
            enhanced_probabilities,
            validation_labels
        )
    )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    recall_difference = (
        enhanced_result[
            "recall_at_1_percent_percentage"
        ]
        -
        original_result[
            "recall_at_1_percent_percentage"
        ]
    )

    precision_difference = (
        enhanced_result[
            "precision_at_top_1_percent_percentage"
        ]
        -
        original_result[
            "precision_at_top_1_percent_percentage"
        ]
    )

    # --------------------------------------------------------
    # Determine winner
    # --------------------------------------------------------

    if recall_difference > 1e-9:

        winner = (
            "Enhanced GraphSAGE"
        )

    elif recall_difference < -1e-9:

        winner = (
            "Original GraphSAGE"
        )

    else:

        winner = (
            "Tie"
        )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "RECALL@1% RESULTS"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Original GraphSAGE"
    )

    print(
        f"  Recall@1%   : "
        f"{original_result['recall_at_1_percent_percentage']:.4f}%"
    )

    print(
        f"  Precision   : "
        f"{original_result['precision_at_top_1_percent_percentage']:.4f}%"
    )

    print(
        f"  Fraud       : "
        f"{original_result['fraud_captured']:,} / "
        f"{original_result['total_fraud']:,}"
    )

    print(
        f"  Top-1% count: "
        f"{original_result['top_1_percent_count']:,}"
    )

    print(
        f"  Cutoff      : "
        f"{original_result['probability_cutoff']:.8f}"
    )

    print()

    print(
        "Enhanced GraphSAGE"
    )

    print(
        f"  Recall@1%   : "
        f"{enhanced_result['recall_at_1_percent_percentage']:.4f}%"
    )

    print(
        f"  Precision   : "
        f"{enhanced_result['precision_at_top_1_percent_percentage']:.4f}%"
    )

    print(
        f"  Fraud       : "
        f"{enhanced_result['fraud_captured']:,} / "
        f"{enhanced_result['total_fraud']:,}"
    )

    print(
        f"  Top-1% count: "
        f"{enhanced_result['top_1_percent_count']:,}"
    )

    print(
        f"  Cutoff      : "
        f"{enhanced_result['probability_cutoff']:.8f}"
    )

    print()

    print(
        "Comparison"
    )

    print(
        f"  Recall difference    : "
        f"{recall_difference:+.4f} percentage points"
    )

    print(
        f"  Precision difference : "
        f"{precision_difference:+.4f} percentage points"
    )

    print(
        f"  Ranking winner       : "
        f"{winner}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {

        "experiment":
            "Client 1 Enhanced GraphSAGE "
            "Recall@1% Comparison",

        "seed":
            SEED,

        "validation_ratio":
            VALIDATION_RATIO,

        "top_percent":
            TOP_PERCENT,

        "graph":
            "graph_enhanced_aligned.pt",

        "validation_samples":
            validation_count,

        "validation_fraud":
            validation_fraud,

        "original_graphsage":
            original_result,

        "enhanced_graphsage":
            enhanced_result,

        "recall_difference_percentage_points":
            float(
                recall_difference
            ),

        "precision_difference_percentage_points":
            float(
                precision_difference
            ),

        "winner":
            winner,

        "original_model_path":
            ORIGINAL_MODEL_PATH,

        "enhanced_model_path":
            ENHANCED_MODEL_PATH,
    }

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "RECALL@1% COMPARISON COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()