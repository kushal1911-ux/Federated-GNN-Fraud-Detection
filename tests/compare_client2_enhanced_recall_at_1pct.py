import sys
from pathlib import Path

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import json
import random

import numpy as np
import torch
from torch_geometric.data import Data

from models.gnn.graphsage_model import GraphSAGE
from models.gnn.enhanced_graphsage_model import EnhancedGraphSAGE


# ============================================================
# CONFIGURATION
# ============================================================

GRAPH_PATH = (
    ROOT
    / "data"
    / "processed"
    / "graphs"
    / "client_2"
    / "graph_enhanced_aligned.pt"
)

ORIGINAL_MODEL_PATH = (
    ROOT
    / "saved_models"
    / "local"
    / "client_2_graphsage_aligned_best.pt"
)

ENHANCED_MODEL_PATH = (
    ROOT
    / "saved_models"
    / "local"
    / "client_2_enhanced_graphsage_best.pt"
)

OUTPUT_PATH = (
    ROOT
    / "saved_models"
    / "local"
    / "client_2_enhanced_recall_at_1pct_comparison.json"
)

INPUT_DIM = 769
HIDDEN_DIM = 128
OUTPUT_DIM = 2
DROPOUT = 0.30

VAL_RATIO = 0.15
SEED = 42
TOP_FRACTION = 0.01


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph():

    print("Loading Client 2 enhanced aligned graph...")

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False
    )

    if not isinstance(graph, Data):
        raise TypeError(
            f"Expected PyG Data object, got {type(graph)}"
        )

    print("\nGraph information:")
    print(f"  Nodes          : {graph.num_nodes:,}")
    print(f"  Features       : {graph.num_node_features}")
    print(f"  Directed edges : {graph.edge_index.shape[1]:,}")
    print(f"  Labels         : {graph.y.shape[0]:,}")

    if hasattr(graph, "edge_type"):
        print(
            f"  Edge types     : "
            f"{torch.unique(graph.edge_type).tolist()}"
        )

    fraud_count = int((graph.y == 1).sum().item())
    legit_count = int((graph.y == 0).sum().item())

    print(f"\n  Class 0 count : {legit_count:,}")
    print(f"  Class 1 count : {fraud_count:,}")

    if graph.num_node_features != INPUT_DIM:
        raise ValueError(
            f"Expected {INPUT_DIM} features, "
            f"got {graph.num_node_features}"
        )

    return graph


# ============================================================
# CREATE VALIDATION SPLIT
# ============================================================

def create_validation_split(num_nodes: int):

    set_seed(SEED)

    indices = torch.randperm(num_nodes)

    val_size = int(num_nodes * VAL_RATIO)

    val_idx = indices[:val_size]

    return val_idx


# ============================================================
# LOAD ORIGINAL GRAPHSAGE
# ============================================================

def load_original_model(device):

    print("\nLoading Original GraphSAGE...")

    model = GraphSAGE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    )

    checkpoint = torch.load(
        ORIGINAL_MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print("  Original model loaded.")

    return model


# ============================================================
# LOAD ENHANCED GRAPHSAGE
# ============================================================

def load_enhanced_model(device):

    print("\nLoading Enhanced GraphSAGE...")

    model = EnhancedGraphSAGE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    )

    checkpoint = torch.load(
        ENHANCED_MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print("  Enhanced model loaded.")

    return model


# ============================================================
# GET FRAUD PROBABILITIES
# ============================================================

@torch.no_grad()
def get_probabilities(model, graph, device):

    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)

    output = model(
        x,
        edge_index
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )[:, 1]

    return probabilities.cpu()


# ============================================================
# CALCULATE RECALL@1%
# ============================================================

def calculate_recall_at_1pct(
    probabilities,
    labels
):

    probabilities = probabilities.numpy()
    labels = labels.numpy()

    total_samples = len(labels)

    top_count = max(
        1,
        int(
            np.ceil(
                total_samples * TOP_FRACTION
            )
        )
    )

    ranked_indices = np.argsort(
        -probabilities
    )

    top_indices = ranked_indices[:top_count]

    top_labels = labels[top_indices]

    fraud_total = int(
        (labels == 1).sum()
    )

    fraud_captured = int(
        (top_labels == 1).sum()
    )

    fraud_missed = (
        fraud_total
        - fraud_captured
    )

    recall = (
        fraud_captured / fraud_total
        if fraud_total > 0
        else 0.0
    )

    precision = (
        fraud_captured / top_count
        if top_count > 0
        else 0.0
    )

    cutoff = float(
        probabilities[
            top_indices[-1]
        ]
    )

    return {
        "validation_samples": int(
            total_samples
        ),
        "validation_fraud": int(
            fraud_total
        ),
        "top_1pct_count": int(
            top_count
        ),
        "fraud_captured": int(
            fraud_captured
        ),
        "fraud_missed": int(
            fraud_missed
        ),
        "recall_at_1pct": float(
            recall
        ),
        "precision_at_1pct": float(
            precision
        ),
        "cutoff_probability": cutoff,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CLIENT 2 ENHANCED GRAPHSAGE RECALL@1% COMPARISON")
    print("=" * 70)

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    graph = load_graph()

    # --------------------------------------------------------
    # Validation split
    # --------------------------------------------------------

    print("\nCreating validation split...")

    val_idx = create_validation_split(
        graph.num_nodes
    )

    val_labels = graph.y[
        val_idx
    ].cpu()

    print(
        f"  Validation samples : "
        f"{len(val_idx):,}"
    )

    print(
        f"  Validation fraud   : "
        f"{int((val_labels == 1).sum()):,}"
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    original_model = load_original_model(
        device
    )

    enhanced_model = load_enhanced_model(
        device
    )

    # --------------------------------------------------------
    # Original inference
    # --------------------------------------------------------

    print(
        "\nRunning Original GraphSAGE inference..."
    )

    original_probabilities_all = (
        get_probabilities(
            original_model,
            graph,
            device
        )
    )

    original_probabilities = (
        original_probabilities_all[
            val_idx
        ]
    )

    print(
        "  Original inference complete."
    )

    # --------------------------------------------------------
    # Enhanced inference
    # --------------------------------------------------------

    print(
        "\nRunning Enhanced GraphSAGE inference..."
    )

    enhanced_probabilities_all = (
        get_probabilities(
            enhanced_model,
            graph,
            device
        )
    )

    enhanced_probabilities = (
        enhanced_probabilities_all[
            val_idx
        ]
    )

    print(
        "  Enhanced inference complete."
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    original_result = (
        calculate_recall_at_1pct(
            original_probabilities,
            val_labels
        )
    )

    enhanced_result = (
        calculate_recall_at_1pct(
            enhanced_probabilities,
            val_labels
        )
    )

    recall_difference = (
        enhanced_result[
            "recall_at_1pct"
        ]
        -
        original_result[
            "recall_at_1pct"
        ]
    )

    precision_difference = (
        enhanced_result[
            "precision_at_1pct"
        ]
        -
        original_result[
            "precision_at_1pct"
        ]
    )

    # --------------------------------------------------------
    # Winner
    # --------------------------------------------------------

    if recall_difference > 0:
        winner = "Enhanced GraphSAGE"

    elif recall_difference < 0:
        winner = "Original GraphSAGE"

    else:
        winner = "Tie"

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLIENT 2 RECALL@1% RESULTS")
    print("=" * 70)

    print("\nOriginal GraphSAGE:")

    print(
        f"  Recall@1%       : "
        f"{original_result['recall_at_1pct']:.4%}"
    )

    print(
        f"  Precision@1%    : "
        f"{original_result['precision_at_1pct']:.4%}"
    )

    print(
        f"  Fraud captured  : "
        f"{original_result['fraud_captured']}"
        f"/"
        f"{original_result['validation_fraud']}"
    )

    print(
        f"  Top 1% count    : "
        f"{original_result['top_1pct_count']:,}"
    )

    print(
        f"  Cutoff          : "
        f"{original_result['cutoff_probability']:.8f}"
    )

    print("\nEnhanced GraphSAGE:")

    print(
        f"  Recall@1%       : "
        f"{enhanced_result['recall_at_1pct']:.4%}"
    )

    print(
        f"  Precision@1%    : "
        f"{enhanced_result['precision_at_1pct']:.4%}"
    )

    print(
        f"  Fraud captured  : "
        f"{enhanced_result['fraud_captured']}"
        f"/"
        f"{enhanced_result['validation_fraud']}"
    )

    print(
        f"  Top 1% count    : "
        f"{enhanced_result['top_1pct_count']:,}"
    )

    print(
        f"  Cutoff          : "
        f"{enhanced_result['cutoff_probability']:.8f}"
    )

    print("\nDifference:")

    print(
        f"  Recall@1%       : "
        f"{recall_difference:+.4%}"
    )

    print(
        f"  Precision@1%    : "
        f"{precision_difference:+.4%}"
    )

    print(
        f"\nWinner by Recall@1%: "
        f"{winner}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    result = {

        "client": 2,

        "seed": SEED,

        "validation_ratio": VAL_RATIO,

        "top_fraction": TOP_FRACTION,

        "graph": str(
            GRAPH_PATH
        ),

        "original_model": str(
            ORIGINAL_MODEL_PATH
        ),

        "enhanced_model": str(
            ENHANCED_MODEL_PATH
        ),

        "original": original_result,

        "enhanced": enhanced_result,

        "difference": {

            "recall_at_1pct": float(
                recall_difference
            ),

            "precision_at_1pct": float(
                precision_difference
            ),
        },

        "winner_by_recall_at_1pct": winner,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CLIENT 2 COMPARISON COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()