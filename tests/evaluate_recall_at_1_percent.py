from pathlib import Path
import json
import sys

import torch
from torch_geometric.data import Data

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.gnn.graphsage_model import GraphSAGE


# ============================================================
# PATHS
# ============================================================

GLOBAL_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "global"
    / "global_graphsage_flower.pt"
)

GRAPH_PATHS = [
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graphs"
    / "client_1"
    / "graph_aligned.pt",

    PROJECT_ROOT
    / "data"
    / "processed"
    / "graphs"
    / "client_2"
    / "graph_aligned.pt",

    PROJECT_ROOT
    / "data"
    / "processed"
    / "graphs"
    / "client_3"
    / "graph_aligned.pt",
]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "global"
    / "recall_at_1_percent.json"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

INPUT_DIM = 769
HIDDEN_DIM = 128
OUTPUT_DIM = 2
DROPOUT = 0.30

TOP_PERCENT = 0.01

SEED = 42

torch.manual_seed(SEED)


# ============================================================
# LOAD GLOBAL MODEL
# ============================================================

def load_global_model():

    if not GLOBAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Global model not found:\n{GLOBAL_MODEL_PATH}"
        )

    model = GraphSAGE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    )

    checkpoint = torch.load(
        GLOBAL_MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        raise TypeError(
            "Unsupported global model checkpoint format."
        )

    model.load_state_dict(state_dict)

    model.eval()

    return model


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Graph not found:\n{path}"
        )

    graph = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(graph, Data):
        raise TypeError(
            f"Expected PyG Data object, got {type(graph)}"
        )

    if graph.x.ndim != 2:
        raise ValueError(
            f"Invalid graph.x shape: {tuple(graph.x.shape)}"
        )

    if graph.x.shape[1] != INPUT_DIM:
        raise ValueError(
            f"Expected {INPUT_DIM} features, "
            f"got {graph.x.shape[1]}"
        )

    if not hasattr(graph, "y"):
        raise ValueError(
            "Graph does not contain labels."
        )

    return graph


# ============================================================
# RECALL@1%
# ============================================================

def calculate_recall_at_1_percent(
    probabilities,
    labels,
):

    probabilities = (
        probabilities
        .detach()
        .cpu()
        .flatten()
    )

    labels = (
        labels
        .detach()
        .cpu()
        .flatten()
        .long()
    )

    if probabilities.numel() != labels.numel():
        raise ValueError(
            "Probability/label length mismatch."
        )

    total_samples = probabilities.numel()

    if total_samples == 0:
        raise ValueError(
            "No samples available."
        )

    total_fraud = int(
        (labels == 1).sum().item()
    )

    if total_fraud == 0:
        raise ValueError(
            "No fraud samples available."
        )

    # --------------------------------------------------------
    # Exact top 1% size
    # --------------------------------------------------------

    top_k = max(
        1,
        int(total_samples * TOP_PERCENT)
    )

    # --------------------------------------------------------
    # Rank by fraud probability
    # --------------------------------------------------------

    sorted_indices = torch.argsort(
        probabilities,
        descending=True,
        stable=True,
    )

    top_indices = sorted_indices[:top_k]

    top_probabilities = probabilities[
        top_indices
    ]

    top_labels = labels[
        top_indices
    ]

    # --------------------------------------------------------
    # Fraud captured
    # --------------------------------------------------------

    fraud_captured = int(
        (top_labels == 1).sum().item()
    )

    fraud_missed = (
        total_fraud
        - fraud_captured
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = (
        fraud_captured
        / total_fraud
    )

    # --------------------------------------------------------
    # Precision within top 1%
    # --------------------------------------------------------

    precision = (
        fraud_captured
        / top_k
    )

    # --------------------------------------------------------
    # Probability cutoff
    # --------------------------------------------------------

    cutoff_probability = float(
        top_probabilities[-1].item()
    )

    return {
        "total_samples": total_samples,
        "total_fraud": total_fraud,
        "total_legitimate": (
            total_samples
            - total_fraud
        ),
        "top_percent": TOP_PERCENT,
        "top_k": top_k,
        "fraud_captured": fraud_captured,
        "fraud_missed": fraud_missed,
        "recall": recall,
        "recall_percentage": (
            recall * 100.0
        ),
        "precision": precision,
        "precision_percentage": (
            precision * 100.0
        ),
        "cutoff_probability": cutoff_probability,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RFGN — RECALL@1% EVALUATION")
    print("=" * 70)

    print()
    print("Authoritative model:")
    print(GLOBAL_MODEL_PATH)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading Global GraphSAGE...")

    model = load_global_model()

    print("Global GraphSAGE loaded successfully.")

    print()
    print("Model configuration:")
    print(f"Input features : {INPUT_DIM}")
    print(f"Hidden features: {HIDDEN_DIM}")
    print(f"Output classes : {OUTPUT_DIM}")
    print(f"Dropout        : {DROPOUT}")

    # --------------------------------------------------------
    # Evaluate clients
    # --------------------------------------------------------

    all_probabilities = []
    all_labels = []

    client_results = []

    for client_number, graph_path in enumerate(
        GRAPH_PATHS,
        start=1,
    ):

        print()
        print("-" * 70)
        print(f"CLIENT {client_number}")
        print("-" * 70)

        graph = load_graph(
            graph_path
        )

        node_count = int(
            graph.x.shape[0]
        )

        fraud_count = int(
            (graph.y == 1).sum().item()
        )

        legitimate_count = (
            node_count
            - fraud_count
        )

        print(
            f"Nodes     : {node_count:,}"
        )

        print(
            f"Features  : {graph.x.shape[1]}"
        )

        print(
            f"Fraud     : {fraud_count:,}"
        )

        print(
            f"Legitimate: {legitimate_count:,}"
        )

        # ----------------------------------------------------
        # GraphSAGE inference
        # ----------------------------------------------------

        with torch.no_grad():

            logits = model(
                graph.x,
                graph.edge_index,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

        # ----------------------------------------------------
        # Numerical validation
        # ----------------------------------------------------

        if torch.isnan(
            probabilities
        ).any():

            raise ValueError(
                f"NaN probabilities detected "
                f"in Client {client_number}."
            )

        if torch.isinf(
            probabilities
        ).any():

            raise ValueError(
                f"Inf probabilities detected "
                f"in Client {client_number}."
            )

        all_probabilities.append(
            probabilities
        )

        all_labels.append(
            graph.y
        )

        client_results.append(
            {
                "client": client_number,
                "samples": node_count,
                "fraud": fraud_count,
                "legitimate": legitimate_count,
                "probability_min": float(
                    probabilities.min().item()
                ),
                "probability_max": float(
                    probabilities.max().item()
                ),
                "probability_mean": float(
                    probabilities.mean().item()
                ),
            }
        )

        print(
            f"Probability min: "
            f"{probabilities.min().item():.6f}"
        )

        print(
            f"Probability max: "
            f"{probabilities.max().item():.6f}"
        )

        print(
            f"Probability mean: "
            f"{probabilities.mean().item():.6f}"
        )

        print(
            "Inference: PASS"
        )

    # --------------------------------------------------------
    # Pool all clients
    # --------------------------------------------------------

    probabilities = torch.cat(
        all_probabilities,
        dim=0,
    )

    labels = torch.cat(
        all_labels,
        dim=0,
    )

    print()
    print("=" * 70)
    print("POOLED THREE-CLIENT EVALUATION")
    print("=" * 70)

    metrics = calculate_recall_at_1_percent(
        probabilities,
        labels,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print(
        f"Total transactions : "
        f"{metrics['total_samples']:,}"
    )

    print(
        f"Total actual fraud : "
        f"{metrics['total_fraud']:,}"
    )

    print(
        f"Top 1% count       : "
        f"{metrics['top_k']:,}"
    )

    print(
        f"Frauds captured    : "
        f"{metrics['fraud_captured']:,}"
    )

    print(
        f"Frauds missed      : "
        f"{metrics['fraud_missed']:,}"
    )

    print()
    print(
        f"Recall@1%          : "
        f"{metrics['recall_percentage']:.4f}%"
    )

    print(
        f"Precision @ top 1% : "
        f"{metrics['precision_percentage']:.4f}%"
    )

    print(
        f"1% cutoff          : "
        f"{metrics['cutoff_probability']:.6f}"
    )

    print()
    print(
        "Reference benchmark:"
    )

    print(
        "IEEE-CIS Recall@1% = 96.9%"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The 96.9% value is the reference-paper "
        "benchmark, not our measured result."
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result = {

        "experiment":
            "RFGN Recall@1% Evaluation",

        "model": {

            "path":
                str(GLOBAL_MODEL_PATH),

            "type":
                "Global GraphSAGE",

            "aggregation":
                "Flower FedAvg",

            "input_features":
                INPUT_DIM,

            "hidden_features":
                HIDDEN_DIM,

            "output_classes":
                OUTPUT_DIM,

            "dropout":
                DROPOUT,
        },

        "evaluation": {

            "clients":
                3,

            "ranking_metric":
                "Recall@1%",

            "top_percent":
                TOP_PERCENT,

            "seed":
                SEED,
        },

        "client_results":
            client_results,

        "pooled_results":
            metrics,

        "reference_benchmark": {

            "dataset":
                "IEEE-CIS",

            "recall_at_1_percent_percentage":
                96.9,

            "note":
                "Reference-paper benchmark",
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULT SAVED")
    print("=" * 70)

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "Recall@1% evaluation completed successfully."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()