# ============================================================
# RFGN CLIENT 1 RECALL@1% COMPARISON
# ============================================================
#
# Purpose:
#   Compare the original Client 1 GraphSAGE model against the
#   Focal Loss GraphSAGE model using the same validation split.
#
# Models:
#   1. Original weighted Cross Entropy model
#   2. Focal Loss model
#
# Evaluation:
#   - Same graph
#   - Same validation split
#   - Same random seed
#   - Same GraphSAGE architecture
#   - Same top 1% ranking protocol
#
# IMPORTANT:
#   Recall@1% is calculated by ranking transactions according
#   to fraud probability and investigating the highest-risk
#   1% of transactions.
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
# IMPORT MODEL
# ============================================================

from models.gnn.graphsage_model import GraphSAGE


# ============================================================
# CONFIGURATION
# ============================================================

CLIENT_ID = "client_1"

INPUT_DIM = 769

HIDDEN_DIM = 128

OUTPUT_DIM = 2

DROPOUT = 0.30


VALIDATION_RATIO = 0.15

RANDOM_SEED = 42

TOP_PERCENT = 0.01


# ============================================================
# PATHS
# ============================================================

GRAPH_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs",
    CLIENT_ID,
    "graph_aligned.pt"
)


ORIGINAL_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "local",
    "client_1_graphsage_aligned_best.pt"
)


FOCAL_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "local",
    "client_1_graphsage_focal_best.pt"
)


RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "local"
)


RESULT_PATH = os.path.join(
    RESULT_DIR,
    "client_1_recall_at_1pct_comparison.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(
    seed
):

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
# SECTION
# ============================================================

def print_section(
    title
):

    print()

    print(
        "=" * 70
    )

    print(
        title
    )

    print(
        "=" * 70
    )


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph():

    print_section(
        "LOADING CLIENT 1 ALIGNED GRAPH"
    )

    if not os.path.exists(
        GRAPH_PATH
    ):

        raise FileNotFoundError(
            "Graph not found:\n"
            f"{GRAPH_PATH}"
        )

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )

    print(
        f"Graph: {GRAPH_PATH}"
    )

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
        f"Labels   : "
        f"{graph.y.shape[0]:,}"
    )

    return graph


# ============================================================
# VALIDATE GRAPH
# ============================================================

def validate_graph(
    graph
):

    print_section(
        "VALIDATING GRAPH"
    )

    if (
        graph.num_node_features
        != INPUT_DIM
    ):

        raise RuntimeError(
            f"Expected {INPUT_DIM} features, "
            f"found {graph.num_node_features}."
        )

    print(
        "Feature dimension: PASS"
    )

    if (
        graph.num_nodes
        != graph.y.shape[0]
    ):

        raise RuntimeError(
            "Node and label counts do not match."
        )

    print(
        "Node/label count: PASS"
    )

    if torch.isnan(
        graph.x
    ).any():

        raise RuntimeError(
            "NaN detected in graph features."
        )

    print(
        "NaN check: PASS"
    )

    if torch.isinf(
        graph.x
    ).any():

        raise RuntimeError(
            "Infinity detected in graph features."
        )

    print(
        "Infinity check: PASS"
    )

    unique_labels = (
        torch.unique(
            graph.y
        )
        .cpu()
        .tolist()
    )

    if not set(
        unique_labels
    ).issubset(
        {0, 1}
    ):

        raise RuntimeError(
            f"Invalid labels: {unique_labels}"
        )

    print(
        "Binary labels: PASS"
    )


# ============================================================
# CREATE SAME VALIDATION SPLIT
# ============================================================

def create_validation_split(
    labels,
    validation_ratio,
    seed,
):

    print_section(
        "RECREATING IDENTICAL VALIDATION SPLIT"
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    labels = labels.cpu()

    class_zero = torch.where(
        labels == 0
    )[0]

    class_one = torch.where(
        labels == 1
    )[0]

    class_zero = class_zero[
        torch.randperm(
            len(class_zero),
            generator=generator
        )
    ]

    class_one = class_one[
        torch.randperm(
            len(class_one),
            generator=generator
        )
    ]

    val_zero_count = int(
        len(class_zero)
        * validation_ratio
    )

    val_one_count = int(
        len(class_one)
        * validation_ratio
    )

    val_indices = torch.cat(
        [
            class_zero[
                :val_zero_count
            ],

            class_one[
                :val_one_count
            ],
        ]
    )

    val_indices = val_indices[
        torch.randperm(
            len(val_indices),
            generator=generator
        )
    ]

    val_labels = labels[
        val_indices
    ]

    legitimate = int(
        (val_labels == 0)
        .sum()
        .item()
    )

    fraud = int(
        (val_labels == 1)
        .sum()
        .item()
    )

    print(
        f"Validation rows : "
        f"{len(val_indices):,}"
    )

    print(
        f"Legitimate      : "
        f"{legitimate:,}"
    )

    print(
        f"Fraud           : "
        f"{fraud:,}"
    )

    print(
        f"Fraud rate      : "
        f"{fraud / len(val_indices) * 100:.4f}%"
    )

    print(
        "Validation split recreation: PASS"
    )

    return val_indices


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model_path,
    device,
):

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            "Model not found:\n"
            f"{model_path}"
        )

    model = GraphSAGE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "model_state_dict not found in:\n"
            f"{model_path}"
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        "Loaded:",
        os.path.basename(
            model_path
        )
    )

    if "best_epoch" in checkpoint:

        print(
            "Best epoch:",
            checkpoint["best_epoch"]
        )

    if "best_val_loss" in checkpoint:

        print(
            "Best validation loss:",
            f"{checkpoint['best_val_loss']:.6f}"
        )

    if "loss_function" in checkpoint:

        print(
            "Loss function:",
            checkpoint["loss_function"]
        )

    return model


# ============================================================
# PREDICT FRAUD PROBABILITIES
# ============================================================

@torch.no_grad()
def predict_probabilities(
    model,
    graph,
    val_indices,
    device,
):

    graph = graph.to(
        device
    )

    val_indices = val_indices.to(
        device
    )

    output = model(
        graph.x,
        graph.edge_index
    )

    if output.ndim != 2:

        raise RuntimeError(
            f"Expected output shape [N, 2], "
            f"found {tuple(output.shape)}."
        )

    probabilities = torch.softmax(
        output,
        dim=1
    )

    fraud_probabilities = (
        probabilities[
            :,
            1
        ]
    )

    validation_probabilities = (
        fraud_probabilities[
            val_indices
        ]
    )

    validation_labels = (
        graph.y[
            val_indices
        ]
    )

    if torch.isnan(
        validation_probabilities
    ).any():

        raise RuntimeError(
            "NaN detected in fraud probabilities."
        )

    if torch.isinf(
        validation_probabilities
    ).any():

        raise RuntimeError(
            "Infinity detected in fraud probabilities."
        )

    return (
        validation_probabilities.cpu(),
        validation_labels.cpu()
    )


# ============================================================
# RECALL@1%
# ============================================================

def calculate_recall_at_1_percent(
    probabilities,
    labels,
):

    total = len(
        labels
    )

    fraud_total = int(
        (labels == 1)
        .sum()
        .item()
    )

    if total == 0:

        raise RuntimeError(
            "No validation transactions."
        )

    if fraud_total == 0:

        raise RuntimeError(
            "No fraud transactions in validation set."
        )

    top_count = int(
        total
        * TOP_PERCENT
    )

    if top_count < 1:

        top_count = 1

    # --------------------------------------------------------
    # Rank transactions by fraud probability.
    # Highest probability = highest suspiciousness.
    # --------------------------------------------------------

    sorted_indices = torch.argsort(
        probabilities,
        descending=True,
        stable=True
    )

    top_indices = (
        sorted_indices[
            :top_count
        ]
    )

    top_labels = (
        labels[
            top_indices
        ]
    )

    fraud_captured = int(
        (top_labels == 1)
        .sum()
        .item()
    )

    fraud_missed = (
        fraud_total
        - fraud_captured
    )

    recall = (
        fraud_captured
        /
        fraud_total
    )

    precision_at_1 = (
        fraud_captured
        /
        top_count
    )

    cutoff_probability = float(
        probabilities[
            sorted_indices[
                top_count - 1
            ]
        ]
        .item()
    )

    predicted_fraud_rate = (
        top_count
        /
        total
    )

    return {

        "total_transactions":
            total,

        "fraud_transactions":
            fraud_total,

        "top_1_percent_count":
            top_count,

        "fraud_captured":
            fraud_captured,

        "fraud_missed":
            fraud_missed,

        "recall_at_1_percent":
            float(recall),

        "recall_at_1_percent_percentage":
            float(recall * 100.0),

        "precision_at_top_1_percent":
            float(precision_at_1),

        "precision_at_top_1_percent_percentage":
            float(precision_at_1 * 100.0),

        "top_1_percent_cutoff_probability":
            cutoff_probability,

        "investigation_rate":
            float(predicted_fraud_rate),

        "investigation_rate_percentage":
            float(predicted_fraud_rate * 100.0),
    }


# ============================================================
# ADD PROBABILITY STATISTICS
# ============================================================

def add_probability_statistics(
    result,
    probabilities,
):

    result["probability_min"] = float(
        probabilities.min().item()
    )

    result["probability_max"] = float(
        probabilities.max().item()
    )

    result["probability_mean"] = float(
        probabilities.mean().item()
    )

    result["probability_std"] = float(
        probabilities.std(
            unbiased=False
        ).item()
    )


# ============================================================
# COMPARE MODELS
# ============================================================

def compare_results(
    original_result,
    focal_result,
):

    print_section(
        "MODEL COMPARISON"
    )

    original_recall = (
        original_result[
            "recall_at_1_percent"
        ]
    )

    focal_recall = (
        focal_result[
            "recall_at_1_percent"
        ]
    )

    recall_difference = (
        focal_recall
        - original_recall
    )

    original_precision = (
        original_result[
            "precision_at_top_1_percent"
        ]
    )

    focal_precision = (
        focal_result[
            "precision_at_top_1_percent"
        ]
    )

    precision_difference = (
        focal_precision
        - original_precision
    )

    print()

    print(
        f"{'Metric':<35}"
        f"{'Original':>15}"
        f"{'Focal Loss':>15}"
        f"{'Difference':>15}"
    )

    print(
        "-" * 80
    )

    print(
        f"{'Recall@1%':<35}"
        f"{original_recall * 100:>14.4f}%"
        f"{focal_recall * 100:>14.4f}%"
        f"{recall_difference * 100:>14.4f}%"
    )

    print(
        f"{'Precision @ top 1%':<35}"
        f"{original_precision * 100:>14.4f}%"
        f"{focal_precision * 100:>14.4f}%"
        f"{precision_difference * 100:>14.4f}%"
    )

    print(
        f"{'Fraud captured':<35}"
        f"{original_result['fraud_captured']:>15,}"
        f"{focal_result['fraud_captured']:>15,}"
        f"{focal_result['fraud_captured'] - original_result['fraud_captured']:>15,}"
    )

    print(
        f"{'Fraud missed':<35}"
        f"{original_result['fraud_missed']:>15,}"
        f"{focal_result['fraud_missed']:>15,}"
        f"{focal_result['fraud_missed'] - original_result['fraud_missed']:>15,}"
    )

    print()

    if recall_difference > 0:

        winner = "Focal Loss"

        print(
            "Recall@1% winner: Focal Loss"
        )

    elif recall_difference < 0:

        winner = "Original"

        print(
            "Recall@1% winner: Original model"
        )

    else:

        winner = "Tie"

        print(
            "Recall@1% winner: TIE"
        )

    return {
        "recall_difference":
            float(recall_difference),

        "recall_difference_percentage_points":
            float(recall_difference * 100.0),

        "precision_difference":
            float(precision_difference),

        "precision_difference_percentage_points":
            float(precision_difference * 100.0),

        "winner":
            winner,
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results
):

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print()

    print(
        "Results saved:"
    )

    print(
        RESULT_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 70
    )

    print(
        "RFGN CLIENT 1 RECALL@1% MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Experiment:"
    )

    print(
        "Original GraphSAGE vs Focal Loss GraphSAGE"
    )

    print()

    print(
        "Evaluation protocol:"
    )

    print(
        "  Client              : Client 1"
    )

    print(
        "  Graph               : graph_aligned.pt"
    )

    print(
        "  Validation ratio    : 15%"
    )

    print(
        "  Random seed         : 42"
    )

    print(
        "  Ranking metric      : Recall@1%"
    )

    print(
        "  Investigation rate  : 1%"
    )

    print(
        "  Architecture        : GraphSAGE"
    )

    # ========================================================
    # SEED
    # ========================================================

    set_seed(
        RANDOM_SEED
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()

    print_section(
        "SELECTING DEVICE"
    )

    print(
        f"Device: {device}"
    )

    # ========================================================
    # LOAD GRAPH
    # ========================================================

    graph = load_graph()

    validate_graph(
        graph
    )

    # ========================================================
    # VALIDATION SPLIT
    # ========================================================

    val_indices = (
        create_validation_split(
            graph.y,
            VALIDATION_RATIO,
            RANDOM_SEED,
        )
    )

    # ========================================================
    # LOAD ORIGINAL MODEL
    # ========================================================

    print_section(
        "LOADING ORIGINAL MODEL"
    )

    original_model = load_model(
        ORIGINAL_MODEL_PATH,
        device,
    )

    # ========================================================
    # ORIGINAL PREDICTIONS
    # ========================================================

    print_section(
        "EVALUATING ORIGINAL MODEL"
    )

    (
        original_probabilities,
        original_labels,
    ) = predict_probabilities(
        original_model,
        graph,
        val_indices,
        device,
    )

    original_result = (
        calculate_recall_at_1_percent(
            original_probabilities,
            original_labels,
        )
    )

    add_probability_statistics(
        original_result,
        original_probabilities,
    )

    print()

    print(
        "Original model results:"
    )

    print(
        f"  Validation transactions : "
        f"{original_result['total_transactions']:,}"
    )

    print(
        f"  Fraud transactions      : "
        f"{original_result['fraud_transactions']:,}"
    )

    print(
        f"  Top 1% count            : "
        f"{original_result['top_1_percent_count']:,}"
    )

    print(
        f"  Fraud captured          : "
        f"{original_result['fraud_captured']:,}"
    )

    print(
        f"  Fraud missed            : "
        f"{original_result['fraud_missed']:,}"
    )

    print(
        f"  Recall@1%               : "
        f"{original_result['recall_at_1_percent_percentage']:.4f}%"
    )

    print(
        f"  Precision @ top 1%     : "
        f"{original_result['precision_at_top_1_percent_percentage']:.4f}%"
    )

    print(
        f"  Top 1% cutoff           : "
        f"{original_result['top_1_percent_cutoff_probability']:.8f}"
    )

    # ========================================================
    # CLEAN MODEL FROM MEMORY
    # ========================================================

    del original_model

    if device.type == "cuda":

        torch.cuda.empty_cache()

    # ========================================================
    # LOAD FOCAL MODEL
    # ========================================================

    print_section(
        "LOADING FOCAL LOSS MODEL"
    )

    focal_model = load_model(
        FOCAL_MODEL_PATH,
        device,
    )

    # ========================================================
    # FOCAL PREDICTIONS
    # ========================================================

    print_section(
        "EVALUATING FOCAL LOSS MODEL"
    )

    (
        focal_probabilities,
        focal_labels,
    ) = predict_probabilities(
        focal_model,
        graph,
        val_indices,
        device,
    )

    # --------------------------------------------------------
    # Ensure identical labels
    # --------------------------------------------------------

    if not torch.equal(
        original_labels,
        focal_labels
    ):

        raise RuntimeError(
            "Original and Focal validation labels "
            "do not match."
        )

    focal_result = (
        calculate_recall_at_1_percent(
            focal_probabilities,
            focal_labels,
        )
    )

    add_probability_statistics(
        focal_result,
        focal_probabilities,
    )

    print()

    print(
        "Focal Loss results:"
    )

    print(
        f"  Validation transactions : "
        f"{focal_result['total_transactions']:,}"
    )

    print(
        f"  Fraud transactions      : "
        f"{focal_result['fraud_transactions']:,}"
    )

    print(
        f"  Top 1% count            : "
        f"{focal_result['top_1_percent_count']:,}"
    )

    print(
        f"  Fraud captured          : "
        f"{focal_result['fraud_captured']:,}"
    )

    print(
        f"  Fraud missed            : "
        f"{focal_result['fraud_missed']:,}"
    )

    print(
        f"  Recall@1%               : "
        f"{focal_result['recall_at_1_percent_percentage']:.4f}%"
    )

    print(
        f"  Precision @ top 1%     : "
        f"{focal_result['precision_at_top_1_percent_percentage']:.4f}%"
    )

    print(
        f"  Top 1% cutoff           : "
        f"{focal_result['top_1_percent_cutoff_probability']:.8f}"
    )

    # ========================================================
    # COMPARE
    # ========================================================

    comparison = compare_results(
        original_result,
        focal_result,
    )

    # ========================================================
    # SAVE
    # ========================================================

    results = {

        "experiment":
            "Client 1 Original vs Focal Loss Recall@1%",

        "client_id":
            CLIENT_ID,

        "graph_path":
            GRAPH_PATH,

        "validation_ratio":
            VALIDATION_RATIO,

        "random_seed":
            RANDOM_SEED,

        "top_percent":
            TOP_PERCENT,

        "metric":
            "Recall@1%",

        "original_model":
            {
                "path":
                    ORIGINAL_MODEL_PATH,

                "results":
                    original_result,
            },

        "focal_loss_model":
            {
                "path":
                    FOCAL_MODEL_PATH,

                "results":
                    focal_result,
            },

        "comparison":
            comparison,
    }

    save_results(
        results
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "RECALL@1% COMPARISON COMPLETED"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Original Recall@1% : "
        f"{original_result['recall_at_1_percent_percentage']:.4f}%"
    )

    print(
        f"Focal Recall@1%    : "
        f"{focal_result['recall_at_1_percent_percentage']:.4f}%"
    )

    print(
        f"Difference          : "
        f"{comparison['recall_difference_percentage_points']:+.4f} percentage points"
    )

    print()

    print(
        f"Winner: "
        f"{comparison['winner']}"
    )

    print()

    print(
        "Existing models were not modified."
    )

    print(
        "No training was performed during this test."
    )

    print()

    print(
        "=" * 70
    )

    print(
        "RFGN RECALL@1% COMPARISON: PASS"
    )

    print(
        "=" * 70
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()