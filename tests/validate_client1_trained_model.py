# ============================================================
# RFGN CLIENT 1 TRAINED GRAPHSAGE MODEL VALIDATION
# ============================================================
#
# Purpose:
#   Validate the trained Client 1 GraphSAGE model.
#
# Checks:
#   - Aligned graph exists
#   - Best model exists
#   - Model checkpoint structure
#   - Graph structure
#   - Shared 769-feature compatibility
#   - Model reload
#   - Forward pass
#   - Finite logits/probabilities
#   - Prediction metrics
#   - Fraud probability statistics
#
# IMPORTANT:
#   - READ-ONLY validation
#   - No training
#   - No optimizer updates
#   - No Client 2/3 data
#   - No external validation/test CSV
#
# ============================================================

import os
import sys
import time

import torch
import torch.nn.functional as F


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
# IMPORT MODEL
# ============================================================

from models.gnn.graphsage_model import GraphSAGE


# ============================================================
# CONFIGURATION
# ============================================================

CLIENT_ID = "client_1"

EXPECTED_INPUT_DIM = 769
EXPECTED_HIDDEN_DIM = 128
EXPECTED_OUTPUT_DIM = 2

GRAPH_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs",
    CLIENT_ID,
    "graph_aligned.pt",
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "local",
    "client_1_graphsage_aligned_best.pt",
)


# ============================================================
# PRINT HELPERS
# ============================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def pass_message(message):

    print(f"{message}: PASS")


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# FILE CHECKS
# ============================================================

def check_files():

    print_section(
        "CHECKING CLIENT 1 TRAINED MODEL FILES"
    )

    print(f"Graph:")
    print(GRAPH_PATH)

    if not os.path.exists(GRAPH_PATH):

        raise FileNotFoundError(
            f"Aligned graph not found:\n{GRAPH_PATH}"
        )

    pass_message("Aligned graph file")


    print()

    print("Best model:")
    print(MODEL_PATH)

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Best trained model not found:\n{MODEL_PATH}"
        )

    pass_message("Best model file")


    graph_size = (
        os.path.getsize(GRAPH_PATH)
        / (1024 ** 2)
    )

    model_size = (
        os.path.getsize(MODEL_PATH)
        / (1024 ** 2)
    )


    print()
    print(f"Graph size : {graph_size:.2f} MB")
    print(f"Model size : {model_size:.2f} MB")


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph():

    print_section(
        "LOADING CLIENT 1 ALIGNED GRAPH"
    )

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )

    print("Graph loaded successfully.")

    print(f"Nodes    : {graph.num_nodes:,}")
    print(f"Features : {graph.num_node_features:,}")
    print(f"Edges    : {graph.num_edges:,}")
    print(f"Labels   : {graph.y.shape[0]:,}")

    return graph


# ============================================================
# VALIDATE GRAPH
# ============================================================

def validate_graph(graph):

    print_section(
        "VALIDATING ALIGNED GRAPH"
    )


    if graph.num_node_features != EXPECTED_INPUT_DIM:

        raise RuntimeError(
            f"Expected {EXPECTED_INPUT_DIM} features, "
            f"found {graph.num_node_features}."
        )

    pass_message("Shared feature dimension")


    if graph.x.shape[0] != graph.num_nodes:

        raise RuntimeError(
            "Feature row count does not match node count."
        )

    pass_message("Feature row count")


    if graph.y.shape[0] != graph.num_nodes:

        raise RuntimeError(
            "Label count does not match node count."
        )

    pass_message("Label count")


    if torch.isnan(graph.x).any():

        raise RuntimeError(
            "NaN values detected in graph features."
        )

    pass_message("Feature NaN check")


    if torch.isinf(graph.x).any():

        raise RuntimeError(
            "Infinite values detected in graph features."
        )

    pass_message("Feature infinity check")


    if graph.edge_index.ndim != 2:

        raise RuntimeError(
            "edge_index must be 2-dimensional."
        )


    if graph.edge_index.shape[0] != 2:

        raise RuntimeError(
            "Invalid edge_index shape."
        )

    pass_message("Edge index shape")


    if graph.edge_index.numel() > 0:

        minimum = int(
            graph.edge_index.min().item()
        )

        maximum = int(
            graph.edge_index.max().item()
        )

        if minimum < 0:

            raise RuntimeError(
                "Negative node index detected."
            )

        if maximum >= graph.num_nodes:

            raise RuntimeError(
                "Edge index exceeds node count."
            )

    pass_message("Edge bounds")


    unique_labels = sorted(
        torch.unique(
            graph.y
        ).cpu().tolist()
    )

    print(
        f"Classes: {unique_labels}"
    )

    if not set(unique_labels).issubset(
        {0, 1}
    ):

        raise RuntimeError(
            "Labels are not binary."
        )

    pass_message("Binary fraud labels")


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint():

    print_section(
        "LOADING CLIENT 1 BEST MODEL"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    print(
        "Best model checkpoint loaded successfully."
    )


    if not isinstance(
        checkpoint,
        dict
    ):

        raise RuntimeError(
            "Saved model is not a checkpoint dictionary."
        )

    pass_message("Checkpoint structure")


    required_keys = [
        "model_state_dict",
        "input_dim",
        "hidden_dim",
        "output_dim",
        "dropout",
    ]


    for key in required_keys:

        if key not in checkpoint:

            raise KeyError(
                f"Required checkpoint key missing: {key}"
            )

        print(
            f"{key}: PRESENT"
        )


    return checkpoint


# ============================================================
# VALIDATE CHECKPOINT CONFIG
# ============================================================

def validate_checkpoint_config(
    checkpoint
):

    print_section(
        "VALIDATING SAVED MODEL CONFIGURATION"
    )


    input_dim = int(
        checkpoint["input_dim"]
    )

    hidden_dim = int(
        checkpoint["hidden_dim"]
    )

    output_dim = int(
        checkpoint["output_dim"]
    )

    dropout = float(
        checkpoint["dropout"]
    )


    print(
        f"Input dimension  : {input_dim}"
    )

    print(
        f"Hidden dimension : {hidden_dim}"
    )

    print(
        f"Output dimension : {output_dim}"
    )

    print(
        f"Dropout          : {dropout}"
    )


    if input_dim != EXPECTED_INPUT_DIM:

        raise RuntimeError(
            "Saved model input dimension "
            "does not match shared feature dimension."
        )

    pass_message("Input dimension")


    if hidden_dim != EXPECTED_HIDDEN_DIM:

        raise RuntimeError(
            "Unexpected hidden dimension."
        )

    pass_message("Hidden dimension")


    if output_dim != EXPECTED_OUTPUT_DIM:

        raise RuntimeError(
            "Unexpected output dimension."
        )

    pass_message("Output dimension")


    if "best_epoch" in checkpoint:

        print(
            f"Best epoch       : "
            f"{checkpoint['best_epoch']}"
        )


    if "best_val_loss" in checkpoint:

        print(
            f"Best val loss    : "
            f"{checkpoint['best_val_loss']:.6f}"
        )


    return (
        input_dim,
        hidden_dim,
        output_dim,
        dropout,
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_and_load_model(
    checkpoint,
    input_dim,
    hidden_dim,
    output_dim,
    dropout,
):

    print_section(
        "RECONSTRUCTING GRAPHSAGE MODEL"
    )


    model = GraphSAGE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
    )


    print(
        "GraphSAGE model reconstructed."
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    pass_message(
        "Model state loading"
    )


    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )


    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


    print()
    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )


    if total_parameters != 230146:

        print(
            "WARNING: Parameter count differs "
            "from the expected Client 1 model."
        )

    else:

        pass_message(
            "Expected parameter count"
        )


    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    labels,
):

    predictions = (
        predictions
        .detach()
        .cpu()
    )

    labels = (
        labels
        .detach()
        .cpu()
    )


    tp = int(
        (
            (predictions == 1)
            &
            (labels == 1)
        )
        .sum()
        .item()
    )


    tn = int(
        (
            (predictions == 0)
            &
            (labels == 0)
        )
        .sum()
        .item()
    )


    fp = int(
        (
            (predictions == 1)
            &
            (labels == 0)
        )
        .sum()
        .item()
    )


    fn = int(
        (
            (predictions == 0)
            &
            (labels == 1)
        )
        .sum()
        .item()
    )


    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )


    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )


    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


    accuracy = (
        (tp + tn)
        /
        (tp + tn + fp + fn)
    )


    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )


    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
    }


# ============================================================
# RUN INFERENCE
# ============================================================

@torch.no_grad()
def run_inference(
    model,
    graph,
    device,
):

    print_section(
        "RUNNING TRAINED MODEL FORWARD PASS"
    )


    model.eval()


    start_time = time.time()


    logits = model(
        graph.x,
        graph.edge_index,
    )


    elapsed = (
        time.time()
        - start_time
    )


    print(
        "Forward pass: COMPLETED"
    )

    print(
        f"Inference time: {elapsed:.2f}s"
    )


    expected_shape = (
        graph.num_nodes,
        EXPECTED_OUTPUT_DIM,
    )


    print(
        f"Output shape: "
        f"{tuple(logits.shape)}"
    )


    if tuple(
        logits.shape
    ) != expected_shape:

        raise RuntimeError(
            f"Unexpected output shape. "
            f"Expected {expected_shape}, "
            f"found {tuple(logits.shape)}."
        )

    pass_message(
        "Output shape"
    )


    nan_count = int(
        torch.isnan(
            logits
        ).sum().item()
    )


    inf_count = int(
        torch.isinf(
            logits
        ).sum().item()
    )


    print(
        f"Output NaN values      : {nan_count}"
    )

    print(
        f"Output infinity values : {inf_count}"
    )


    if nan_count != 0:

        raise RuntimeError(
            "NaN values detected in trained model output."
        )

    pass_message(
        "Output NaN check"
    )


    if inf_count != 0:

        raise RuntimeError(
            "Infinite values detected in trained model output."
        )

    pass_message(
        "Output infinity check"
    )


    return logits


# ============================================================
# PROBABILITY VALIDATION
# ============================================================

def validate_probabilities(
    logits
):

    print_section(
        "VALIDATING FRAUD PROBABILITIES"
    )


    probabilities = F.softmax(
        logits,
        dim=1,
    )


    if torch.isnan(
        probabilities
    ).any():

        raise RuntimeError(
            "NaN detected in probabilities."
        )

    pass_message(
        "Probability NaN check"
    )


    if torch.isinf(
        probabilities
    ).any():

        raise RuntimeError(
            "Infinity detected in probabilities."
        )

    pass_message(
        "Probability infinity check"
    )


    row_sums = probabilities.sum(
        dim=1
    )


    if not torch.allclose(
        row_sums,
        torch.ones_like(
            row_sums
        ),
        atol=1e-5,
    ):

        raise RuntimeError(
            "Probability rows do not sum to 1."
        )

    pass_message(
        "Probability normalization"
    )


    fraud_probabilities = (
        probabilities[:, 1]
    )


    print()

    print(
        "Fraud probability statistics:"
    )

    print(
        f"  Minimum : "
        f"{fraud_probabilities.min().item():.6f}"
    )

    print(
        f"  Maximum : "
        f"{fraud_probabilities.max().item():.6f}"
    )

    print(
        f"  Mean    : "
        f"{fraud_probabilities.mean().item():.6f}"
    )

    print(
        f"  Std     : "
        f"{fraud_probabilities.std().item():.6f}"
    )


    return probabilities


# ============================================================
# EVALUATE
# ============================================================

def evaluate_predictions(
    probabilities,
    labels,
):

    print_section(
        "EVALUATING TRAINED CLIENT 1 MODEL"
    )


    predictions = probabilities.argmax(
        dim=1
    )


    unique_predictions = sorted(
        torch.unique(
            predictions
        )
        .detach()
        .cpu()
        .tolist()
    )


    print(
        f"Prediction classes: "
        f"{unique_predictions}"
    )


    if not set(
        unique_predictions
    ).issubset({0, 1}):

        raise RuntimeError(
            "Invalid prediction classes."
        )

    pass_message(
        "Prediction classes"
    )


    metrics = calculate_metrics(
        predictions,
        labels,
    )


    predicted_legitimate = int(
        (predictions == 0)
        .sum()
        .item()
    )

    predicted_fraud = int(
        (predictions == 1)
        .sum()
        .item()
    )


    actual_legitimate = int(
        (labels == 0)
        .sum()
        .item()
    )

    actual_fraud = int(
        (labels == 1)
        .sum()
        .item()
    )


    print()

    print(
        "Prediction distribution:"
    )

    print(
        f"  Predicted legitimate : "
        f"{predicted_legitimate:,}"
    )

    print(
        f"  Predicted fraud      : "
        f"{predicted_fraud:,}"
    )


    print()

    print(
        "Actual distribution:"
    )

    print(
        f"  Actual legitimate    : "
        f"{actual_legitimate:,}"
    )

    print(
        f"  Actual fraud         : "
        f"{actual_fraud:,}"
    )


    print()

    print(
        "Confusion matrix:"
    )

    print(
        f"  True Negative  : "
        f"{metrics['tn']:,}"
    )

    print(
        f"  False Positive : "
        f"{metrics['fp']:,}"
    )

    print(
        f"  False Negative : "
        f"{metrics['fn']:,}"
    )

    print(
        f"  True Positive  : "
        f"{metrics['tp']:,}"
    )


    print()

    print(
        "Metrics:"
    )

    print(
        f"  Accuracy    : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"  Precision   : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"  Recall      : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"  F1 Score    : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"  Specificity : "
        f"{metrics['specificity']:.4f}"
    )


    return metrics


# ============================================================
# PARAMETER VALIDATION
# ============================================================

def validate_model_parameters(
    model
):

    print_section(
        "VALIDATING TRAINED MODEL PARAMETERS"
    )


    total_values = 0


    for name, parameter in (
        model.named_parameters()
    ):

        total_values += (
            parameter.numel()
        )


        if torch.isnan(
            parameter
        ).any():

            raise RuntimeError(
                f"NaN parameter detected: {name}"
            )


        if torch.isinf(
            parameter
        ).any():

            raise RuntimeError(
                f"Infinite parameter detected: {name}"
            )


    print(
        f"Parameter values checked: "
        f"{total_values:,}"
    )

    pass_message(
        "Parameter NaN check"
    )

    pass_message(
        "Parameter infinity check"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "RFGN CLIENT 1 TRAINED GRAPHSAGE MODEL VALIDATION"
    )

    print("=" * 70)

    print()

    print(
        "READ-ONLY VALIDATION"
    )

    print(
        "No model training will be performed."
    )

    print(
        "No model parameters will be updated."
    )

    print(
        "No validation/test CSV dataset will be used."
    )


    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    check_files()


    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    graph = load_graph()

    validate_graph(
        graph
    )


    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint = load_checkpoint()


    (
        input_dim,
        hidden_dim,
        output_dim,
        dropout,
    ) = validate_checkpoint_config(
        checkpoint
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_and_load_model(
        checkpoint,
        input_dim,
        hidden_dim,
        output_dim,
        dropout,
    )


    validate_model_parameters(
        model
    )


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print_section(
        "SELECTING DEVICE"
    )


    device = get_device()


    print(
        f"Device: {device}"
    )


    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )


    # --------------------------------------------------------
    # Transfer
    # --------------------------------------------------------

    print_section(
        "MOVING MODEL AND GRAPH TO DEVICE"
    )


    model = model.to(
        device
    )

    graph = graph.to(
        device
    )


    pass_message(
        "Model transfer"
    )

    pass_message(
        "Graph transfer"
    )


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    logits = run_inference(
        model,
        graph,
        device,
    )


    probabilities = (
        validate_probabilities(
            logits
        )
    )


    metrics = evaluate_predictions(
        probabilities,
        graph.y,
    )


    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if device.type == "cuda":

        print_section(
            "GPU MEMORY"
        )


        allocated = (
            torch.cuda.memory_allocated()
            /
            (1024 ** 3)
        )


        reserved = (
            torch.cuda.memory_reserved()
            /
            (1024 ** 3)
        )


        print(
            f"Allocated : "
            f"{allocated:.3f} GB"
        )

        print(
            f"Reserved  : "
            f"{reserved:.3f} GB"
        )


    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "CLIENT 1 TRAINED MODEL VALIDATION: PASSED"
    )

    print("=" * 70)


    print()

    print(
        f"Graph nodes      : "
        f"{graph.num_nodes:,}"
    )

    print(
        f"Input features   : "
        f"{graph.num_node_features:,}"
    )

    print(
        f"Graph edges      : "
        f"{graph.num_edges:,}"
    )

    print(
        f"Accuracy         : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision        : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall           : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score         : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Device           : "
        f"{device}"
    )


    print()

    print(
        "Saved Client 1 GraphSAGE model "
        "loads and performs inference correctly."
    )

    print()

    print(
        "TRAINING WAS NOT PERFORMED BY THIS TEST."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Train Client 2 using its aligned graph."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()