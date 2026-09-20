# ============================================================
# RFGN CLIENT 3 NORMALIZED GRAPHSAGE FORWARD-PASS TEST
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
# IMPORT MODEL
# ============================================================

from models.gnn.graphsage_model import GraphSAGE


# ============================================================
# CONFIGURATION
# ============================================================

CLIENT_ID = "client_3"

INPUT_DIM = 770
HIDDEN_DIM = 128
OUTPUT_DIM = 2
DROPOUT = 0.30


# ============================================================
# PATH
# ============================================================

GRAPH_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs",
    CLIENT_ID,
    "graph_normalized.pt",
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "RFGN CLIENT 3 NORMALIZED GRAPHSAGE FORWARD-PASS TEST"
    )
    print("=" * 70)

    print()
    print("This test performs:")
    print("  Normalized graph loading")
    print("  Graph validation")
    print("  Model creation")
    print("  Forward pass")
    print("  Output validation")

    print()
    print("No training will be performed.")
    print("No model parameters will be updated.")
    print("No validation/test dataset will be used.")

    # ========================================================
    # CHECK GRAPH
    # ========================================================

    print()
    print("=" * 70)
    print("CHECKING NORMALIZED CLIENT 3 GRAPH")
    print("=" * 70)

    if not os.path.exists(GRAPH_PATH):

        raise FileNotFoundError(
            f"Normalized graph not found:\n{GRAPH_PATH}"
        )

    print("Graph file: PRESENT")
    print("Path:", GRAPH_PATH)

    # ========================================================
    # LOAD GRAPH
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING NORMALIZED GRAPH")
    print("=" * 70)

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
        f"Features : {graph.num_node_features:,}"
    )

    print(
        f"Edges    : {graph.num_edges:,}"
    )

    print(
        f"Labels   : {graph.y.shape[0]:,}"
    )

    # ========================================================
    # VALIDATE DIMENSIONS
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATING NORMALIZED GRAPH DIMENSIONS")
    print("=" * 70)

    if graph.num_node_features != INPUT_DIM:

        raise ValueError(
            f"Expected {INPUT_DIM} input features, "
            f"found {graph.num_node_features}."
        )

    print(
        "Input feature dimension: PASS"
    )

    if graph.x.shape[0] != graph.num_nodes:

        raise ValueError(
            "Feature row count does not match node count."
        )

    print(
        "Feature row count: PASS"
    )

    if graph.y.shape[0] != graph.num_nodes:

        raise ValueError(
            "Label count does not match node count."
        )

    print(
        "Label count: PASS"
    )

    if graph.edge_index.shape[0] != 2:

        raise ValueError(
            "Edge index must have shape (2, E)."
        )

    print(
        "Edge index shape: PASS"
    )

    # ========================================================
    # VALIDATE FEATURES
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATING NORMALIZED FEATURES")
    print("=" * 70)

    nan_count = int(
        torch.isnan(
            graph.x
        ).sum().item()
    )

    inf_count = int(
        torch.isinf(
            graph.x
        ).sum().item()
    )

    print(
        f"NaN values      : {nan_count:,}"
    )

    print(
        f"Infinite values : {inf_count:,}"
    )

    if nan_count != 0:

        raise ValueError(
            "Normalized graph contains NaN values."
        )

    print(
        "Feature NaN check: PASS"
    )

    if inf_count != 0:

        raise ValueError(
            "Normalized graph contains infinity."
        )

    print(
        "Feature infinity check: PASS"
    )

    # ========================================================
    # VALIDATE EDGES
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATING GRAPH EDGES")
    print("=" * 70)

    if graph.edge_index.numel() == 0:

        raise ValueError(
            "Graph contains no edges."
        )

    minimum_index = int(
        graph.edge_index.min().item()
    )

    maximum_index = int(
        graph.edge_index.max().item()
    )

    print(
        f"Minimum node index: {minimum_index}"
    )

    print(
        f"Maximum node index: {maximum_index}"
    )

    if minimum_index < 0:

        raise ValueError(
            "Negative node index detected."
        )

    if maximum_index >= graph.num_nodes:

        raise ValueError(
            "Edge index exceeds node count."
        )

    print(
        "Edge bounds: PASS"
    )

    # ========================================================
    # VALIDATE LABELS
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATING LABELS")
    print("=" * 70)

    unique_labels = torch.unique(
        graph.y
    ).tolist()

    print(
        f"Classes: {unique_labels}"
    )

    if not set(
        unique_labels
    ).issubset({0, 1}):

        raise ValueError(
            f"Unexpected label classes: {unique_labels}"
        )

    if len(unique_labels) != 2:

        raise ValueError(
            "Expected both legitimate and fraud classes."
        )

    print(
        "Binary fraud labels: PASS"
    )

    # ========================================================
    # DEVICE
    # ========================================================

    print()
    print("=" * 70)
    print("SELECTING DEVICE")
    print("=" * 70)

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("CREATING GRAPHSAGE MODEL")
    print("=" * 70)

    model = GraphSAGE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    )

    model.eval()

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "GraphSAGE model: CREATED"
    )

    print()
    print(
        "Model configuration:"
    )

    print(
        f"input_dim             : {INPUT_DIM}"
    )

    print(
        f"hidden_dim            : {HIDDEN_DIM}"
    )

    print(
        f"output_dim            : {OUTPUT_DIM}"
    )

    print(
        f"dropout               : {DROPOUT}"
    )

    print(
        f"total_parameters      : {total_parameters:,}"
    )

    print(
        f"trainable_parameters  : {trainable_parameters:,}"
    )

    if INPUT_DIM != graph.num_node_features:

        raise ValueError(
            "Model input dimension does not match graph."
        )

    print(
        "Model input dimension: PASS"
    )

    if OUTPUT_DIM != 2:

        raise ValueError(
            "Output dimension must be 2."
        )

    print(
        "Model output dimension: PASS"
    )

    # ========================================================
    # MOVE TO DEVICE
    # ========================================================

    print()
    print("=" * 70)
    print("MOVING MODEL AND GRAPH TO DEVICE")
    print("=" * 70)

    try:

        model = model.to(device)

        print(
            "Model transfer: PASS"
        )

        graph = graph.to(device)

        print(
            "Graph transfer: PASS"
        )

    except RuntimeError as error:

        if "out of memory" in str(
            error
        ).lower():

            print()
            print(
                "CUDA OUT OF MEMORY."
            )

            print(
                "Try closing other GPU applications "
                "and run the test again."
            )

            return

        raise

    # ========================================================
    # FORWARD PASS
    # ========================================================

    print()
    print("=" * 70)
    print("RUNNING GRAPHSAGE FORWARD PASS")
    print("=" * 70)

    model.eval()

    with torch.no_grad():

        logits = model(
            graph.x,
            graph.edge_index,
        )

    print(
        "Forward pass: COMPLETED"
    )

    # ========================================================
    # OUTPUT VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATING MODEL OUTPUT")
    print("=" * 70)

    expected_shape = (
        graph.num_nodes,
        OUTPUT_DIM,
    )

    print(
        f"Output shape: {tuple(logits.shape)}"
    )

    if tuple(
        logits.shape
    ) != expected_shape:

        raise ValueError(
            f"Expected output shape "
            f"{expected_shape}, "
            f"found {tuple(logits.shape)}."
        )

    print(
        "Output shape: PASS"
    )

    output_nan = int(
        torch.isnan(
            logits
        ).sum().item()
    )

    output_inf = int(
        torch.isinf(
            logits
        ).sum().item()
    )

    print(
        f"Output NaN values      : "
        f"{output_nan:,}"
    )

    print(
        f"Output infinity values : "
        f"{output_inf:,}"
    )

    if output_nan != 0:

        raise ValueError(
            "Model output contains NaN."
        )

    print(
        "Output NaN check: PASS"
    )

    if output_inf != 0:

        raise ValueError(
            "Model output contains infinity."
        )

    print(
        "Output infinity check: PASS"
    )

    # ========================================================
    # PROBABILITIES
    # ========================================================

    print()
    print("=" * 70)
    print("TESTING OUTPUT PROBABILITIES")
    print("=" * 70)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    print(
        f"Probability tensor shape: "
        f"{tuple(probabilities.shape)}"
    )

    if tuple(
        probabilities.shape
    ) != expected_shape:

        raise ValueError(
            "Probability tensor shape mismatch."
        )

    print(
        "Probability shape: PASS"
    )

    probability_nan = int(
        torch.isnan(
            probabilities
        ).sum().item()
    )

    probability_inf = int(
        torch.isinf(
            probabilities
        ).sum().item()
    )

    if probability_nan != 0:

        raise ValueError(
            "Probability tensor contains NaN."
        )

    if probability_inf != 0:

        raise ValueError(
            "Probability tensor contains infinity."
        )

    print(
        "Probability NaN/Inf check: PASS"
    )

    probability_sums = (
        probabilities.sum(
            dim=1
        )
    )

    normalization_error = torch.max(
        torch.abs(
            probability_sums - 1.0
        )
    ).item()

    if normalization_error > 1e-5:

        raise ValueError(
            "Probability normalization failed."
        )

    print(
        "Probability normalization: PASS"
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATING PREDICTIONS")
    print("=" * 70)

    predictions = torch.argmax(
        probabilities,
        dim=1,
    )

    prediction_count = (
        predictions.shape[0]
    )

    print(
        f"Prediction count: "
        f"{prediction_count:,}"
    )

    if prediction_count != graph.num_nodes:

        raise ValueError(
            "Prediction count does not match nodes."
        )

    print(
        "Prediction count: PASS"
    )

    prediction_classes = torch.unique(
        predictions
    ).tolist()

    print(
        f"Prediction classes: "
        f"{prediction_classes}"
    )

    if not set(
        prediction_classes
    ).issubset({0, 1}):

        raise ValueError(
            "Invalid prediction classes."
        )

    print(
        "Prediction classes: PASS"
    )

    # ========================================================
    # OUTPUT STATISTICS
    # ========================================================

    print()
    print("=" * 70)
    print("OUTPUT STATISTICS")
    print("=" * 70)

    print(
        f"Logit minimum : "
        f"{logits.min().item():.6f}"
    )

    print(
        f"Logit maximum : "
        f"{logits.max().item():.6f}"
    )

    print(
        f"Logit mean    : "
        f"{logits.mean().item():.6f}"
    )

    print(
        f"Logit std     : "
        f"{logits.std(unbiased=False).item():.6f}"
    )

    prediction_0 = int(
        (predictions == 0)
        .sum()
        .item()
    )

    prediction_1 = int(
        (predictions == 1)
        .sum()
        .item()
    )

    print(
        f"Prediction 0  : "
        f"{prediction_0:,}"
    )

    print(
        f"Prediction 1  : "
        f"{prediction_1:,}"
    )

    # ========================================================
    # GPU MEMORY
    # ========================================================

    if device.type == "cuda":

        allocated = (
            torch.cuda.memory_allocated(0)
            / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved(0)
            / (1024 ** 3)
        )

        print()
        print("GPU memory:")

        print(
            f"  Allocated : "
            f"{allocated:.3f} GB"
        )

        print(
            f"  Reserved  : "
            f"{reserved:.3f} GB"
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "CLIENT 3 NORMALIZED GRAPHSAGE "
        "FORWARD-PASS TEST: PASSED"
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
        f"Output classes   : "
        f"{OUTPUT_DIM}"
    )

    print(
        f"Output shape     : "
        f"{tuple(logits.shape)}"
    )

    print(
        f"Device           : "
        f"{device}"
    )

    print()

    print(
        "Normalized GraphSAGE forward pipeline "
        "is working correctly."
    )

    print()
    print(
        "TRAINING HAS NOT STARTED."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()