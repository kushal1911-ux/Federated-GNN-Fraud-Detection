# ============================================================
# RFGN CLIENT 1 NORMALIZED GRAPHSAGE FORWARD-PASS TEST
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
# PATH
# ============================================================

GRAPH_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs",
    "client_1",
    "graph_normalized.pt",
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

INPUT_DIM = 770
HIDDEN_DIM = 128
OUTPUT_DIM = 2
DROPOUT = 0.30


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RFGN CLIENT 1 NORMALIZED GRAPHSAGE FORWARD-PASS TEST")
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
    print("CHECKING NORMALIZED CLIENT 1 GRAPH")
    print("=" * 70)

    if not os.path.exists(GRAPH_PATH):

        raise FileNotFoundError(
            f"Normalized graph not found:\n{GRAPH_PATH}"
        )

    print(
        "Graph file: PRESENT"
    )

    print(
        "Path:",
        GRAPH_PATH
    )

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

    print(
        "Graph loaded successfully."
    )

    print(
        f"Nodes    : {graph.num_nodes:,}"
    )

    print(
        f"Features : {graph.num_node_features}"
    )

    print(
        f"Edges    : {graph.num_edges:,}"
    )

    print(
        f"Labels   : {graph.y.shape[0]:,}"
    )

    # ========================================================
    # VALIDATE GRAPH DIMENSIONS
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATING NORMALIZED GRAPH DIMENSIONS")
    print("=" * 70)

    if graph.num_node_features != INPUT_DIM:

        raise ValueError(
            f"Expected {INPUT_DIM} features, "
            f"but found {graph.num_node_features}."
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
    # FEATURE SAFETY
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
            "Normalized graph contains infinity values."
        )

    print(
        "Feature infinity check: PASS"
    )

    # ========================================================
    # EDGE VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATING GRAPH EDGES")
    print("=" * 70)

    edge_min = int(
        graph.edge_index.min().item()
    )

    edge_max = int(
        graph.edge_index.max().item()
    )

    print(
        f"Minimum node index: {edge_min}"
    )

    print(
        f"Maximum node index: {edge_max}"
    )

    if edge_min < 0:

        raise ValueError(
            "Negative edge index detected."
        )

    if edge_max >= graph.num_nodes:

        raise ValueError(
            "Edge index exceeds node count."
        )

    print(
        "Edge bounds: PASS"
    )

    # ========================================================
    # LABEL VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATING LABELS")
    print("=" * 70)

    unique_labels = (
        torch.unique(
            graph.y
        )
        .tolist()
    )

    print(
        "Classes:",
        unique_labels
    )

    if not set(unique_labels).issubset(
        {0, 1}
    ):

        raise ValueError(
            "Unexpected target class found."
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

        print(
            "Device: cuda"
        )

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        device = torch.device(
            "cpu"
        )

        print(
            "Device: cpu"
        )

        print(
            "CUDA is not available."
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

    print(
        "GraphSAGE model: CREATED"
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

    # ========================================================
    # MOVE MODEL + GRAPH
    # ========================================================

    print()
    print("=" * 70)
    print("MOVING MODEL AND GRAPH TO DEVICE")
    print("=" * 70)

    try:

        model = model.to(
            device
        )

        graph = graph.to(
            device
        )

        print(
            "Model transfer: PASS"
        )

        print(
            "Graph transfer: PASS"
        )

    except RuntimeError as error:

        print()
        print(
            "Device transfer failed."
        )

        print(
            "Error:",
            error
        )

        return

    # ========================================================
    # FORWARD PASS
    # ========================================================

    print()
    print("=" * 70)
    print("RUNNING GRAPHSAGE FORWARD PASS")
    print("=" * 70)

    model.eval()

    try:

        with torch.no_grad():

            logits = model(
                graph.x,
                graph.edge_index,
            )

        print(
            "Forward pass: COMPLETED"
        )

    except RuntimeError as error:

        if (
            "out of memory"
            in str(error).lower()
        ):

            print()
            print(
                "CUDA OUT OF MEMORY."
            )

            print(
                "The normalized graph still does not "
                "fit the current forward-pass configuration."
            )

            if device.type == "cuda":

                torch.cuda.empty_cache()

            return

        raise

    # ========================================================
    # OUTPUT VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATING MODEL OUTPUT")
    print("=" * 70)

    print(
        "Output shape:",
        tuple(logits.shape)
    )

    expected_shape = (
        graph.num_nodes,
        OUTPUT_DIM,
    )

    if tuple(logits.shape) != expected_shape:

        raise ValueError(
            f"Expected output shape "
            f"{expected_shape}, "
            f"got {tuple(logits.shape)}."
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
            "Model output contains NaN values."
        )

    print(
        "Output NaN check: PASS"
    )

    if output_inf != 0:

        raise ValueError(
            "Model output contains infinity values."
        )

    print(
        "Output infinity check: PASS"
    )

    # ========================================================
    # PROBABILITY TEST
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
        "Probability tensor shape:",
        tuple(probabilities.shape)
    )

    if tuple(
        probabilities.shape
    ) != expected_shape:

        raise ValueError(
            "Probability tensor shape is incorrect."
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

    # --------------------------------------------------------
    # Probability normalization
    # --------------------------------------------------------

    probability_sums = (
        probabilities.sum(
            dim=1
        )
    )

    expected_ones = torch.ones_like(
        probability_sums
    )

    if not torch.allclose(
        probability_sums,
        expected_ones,
        atol=1e-5,
    ):

        raise ValueError(
            "Probability rows do not sum to 1."
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

    if (
        predictions.shape[0]
        != graph.num_nodes
    ):

        raise ValueError(
            "Prediction count does not match node count."
        )

    print(
        "Prediction count: PASS"
    )

    prediction_classes = (
        torch.unique(
            predictions
        )
        .detach()
        .cpu()
        .tolist()
    )

    print(
        "Prediction classes:",
        prediction_classes
    )

    if not set(
        prediction_classes
    ).issubset({0, 1}):

        raise ValueError(
            "Unexpected prediction class."
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
        f"{logits.std().item():.6f}"
    )

    print(
        f"Prediction 0  : "
        f"{int((predictions == 0).sum().item()):,}"
    )

    print(
        f"Prediction 1  : "
        f"{int((predictions == 1).sum().item()):,}"
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
        print(
            "GPU memory:"
        )

        print(
            f"  Allocated : "
            f"{allocated:.3f} GB"
        )

        print(
            f"  Reserved  : "
            f"{reserved:.3f} GB"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print(
        "CLIENT 1 NORMALIZED GRAPHSAGE "
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