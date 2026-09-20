import os
import sys

import torch


# ============================================================
# PROJECT PATH
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


from models.gnn.graphsage_model import (
    GraphSAGE,
    model_summary,
)


# ============================================================
# CONFIGURATION
# ============================================================

GRAPH_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs",
    "client_1",
    "graph.pt",
)

EXPECTED_NODES = 136987
EXPECTED_FEATURES = 814
EXPECTED_CLASSES = 2


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RFGN CLIENT 1 GRAPHSAGE FORWARD-PASS TEST")
    print("=" * 70)

    print()
    print("This test performs:")
    print("  Graph loading")
    print("  Model creation")
    print("  Forward pass")
    print("  Output validation")

    print()
    print("No training will be performed.")
    print("No validation/test data will be used.")
    print("No model parameters will be updated.")

    # ========================================================
    # CHECK GRAPH
    # ========================================================

    print()
    print("=" * 70)
    print("CHECKING CLIENT 1 GRAPH")
    print("=" * 70)

    if not os.path.exists(
        GRAPH_PATH
    ):

        raise FileNotFoundError(
            f"Graph not found:\n{GRAPH_PATH}"
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
    print("LOADING CLIENT 1 GRAPH")
    print("=" * 70)

    data = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )

    print(
        "Graph loaded successfully."
    )

    print(
        "Nodes:",
        f"{data.num_nodes:,}"
    )

    print(
        "Features:",
        data.num_node_features
    )

    print(
        "Edges:",
        f"{data.num_edges:,}"
    )

    print(
        "Labels:",
        data.y.shape[0]
    )

    # ========================================================
    # GRAPH VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATING GRAPH DIMENSIONS")
    print("=" * 70)

    assert (
        data.num_nodes
        == EXPECTED_NODES
    )

    print(
        "Node count: PASS"
    )

    assert (
        data.num_node_features
        == EXPECTED_FEATURES
    )

    print(
        "Input feature dimension: PASS"
    )

    assert (
        data.y.shape[0]
        == EXPECTED_NODES
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

    # ========================================================
    # DEVICE
    # ========================================================

    print()
    print("=" * 70)
    print("SELECTING DEVICE")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
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
        input_dim=EXPECTED_FEATURES,
        hidden_dim=128,
        output_dim=EXPECTED_CLASSES,
        dropout=0.30,
    )

    print(
        "GraphSAGE model: CREATED"
    )

    summary = model_summary(
        model
    )

    print()
    print("Model configuration:")

    for key, value in summary.items():

        print(
            f"{key:<22}: {value}"
        )

    # ========================================================
    # MODEL VALIDATION
    # ========================================================

    assert (
        model.input_dim
        == EXPECTED_FEATURES
    )

    print()
    print(
        "Model input dimension: PASS"
    )

    assert (
        model.output_dim
        == EXPECTED_CLASSES
    )

    print(
        "Model output dimension: PASS"
    )

    # ========================================================
    # MOVE TO DEVICE
    # ========================================================

    data = data.to(
        device
    )

    model = model.to(
        device
    )

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
            data.x,
            data.edge_index,
        )

    print(
        "Forward pass: COMPLETED"
    )

    print(
        "Output shape:",
        tuple(logits.shape)
    )

    # ========================================================
    # OUTPUT VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATING MODEL OUTPUT")
    print("=" * 70)

    expected_shape = (
        EXPECTED_NODES,
        EXPECTED_CLASSES,
    )

    assert (
        tuple(logits.shape)
        == expected_shape
    )

    print(
        "Output shape: PASS"
    )

    assert not torch.isnan(
        logits
    ).any()

    print(
        "Output NaN check: PASS"
    )

    assert not torch.isinf(
        logits
    ).any()

    print(
        "Output infinity check: PASS"
    )

    # ========================================================
    # SOFTMAX PROBABILITY CHECK
    # ========================================================

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    print()
    print(
        "Probability tensor shape:",
        tuple(probabilities.shape)
    )

    assert (
        tuple(probabilities.shape)
        == expected_shape
    )

    print(
        "Probability shape: PASS"
    )

    probability_sums = (
        probabilities.sum(
            dim=1
        )
    )

    assert torch.allclose(
        probability_sums,
        torch.ones_like(
            probability_sums
        ),
        atol=1e-5,
    )

    print(
        "Probability normalization: PASS"
    )

    # ========================================================
    # PREDICTION CHECK
    # ========================================================

    predictions = (
        torch.argmax(
            probabilities,
            dim=1,
        )
    )

    assert (
        predictions.shape[0]
        == EXPECTED_NODES
    )

    print(
        "Prediction count: PASS"
    )

    unique_predictions = (
        torch.unique(
            predictions
        )
        .cpu()
        .tolist()
    )

    print(
        "Prediction classes:",
        unique_predictions
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("CLIENT 1 GRAPHSAGE FORWARD-PASS TEST: PASSED")
    print("=" * 70)

    print()
    print(
        "Graph nodes:",
        f"{data.num_nodes:,}"
    )

    print(
        "Input features:",
        data.num_node_features
    )

    print(
        "Output classes:",
        EXPECTED_CLASSES
    )

    print(
        "Output shape:",
        tuple(logits.shape)
    )

    print(
        "Device:",
        device
    )

    print()
    print(
        "GraphSAGE forward pipeline is working correctly."
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