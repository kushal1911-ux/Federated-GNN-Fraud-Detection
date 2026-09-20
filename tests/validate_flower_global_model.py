# ============================================================
# RFGN FLOWER GLOBAL MODEL VALIDATION
# ============================================================
#
# LEVEL 5 - FEDERATED LEARNING
# STAGE 5 - GLOBAL MODEL VALIDATION
#
# READ-ONLY
#
# No training
# No parameter updates
# No FedAvg aggregation
# No graph modification
# No CSV modification
#
# ============================================================

import os
import sys
import json

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
# IMPORT GRAPHSAGE MODEL
# ============================================================

from models.gnn.graphsage_model import GraphSAGE


# ============================================================
# PATHS
# ============================================================

GLOBAL_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "global",
    "global_graphsage_flower.pt",
)

MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "global",
    "flower_fedavg_manifest.json",
)

GRAPH_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs",
    "client_1",
    "graph_aligned.pt",
)


# ============================================================
# EXPECTED MODEL CONFIGURATION
# ============================================================

EXPECTED_INPUT_DIM = 769
EXPECTED_HIDDEN_DIM = 128
EXPECTED_OUTPUT_DIM = 2
EXPECTED_DROPOUT = 0.3

EXPECTED_PARAMETER_COUNT = 230146
EXPECTED_PARAMETER_TENSORS = 8

EXPECTED_CLIENTS = 3
EXPECTED_ROUNDS = 3


# ============================================================
# SECTION HEADER
# ============================================================

def section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)
    print("RFGN FLOWER GLOBAL MODEL VALIDATION")
    print("=" * 70)

    print()

    print(
        "LEVEL 5 - FEDERATED LEARNING"
    )

    print(
        "STAGE 5 - GLOBAL MODEL VALIDATION"
    )

    print()

    print(
        "READ-ONLY VALIDATION"
    )

    print(
        "No training will be performed."
    )

    print(
        "No model parameters will be updated."
    )

    print(
        "No FedAvg aggregation will be performed."
    )

    print(
        "No graph will be modified."
    )

    print(
        "No CSV dataset will be modified."
    )


    # ========================================================
    # CHECK GLOBAL MODEL
    # ========================================================

    section(
        "CHECKING FLOWER GLOBAL MODEL"
    )

    if not os.path.exists(
        GLOBAL_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Flower global model not found:\n"
            f"{GLOBAL_MODEL_PATH}"
        )

    model_size = (
        os.path.getsize(
            GLOBAL_MODEL_PATH
        )
        / (1024 * 1024)
    )

    print(
        "Global model file: PRESENT"
    )

    print(
        f"Path: {GLOBAL_MODEL_PATH}"
    )

    print(
        f"Size: {model_size:.2f} MB"
    )


    # ========================================================
    # LOAD GLOBAL CHECKPOINT
    # ========================================================

    section(
        "LOADING FLOWER GLOBAL CHECKPOINT"
    )

    checkpoint = torch.load(
        GLOBAL_MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    print(
        "Global checkpoint loaded successfully."
    )


    # ========================================================
    # CHECK CHECKPOINT STRUCTURE
    # ========================================================

    section(
        "VALIDATING GLOBAL CHECKPOINT STRUCTURE"
    )

    required_keys = [

        "model_state_dict",

        "input_dim",

        "hidden_dim",

        "output_dim",

        "dropout",

        "aggregation",

        "num_clients",

        "num_rounds",

        "parameter_tensors",

        "parameter_count",
    ]

    for key in required_keys:

        if key not in checkpoint:

            raise KeyError(
                f"Missing checkpoint key: {key}"
            )

        print(
            f"{key}: PRESENT"
        )

    print(
        "Checkpoint structure: PASS"
    )


    # ========================================================
    # MODEL CONFIGURATION
    # ========================================================

    section(
        "VALIDATING GLOBAL MODEL CONFIGURATION"
    )

    input_dim = checkpoint[
        "input_dim"
    ]

    hidden_dim = checkpoint[
        "hidden_dim"
    ]

    output_dim = checkpoint[
        "output_dim"
    ]

    dropout = checkpoint[
        "dropout"
    ]

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
            "Input dimension mismatch."
        )

    print(
        "Input dimension: PASS"
    )


    if hidden_dim != EXPECTED_HIDDEN_DIM:

        raise RuntimeError(
            "Hidden dimension mismatch."
        )

    print(
        "Hidden dimension: PASS"
    )


    if output_dim != EXPECTED_OUTPUT_DIM:

        raise RuntimeError(
            "Output dimension mismatch."
        )

    print(
        "Output dimension: PASS"
    )


    if abs(
        float(dropout)
        - EXPECTED_DROPOUT
    ) > 1e-9:

        raise RuntimeError(
            "Dropout mismatch."
        )

    print(
        "Dropout: PASS"
    )


    # ========================================================
    # FEDERATED METADATA
    # ========================================================

    section(
        "VALIDATING FEDERATED METADATA"
    )

    aggregation = checkpoint[
        "aggregation"
    ]

    num_clients = checkpoint[
        "num_clients"
    ]

    num_rounds = checkpoint[
        "num_rounds"
    ]

    parameter_tensors_metadata = checkpoint[
        "parameter_tensors"
    ]

    parameter_count_metadata = checkpoint[
        "parameter_count"
    ]

    print(
        f"Aggregation : {aggregation}"
    )

    print(
        f"Clients     : {num_clients}"
    )

    print(
        f"Rounds      : {num_rounds}"
    )

    print(
        f"Parameter tensors : "
        f"{parameter_tensors_metadata}"
    )

    print(
        f"Parameter count   : "
        f"{parameter_count_metadata:,}"
    )


    if aggregation != "Flower FedAvg":

        raise RuntimeError(
            "Aggregation method mismatch."
        )

    print(
        "Aggregation method: PASS"
    )


    if num_clients != EXPECTED_CLIENTS:

        raise RuntimeError(
            "Expected 3 federated clients."
        )

    print(
        "Client count: PASS"
    )


    if num_rounds != EXPECTED_ROUNDS:

        raise RuntimeError(
            "Expected 3 federated rounds."
        )

    print(
        "Round count: PASS"
    )


    # ========================================================
    # MODEL STATE DICTIONARY
    # ========================================================

    section(
        "VALIDATING GLOBAL MODEL PARAMETERS"
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    actual_parameter_tensors = len(
        state_dict
    )

    actual_parameter_count = sum(
        tensor.numel()
        for tensor in state_dict.values()
    )

    print(
        f"Parameter tensors: "
        f"{actual_parameter_tensors}"
    )

    print(
        f"Parameter values: "
        f"{actual_parameter_count:,}"
    )


    if (
        actual_parameter_tensors
        != EXPECTED_PARAMETER_TENSORS
    ):

        raise RuntimeError(
            "Parameter tensor count mismatch."
        )

    print(
        "Parameter tensor count: PASS"
    )


    if (
        actual_parameter_count
        != EXPECTED_PARAMETER_COUNT
    ):

        raise RuntimeError(
            "Parameter value count mismatch."
        )

    print(
        "Parameter count: PASS"
    )


    # ========================================================
    # PARAMETER SAFETY
    # ========================================================

    section(
        "VALIDATING GLOBAL PARAMETER SAFETY"
    )

    checked_values = 0

    for name, tensor in state_dict.items():

        if not torch.is_tensor(
            tensor
        ):

            raise RuntimeError(
                f"Invalid tensor: {name}"
            )


        if torch.is_floating_point(
            tensor
        ):

            if torch.isnan(
                tensor
            ).any():

                raise RuntimeError(
                    f"NaN detected in: {name}"
                )


            if torch.isinf(
                tensor
            ).any():

                raise RuntimeError(
                    f"Infinity detected in: {name}"
                )


        checked_values += tensor.numel()


    print(
        f"Parameter values checked: "
        f"{checked_values:,}"
    )

    print(
        "Parameter NaN check: PASS"
    )

    print(
        "Parameter infinity check: PASS"
    )


    # ========================================================
    # RECONSTRUCT MODEL
    # ========================================================

    section(
        "RECONSTRUCTING FLOWER GLOBAL GRAPHSAGE"
    )

    model = GraphSAGE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print(
        "GraphSAGE model reconstructed."
    )

    print(
        "Global model state loading: PASS"
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


    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )


    if (
        total_parameters
        != EXPECTED_PARAMETER_COUNT
    ):

        raise RuntimeError(
            "Reconstructed model parameter "
            "count mismatch."
        )


    print(
        "Expected parameter count: PASS"
    )


    # ========================================================
    # LOAD ALIGNED GRAPH
    # ========================================================

    section(
        "LOADING ALIGNED GRAPH FOR FORWARD VALIDATION"
    )

    if not os.path.exists(
        GRAPH_PATH
    ):

        raise FileNotFoundError(
            "Aligned graph not found:\n"
            f"{GRAPH_PATH}"
        )


    print(
        "Client 1 aligned graph: PRESENT"
    )

    print(
        f"Path: {GRAPH_PATH}"
    )


    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )


    print(
        "Aligned graph loaded successfully."
    )

    print(
        f"Nodes    : "
        f"{graph.num_nodes:,}"
    )

    print(
        f"Features : "
        f"{graph.num_node_features}"
    )

    print(
        f"Edges    : "
        f"{graph.num_edges:,}"
    )


    # ========================================================
    # GRAPH VALIDATION
    # ========================================================

    section(
        "VALIDATING ALIGNED GRAPH"
    )


    if (
        graph.num_node_features
        != EXPECTED_INPUT_DIM
    ):

        raise RuntimeError(
            "Graph feature dimension mismatch."
        )

    print(
        "Feature dimension: PASS"
    )


    if (
        graph.x.shape[0]
        != graph.num_nodes
    ):

        raise RuntimeError(
            "Feature row count mismatch."
        )

    print(
        "Feature row count: PASS"
    )


    if (
        graph.y.shape[0]
        != graph.num_nodes
    ):

        raise RuntimeError(
            "Label count mismatch."
        )

    print(
        "Label count: PASS"
    )


    if graph.x.isnan().any():

        raise RuntimeError(
            "NaN detected in graph features."
        )

    print(
        "Feature NaN check: PASS"
    )


    if graph.x.isinf().any():

        raise RuntimeError(
            "Infinity detected in graph features."
        )

    print(
        "Feature infinity check: PASS"
    )


    if graph.edge_index.shape[0] != 2:

        raise RuntimeError(
            "Invalid edge index shape."
        )

    print(
        "Edge index shape: PASS"
    )


    minimum_index = int(
        graph.edge_index.min()
    )

    maximum_index = int(
        graph.edge_index.max()
    )


    print(
        f"Minimum node index: "
        f"{minimum_index}"
    )

    print(
        f"Maximum node index: "
        f"{maximum_index}"
    )


    if minimum_index < 0:

        raise RuntimeError(
            "Negative node index detected."
        )


    if maximum_index >= graph.num_nodes:

        raise RuntimeError(
            "Out-of-range node index detected."
        )


    print(
        "Edge bounds: PASS"
    )


    classes = sorted(
        graph.y.unique().tolist()
    )


    print(
        f"Classes: {classes}"
    )


    if classes != [0, 1]:

        raise RuntimeError(
            "Labels are not binary."
        )


    print(
        "Binary fraud labels: PASS"
    )


    # ========================================================
    # FORWARD PASS
    # ========================================================

    section(
        "RUNNING FLOWER GLOBAL GRAPHSAGE FORWARD PASS"
    )


    with torch.no_grad():

        logits = model(
            graph.x,
            graph.edge_index,
        )


    print(
        "Forward pass: COMPLETED"
    )

    print(
        f"Output shape: "
        f"{tuple(logits.shape)}"
    )


    expected_shape = (
        graph.num_nodes,
        EXPECTED_OUTPUT_DIM,
    )


    if (
        tuple(logits.shape)
        != expected_shape
    ):

        raise RuntimeError(
            "Output shape mismatch."
        )


    print(
        "Output shape: PASS"
    )


    # ========================================================
    # OUTPUT SAFETY
    # ========================================================

    section(
        "VALIDATING GLOBAL MODEL OUTPUT"
    )


    nan_count = int(
        torch.isnan(logits).sum()
    )

    infinity_count = int(
        torch.isinf(logits).sum()
    )


    print(
        f"Output NaN values      : "
        f"{nan_count}"
    )

    print(
        f"Output infinity values : "
        f"{infinity_count}"
    )


    if nan_count != 0:

        raise RuntimeError(
            "NaN detected in global output."
        )


    if infinity_count != 0:

        raise RuntimeError(
            "Infinity detected in global output."
        )


    print(
        "Output NaN check: PASS"
    )

    print(
        "Output infinity check: PASS"
    )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    section(
        "VALIDATING GLOBAL FRAUD PROBABILITIES"
    )


    probabilities = torch.softmax(
        logits,
        dim=1,
    )


    if torch.isnan(
        probabilities
    ).any():

        raise RuntimeError(
            "NaN detected in probabilities."
        )


    if torch.isinf(
        probabilities
    ).any():

        raise RuntimeError(
            "Infinity detected in probabilities."
        )


    probability_sums = (
        probabilities.sum(
            dim=1
        )
    )


    if not torch.allclose(
        probability_sums,
        torch.ones_like(
            probability_sums
        ),
        atol=1e-5,
    ):

        raise RuntimeError(
            "Probability normalization failed."
        )


    print(
        "Probability NaN check: PASS"
    )

    print(
        "Probability infinity check: PASS"
    )

    print(
        "Probability normalization: PASS"
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


    # ========================================================
    # PREDICTIONS
    # ========================================================

    section(
        "VALIDATING GLOBAL PREDICTIONS"
    )


    predictions = torch.argmax(
        probabilities,
        dim=1,
    )


    if (
        predictions.shape[0]
        != graph.num_nodes
    ):

        raise RuntimeError(
            "Prediction count mismatch."
        )


    print(
        f"Prediction count: "
        f"{predictions.shape[0]:,}"
    )

    print(
        "Prediction count: PASS"
    )


    prediction_classes = sorted(
        predictions.unique().tolist()
    )


    print(
        f"Prediction classes: "
        f"{prediction_classes}"
    )


    if not all(
        value in [0, 1]
        for value in prediction_classes
    ):

        raise RuntimeError(
            "Invalid prediction class."
        )


    print(
        "Prediction classes: PASS"
    )


    predicted_legitimate = int(
        (predictions == 0).sum()
    )

    predicted_fraud = int(
        (predictions == 1).sum()
    )

    actual_legitimate = int(
        (graph.y == 0).sum()
    )

    actual_fraud = int(
        (graph.y == 1).sum()
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


    # ========================================================
    # MANIFEST
    # ========================================================

    section(
        "VALIDATING FLOWER FEDAVG MANIFEST"
    )


    if not os.path.exists(
        MANIFEST_PATH
    ):

        raise FileNotFoundError(
            "Flower FedAvg manifest not found:\n"
            f"{MANIFEST_PATH}"
        )


    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        manifest = json.load(
            file
        )


    print(
        "Manifest loaded successfully."
    )


    if manifest.get(
        "technology"
    ) != "Flower + FedAvg":

        raise RuntimeError(
            "Manifest technology mismatch."
        )


    print(
        "Technology: Flower + FedAvg: PASS"
    )


    if manifest.get(
        "num_clients"
    ) != EXPECTED_CLIENTS:

        raise RuntimeError(
            "Manifest client count mismatch."
        )


    print(
        "Manifest client count: PASS"
    )


    if manifest.get(
        "num_rounds"
    ) != EXPECTED_ROUNDS:

        raise RuntimeError(
            "Manifest round count mismatch."
        )


    print(
        "Manifest round count: PASS"
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    section(
        "RFGN FLOWER GLOBAL MODEL VALIDATION SUMMARY"
    )


    print()

    print(
        "Global model file       : PRESENT"
    )

    print(
        "Aggregation method      : Flower FedAvg"
    )

    print(
        f"Clients                 : "
        f"{EXPECTED_CLIENTS}"
    )

    print(
        f"Federated rounds        : "
        f"{EXPECTED_ROUNDS}"
    )

    print(
        f"Input features          : "
        f"{EXPECTED_INPUT_DIM}"
    )

    print(
        f"Hidden dimension        : "
        f"{EXPECTED_HIDDEN_DIM}"
    )

    print(
        f"Output classes          : "
        f"{EXPECTED_OUTPUT_DIM}"
    )

    print(
        f"Parameters              : "
        f"{EXPECTED_PARAMETER_COUNT:,}"
    )

    print(
        "Parameter safety        : PASS"
    )

    print(
        "Graph validation        : PASS"
    )

    print(
        "Forward pass            : PASS"
    )

    print(
        "Output safety           : PASS"
    )

    print(
        "Probability validation  : PASS"
    )

    print(
        "Prediction validation   : PASS"
    )

    print(
        "FedAvg manifest         : PASS"
    )


    print()

    print("=" * 70)

    print(
        "FLOWER GLOBAL MODEL VALIDATION: PASSED"
    )

    print("=" * 70)

    print()

    print(
        "The actual Flower + FedAvg global "
        "GraphSAGE model is validated."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Move to Level 6 - Reinforcement Learning."
    )

    print(
        "Stage 1 - RL environment and threshold "
        "optimization."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()