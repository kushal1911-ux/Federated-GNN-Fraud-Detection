# ============================================================
# RFGN GLOBAL GRAPHSAGE MODEL VALIDATION
# ============================================================
#
# Purpose:
#   Validate the global GraphSAGE model created by FedAvg.
#
# This script is READ-ONLY.
#
# It will NOT:
#   - modify the global model
#   - modify client models
#   - modify graphs
#   - perform training
#   - perform FedAvg
#   - modify CSV datasets
#
# It validates:
#   1. Global model file
#   2. Global model checkpoint
#   3. Architecture
#   4. Parameter names
#   5. Parameter shapes
#   6. Parameter count
#   7. NaN / infinity safety
#   8. Global model reconstruction
#   9. Forward pass
#   10. Output probabilities
#   11. Predictions
# ============================================================

import os
import sys

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
# MODEL
# ============================================================

from models.gnn.graphsage_model import GraphSAGE


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIM = 769
HIDDEN_DIM = 128
OUTPUT_DIM = 2
DROPOUT = 0.3

EXPECTED_PARAMETER_COUNT = 230146


# ============================================================
# PATHS
# ============================================================

GLOBAL_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "global",
    "global_graphsage_fedavg.pt",
)

CLIENT_GRAPH_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs",
    "client_1",
    "graph_aligned.pt",
)


# ============================================================
# SECTION
# ============================================================

def section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# CHECK FILE
# ============================================================

def check_file(path, description):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


    size_mb = (
        os.path.getsize(path)
        /
        (1024 * 1024)
    )


    print(
        f"{description}: PRESENT"
    )

    print(
        f"Path: {path}"
    )

    print(
        f"Size: {size_mb:.2f} MB"
    )


# ============================================================
# VALIDATE STATE DICTIONARY
# ============================================================

def validate_state_dict(
    state_dict
):

    if not isinstance(
        state_dict,
        dict,
    ):

        raise RuntimeError(
            "Global model_state_dict "
            "is not a dictionary."
        )


    print(
        f"Parameter tensors: "
        f"{len(state_dict)}"
    )


    parameter_count = 0


    for name, tensor in state_dict.items():

        if not torch.is_tensor(tensor):

            raise RuntimeError(
                f"Parameter {name} "
                f"is not a tensor."
            )


        if torch.is_floating_point(
            tensor
        ):

            if torch.isnan(
                tensor
            ).any():

                raise RuntimeError(
                    f"NaN detected in "
                    f"parameter: {name}"
                )


            if torch.isinf(
                tensor
            ).any():

                raise RuntimeError(
                    f"Infinity detected in "
                    f"parameter: {name}"
                )


        parameter_count += tensor.numel()


    return parameter_count


# ============================================================
# VALIDATE GRAPH
# ============================================================

def validate_graph(graph):

    section(
        "VALIDATING ALIGNED GRAPH"
    )


    if graph.num_node_features != INPUT_DIM:

        raise RuntimeError(
            f"Expected {INPUT_DIM} features, "
            f"found {graph.num_node_features}."
        )


    print(
        "Feature dimension: PASS"
    )


    if graph.x.shape[0] != graph.num_nodes:

        raise RuntimeError(
            "Feature row count mismatch."
        )


    print(
        "Feature row count: PASS"
    )


    if graph.y.shape[0] != graph.num_nodes:

        raise RuntimeError(
            "Label count mismatch."
        )


    print(
        "Label count: PASS"
    )


    if torch.isnan(
        graph.x
    ).any():

        raise RuntimeError(
            "NaN found in graph features."
        )


    print(
        "Feature NaN check: PASS"
    )


    if torch.isinf(
        graph.x
    ).any():

        raise RuntimeError(
            "Infinity found in graph features."
        )


    print(
        "Feature infinity check: PASS"
    )


    if graph.edge_index.ndim != 2:

        raise RuntimeError(
            "Invalid edge index dimensions."
        )


    if graph.edge_index.shape[0] != 2:

        raise RuntimeError(
            "Invalid edge index shape."
        )


    print(
        "Edge index shape: PASS"
    )


    if graph.edge_index.numel() > 0:

        minimum_index = int(
            graph.edge_index.min().item()
        )

        maximum_index = int(
            graph.edge_index.max().item()
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
                "Negative edge index detected."
            )


        if maximum_index >= graph.num_nodes:

            raise RuntimeError(
                "Edge index exceeds node count."
            )


    print(
        "Edge bounds: PASS"
    )


    classes = sorted(
        torch.unique(
            graph.y
        ).cpu().tolist()
    )


    print(
        f"Classes: {classes}"
    )


    if not set(classes).issubset(
        {0, 1}
    ):

        raise RuntimeError(
            "Labels are not binary."
        )


    print(
        "Binary fraud labels: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "RFGN GLOBAL GRAPHSAGE MODEL VALIDATION"
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
        "No FedAvg aggregation will be performed."
    )

    print(
        "No graph will be modified."
    )


    # ========================================================
    # GLOBAL MODEL FILE
    # ========================================================

    section(
        "CHECKING GLOBAL MODEL FILE"
    )


    check_file(
        GLOBAL_MODEL_PATH,
        "Global model file",
    )


    # ========================================================
    # LOAD GLOBAL CHECKPOINT
    # ========================================================

    section(
        "LOADING GLOBAL MODEL"
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

        "clients",

        "client_node_counts",

        "client_weights",

        "total_nodes",
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
    # VALIDATE CONFIGURATION
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

    aggregation = checkpoint[
        "aggregation"
    ]

    num_clients = checkpoint[
        "num_clients"
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

    print(
        f"Aggregation      : {aggregation}"
    )

    print(
        f"Clients           : {num_clients}"
    )


    if input_dim != INPUT_DIM:

        raise RuntimeError(
            "Invalid global input dimension."
        )


    print(
        "Input dimension: PASS"
    )


    if hidden_dim != HIDDEN_DIM:

        raise RuntimeError(
            "Invalid global hidden dimension."
        )


    print(
        "Hidden dimension: PASS"
    )


    if output_dim != OUTPUT_DIM:

        raise RuntimeError(
            "Invalid global output dimension."
        )


    print(
        "Output dimension: PASS"
    )


    if abs(
        float(dropout)
        -
        DROPOUT
    ) > 1e-8:

        raise RuntimeError(
            "Invalid global dropout."
        )


    print(
        "Dropout: PASS"
    )


    if aggregation != "FedAvg":

        raise RuntimeError(
            "Global model was not created "
            "using FedAvg."
        )


    print(
        "Aggregation method: PASS"
    )


    if num_clients != 3:

        raise RuntimeError(
            "Expected exactly 3 clients."
        )


    print(
        "Client count: PASS"
    )


    # ========================================================
    # CLIENT INFORMATION
    # ========================================================

    section(
        "VALIDATING FEDAVG CLIENT INFORMATION"
    )


    clients = checkpoint[
        "clients"
    ]

    node_counts = checkpoint[
        "client_node_counts"
    ]

    weights = checkpoint[
        "client_weights"
    ]

    total_nodes = checkpoint[
        "total_nodes"
    ]


    print(
        f"Clients: {clients}"
    )


    if clients != [
        "client_1",
        "client_2",
        "client_3",
    ]:

        raise RuntimeError(
            "Unexpected client list."
        )


    print(
        "Client list: PASS"
    )


    calculated_total = sum(
        int(node_counts[client])
        for client in clients
    )


    print()

    for client in clients:

        print(
            f"{client}:"
        )

        print(
            f"  Nodes  : "
            f"{int(node_counts[client]):,}"
        )

        print(
            f"  Weight : "
            f"{float(weights[client]):.8f}"
        )


    if calculated_total != total_nodes:

        raise RuntimeError(
            "Total node count mismatch."
        )


    print()

    print(
        f"Total nodes: {total_nodes:,}"
    )

    print(
        "Client node counts: PASS"
    )


    weight_sum = sum(
        float(weights[client])
        for client in clients
    )


    print(
        f"Weight sum: {weight_sum:.8f}"
    )


    if abs(
        weight_sum - 1.0
    ) > 1e-8:

        raise RuntimeError(
            "FedAvg weights do not sum to 1."
        )


    print(
        "FedAvg weight normalization: PASS"
    )


    # ========================================================
    # GLOBAL STATE
    # ========================================================

    section(
        "VALIDATING GLOBAL MODEL PARAMETERS"
    )


    global_state = checkpoint[
        "model_state_dict"
    ]


    parameter_count = (
        validate_state_dict(
            global_state
        )
    )


    print()

    print(
        f"Parameter values checked: "
        f"{parameter_count:,}"
    )


    if parameter_count != EXPECTED_PARAMETER_COUNT:

        raise RuntimeError(
            "Global parameter count does "
            "not match expected value."
        )


    print(
        "Parameter count: PASS"
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
        "RECONSTRUCTING GLOBAL GRAPHSAGE MODEL"
    )


    model = GraphSAGE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    )


    print(
        "Global GraphSAGE model reconstructed."
    )


    model.load_state_dict(
        global_state,
        strict=True,
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


    print()

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )


    if total_parameters != EXPECTED_PARAMETER_COUNT:

        raise RuntimeError(
            "Unexpected reconstructed "
            "parameter count."
        )


    print(
        "Expected parameter count: PASS"
    )


    # ========================================================
    # LOAD GRAPH
    # ========================================================

    section(
        "LOADING ALIGNED GRAPH FOR FORWARD VALIDATION"
    )


    check_file(
        CLIENT_GRAPH_PATH,
        "Client 1 aligned graph",
    )


    graph = torch.load(
        CLIENT_GRAPH_PATH,
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


    # ========================================================
    # GRAPH VALIDATION
    # ========================================================

    validate_graph(
        graph
    )


    # ========================================================
    # FORWARD PASS
    # ========================================================

    section(
        "RUNNING GLOBAL GRAPHSAGE FORWARD PASS"
    )


    model.eval()


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
        OUTPUT_DIM,
    )


    if tuple(
        logits.shape
    ) != expected_shape:

        raise RuntimeError(
            "Global model output shape "
            "is incorrect."
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
        torch.isnan(
            logits
        ).sum().item()
    )


    infinity_count = int(
        torch.isinf(
            logits
        ).sum().item()
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


    print(
        "Output NaN check: PASS"
    )


    if infinity_count != 0:

        raise RuntimeError(
            "Infinity detected in global output."
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


    print(
        "Probability NaN check: PASS"
    )


    if torch.isinf(
        probabilities
    ).any():

        raise RuntimeError(
            "Infinity detected in probabilities."
        )


    print(
        "Probability infinity check: PASS"
    )


    probability_sums = probabilities.sum(
        dim=1
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
        "Probability normalization: PASS"
    )


    fraud_probabilities = probabilities[
        :,
        1,
    ]


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


    predictions = probabilities.argmax(
        dim=1
    )


    prediction_count = (
        predictions.shape[0]
    )


    print(
        f"Prediction count: "
        f"{prediction_count:,}"
    )


    if prediction_count != graph.num_nodes:

        raise RuntimeError(
            "Prediction count does not "
            "match graph node count."
        )


    print(
        "Prediction count: PASS"
    )


    prediction_classes = sorted(
        torch.unique(
            predictions
        ).cpu().tolist()
    )


    print(
        f"Prediction classes: "
        f"{prediction_classes}"
    )


    if not set(
        prediction_classes
    ).issubset({0, 1}):

        raise RuntimeError(
            "Invalid prediction classes."
        )


    print(
        "Prediction classes: PASS"
    )


    print()

    predicted_legitimate = int(
        (predictions == 0).sum().item()
    )

    predicted_fraud = int(
        (predictions == 1).sum().item()
    )


    actual_legitimate = int(
        (graph.y == 0).sum().item()
    )

    actual_fraud = int(
        (graph.y == 1).sum().item()
    )


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
    # FINAL SUMMARY
    # ========================================================

    section(
        "RFGN GLOBAL GRAPHSAGE VALIDATION SUMMARY"
    )


    print()

    print(
        "Global model file       : PRESENT"
    )

    print(
        "Aggregation method      : FedAvg"
    )

    print(
        "Clients aggregated     : 3"
    )

    print(
        "Input features          : 769"
    )

    print(
        "Hidden dimension        : 128"
    )

    print(
        "Output classes          : 2"
    )

    print(
        f"Parameters              : "
        f"{parameter_count:,}"
    )

    print(
        "Parameter safety        : PASS"
    )

    print(
        f"Graph nodes             : "
        f"{graph.num_nodes:,}"
    )

    print(
        f"Graph edges             : "
        f"{graph.num_edges:,}"
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


    print()

    print("=" * 70)

    print(
        "GLOBAL GRAPHSAGE MODEL VALIDATION: PASSED"
    )

    print("=" * 70)


    print()

    print(
        "The FedAvg global GraphSAGE model "
        "is ready for the next RFGN component."
    )


    print()

    print(
        "NEXT:"
    )

    print(
        "Begin RL threshold optimization."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()