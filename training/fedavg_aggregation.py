# ============================================================
# RFGN FEDAVG AGGREGATION
# ============================================================
#
# Purpose:
#   Aggregate the three locally trained GraphSAGE models
#   using Federated Averaging (FedAvg).
#
# Clients:
#   Client 1
#   Client 2
#   Client 3
#
# Important:
#   - Local models are NOT modified.
#   - Graphs are NOT modified.
#   - CSV datasets are NOT modified.
#   - No additional local training is performed.
#   - Validation/test data is NOT used for aggregation.
#
# Output:
#   Global GraphSAGE model
#   FedAvg aggregation manifest
# ============================================================

import os
import sys
import json
import copy

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
# CONFIGURATION
# ============================================================

INPUT_DIM = 769
HIDDEN_DIM = 128
OUTPUT_DIM = 2
DROPOUT = 0.3

CLIENTS = [
    "client_1",
    "client_2",
    "client_3",
]


# ============================================================
# LOCAL MODEL PATHS
# ============================================================

LOCAL_MODEL_PATHS = {

    "client_1": os.path.join(
        PROJECT_ROOT,
        "saved_models",
        "local",
        "client_1_graphsage_aligned_best.pt",
    ),

    "client_2": os.path.join(
        PROJECT_ROOT,
        "saved_models",
        "local",
        "client_2_graphsage_aligned_best.pt",
    ),

    "client_3": os.path.join(
        PROJECT_ROOT,
        "saved_models",
        "local",
        "client_3_graphsage_aligned_best.pt",
    ),
}


# ============================================================
# CLIENT GRAPH PATHS
# ============================================================

CLIENT_GRAPH_PATHS = {

    "client_1": os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "graphs",
        "client_1",
        "graph_aligned.pt",
    ),

    "client_2": os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "graphs",
        "client_2",
        "graph_aligned.pt",
    ),

    "client_3": os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "graphs",
        "client_3",
        "graph_aligned.pt",
    ),
}


# ============================================================
# OUTPUT PATHS
# ============================================================

GLOBAL_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "global",
    "global_graphsage_fedavg.pt",
)

FEDAVG_MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "global",
    "fedavg_manifest.json",
)


# ============================================================
# SECTION PRINT
# ============================================================

def section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD LOCAL MODEL
# ============================================================

def load_local_model(client):

    path = LOCAL_MODEL_PATHS[client]

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Local model not found for {client}:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    return checkpoint


# ============================================================
# VALIDATE CHECKPOINT
# ============================================================

def validate_checkpoint(
    client,
    checkpoint,
):

    required_keys = [
        "model_state_dict",
        "input_dim",
        "hidden_dim",
        "output_dim",
        "dropout",
        "best_epoch",
        "best_val_loss",
    ]

    for key in required_keys:

        if key not in checkpoint:

            raise KeyError(
                f"{client} checkpoint missing: {key}"
            )


    if checkpoint["input_dim"] != INPUT_DIM:

        raise RuntimeError(
            f"{client}: invalid input dimension."
        )


    if checkpoint["hidden_dim"] != HIDDEN_DIM:

        raise RuntimeError(
            f"{client}: invalid hidden dimension."
        )


    if checkpoint["output_dim"] != OUTPUT_DIM:

        raise RuntimeError(
            f"{client}: invalid output dimension."
        )


    if abs(
        float(checkpoint["dropout"])
        -
        DROPOUT
    ) > 1e-8:

        raise RuntimeError(
            f"{client}: invalid dropout."
        )


# ============================================================
# VALIDATE STATE DICTIONARY
# ============================================================

def validate_state_dictionary(
    client,
    state_dict,
):

    if not isinstance(
        state_dict,
        dict,
    ):

        raise RuntimeError(
            f"{client}: invalid model state dictionary."
        )


    for name, tensor in state_dict.items():

        if not torch.is_tensor(tensor):

            raise RuntimeError(
                f"{client}: parameter {name} "
                f"is not a tensor."
            )


        if torch.is_floating_point(tensor):

            if torch.isnan(tensor).any():

                raise RuntimeError(
                    f"{client}: NaN found in {name}."
                )


            if torch.isinf(tensor).any():

                raise RuntimeError(
                    f"{client}: infinity found in {name}."
                )


# ============================================================
# VALIDATE MODEL COMPATIBILITY
# ============================================================

def validate_model_compatibility(
    state_dicts,
):

    reference_client = CLIENTS[0]

    reference_state = state_dicts[
        reference_client
    ]

    reference_names = list(
        reference_state.keys()
    )


    for client in CLIENTS:

        current_state = state_dicts[
            client
        ]

        current_names = list(
            current_state.keys()
        )


        if current_names != reference_names:

            raise RuntimeError(
                f"{client}: parameter names "
                f"do not match {reference_client}."
            )


        for name in reference_names:

            reference_shape = (
                reference_state[name].shape
            )

            current_shape = (
                current_state[name].shape
            )


            if current_shape != reference_shape:

                raise RuntimeError(
                    f"{client}: shape mismatch "
                    f"for parameter {name}."
                )


# ============================================================
# GET CLIENT NODE COUNT
# ============================================================

def get_client_node_counts():

    node_counts = {}


    for client in CLIENTS:

        graph_path = CLIENT_GRAPH_PATHS[
            client
        ]


        if not os.path.exists(
            graph_path
        ):

            raise FileNotFoundError(
                f"Aligned graph not found for "
                f"{client}:\n{graph_path}"
            )


        graph = torch.load(
            graph_path,
            map_location="cpu",
            weights_only=False,
        )


        node_count = int(
            graph.num_nodes
        )


        if node_count <= 0:

            raise RuntimeError(
                f"{client}: invalid node count."
            )


        node_counts[
            client
        ] = node_count


        print(
            f"{client}: "
            f"{node_count:,} nodes"
        )


        del graph


    return node_counts


# ============================================================
# CALCULATE FEDAVG WEIGHTS
# ============================================================

def calculate_fedavg_weights(
    node_counts,
):

    total_nodes = sum(
        node_counts.values()
    )


    if total_nodes <= 0:

        raise RuntimeError(
            "Total client node count is zero."
        )


    weights = {}


    for client in CLIENTS:

        weights[client] = (
            node_counts[client]
            /
            total_nodes
        )


    return (
        weights,
        total_nodes,
    )


# ============================================================
# FEDAVG
# ============================================================

def federated_average(
    state_dicts,
    weights,
):

    reference_state = state_dicts[
        CLIENTS[0]
    ]


    global_state = copy.deepcopy(
        reference_state
    )


    for name in global_state:

        reference_tensor = (
            reference_state[name]
        )


        # ----------------------------------------------------
        # Floating-point parameters
        # ----------------------------------------------------

        if torch.is_floating_point(
            reference_tensor
        ):

            aggregated = torch.zeros_like(
                reference_tensor
            )


            for client in CLIENTS:

                client_tensor = (
                    state_dicts[
                        client
                    ][name]
                )


                aggregated += (
                    client_tensor
                    *
                    weights[client]
                )


            global_state[name] = (
                aggregated
            )


        # ----------------------------------------------------
        # Non-floating parameters
        # ----------------------------------------------------

        else:

            global_state[name] = (
                reference_tensor.clone()
            )


    return global_state


# ============================================================
# VALIDATE GLOBAL STATE
# ============================================================

def validate_global_state(
    global_state,
):

    parameter_count = 0


    for name, tensor in global_state.items():

        if not torch.is_tensor(tensor):

            raise RuntimeError(
                f"Global parameter {name} "
                f"is not a tensor."
            )


        if torch.is_floating_point(tensor):

            if torch.isnan(tensor).any():

                raise RuntimeError(
                    f"NaN detected in "
                    f"global parameter {name}."
                )


            if torch.isinf(tensor).any():

                raise RuntimeError(
                    f"Infinity detected in "
                    f"global parameter {name}."
                )


        parameter_count += tensor.numel()


    return parameter_count


# ============================================================
# SAVE GLOBAL MODEL
# ============================================================

def save_global_model(
    global_state,
    node_counts,
    weights,
    total_nodes,
):

    os.makedirs(
        os.path.dirname(
            GLOBAL_MODEL_PATH
        ),
        exist_ok=True,
    )


    checkpoint = {

        "model_state_dict":
            global_state,

        "input_dim":
            INPUT_DIM,

        "hidden_dim":
            HIDDEN_DIM,

        "output_dim":
            OUTPUT_DIM,

        "dropout":
            DROPOUT,

        "aggregation":
            "FedAvg",

        "num_clients":
            len(CLIENTS),

        "clients":
            CLIENTS,

        "client_node_counts":
            node_counts,

        "client_weights":
            weights,

        "total_nodes":
            total_nodes,
    }


    torch.save(
        checkpoint,
        GLOBAL_MODEL_PATH,
    )


# ============================================================
# SAVE MANIFEST
# ============================================================

def save_manifest(
    node_counts,
    weights,
    total_nodes,
    parameter_count,
):

    os.makedirs(
        os.path.dirname(
            FEDAVG_MANIFEST_PATH
        ),
        exist_ok=True,
    )


    manifest = {

        "project":
            "RFGN",

        "aggregation":
            "FedAvg",

        "num_clients":
            len(CLIENTS),

        "clients":
            CLIENTS,

        "input_dimension":
            INPUT_DIM,

        "hidden_dimension":
            HIDDEN_DIM,

        "output_dimension":
            OUTPUT_DIM,

        "dropout":
            DROPOUT,

        "parameter_count":
            parameter_count,

        "client_node_counts":
            node_counts,

        "client_weights":
            weights,

        "total_nodes":
            total_nodes,

        "local_models":

            {
                client:
                    LOCAL_MODEL_PATHS[
                        client
                    ]
                for client in CLIENTS
            },

        "global_model":
            GLOBAL_MODEL_PATH,
    }


    with open(
        FEDAVG_MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
        )


# ============================================================
# RELOAD GLOBAL MODEL
# ============================================================

def reload_global_model():

    checkpoint = torch.load(
        GLOBAL_MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )


    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Global model checkpoint does not "
            "contain model_state_dict."
        )


    state_dict = checkpoint[
        "model_state_dict"
    ]


    parameter_count = (
        validate_global_state(
            state_dict
        )
    )


    if parameter_count != 230146:

        raise RuntimeError(
            "Global model parameter count "
            "does not match expected value."
        )


    return checkpoint


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "RFGN FEDAVG AGGREGATION"
    )

    print("=" * 70)

    print()

    print(
        "Purpose:"
    )

    print(
        "Aggregate Client 1, Client 2 and "
        "Client 3 GraphSAGE models."
    )

    print()

    print(
        "Local models will NOT be modified."
    )

    print(
        "Graphs will NOT be modified."
    )

    print(
        "CSV datasets will NOT be modified."
    )

    print(
        "No additional training will be performed."
    )


    # ========================================================
    # LOAD MODELS
    # ========================================================

    section(
        "LOADING LOCAL GRAPHSAGE MODELS"
    )


    checkpoints = {}


    for client in CLIENTS:

        print()

        print(
            f"Loading {client}..."
        )


        checkpoint = load_local_model(
            client
        )


        validate_checkpoint(
            client,
            checkpoint,
        )


        checkpoints[
            client
        ] = checkpoint


        print(
            f"{client}: LOAD PASS"
        )


    # ========================================================
    # STATE DICTIONARIES
    # ========================================================

    section(
        "VALIDATING LOCAL MODEL PARAMETERS"
    )


    state_dicts = {}


    for client in CLIENTS:

        state_dict = checkpoints[
            client
        ][
            "model_state_dict"
        ]


        validate_state_dictionary(
            client,
            state_dict,
        )


        state_dicts[
            client
        ] = state_dict


        print(
            f"{client}: parameter validation: PASS"
        )


        print(
            f"{client}: parameter tensors: "
            f"{len(state_dict)}"
        )


    # ========================================================
    # COMPATIBILITY
    # ========================================================

    section(
        "VALIDATING FEDERATED PARAMETER COMPATIBILITY"
    )


    validate_model_compatibility(
        state_dicts
    )


    print(
        "Parameter names: PASS"
    )

    print(
        "Parameter shapes: PASS"
    )

    print(
        "Parameter compatibility: PASS"
    )


    # ========================================================
    # CLIENT SIZES
    # ========================================================

    section(
        "CALCULATING CLIENT DATA WEIGHTS"
    )


    print()

    node_counts = (
        get_client_node_counts()
    )


    (
        weights,
        total_nodes,
    ) = calculate_fedavg_weights(
        node_counts
    )


    print()

    print(
        f"Total client nodes: "
        f"{total_nodes:,}"
    )


    print()

    for client in CLIENTS:

        print(
            f"{client}:"
        )

        print(
            f"  Nodes  : "
            f"{node_counts[client]:,}"
        )

        print(
            f"  Weight : "
            f"{weights[client]:.8f}"
        )


    weight_sum = sum(
        weights.values()
    )


    print()

    print(
        f"Weight sum: "
        f"{weight_sum:.8f}"
    )


    if abs(
        weight_sum - 1.0
    ) > 1e-8:

        raise RuntimeError(
            "FedAvg weights do not sum to 1."
        )


    print(
        "Weight normalization: PASS"
    )


    # ========================================================
    # FEDAVG
    # ========================================================

    section(
        "PERFORMING FEDAVG AGGREGATION"
    )


    print()

    print(
        "Aggregation method:"
    )

    print(
        "Weighted Federated Averaging"
    )


    print()

    print(
        "Formula:"
    )

    print(
        "Global = Σ(client_weight × local_parameters)"
    )


    print()

    print(
        "Aggregating:"
    )

    for client in CLIENTS:

        print(
            f"  {client} "
            f"weight = "
            f"{weights[client]:.8f}"
        )


    global_state = federated_average(
        state_dicts,
        weights,
    )


    print()

    print(
        "FedAvg aggregation: COMPLETED"
    )


    # ========================================================
    # GLOBAL VALIDATION
    # ========================================================

    section(
        "VALIDATING GLOBAL MODEL PARAMETERS"
    )


    parameter_count = (
        validate_global_state(
            global_state
        )
    )


    print(
        f"Parameter values: "
        f"{parameter_count:,}"
    )


    if parameter_count != 230146:

        raise RuntimeError(
            "Unexpected global parameter count."
        )


    print(
        "Parameter count: PASS"
    )

    print(
        "NaN check: PASS"
    )

    print(
        "Infinity check: PASS"
    )


    # ========================================================
    # SAVE GLOBAL MODEL
    # ========================================================

    section(
        "SAVING GLOBAL GRAPHSAGE MODEL"
    )


    save_global_model(
        global_state,
        node_counts,
        weights,
        total_nodes,
    )


    if not os.path.exists(
        GLOBAL_MODEL_PATH
    ):

        raise RuntimeError(
            "Global model was not saved."
        )


    size_mb = (
        os.path.getsize(
            GLOBAL_MODEL_PATH
        )
        /
        (
            1024 * 1024
        )
    )


    print(
        "Global model saved:"
    )

    print(
        GLOBAL_MODEL_PATH
    )


    print(
        f"File size: "
        f"{size_mb:.2f} MB"
    )


    print(
        "Global model save: PASS"
    )


    # ========================================================
    # MANIFEST
    # ========================================================

    section(
        "CREATING FEDAVG MANIFEST"
    )


    save_manifest(
        node_counts,
        weights,
        total_nodes,
        parameter_count,
    )


    if not os.path.exists(
        FEDAVG_MANIFEST_PATH
    ):

        raise RuntimeError(
            "FedAvg manifest was not saved."
        )


    print(
        "Manifest saved:"
    )

    print(
        FEDAVG_MANIFEST_PATH
    )


    print(
        "Manifest save: PASS"
    )


    # ========================================================
    # RELOAD VALIDATION
    # ========================================================

    section(
        "RELOADING GLOBAL MODEL"
    )


    reloaded_checkpoint = (
        reload_global_model()
    )


    print(
        "Global model reload: PASS"
    )


    print(
        "Global model parameter validation: PASS"
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    section(
        "RFGN FEDAVG AGGREGATION SUMMARY"
    )


    print()

    print(
        "Clients aggregated       : 3"
    )

    print(
        "Client 1 nodes           : "
        f"{node_counts['client_1']:,}"
    )

    print(
        "Client 2 nodes           : "
        f"{node_counts['client_2']:,}"
    )

    print(
        "Client 3 nodes           : "
        f"{node_counts['client_3']:,}"
    )

    print(
        "Total client nodes       : "
        f"{total_nodes:,}"
    )

    print(
        "Shared input features    : 769"
    )

    print(
        "Hidden dimension         : 128"
    )

    print(
        "Output classes           : 2"
    )

    print(
        "Global parameters        : "
        f"{parameter_count:,}"
    )

    print(
        "Aggregation method       : FedAvg"
    )

    print(
        "Parameter compatibility  : YES"
    )

    print(
        "Global NaN values        : 0"
    )

    print(
        "Global infinity values   : 0"
    )

    print(
        "Global model saved       : YES"
    )

    print(
        "FedAvg manifest saved    : YES"
    )


    print()

    print("=" * 70)

    print(
        "RFGN FEDAVG AGGREGATION: COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)

    print()

    print(
        "The three local GraphSAGE models "
        "have been aggregated into the global model."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Validate the global GraphSAGE model "
        "before starting RL threshold optimization."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()