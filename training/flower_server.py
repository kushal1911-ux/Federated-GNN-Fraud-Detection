# ============================================================
# RFGN FLOWER FEDERATED SERVER
# ============================================================
# LEVEL 5 - FEDERATED LEARNING
# STAGE 2 - FLOWER SERVER + FEDAVG
# ============================================================

import os
import sys
import json

import torch
import flwr as fl

from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.strategy import FedAvg


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
# GRAPHSAGE MODEL
# ============================================================

from models.gnn.graphsage_model import GraphSAGE


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIM = 769
HIDDEN_DIM = 128
OUTPUT_DIM = 2
DROPOUT = 0.3

NUM_CLIENTS = 3
NUM_ROUNDS = 3

MIN_CLIENTS = 3


# ============================================================
# OUTPUT PATHS
# ============================================================

GLOBAL_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "global",
)

GLOBAL_MODEL_PATH = os.path.join(
    GLOBAL_DIR,
    "global_graphsage_flower.pt",
)

MANIFEST_PATH = os.path.join(
    GLOBAL_DIR,
    "flower_fedavg_manifest.json",
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
# CREATE GRAPHSAGE MODEL
# ============================================================

def create_model():

    model = GraphSAGE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    )

    model.eval()

    return model


# ============================================================
# INITIAL PARAMETERS
# ============================================================

def get_initial_parameters():

    model = create_model()

    ndarrays = [
        parameter.detach()
        .cpu()
        .numpy()
        for parameter in
        model.state_dict().values()
    ]

    return ndarrays_to_parameters(
        ndarrays
    )


# ============================================================
# CUSTOM FEDAVG STRATEGY
# ============================================================

class RFGNFedAvg(FedAvg):

    def __init__(self, **kwargs):

        super().__init__(
            **kwargs
        )

        self.latest_parameters = None

        self.round_metrics = []


    # ========================================================
    # FEDAVG FIT AGGREGATION
    # ========================================================

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):

        print()

        print(
            f"[SERVER] FEDERATED ROUND "
            f"{server_round}"
        )

        print(
            f"[SERVER] Successful clients: "
            f"{len(results)}"
        )

        print(
            f"[SERVER] Failed clients: "
            f"{len(failures)}"
        )


        if len(results) < MIN_CLIENTS:

            raise RuntimeError(
                "Not enough clients completed "
                "local training."
            )


        aggregated_parameters, metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures,
            )
        )


        if aggregated_parameters is None:

            raise RuntimeError(
                "FedAvg returned no parameters."
            )


        self.latest_parameters = (
            aggregated_parameters
        )


        round_record = {

            "round":
                int(server_round),

            "successful_clients":
                int(len(results)),

            "failed_clients":
                int(len(failures)),
        }


        if metrics:

            round_record["metrics"] = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(
                    value,
                    (int, float)
                )
            }


        self.round_metrics.append(
            round_record
        )


        print(
            f"[SERVER] FedAvg aggregation: PASS"
        )


        return (
            aggregated_parameters,
            metrics,
        )


    # ========================================================
    # FEDAVG EVALUATION
    # ========================================================

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):

        print()

        print(
            f"[SERVER] GLOBAL EVALUATION "
            f"ROUND {server_round}"
        )

        print(
            f"[SERVER] Successful evaluations: "
            f"{len(results)}"
        )


        loss, metrics = (
            super().aggregate_evaluate(
                server_round,
                results,
                failures,
            )
        )


        if loss is not None:

            print(
                f"[SERVER] Aggregated loss: "
                f"{loss:.6f}"
            )


        if metrics:

            for key, value in metrics.items():

                print(
                    f"[SERVER] {key}: {value}"
                )


        return (
            loss,
            metrics,
        )


# ============================================================
# SAVE GLOBAL MODEL
# ============================================================

def save_global_model(
    parameters,
    strategy,
):

    section(
        "SAVING GLOBAL GRAPHSAGE MODEL"
    )


    if parameters is None:

        raise RuntimeError(
            "Final global parameters are missing."
        )


    arrays = parameters_to_ndarrays(
        parameters
    )


    model = create_model()

    original_state = (
        model.state_dict()
    )


    if len(arrays) != len(
        original_state
    ):

        raise RuntimeError(
            "Global parameter tensor count "
            "does not match GraphSAGE."
        )


    new_state = {}


    for key, array in zip(
        original_state.keys(),
        arrays,
    ):

        tensor = torch.tensor(
            array,
            dtype=original_state[key].dtype,
        )


        if tensor.shape != (
            original_state[key].shape
        ):

            raise RuntimeError(
                f"Parameter shape mismatch: "
                f"{key}"
            )


        if torch.is_floating_point(
            tensor
        ):

            if torch.isnan(
                tensor
            ).any():

                raise RuntimeError(
                    f"NaN detected in parameter: "
                    f"{key}"
                )


            if torch.isinf(
                tensor
            ).any():

                raise RuntimeError(
                    f"Infinity detected in parameter: "
                    f"{key}"
                )


        new_state[key] = tensor


    model.load_state_dict(
        new_state,
        strict=True,
    )


    parameter_count = sum(
        parameter.numel()
        for parameter in
        model.parameters()
    )


    if parameter_count != 230146:

        raise RuntimeError(
            "Unexpected GraphSAGE "
            "parameter count."
        )


    os.makedirs(
        GLOBAL_DIR,
        exist_ok=True,
    )


    # ========================================================
    # GLOBAL CHECKPOINT
    # ========================================================

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "input_dim":
            INPUT_DIM,

        "hidden_dim":
            HIDDEN_DIM,

        "output_dim":
            OUTPUT_DIM,

        "dropout":
            DROPOUT,

        "aggregation":
            "Flower FedAvg",

        "num_clients":
            NUM_CLIENTS,

        "num_rounds":
            NUM_ROUNDS,

        "parameter_tensors":
            len(arrays),

        "parameter_count":
            parameter_count,
    }


    torch.save(
        checkpoint,
        GLOBAL_MODEL_PATH,
    )


    print(
        "Global model saved:"
    )

    print(
        GLOBAL_MODEL_PATH
    )

    print(
        "Global model save: PASS"
    )


    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {

        "project":
            "RFGN",

        "level":
            5,

        "component":
            "Federated Learning",

        "technology":
            "Flower + FedAvg",

        "num_clients":
            NUM_CLIENTS,

        "num_rounds":
            NUM_ROUNDS,

        "input_dimension":
            INPUT_DIM,

        "hidden_dimension":
            HIDDEN_DIM,

        "output_dimension":
            OUTPUT_DIM,

        "dropout":
            DROPOUT,

        "parameter_tensors":
            len(arrays),

        "parameter_count":
            parameter_count,

        "global_model":
            GLOBAL_MODEL_PATH,

        "round_metrics":
            strategy.round_metrics,
    }


    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
        )


    print(
        "FedAvg manifest saved:"
    )

    print(
        MANIFEST_PATH
    )

    print(
        "Manifest save: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "RFGN FLOWER FEDERATED SERVER"
    )

    print("=" * 70)

    print()

    print(
        "LEVEL 5 - FEDERATED LEARNING"
    )

    print(
        "STAGE 2 - FLOWER SERVER + FEDAVG"
    )

    print()

    print(
        f"Input features   : {INPUT_DIM}"
    )

    print(
        f"Hidden dimension : {HIDDEN_DIM}"
    )

    print(
        f"Output classes   : {OUTPUT_DIM}"
    )

    print(
        f"Clients          : {NUM_CLIENTS}"
    )

    print(
        f"Rounds           : {NUM_ROUNDS}"
    )

    print(
        "Aggregation      : FedAvg"
    )


    # ========================================================
    # INITIAL MODEL
    # ========================================================

    section(
        "CREATING INITIAL GRAPHSAGE MODEL"
    )


    model = create_model()


    parameter_count = sum(
        parameter.numel()
        for parameter in
        model.parameters()
    )


    parameter_tensors = len(
        model.state_dict()
    )


    print(
        "GraphSAGE model: CREATED"
    )

    print(
        f"Parameter tensors: "
        f"{parameter_tensors}"
    )

    print(
        f"Parameter values: "
        f"{parameter_count:,}"
    )


    if parameter_count != 230146:

        raise RuntimeError(
            "GraphSAGE parameter count mismatch."
        )


    print(
        "Parameter count: PASS"
    )


    # ========================================================
    # INITIAL PARAMETERS
    # ========================================================

    section(
        "PREPARING INITIAL PARAMETERS"
    )


    initial_parameters = (
        get_initial_parameters()
    )


    print(
        "Initial parameters: PASS"
    )


    # ========================================================
    # FEDAVG STRATEGY
    # ========================================================

    section(
        "CREATING FEDAVG STRATEGY"
    )


    strategy = RFGNFedAvg(

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=MIN_CLIENTS,

        min_evaluate_clients=MIN_CLIENTS,

        min_available_clients=MIN_CLIENTS,

        initial_parameters=
            initial_parameters,

        on_fit_config_fn=lambda
            server_round:
            {
                "local_epochs": 1,

                "server_round":
                    int(server_round),
            },
    )


    print(
        "FedAvg strategy: CREATED"
    )

    print(
        "Minimum clients: 3"
    )

    print(
        "Strategy compatibility: PASS"
    )


    # ========================================================
    # START FLOWER SERVER
    # ========================================================

    section(
        "STARTING FLOWER SERVER"
    )


    print(
        "Server address: 0.0.0.0:8080"
    )

    print(
        f"Federated rounds: {NUM_ROUNDS}"
    )

    print()

    print(
        "Waiting for 3 clients..."
    )


    fl.server.start_server(

        server_address=
            "0.0.0.0:8080",

        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),

        strategy=strategy,
    )


    # ========================================================
    # FLOWER FINISHED
    # ========================================================

    section(
        "FLOWER FEDERATED TRAINING FINISHED"
    )


    if strategy.latest_parameters is None:

        raise RuntimeError(
            "Flower completed but no final "
            "FedAvg parameters were captured."
        )


    print(
        "Final global parameters: PRESENT"
    )


    # ========================================================
    # SAVE GLOBAL MODEL
    # ========================================================

    save_global_model(
        strategy.latest_parameters,
        strategy,
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    section(
        "RFGN FLOWER FEDERATED LEARNING SUMMARY"
    )


    print()

    print(
        "Clients aggregated       : 3"
    )

    print(
        f"Federated rounds         : "
        f"{NUM_ROUNDS}"
    )

    print(
        "Aggregation              : Flower FedAvg"
    )

    print(
        f"Input features           : "
        f"{INPUT_DIM}"
    )

    print(
        f"Hidden dimension         : "
        f"{HIDDEN_DIM}"
    )

    print(
        f"Output classes           : "
        f"{OUTPUT_DIM}"
    )

    print(
        f"Parameters               : "
        f"{parameter_count:,}"
    )

    print(
        "Final parameters         : PRESENT"
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
        "RFGN FLOWER SERVER: COMPLETED"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()