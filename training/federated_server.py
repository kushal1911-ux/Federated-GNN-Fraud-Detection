from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SERVER_ADDRESS = "127.0.0.1:8080"

NUM_CLIENTS = 3
NUM_ROUNDS = 3

INPUT_DIM = 769
HIDDEN_DIM = 128
OUTPUT_DIM = 2
DROPOUT = 0.30

GLOBAL_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "global"
    / "global_graphsage_flower_optimized.pt"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "global"
    / "global_graphsage_flower_optimized_metrics.json"
)


# ============================================================
# MODEL
# ============================================================

def build_model() -> torch.nn.Module:

    from torch_geometric.nn import SAGEConv
    import torch.nn.functional as F

    class GraphSAGE(torch.nn.Module):

        def __init__(self):
            super().__init__()

            self.conv1 = SAGEConv(
                INPUT_DIM,
                HIDDEN_DIM,
            )

            self.conv2 = SAGEConv(
                HIDDEN_DIM,
                HIDDEN_DIM,
            )

            self.classifier = torch.nn.Linear(
                HIDDEN_DIM,
                OUTPUT_DIM,
            )

            self.dropout = DROPOUT

        def forward(
            self,
            x,
            edge_index,
        ):

            x = self.conv1(
                x,
                edge_index,
            )

            x = F.relu(x)

            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training,
            )

            x = self.conv2(
                x,
                edge_index,
            )

            x = F.relu(x)

            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training,
            )

            return self.classifier(x)

    return GraphSAGE()


# ============================================================
# METRICS AGGREGATION
# ============================================================

def weighted_average_metrics(
    metrics: List[Tuple[int, Dict[str, Scalar]]],
) -> Dict[str, Scalar]:

    if not metrics:
        return {}

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    if total_examples == 0:
        return {}

    result: Dict[str, Scalar] = {}

    keys = set()

    for _, metric_dict in metrics:
        keys.update(metric_dict.keys())

    for key in keys:

        numeric_values = []

        for num_examples, metric_dict in metrics:

            if key in metric_dict:

                value = metric_dict[key]

                if isinstance(
                    value,
                    (int, float, np.integer, np.floating),
                ):
                    numeric_values.append(
                        (
                            num_examples,
                            float(value),
                        )
                    )

        if numeric_values:

            weighted_sum = sum(
                num_examples * value
                for num_examples, value
                in numeric_values
            )

            denominator = sum(
                num_examples
                for num_examples, _
                in numeric_values
            )

            if denominator > 0:

                result[key] = (
                    weighted_sum / denominator
                )

    return result


# ============================================================
# CUSTOM FEDAVG
# ============================================================

class RFGNFedAvg(fl.server.strategy.FedAvg):

    def __init__(
        self,
        initial_parameters: Parameters,
    ):

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,

            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,

            evaluate_metrics_aggregation_fn=(
                weighted_average_metrics
            ),

            fit_metrics_aggregation_fn=(
                weighted_average_metrics
            ),

            initial_parameters=initial_parameters,
        )

        self.round_history = []

    # --------------------------------------------------------
    # FIT CONFIG
    # --------------------------------------------------------

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ):

        print()
        print("=" * 70)
        print(f"FEDERATED ROUND {server_round}")
        print("=" * 70)

        config = {
            "server_round": str(server_round),
        }

        fit_config = []

        clients = client_manager.sample(
            num_clients=NUM_CLIENTS,
            min_num_clients=NUM_CLIENTS,
        )

        for client in clients:

            fit_config.append(
                (
                    client,
                    fl.common.FitIns(
                        parameters,
                        config,
                    ),
                )
            )

        return fit_config

    # --------------------------------------------------------
    # AGGREGATE FIT
    # --------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):

        print()
        print(
            f"Aggregating federated round "
            f"{server_round}"
        )

        print(
            f"Successful clients: {len(results)}"
        )

        print(
            f"Failures: {len(failures)}"
        )

        if not results:

            print(
                "ERROR: No successful client results."
            )

            return None, {}

        aggregated_parameters, metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures,
            )
        )

        if aggregated_parameters is None:

            print(
                "ERROR: Aggregation returned no parameters."
            )

            return None, metrics

        # Save the current global model.
        self.save_global_model(
            aggregated_parameters,
            server_round,
        )

        return aggregated_parameters, metrics

    # --------------------------------------------------------
    # SAVE GLOBAL MODEL
    # --------------------------------------------------------

    def save_global_model(
        self,
        parameters: Parameters,
        server_round: int,
    ):

        ndarrays = parameters_to_ndarrays(
            parameters
        )

        model = build_model()

        state_dict = model.state_dict()

        if len(ndarrays) != len(state_dict):

            raise RuntimeError(
                "Global parameter count mismatch."
            )

        new_state_dict = {}

        for (
            (name, old_tensor),
            array,
        ) in zip(
            state_dict.items(),
            ndarrays,
        ):

            tensor = torch.tensor(
                array,
                dtype=old_tensor.dtype,
            )

            if tensor.shape != old_tensor.shape:

                raise RuntimeError(
                    f"Shape mismatch for {name}: "
                    f"{tensor.shape} != "
                    f"{old_tensor.shape}"
                )

            new_state_dict[name] = tensor

        model.load_state_dict(
            new_state_dict
        )

        GLOBAL_MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            model.state_dict(),
            GLOBAL_MODEL_PATH,
        )

        print()
        print(
            f"Global model saved:"
        )

        print(
            f"  {GLOBAL_MODEL_PATH}"
        )

        self.round_history.append(
            {
                "round": server_round,
                "model_path": str(
                    GLOBAL_MODEL_PATH
                ),
            }
        )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):

        loss, metrics = (
            super().aggregate_evaluate(
                server_round,
                results,
                failures,
            )
        )

        if loss is not None:

            print()
            print(
                f"Round {server_round} "
                f"federated evaluation"
            )

            print(
                f"Loss: {loss:.6f}"
            )

            if metrics:

                for key, value in metrics.items():

                    if isinstance(
                        value,
                        (int, float),
                    ):

                        print(
                            f"{key}: {float(value):.6f}"
                        )

        return loss, metrics


# ============================================================
# INITIAL GLOBAL MODEL
# ============================================================

def create_initial_parameters() -> Parameters:

    model = build_model()

    print(
        "Creating initial global parameters..."
    )

    ndarrays = [
        value.detach().cpu().numpy()
        for value in model.state_dict().values()
    ]

    parameter_count = sum(
        array.size
        for array in ndarrays
    )

    print(
        f"Initial parameters: "
        f"{parameter_count:,}"
    )

    return ndarrays_to_parameters(
        ndarrays
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("RFGN FEDERATED LEARNING SERVER")
    print("=" * 70)

    print()
    print(f"Server address : {SERVER_ADDRESS}")
    print(f"Clients        : {NUM_CLIENTS}")
    print(f"Rounds         : {NUM_ROUNDS}")
    print("Algorithm      : FedAvg")
    print("Architecture   : Original GraphSAGE")
    print(f"Input features : {INPUT_DIM}")
    print(f"Hidden features: {HIDDEN_DIM}")

    model = build_model()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters     : {parameter_count:,}"
    )

    print()
    print("Creating initial global model...")

    initial_parameters = (
        create_initial_parameters()
    )

    strategy = RFGNFedAvg(
        initial_parameters=initial_parameters
    )

    print()
    print("Starting Flower server...")
    print()

    history = fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),
        strategy=strategy,
    )

    # --------------------------------------------------------
    # SAVE TRAINING SUMMARY
    # --------------------------------------------------------

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "algorithm": "FedAvg",
        "architecture": "Original GraphSAGE",
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": OUTPUT_DIM,
        "dropout": DROPOUT,
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS,
        "server_address": SERVER_ADDRESS,
        "global_model_path": str(
            GLOBAL_MODEL_PATH
        ),
        "round_history": strategy.round_history,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("FEDERATED TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Global model:"
    )

    print(
        f"{GLOBAL_MODEL_PATH}"
    )

    print()
    print(
        f"Summary:"
    )

    print(
        f"{METRICS_PATH}"
    )


if __name__ == "__main__":
    main()