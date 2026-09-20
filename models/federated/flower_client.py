# ============================================================
# RFGN FLOWER FEDERATED CLIENT
# ============================================================
#
# Purpose:
#   Adapt the existing RFGN GraphSAGE model to Flower.
#
# This client:
#   - Loads one client's aligned graph
#   - Receives global parameters
#   - Performs local GraphSAGE training
#   - Returns updated parameters
#   - Evaluates the local model
#
# Raw client data never leaves the client.
# ============================================================

import os
import sys

import flwr as fl
import torch
import torch.nn.functional as F

from torch.optim import AdamW


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# GRAPH MODEL
# ============================================================

from models.gnn.graphsage_model import GraphSAGE


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIM = 769
HIDDEN_DIM = 128
OUTPUT_DIM = 2
DROPOUT = 0.3

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

LOCAL_EPOCHS = 1


# ============================================================
# CLIENT GRAPH PATHS
# ============================================================

GRAPH_PATHS = {

    1: os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "graphs",
        "client_1",
        "graph_aligned.pt",
    ),

    2: os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "graphs",
        "client_2",
        "graph_aligned.pt",
    ),

    3: os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "graphs",
        "client_3",
        "graph_aligned.pt",
    ),
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# FEDERATED CLIENT
# ============================================================

class FraudGraphSAGEClient(
    fl.client.NumPyClient
):

    def __init__(
        self,
        client_id,
    ):

        self.client_id = int(
            client_id
        )

        if self.client_id not in GRAPH_PATHS:

            raise ValueError(
                "client_id must be 1, 2 or 3."
            )


        # ----------------------------------------------------
        # Load graph
        # ----------------------------------------------------

        graph_path = GRAPH_PATHS[
            self.client_id
        ]


        if not os.path.exists(
            graph_path
        ):

            raise FileNotFoundError(
                f"Graph not found:\n{graph_path}"
            )


        self.graph = torch.load(
            graph_path,
            map_location="cpu",
            weights_only=False,
        )


        # ----------------------------------------------------
        # Validate graph
        # ----------------------------------------------------

        if self.graph.num_node_features != INPUT_DIM:

            raise RuntimeError(
                f"Client {self.client_id}: "
                f"expected {INPUT_DIM} features, "
                f"found "
                f"{self.graph.num_node_features}."
            )


        if self.graph.y.shape[0] != self.graph.num_nodes:

            raise RuntimeError(
                f"Client {self.client_id}: "
                f"label count mismatch."
            )


        if torch.isnan(
            self.graph.x
        ).any():

            raise RuntimeError(
                f"Client {self.client_id}: "
                f"NaN detected in features."
            )


        if torch.isinf(
            self.graph.x
        ).any():

            raise RuntimeError(
                f"Client {self.client_id}: "
                f"infinity detected in features."
            )


        # ----------------------------------------------------
        # Move graph to device
        # ----------------------------------------------------

        self.graph = self.graph.to(
            DEVICE
        )


        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.model = GraphSAGE(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            dropout=DROPOUT,
        ).to(
            DEVICE
        )


        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )


        # ----------------------------------------------------
        # Class weights
        # ----------------------------------------------------

        labels = self.graph.y


        class_counts = torch.bincount(
            labels,
            minlength=OUTPUT_DIM,
        ).float()


        if torch.any(
            class_counts <= 0
        ):

            raise RuntimeError(
                f"Client {self.client_id}: "
                f"missing one of the classes."
            )


        total = class_counts.sum()


        class_weights = (
            total
            /
            (
                OUTPUT_DIM
                *
                class_counts
            )
        )


        self.class_weights = (
            class_weights.to(
                DEVICE
            )
        )


    # ========================================================
    # GET PARAMETERS
    # ========================================================

    def get_parameters(
        self,
        config,
    ):

        return [
            parameter.detach()
            .cpu()
            .numpy()
            for parameter in
            self.model.state_dict().values()
        ]


    # ========================================================
    # SET PARAMETERS
    # ========================================================

    def set_parameters(
        self,
        parameters,
    ):

        state_dict = (
            self.model.state_dict()
        )


        keys = list(
            state_dict.keys()
        )


        if len(parameters) != len(keys):

            raise RuntimeError(
                f"Client {self.client_id}: "
                f"parameter count mismatch."
            )


        new_state_dict = {}


        for key, value in zip(
            keys,
            parameters,
        ):

            tensor = torch.tensor(
                value
            )


            if tensor.shape != (
                state_dict[key].shape
            ):

                raise RuntimeError(
                    f"Client {self.client_id}: "
                    f"shape mismatch for {key}."
                )


            new_state_dict[
                key
            ] = tensor


        self.model.load_state_dict(
            new_state_dict,
            strict=True,
        )


    # ========================================================
    # LOCAL TRAINING
    # ========================================================

    def fit(
        self,
        parameters,
        config,
    ):

        self.set_parameters(
            parameters
        )


        self.model.train()


        epochs = int(
            config.get(
                "local_epochs",
                LOCAL_EPOCHS,
            )
        )


        total_loss = 0.0


        for _ in range(
            epochs
        ):

            self.optimizer.zero_grad(
                set_to_none=True
            )


            logits = self.model(
                self.graph.x,
                self.graph.edge_index,
            )


            loss = F.cross_entropy(
                logits,
                self.graph.y,
                weight=self.class_weights,
            )


            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    f"Client {self.client_id}: "
                    f"non-finite training loss."
                )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=5.0,
            )


            self.optimizer.step()


            total_loss += (
                loss.item()
            )


        average_loss = (
            total_loss
            /
            max(epochs, 1)
        )


        print(
            f"Client {self.client_id} "
            f"local training completed | "
            f"loss={average_loss:.6f}"
        )


        return (
            self.get_parameters(
                config
            ),
            self.graph.num_nodes,
            {
                "train_loss":
                    float(average_loss)
            },
        )


    # ========================================================
    # EVALUATION
    # ========================================================

    def evaluate(
        self,
        parameters,
        config,
    ):

        self.set_parameters(
            parameters
        )


        self.model.eval()


        with torch.no_grad():

            logits = self.model(
                self.graph.x,
                self.graph.edge_index,
            )


            loss = F.cross_entropy(
                logits,
                self.graph.y,
                weight=self.class_weights,
            )


            predictions = logits.argmax(
                dim=1
            )


            accuracy = (
                (
                    predictions
                    ==
                    self.graph.y
                )
                .float()
                .mean()
                .item()
            )


        print(
            f"Client {self.client_id} "
            f"evaluation | "
            f"loss={loss.item():.6f} | "
            f"accuracy={accuracy:.6f}"
        )


        return (
            float(loss.item()),
            self.graph.num_nodes,
            {
                "accuracy":
                    float(accuracy)
            },
        )


# ============================================================
# FACTORY
# ============================================================

def create_flower_client(
    client_id,
):

    return FraudGraphSAGEClient(
        client_id
    )