from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SERVER_ADDRESS = "127.0.0.1:8080"

INPUT_DIM = 769
HIDDEN_DIM = 128
OUTPUT_DIM = 2
DROPOUT = 0.30

LOCAL_EPOCHS = 1
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0

SEED = 42

GRAPH_PATHS = {
    1: PROJECT_ROOT / "data" / "processed" / "graphs" / "client_1" / "graph_aligned.pt",
    2: PROJECT_ROOT / "data" / "processed" / "graphs" / "client_2" / "graph_aligned.pt",
    3: PROJECT_ROOT / "data" / "processed" / "graphs" / "client_3" / "graph_aligned.pt",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# GRAPH DATA
# ============================================================

def load_graph(client_id: int) -> Data:
    path = GRAPH_PATHS[client_id]

    if not path.exists():
        raise FileNotFoundError(
            f"Graph file not found for Client {client_id}: {path}"
        )

    data = torch.load(path, map_location="cpu", weights_only=False)

    if not hasattr(data, "x"):
        raise ValueError("Graph does not contain node features 'x'.")

    if not hasattr(data, "edge_index"):
        raise ValueError("Graph does not contain 'edge_index'.")

    if not hasattr(data, "y"):
        raise ValueError("Graph does not contain labels 'y'.")

    if data.x.shape[1] != INPUT_DIM:
        raise ValueError(
            f"Client {client_id}: expected {INPUT_DIM} features, "
            f"found {data.x.shape[1]}."
        )

    return data


# ============================================================
# MODEL
# ============================================================

class GraphSAGE(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = OUTPUT_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, output_dim)

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        return self.classifier(x)


# ============================================================
# TRAINING
# ============================================================

def train_local_model(
    model: nn.Module,
    data: Data,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> float:

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    x = data.x.float()
    edge_index = data.edge_index.long()
    y = data.y.long()

    # Use the same full-graph training behavior as the existing
    # local GraphSAGE training pipeline.
    train_mask = getattr(data, "train_mask", None)

    if train_mask is None:
        train_mask = torch.ones(
            y.shape[0],
            dtype=torch.bool,
        )

    last_loss = 0.0

    for _ in range(epochs):

        optimizer.zero_grad(set_to_none=True)

        logits = model(x, edge_index)

        loss = criterion(
            logits[train_mask],
            y[train_mask],
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Training produced a non-finite loss."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        last_loss = float(loss.item())

    return last_loss


# ============================================================
# EVALUATION
# ============================================================

def evaluate_local_model(
    model: nn.Module,
    data: Data,
) -> Tuple[float, Dict[str, float]]:

    model.eval()

    x = data.x.float()
    edge_index = data.edge_index.long()
    y = data.y.long()

    eval_mask = getattr(data, "val_mask", None)

    if eval_mask is None:
        eval_mask = torch.ones(
            y.shape[0],
            dtype=torch.bool,
        )

    with torch.no_grad():

        logits = model(x, edge_index)

        loss = F.cross_entropy(
            logits[eval_mask],
            y[eval_mask],
        )

        predictions = logits.argmax(dim=1)

        correct = (
            predictions[eval_mask] == y[eval_mask]
        ).sum()

        total = int(eval_mask.sum().item())

        accuracy = (
            float(correct.item()) / total
            if total > 0
            else 0.0
        )

    metrics = {
        "accuracy": accuracy,
        "validation_samples": float(total),
    }

    return float(loss.item()), metrics


# ============================================================
# NUMPY PARAMETER CONVERSION
# ============================================================

def get_parameters(model: nn.Module) -> List[np.ndarray]:

    return [
        value.detach().cpu().numpy()
        for value in model.state_dict().values()
    ]


def set_parameters(
    model: nn.Module,
    parameters: List[np.ndarray],
) -> None:

    state_dict = model.state_dict()

    if len(parameters) != len(state_dict):
        raise ValueError(
            f"Parameter count mismatch: "
            f"received {len(parameters)}, "
            f"expected {len(state_dict)}."
        )

    new_state_dict = {}

    for (
        (name, old_tensor),
        new_array,
    ) in zip(state_dict.items(), parameters):

        tensor = torch.tensor(
            new_array,
            dtype=old_tensor.dtype,
        )

        if tensor.shape != old_tensor.shape:
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"received {tuple(tensor.shape)}, "
                f"expected {tuple(old_tensor.shape)}."
            )

        new_state_dict[name] = tensor

    model.load_state_dict(new_state_dict)


# ============================================================
# FEDERATED CLIENT
# ============================================================

class RFGNFederatedClient(fl.client.NumPyClient):

    def __init__(self, client_id: int):

        self.client_id = client_id

        set_seed(SEED + client_id)

        self.device = torch.device("cpu")

        print()
        print("=" * 70)
        print(f"INITIALIZING CLIENT {client_id}")
        print("=" * 70)

        print(f"Graph: {GRAPH_PATHS[client_id]}")

        self.data = load_graph(client_id)

        self.model = GraphSAGE().to(self.device)

        print(
            f"Nodes         : {self.data.num_nodes:,}"
        )

        print(
            f"Features      : {self.data.x.shape[1]}"
        )

        print(
            f"Edges         : {self.data.edge_index.shape[1]:,}"
        )

        fraud_count = int(
            (self.data.y == 1).sum().item()
        )

        print(
            f"Fraud samples : {fraud_count:,}"
        )

        print(
            f"Parameters    : "
            f"{sum(p.numel() for p in self.model.parameters()):,}"
        )

    # --------------------------------------------------------
    # GET PARAMETERS
    # --------------------------------------------------------

    def get_parameters(
        self,
        config: Dict[str, str],
    ) -> List[np.ndarray]:

        return get_parameters(self.model)

    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:

        server_round = int(
            config.get("server_round", 0)
        )

        print()
        print("-" * 70)
        print(
            f"CLIENT {self.client_id} "
            f"TRAINING — ROUND {server_round}"
        )
        print("-" * 70)

        set_parameters(
            self.model,
            parameters,
        )

        loss = train_local_model(
            model=self.model,
            data=self.data,
            epochs=LOCAL_EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        print(
            f"Client {self.client_id} local loss: "
            f"{loss:.6f}"
        )

        return (
            get_parameters(self.model),
            int(self.data.num_nodes),
            {
                "train_loss": loss,
                "client_id": float(self.client_id),
                "server_round": float(server_round),
            },
        )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[float, int, Dict[str, float]]:

        set_parameters(
            self.model,
            parameters,
        )

        loss, metrics = evaluate_local_model(
            self.model,
            self.data,
        )

        print(
            f"Client {self.client_id} "
            f"evaluation loss: {loss:.6f} | "
            f"accuracy: {metrics['accuracy']:.6f}"
        )

        return (
            loss,
            int(self.data.num_nodes),
            metrics,
        )


# ============================================================
# CLIENT FACTORY
# ============================================================

def client_fn(context) -> RFGNFederatedClient:

    client_id_text = os.environ.get(
        "RFGN_CLIENT_ID"
    )

    if client_id_text is None:

        raise RuntimeError(
            "RFGN_CLIENT_ID environment variable is not set.\n"
            "Use one of:\n"
            "  $env:RFGN_CLIENT_ID='1'\n"
            "  $env:RFGN_CLIENT_ID='2'\n"
            "  $env:RFGN_CLIENT_ID='3'"
        )

    try:
        client_id = int(client_id_text)

    except ValueError:

        raise ValueError(
            f"Invalid RFGN_CLIENT_ID: {client_id_text}"
        )

    if client_id not in GRAPH_PATHS:

        raise ValueError(
            f"Invalid client ID {client_id}. "
            f"Expected 1, 2, or 3."
        )

    print(
        f"\nFlower assigned RFGN Client ID: "
        f"{client_id}"
    )

    return RFGNFederatedClient(
        client_id=client_id
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("RFGN FEDERATED CLIENT")
    print("=" * 70)

    print()
    print(f"Server address : {SERVER_ADDRESS}")
    print("Model          : Original GraphSAGE")
    print(f"Input features : {INPUT_DIM}")
    print(f"Hidden features: {HIDDEN_DIM}")
    print(f"Output classes : {OUTPUT_DIM}")
    print(f"Dropout        : {DROPOUT}")
    print(f"Local epochs   : {LOCAL_EPOCHS}")

    client_id = os.environ.get(
        "RFGN_CLIENT_ID",
        "NOT SET",
    )

    print(f"Client ID      : {client_id}")

    print()
    print("Connecting to Flower server...")

    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client_fn=client_fn,
    )


if __name__ == "__main__":
    main()