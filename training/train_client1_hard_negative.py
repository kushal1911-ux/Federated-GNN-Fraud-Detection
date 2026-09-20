import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from models.gnn.graphsage_model import GraphSAGE


# ============================================================
# CONFIGURATION
# ============================================================

GRAPH_PATH = (
    ROOT
    / "data"
    / "processed"
    / "graphs"
    / "client_1"
    / "graph_aligned.pt"
)

OUTPUT_MODEL_PATH = (
    ROOT
    / "saved_models"
    / "local"
    / "client_1_graphsage_hard_negative_best.pt"
)

OUTPUT_METRICS_PATH = (
    ROOT
    / "saved_models"
    / "local"
    / "client_1_graphsage_hard_negative_metrics.json"
)

INPUT_DIM = 769
HIDDEN_DIM = 128
OUTPUT_DIM = 2
DROPOUT = 0.30

VAL_RATIO = 0.15
SEED = 42

INITIAL_EPOCHS = 20
MINING_EPOCHS = 40

PATIENCE = 8

LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001

# Percentage of legitimate training samples considered
# hard negatives after initial model scoring.
HARD_NEGATIVE_FRACTION = 0.10

# Weight applied to hard negative samples.
HARD_NEGATIVE_WEIGHT = 3.0

GRAD_CLIP = 1.0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph():

    print("Loading Client 1 aligned graph...")

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False
    )

    if not isinstance(graph, Data):
        raise TypeError(
            f"Expected PyG Data object, got {type(graph)}"
        )

    print("\nGraph information:")
    print(f"  Nodes          : {graph.num_nodes:,}")
    print(f"  Features       : {graph.num_node_features}")
    print(f"  Directed edges : {graph.edge_index.shape[1]:,}")
    print(f"  Labels         : {graph.y.shape[0]:,}")

    class_0 = int(
        (graph.y == 0).sum().item()
    )

    class_1 = int(
        (graph.y == 1).sum().item()
    )

    print(f"\n  Class 0 count : {class_0:,}")
    print(f"  Class 1 count : {class_1:,}")

    if graph.num_node_features != INPUT_DIM:
        raise ValueError(
            f"Expected {INPUT_DIM} features, "
            f"got {graph.num_node_features}"
        )

    if torch.isnan(graph.x).any():
        raise ValueError(
            "Graph contains NaN values."
        )

    if torch.isinf(graph.x).any():
        raise ValueError(
            "Graph contains Inf values."
        )

    return graph


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def create_split(num_nodes):

    set_seed(SEED)

    indices = torch.randperm(
        num_nodes
    )

    val_size = int(
        num_nodes * VAL_RATIO
    )

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return train_idx, val_idx


# ============================================================
# MODEL
# ============================================================

def create_model(device):

    model = GraphSAGE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    )

    model.to(device)

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    probabilities,
    labels,
    threshold=0.50
):

    predictions = (
        probabilities >= threshold
    ).long()

    labels = labels.long()

    tp = int(
        ((predictions == 1) & (labels == 1))
        .sum()
        .item()
    )

    tn = int(
        ((predictions == 0) & (labels == 0))
        .sum()
        .item()
    )

    fp = int(
        ((predictions == 1) & (labels == 0))
        .sum()
        .item()
    )

    fn = int(
        ((predictions == 0) & (labels == 1))
        .sum()
        .item()
    )

    accuracy = (
        (tp + tn) / len(labels)
        if len(labels) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def predict(
    model,
    graph,
    device
):

    model.eval()

    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)

    output = model(
        x,
        edge_index
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )[:, 1]

    return probabilities.cpu()


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    graph,
    val_idx,
    device
):

    probabilities = predict(
        model,
        graph,
        device
    )

    val_probabilities = (
        probabilities[val_idx]
    )

    val_labels = (
        graph.y[val_idx].cpu()
    )

    val_loss = F.cross_entropy(
        torch.stack(
            [
                1.0 - val_probabilities,
                val_probabilities
            ],
            dim=1
        ).clamp(min=1e-8),
        val_labels
    )

    metrics = calculate_metrics(
        val_probabilities,
        val_labels
    )

    metrics["val_loss"] = float(
        val_loss.item()
    )

    return metrics


# ============================================================
# INITIAL TRAINING
# ============================================================

def initial_training(
    model,
    graph,
    train_idx,
    val_idx,
    device
):

    print("\n" + "=" * 70)
    print("INITIAL GRAPH SAGE TRAINING")
    print("=" * 70)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_loss = float("inf")
    best_state = None

    patience_counter = 0

    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    y = graph.y.to(device)

    train_idx_device = train_idx.to(device)

    for epoch in range(
        1,
        INITIAL_EPOCHS + 1
    ):

        model.train()

        optimizer.zero_grad()

        output = model(
            x,
            edge_index
        )

        loss = F.cross_entropy(
            output[train_idx_device],
            y[train_idx_device]
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        val_metrics = evaluate(
            model,
            graph,
            val_idx,
            device
        )

        val_loss = val_metrics[
            "val_loss"
        ]

        scheduler.step(
            val_loss
        )

        lr = optimizer.param_groups[0][
            "lr"
        ]

        print(
            f"Epoch {epoch:02d}/{INITIAL_EPOCHS} "
            f"| Train Loss: {loss.item():.6f} "
            f"| Val Loss: {val_loss:.6f} "
            f"| Acc: {val_metrics['accuracy']:.4f} "
            f"| Prec: {val_metrics['precision']:.4f} "
            f"| Recall: {val_metrics['recall']:.4f} "
            f"| F1: {val_metrics['f1']:.4f} "
            f"| LR: {lr:.6f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    "Early stopping initial training."
                )

                break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model


# ============================================================
# HARD NEGATIVE MINING
# ============================================================

@torch.no_grad()
def mine_hard_negatives(
    model,
    graph,
    train_idx,
    device
):

    print("\n" + "=" * 70)
    print("HARD NEGATIVE MINING")
    print("=" * 70)

    probabilities = predict(
        model,
        graph,
        device
    )

    train_labels = (
        graph.y[train_idx]
    )

    train_probabilities = (
        probabilities[train_idx]
    )

    # Legitimate training samples only.
    negative_mask = (
        train_labels == 0
    )

    negative_positions = torch.nonzero(
        negative_mask,
        as_tuple=False
    ).view(-1)

    negative_probabilities = (
        train_probabilities[
            negative_positions
        ]
    )

    hard_count = max(
        1,
        int(
            len(negative_positions)
            * HARD_NEGATIVE_FRACTION
        )
    )

    # Highest fraud probability among legitimate
    # transactions = hardest negatives.
    sorted_positions = torch.argsort(
        negative_probabilities,
        descending=True
    )

    selected_positions = (
        negative_positions[
            sorted_positions[:hard_count]
        ]
    )

    hard_negative_idx = (
        train_idx[selected_positions]
    )

    threshold_index = (
        sorted_positions[
            min(
                hard_count - 1,
                len(sorted_positions) - 1
            )
        ]
    )

    probability_cutoff = float(
        negative_probabilities[
            threshold_index
        ].item()
    )

    print(
        f"Legitimate training samples : "
        f"{len(negative_positions):,}"
    )

    print(
        f"Hard negative fraction       : "
        f"{HARD_NEGATIVE_FRACTION:.2%}"
    )

    print(
        f"Hard negatives selected      : "
        f"{len(hard_negative_idx):,}"
    )

    print(
        f"Hard-negative probability cutoff: "
        f"{probability_cutoff:.8f}"
    )

    print(
        f"Maximum hard-negative probability: "
        f"{float(negative_probabilities.max()):.8f}"
    )

    return hard_negative_idx


# ============================================================
# HARD NEGATIVE RETRAINING
# ============================================================

def hard_negative_training(
    model,
    graph,
    train_idx,
    val_idx,
    hard_negative_idx,
    device
):

    print("\n" + "=" * 70)
    print("HARD NEGATIVE RETRAINING")
    print("=" * 70)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    patience_counter = 0

    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    y = graph.y.to(device)

    train_idx_device = (
        train_idx.to(device)
    )

    hard_negative_device = (
        hard_negative_idx.to(device)
    )

    # Weight vector for the complete training set.
    sample_weights = torch.ones(
        len(train_idx),
        device=device
    )

    # Map hard-negative global indices to their
    # positions inside train_idx.
    train_position = {
        int(index): position
        for position, index
        in enumerate(
            train_idx.tolist()
        )
    }

    for global_idx in (
        hard_negative_idx.tolist()
    ):

        position = train_position[
            int(global_idx)
        ]

        sample_weights[position] = (
            HARD_NEGATIVE_WEIGHT
        )

    for epoch in range(
        1,
        MINING_EPOCHS + 1
    ):

        model.train()

        optimizer.zero_grad()

        output = model(
            x,
            edge_index
        )

        per_sample_loss = F.cross_entropy(
            output[train_idx_device],
            y[train_idx_device],
            reduction="none"
        )

        weighted_loss = (
            per_sample_loss
            * sample_weights
        )

        loss = weighted_loss.mean()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        val_metrics = evaluate(
            model,
            graph,
            val_idx,
            device
        )

        val_loss = val_metrics[
            "val_loss"
        ]

        scheduler.step(
            val_loss
        )

        lr = optimizer.param_groups[0][
            "lr"
        ]

        print(
            f"Epoch {epoch:02d}/{MINING_EPOCHS} "
            f"| Train Loss: {loss.item():.6f} "
            f"| Val Loss: {val_loss:.6f} "
            f"| Acc: {val_metrics['accuracy']:.4f} "
            f"| Prec: {val_metrics['precision']:.4f} "
            f"| Recall: {val_metrics['recall']:.4f} "
            f"| F1: {val_metrics['f1']:.4f} "
            f"| LR: {lr:.6f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss
            best_epoch = epoch

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    "Early stopping hard-negative training."
                )

                break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model, best_epoch


# ============================================================
# FINAL VALIDATION
# ============================================================

def final_evaluation(
    model,
    graph,
    val_idx,
    device
):

    metrics = evaluate(
        model,
        graph,
        val_idx,
        device
    )

    return metrics


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    metrics,
    hard_negative_count,
    best_epoch
):

    OUTPUT_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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

        "architecture":
            "Original GraphSAGE",

        "training_method":
            "Hard Negative Mining",

        "hard_negative_fraction":
            HARD_NEGATIVE_FRACTION,

        "hard_negative_weight":
            HARD_NEGATIVE_WEIGHT,

        "hard_negative_count":
            hard_negative_count,

        "best_epoch":
            best_epoch,

        "metrics":
            metrics,
    }

    torch.save(
        checkpoint,
        OUTPUT_MODEL_PATH
    )

    print(
        "\nModel saved to:"
    )

    print(
        OUTPUT_MODEL_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "RFGN CLIENT 1 "
        "GRAPH SAGE HARD NEGATIVE MINING"
    )

    print("=" * 70)

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    graph = load_graph()

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    print(
        "\nCreating train/validation split..."
    )

    train_idx, val_idx = (
        create_split(
            graph.num_nodes
        )
    )

    print(
        f"  Training   : "
        f"{len(train_idx):,}"
    )

    print(
        f"  Validation : "
        f"{len(val_idx):,}"
    )

    print(
        f"  Train fraud: "
        f"{int((graph.y[train_idx] == 1).sum()):,}"
    )

    print(
        f"  Val fraud  : "
        f"{int((graph.y[val_idx] == 1).sum()):,}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model(
        device
    )

    print(
        "\nModel:"
    )

    print(
        "  Architecture       : Original GraphSAGE"
    )

    print(
        f"  Input features     : {INPUT_DIM}"
    )

    print(
        f"  Hidden features    : {HIDDEN_DIM}"
    )

    print(
        f"  Output classes     : {OUTPUT_DIM}"
    )

    print(
        f"  Dropout            : {DROPOUT}"
    )

    print(
        f"  Parameters         : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # --------------------------------------------------------
    # Initial training
    # --------------------------------------------------------

    model = initial_training(
        model,
        graph,
        train_idx,
        val_idx,
        device
    )

    # --------------------------------------------------------
    # Mine hard negatives
    # --------------------------------------------------------

    hard_negative_idx = (
        mine_hard_negatives(
            model,
            graph,
            train_idx,
            device
        )
    )

    # --------------------------------------------------------
    # Hard-negative retraining
    # --------------------------------------------------------

    model, best_epoch = (
        hard_negative_training(
            model,
            graph,
            train_idx,
            val_idx,
            hard_negative_idx,
            device
        )
    )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    final_metrics = final_evaluation(
        model,
        graph,
        val_idx,
        device
    )

    # --------------------------------------------------------
    # Display final results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CLIENT 1 HARD NEGATIVE MINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nBest mining epoch     : "
        f"{best_epoch}"
    )

    print(
        f"Validation loss       : "
        f"{final_metrics['val_loss']:.6f}"
    )

    print(
        f"Accuracy              : "
        f"{final_metrics['accuracy']:.6f}"
    )

    print(
        f"Precision             : "
        f"{final_metrics['precision']:.6f}"
    )

    print(
        f"Recall                : "
        f"{final_metrics['recall']:.6f}"
    )

    print(
        f"F1                    : "
        f"{final_metrics['f1']:.6f}"
    )

    print(
        f"Specificity           : "
        f"{final_metrics['specificity']:.6f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        f"  TP: {final_metrics['tp']:,}"
    )

    print(
        f"  TN: {final_metrics['tn']:,}"
    )

    print(
        f"  FP: {final_metrics['fp']:,}"
    )

    print(
        f"  FN: {final_metrics['fn']:,}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_model(
        model,
        final_metrics,
        len(hard_negative_idx),
        best_epoch
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    result = {

        "client": 1,

        "seed": SEED,

        "graph": str(
            GRAPH_PATH
        ),

        "architecture":
            "Original GraphSAGE",

        "training_method":
            "Hard Negative Mining",

        "input_dim":
            INPUT_DIM,

        "hidden_dim":
            HIDDEN_DIM,

        "output_dim":
            OUTPUT_DIM,

        "dropout":
            DROPOUT,

        "validation_ratio":
            VAL_RATIO,

        "initial_epochs":
            INITIAL_EPOCHS,

        "mining_epochs":
            MINING_EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "hard_negative_fraction":
            HARD_NEGATIVE_FRACTION,

        "hard_negative_weight":
            HARD_NEGATIVE_WEIGHT,

        "hard_negative_count":
            len(hard_negative_idx),

        "best_epoch":
            best_epoch,

        "metrics":
            final_metrics,

        "model_path":
            str(
                OUTPUT_MODEL_PATH
            ),
    }

    OUTPUT_METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print(
        "\nMetrics saved to:"
    )

    print(
        OUTPUT_METRICS_PATH
    )

    print(
        "\n" + "=" * 70
    )


if __name__ == "__main__":
    main()