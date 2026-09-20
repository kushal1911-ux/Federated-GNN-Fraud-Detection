# ============================================================
# RFGN CLIENT 3 LOCAL GRAPHSAGE TRAINING
# ============================================================

import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


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
# TRAINING CONFIGURATION
# ============================================================

CLIENT_ID = "client_3"

INPUT_DIM = 769
HIDDEN_DIM = 128
OUTPUT_DIM = 2
DROPOUT = 0.3

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

MAX_EPOCHS = 30

VALIDATION_RATIO = 0.15

RANDOM_SEED = 42

PATIENCE = 8


# ============================================================
# PATHS
# ============================================================

GRAPH_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs",
    CLIENT_ID,
    "graph_aligned.pt",
)

BEST_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "local",
    "client_3_graphsage_aligned_best.pt",
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "checkpoints",
    "client_3_graphsage_aligned_checkpoint.pt",
)


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SECTION
# ============================================================

def section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# GRAPH VALIDATION
# ============================================================

def validate_graph(graph):

    section(
        "VALIDATING TRAINING GRAPH"
    )

    if graph.num_node_features != INPUT_DIM:

        raise RuntimeError(
            f"Expected {INPUT_DIM} features, "
            f"found {graph.num_node_features}."
        )

    print(
        "Input feature dimension: PASS"
    )


    if graph.x.shape[0] != graph.num_nodes:

        raise RuntimeError(
            "Feature row count does not match node count."
        )

    print(
        "Feature row count: PASS"
    )


    if graph.y.shape[0] != graph.num_nodes:

        raise RuntimeError(
            "Label count does not match node count."
        )

    print(
        "Label count: PASS"
    )


    if torch.isnan(graph.x).any():

        raise RuntimeError(
            "NaN values found in graph features."
        )

    print(
        "Feature NaN check: PASS"
    )


    if torch.isinf(graph.x).any():

        raise RuntimeError(
            "Infinite values found in graph features."
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


        if minimum_index < 0:

            raise RuntimeError(
                "Negative node index detected."
            )


        if maximum_index >= graph.num_nodes:

            raise RuntimeError(
                "Edge index exceeds node count."
            )


    print(
        "Edge bounds: PASS"
    )


    unique_labels = sorted(
        torch.unique(
            graph.y
        ).cpu().tolist()
    )


    if not set(
        unique_labels
    ).issubset({0, 1}):

        raise RuntimeError(
            "Labels are not binary."
        )


    print(
        "Binary labels: PASS"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    labels
):

    section(
        "CALCULATING CLASS WEIGHTS"
    )


    labels = labels.long()


    class_counts = torch.bincount(
        labels,
        minlength=2,
    ).float()


    total = class_counts.sum()


    weights = (
        total
        /
        (
            2.0
            *
            class_counts
        )
    )


    weights = weights.float()


    print(
        f"Legitimate weight: "
        f"{weights[0].item():.6f}"
    )


    print(
        f"Fraud weight     : "
        f"{weights[1].item():.6f}"
    )


    if weights[1] <= weights[0]:

        raise RuntimeError(
            "Fraud class did not receive higher weight."
        )


    print(
        "Fraud class weighting: PASS"
    )


    return weights


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def create_split(
    labels
):

    section(
        "CREATING LOCAL TRAINING / VALIDATION SPLIT"
    )


    labels = labels.cpu()


    generator = torch.Generator()

    generator.manual_seed(
        RANDOM_SEED
    )


    fraud_indices = torch.where(
        labels == 1
    )[0]


    legitimate_indices = torch.where(
        labels == 0
    )[0]


    fraud_indices = fraud_indices[
        torch.randperm(
            len(fraud_indices),
            generator=generator,
        )
    ]


    legitimate_indices = legitimate_indices[
        torch.randperm(
            len(legitimate_indices),
            generator=generator,
        )
    ]


    fraud_val_count = max(
        1,
        int(
            len(fraud_indices)
            *
            VALIDATION_RATIO
        ),
    )


    legitimate_val_count = max(
        1,
        int(
            len(legitimate_indices)
            *
            VALIDATION_RATIO
        ),
    )


    val_indices = torch.cat(
        [
            fraud_indices[
                :fraud_val_count
            ],

            legitimate_indices[
                :legitimate_val_count
            ],
        ]
    )


    train_indices = torch.cat(
        [
            fraud_indices[
                fraud_val_count:
            ],

            legitimate_indices[
                legitimate_val_count:
            ],
        ]
    )


    train_indices = train_indices[
        torch.randperm(
            len(train_indices),
            generator=generator,
        )
    ]


    val_indices = val_indices[
        torch.randperm(
            len(val_indices),
            generator=generator,
        )
    ]


    train_labels = labels[
        train_indices
    ]


    val_labels = labels[
        val_indices
    ]


    print()

    print(
        "LOCAL TRAINING NODES"
    )


    print(
        f"  Rows       : "
        f"{len(train_indices):,}"
    )


    print(
        f"  Legitimate : "
        f"{int((train_labels == 0).sum()):,}"
    )


    print(
        f"  Fraud      : "
        f"{int((train_labels == 1).sum()):,}"
    )


    print(
        f"  Fraud rate : "
        f"{(train_labels == 1).float().mean().item() * 100:.4f}%"
    )


    print()

    print(
        "LOCAL VALIDATION NODES"
    )


    print(
        f"  Rows       : "
        f"{len(val_indices):,}"
    )


    print(
        f"  Legitimate : "
        f"{int((val_labels == 0).sum()):,}"
    )


    print(
        f"  Fraud      : "
        f"{int((val_labels == 1).sum()):,}"
    )


    print(
        f"  Fraud rate : "
        f"{(val_labels == 1).float().mean().item() * 100:.4f}%"
    )


    return (
        train_indices,
        val_indices,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    labels
):

    predictions = (
        predictions
        .detach()
        .cpu()
    )


    labels = (
        labels
        .detach()
        .cpu()
    )


    precision = precision_score(
        labels.numpy(),
        predictions.numpy(),
        zero_division=0,
    )


    recall = recall_score(
        labels.numpy(),
        predictions.numpy(),
        zero_division=0,
    )


    f1 = f1_score(
        labels.numpy(),
        predictions.numpy(),
        zero_division=0,
    )


    accuracy = accuracy_score(
        labels.numpy(),
        predictions.numpy(),
    )


    return (
        precision,
        recall,
        f1,
        accuracy,
    )


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_best_model(
    model,
    epoch,
    val_loss,
):

    os.makedirs(
        os.path.dirname(
            BEST_MODEL_PATH
        ),
        exist_ok=True,
    )


    checkpoint = {

        "client_id":
            CLIENT_ID,

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

        "best_epoch":
            epoch,

        "best_val_loss":
            val_loss,
    }


    torch.save(
        checkpoint,
        BEST_MODEL_PATH,
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss,
    patience_counter,
):

    os.makedirs(
        os.path.dirname(
            CHECKPOINT_PATH
        ),
        exist_ok=True,
    )


    checkpoint = {

        "client_id":
            CLIENT_ID,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "epoch":
            epoch,

        "best_val_loss":
            best_val_loss,

        "patience_counter":
            patience_counter,

        "input_dim":
            INPUT_DIM,

        "hidden_dim":
            HIDDEN_DIM,

        "output_dim":
            OUTPUT_DIM,

        "dropout":
            DROPOUT,
    }


    torch.save(
        checkpoint,
        CHECKPOINT_PATH,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        RANDOM_SEED
    )


    print()

    print("=" * 70)

    print(
        "RFGN CLIENT 3 LOCAL GRAPHSAGE TRAINING"
    )

    print("=" * 70)


    print()

    print(
        "Training configuration:"
    )


    print(
        f"  Input dimension      : "
        f"{INPUT_DIM}"
    )


    print(
        f"  Hidden dimension     : "
        f"{HIDDEN_DIM}"
    )


    print(
        f"  Learning rate        : "
        f"{LEARNING_RATE}"
    )


    print(
        f"  Weight decay         : "
        f"{WEIGHT_DECAY}"
    )


    print(
        f"  Maximum epochs       : "
        f"{MAX_EPOCHS}"
    )


    print(
        f"  Validation ratio     : "
        f"{VALIDATION_RATIO}"
    )


    print(
        f"  Random seed          : "
        f"{RANDOM_SEED}"
    )


    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()


    print()

    print(
        f"Device: {device}"
    )


    # ========================================================
    # GRAPH
    # ========================================================

    section(
        "LOADING CLIENT 3 ALIGNED GRAPH"
    )


    if not os.path.exists(
        GRAPH_PATH
    ):

        raise FileNotFoundError(
            f"Graph not found:\n{GRAPH_PATH}"
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
        "Graph loaded successfully."
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
    # VALIDATE
    # ========================================================

    validate_graph(
        graph
    )


    # ========================================================
    # SPLIT
    # ========================================================

    (
        train_indices,
        val_indices,
    ) = create_split(
        graph.y
    )


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = (
        calculate_class_weights(
            graph.y[
                train_indices
            ]
        )
    )


    # ========================================================
    # MODEL
    # ========================================================

    section(
        "CREATING GRAPHSAGE MODEL"
    )


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
        p.numel()
        for p in model.parameters()
    )


    print(
        f"Total parameters : "
        f"{total_parameters:,}"
    )


    if total_parameters != 230146:

        raise RuntimeError(
            "Unexpected model parameter count."
        )


    print(
        "Model parameter count: PASS"
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    print(
        "AdamW optimizer: READY"
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )


    print(
        "Learning-rate scheduler: READY"
    )


    # ========================================================
    # LOSS
    # ========================================================

    class_weights = class_weights.to(
        device
    )


    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )


    print(
        "Weighted loss: READY"
    )


    print(
        "Early stopping: READY"
    )


    # ========================================================
    # DEVICE TRANSFER
    # ========================================================

    section(
        "MOVING GRAPH TO DEVICE"
    )


    model = model.to(
        device
    )


    graph = graph.to(
        device
    )


    train_indices = train_indices.to(
        device
    )


    val_indices = val_indices.to(
        device
    )


    print(
        "Graph transferred to device: PASS"
    )


    # ========================================================
    # TRAINING
    # ========================================================

    section(
        "STARTING LOCAL TRAINING"
    )


    print()

    print(
        "Training will stop automatically "
        "when validation loss stops improving."
    )


    best_val_loss = float(
        "inf"
    )


    patience_counter = 0


    training_start = time.time()


    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        epoch_start = time.time()


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()


        optimizer.zero_grad(
            set_to_none=True
        )


        logits = model(
            graph.x,
            graph.edge_index,
        )


        train_logits = logits[
            train_indices
        ]


        train_labels = graph.y[
            train_indices
        ]


        train_loss = criterion(
            train_logits,
            train_labels,
        )


        if not torch.isfinite(
            train_loss
        ):

            raise RuntimeError(
                f"Non-finite training loss "
                f"at epoch {epoch}."
            )


        train_loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )


        optimizer.step()


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()


        with torch.no_grad():

            val_logits = model(
                graph.x,
                graph.edge_index,
            )


            val_logits = val_logits[
                val_indices
            ]


            val_labels = graph.y[
                val_indices
            ]


            val_loss = criterion(
                val_logits,
                val_labels,
            )


            probabilities = F.softmax(
                val_logits,
                dim=1,
            )


            predictions = probabilities.argmax(
                dim=1
            )


        if not torch.isfinite(
            val_loss
        ):

            raise RuntimeError(
                f"Non-finite validation loss "
                f"at epoch {epoch}."
            )


        (
            precision,
            recall,
            f1,
            accuracy,
        ) = calculate_metrics(
            predictions,
            val_labels,
        )


        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        scheduler.step(
            val_loss.item()
        )


        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )


        epoch_time = (
            time.time()
            -
            epoch_start
        )


        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        is_best = (
            val_loss.item()
            <
            best_val_loss
        )


        if is_best:

            best_val_loss = (
                val_loss.item()
            )

            patience_counter = 0


            save_best_model(
                model,
                epoch,
                best_val_loss,
            )


        else:

            patience_counter += 1


        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            patience_counter,
        )


        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        print()

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS}"
        )


        print(
            f"  Train loss      : "
            f"{train_loss.item():.6f}"
        )


        print(
            f"  Validation loss : "
            f"{val_loss.item():.6f}"
        )


        print(
            f"  Precision       : "
            f"{precision:.4f}"
        )


        print(
            f"  Recall          : "
            f"{recall:.4f}"
        )


        print(
            f"  F1              : "
            f"{f1:.4f}"
        )


        print(
            f"  Accuracy        : "
            f"{accuracy:.4f}"
        )


        print(
            f"  Learning rate   : "
            f"{current_lr:.8f}"
        )


        print(
            f"  Epoch time      : "
            f"{epoch_time:.2f}s"
        )


        if is_best:

            print(
                "  Best model: SAVED"
            )

        else:

            print(
                f"  Patience        : "
                f"{patience_counter}/{PATIENCE}"
            )


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if patience_counter >= PATIENCE:

            print()

            print(
                "Early stopping triggered."
            )

            break


    # ========================================================
    # COMPLETE
    # ========================================================

    total_time = (
        time.time()
        -
        training_start
    )


    print()

    print("=" * 70)

    print(
        "CLIENT 3 TRAINING COMPLETED"
    )

    print("=" * 70)


    print()

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )


    print(
        f"Training time: "
        f"{total_time / 60:.2f} minutes"
    )


    print()

    print(
        "Best model:"
    )


    print(
        BEST_MODEL_PATH
    )


    print()

    print(
        "Checkpoint:"
    )


    print(
        CHECKPOINT_PATH
    )


    if not os.path.exists(
        BEST_MODEL_PATH
    ):

        raise RuntimeError(
            "Best model file was not created."
        )


    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        raise RuntimeError(
            "Checkpoint file was not created."
        )


    print()

    print(
        "Best model file: PRESENT"
    )


    print(
        "Checkpoint file: PRESENT"
    )


    print()

    print("=" * 70)

    print(
        "CLIENT 3 LOCAL GRAPHSAGE TRAINING: COMPLETE"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()