# ============================================================
# RFGN CLIENT 1 LOCAL GRAPHSAGE TRAINING
# FOCAL LOSS VERSION
# ============================================================
#
# Purpose:
#   Train the Client 1 local GraphSAGE fraud-detection model
#   using Focal Loss to improve learning on difficult and
#   minority fraud examples.
#
# Input:
#   graph_aligned.pt
#
# IMPORTANT:
#   - Uses only Client 1 training graph
#   - No validation/test CSV is used
#   - No federated aggregation is performed
#   - No Client 2/3 data is used
#   - Uses 769 shared features
#   - Uses Focal Loss instead of weighted Cross Entropy
#
# ============================================================


import os
import sys
import time
import random

import numpy as np
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
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# IMPORT MODEL
# ============================================================

from models.gnn.graphsage_model import GraphSAGE

from models.losses.focal_loss import FocalLoss


# ============================================================
# CONFIGURATION
# ============================================================

CLIENT_ID = "client_1"


GRAPH_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs",
    CLIENT_ID,
    "graph_aligned.pt",
)


MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "local",
)


CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "checkpoints",
)


BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "client_1_graphsage_focal_best.pt",
)


CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "client_1_graphsage_focal_checkpoint.pt",
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

INPUT_DIM = 769

HIDDEN_DIM = 128

OUTPUT_DIM = 2

DROPOUT = 0.30


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

LEARNING_RATE = 0.0005

WEIGHT_DECAY = 0.0001

MAX_EPOCHS = 30

VALIDATION_RATIO = 0.15

RANDOM_SEED = 42

EARLY_STOPPING_PATIENCE = 6

LR_PATIENCE = 3

LR_FACTOR = 0.5

GRADIENT_CLIP_NORM = 1.0


# ============================================================
# FOCAL LOSS CONFIGURATION
# ============================================================
#
# gamma controls how strongly the loss focuses on difficult
# examples.
#
# alpha provides class weighting.
#
# The fraud class receives the larger weight because the
# dataset is highly imbalanced.
#
# ============================================================

FOCAL_GAMMA = 2.0

FOCAL_ALPHA = [
    0.25,
    0.75,
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(
    seed
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


# ============================================================
# PRINT SECTION
# ============================================================

def print_section(
    title
):

    print()

    print(
        "=" * 70
    )

    print(
        title
    )

    print(
        "=" * 70
    )


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph():

    print_section(
        "LOADING CLIENT 1 ALIGNED GRAPH"
    )

    print(
        f"Path: {GRAPH_PATH}"
    )

    if not os.path.exists(
        GRAPH_PATH
    ):

        raise FileNotFoundError(
            "Aligned graph not found:\n"
            f"{GRAPH_PATH}"
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

    return graph


# ============================================================
# VALIDATE GRAPH
# ============================================================

def validate_graph(
    graph
):

    print_section(
        "VALIDATING TRAINING GRAPH"
    )

    # --------------------------------------------------------
    # Feature dimension
    # --------------------------------------------------------

    if (
        graph.num_node_features
        != INPUT_DIM
    ):

        raise RuntimeError(
            f"Expected {INPUT_DIM} features, "
            f"found "
            f"{graph.num_node_features}."
        )

    print(
        "Input feature dimension: PASS"
    )

    # --------------------------------------------------------
    # Feature rows
    # --------------------------------------------------------

    if (
        graph.x.shape[0]
        != graph.num_nodes
    ):

        raise RuntimeError(
            "Feature row count does not "
            "match node count."
        )

    print(
        "Feature row count: PASS"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    if (
        graph.y.shape[0]
        != graph.num_nodes
    ):

        raise RuntimeError(
            "Label count does not "
            "match node count."
        )

    print(
        "Label count: PASS"
    )

    # --------------------------------------------------------
    # Edge index
    # --------------------------------------------------------

    if (
        graph.edge_index.shape[0]
        != 2
    ):

        raise RuntimeError(
            "Invalid edge index shape."
        )

    print(
        "Edge index shape: PASS"
    )

    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    if torch.isnan(
        graph.x
    ).any():

        raise RuntimeError(
            "NaN values detected in features."
        )

    print(
        "Feature NaN check: PASS"
    )

    # --------------------------------------------------------
    # Infinity
    # --------------------------------------------------------

    if torch.isinf(
        graph.x
    ).any():

        raise RuntimeError(
            "Infinite values detected in features."
        )

    print(
        "Feature infinity check: PASS"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    unique_labels = (
        torch.unique(
            graph.y
        )
        .cpu()
        .tolist()
    )

    if not set(
        unique_labels
    ).issubset(
        {0, 1}
    ):

        raise RuntimeError(
            f"Invalid labels: "
            f"{unique_labels}"
        )

    print(
        "Binary labels: PASS"
    )

    # --------------------------------------------------------
    # Edge bounds
    # --------------------------------------------------------

    if graph.edge_index.numel() > 0:

        minimum = int(
            graph.edge_index
            .min()
            .item()
        )

        maximum = int(
            graph.edge_index
            .max()
            .item()
        )

        if minimum < 0:

            raise RuntimeError(
                "Negative edge index detected."
            )

        if maximum >= graph.num_nodes:

            raise RuntimeError(
                "Edge index exceeds node count."
            )

    print(
        "Edge bounds: PASS"
    )


# ============================================================
# CREATE STRATIFIED SPLIT
# ============================================================

def create_split(
    labels,
    validation_ratio,
    seed,
):

    print_section(
        "CREATING LOCAL TRAINING / VALIDATION SPLIT"
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    labels = labels.cpu()

    class_zero = torch.where(
        labels == 0
    )[0]

    class_one = torch.where(
        labels == 1
    )[0]

    # --------------------------------------------------------
    # Shuffle each class independently
    # --------------------------------------------------------

    class_zero = class_zero[
        torch.randperm(
            len(class_zero),
            generator=generator,
        )
    ]

    class_one = class_one[
        torch.randperm(
            len(class_one),
            generator=generator,
        )
    ]

    # --------------------------------------------------------
    # Validation counts
    # --------------------------------------------------------

    val_zero_count = int(
        len(class_zero)
        * validation_ratio
    )

    val_one_count = int(
        len(class_one)
        * validation_ratio
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_indices = torch.cat(
        [
            class_zero[
                :val_zero_count
            ],

            class_one[
                :val_one_count
            ],
        ]
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_indices = torch.cat(
        [
            class_zero[
                val_zero_count:
            ],

            class_one[
                val_one_count:
            ],
        ]
    )

    # --------------------------------------------------------
    # Shuffle final sets
    # --------------------------------------------------------

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

    return (
        train_indices,
        val_indices,
    )


# ============================================================
# PRINT SPLIT SUMMARY
# ============================================================

def print_split_summary(
    labels,
    train_indices,
    val_indices,
):

    train_labels = labels[
        train_indices
    ]

    val_labels = labels[
        val_indices
    ]

    train_legitimate = int(
        (
            train_labels == 0
        )
        .sum()
        .item()
    )

    train_fraud = int(
        (
            train_labels == 1
        )
        .sum()
        .item()
    )

    val_legitimate = int(
        (
            val_labels == 0
        )
        .sum()
        .item()
    )

    val_fraud = int(
        (
            val_labels == 1
        )
        .sum()
        .item()
    )

    train_total = len(
        train_indices
    )

    val_total = len(
        val_indices
    )

    print()

    print(
        "LOCAL TRAINING NODES"
    )

    print(
        f"  Rows       : "
        f"{train_total:,}"
    )

    print(
        f"  Legitimate : "
        f"{train_legitimate:,}"
    )

    print(
        f"  Fraud      : "
        f"{train_fraud:,}"
    )

    print(
        f"  Fraud rate : "
        f"{train_fraud / train_total * 100:.4f}%"
    )

    print()

    print(
        "LOCAL VALIDATION NODES"
    )

    print(
        f"  Rows       : "
        f"{val_total:,}"
    )

    print(
        f"  Legitimate : "
        f"{val_legitimate:,}"
    )

    print(
        f"  Fraud      : "
        f"{val_fraud:,}"
    )

    print(
        f"  Fraud rate : "
        f"{val_fraud / val_total * 100:.4f}%"
    )


# ============================================================
# CREATE FOCAL LOSS
# ============================================================

def create_focal_loss(
    device
):

    print_section(
        "CREATING FOCAL LOSS"
    )

    criterion = FocalLoss(
        alpha=FOCAL_ALPHA,
        gamma=FOCAL_GAMMA,
        reduction="mean",
    )

    criterion = criterion.to(
        device
    )

    print(
        "Focal Loss: CREATED"
    )

    print()

    print(
        f"Gamma: "
        f"{FOCAL_GAMMA}"
    )

    print(
        f"Alpha legitimate: "
        f"{FOCAL_ALPHA[0]}"
    )

    print(
        f"Alpha fraud: "
        f"{FOCAL_ALPHA[1]}"
    )

    if (
        FOCAL_ALPHA[1]
        <= FOCAL_ALPHA[0]
    ):

        raise RuntimeError(
            "Fraud class must receive "
            "the higher alpha weight."
        )

    print(
        "Fraud class emphasis: PASS"
    )

    return criterion


# ============================================================
# MODEL CREATION
# ============================================================

def create_model():

    print_section(
        "CREATING GRAPHSAGE MODEL"
    )

    model = GraphSAGE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
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
        "GraphSAGE model: CREATED"
    )

    print()

    print(
        "Model configuration:"
    )

    print(
        f"input_dim            : "
        f"{INPUT_DIM}"
    )

    print(
        f"hidden_dim           : "
        f"{HIDDEN_DIM}"
    )

    print(
        f"output_dim           : "
        f"{OUTPUT_DIM}"
    )

    print(
        f"dropout              : "
        f"{DROPOUT}"
    )

    print(
        f"total_parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"trainable_parameters : "
        f"{trainable_parameters:,}"
    )

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    labels,
):

    predictions = predictions.cpu()

    labels = labels.cpu()

    true_positive = int(
        (
            (predictions == 1)
            & (labels == 1)
        )
        .sum()
        .item()
    )

    false_positive = int(
        (
            (predictions == 1)
            & (labels == 0)
        )
        .sum()
        .item()
    )

    false_negative = int(
        (
            (predictions == 0)
            & (labels == 1)
        )
        .sum()
        .item()
    )

    true_negative = int(
        (
            (predictions == 0)
            & (labels == 0)
        )
        .sum()
        .item()
    )

    correct = (
        true_positive
        + true_negative
    )

    total = len(
        labels
    )

    precision_denominator = (
        true_positive
        + false_positive
    )

    recall_denominator = (
        true_positive
        + false_negative
    )

    specificity_denominator = (
        true_negative
        + false_positive
    )

    precision = (
        true_positive
        /
        precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        true_positive
        /
        recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    specificity = (
        true_negative
        /
        specificity_denominator
        if specificity_denominator > 0
        else 0.0
    )

    f1_denominator = (
        precision
        + recall
    )

    f1 = (
        2.0
        * precision
        * recall
        /
        f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    accuracy = (
        correct
        /
        total
        if total > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


# ============================================================
# VALIDATE LOSS
# ============================================================

def validate_loss(
    loss,
    name,
):

    if not torch.isfinite(
        loss
    ).item():

        raise RuntimeError(
            f"{name} is not finite: "
            f"{loss.item()}"
        )

    if loss.item() < 0:

        raise RuntimeError(
            f"{name} is negative: "
            f"{loss.item()}"
        )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    graph,
    train_indices,
    optimizer,
    criterion,
):

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    output = model(
        graph.x,
        graph.edge_index,
    )

    train_output = output[
        train_indices
    ]

    train_labels = graph.y[
        train_indices
    ]

    loss = criterion(
        train_output,
        train_labels,
    )

    validate_loss(
        loss,
        "Training loss",
    )

    loss.backward()

    # --------------------------------------------------------
    # Gradient validation
    # --------------------------------------------------------

    total_gradient_norm = 0.0

    for parameter in model.parameters():

        if parameter.grad is None:

            continue

        if not torch.isfinite(
            parameter.grad
        ).all().item():

            raise RuntimeError(
                "Non-finite gradient detected."
            )

        gradient_norm = (
            parameter.grad
            .detach()
            .norm(2)
            .item()
        )

        total_gradient_norm += (
            gradient_norm ** 2
        )

    total_gradient_norm = (
        total_gradient_norm
        ** 0.5
    )

    if not np.isfinite(
        total_gradient_norm
    ):

        raise RuntimeError(
            "Gradient norm is not finite."
        )

    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRADIENT_CLIP_NORM,
    )

    optimizer.step()

    predictions = (
        train_output
        .argmax(
            dim=1
        )
    )

    metrics = calculate_metrics(
        predictions,
        train_labels,
    )

    return (
        float(loss.item()),
        metrics,
        total_gradient_norm,
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    graph,
    val_indices,
    criterion,
):

    model.eval()

    output = model(
        graph.x,
        graph.edge_index,
    )

    val_output = output[
        val_indices
    ]

    val_labels = graph.y[
        val_indices
    ]

    loss = criterion(
        val_output,
        val_labels,
    )

    validate_loss(
        loss,
        "Validation loss",
    )

    predictions = (
        val_output
        .argmax(
            dim=1
        )
    )

    metrics = calculate_metrics(
        predictions,
        val_labels,
    )

    return (
        float(loss.item()),
        metrics,
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
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    checkpoint = {

        "client_id":
            CLIENT_ID,

        "loss_function":
            "FocalLoss",

        "focal_gamma":
            FOCAL_GAMMA,

        "focal_alpha":
            FOCAL_ALPHA,

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

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

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "random_seed":
            RANDOM_SEED,

        "graph_file":
            GRAPH_PATH,

        "feature_count":
            INPUT_DIM,
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print()

    print(
        "=" * 70
    )

    print(
        "RFGN CLIENT 1 LOCAL GRAPHSAGE TRAINING"
    )

    print(
        "FOCAL LOSS VERSION"
    )

    print(
        "=" * 70
    )

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

    print(
        f"  Early stopping       : "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    print(
        f"  Gradient clip        : "
        f"{GRADIENT_CLIP_NORM}"
    )

    print()

    print(
        "Focal Loss configuration:"
    )

    print(
        f"  Gamma                : "
        f"{FOCAL_GAMMA}"
    )

    print(
        f"  Alpha legitimate     : "
        f"{FOCAL_ALPHA[0]}"
    )

    print(
        f"  Alpha fraud         : "
        f"{FOCAL_ALPHA[1]}"
    )

    print()

    print(
        "Loss function:"
    )

    print(
        "  Focal Loss"
    )

    print(
        "  Weighted Cross Entropy: DISABLED"
    )

    # ========================================================
    # SEED
    # ========================================================

    set_seed(
        RANDOM_SEED
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()

    print_section(
        "SELECTING DEVICE"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        total_memory = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
            /
            (1024 ** 3)
        )

        print(
            f"GPU memory: "
            f"{total_memory:.2f} GB"
        )

    # ========================================================
    # LOAD GRAPH
    # ========================================================

    graph = load_graph()

    # ========================================================
    # VALIDATE GRAPH
    # ========================================================

    validate_graph(
        graph
    )

    # ========================================================
    # CREATE SPLIT
    # ========================================================

    (
        train_indices,
        val_indices,
    ) = create_split(
        graph.y,
        VALIDATION_RATIO,
        RANDOM_SEED,
    )

    print_split_summary(
        graph.y,
        train_indices,
        val_indices,
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = create_model()

    model = model.to(
        device
    )

    # ========================================================
    # CREATE FOCAL LOSS
    # ========================================================

    criterion = create_focal_loss(
        device
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=1e-6,
    )

    print()

    print(
        "AdamW optimizer: READY"
    )

    print(
        "Learning-rate scheduler: READY"
    )

    print(
        "Focal Loss: READY"
    )

    print(
        "Gradient clipping: READY"
    )

    print(
        "Early stopping: READY"
    )

    # ========================================================
    # MOVE GRAPH TO DEVICE
    # ========================================================

    print_section(
        "MOVING GRAPH TO DEVICE"
    )

    graph = graph.to(
        device
    )

    train_indices = (
        train_indices.to(
            device
        )
    )

    val_indices = (
        val_indices.to(
            device
        )
    )

    print(
        "Graph transferred to device: PASS"
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print_section(
        "STARTING LOCAL FOCAL-LOSS TRAINING"
    )

    print(
        "Training uses the aligned 769-feature graph."
    )

    print(
        "Focal Loss is active."
    )

    print(
        "Weighted Cross Entropy is not used."
    )

    print(
        "Loss and gradient safety checks are enabled."
    )

    print(
        "Training will stop automatically when "
        "validation loss stops improving."
    )

    best_val_loss = float(
        "inf"
    )

    patience_counter = 0

    best_epoch = 0

    best_metrics = None

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        epoch_start = time.time()

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        (
            train_loss,
            train_metrics,
            gradient_norm,
        ) = train_one_epoch(
            model,
            graph,
            train_indices,
            optimizer,
            criterion,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            val_loss,
            val_metrics,
        ) = validate(
            model,
            graph,
            val_indices,
            criterion,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        # ----------------------------------------------------
        # Improvement
        # ----------------------------------------------------

        improved = (
            val_loss
            <
            best_val_loss
        )

        if improved:

            best_val_loss = val_loss

            best_epoch = epoch

            patience_counter = 0

            best_metrics = (
                val_metrics.copy()
            )

            os.makedirs(
                MODEL_DIR,
                exist_ok=True,
            )

            torch.save(
                {
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

                    "feature_count":
                        INPUT_DIM,

                    "best_epoch":
                        epoch,

                    "best_val_loss":
                        best_val_loss,

                    "best_validation_metrics":
                        best_metrics,

                    "loss_function":
                        "FocalLoss",

                    "focal_gamma":
                        FOCAL_GAMMA,

                    "focal_alpha":
                        FOCAL_ALPHA,

                    "graph":
                        "graph_aligned.pt",
                },
                BEST_MODEL_PATH,
            )

        else:

            patience_counter += 1

        # ----------------------------------------------------
        # Checkpoint
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
        # GPU memory
        # ----------------------------------------------------

        gpu_memory_text = ""

        if device.type == "cuda":

            allocated = (
                torch.cuda
                .memory_allocated()
                /
                (1024 ** 3)
            )

            gpu_memory_text = (
                f"  GPU memory      : "
                f"{allocated:.3f} GB"
            )

        # ----------------------------------------------------
        # Epoch output
        # ----------------------------------------------------

        print()

        print(
            f"Epoch "
            f"{epoch:03d}/{MAX_EPOCHS}"
        )

        print(
            f"  Train loss       : "
            f"{train_loss:.6f}"
        )

        print(
            f"  Validation loss  : "
            f"{val_loss:.6f}"
        )

        print(
            f"  Train Precision  : "
            f"{train_metrics['precision']:.4f}"
        )

        print(
            f"  Train Recall     : "
            f"{train_metrics['recall']:.4f}"
        )

        print(
            f"  Train F1         : "
            f"{train_metrics['f1']:.4f}"
        )

        print(
            f"  Train Accuracy   : "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            f"  Val Precision    : "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"  Val Recall       : "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            f"  Val F1           : "
            f"{val_metrics['f1']:.4f}"
        )

        print(
            f"  Val Accuracy     : "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"  Val Specificity  : "
            f"{val_metrics['specificity']:.4f}"
        )

        print(
            f"  Gradient norm    : "
            f"{gradient_norm:.6f}"
        )

        print(
            f"  Learning rate    : "
            f"{current_lr:.8f}"
        )

        print(
            f"  Epoch time       : "
            f"{epoch_time:.2f}s"
        )

        if improved:

            print(
                "  Best model: SAVED"
            )

        else:

            print(
                f"  No improvement: "
                f"{patience_counter}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        if gpu_memory_text:

            print(
                gpu_memory_text
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print()

            print(
                "Early stopping triggered."
            )

            break

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    print()

    print(
        "=" * 70
    )

    print(
        "CLIENT 1 FOCAL-LOSS TRAINING COMPLETED"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    if best_metrics is not None:

        print()

        print(
            "Best validation metrics:"
        )

        print(
            f"  Precision   : "
            f"{best_metrics['precision']:.4f}"
        )

        print(
            f"  Recall      : "
            f"{best_metrics['recall']:.4f}"
        )

        print(
            f"  F1          : "
            f"{best_metrics['f1']:.4f}"
        )

        print(
            f"  Accuracy    : "
            f"{best_metrics['accuracy']:.4f}"
        )

        print(
            f"  Specificity : "
            f"{best_metrics['specificity']:.4f}"
        )

        print()

        print(
            "Confusion matrix:"
        )

        print(
            f"  TP: "
            f"{best_metrics['true_positive']:,}"
        )

        print(
            f"  TN: "
            f"{best_metrics['true_negative']:,}"
        )

        print(
            f"  FP: "
            f"{best_metrics['false_positive']:,}"
        )

        print(
            f"  FN: "
            f"{best_metrics['false_negative']:,}"
        )

    print()

    print(
        f"Training time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()

    print(
        "Loss function:"
    )

    print(
        "Focal Loss"
    )

    print()

    print(
        "Focal gamma:"
    )

    print(
        FOCAL_GAMMA
    )

    print()

    print(
        "Focal alpha:"
    )

    print(
        FOCAL_ALPHA
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

    print(
        "=" * 70
    )

    print(
        "CLIENT 1 LOCAL FOCAL-LOSS TRAINING: PASS"
    )

    print(
        "=" * 70
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()