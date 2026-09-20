# ============================================================
# RFGN - CLIENT 3 ENHANCED GRAPHSAGE TRAINING
# ============================================================

import os
import sys
import json
import random

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
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
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# MODEL
# ============================================================

from models.gnn.enhanced_graphsage_model import (
    EnhancedGraphSAGE
)


# ============================================================
# PATHS
# ============================================================

GRAPH_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs",
    "client_3",
    "graph_enhanced_aligned.pt",
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "local",
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "client_3_enhanced_graphsage_best.pt",
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "client_3_enhanced_graphsage_metrics.json",
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

INPUT_DIM = 769

HIDDEN_DIM = 128

OUTPUT_DIM = 2

DROPOUT = 0.30

LEARNING_RATE = 0.0005

WEIGHT_DECAY = 0.0001

EPOCHS = 40

VALIDATION_RATIO = 0.15

PATIENCE = 8

GRAD_CLIP = 1.0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

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
# LOAD GRAPH
# ============================================================

def load_graph():

    if not os.path.exists(
        GRAPH_PATH
    ):

        raise FileNotFoundError(
            f"Graph not found:\n{GRAPH_PATH}"
        )

    print()

    print(
        "Loading Client 3 enhanced graph..."
    )

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False
    )

    required_attributes = [
        "x",
        "edge_index",
        "y",
    ]

    for attribute in required_attributes:

        if not hasattr(
            graph,
            attribute
        ):

            raise ValueError(
                f"Graph is missing required "
                f"attribute: {attribute}"
            )

    return graph


# ============================================================
# VALIDATE GRAPH
# ============================================================

def validate_graph(graph):

    x = graph.x

    edge_index = graph.edge_index

    y = graph.y

    print()

    print(
        "Graph information:"
    )

    print(
        f"  Nodes          : "
        f"{x.shape[0]:,}"
    )

    print(
        f"  Features       : "
        f"{x.shape[1]}"
    )

    print(
        f"  Directed edges : "
        f"{edge_index.shape[1]:,}"
    )

    print(
        f"  Labels         : "
        f"{y.shape[0]:,}"
    )

    if hasattr(
        graph,
        "edge_type"
    ):

        print(
            f"  Edge types     : "
            f"{torch.unique(graph.edge_type).tolist()}"
        )

    if x.shape[1] != INPUT_DIM:

        raise ValueError(
            f"Expected {INPUT_DIM} features, "
            f"received {x.shape[1]}."
        )

    if x.shape[0] != y.shape[0]:

        raise ValueError(
            "Number of nodes and labels "
            "do not match."
        )

    if edge_index.ndim != 2:

        raise ValueError(
            "edge_index must have shape [2, E]."
        )

    if edge_index.shape[0] != 2:

        raise ValueError(
            "edge_index first dimension must be 2."
        )

    if not torch.isfinite(
        x
    ).all():

        raise ValueError(
            "Graph features contain NaN or Inf."
        )

    if not torch.isfinite(
        y.float()
    ).all():

        raise ValueError(
            "Graph labels contain NaN or Inf."
        )

    unique_labels = torch.unique(
        y
    )

    print()

    print(
        f"  Classes         : "
        f"{unique_labels.tolist()}"
    )

    for label in unique_labels.tolist():

        count = int(
            (y == label).sum().item()
        )

        print(
            f"  Class {label} count : "
            f"{count:,}"
        )


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def create_stratified_split(
    labels,
    validation_ratio,
    seed
):

    labels_np = labels.cpu().numpy()

    rng = np.random.default_rng(
        seed
    )

    train_indices = []

    validation_indices = []

    unique_classes = np.unique(
        labels_np
    )

    for class_id in unique_classes:

        class_indices = np.where(
            labels_np == class_id
        )[0]

        rng.shuffle(
            class_indices
        )

        validation_count = int(
            len(class_indices)
            * validation_ratio
        )

        validation_count = max(
            1,
            validation_count
        )

        val_part = class_indices[
            :validation_count
        ]

        train_part = class_indices[
            validation_count:
        ]

        validation_indices.extend(
            val_part.tolist()
        )

        train_indices.extend(
            train_part.tolist()
        )

    rng.shuffle(
        train_indices
    )

    rng.shuffle(
        validation_indices
    )

    train_mask = torch.zeros(
        len(labels),
        dtype=torch.bool
    )

    validation_mask = torch.zeros(
        len(labels),
        dtype=torch.bool
    )

    train_mask[
        torch.tensor(
            train_indices,
            dtype=torch.long
        )
    ] = True

    validation_mask[
        torch.tensor(
            validation_indices,
            dtype=torch.long
        )
    ] = True

    return (
        train_mask,
        validation_mask
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions
):

    labels_np = labels.cpu().numpy()

    predictions_np = predictions.cpu().numpy()

    accuracy = accuracy_score(
        labels_np,
        predictions_np
    )

    precision = precision_score(
        labels_np,
        predictions_np,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        labels_np,
        predictions_np,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        labels_np,
        predictions_np,
        pos_label=1,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        labels_np,
        predictions_np,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "specificity":
            float(specificity),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    }


# ============================================================
# TRAIN
# ============================================================

def train():

    set_seed(
        SEED
    )

    device = get_device()

    print()

    print(
        "=" * 70
    )

    print(
        "RFGN CLIENT 3 ENHANCED GRAPHSAGE TRAINING"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    graph = load_graph()

    validate_graph(
        graph
    )

    # --------------------------------------------------------
    # Move tensors
    # --------------------------------------------------------

    x = graph.x.float().to(
        device
    )

    edge_index = graph.edge_index.long().to(
        device
    )

    y = graph.y.long().to(
        device
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_mask,
        validation_mask
    ) = create_stratified_split(
        y,
        VALIDATION_RATIO,
        SEED
    )

    train_mask = train_mask.to(
        device
    )

    validation_mask = validation_mask.to(
        device
    )

    train_count = int(
        train_mask.sum().item()
    )

    validation_count = int(
        validation_mask.sum().item()
    )

    print()

    print(
        "Data split:"
    )

    print(
        f"  Training   : "
        f"{train_count:,}"
    )

    print(
        f"  Validation : "
        f"{validation_count:,}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = EnhancedGraphSAGE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    ).to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()

    print(
        "Model:"
    )

    print(
        "  Architecture       : "
        "Enhanced GraphSAGE"
    )

    print(
        f"  Input features     : "
        f"{INPUT_DIM}"
    )

    print(
        f"  Hidden features    : "
        f"{HIDDEN_DIM}"
    )

    print(
        f"  Output classes     : "
        f"{OUTPUT_DIM}"
    )

    print(
        f"  Dropout            : "
        f"{DROPOUT}"
    )

    print(
        f"  Parameters         : "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    best_epoch = 0

    patience_counter = 0

    best_metrics = None

    history = []

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        # ----------------------------------------------------
        # Training mode
        # ----------------------------------------------------

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            x,
            edge_index
        )

        train_loss = criterion(
            logits[
                train_mask
            ],
            y[
                train_mask
            ]
        )

        train_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            validation_logits = model(
                x,
                edge_index
            )

            validation_loss = criterion(
                validation_logits[
                    validation_mask
                ],
                y[
                    validation_mask
                ]
            )

            validation_predictions = (
                validation_logits[
                    validation_mask
                ].argmax(
                    dim=1
                )
            )

            validation_labels = y[
                validation_mask
            ]

            metrics = calculate_metrics(
                validation_labels,
                validation_predictions
            )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            validation_loss.item()
        )

        current_lr = optimizer.param_groups[
            0
        ][
            "lr"
        ]

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        epoch_record = {
            "epoch":
                epoch,

            "train_loss":
                float(
                    train_loss.item()
                ),

            "validation_loss":
                float(
                    validation_loss.item()
                ),

            "accuracy":
                metrics[
                    "accuracy"
                ],

            "precision":
                metrics[
                    "precision"
                ],

            "recall":
                metrics[
                    "recall"
                ],

            "f1":
                metrics[
                    "f1"
                ],

            "specificity":
                metrics[
                    "specificity"
                ],

            "learning_rate":
                float(
                    current_lr
                ),
        }

        history.append(
            epoch_record
        )

        # ----------------------------------------------------
        # Epoch output
        # ----------------------------------------------------

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: "
            f"{train_loss.item():.6f} | "
            f"Val Loss: "
            f"{validation_loss.item():.6f} | "
            f"Acc: "
            f"{metrics['accuracy']:.4f} | "
            f"Prec: "
            f"{metrics['precision']:.4f} | "
            f"Recall: "
            f"{metrics['recall']:.4f} | "
            f"F1: "
            f"{metrics['f1']:.4f} | "
            f"LR: "
            f"{current_lr:.6f}"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if validation_loss.item() < (
            best_validation_loss - 1e-6
        ):

            best_validation_loss = (
                validation_loss.item()
            )

            best_epoch = epoch

            patience_counter = 0

            best_metrics = metrics.copy()

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

                "epoch":
                    epoch,

                "validation_loss":
                    best_validation_loss,

                "metrics":
                    best_metrics,

                "seed":
                    SEED,
            }

            os.makedirs(
                MODEL_DIR,
                exist_ok=True
            )

            torch.save(
                checkpoint,
                MODEL_PATH
            )

        else:

            patience_counter += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if patience_counter >= PATIENCE:

            print()

            print(
                f"Early stopping at epoch "
                f"{epoch}."
            )

            break

    # ========================================================
    # SAVE METRICS
    # ========================================================

    result = {
        "model":
            "EnhancedGraphSAGE",

        "client":
            "client_3",

        "graph":
            "graph_enhanced_aligned.pt",

        "input_dim":
            INPUT_DIM,

        "hidden_dim":
            HIDDEN_DIM,

        "output_dim":
            OUTPUT_DIM,

        "dropout":
            DROPOUT,

        "parameters":
            parameter_count,

        "optimizer":
            "AdamW",

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "loss":
            "CrossEntropyLoss",

        "epochs_requested":
            EPOCHS,

        "best_epoch":
            best_epoch,

        "validation_ratio":
            VALIDATION_RATIO,

        "seed":
            SEED,

        "best_validation_loss":
            best_validation_loss,

        "best_metrics":
            best_metrics,

        "history":
            history,

        "model_path":
            MODEL_PATH,
    }

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "CLIENT 3 ENHANCED GRAPHSAGE TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Best epoch          : "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )

    if best_metrics is not None:

        print(
            f"Accuracy            : "
            f"{best_metrics['accuracy']:.6f}"
        )

        print(
            f"Precision           : "
            f"{best_metrics['precision']:.6f}"
        )

        print(
            f"Recall              : "
            f"{best_metrics['recall']:.6f}"
        )

        print(
            f"F1                  : "
            f"{best_metrics['f1']:.6f}"
        )

        print(
            f"Specificity         : "
            f"{best_metrics['specificity']:.6f}"
        )

        print()

        print(
            "Confusion matrix:"
        )

        print(
            f"  TP: "
            f"{best_metrics['tp']:,}"
        )

        print(
            f"  TN: "
            f"{best_metrics['tn']:,}"
        )

        print(
            f"  FP: "
            f"{best_metrics['fp']:,}"
        )

        print(
            f"  FN: "
            f"{best_metrics['fn']:,}"
        )

    print()

    print(
        "Best model saved to:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        "Metrics saved to:"
    )

    print(
        METRICS_PATH
    )

    print()

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    train()