"""
RFGN - Global GraphSAGE Evaluation Across All 3 Federated Clients

Purpose:
    Evaluate the existing Flower FedAvg Global GraphSAGE model
    on Client 1, Client 2, and Client 3.

Important:
    - No training
    - No parameter updates
    - No graph modification
    - No CSV modification
    - Uses existing Flower global model
    - Uses existing aligned client graphs
    - Uses DQN-selected threshold = 0.80

Metrics:
    Accuracy
    Precision
    Recall
    F1 Score
    Specificity
    TP
    TN
    FP
    FN
    Fraud rate
    Predicted fraud rate
    Raw reward
    Normalized reward

Also calculates:
    Macro average
    Pooled / micro performance
"""

import os
import sys
import json

import numpy as np
import torch


# ============================================================
# PROJECT ROOT
# ============================================================

# This file is:
#
# E:\RFGN\tests\evaluate_global_all_clients.py
#
# Therefore the project root is:
#
# E:\RFGN

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


# ============================================================
# PYTHON IMPORT PATH
# ============================================================

# Add project root so imports such as:
#
# from models.gnn.graphsage_model import GraphSAGE
#
# work when this script is launched from the tests folder.

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from models.gnn.graphsage_model import GraphSAGE


# ============================================================
# MODEL PATH
# ============================================================

GLOBAL_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "global",
    "global_graphsage_flower.pt"
)


# ============================================================
# CLIENT GRAPH PATHS
# ============================================================

CLIENT_GRAPH_PATHS = {

    1: os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "graphs",
        "client_1",
        "graph_aligned.pt"
    ),

    2: os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "graphs",
        "client_2",
        "graph_aligned.pt"
    ),

    3: os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "graphs",
        "client_3",
        "graph_aligned.pt"
    )
}


# ============================================================
# OUTPUT PATH
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "global"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "global_all_clients_evaluation.json"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

INPUT_DIM = 769
HIDDEN_DIM = 128
OUTPUT_DIM = 2
DROPOUT = 0.3


# ============================================================
# DQN THRESHOLD
# ============================================================

DQN_THRESHOLD = 0.80


# ============================================================
# REWARD CONFIGURATION
# ============================================================

TRUE_POSITIVE_REWARD = 5.0
TRUE_NEGATIVE_REWARD = 1.0

FALSE_POSITIVE_COST = 1.0
FALSE_NEGATIVE_COST = 5.0

REVIEW_COST = 0.10
MAX_REVIEW_RATE = 0.01


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HELPER
# ============================================================

def separator():

    print("-" * 70)


# ============================================================
# LOAD GLOBAL GRAPHSAGE MODEL
# ============================================================

def load_global_model():

    print(
        "Loading Flower global GraphSAGE model..."
    )

    print()
    print(
        "Global model path:"
    )

    print(
        GLOBAL_MODEL_PATH
    )

    # --------------------------------------------------------
    # Check model exists
    # --------------------------------------------------------

    if not os.path.isfile(
        GLOBAL_MODEL_PATH
    ):

        raise FileNotFoundError(
            "\nGlobal model not found:\n"
            +
            GLOBAL_MODEL_PATH
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        GLOBAL_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = GraphSAGE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT
    )

    # --------------------------------------------------------
    # Extract state dictionary
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        raise RuntimeError(
            "Unsupported global model checkpoint format."
        )

    # --------------------------------------------------------
    # Load parameters
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict
    )

    model.to(
        DEVICE
    )

    model.eval()

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameter count: "
        f"{parameter_count:,}"
    )

    if parameter_count != 230146:

        raise RuntimeError(
            "Unexpected GraphSAGE parameter count: "
            +
            str(parameter_count)
        )

    # --------------------------------------------------------
    # NaN / Inf validation
    # --------------------------------------------------------

    for name, parameter in model.named_parameters():

        if not torch.isfinite(
            parameter
        ).all():

            raise RuntimeError(
                "NaN/Inf detected in model parameter: "
                +
                name
            )

    print(
        "Global model checkpoint: PASS"
    )

    print(
        "Global GraphSAGE model: PASS"
    )

    return model


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities,
    threshold
):

    labels = np.asarray(
        labels
    ).astype(
        np.int64
    )

    probabilities = np.asarray(
        probabilities
    ).astype(
        np.float64
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = (
        probabilities >= threshold
    ).astype(
        np.int64
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tp = int(
        np.sum(
            (predictions == 1)
            &
            (labels == 1)
        )
    )

    tn = int(
        np.sum(
            (predictions == 0)
            &
            (labels == 0)
        )
    )

    fp = int(
        np.sum(
            (predictions == 1)
            &
            (labels == 0)
        )
    )

    fn = int(
        np.sum(
            (predictions == 0)
            &
            (labels == 1)
        )
    )

    total = len(
        labels
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = (

        (tp + tn) / total

        if total > 0

        else 0.0
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (

        tp / (tp + fp)

        if (tp + fp) > 0

        else 0.0
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = (

        tp / (tp + fn)

        if (tp + fn) > 0

        else 0.0
    )

    # --------------------------------------------------------
    # F1 Score
    # --------------------------------------------------------

    f1_score = (

        2.0
        * precision
        * recall
        / (precision + recall)

        if (precision + recall) > 0

        else 0.0
    )

    # --------------------------------------------------------
    # Specificity
    # --------------------------------------------------------

    specificity = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else 0.0
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    fraud_samples = int(
        np.sum(
            labels == 1
        )
    )

    legitimate_samples = int(
        np.sum(
            labels == 0
        )
    )

    predicted_fraud_samples = int(
        np.sum(
            predictions == 1
        )
    )

    predicted_legitimate_samples = int(
        np.sum(
            predictions == 0
        )
    )

    # --------------------------------------------------------
    # Rates
    # --------------------------------------------------------

    fraud_rate = (

        fraud_samples / total

        if total > 0

        else 0.0
    )

    predicted_fraud_rate = (

        predicted_fraud_samples / total

        if total > 0

        else 0.0
    )

    # --------------------------------------------------------
    # Raw reward
    # --------------------------------------------------------

    raw_reward = (

        tp * TRUE_POSITIVE_REWARD

        +

        tn * TRUE_NEGATIVE_REWARD

        -

        fp * FALSE_POSITIVE_COST

        -

        fn * FALSE_NEGATIVE_COST
    )

    # --------------------------------------------------------
    # Review budget penalty
    # --------------------------------------------------------

    review_penalty = 0.0

    if predicted_fraud_rate > MAX_REVIEW_RATE:

        excess_rate = (

            predicted_fraud_rate
            -
            MAX_REVIEW_RATE
        )

        review_penalty = (

            excess_rate
            *
            total
            *
            REVIEW_COST
        )

        raw_reward -= (
            review_penalty
        )

    # --------------------------------------------------------
    # Normalized reward
    # --------------------------------------------------------

    normalized_reward = (

        raw_reward / total

        if total > 0

        else 0.0
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "samples":
            total,

        "fraud_samples":
            fraud_samples,

        "legitimate_samples":
            legitimate_samples,

        "threshold":
            float(threshold),

        "true_positive":
            tp,

        "true_negative":
            tn,

        "false_positive":
            fp,

        "false_negative":
            fn,

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1_score":
            float(f1_score),

        "specificity":
            float(specificity),

        "predicted_fraud_samples":
            predicted_fraud_samples,

        "predicted_legitimate_samples":
            predicted_legitimate_samples,

        "fraud_rate":
            float(fraud_rate),

        "predicted_fraud_rate":
            float(predicted_fraud_rate),

        "review_penalty":
            float(review_penalty),

        "raw_reward":
            float(raw_reward),

        "normalized_reward":
            float(normalized_reward)
    }


# ============================================================
# EVALUATE ONE CLIENT
# ============================================================

def evaluate_client(
    model,
    client_id
):

    print()
    print("=" * 70)

    print(
        f"CLIENT {client_id} - "
        f"GLOBAL MODEL EVALUATION"
    )

    print("=" * 70)

    graph_path = (
        CLIENT_GRAPH_PATHS[
            client_id
        ]
    )

    print()
    print(
        "Graph path:"
    )

    print(
        graph_path
    )

    # --------------------------------------------------------
    # Check graph
    # --------------------------------------------------------

    if not os.path.isfile(
        graph_path
    ):

        raise FileNotFoundError(

            "\nClient graph not found:\n"
            +
            graph_path
        )

    # --------------------------------------------------------
    # Load PyG graph
    #
    # The graph is a trusted RFGN-generated file.
    #
    # PyTorch 2.6+ defaults weights_only=True.
    #
    # PyG Data objects require weights_only=False here.
    # --------------------------------------------------------

    graph = torch.load(
        graph_path,
        map_location=DEVICE,
        weights_only=False
    )

    graph = graph.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Graph validation
    # --------------------------------------------------------

    if graph.x.shape[1] != INPUT_DIM:

        raise RuntimeError(

            f"Client {client_id}: "
            f"expected {INPUT_DIM} features, "
            f"found {graph.x.shape[1]}"
        )

    if not torch.isfinite(
        graph.x
    ).all():

        raise RuntimeError(

            f"Client {client_id}: "
            "NaN/Inf detected in graph features."
        )

    labels = graph.y.long()

    print()
    print(
        "Graph validation: PASS"
    )

    print(
        f"Nodes       : "
        f"{graph.x.shape[0]:,}"
    )

    print(
        f"Features    : "
        f"{graph.x.shape[1]:,}"
    )

    print(
        f"Edges       : "
        f"{graph.edge_index.shape[1]:,}"
    )

    print(
        f"Fraud       : "
        f"{int((labels == 1).sum().item()):,}"
    )

    print(
        f"Legitimate  : "
        f"{int((labels == 0).sum().item()):,}"
    )

    # --------------------------------------------------------
    # Global GraphSAGE inference
    # --------------------------------------------------------

    print()
    print(
        "Running Global GraphSAGE inference..."
    )

    with torch.no_grad():

        logits = model(
            graph.x,
            graph.edge_index
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[:, 1]

    probabilities_np = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    labels_np = (
        labels
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Probability validation
    # --------------------------------------------------------

    if not np.isfinite(
        probabilities_np
    ).all():

        raise RuntimeError(

            f"Client {client_id}: "
            "NaN/Inf detected in probabilities."
        )

    print(
        "Global GraphSAGE inference: PASS"
    )

    # --------------------------------------------------------
    # Probability statistics
    # --------------------------------------------------------

    probability_min = float(
        np.min(
            probabilities_np
        )
    )

    probability_max = float(
        np.max(
            probabilities_np
        )
    )

    probability_mean = float(
        np.mean(
            probabilities_np
        )
    )

    probability_std = float(
        np.std(
            probabilities_np
        )
    )

    # --------------------------------------------------------
    # Apply DQN threshold
    # --------------------------------------------------------

    metrics = calculate_metrics(
        labels_np,
        probabilities_np,
        DQN_THRESHOLD
    )

    # Add probability statistics

    metrics[
        "probability_min"
    ] = probability_min

    metrics[
        "probability_max"
    ] = probability_max

    metrics[
        "probability_mean"
    ] = probability_mean

    metrics[
        "probability_std"
    ] = probability_std

    # --------------------------------------------------------
    # Print threshold
    # --------------------------------------------------------

    print()
    print(
        "DQN DECISION THRESHOLD"
    )

    separator()

    print(
        f"Threshold       : "
        f"{DQN_THRESHOLD:.2f}"
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    print()
    print(
        "CLASSIFICATION RESULTS"
    )

    separator()

    print(
        f"Accuracy        : "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Precision       : "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall          : "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1 Score        : "
        f"{metrics['f1_score']:.6f}"
    )

    print(
        f"Specificity     : "
        f"{metrics['specificity']:.6f}"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print()
    print(
        "CONFUSION MATRIX"
    )

    separator()

    print(
        f"True Positive   : "
        f"{metrics['true_positive']:,}"
    )

    print(
        f"True Negative   : "
        f"{metrics['true_negative']:,}"
    )

    print(
        f"False Positive  : "
        f"{metrics['false_positive']:,}"
    )

    print(
        f"False Negative  : "
        f"{metrics['false_negative']:,}"
    )

    # --------------------------------------------------------
    # Decision rates
    # --------------------------------------------------------

    print()
    print(
        "DECISION RATES"
    )

    separator()

    print(
        f"Actual fraud rate    : "
        f"{metrics['fraud_rate'] * 100:.4f}%"
    )

    print(
        f"Predicted fraud      : "
        f"{metrics['predicted_fraud_samples']:,}"
    )

    print(
        f"Predicted fraud rate : "
        f"{metrics['predicted_fraud_rate'] * 100:.4f}%"
    )

    # --------------------------------------------------------
    # Probability statistics
    # --------------------------------------------------------

    print()
    print(
        "PROBABILITY STATISTICS"
    )

    separator()

    print(
        f"Minimum probability : "
        f"{probability_min:.6f}"
    )

    print(
        f"Maximum probability : "
        f"{probability_max:.6f}"
    )

    print(
        f"Mean probability    : "
        f"{probability_mean:.6f}"
    )

    print(
        f"Std probability     : "
        f"{probability_std:.6f}"
    )

    # --------------------------------------------------------
    # Reward
    # --------------------------------------------------------

    print()
    print(
        "REWARD"
    )

    separator()

    print(
        f"Raw reward          : "
        f"{metrics['raw_reward']:.4f}"
    )

    print(
        f"Review penalty      : "
        f"{metrics['review_penalty']:.4f}"
    )

    print(
        f"Normalized reward   : "
        f"{metrics['normalized_reward']:.6f}"
    )

    print()
    print(
        f"Client {client_id} evaluation: PASS"
    )

    return metrics


# ============================================================
# MACRO AVERAGE
# ============================================================

def calculate_macro_average(
    results
):

    metric_names = [

        "accuracy",

        "precision",

        "recall",

        "f1_score",

        "specificity",

        "fraud_rate",

        "predicted_fraud_rate",

        "normalized_reward"
    ]

    macro = {}

    for metric in metric_names:

        macro[metric] = float(

            np.mean(
                [
                    results[1][metric],
                    results[2][metric],
                    results[3][metric]
                ]
            )
        )

    return macro


# ============================================================
# POOLED / MICRO RESULTS
# ============================================================

def calculate_pooled_results(
    results
):

    # --------------------------------------------------------
    # Sum confusion matrices
    # --------------------------------------------------------

    tp = sum(
        results[i]["true_positive"]
        for i in [1, 2, 3]
    )

    tn = sum(
        results[i]["true_negative"]
        for i in [1, 2, 3]
    )

    fp = sum(
        results[i]["false_positive"]
        for i in [1, 2, 3]
    )

    fn = sum(
        results[i]["false_negative"]
        for i in [1, 2, 3]
    )

    # --------------------------------------------------------
    # Sample counts
    # --------------------------------------------------------

    total = sum(
        results[i]["samples"]
        for i in [1, 2, 3]
    )

    fraud_samples = sum(
        results[i]["fraud_samples"]
        for i in [1, 2, 3]
    )

    legitimate_samples = sum(
        results[i]["legitimate_samples"]
        for i in [1, 2, 3]
    )

    predicted_fraud_samples = sum(
        results[i]["predicted_fraud_samples"]
        for i in [1, 2, 3]
    )

    predicted_legitimate_samples = sum(
        results[i]["predicted_legitimate_samples"]
        for i in [1, 2, 3]
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = (

        (tp + tn) / total

        if total > 0

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

    f1_score = (

        2.0
        * precision
        * recall
        / (precision + recall)

        if (precision + recall) > 0

        else 0.0
    )

    specificity = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else 0.0
    )

    fraud_rate = (

        fraud_samples / total

        if total > 0

        else 0.0
    )

    predicted_fraud_rate = (

        predicted_fraud_samples / total

        if total > 0

        else 0.0
    )

    # --------------------------------------------------------
    # Reward
    # --------------------------------------------------------

    raw_reward = (

        tp * TRUE_POSITIVE_REWARD

        +

        tn * TRUE_NEGATIVE_REWARD

        -

        fp * FALSE_POSITIVE_COST

        -

        fn * FALSE_NEGATIVE_COST
    )

    # --------------------------------------------------------
    # Review penalty
    # --------------------------------------------------------

    review_penalty = 0.0

    if predicted_fraud_rate > MAX_REVIEW_RATE:

        excess_rate = (

            predicted_fraud_rate
            -
            MAX_REVIEW_RATE
        )

        review_penalty = (

            excess_rate
            *
            total
            *
            REVIEW_COST
        )

        raw_reward -= (
            review_penalty
        )

    # --------------------------------------------------------
    # Normalized reward
    # --------------------------------------------------------

    normalized_reward = (

        raw_reward / total

        if total > 0

        else 0.0
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "samples":
            total,

        "fraud_samples":
            fraud_samples,

        "legitimate_samples":
            legitimate_samples,

        "threshold":
            DQN_THRESHOLD,

        "true_positive":
            tp,

        "true_negative":
            tn,

        "false_positive":
            fp,

        "false_negative":
            fn,

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1_score":
            float(f1_score),

        "specificity":
            float(specificity),

        "predicted_fraud_samples":
            predicted_fraud_samples,

        "predicted_legitimate_samples":
            predicted_legitimate_samples,

        "fraud_rate":
            float(fraud_rate),

        "predicted_fraud_rate":
            float(predicted_fraud_rate),

        "review_penalty":
            float(review_penalty),

        "raw_reward":
            float(raw_reward),

        "normalized_reward":
            float(normalized_reward)
    }


# ============================================================
# FINAL RESULTS TABLE
# ============================================================

def print_final_results(
    results,
    macro,
    pooled
):

    print()
    print()
    print("=" * 70)

    print(
        "RFGN - FINAL ALL 3 CLIENT RESULTS"
    )

    print("=" * 70)

    print()

    print(
        f"{'Metric':<24}"
        f"{'Client 1':>14}"
        f"{'Client 2':>14}"
        f"{'Client 3':>14}"
    )

    separator()

    rows = [

        (
            "Accuracy",
            "accuracy"
        ),

        (
            "Precision",
            "precision"
        ),

        (
            "Recall",
            "recall"
        ),

        (
            "F1 Score",
            "f1_score"
        ),

        (
            "Specificity",
            "specificity"
        ),

        (
            "Fraud Rate",
            "fraud_rate"
        ),

        (
            "Predicted Fraud Rate",
            "predicted_fraud_rate"
        ),

        (
            "Normalized Reward",
            "normalized_reward"
        )
    ]

    for label, key in rows:

        print(

            f"{label:<24}"
            f"{results[1][key]:>14.6f}"
            f"{results[2][key]:>14.6f}"
            f"{results[3][key]:>14.6f}"
        )

    # --------------------------------------------------------
    # Macro average
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "MACRO AVERAGE"
    )

    print("=" * 70)

    for label, key in rows:

        print(

            f"{label:<24}: "
            f"{macro[key]:.6f}"
        )

    # --------------------------------------------------------
    # Pooled
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "POOLED / MICRO RESULTS"
    )

    print("=" * 70)

    print(
        f"Total samples        : "
        f"{pooled['samples']:,}"
    )

    print(
        f"Fraud samples        : "
        f"{pooled['fraud_samples']:,}"
    )

    print(
        f"Legitimate samples   : "
        f"{pooled['legitimate_samples']:,}"
    )

    print()

    print(
        f"Accuracy             : "
        f"{pooled['accuracy']:.6f}"
    )

    print(
        f"Precision            : "
        f"{pooled['precision']:.6f}"
    )

    print(
        f"Recall               : "
        f"{pooled['recall']:.6f}"
    )

    print(
        f"F1 Score             : "
        f"{pooled['f1_score']:.6f}"
    )

    print(
        f"Specificity          : "
        f"{pooled['specificity']:.6f}"
    )

    print()

    print(
        f"True Positive        : "
        f"{pooled['true_positive']:,}"
    )

    print(
        f"True Negative        : "
        f"{pooled['true_negative']:,}"
    )

    print(
        f"False Positive       : "
        f"{pooled['false_positive']:,}"
    )

    print(
        f"False Negative       : "
        f"{pooled['false_negative']:,}"
    )

    print()

    print(
        f"Predicted fraud      : "
        f"{pooled['predicted_fraud_samples']:,}"
    )

    print(
        f"Predicted fraud rate : "
        f"{pooled['predicted_fraud_rate'] * 100:.4f}%"
    )

    print()

    print(
        f"Raw reward           : "
        f"{pooled['raw_reward']:.4f}"
    )

    print(
        f"Review penalty       : "
        f"{pooled['review_penalty']:.4f}"
    )

    print(
        f"Normalized reward    : "
        f"{pooled['normalized_reward']:.6f}"
    )

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "MODEL CONFIGURATION"
    )

    print("=" * 70)

    print(
        "Global model        : "
        "Flower FedAvg GraphSAGE"
    )

    print(
        f"Input features      : "
        f"{INPUT_DIM}"
    )

    print(
        f"Hidden dimension    : "
        f"{HIDDEN_DIM}"
    )

    print(
        f"Output classes      : "
        f"{OUTPUT_DIM}"
    )

    print(
        f"Federated clients   : "
        f"3"
    )

    print(
        f"Federated rounds    : "
        f"3"
    )

    print(
        f"DQN threshold       : "
        f"{DQN_THRESHOLD:.2f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "RFGN GLOBAL GRAPHSAGE - "
        "ALL 3 CLIENT EVALUATION"
    )

    print("=" * 70)

    print()
    print(
        "No training will be performed."
    )

    print(
        "No model parameters will be updated."
    )

    print(
        "No graph will be modified."
    )

    print(
        "No CSV dataset will be modified."
    )

    print()

    print(
        f"Project root       : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"DQN threshold      : "
        f"{DQN_THRESHOLD:.2f}"
    )

    print(
        f"Device             : "
        f"{DEVICE}"
    )

    # --------------------------------------------------------
    # Verify project root
    # --------------------------------------------------------

    if not os.path.isdir(
        PROJECT_ROOT
    ):

        raise RuntimeError(

            "Project root does not exist:\n"
            +
            PROJECT_ROOT
        )

    # --------------------------------------------------------
    # Verify global model
    # --------------------------------------------------------

    if not os.path.isfile(
        GLOBAL_MODEL_PATH
    ):

        raise FileNotFoundError(

            "\nExpected global model at:\n"
            +
            GLOBAL_MODEL_PATH
        )

    # --------------------------------------------------------
    # Load global model
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "LOADING GLOBAL FLOWER MODEL"
    )

    print("=" * 70)

    model = load_global_model()

    # --------------------------------------------------------
    # Evaluate all three clients
    # --------------------------------------------------------

    results = {}

    for client_id in [1, 2, 3]:

        results[client_id] = evaluate_client(
            model,
            client_id
        )

    # --------------------------------------------------------
    # Macro average
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "CALCULATING MACRO AVERAGE"
    )

    print("=" * 70)

    macro = calculate_macro_average(
        results
    )

    print(
        "Macro average: PASS"
    )

    # --------------------------------------------------------
    # Pooled / micro
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "CALCULATING POOLED / MICRO RESULTS"
    )

    print("=" * 70)

    pooled = calculate_pooled_results(
        results
    )

    print(
        "Pooled evaluation: PASS"
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print_final_results(
        results,
        macro,
        pooled
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output = {

        "experiment":
            "RFGN Global GraphSAGE "
            "Evaluation Across All 3 Clients",

        "project_root":
            PROJECT_ROOT,

        "global_model":
            GLOBAL_MODEL_PATH,

        "threshold_source":
            "DQN",

        "dqn_threshold":
            DQN_THRESHOLD,

        "device":
            str(DEVICE),

        "model_configuration": {

            "input_dim":
                INPUT_DIM,

            "hidden_dim":
                HIDDEN_DIM,

            "output_dim":
                OUTPUT_DIM,

            "dropout":
                DROPOUT,

            "federated_clients":
                3,

            "federated_rounds":
                3
        },

        "reward_configuration": {

            "true_positive_reward":
                TRUE_POSITIVE_REWARD,

            "true_negative_reward":
                TRUE_NEGATIVE_REWARD,

            "false_positive_cost":
                FALSE_POSITIVE_COST,

            "false_negative_cost":
                FALSE_NEGATIVE_COST,

            "review_cost":
                REVIEW_COST,

            "max_review_rate":
                MAX_REVIEW_RATE
        },

        "clients": {

            "client_1":
                results[1],

            "client_2":
                results[2],

            "client_3":
                results[3]
        },

        "macro_average":
            macro,

        "pooled_micro_results":
            pooled
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "RESULTS SAVED"
    )

    print("=" * 70)

    print(
        OUTPUT_PATH
    )

    print()
    print("=" * 70)

    print(
        "RFGN ALL-CLIENT EVALUATION: "
        "COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()