# ============================================================
# RFGN ENHANCED GRAPHSAGE MODEL
# ============================================================
#
# Architecture:
#
#   Input 769
#       ↓
#   SAGEConv
#       ↓
#   BatchNorm
#       ↓
#   ReLU
#       ↓
#   Dropout
#       ↓
#   SAGEConv
#       ↓
#   BatchNorm
#       ↓
#   ReLU
#       ↓
#   Dropout
#       ↓
#   SAGEConv
#       ↓
#   Residual Connection
#       ↓
#   BatchNorm
#       ↓
#   ReLU
#       ↓
#   Dropout
#       ↓
#   Classifier
#       ↓
#   2 Classes
#
# IMPORTANT:
#   - Does NOT modify the original GraphSAGE model.
#   - Compatible with graph_enhanced_aligned.pt.
#   - Uses x and edge_index.
#   - edge_type remains stored in the graph but is not
#     directly consumed by vanilla SAGEConv.
#
# ============================================================

import os
import sys

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


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
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# MODEL
# ============================================================

class EnhancedGraphSAGE(nn.Module):

    def __init__(
        self,
        input_dim=769,
        hidden_dim=128,
        output_dim=2,
        dropout=0.30,
    ):

        super().__init__()

        self.input_dim = input_dim

        self.hidden_dim = hidden_dim

        self.output_dim = output_dim

        self.dropout_rate = dropout

        # ----------------------------------------------------
        # First GraphSAGE layer
        # ----------------------------------------------------

        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )

        self.bn1 = nn.BatchNorm1d(
            hidden_dim
        )

        # ----------------------------------------------------
        # Second GraphSAGE layer
        # ----------------------------------------------------

        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )

        self.bn2 = nn.BatchNorm1d(
            hidden_dim
        )

        # ----------------------------------------------------
        # Third GraphSAGE layer
        # ----------------------------------------------------

        self.conv3 = SAGEConv(
            hidden_dim,
            hidden_dim
        )

        self.bn3 = nn.BatchNorm1d(
            hidden_dim
        )

        # ----------------------------------------------------
        # Activation
        # ----------------------------------------------------

        self.relu = nn.ReLU()

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            p=dropout
        )

        # ----------------------------------------------------
        # Final classifier
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            hidden_dim,
            output_dim
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        edge_index
    ):

        # ----------------------------------------------------
        # Input validation
        # ----------------------------------------------------

        if x.ndim != 2:

            raise ValueError(
                "Input x must have shape [N, F]."
            )

        if x.shape[1] != self.input_dim:

            raise ValueError(
                f"Expected {self.input_dim} input features, "
                f"received {x.shape[1]}."
            )

        if edge_index.ndim != 2:

            raise ValueError(
                "edge_index must have shape [2, E]."
            )

        if edge_index.shape[0] != 2:

            raise ValueError(
                "edge_index first dimension must be 2."
            )

        # ----------------------------------------------------
        # Layer 1
        # ----------------------------------------------------

        x = self.conv1(
            x,
            edge_index
        )

        x = self.bn1(
            x
        )

        x = self.relu(
            x
        )

        x = self.dropout(
            x
        )

        # ----------------------------------------------------
        # Layer 2
        # ----------------------------------------------------

        x = self.conv2(
            x,
            edge_index
        )

        x = self.bn2(
            x
        )

        x = self.relu(
            x
        )

        x = self.dropout(
            x
        )

        # ----------------------------------------------------
        # Save residual
        # ----------------------------------------------------

        residual = x

        # ----------------------------------------------------
        # Layer 3
        # ----------------------------------------------------

        x = self.conv3(
            x,
            edge_index
        )

        # ----------------------------------------------------
        # Residual connection
        # ----------------------------------------------------

        x = x + residual

        # ----------------------------------------------------
        # Batch normalization
        # ----------------------------------------------------

        x = self.bn3(
            x
        )

        # ----------------------------------------------------
        # Activation
        # ----------------------------------------------------

        x = self.relu(
            x
        )

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        x = self.dropout(
            x
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        output = self.classifier(
            x
        )

        return output


# ============================================================
# MODEL INFORMATION
# ============================================================

def model_summary(
    model
):

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "input_features":
            model.input_dim,

        "hidden_features":
            model.hidden_dim,

        "output_classes":
            model.output_dim,

        "dropout":
            model.dropout_rate,

        "total_parameters":
            total_parameters,

        "trainable_parameters":
            trainable_parameters,
    }


# ============================================================
# SELF TEST
# ============================================================

def run_self_test():

    print()

    print(
        "=" * 70
    )

    print(
        "RFGN ENHANCED GRAPHSAGE SELF TEST"
    )

    print(
        "=" * 70
    )

    torch.manual_seed(
        42
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = EnhancedGraphSAGE(
        input_dim=769,
        hidden_dim=128,
        output_dim=2,
        dropout=0.30,
    )

    model.eval()

    print()

    print(
        "Model creation: PASS"
    )

    # --------------------------------------------------------
    # Model summary
    # --------------------------------------------------------

    summary = model_summary(
        model
    )

    print()

    print(
        "Model configuration:"
    )

    print(
        f"  Input features      : "
        f"{summary['input_features']}"
    )

    print(
        f"  Hidden features     : "
        f"{summary['hidden_features']}"
    )

    print(
        f"  Output classes      : "
        f"{summary['output_classes']}"
    )

    print(
        f"  Dropout             : "
        f"{summary['dropout']}"
    )

    print(
        f"  Total parameters    : "
        f"{summary['total_parameters']:,}"
    )

    print(
        f"  Trainable parameters: "
        f"{summary['trainable_parameters']:,}"
    )

    # --------------------------------------------------------
    # Synthetic graph
    # --------------------------------------------------------

    num_nodes = 128

    num_edges = 512

    x = torch.randn(
        num_nodes,
        769
    )

    edge_index = torch.randint(
        0,
        num_nodes,
        (
            2,
            num_edges
        )
    )

    print()

    print(
        "Synthetic graph:"
    )

    print(
        f"  Nodes    : {num_nodes}"
    )

    print(
        f"  Features : {x.shape[1]}"
    )

    print(
        f"  Edges    : {num_edges}"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            x,
            edge_index
        )

    expected_shape = (
        num_nodes,
        2
    )

    if tuple(
        output.shape
    ) != expected_shape:

        raise RuntimeError(
            f"Unexpected output shape: "
            f"{tuple(output.shape)}"
        )

    print()

    print(
        f"Forward output shape: "
        f"{tuple(output.shape)}"
    )

    print(
        "Forward pass: PASS"
    )

    # --------------------------------------------------------
    # NaN / infinity
    # --------------------------------------------------------

    if torch.isnan(
        output
    ).any():

        raise RuntimeError(
            "NaN detected in model output."
        )

    print(
        "Output NaN check: PASS"
    )

    if torch.isinf(
        output
    ).any():

        raise RuntimeError(
            "Infinity detected in model output."
        )

    print(
        "Output infinity check: PASS"
    )

    # --------------------------------------------------------
    # Softmax
    # --------------------------------------------------------

    probabilities = torch.softmax(
        output,
        dim=1
    )

    probability_sums = (
        probabilities.sum(
            dim=1
        )
    )

    if not torch.allclose(
        probability_sums,
        torch.ones_like(
            probability_sums
        ),
        atol=1e-5
    ):

        raise RuntimeError(
            "Softmax probabilities do not sum to 1."
        )

    print(
        "Softmax probability check: PASS"
    )

    # --------------------------------------------------------
    # Training mode
    # --------------------------------------------------------

    model.train()

    training_output = model(
        x,
        edge_index
    )

    if training_output.shape != (
        num_nodes,
        2
    ):

        raise RuntimeError(
            "Training-mode forward pass failed."
        )

    print(
        "Training-mode forward pass: PASS"
    )

    # --------------------------------------------------------
    # Gradient propagation
    # --------------------------------------------------------

    labels = torch.randint(
        0,
        2,
        (
            num_nodes,
        )
    )

    criterion = nn.CrossEntropyLoss()

    loss = criterion(
        training_output,
        labels
    )

    if not torch.isfinite(
        loss
    ).item():

        raise RuntimeError(
            "Training loss is not finite."
        )

    model.zero_grad(
        set_to_none=True
    )

    loss.backward()

    gradients_valid = True

    for parameter in model.parameters():

        if parameter.grad is None:

            continue

        if not torch.isfinite(
            parameter.grad
        ).all().item():

            gradients_valid = False

            break

    if not gradients_valid:

        raise RuntimeError(
            "Invalid gradient detected."
        )

    print(
        "Gradient propagation: PASS"
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    if summary[
        "trainable_parameters"
    ] <= 0:

        raise RuntimeError(
            "No trainable parameters found."
        )

    print(
        "Trainable parameter check: PASS"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "RFGN ENHANCED GRAPHSAGE: PASS"
    )

    print(
        "=" * 70
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_self_test()