import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv


# ============================================================
# RFGN GRAPH SAGE MODEL
# ============================================================


class GraphSAGE(nn.Module):
    """
    GraphSAGE model for transaction-level fraud detection.

    Input:
        Node features + graph connectivity

    Output:
        Two-class logits:
            0 -> Legitimate
            1 -> Fraud
    """

    def __init__(
        self,
        input_dim=814,
        hidden_dim=128,
        output_dim=2,
        dropout=0.30,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # ----------------------------------------------------
        # GraphSAGE Layer 1
        # ----------------------------------------------------

        self.conv1 = SAGEConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
        )

        # ----------------------------------------------------
        # GraphSAGE Layer 2
        # ----------------------------------------------------

        self.conv2 = SAGEConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
        )

        # ----------------------------------------------------
        # Classification Layer
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            hidden_dim,
            output_dim,
        )

    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(
        self,
        x,
        edge_index,
    ):
        """
        Forward pass.

        Parameters
        ----------
        x:
            Node feature matrix
            Shape: [num_nodes, input_dim]

        edge_index:
            PyTorch Geometric edge index
            Shape: [2, num_edges]

        Returns
        -------
        logits:
            Shape: [num_nodes, output_dim]
        """

        # GraphSAGE layer 1
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

        # GraphSAGE layer 2
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

        # Fraud classification
        logits = self.classifier(
            x
        )

        return logits


# ============================================================
# MODEL INFORMATION
# ============================================================

def model_summary(
    model,
):
    """
    Return basic model information.
    """

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
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "output_dim": model.output_dim,
        "dropout": model.dropout,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }


# ============================================================
# TEST MODEL DIRECTLY
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("RFGN GRAPHSAGE MODEL")
    print("=" * 70)

    model = GraphSAGE(
        input_dim=814,
        hidden_dim=128,
        output_dim=2,
        dropout=0.30,
    )

    print()
    print("Model created successfully.")

    print()
    print("Model configuration:")

    summary = model_summary(
        model
    )

    for key, value in summary.items():

        print(
            f"{key:<22}: {value}"
        )

    print()
    print("GraphSAGE architecture:")
    print(model)

    print()
    print("=" * 70)