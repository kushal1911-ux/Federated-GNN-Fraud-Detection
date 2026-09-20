import os
import sys

import torch


# ============================================================
# IMPORT PROJECT
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


from models.gnn.graph_builder import (
    build_client_graph
)


# ============================================================
# TEST
# ============================================================

def main():

    print()
    print(
        "=" * 70
    )
    print(
        "RFGN CLIENT 1 GRAPH BUILDER TEST"
    )
    print(
        "=" * 70
    )

    print()

    print(
        "Building graph for Client 1 only."
    )

    print(
        "No model training will be performed."
    )

    print(
        "No validation/test data will be used."
    )

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    data = build_client_graph(
        "client_1"
    )

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    assert data is not None

    assert hasattr(
        data,
        "x"
    )

    assert hasattr(
        data,
        "edge_index"
    )

    assert hasattr(
        data,
        "y"
    )

    print()
    print(
        "Basic graph object: PASS"
    )

    # --------------------------------------------------------
    # Node count
    # --------------------------------------------------------

    assert (
        data.num_nodes
        == 136987
    )

    print(
        "Client 1 node count: PASS"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    assert (
        data.y.shape[0]
        == data.num_nodes
    )

    print(
        "Client 1 label count: PASS"
    )

    # --------------------------------------------------------
    # Feature matrix
    # --------------------------------------------------------

    assert (
        data.x.shape[0]
        == data.num_nodes
    )

    print(
        "Feature row count: PASS"
    )

    # --------------------------------------------------------
    # Edge index
    # --------------------------------------------------------

    assert (
        data.edge_index.shape[0]
        == 2
    )

    assert (
        data.edge_index.shape[1]
        > 0
    )

    print(
        "Graph edges exist: PASS"
    )

    # --------------------------------------------------------
    # Edge bounds
    # --------------------------------------------------------

    assert (
        int(data.edge_index.min())
        >= 0
    )

    assert (
        int(data.edge_index.max())
        < data.num_nodes
    )

    print(
        "Edge bounds: PASS"
    )

    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    assert not torch.isnan(
        data.x
    ).any()

    print(
        "Feature NaN check: PASS"
    )

    # --------------------------------------------------------
    # Infinity
    # --------------------------------------------------------

    assert not torch.isinf(
        data.x
    ).any()

    print(
        "Feature infinity check: PASS"
    )

    # --------------------------------------------------------
    # Fraud count
    # --------------------------------------------------------

    fraud_count = int(
        data.y.sum().item()
    )

    assert (
        fraud_count == 5102
    )

    print(
        "Fraud label preservation: PASS"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "CLIENT 1 GRAPH BUILDER TEST: PASSED"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Nodes:",
        f"{data.num_nodes:,}"
    )

    print(
        "Features:",
        data.num_node_features
    )

    print(
        "Edges:",
        f"{data.num_edges:,}"
    )

    print(
        "Fraud nodes:",
        fraud_count
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()