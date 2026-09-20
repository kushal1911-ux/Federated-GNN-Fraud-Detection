import os
import sys
import torch


# ============================================================
# RFGN CLIENT 3 GRAPH BUILDER TEST
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


from models.gnn.graph_builder import build_client_graph


# ============================================================
# EXPECTED CLIENT 3 VALUES
# ============================================================

EXPECTED_NODES = 137272

EXPECTED_FEATURES = 814

EXPECTED_FRAUD = 4223


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RFGN CLIENT 3 GRAPH BUILDER TEST")
    print("=" * 70)

    print()

    print(
        "Building graph for Client 3 only."
    )

    print(
        "No model training will be performed."
    )

    print(
        "No validation/test data will be used."
    )

    print()

    # ========================================================
    # BUILD CLIENT 3 GRAPH
    # ========================================================

    data = build_client_graph(
        "client_3"
    )

    # ========================================================
    # BASIC GRAPH OBJECT
    # ========================================================

    print()
    print("=" * 70)
    print("CLIENT 3 GRAPH VALIDATION")
    print("=" * 70)

    assert data is not None

    print(
        "Basic graph object: PASS"
    )

    # ========================================================
    # NODE COUNT
    # ========================================================

    assert (
        data.num_nodes
        == EXPECTED_NODES
    )

    print(
        "Client 3 node count: PASS"
    )

    # ========================================================
    # FEATURE DIMENSION
    # ========================================================

    assert (
        data.num_node_features
        == EXPECTED_FEATURES
    )

    print(
        "Feature dimension: PASS"
    )

    # ========================================================
    # LABEL COUNT
    # ========================================================

    assert (
        data.y.shape[0]
        == EXPECTED_NODES
    )

    print(
        "Client 3 label count: PASS"
    )

    # ========================================================
    # FEATURE ROW COUNT
    # ========================================================

    assert (
        data.x.shape[0]
        == EXPECTED_NODES
    )

    print(
        "Feature row count: PASS"
    )

    # ========================================================
    # EDGE EXISTENCE
    # ========================================================

    assert (
        data.num_edges > 0
    )

    print(
        "Graph edges exist: PASS"
    )

    # ========================================================
    # EDGE INDEX SHAPE
    # ========================================================

    assert (
        data.edge_index.shape[0]
        == 2
    )

    print(
        "Edge index shape: PASS"
    )

    # ========================================================
    # EDGE BOUNDS
    # ========================================================

    assert (
        int(data.edge_index.min())
        >= 0
    )

    assert (
        int(data.edge_index.max())
        < EXPECTED_NODES
    )

    print(
        "Edge bounds: PASS"
    )

    # ========================================================
    # FEATURE NaN CHECK
    # ========================================================

    assert not torch.isnan(
        data.x
    ).any()

    print(
        "Feature NaN check: PASS"
    )

    # ========================================================
    # FEATURE INFINITY CHECK
    # ========================================================

    assert not torch.isinf(
        data.x
    ).any()

    print(
        "Feature infinity check: PASS"
    )

    # ========================================================
    # FRAUD LABEL CHECK
    # ========================================================

    fraud_count = int(
        data.y.sum().item()
    )

    assert (
        fraud_count
        == EXPECTED_FRAUD
    )

    print(
        "Fraud label preservation: PASS"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("CLIENT 3 GRAPH BUILDER TEST: PASSED")
    print("=" * 70)

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

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()