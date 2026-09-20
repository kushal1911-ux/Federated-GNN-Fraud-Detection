# ============================================================
# RFGN FLOWER CLIENT LAUNCHER TEST
# ============================================================

import os
import sys


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
# IMPORT FLOWER CLIENT
# ============================================================

from models.federated.flower_client import (
    FraudGraphSAGEClient,
)


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print()

    print("=" * 70)
    print("RFGN FLOWER CLIENT LAUNCHER TEST")
    print("=" * 70)

    print()

    # ========================================================
    # CHECK ALL THREE CLIENTS
    # ========================================================

    for client_id in (1, 2, 3):

        print(
            f"Checking Client {client_id}..."
        )

        # ----------------------------------------------------
        # Create client
        # ----------------------------------------------------

        client = FraudGraphSAGEClient(
            client_id
        )

        # ----------------------------------------------------
        # Get parameters
        # ----------------------------------------------------

        parameters = client.get_parameters(
            {}
        )

        # ----------------------------------------------------
        # Validate parameter count
        # ----------------------------------------------------

        if len(parameters) != 8:

            raise RuntimeError(
                f"Client {client_id}: "
                f"expected 8 parameter tensors, "
                f"found {len(parameters)}."
            )

        # ----------------------------------------------------
        # Validate feature dimension
        # ----------------------------------------------------

        if (
            client.graph.num_node_features
            != 769
        ):

            raise RuntimeError(
                f"Client {client_id}: "
                f"expected 769 features, "
                f"found "
                f"{client.graph.num_node_features}."
            )

        # ----------------------------------------------------
        # Validate graph
        # ----------------------------------------------------

        if (
            client.graph.y.shape[0]
            != client.graph.num_nodes
        ):

            raise RuntimeError(
                f"Client {client_id}: "
                f"label count does not match "
                f"node count."
            )

        # ----------------------------------------------------
        # Validate features
        # ----------------------------------------------------

        if client.graph.x.isnan().any():

            raise RuntimeError(
                f"Client {client_id}: "
                f"NaN detected in features."
            )

        if client.graph.x.isinf().any():

            raise RuntimeError(
                f"Client {client_id}: "
                f"infinity detected in features."
            )

        # ----------------------------------------------------
        # PASS
        # ----------------------------------------------------

        print(
            f"Client {client_id}: PASS"
        )

        print(
            f"  Nodes     : "
            f"{client.graph.num_nodes:,}"
        )

        print(
            f"  Features  : "
            f"{client.graph.num_node_features}"
        )

        print(
            f"  Edges     : "
            f"{client.graph.num_edges:,}"
        )

        print(
            f"  Parameters: "
            f"{len(parameters)} tensors"
        )

        print()


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("=" * 70)

    print(
        "ALL THREE FLOWER CLIENTS: PASS"
    )

    print()

    print(
        "Flower Client 1: READY"
    )

    print(
        "Flower Client 2: READY"
    )

    print(
        "Flower Client 3: READY"
    )

    print()

    print(
        "Flower client launcher is ready."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()