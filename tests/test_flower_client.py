# ============================================================
# RFGN FLOWER CLIENT TEST
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
# IMPORT
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
    print("RFGN FLOWER CLIENT TEST")
    print("=" * 70)

    print()
    print(
        "Creating Client 1 Flower client..."
    )

    client = FraudGraphSAGEClient(
        1
    )

    print(
        "Flower client creation: PASS"
    )

    print()
    print(
        "Client 1 graph:"
    )

    print(
        f"Nodes    : "
        f"{client.graph.num_nodes:,}"
    )

    print(
        f"Features : "
        f"{client.graph.num_node_features:,}"
    )

    print(
        f"Edges    : "
        f"{client.graph.num_edges:,}"
    )

    print()
    print(
        "Testing parameter extraction..."
    )

    parameters = client.get_parameters(
        {}
    )

    print(
        f"Parameter tensors: "
        f"{len(parameters)}"
    )

    if len(parameters) != 8:

        raise RuntimeError(
            "Expected 8 parameter tensors."
        )

    print(
        "Parameter extraction: PASS"
    )

    print()
    print(
        "Testing parameter loading..."
    )

    client.set_parameters(
        parameters
    )

    print(
        "Parameter loading: PASS"
    )

    print()
    print(
        "Testing Flower client interface..."
    )

    required_methods = [
        "get_parameters",
        "set_parameters",
        "fit",
        "evaluate",
    ]

    for method in required_methods:

        if not hasattr(
            client,
            method
        ):

            raise RuntimeError(
                f"Missing method: {method}"
            )

        print(
            f"{method}: PASS"
        )

    print()
    print("=" * 70)
    print(
        "FLOWER CLIENT TEST: PASSED"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()