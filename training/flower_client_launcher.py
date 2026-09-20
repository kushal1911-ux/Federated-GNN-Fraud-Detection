# ============================================================
# RFGN FLOWER CLIENT LAUNCHER
# ============================================================
#
# LEVEL 5 - FEDERATED LEARNING
# STAGE 3 - CONNECT CLIENTS TO FLOWER SERVER
#
# Usage:
#   python training\flower_client_launcher.py 1
#   python training\flower_client_launcher.py 2
#   python training\flower_client_launcher.py 3
#
# ============================================================

import os
import sys

import flwr as fl


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
# SERVER CONFIGURATION
# ============================================================

SERVER_ADDRESS = "127.0.0.1:8080"


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "RFGN FLOWER CLIENT LAUNCHER"
    )

    print("=" * 70)

    print()

    print(
        "LEVEL 5 - FEDERATED LEARNING"
    )

    print(
        "STAGE 3 - CLIENT CONNECTION"
    )

    print()


    # ========================================================
    # CLIENT ID
    # ========================================================

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print()

        print(
            "python training\\flower_client_launcher.py 1"
        )

        print(
            "python training\\flower_client_launcher.py 2"
        )

        print(
            "python training\\flower_client_launcher.py 3"
        )

        raise SystemExit(1)


    try:

        client_id = int(
            sys.argv[1]
        )

    except ValueError:

        raise SystemExit(
            "Client ID must be 1, 2 or 3."
        )


    if client_id not in (1, 2, 3):

        raise SystemExit(
            "Client ID must be 1, 2 or 3."
        )


    # ========================================================
    # CREATE CLIENT
    # ========================================================

    print(
        f"Creating Client {client_id}..."
    )


    client = FraudGraphSAGEClient(
        client_id
    )


    print(
        f"Client {client_id}: CREATED"
    )


    # ========================================================
    # CLIENT INFORMATION
    # ========================================================

    print()

    print(
        f"Client ID       : {client_id}"
    )

    print(
        f"Nodes           : "
        f"{client.graph.num_nodes:,}"
    )

    print(
        f"Features        : "
        f"{client.graph.num_node_features}"
    )

    print(
        f"Edges           : "
        f"{client.graph.num_edges:,}"
    )

    print(
        f"Server          : "
        f"{SERVER_ADDRESS}"
    )


    # ========================================================
    # INITIAL PARAMETERS
    # ========================================================

    parameters = client.get_parameters(
        {}
    )


    print()

    print(
        f"Parameter tensors: "
        f"{len(parameters)}"
    )

    print(
        "Parameter interface: PASS"
    )


    # ========================================================
    # START FLOWER CLIENT
    # ========================================================

    print()

    print("=" * 70)

    print(
        f"CONNECTING CLIENT {client_id} "
        f"TO FLOWER SERVER"
    )

    print("=" * 70)

    print()

    print(
        "Waiting for federated instructions..."
    )

    print()


    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client.to_client(),
    )


    # ========================================================
    # CLIENT FINISHED
    # ========================================================

    print()

    print("=" * 70)

    print(
        f"CLIENT {client_id} "
        f"FEDERATED SESSION COMPLETED"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()