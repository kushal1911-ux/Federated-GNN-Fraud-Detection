# ============================================================
# RFGN FLOWER SERVER TEST
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
# IMPORT SERVER COMPONENTS
# ============================================================

from training.flower_server import (
    create_model,
    get_initial_parameters,
    RFGNFedAvg,
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "RFGN FLOWER SERVER TEST"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()

    print(
        "Creating GraphSAGE model..."
    )


    model = create_model()


    print(
        "GraphSAGE model: PASS"
    )


    parameter_count = sum(
        parameter.numel()
        for parameter in
        model.parameters()
    )


    print(
        f"Parameter count: "
        f"{parameter_count:,}"
    )


    if parameter_count != 230146:

        raise RuntimeError(
            "Unexpected parameter count."
        )


    print(
        "Parameter count: PASS"
    )


    # --------------------------------------------------------
    # INITIAL PARAMETERS
    # --------------------------------------------------------

    print()

    print(
        "Testing initial parameter conversion..."
    )


    parameters = (
        get_initial_parameters()
    )


    print(
        "Initial parameters: PASS"
    )


    # --------------------------------------------------------
    # FEDAVG STRATEGY
    # --------------------------------------------------------

    print()

    print(
        "Creating FedAvg strategy..."
    )


    strategy = RFGNFedAvg(

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=3,

        min_evaluate_clients=3,

        min_available_clients=3,

        initial_parameters=parameters,
    )


    print(
        "FedAvg strategy: PASS"
    )


    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "FLOWER SERVER TEST: PASSED"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()