# ============================================================
# RFGN FEDERATED MODEL COMPATIBILITY VALIDATION
# ============================================================

import os
import sys

import torch


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
# CONFIGURATION
# ============================================================

EXPECTED_INPUT_DIM = 769
EXPECTED_HIDDEN_DIM = 128
EXPECTED_OUTPUT_DIM = 2
EXPECTED_DROPOUT = 0.3


CLIENTS = [
    "client_1",
    "client_2",
    "client_3",
]


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATHS = {
    client: os.path.join(
        PROJECT_ROOT,
        "saved_models",
        "local",
        f"{client}_graphsage_aligned_best.pt",
    )
    for client in CLIENTS
}


# ============================================================
# SECTION
# ============================================================

def section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "RFGN FEDERATED MODEL COMPATIBILITY VALIDATION"
    )

    print("=" * 70)

    print()

    print(
        "This script is READ-ONLY."
    )

    print(
        "No model will be modified."
    )

    print(
        "No training will be performed."
    )

    print(
        "No FedAvg aggregation will be performed."
    )


    # ========================================================
    # CHECK MODEL FILES
    # ========================================================

    section(
        "CHECKING LOCAL MODEL FILES"
    )


    for client in CLIENTS:

        path = MODEL_PATHS[client]

        print()
        print(client.upper())

        print(
            f"Path: {path}"
        )


        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Model file not found:\n{path}"
            )


        size_mb = (
            os.path.getsize(path)
            /
            (
                1024 * 1024
            )
        )


        print(
            f"Size: {size_mb:.2f} MB"
        )

        print(
            "Model file: PRESENT"
        )


    # ========================================================
    # LOAD MODELS
    # ========================================================

    section(
        "LOADING LOCAL MODELS"
    )


    checkpoints = {}


    for client in CLIENTS:

        print()

        print(
            f"Loading {client}..."
        )


        checkpoint = torch.load(
            MODEL_PATHS[client],
            map_location="cpu",
            weights_only=False,
        )


        checkpoints[client] = checkpoint


        print(
            f"{client}: LOAD PASS"
        )


    # ========================================================
    # CHECK CHECKPOINT STRUCTURE
    # ========================================================

    section(
        "VALIDATING CHECKPOINT STRUCTURE"
    )


    required_keys = [
        "model_state_dict",
        "input_dim",
        "hidden_dim",
        "output_dim",
        "dropout",
        "best_epoch",
        "best_val_loss",
    ]


    for client in CLIENTS:

        checkpoint = checkpoints[client]

        print()
        print(client.upper())


        for key in required_keys:

            if key not in checkpoint:

                raise KeyError(
                    f"{client} missing checkpoint key: {key}"
                )


            print(
                f"{key}: PRESENT"
            )


        print(
            "Checkpoint structure: PASS"
        )


    # ========================================================
    # MODEL CONFIGURATION
    # ========================================================

    section(
        "VALIDATING MODEL CONFIGURATION"
    )


    configurations = {}


    for client in CLIENTS:

        checkpoint = checkpoints[client]


        input_dim = checkpoint[
            "input_dim"
        ]

        hidden_dim = checkpoint[
            "hidden_dim"
        ]

        output_dim = checkpoint[
            "output_dim"
        ]

        dropout = checkpoint[
            "dropout"
        ]


        configurations[client] = (
            input_dim,
            hidden_dim,
            output_dim,
            dropout,
        )


        print()

        print(client.upper())

        print(
            f"Input dimension  : {input_dim}"
        )

        print(
            f"Hidden dimension : {hidden_dim}"
        )

        print(
            f"Output dimension : {output_dim}"
        )

        print(
            f"Dropout          : {dropout}"
        )


        if input_dim != EXPECTED_INPUT_DIM:

            raise RuntimeError(
                f"{client}: invalid input dimension."
            )


        print(
            "Input dimension: PASS"
        )


        if hidden_dim != EXPECTED_HIDDEN_DIM:

            raise RuntimeError(
                f"{client}: invalid hidden dimension."
            )


        print(
            "Hidden dimension: PASS"
        )


        if output_dim != EXPECTED_OUTPUT_DIM:

            raise RuntimeError(
                f"{client}: invalid output dimension."
            )


        print(
            "Output dimension: PASS"
        )


        if abs(
            float(dropout)
            -
            EXPECTED_DROPOUT
        ) > 1e-8:

            raise RuntimeError(
                f"{client}: invalid dropout."
            )


        print(
            "Dropout: PASS"
        )


    # ========================================================
    # CROSS-CLIENT CONFIGURATION
    # ========================================================

    section(
        "CROSS-CLIENT CONFIGURATION COMPATIBILITY"
    )


    reference_config = configurations[
        "client_1"
    ]


    for client in CLIENTS:

        if configurations[client] != reference_config:

            raise RuntimeError(
                f"{client} configuration "
                f"does not match Client 1."
            )


        print(
            f"{client}: configuration compatibility: PASS"
        )


    # ========================================================
    # STATE DICTIONARIES
    # ========================================================

    section(
        "VALIDATING MODEL STATE DICTIONARIES"
    )


    state_dicts = {
        client:
            checkpoints[client][
                "model_state_dict"
            ]
        for client in CLIENTS
    }


    for client in CLIENTS:

        state_dict = state_dicts[client]


        if not isinstance(
            state_dict,
            dict,
        ):

            raise RuntimeError(
                f"{client} model_state_dict "
                f"is not a dictionary."
            )


        print(
            f"{client}: state dictionary: PASS"
        )

        print(
            f"{client}: parameter tensors: "
            f"{len(state_dict)}"
        )


    # ========================================================
    # PARAMETER NAME COMPATIBILITY
    # ========================================================

    section(
        "VALIDATING PARAMETER NAMES"
    )


    reference_names = list(
        state_dicts[
            "client_1"
        ].keys()
    )


    for client in CLIENTS:

        names = list(
            state_dicts[
                client
            ].keys()
        )


        if names != reference_names:

            raise RuntimeError(
                f"{client} parameter names "
                f"do not match Client 1."
            )


        print(
            f"{client}: parameter names: PASS"
        )


    print()

    print(
        f"Total parameters/tensors: "
        f"{len(reference_names)}"
    )


    # ========================================================
    # PARAMETER SHAPE COMPATIBILITY
    # ========================================================

    section(
        "VALIDATING PARAMETER SHAPES"
    )


    for client in CLIENTS:

        state_dict = state_dicts[
            client
        ]


        all_shapes_valid = True


        for name in reference_names:

            reference_shape = (
                state_dicts[
                    "client_1"
                ][name].shape
            )


            current_shape = (
                state_dict[name].shape
            )


            if current_shape != reference_shape:

                all_shapes_valid = False

                print()

                print(
                    f"Shape mismatch:"
                )

                print(
                    f"Parameter: {name}"
                )

                print(
                    f"Client 1 : {reference_shape}"
                )

                print(
                    f"{client} : {current_shape}"
                )


        if not all_shapes_valid:

            raise RuntimeError(
                f"{client} parameter shapes "
                f"are incompatible."
            )


        print(
            f"{client}: parameter shapes: PASS"
        )


    # ========================================================
    # PARAMETER FINITE CHECK
    # ========================================================

    section(
        "VALIDATING MODEL PARAMETERS"
    )


    total_parameter_values = {}


    for client in CLIENTS:

        state_dict = state_dicts[
            client
        ]


        total_values = 0


        for name, tensor in state_dict.items():

            if not torch.is_tensor(tensor):

                raise RuntimeError(
                    f"{client}: {name} "
                    f"is not a tensor."
                )


            if not torch.is_floating_point(
                tensor
            ):

                continue


            if torch.isnan(
                tensor
            ).any():

                raise RuntimeError(
                    f"{client}: NaN detected "
                    f"in parameter {name}."
                )


            if torch.isinf(
                tensor
            ).any():

                raise RuntimeError(
                    f"{client}: infinity detected "
                    f"in parameter {name}."
                )


            total_values += tensor.numel()


        total_parameter_values[
            client
        ] = total_values


        print()

        print(
            f"{client}"
        )

        print(
            f"Parameter values checked: "
            f"{total_values:,}"
        )

        print(
            "Parameter NaN check: PASS"
        )

        print(
            "Parameter infinity check: PASS"
        )


    # ========================================================
    # PARAMETER COUNT COMPATIBILITY
    # ========================================================

    section(
        "VALIDATING PARAMETER COUNTS"
    )


    reference_count = (
        total_parameter_values[
            "client_1"
        ]
    )


    for client in CLIENTS:

        if (
            total_parameter_values[
                client
            ]
            !=
            reference_count
        ):

            raise RuntimeError(
                f"{client} parameter count "
                f"does not match Client 1."
            )


        print(
            f"{client}: parameter count: PASS"
        )


    # ========================================================
    # TRAINING METADATA
    # ========================================================

    section(
        "CHECKING TRAINING METADATA"
    )


    for client in CLIENTS:

        checkpoint = checkpoints[
            client
        ]


        print()

        print(
            client.upper()
        )


        print(
            f"Best epoch       : "
            f"{checkpoint['best_epoch']}"
        )


        print(
            f"Best validation loss : "
            f"{checkpoint['best_val_loss']:.6f}"
        )


        if checkpoint[
            "best_epoch"
        ] < 1:

            raise RuntimeError(
                f"{client}: invalid best epoch."
            )


        if not np_is_finite(
            checkpoint[
                "best_val_loss"
            ]
        ):

            raise RuntimeError(
                f"{client}: invalid validation loss."
            )


        print(
            "Training metadata: PASS"
        )


    # ========================================================
    # FINAL FEDERATED COMPATIBILITY
    # ========================================================

    section(
        "FINAL FEDERATED COMPATIBILITY CHECK"
    )


    print()

    print(
        "Client 1 input dimension : 769"
    )

    print(
        "Client 2 input dimension : 769"
    )

    print(
        "Client 3 input dimension : 769"
    )


    print()

    print(
        "Identical architecture: PASS"
    )

    print(
        "Identical parameter names: PASS"
    )

    print(
        "Identical parameter shapes: PASS"
    )

    print(
        "Identical parameter counts: PASS"
    )

    print(
        "Finite model parameters: PASS"
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    section(
        "RFGN FEDERATED MODEL COMPATIBILITY SUMMARY"
    )


    print()

    print(
        "Clients validated       : 3"
    )

    print(
        "Input features          : 769"
    )

    print(
        "Hidden dimension        : 128"
    )

    print(
        "Output classes          : 2"
    )

    print(
        "Dropout                 : 0.3"
    )

    print(
        f"Parameter values       : "
        f"{reference_count:,}"
    )

    print(
        "Parameter names match   : YES"
    )

    print(
        "Parameter shapes match  : YES"
    )

    print(
        "Parameters finite       : YES"
    )

    print(
        "FedAvg compatible       : YES"
    )


    print()

    print("=" * 70)

    print(
        "FEDERATED MODEL COMPATIBILITY: PASSED"
    )

    print("=" * 70)

    print()

    print(
        "All three local GraphSAGE models "
        "are ready for FedAvg aggregation."
    )


# ============================================================
# FINITE NUMBER CHECK
# ============================================================

def np_is_finite(value):

    try:

        return bool(
            np.isfinite(
                float(value)
            )
        )

    except Exception:

        return False


# ============================================================
# IMPORT NUMPY
# ============================================================

# Imported here so the helper remains isolated.
import numpy as np


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()