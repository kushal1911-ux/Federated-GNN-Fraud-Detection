# ============================================================
# RFGN DQN MODEL VALIDATION
# ============================================================
#
# Purpose:
# Validate the trained DQN threshold-optimization model.
#
# This script is READ-ONLY.
#
# It does NOT:
#   - Train the DQN
#   - Modify the DQN model
#   - Modify the GraphSAGE model
#   - Modify graphs
#   - Modify CSV datasets
#
# ============================================================

import os
import sys
import json

import numpy as np
import torch

from stable_baselines3 import DQN


# ============================================================
# PROJECT ROOT
# ============================================================

# File:
# E:\RFGN\tests\validate_dqn_model.py
#
# Project root:
# E:\RFGN

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PATHS
# ============================================================

DQN_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "reinforcement",
    "dqn_fraud_threshold.zip",
)

DQN_RESULTS_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "reinforcement",
    "dqn_threshold_results.json",
)

DQN_MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "reinforcement",
    "dqn_training_manifest.json",
)


# ============================================================
# EXPECTED CONFIGURATION
# ============================================================

EXPECTED_OBSERVATION_DIM = 5

EXPECTED_ACTION_COUNT = 19

EXPECTED_MIN_THRESHOLD = 0.05

EXPECTED_MAX_THRESHOLD = 0.95

EXPECTED_THRESHOLD_STEP = 0.05

EXPECTED_TIMESTEPS = 5000


# ============================================================
# IMPORT ENVIRONMENT
# ============================================================

from models.reinforcement.environment import (
    FraudThresholdEnvironment,
    load_global_predictions,
)

from models.reinforcement.dqn_agent import (
    FraudThresholdGymEnvironment,
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(
    path,
    name,
):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )


    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )


    return data


# ============================================================
# VALIDATE DQN FILE
# ============================================================

def validate_dqn_file():

    print()

    print("-" * 70)

    print(
        "CHECKING TRAINED DQN MODEL"
    )

    print("-" * 70)


    if not os.path.exists(
        DQN_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Trained DQN model not found:\n"
            f"{DQN_MODEL_PATH}"
        )


    file_size = (
        os.path.getsize(
            DQN_MODEL_PATH
        )
        / (
            1024 * 1024
        )
    )


    print(
        "DQN model file: PRESENT"
    )


    print(
        f"Path: {DQN_MODEL_PATH}"
    )


    print(
        f"Size: {file_size:.2f} MB"
    )


    return True


# ============================================================
# LOAD DQN
# ============================================================

def load_dqn():

    print()

    print("-" * 70)

    print(
        "LOADING TRAINED DQN"
    )

    print("-" * 70)


    model = DQN.load(
        DQN_MODEL_PATH,
        device="auto",
    )


    print(
        "DQN model loaded successfully."
    )


    return model


# ============================================================
# VALIDATE DQN CONFIGURATION
# ============================================================

def validate_dqn_configuration(
    model,
):

    print()

    print("-" * 70)

    print(
        "VALIDATING DQN CONFIGURATION"
    )

    print("-" * 70)


    observation_space = (
        model.observation_space
    )

    action_space = (
        model.action_space
    )


    print(
        f"Observation space: "
        f"{observation_space}"
    )


    print(
        f"Action space: "
        f"{action_space}"
    )


    # --------------------------------------------------------
    # Observation dimension
    # --------------------------------------------------------

    if (
        observation_space.shape
        != (
            EXPECTED_OBSERVATION_DIM,
        )
    ):

        raise RuntimeError(
            "DQN observation dimension mismatch."
        )


    print(
        "Observation dimension: PASS"
    )


    # --------------------------------------------------------
    # Action count
    # --------------------------------------------------------

    if (
        action_space.n
        != EXPECTED_ACTION_COUNT
    ):

        raise RuntimeError(
            "DQN action count mismatch."
        )


    print(
        "Action count: PASS"
    )


    # --------------------------------------------------------
    # Policy
    # --------------------------------------------------------

    print(
        f"Policy: "
        f"{model.policy.__class__.__name__}"
    )


    print(
        "DQN configuration: PASS"
    )


# ============================================================
# VALIDATE DQN PARAMETERS
# ============================================================

def validate_dqn_parameters(
    model,
):

    print()

    print("-" * 70)

    print(
        "VALIDATING DQN PARAMETERS"
    )

    print("-" * 70)


    parameters = list(
        model.policy.parameters()
    )


    if len(parameters) == 0:

        raise RuntimeError(
            "DQN contains no parameters."
        )


    total_values = 0


    nan_values = 0

    infinity_values = 0


    for parameter in parameters:

        values = (
            parameter
            .detach()
            .cpu()
        )


        total_values += (
            values.numel()
        )


        nan_values += int(
            torch.isnan(
                values
            ).sum().item()
        )


        infinity_values += int(
            torch.isinf(
                values
            ).sum().item()
        )


    print(
        f"Parameter tensors: "
        f"{len(parameters)}"
    )


    print(
        f"Parameter values: "
        f"{total_values:,}"
    )


    print(
        f"NaN values: "
        f"{nan_values}"
    )


    print(
        f"Infinity values: "
        f"{infinity_values}"
    )


    if nan_values != 0:

        raise RuntimeError(
            "DQN parameters contain NaN."
        )


    if infinity_values != 0:

        raise RuntimeError(
            "DQN parameters contain infinity."
        )


    print(
        "Parameter NaN check: PASS"
    )


    print(
        "Parameter infinity check: PASS"
    )


# ============================================================
# LOAD GLOBAL GRAPHSAGE OUTPUT
# ============================================================

def load_environment_data():

    print()

    print("-" * 70)

    print(
        "LOADING GLOBAL GRAPHSAGE OUTPUT"
    )

    print("-" * 70)


    (
        fraud_probabilities,
        labels,
    ) = load_global_predictions()


    print(
        "Global GraphSAGE probabilities: PASS"
    )


    return (
        fraud_probabilities,
        labels,
    )


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

def create_environment(
    fraud_probabilities,
    labels,
):

    print()

    print("-" * 70)

    print(
        "CREATING REINFORCEMENT LEARNING ENVIRONMENT"
    )

    print("-" * 70)


    environment = (
        FraudThresholdGymEnvironment(
            fraud_probabilities,
            labels,
        )
    )


    print(
        "Environment created: PASS"
    )


    return environment


# ============================================================
# VALIDATE THRESHOLD ACTIONS
# ============================================================

def validate_threshold_actions(
    environment,
):

    print()

    print("-" * 70)

    print(
        "VALIDATING THRESHOLD ACTION SPACE"
    )

    print("-" * 70)


    thresholds = (
        environment.environment.thresholds
    )


    if len(thresholds) != 19:

        raise RuntimeError(
            "Unexpected threshold count."
        )


    print(
        f"Threshold count: "
        f"{len(thresholds)}"
    )


    print(
        f"Minimum threshold: "
        f"{thresholds.min():.2f}"
    )


    print(
        f"Maximum threshold: "
        f"{thresholds.max():.2f}"
    )


    # --------------------------------------------------------
    # Validate range
    # --------------------------------------------------------

    if not np.isclose(
        thresholds.min(),
        EXPECTED_MIN_THRESHOLD,
    ):

        raise RuntimeError(
            "Minimum threshold mismatch."
        )


    if not np.isclose(
        thresholds.max(),
        EXPECTED_MAX_THRESHOLD,
    ):

        raise RuntimeError(
            "Maximum threshold mismatch."
        )


    # --------------------------------------------------------
    # Validate step
    # --------------------------------------------------------

    differences = np.diff(
        thresholds
    )


    if not np.allclose(
        differences,
        EXPECTED_THRESHOLD_STEP,
        atol=1e-6,
    ):

        raise RuntimeError(
            "Threshold step mismatch."
        )


    print(
        "Threshold range: PASS"
    )


    print(
        "Threshold step: PASS"
    )


# ============================================================
# TEST DQN RESET
# ============================================================

def test_reset(
    environment,
):

    print()

    print("-" * 70)

    print(
        "TESTING DQN ENVIRONMENT RESET"
    )

    print("-" * 70)


    observation, info = (
        environment.reset()
    )


    print(
        f"Observation shape: "
        f"{observation.shape}"
    )


    print(
        f"Observation: "
        f"{observation}"
    )


    if observation.shape != (
        EXPECTED_OBSERVATION_DIM,
    ):

        raise RuntimeError(
            "Invalid observation shape."
        )


    if not np.isfinite(
        observation
    ).all():

        raise RuntimeError(
            "Observation contains NaN or infinity."
        )


    print(
        "Environment reset: PASS"
    )


    return observation


# ============================================================
# GET DQN Q VALUES
# ============================================================

def get_q_values(
    model,
    observation,
):

    observation_tensor = (
        torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=model.device,
        )
        .unsqueeze(0)
    )


    with torch.no_grad():

        q_values = (
            model.policy.q_net(
                observation_tensor
            )
        )


    q_values = (
        q_values
        .detach()
        .cpu()
        .numpy()
        .flatten()
    )


    return q_values


# ============================================================
# VALIDATE Q VALUES
# ============================================================

def validate_q_values(
    q_values,
):

    print()

    print("-" * 70)

    print(
        "VALIDATING DQN Q-VALUES"
    )

    print("-" * 70)


    print(
        f"Q-value count: "
        f"{len(q_values)}"
    )


    if len(q_values) != (
        EXPECTED_ACTION_COUNT
    ):

        raise RuntimeError(
            "Q-value count does not match "
            "action count."
        )


    if not np.isfinite(
        q_values
    ).all():

        raise RuntimeError(
            "DQN Q-values contain NaN or infinity."
        )


    print(
        "Q-value count: PASS"
    )


    print(
        "Q-value NaN check: PASS"
    )


    print(
        "Q-value infinity check: PASS"
    )


# ============================================================
# EVALUATE DQN DECISION
# ============================================================

def evaluate_dqn_decision(
    model,
    environment,
    observation,
):

    print()

    print("-" * 70)

    print(
        "EVALUATING TRAINED DQN DECISION"
    )

    print("-" * 70)


    q_values = get_q_values(
        model,
        observation,
    )


    validate_q_values(
        q_values
    )


    selected_action = int(
        np.argmax(
            q_values
        )
    )


    threshold = float(
        environment.environment.thresholds[
            selected_action
        ]
    )


    (
        next_state,
        reward,
        terminated,
        truncated,
        info,
    ) = environment.step(
        selected_action
    )


    print(
        f"Selected action : "
        f"{selected_action}"
    )


    print(
        f"Selected threshold: "
        f"{threshold:.2f}"
    )


    print(
        f"Reward: "
        f"{reward:.4f}"
    )


    print(
        f"Terminated: "
        f"{terminated}"
    )


    print(
        f"Truncated: "
        f"{truncated}"
    )


    # --------------------------------------------------------
    # Validate threshold
    # --------------------------------------------------------

    if (
        threshold
        < EXPECTED_MIN_THRESHOLD
        or
        threshold
        > EXPECTED_MAX_THRESHOLD
    ):

        raise RuntimeError(
            "Selected threshold is outside "
            "the allowed range."
        )


    print(
        "Selected threshold: PASS"
    )


    # --------------------------------------------------------
    # Validate reward
    # --------------------------------------------------------

    if not np.isfinite(
        reward
    ):

        raise RuntimeError(
            "DQN reward contains NaN or infinity."
        )


    print(
        "Reward validation: PASS"
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print()

    print(
        "DQN decision metrics:"
    )


    print(
        f"Accuracy       : "
        f"{info['accuracy']:.6f}"
    )


    print(
        f"Precision      : "
        f"{info['precision']:.6f}"
    )


    print(
        f"Recall         : "
        f"{info['recall']:.6f}"
    )


    print(
        f"F1 Score       : "
        f"{info['f1']:.6f}"
    )


    print(
        f"Review rate    : "
        f"{info['review_rate'] * 100:.4f}%"
    )


    print(
        f"True positive  : "
        f"{info['true_positive']:,}"
    )


    print(
        f"True negative  : "
        f"{info['true_negative']:,}"
    )


    print(
        f"False positive : "
        f"{info['false_positive']:,}"
    )


    print(
        f"False negative : "
        f"{info['false_negative']:,}"
    )


    return (
        selected_action,
        threshold,
        reward,
        info,
        q_values,
    )


# ============================================================
# LOAD SAVED RESULTS
# ============================================================

def validate_saved_results():

    print()

    print("-" * 70)

    print(
        "VALIDATING SAVED TRAINING RESULTS"
    )

    print("-" * 70)


    results = load_json(
        DQN_RESULTS_PATH,
        "DQN results",
    )


    required_keys = [

        "selected_action",

        "selected_threshold",

        "dqn_reward",

        "accuracy",

        "precision",

        "recall",

        "f1",

        "review_rate",

        "direct_optimum_threshold",

        "direct_optimum_reward",

    ]


    for key in required_keys:

        if key not in results:

            raise KeyError(
                f"Missing result field: {key}"
            )


    print(
        "Results structure: PASS"
    )


    print(
        f"Saved selected action: "
        f"{results['selected_action']}"
    )


    print(
        f"Saved selected threshold: "
        f"{results['selected_threshold']:.2f}"
    )


    print(
        f"Saved DQN reward: "
        f"{results['dqn_reward']:.4f}"
    )


    print(
        f"Saved direct optimum threshold: "
        f"{results['direct_optimum_threshold']:.2f}"
    )


    print(
        f"Saved direct optimum reward: "
        f"{results['direct_optimum_reward']:.4f}"
    )


    return results


# ============================================================
# VALIDATE MANIFEST
# ============================================================

def validate_manifest():

    print()

    print("-" * 70)

    print(
        "VALIDATING DQN TRAINING MANIFEST"
    )

    print("-" * 70)


    manifest = load_json(
        DQN_MANIFEST_PATH,
        "DQN training manifest",
    )


    required_keys = [

        "component",

        "algorithm",

        "framework",

        "environment",

        "global_model",

        "input_dimension",

        "output_classes",

        "threshold_min",

        "threshold_max",

        "threshold_step",

        "number_of_actions",

        "total_timesteps",

        "selected_action",

        "selected_threshold",

        "model_path",

    ]


    for key in required_keys:

        if key not in manifest:

            raise KeyError(
                f"Missing manifest field: {key}"
            )


    print(
        "Manifest structure: PASS"
    )


    if (
        manifest["component"]
        != "Reinforcement Learning"
    ):

        raise RuntimeError(
            "Manifest component mismatch."
        )


    if (
        manifest["algorithm"]
        != "Deep Q-Network"
    ):

        raise RuntimeError(
            "Manifest algorithm mismatch."
        )


    if (
        manifest["framework"]
        != "Stable-Baselines3"
    ):

        raise RuntimeError(
            "Manifest framework mismatch."
        )


    if (
        manifest["input_dimension"]
        != 769
    ):

        raise RuntimeError(
            "Manifest input dimension mismatch."
        )


    if (
        manifest["output_classes"]
        != 2
    ):

        raise RuntimeError(
            "Manifest output classes mismatch."
        )


    if (
        manifest["number_of_actions"]
        != EXPECTED_ACTION_COUNT
    ):

        raise RuntimeError(
            "Manifest action count mismatch."
        )


    print(
        "Component: PASS"
    )


    print(
        "Algorithm: PASS"
    )


    print(
        "Framework: PASS"
    )


    print(
        "Input dimension: PASS"
    )


    print(
        "Output classes: PASS"
    )


    print(
        "Action count: PASS"
    )


    return manifest


# ============================================================
# COMPARE SAVED RESULT WITH LIVE RESULT
# ============================================================

def compare_results(
    saved_results,
    selected_action,
    selected_threshold,
    reward,
    info,
):

    print()

    print("-" * 70)

    print(
        "COMPARING SAVED AND LIVE DQN RESULTS"
    )

    print("-" * 70)


    saved_action = int(
        saved_results[
            "selected_action"
        ]
    )


    saved_threshold = float(
        saved_results[
            "selected_threshold"
        ]
    )


    saved_reward = float(
        saved_results[
            "dqn_reward"
        ]
    )


    # --------------------------------------------------------
    # Action
    # --------------------------------------------------------

    if (
        saved_action
        == selected_action
    ):

        print(
            "Selected action consistency: PASS"
        )

    else:

        print(
            "Selected action consistency: DIFFERENT"
        )


    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    if np.isclose(
        saved_threshold,
        selected_threshold,
        atol=1e-6,
    ):

        print(
            "Selected threshold consistency: PASS"
        )

    else:

        print(
            "Selected threshold consistency: DIFFERENT"
        )


    # --------------------------------------------------------
    # Reward
    # --------------------------------------------------------

    if np.isclose(
        saved_reward,
        reward,
        atol=1e-3,
    ):

        print(
            "Reward consistency: PASS"
        )

    else:

        print(
            "Reward consistency: DIFFERENT"
        )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    saved_precision = float(
        saved_results[
            "precision"
        ]
    )


    saved_recall = float(
        saved_results[
            "recall"
        ]
    )


    saved_f1 = float(
        saved_results[
            "f1"
        ]
    )


    if np.isclose(
        saved_precision,
        info["precision"],
        atol=1e-6,
    ):

        print(
            "Precision consistency: PASS"
        )

    else:

        print(
            "Precision consistency: DIFFERENT"
        )


    if np.isclose(
        saved_recall,
        info["recall"],
        atol=1e-6,
    ):

        print(
            "Recall consistency: PASS"
        )

    else:

        print(
            "Recall consistency: DIFFERENT"
        )


    if np.isclose(
        saved_f1,
        info["f1"],
        atol=1e-6,
    ):

        print(
            "F1 consistency: PASS"
        )

    else:

        print(
            "F1 consistency: DIFFERENT"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "RFGN DQN MODEL VALIDATION"
    )

    print("=" * 70)

    print()

    print(
        "READ-ONLY VALIDATION"
    )

    print(
        "No DQN training will be performed."
    )

    print(
        "No model parameters will be updated."
    )

    print(
        "No GraphSAGE model will be modified."
    )

    print(
        "No graph will be modified."
    )

    print(
        "No CSV dataset will be modified."
    )


    # ========================================================
    # CHECK MODEL
    # ========================================================

    validate_dqn_file()


    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_dqn()


    # ========================================================
    # CONFIGURATION
    # ========================================================

    validate_dqn_configuration(
        model
    )


    # ========================================================
    # PARAMETERS
    # ========================================================

    validate_dqn_parameters(
        model
    )


    # ========================================================
    # GLOBAL GRAPHSAGE OUTPUT
    # ========================================================

    (
        fraud_probabilities,
        labels,
    ) = load_environment_data()


    # ========================================================
    # ENVIRONMENT
    # ========================================================

    environment = create_environment(
        fraud_probabilities,
        labels,
    )


    # ========================================================
    # THRESHOLDS
    # ========================================================

    validate_threshold_actions(
        environment
    )


    # ========================================================
    # RESET
    # ========================================================

    observation = test_reset(
        environment
    )


    # ========================================================
    # DQN DECISION
    # ========================================================

    (
        selected_action,
        selected_threshold,
        reward,
        info,
        q_values,
    ) = evaluate_dqn_decision(
        model,
        environment,
        observation,
    )


    # ========================================================
    # Q-VALUE TABLE
    # ========================================================

    print()

    print("-" * 70)

    print(
        "DQN Q-VALUE TABLE"
    )

    print("-" * 70)


    thresholds = (
        environment.environment.thresholds
    )


    for action in range(
        len(q_values)
    ):

        print(
            f"Action {action:2d} | "
            f"Threshold {thresholds[action]:.2f} | "
            f"Q-value {q_values[action]:.6f}"
        )


    # ========================================================
    # SAVED RESULTS
    # ========================================================

    saved_results = (
        validate_saved_results()
    )


    # ========================================================
    # MANIFEST
    # ========================================================

    validate_manifest()


    # ========================================================
    # CONSISTENCY
    # ========================================================

    compare_results(
        saved_results,
        selected_action,
        selected_threshold,
        reward,
        info,
    )


    # ========================================================
    # DIRECT OPTIMUM
    # ========================================================

    print()

    print("-" * 70)

    print(
        "DQN VS DIRECT THRESHOLD REFERENCE"
    )

    print("-" * 70)


    direct_threshold = float(
        saved_results[
            "direct_optimum_threshold"
        ]
    )


    direct_reward = float(
        saved_results[
            "direct_optimum_reward"
        ]
    )


    print(
        f"Direct optimum threshold : "
        f"{direct_threshold:.2f}"
    )


    print(
        f"DQN selected threshold   : "
        f"{selected_threshold:.2f}"
    )


    print(
        f"Direct optimum reward    : "
        f"{direct_reward:.4f}"
    )


    print(
        f"DQN reward               : "
        f"{reward:.4f}"
    )


    reward_difference = (
        direct_reward
        - reward
    )


    print(
        f"Reward difference        : "
        f"{reward_difference:.4f}"
    )


    if np.isclose(
        direct_threshold,
        selected_threshold,
        atol=1e-6,
    ):

        print(
            "Threshold agreement: PASS"
        )

    else:

        print(
            "Threshold agreement: DIFFERENT"
        )


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print()

    print("=" * 70)

    print(
        "RFGN DQN MODEL VALIDATION SUMMARY"
    )

    print("=" * 70)

    print()

    print(
        "DQN model file       : PRESENT"
    )


    print(
        "DQN model loading    : PASS"
    )


    print(
        "DQN configuration    : PASS"
    )


    print(
        "DQN parameters       : PASS"
    )


    print(
        "Environment          : PASS"
    )


    print(
        "Threshold actions    : PASS"
    )


    print(
        "Q-values             : PASS"
    )


    print(
        "DQN decision         : PASS"
    )


    print(
        f"Selected threshold   : "
        f"{selected_threshold:.2f}"
    )


    print(
        f"Reward               : "
        f"{reward:.4f}"
    )


    print(
        f"Precision            : "
        f"{info['precision']:.6f}"
    )


    print(
        f"Recall               : "
        f"{info['recall']:.6f}"
    )


    print(
        f"F1 Score             : "
        f"{info['f1']:.6f}"
    )


    print(
        f"Review rate          : "
        f"{info['review_rate'] * 100:.4f}%"
    )


    print(
        "Saved results        : PASS"
    )


    print(
        "Training manifest    : PASS"
    )


    print()

    print("=" * 70)

    print(
        "RFGN DQN MODEL VALIDATION: PASSED"
    )

    print("=" * 70)

    print()

    print(
        "The trained DQN model is valid and "
        "ready for final threshold optimization."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()