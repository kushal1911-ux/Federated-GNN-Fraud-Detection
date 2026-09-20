# ============================================================
# RFGN REINFORCEMENT LEARNING - DQN TRAINING V2
# ============================================================
#
# Purpose:
# Train the DQN agent for improved fraud-threshold optimization.
#
# Previous model:
#   dqn_fraud_threshold.zip
#   5,000 timesteps
#
# New model:
#   dqn_fraud_threshold_20000.zip
#   20,000 timesteps
#
# Existing GraphSAGE model will NOT be modified.
# Existing graphs will NOT be modified.
# CSV datasets will NOT be modified.
#
# ============================================================

import os
import sys
import json
import random

import numpy as np
import torch

from stable_baselines3 import DQN


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
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from models.reinforcement.environment import (
    FraudThresholdEnvironment,
    load_global_predictions,
)

from models.reinforcement.dqn_agent import (
    FraudThresholdGymEnvironment,
    create_dqn_agent,
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

MODEL_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "reinforcement",
)

os.makedirs(
    MODEL_OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# NEW MODEL PATHS
# ============================================================

DQN_MODEL_PATH = os.path.join(
    MODEL_OUTPUT_DIR,
    "dqn_fraud_threshold_20000",
)

DQN_RESULTS_PATH = os.path.join(
    MODEL_OUTPUT_DIR,
    "dqn_threshold_results_20000.json",
)

DQN_MANIFEST_PATH = os.path.join(
    MODEL_OUTPUT_DIR,
    "dqn_training_manifest_20000.json",
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

TOTAL_TIMESTEPS = 20000

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# DIRECT THRESHOLD EVALUATION
# ============================================================

def evaluate_all_thresholds(
    fraud_probabilities,
    labels,
):

    environment = FraudThresholdEnvironment(
        fraud_probabilities,
        labels,
    )

    results = []

    for action in range(
        environment.num_actions
    ):

        (
            _state,
            reward,
            _terminated,
            _truncated,
            info,
        ) = environment.step(
            action
        )

        results.append(
            {
                "action": int(action),

                "threshold": float(
                    info["threshold"]
                ),

                "reward": float(
                    reward
                ),

                "accuracy": float(
                    info["accuracy"]
                ),

                "precision": float(
                    info["precision"]
                ),

                "recall": float(
                    info["recall"]
                ),

                "f1": float(
                    info["f1"]
                ),

                "review_rate": float(
                    info["review_rate"]
                ),

                "true_positive": int(
                    info["true_positive"]
                ),

                "true_negative": int(
                    info["true_negative"]
                ),

                "false_positive": int(
                    info["false_positive"]
                ),

                "false_negative": int(
                    info["false_negative"]
                ),
            }
        )


    best_result = max(
        results,
        key=lambda item: item["reward"],
    )


    return (
        results,
        best_result,
    )


# ============================================================
# GET DQN Q-VALUES
# ============================================================

def get_q_values(
    model,
    observation,
):

    observation_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=model.device,
    ).unsqueeze(0)


    with torch.no_grad():

        q_values = model.policy.q_net(
            observation_tensor
        )


    return (
        q_values
        .detach()
        .cpu()
        .numpy()
        .flatten()
    )


# ============================================================
# EVALUATE TRAINED DQN
# ============================================================

def evaluate_dqn(
    model,
    environment,
):

    observation, _ = (
        environment.reset()
    )


    q_values = get_q_values(
        model,
        observation,
    )


    if len(q_values) != (
        environment.action_space.n
    ):

        raise RuntimeError(
            "DQN Q-value count does not "
            "match action count."
        )


    if not np.isfinite(
        q_values
    ).all():

        raise RuntimeError(
            "DQN Q-values contain NaN "
            "or infinity."
        )


    selected_action = int(
        np.argmax(q_values)
    )


    (
        _state,
        reward,
        _terminated,
        _truncated,
        info,
    ) = environment.step(
        selected_action
    )


    threshold = float(
        info["threshold"]
    )


    return (
        selected_action,
        threshold,
        reward,
        info,
        q_values,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "RFGN DQN TRAINING - 20,000 TIMESTEPS"
    )

    print("=" * 70)

    print()

    print(
        "Training a new DQN model."
    )

    print(
        "The previous 5,000-timestep model "
        "will NOT be modified."
    )

    print(
        "Global GraphSAGE model will NOT be modified."
    )

    print(
        "Graphs will NOT be modified."
    )

    print(
        "CSV datasets will NOT be modified."
    )

    print()

    print(
        f"Training timesteps : "
        f"{TOTAL_TIMESTEPS:,}"
    )

    print(
        f"Random seed        : "
        f"{SEED}"
    )


    # ========================================================
    # LOAD GLOBAL GRAPHSAGE OUTPUT
    # ========================================================

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
        "Global fraud probabilities: PASS"
    )


    # ========================================================
    # CREATE ENVIRONMENT
    # ========================================================

    print()

    print("-" * 70)

    print(
        "CREATING DQN ENVIRONMENT"
    )

    print("-" * 70)


    environment = FraudThresholdGymEnvironment(
        fraud_probabilities,
        labels,
    )


    print(
        "DQN environment: CREATED"
    )


    print(
        f"Observation shape: "
        f"{environment.observation_space.shape}"
    )


    print(
        f"Action count: "
        f"{environment.action_space.n}"
    )


    # ========================================================
    # DIRECT REFERENCE
    # ========================================================

    print()

    print("-" * 70)

    print(
        "CALCULATING DIRECT THRESHOLD REFERENCE"
    )

    print("-" * 70)


    (
        threshold_results,
        best_direct_result,
    ) = evaluate_all_thresholds(
        fraud_probabilities,
        labels,
    )


    print(
        "Threshold evaluation: PASS"
    )


    print()

    print(
        f"Direct optimum threshold : "
        f"{best_direct_result['threshold']:.2f}"
    )


    print(
        f"Direct optimum reward    : "
        f"{best_direct_result['reward']:.4f}"
    )


    print(
        f"Precision               : "
        f"{best_direct_result['precision']:.6f}"
    )


    print(
        f"Recall                  : "
        f"{best_direct_result['recall']:.6f}"
    )


    print(
        f"F1 Score                : "
        f"{best_direct_result['f1']:.6f}"
    )


    print(
        f"Review rate             : "
        f"{best_direct_result['review_rate'] * 100:.4f}%"
    )


    # ========================================================
    # CREATE DQN
    # ========================================================

    print()

    print("-" * 70)

    print(
        "CREATING NEW DQN AGENT"
    )

    print("-" * 70)


    model = create_dqn_agent(
        environment
    )


    print(
        "DQN agent: CREATED"
    )


    print(
        f"Device: "
        f"{model.device}"
    )


    # ========================================================
    # TRAIN
    # ========================================================

    print()

    print("-" * 70)

    print(
        "TRAINING DQN"
    )

    print("-" * 70)


    print(
        "Training started..."
    )


    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=False,
    )


    print(
        "DQN training: COMPLETED"
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    print()

    print("-" * 70)

    print(
        "SAVING NEW DQN MODEL"
    )

    print("-" * 70)


    model.save(
        DQN_MODEL_PATH
    )


    print(
        "New DQN model saved:"
    )


    print(
        f"{DQN_MODEL_PATH}.zip"
    )


    # ========================================================
    # EVALUATE
    # ========================================================

    print()

    print("-" * 70)

    print(
        "EVALUATING TRAINED DQN"
    )

    print("-" * 70)


    (
        selected_action,
        selected_threshold,
        dqn_reward,
        dqn_info,
        q_values,
    ) = evaluate_dqn(
        model,
        environment,
    )


    print(
        "DQN evaluation: PASS"
    )


    print()

    print(
        f"Selected action   : "
        f"{selected_action}"
    )


    print(
        f"Selected threshold: "
        f"{selected_threshold:.2f}"
    )


    print(
        f"DQN reward        : "
        f"{dqn_reward:.4f}"
    )


    print(
        f"Accuracy          : "
        f"{dqn_info['accuracy']:.6f}"
    )


    print(
        f"Precision         : "
        f"{dqn_info['precision']:.6f}"
    )


    print(
        f"Recall            : "
        f"{dqn_info['recall']:.6f}"
    )


    print(
        f"F1 Score          : "
        f"{dqn_info['f1']:.6f}"
    )


    print(
        f"Review rate       : "
        f"{dqn_info['review_rate'] * 100:.4f}%"
    )


    # ========================================================
    # Q-VALUE ANALYSIS
    # ========================================================

    print()

    print("-" * 70)

    print(
        "DQN Q-VALUE ANALYSIS"
    )

    print("-" * 70)


    thresholds = (
        environment.environment.thresholds
    )


    for action, q_value in enumerate(
        q_values
    ):

        print(
            f"Action {action:2d} | "
            f"Threshold {thresholds[action]:.2f} | "
            f"Q-value {q_value:.6f}"
        )


    # ========================================================
    # COMPARE WITH DIRECT OPTIMUM
    # ========================================================

    print()

    print("-" * 70)

    print(
        "DQN VS DIRECT THRESHOLD REFERENCE"
    )

    print("-" * 70)


    direct_threshold = float(
        best_direct_result[
            "threshold"
        ]
    )


    direct_reward = float(
        best_direct_result[
            "reward"
        ]
    )


    reward_difference = (
        direct_reward
        - dqn_reward
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
        f"{dqn_reward:.4f}"
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

        threshold_agreement = True

        print(
            "Threshold agreement: PASS"
        )

    else:

        threshold_agreement = False

        print(
            "Threshold agreement: DIFFERENT"
        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print()

    print("-" * 70)

    print(
        "SAVING DQN RESULTS"
    )

    print("-" * 70)


    results = {

        "training_timesteps":
            TOTAL_TIMESTEPS,

        "seed":
            SEED,

        "selected_action":
            int(selected_action),

        "selected_threshold":
            float(selected_threshold),

        "dqn_reward":
            float(dqn_reward),

        "accuracy":
            float(dqn_info["accuracy"]),

        "precision":
            float(dqn_info["precision"]),

        "recall":
            float(dqn_info["recall"]),

        "f1":
            float(dqn_info["f1"]),

        "review_rate":
            float(dqn_info["review_rate"]),

        "true_positive":
            int(dqn_info["true_positive"]),

        "true_negative":
            int(dqn_info["true_negative"]),

        "false_positive":
            int(dqn_info["false_positive"]),

        "false_negative":
            int(dqn_info["false_negative"]),

        "direct_optimum_threshold":
            direct_threshold,

        "direct_optimum_reward":
            direct_reward,

        "reward_difference":
            float(reward_difference),

        "threshold_agreement":
            bool(threshold_agreement),

        "model_path":
            f"{DQN_MODEL_PATH}.zip",

        "threshold_results":
            threshold_results,
    }


    with open(
        DQN_RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )


    print(
        f"Results saved:"
    )


    print(
        DQN_RESULTS_PATH
    )


    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    manifest = {

        "component":
            "Reinforcement Learning",

        "algorithm":
            "Deep Q-Network",

        "framework":
            "Stable-Baselines3",

        "environment":
            "FraudThresholdEnvironment",

        "global_model":
            "global_graphsage_flower.pt",

        "input_dimension":
            769,

        "output_classes":
            2,

        "threshold_min":
            0.05,

        "threshold_max":
            0.95,

        "threshold_step":
            0.05,

        "number_of_actions":
            19,

        "total_timesteps":
            TOTAL_TIMESTEPS,

        "seed":
            SEED,

        "selected_action":
            int(selected_action),

        "selected_threshold":
            float(selected_threshold),

        "dqn_reward":
            float(dqn_reward),

        "direct_optimum_threshold":
            direct_threshold,

        "direct_optimum_reward":
            direct_reward,

        "reward_difference":
            float(reward_difference),

        "threshold_agreement":
            bool(threshold_agreement),

        "model_path":
            f"{DQN_MODEL_PATH}.zip",

        "results_path":
            DQN_RESULTS_PATH,
    }


    with open(
        DQN_MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
        )


    print(
        "Training manifest saved:"
    )


    print(
        DQN_MANIFEST_PATH
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print("=" * 70)

    print(
        "RFGN DQN TRAINING SUMMARY"
    )

    print("=" * 70)

    print()

    print(
        f"Training timesteps : "
        f"{TOTAL_TIMESTEPS:,}"
    )


    print(
        f"Selected action    : "
        f"{selected_action}"
    )


    print(
        f"Selected threshold : "
        f"{selected_threshold:.2f}"
    )


    print(
        f"DQN reward         : "
        f"{dqn_reward:.4f}"
    )


    print(
        f"Precision           : "
        f"{dqn_info['precision']:.6f}"
    )


    print(
        f"Recall              : "
        f"{dqn_info['recall']:.6f}"
    )


    print(
        f"F1 Score            : "
        f"{dqn_info['f1']:.6f}"
    )


    print(
        f"Review rate         : "
        f"{dqn_info['review_rate'] * 100:.4f}%"
    )


    print()

    print(
        f"Direct optimum      : "
        f"{direct_threshold:.2f}"
    )


    print(
        f"Reward difference   : "
        f"{reward_difference:.4f}"
    )


    print()

    print(
        "New DQN model saved : YES"
    )


    print(
        "Results saved       : YES"
    )


    print(
        "Manifest saved      : YES"
    )


    print()

    print("=" * 70)

    print(
        "RFGN DQN TRAINING: COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()