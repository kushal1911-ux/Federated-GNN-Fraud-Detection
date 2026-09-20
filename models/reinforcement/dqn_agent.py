# ============================================================
# RFGN REINFORCEMENT LEARNING - DQN AGENT
# ============================================================
#
# Purpose:
# Create and test the DQN agent used for fraud-threshold
# optimization.
#
# This file does NOT perform full DQN training.
# It only creates the DQN-compatible environment wrapper,
# creates the DQN model, and performs a basic prediction test.
#
# ============================================================

import os
import sys

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN


# ============================================================
# PROJECT ROOT
# ============================================================

# File:
# E:\RFGN\models\reinforcement\dqn_agent.py
#
# Project root:
# E:\RFGN

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
# IMPORT REINFORCEMENT LEARNING ENVIRONMENT
# ============================================================

from models.reinforcement.environment import (
    FraudThresholdEnvironment,
    load_global_predictions,
)


# ============================================================
# DQN CONFIGURATION
# ============================================================

DQN_LEARNING_RATE = 0.0001

DQN_BUFFER_SIZE = 10000

DQN_LEARNING_STARTS = 100

DQN_BATCH_SIZE = 32

DQN_GAMMA = 0.99

DQN_TRAIN_FREQUENCY = 1

DQN_TARGET_UPDATE_INTERVAL = 500

DQN_EXPLORATION_FRACTION = 0.10

DQN_EXPLORATION_INITIAL_EPS = 1.0

DQN_EXPLORATION_FINAL_EPS = 0.05

DQN_POLICY = "MlpPolicy"


# ============================================================
# GYMNASIUM WRAPPER
# ============================================================

class FraudThresholdGymEnvironment(
    gym.Env
):
    """
    Gymnasium-compatible wrapper around the
    FraudThresholdEnvironment.

    Observation:
        5 continuous state values.

    Action:
        Discrete threshold index.
    """

    metadata = {
        "render_modes": []
    }


    def __init__(
        self,
        fraud_probabilities,
        labels,
    ):

        super().__init__()


        # ----------------------------------------------------
        # Base environment
        # ----------------------------------------------------

        self.environment = (
            FraudThresholdEnvironment(
                fraud_probabilities,
                labels,
            )
        )


        # ----------------------------------------------------
        # Observation space
        # ----------------------------------------------------

        self.observation_space = (
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(5,),
                dtype=np.float32,
            )
        )


        # ----------------------------------------------------
        # Action space
        # ----------------------------------------------------

        self.action_space = (
            spaces.Discrete(
                self.environment.num_actions
            )
        )


    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):

        super().reset(
            seed=seed
        )


        state = (
            self.environment.reset()
        )


        return (
            state.astype(
                np.float32
            ),
            {},
        )


    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        action,
    ):

        (
            state,
            reward,
            terminated,
            truncated,
            info,
        ) = self.environment.step(
            int(action)
        )


        return (
            state.astype(
                np.float32
            ),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )


    # ========================================================
    # RENDER
    # ========================================================

    def render(self):

        return None


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        return None


# ============================================================
# CREATE DQN MODEL
# ============================================================

def create_dqn_agent(
    environment,
):
    """
    Create the Stable-Baselines3 DQN agent.
    """

    model = DQN(
        policy=DQN_POLICY,

        env=environment,

        learning_rate=DQN_LEARNING_RATE,

        buffer_size=DQN_BUFFER_SIZE,

        learning_starts=DQN_LEARNING_STARTS,

        batch_size=DQN_BATCH_SIZE,

        gamma=DQN_GAMMA,

        train_freq=DQN_TRAIN_FREQUENCY,

        target_update_interval=(
            DQN_TARGET_UPDATE_INTERVAL
        ),

        exploration_fraction=(
            DQN_EXPLORATION_FRACTION
        ),

        exploration_initial_eps=(
            DQN_EXPLORATION_INITIAL_EPS
        ),

        exploration_final_eps=(
            DQN_EXPLORATION_FINAL_EPS
        ),

        verbose=0,

        device="auto",
    )


    return model


# ============================================================
# TEST DQN PREDICTION
# ============================================================

def test_dqn_prediction(
    model,
    environment,
):
    """
    Perform a prediction using the newly created DQN agent.

    No training is performed.
    """

    observation, _ = (
        environment.reset()
    )


    action, _states = (
        model.predict(
            observation,
            deterministic=True,
        )
    )


    action = int(
        np.asarray(action).item()
    )


    if (
        action < 0
        or
        action >= environment.action_space.n
    ):

        raise RuntimeError(
            "DQN produced an invalid action."
        )


    threshold = float(
        environment.environment.thresholds[
            action
        ]
    )


    return (
        observation,
        action,
        threshold,
    )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "RFGN DQN AGENT TEST"
    )

    print("=" * 70)

    print()

    print(
        "No DQN training will be performed."
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
        "Global GraphSAGE probabilities: PASS"
    )


    # ========================================================
    # CREATE GYMNASIUM ENVIRONMENT
    # ========================================================

    print()

    print("-" * 70)

    print(
        "CREATING DQN ENVIRONMENT"
    )

    print("-" * 70)


    environment = (
        FraudThresholdGymEnvironment(
            fraud_probabilities,
            labels,
        )
    )


    print(
        "Gymnasium environment: CREATED"
    )


    # ========================================================
    # VALIDATE OBSERVATION SPACE
    # ========================================================

    print()

    print(
        "Observation space:"
    )

    print(
        environment.observation_space
    )


    if (
        environment.observation_space.shape
        != (5,)
    ):

        raise RuntimeError(
            "Observation space shape is incorrect."
        )


    print(
        "Observation space: PASS"
    )


    # ========================================================
    # VALIDATE ACTION SPACE
    # ========================================================

    print()

    print(
        "Action space:"
    )

    print(
        environment.action_space
    )


    if (
        environment.action_space.n
        != 19
    ):

        raise RuntimeError(
            "Expected 19 threshold actions."
        )


    print(
        "Action space: PASS"
    )


    # ========================================================
    # RESET ENVIRONMENT
    # ========================================================

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


    if observation.shape != (5,):

        raise RuntimeError(
            "Observation shape is incorrect."
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


    # ========================================================
    # CREATE DQN AGENT
    # ========================================================

    print()

    print("-" * 70)

    print(
        "CREATING DQN AGENT"
    )

    print("-" * 70)


    model = create_dqn_agent(
        environment
    )


    print(
        "DQN model: CREATED"
    )


    print(
        f"Policy: "
        f"{DQN_POLICY}"
    )


    print(
        f"Learning rate: "
        f"{DQN_LEARNING_RATE}"
    )


    print(
        f"Discount factor: "
        f"{DQN_GAMMA}"
    )


    print(
        f"Batch size: "
        f"{DQN_BATCH_SIZE}"
    )


    print(
        f"Replay buffer size: "
        f"{DQN_BUFFER_SIZE}"
    )


    print(
        f"Action count: "
        f"{environment.action_space.n}"
    )


    # ========================================================
    # PARAMETER VALIDATION
    # ========================================================

    print()

    print("-" * 70)

    print(
        "VALIDATING DQN MODEL"
    )

    print("-" * 70)


    parameter_count = sum(
        parameter.numel()
        for parameter in model.policy.parameters()
    )


    print(
        f"DQN policy parameters: "
        f"{parameter_count:,}"
    )


    if parameter_count <= 0:

        raise RuntimeError(
            "DQN policy contains no parameters."
        )


    print(
        "DQN policy parameters: PASS"
    )


    # ========================================================
    # PREDICTION TEST
    # ========================================================

    print()

    print("-" * 70)

    print(
        "TESTING DQN PREDICTION"
    )

    print("-" * 70)


    (
        observation,
        action,
        threshold,
    ) = test_dqn_prediction(
        model,
        environment,
    )


    print(
        f"Predicted action: "
        f"{action}"
    )


    print(
        f"Selected threshold: "
        f"{threshold:.2f}"
    )


    if (
        action < 0
        or
        action >= environment.action_space.n
    ):

        raise RuntimeError(
            "Invalid DQN action."
        )


    if not (
        0.05
        <= threshold
        <= 0.95
    ):

        raise RuntimeError(
            "DQN selected threshold outside "
            "the allowed range."
        )


    print(
        "DQN prediction: PASS"
    )


    # ========================================================
    # THRESHOLD ACTION MAP
    # ========================================================

    print()

    print("-" * 70)

    print(
        "DQN THRESHOLD ACTION SPACE"
    )

    print("-" * 70)


    for index, value in enumerate(
        environment.environment.thresholds
    ):

        print(
            f"Action {index:2d} "
            f"-> threshold {value:.2f}"
        )


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print()

    print("=" * 70)

    print(
        "RFGN DQN AGENT TEST: PASSED"
    )

    print("=" * 70)

    print()

    print(
        "Global GraphSAGE integration: PASS"
    )

    print(
        "Gymnasium environment: PASS"
    )

    print(
        "Observation space: PASS"
    )

    print(
        "Action space: PASS"
    )

    print(
        "DQN model creation: PASS"
    )

    print(
        "DQN policy parameters: PASS"
    )

    print(
        "DQN prediction: PASS"
    )

    print()

    print(
        "DQN agent is ready for training."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()