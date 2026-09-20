"""
RFGN - DQN Decision Layer

Connects the Global GraphSAGE fraud probability to the
trained DQN policy and produces the final transaction decision.

Input:
    Fraud probability from Global GraphSAGE

Output:
    DQN-selected threshold
    Final FRAUD / LEGITIMATE decision
    Decision metadata

The trained DQN model is used directly.
No threshold is hard-coded as the decision policy.

Class convention:
    0 = Legitimate
    1 = Fraud
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# DQN MODEL PATH
# ============================================================

DQN_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "reinforcement"
    / "dqn_fraud_threshold_20000.zip"
)


# ============================================================
# DQN CONFIGURATION
# ============================================================

EXPECTED_ACTION_COUNT = 19

MIN_THRESHOLD = 0.05
MAX_THRESHOLD = 0.95
THRESHOLD_STEP = 0.05

DEFAULT_THRESHOLD = 0.80

STATE_SIZE = 5


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cpu"


# ============================================================
# DQN DECISION LAYER
# ============================================================

class DQNDecisionLayer:
    """
    Loads the trained DQN policy and converts a GraphSAGE
    fraud probability into a final transaction decision.
    """

    def __init__(
        self,
        model_path: Path = DQN_MODEL_PATH,
        device: str = DEVICE,
    ) -> None:

        self.model_path = Path(model_path)
        self.device = device

        self.model = self._load_model()

        self.thresholds = self._build_thresholds()

        self._validate_model()

    # ========================================================
    # BUILD THRESHOLD ACTIONS
    # ========================================================

    def _build_thresholds(self) -> np.ndarray:
        """
        Build the same 19 threshold actions used by the
        RFGN reinforcement-learning environment.
        """

        thresholds = np.arange(
            MIN_THRESHOLD,
            MAX_THRESHOLD + (
                THRESHOLD_STEP / 2.0
            ),
            THRESHOLD_STEP,
            dtype=np.float32,
        )

        thresholds = np.round(
            thresholds,
            2,
        )

        if len(thresholds) != EXPECTED_ACTION_COUNT:

            raise ValueError(
                "Unexpected DQN action count.\n"
                f"Expected: {EXPECTED_ACTION_COUNT}\n"
                f"Actual:   {len(thresholds)}"
            )

        return thresholds

    # ========================================================
    # LOAD DQN
    # ========================================================

    def _load_model(self) -> Any:
        """
        Load the trained Stable-Baselines3 DQN model.
        """

        if not self.model_path.exists():

            raise FileNotFoundError(
                "Trained DQN model not found:\n"
                f"{self.model_path}"
            )

        try:

            from stable_baselines3 import DQN

        except ImportError as exc:

            raise ImportError(
                "stable-baselines3 is required to load "
                "the trained DQN model."
            ) from exc

        model = DQN.load(
            str(self.model_path),
            device=self.device,
        )

        return model

    # ========================================================
    # VALIDATE DQN
    # ========================================================

    def _validate_model(self) -> None:

        # ----------------------------------------------------
        # Action space
        # ----------------------------------------------------

        action_space = self.model.action_space

        if not hasattr(
            action_space,
            "n",
        ):

            raise ValueError(
                "DQN action space is not discrete."
            )

        if action_space.n != EXPECTED_ACTION_COUNT:

            raise ValueError(
                "DQN action count mismatch.\n"
                f"Expected: {EXPECTED_ACTION_COUNT}\n"
                f"Actual:   {action_space.n}"
            )

        # ----------------------------------------------------
        # Observation space
        # ----------------------------------------------------

        observation_space = (
            self.model.observation_space
        )

        if not hasattr(
            observation_space,
            "shape",
        ):

            raise ValueError(
                "DQN observation space has no shape."
            )

        if observation_space.shape != (
            STATE_SIZE,
        ):

            raise ValueError(
                "DQN observation dimension mismatch.\n"
                f"Expected: {(STATE_SIZE,)}\n"
                f"Actual:   {observation_space.shape}"
            )

    # ========================================================
    # BUILD STATE
    # ========================================================

    def build_state(
        self,
        fraud_probability: float,
        threshold: float = DEFAULT_THRESHOLD,
        fraud_rate: float = 0.0,
        predicted_fraud_rate: float = 0.0,
        review_rate: float = 0.0,
    ) -> np.ndarray:
        """
        Build the five-dimensional DQN state.

        State convention from the RFGN environment:

            [threshold,
             fraud_rate,
             mean_probability,
             predicted_fraud_rate,
             review_rate]

        For real-time single-transaction inference,
        the current transaction's fraud probability is used
        as mean_probability.
        """

        self._validate_probability(
            fraud_probability
        )

        self._validate_probability(
            threshold
        )

        self._validate_probability(
            fraud_rate
        )

        self._validate_probability(
            predicted_fraud_rate
        )

        self._validate_probability(
            review_rate
        )

        state = np.array(
            [
                threshold,
                fraud_rate,
                fraud_probability,
                predicted_fraud_rate,
                review_rate,
            ],
            dtype=np.float32,
        )

        if state.shape != (
            STATE_SIZE,
        ):

            raise ValueError(
                "Invalid DQN state shape."
            )

        if not np.all(
            np.isfinite(state)
        ):

            raise ValueError(
                "NaN or Infinity detected in DQN state."
            )

        return state

    # ========================================================
    # VALIDATE PROBABILITY
    # ========================================================

    @staticmethod
    def _validate_probability(
        value: float,
    ) -> None:

        if not np.isfinite(
            value
        ):

            raise ValueError(
                "Probability/state value must be finite."
            )

        if value < 0.0 or value > 1.0:

            raise ValueError(
                "Probability/state value must be between 0 and 1."
            )

    # ========================================================
    # SELECT ACTION
    # ========================================================

    def select_action(
        self,
        state: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Use the trained DQN policy to select a threshold action.
        """

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        if state.shape != (
            STATE_SIZE,
        ):

            raise ValueError(
                "DQN state must have shape [5].\n"
                f"Actual: {state.shape}"
            )

        if not np.all(
            np.isfinite(state)
        ):

            raise ValueError(
                "DQN state contains NaN or Infinity."
            )

        action, _ = self.model.predict(
            state,
            deterministic=True,
        )

        action = int(
            np.asarray(
                action
            ).reshape(-1)[0]
        )

        if not (
            0
            <= action
            < EXPECTED_ACTION_COUNT
        ):

            raise ValueError(
                "DQN produced an invalid action.\n"
                f"Action: {action}"
            )

        threshold = float(
            self.thresholds[action]
        )

        return {
            "action":
                action,

            "threshold":
                threshold,
        }

    # ========================================================
    # FINAL DECISION
    # ========================================================

    def decide(
        self,
        fraud_probability: float,
        fraud_rate: float = 0.0,
        predicted_fraud_rate: Optional[float] = None,
        review_rate: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Produce the final transaction decision.

        Decision rule:

            fraud_probability >= DQN threshold
                -> FRAUD

            fraud_probability < DQN threshold
                -> LEGITIMATE
        """

        self._validate_probability(
            fraud_probability
        )

        if predicted_fraud_rate is None:

            predicted_fraud_rate = (
                fraud_probability
            )

        # ----------------------------------------------------
        # Use the current default threshold as the state
        # context before DQN chooses an action.
        # ----------------------------------------------------

        state = self.build_state(
            fraud_probability=fraud_probability,
            threshold=DEFAULT_THRESHOLD,
            fraud_rate=fraud_rate,
            predicted_fraud_rate=predicted_fraud_rate,
            review_rate=review_rate,
        )

        selected = self.select_action(
            state
        )

        action = selected[
            "action"
        ]

        threshold = selected[
            "threshold"
        ]

        # ----------------------------------------------------
        # Final decision
        # ----------------------------------------------------

        if fraud_probability >= threshold:

            decision = "FRAUD"

            predicted_class = 1

        else:

            decision = "LEGITIMATE"

            predicted_class = 0

        return {
            "fraud_probability":
                float(
                    fraud_probability
                ),

            "dqn_action":
                int(action),

            "dqn_threshold":
                float(threshold),

            "decision":
                decision,

            "predicted_class":
                predicted_class,

            "model":
                "GraphSAGE + DQN",
        }

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:

        return {
            "module":
                "DQNDecisionLayer",

            "status":
                "healthy",

            "model_path":
                str(
                    self.model_path
                ),

            "action_count":
                int(
                    self.model.action_space.n
                ),

            "state_size":
                STATE_SIZE,

            "minimum_threshold":
                MIN_THRESHOLD,

            "maximum_threshold":
                MAX_THRESHOLD,

            "threshold_step":
                THRESHOLD_STEP,

            "default_threshold":
                DEFAULT_THRESHOLD,
        }


# ============================================================
# SELF TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 70)
    print(
        "RFGN DQN DECISION LAYER SELF-TEST"
    )
    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        dqn = DQNDecisionLayer()

        print(
            "DQN model loading     : PASS"
        )

    except Exception as exc:

        print(
            "DQN model loading     : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Health check
    # --------------------------------------------------------

    health = dqn.health_check()

    if health[
        "status"
    ] != "healthy":

        print(
            "DQN health check      : FAIL"
        )

        return False

    print(
        "DQN health check      : PASS"
    )

    # --------------------------------------------------------
    # Action count
    # --------------------------------------------------------

    if health[
        "action_count"
    ] != EXPECTED_ACTION_COUNT:

        print(
            "Action count          : FAIL"
        )

        return False

    print(
        "Action count          : PASS"
    )

    # --------------------------------------------------------
    # Threshold mapping
    # --------------------------------------------------------

    expected_thresholds = np.array(
        [
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
        ],
        dtype=np.float32,
    )

    if not np.allclose(
        dqn.thresholds,
        expected_thresholds,
        atol=1e-6,
    ):

        print(
            "Threshold mapping    : FAIL"
        )

        return False

    print(
        "Threshold mapping    : PASS"
    )

    # --------------------------------------------------------
    # State creation
    # --------------------------------------------------------

    try:

        state = dqn.build_state(
            fraud_probability=0.82,
            threshold=DEFAULT_THRESHOLD,
            fraud_rate=0.035,
            predicted_fraud_rate=0.03,
            review_rate=0.01,
        )

        print(
            "DQN state creation   : PASS"
        )

    except Exception as exc:

        print(
            "DQN state creation   : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # State validation
    # --------------------------------------------------------

    if state.shape != (
        STATE_SIZE,
    ):

        print(
            "DQN state shape      : FAIL"
        )

        return False

    if not np.all(
        np.isfinite(state)
    ):

        print(
            "DQN state safety     : FAIL"
        )

        return False

    print(
        "DQN state validation : PASS"
    )

    # --------------------------------------------------------
    # Action selection
    # --------------------------------------------------------

    try:

        selected = dqn.select_action(
            state
        )

        print(
            "DQN action selection : PASS"
        )

    except Exception as exc:

        print(
            "DQN action selection : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    action = selected[
        "action"
    ]

    threshold = selected[
        "threshold"
    ]

    # --------------------------------------------------------
    # Action validation
    # --------------------------------------------------------

    if not (
        0
        <= action
        < EXPECTED_ACTION_COUNT
    ):

        print(
            "Action validation    : FAIL"
        )

        return False

    print(
        "Action validation    : PASS"
    )

    # --------------------------------------------------------
    # Threshold validation
    # --------------------------------------------------------

    if not (
        MIN_THRESHOLD
        <= threshold
        <= MAX_THRESHOLD
    ):

        print(
            "Threshold validation : FAIL"
        )

        return False

    print(
        "Threshold validation : PASS"
    )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    try:

        result = dqn.decide(
            fraud_probability=0.82,
            fraud_rate=0.035,
            predicted_fraud_rate=0.03,
            review_rate=0.01,
        )

        print(
            "Final decision       : PASS"
        )

    except Exception as exc:

        print(
            "Final decision       : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Decision validation
    # --------------------------------------------------------

    if result[
        "decision"
    ] not in (
        "FRAUD",
        "LEGITIMATE",
    ):

        print(
            "Decision validation  : FAIL"
        )

        return False

    if result[
        "predicted_class"
    ] not in (
        0,
        1,
    ):

        print(
            "Class validation     : FAIL"
        )

        return False

    print(
        "Decision validation  : PASS"
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()

    print(
        f"Test fraud probability : "
        f"{result['fraud_probability']:.6f}"
    )

    print(
        f"DQN selected action    : "
        f"{result['dqn_action']}"
    )

    print(
        f"DQN selected threshold : "
        f"{result['dqn_threshold']:.2f}"
    )

    print(
        f"Final decision         : "
        f"{result['decision']}"
    )

    print()

    print("=" * 70)
    print(
        "DQN DECISION LAYER: PASS"
    )
    print("=" * 70)

    return True


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    success = run_self_test()

    if not success:
        raise SystemExit(1)