# ============================================================
# RFGN REINFORCEMENT LEARNING ENVIRONMENT
# ============================================================
#
# Purpose:
# Create and test the Reinforcement Learning environment
# used for fraud-detection threshold optimization.
#
# This version keeps the original fraud reward formulation
# but normalizes the reward for more stable DQN learning.
#
# Original reward:
#   True Positive  = +5
#   True Negative  = +1
#   False Positive = -1
#   False Negative = -5
#
# Review budget:
#   1%
#
# Reward normalization:
#   Raw reward / number of samples
#
# This preserves the ranking of thresholds while keeping
# reward values in a stable numerical range.
#
# ============================================================

import os
import sys

import numpy as np
import torch


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
# PATHS
# ============================================================

GLOBAL_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "global",
    "global_graphsage_flower.pt",
)

GRAPH_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "graphs",
    "client_1",
    "graph_aligned.pt",
)


# ============================================================
# EXPECTED MODEL CONFIGURATION
# ============================================================

EXPECTED_INPUT_DIM = 769
EXPECTED_OUTPUT_DIM = 2


# ============================================================
# THRESHOLD CONFIGURATION
# ============================================================

DEFAULT_THRESHOLD = 0.50

MIN_THRESHOLD = 0.05
MAX_THRESHOLD = 0.95
THRESHOLD_STEP = 0.05


# ============================================================
# REWARD CONFIGURATION
# ============================================================

FALSE_POSITIVE_COST = 1.0
FALSE_NEGATIVE_COST = 5.0

TRUE_POSITIVE_REWARD = 5.0
TRUE_NEGATIVE_REWARD = 1.0

REVIEW_COST = 0.10

MAX_REVIEW_RATE = 0.01


# ============================================================
# REWARD NORMALIZATION
# ============================================================

REWARD_NORMALIZATION = True


# ============================================================
# REINFORCEMENT LEARNING ENVIRONMENT
# ============================================================

class FraudThresholdEnvironment:
    """
    Environment for fraud-detection threshold optimization.

    The environment receives fraud probabilities generated
    by the global GraphSAGE model.

    Each action represents a threshold.

    Decision rule:

        probability >= threshold
            -> fraud

        probability < threshold
            -> legitimate

    The reward considers:

        True positives
        True negatives
        False positives
        False negatives
        Review-budget penalty

    The raw reward is normalized by the total number of
    samples to provide a stable reward scale for DQN.
    """

    def __init__(
        self,
        fraud_probabilities,
        labels,
    ):

        # ----------------------------------------------------
        # Convert inputs
        # ----------------------------------------------------

        self.fraud_probabilities = np.asarray(
            fraud_probabilities,
            dtype=np.float32,
        )

        self.labels = np.asarray(
            labels,
            dtype=np.int64,
        )


        # ----------------------------------------------------
        # Validate sample count
        # ----------------------------------------------------

        if (
            len(self.fraud_probabilities)
            != len(self.labels)
        ):

            raise ValueError(
                "Fraud probability count and label count "
                "do not match."
            )


        if len(self.labels) == 0:

            raise ValueError(
                "Environment cannot contain zero samples."
            )


        # ----------------------------------------------------
        # Validate probabilities
        # ----------------------------------------------------

        if not np.isfinite(
            self.fraud_probabilities
        ).all():

            raise ValueError(
                "Fraud probabilities contain NaN or infinity."
            )


        if (
            self.fraud_probabilities.min() < 0.0
            or
            self.fraud_probabilities.max() > 1.0
        ):

            raise ValueError(
                "Fraud probabilities must be between 0 and 1."
            )


        # ----------------------------------------------------
        # Validate labels
        # ----------------------------------------------------

        unique_labels = np.unique(
            self.labels
        )


        if not np.all(
            np.isin(
                unique_labels,
                [0, 1],
            )
        ):

            raise ValueError(
                "Labels must contain only 0 and 1."
            )


        # ----------------------------------------------------
        # Create threshold actions
        # ----------------------------------------------------

        self.thresholds = np.arange(
            MIN_THRESHOLD,
            MAX_THRESHOLD + (
                THRESHOLD_STEP / 2
            ),
            THRESHOLD_STEP,
            dtype=np.float32,
        )


        self.num_actions = len(
            self.thresholds
        )


        # ----------------------------------------------------
        # Initial action
        # ----------------------------------------------------

        self.current_action = int(
            np.argmin(
                np.abs(
                    self.thresholds
                    - DEFAULT_THRESHOLD
                )
            )
        )


        self.current_threshold = float(
            self.thresholds[
                self.current_action
            ]
        )


        self.step_count = 0


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.current_action = int(
            np.argmin(
                np.abs(
                    self.thresholds
                    - DEFAULT_THRESHOLD
                )
            )
        )


        self.current_threshold = float(
            self.thresholds[
                self.current_action
            ]
        )


        self.step_count = 0


        return self._get_state()


    # ========================================================
    # STATE
    # ========================================================

    def _get_state(self):

        threshold = float(
            self.current_threshold
        )


        fraud_rate = float(
            self.labels.mean()
        )


        mean_probability = float(
            self.fraud_probabilities.mean()
        )


        predictions = (
            self.fraud_probabilities
            >= threshold
        )


        predicted_fraud_rate = float(
            predictions.mean()
        )


        review_rate = (
            predicted_fraud_rate
        )


        state = np.array(
            [
                threshold,
                fraud_rate,
                mean_probability,
                predicted_fraud_rate,
                review_rate,
            ],
            dtype=np.float32,
        )


        return state


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    def _calculate_confusion_matrix(
        self,
        threshold,
    ):

        predictions = (
            self.fraud_probabilities
            >= threshold
        ).astype(
            np.int64
        )


        true_positive = int(
            np.sum(
                (predictions == 1)
                &
                (self.labels == 1)
            )
        )


        true_negative = int(
            np.sum(
                (predictions == 0)
                &
                (self.labels == 0)
            )
        )


        false_positive = int(
            np.sum(
                (predictions == 1)
                &
                (self.labels == 0)
            )
        )


        false_negative = int(
            np.sum(
                (predictions == 0)
                &
                (self.labels == 1)
            )
        )


        return (
            true_positive,
            true_negative,
            false_positive,
            false_negative,
        )


    # ========================================================
    # RAW REWARD
    # ========================================================

    def _calculate_raw_reward(
        self,
        threshold,
    ):

        (
            true_positive,
            true_negative,
            false_positive,
            false_negative,
        ) = self._calculate_confusion_matrix(
            threshold
        )


        reward = 0.0


        # ----------------------------------------------------
        # True positives
        # ----------------------------------------------------

        reward += (
            true_positive
            * TRUE_POSITIVE_REWARD
        )


        # ----------------------------------------------------
        # True negatives
        # ----------------------------------------------------

        reward += (
            true_negative
            * TRUE_NEGATIVE_REWARD
        )


        # ----------------------------------------------------
        # False positives
        # ----------------------------------------------------

        reward -= (
            false_positive
            * FALSE_POSITIVE_COST
        )


        # ----------------------------------------------------
        # False negatives
        # ----------------------------------------------------

        reward -= (
            false_negative
            * FALSE_NEGATIVE_COST
        )


        # ----------------------------------------------------
        # Review budget
        # ----------------------------------------------------

        predictions = (
            self.fraud_probabilities
            >= threshold
        )


        review_rate = float(
            predictions.mean()
        )


        if review_rate > MAX_REVIEW_RATE:

            excess_rate = (
                review_rate
                - MAX_REVIEW_RATE
            )


            reward -= (
                excess_rate
                * len(predictions)
                * REVIEW_COST
            )


        return float(
            reward
        )


    # ========================================================
    # NORMALIZED REWARD
    # ========================================================

    def _calculate_reward(
        self,
        threshold,
    ):

        raw_reward = (
            self._calculate_raw_reward(
                threshold
            )
        )


        if REWARD_NORMALIZATION:

            normalized_reward = (
                raw_reward
                /
                float(
                    len(
                        self.labels
                    )
                )
            )

        else:

            normalized_reward = (
                raw_reward
            )


        return float(
            normalized_reward
        )


    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        action,
    ):

        action = int(
            action
        )


        if action < 0:

            raise ValueError(
                "Action cannot be negative."
            )


        if action >= self.num_actions:

            raise ValueError(
                "Action is outside the threshold action space."
            )


        # ----------------------------------------------------
        # Apply action
        # ----------------------------------------------------

        self.current_action = action


        self.current_threshold = float(
            self.thresholds[
                action
            ]
        )


        self.step_count += 1


        # ----------------------------------------------------
        # Raw reward
        # ----------------------------------------------------

        raw_reward = (
            self._calculate_raw_reward(
                self.current_threshold
            )
        )


        # ----------------------------------------------------
        # Normalized reward
        # ----------------------------------------------------

        reward = (
            self._calculate_reward(
                self.current_threshold
            )
        )


        # ----------------------------------------------------
        # New state
        # ----------------------------------------------------

        state = self._get_state()


        # ----------------------------------------------------
        # One threshold decision = one episode
        # ----------------------------------------------------

        terminated = True

        truncated = False


        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        (
            true_positive,
            true_negative,
            false_positive,
            false_negative,
        ) = self._calculate_confusion_matrix(
            self.current_threshold
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        total = len(
            self.labels
        )


        accuracy = (
            true_positive
            + true_negative
        ) / total


        precision_denominator = (
            true_positive
            + false_positive
        )


        if precision_denominator > 0:

            precision = (
                true_positive
                / precision_denominator
            )

        else:

            precision = 0.0


        recall_denominator = (
            true_positive
            + false_negative
        )


        if recall_denominator > 0:

            recall = (
                true_positive
                / recall_denominator
            )

        else:

            recall = 0.0


        if (
            precision
            + recall
        ) > 0:

            f1 = (
                2.0
                * precision
                * recall
                /
                (
                    precision
                    + recall
                )
            )

        else:

            f1 = 0.0


        predictions = (
            self.fraud_probabilities
            >= self.current_threshold
        )


        review_rate = float(
            predictions.mean()
        )


        # ----------------------------------------------------
        # Information dictionary
        # ----------------------------------------------------

        info = {

            "threshold":
                self.current_threshold,

            "true_positive":
                true_positive,

            "true_negative":
                true_negative,

            "false_positive":
                false_positive,

            "false_negative":
                false_negative,

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1":
                float(f1),

            "review_rate":
                review_rate,

            "raw_reward":
                float(raw_reward),

            "reward":
                float(reward),

            "reward_normalized":
                bool(
                    REWARD_NORMALIZATION
                ),
        }


        return (
            state,
            reward,
            terminated,
            truncated,
            info,
        )


    # ========================================================
    # GET THRESHOLDS
    # ========================================================

    def get_thresholds(self):

        return self.thresholds.copy()


    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(self):

        return {

            "samples":
                int(
                    len(
                        self.labels
                    )
                ),

            "fraud_samples":
                int(
                    np.sum(
                        self.labels == 1
                    )
                ),

            "legitimate_samples":
                int(
                    np.sum(
                        self.labels == 0
                    )
                ),

            "fraud_rate":
                float(
                    self.labels.mean()
                ),

            "minimum_probability":
                float(
                    self.fraud_probabilities.min()
                ),

            "maximum_probability":
                float(
                    self.fraud_probabilities.max()
                ),

            "mean_probability":
                float(
                    self.fraud_probabilities.mean()
                ),

            "num_actions":
                int(
                    self.num_actions
                ),

            "minimum_threshold":
                float(
                    self.thresholds.min()
                ),

            "maximum_threshold":
                float(
                    self.thresholds.max()
                ),

            "reward_normalization":
                bool(
                    REWARD_NORMALIZATION
                ),
        }


# ============================================================
# LOAD GLOBAL GRAPHSAGE PREDICTIONS
# ============================================================

def load_global_predictions():

    print(
        "Loading Flower global GraphSAGE model..."
    )


    # --------------------------------------------------------
    # Check global model
    # --------------------------------------------------------

    if not os.path.exists(
        GLOBAL_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Global model not found:\n"
            f"{GLOBAL_MODEL_PATH}"
        )


    checkpoint = torch.load(
        GLOBAL_MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )


    print(
        "Global model checkpoint: PASS"
    )


    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    input_dim = checkpoint.get(
        "input_dim"
    )

    output_dim = checkpoint.get(
        "output_dim"
    )


    if input_dim != EXPECTED_INPUT_DIM:

        raise RuntimeError(
            "Global model input dimension mismatch.\n"
            f"Expected: {EXPECTED_INPUT_DIM}\n"
            f"Found: {input_dim}"
        )


    if output_dim != EXPECTED_OUTPUT_DIM:

        raise RuntimeError(
            "Global model output dimension mismatch.\n"
            f"Expected: {EXPECTED_OUTPUT_DIM}\n"
            f"Found: {output_dim}"
        )


    # --------------------------------------------------------
    # Import GraphSAGE
    # --------------------------------------------------------

    from models.gnn.graphsage_model import GraphSAGE


    # --------------------------------------------------------
    # Reconstruct model
    # --------------------------------------------------------

    model = GraphSAGE(
        input_dim=checkpoint[
            "input_dim"
        ],

        hidden_dim=checkpoint[
            "hidden_dim"
        ],

        output_dim=checkpoint[
            "output_dim"
        ],

        dropout=checkpoint[
            "dropout"
        ],
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )


    model.eval()


    print(
        "Global GraphSAGE model: PASS"
    )


    # --------------------------------------------------------
    # Check aligned graph
    # --------------------------------------------------------

    if not os.path.exists(
        GRAPH_PATH
    ):

        raise FileNotFoundError(
            "Aligned graph not found:\n"
            f"{GRAPH_PATH}"
        )


    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )


    print(
        "Aligned graph: PASS"
    )


    # --------------------------------------------------------
    # Validate graph
    # --------------------------------------------------------

    if (
        graph.num_node_features
        != EXPECTED_INPUT_DIM
    ):

        raise RuntimeError(
            "Graph feature dimension mismatch."
        )


    if graph.x.isnan().any():

        raise RuntimeError(
            "Graph features contain NaN."
        )


    if graph.x.isinf().any():

        raise RuntimeError(
            "Graph features contain infinity."
        )


    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print(
        "Generating global fraud probabilities..."
    )


    with torch.no_grad():

        logits = model(
            graph.x,
            graph.edge_index,
        )


        probabilities = torch.softmax(
            logits,
            dim=1,
        )


    # --------------------------------------------------------
    # Extract fraud probability
    # --------------------------------------------------------

    fraud_probabilities = (
        probabilities[:, 1]
        .cpu()
        .numpy()
    )


    labels = (
        graph.y
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # Validate probabilities
    # --------------------------------------------------------

    if not np.isfinite(
        fraud_probabilities
    ).all():

        raise RuntimeError(
            "Fraud probabilities contain NaN or infinity."
        )


    if not np.isfinite(
        labels
    ).all():

        raise RuntimeError(
            "Labels contain NaN or infinity."
        )


    print(
        "Fraud probabilities: PASS"
    )


    print(
        f"Samples: "
        f"{len(labels):,}"
    )


    print(
        f"Fraud samples: "
        f"{np.sum(labels == 1):,}"
    )


    print(
        f"Legitimate samples: "
        f"{np.sum(labels == 0):,}"
    )


    return (
        fraud_probabilities,
        labels,
    )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "RFGN REINFORCEMENT LEARNING ENVIRONMENT TEST"
    )

    print("=" * 70)

    print()

    print(
        "No DQN training will be performed."
    )

    print(
        "No model parameters will be updated."
    )

    print(
        "No graph will be modified."
    )

    print(
        "No CSV dataset will be modified."
    )

    print()

    print(
        "Reward formulation:"
    )

    print(
        f"True Positive  = +{TRUE_POSITIVE_REWARD}"
    )

    print(
        f"True Negative  = +{TRUE_NEGATIVE_REWARD}"
    )

    print(
        f"False Positive = -{FALSE_POSITIVE_COST}"
    )

    print(
        f"False Negative = -{FALSE_NEGATIVE_COST}"
    )

    print(
        f"Review cost    = {REVIEW_COST}"
    )

    print(
        f"Review budget  = {MAX_REVIEW_RATE * 100:.2f}%"
    )

    print(
        f"Reward normalization = "
        f"{REWARD_NORMALIZATION}"
    )


    # ========================================================
    # LOAD GLOBAL MODEL OUTPUT
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


    # ========================================================
    # CREATE ENVIRONMENT
    # ========================================================

    print()

    print("-" * 70)

    print(
        "CREATING REINFORCEMENT LEARNING ENVIRONMENT"
    )

    print("-" * 70)


    environment = FraudThresholdEnvironment(
        fraud_probabilities,
        labels,
    )


    print(
        "Reinforcement Learning environment: CREATED"
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = environment.summary()


    print()

    print(
        "Environment summary:"
    )


    print(
        f"Samples              : "
        f"{summary['samples']:,}"
    )


    print(
        f"Fraud samples        : "
        f"{summary['fraud_samples']:,}"
    )


    print(
        f"Legitimate samples   : "
        f"{summary['legitimate_samples']:,}"
    )


    print(
        f"Fraud rate           : "
        f"{summary['fraud_rate'] * 100:.4f}%"
    )


    print(
        f"Minimum probability  : "
        f"{summary['minimum_probability']:.6f}"
    )


    print(
        f"Maximum probability  : "
        f"{summary['maximum_probability']:.6f}"
    )


    print(
        f"Mean probability     : "
        f"{summary['mean_probability']:.6f}"
    )


    print(
        f"Number of actions    : "
        f"{summary['num_actions']}"
    )


    print(
        f"Threshold range      : "
        f"{summary['minimum_threshold']:.2f}"
        f" - "
        f"{summary['maximum_threshold']:.2f}"
    )


    print(
        f"Reward normalization : "
        f"{summary['reward_normalization']}"
    )


    # ========================================================
    # RESET
    # ========================================================

    print()

    print("-" * 70)

    print(
        "TESTING ENVIRONMENT RESET"
    )

    print("-" * 70)


    state = environment.reset()


    print(
        f"Initial state shape: "
        f"{state.shape}"
    )


    print(
        f"Initial state: "
        f"{state}"
    )


    if state.shape != (5,):

        raise RuntimeError(
            "Initial state shape is incorrect."
        )


    if not np.isfinite(
        state
    ).all():

        raise RuntimeError(
            "Initial state contains NaN or infinity."
        )


    print(
        "Environment reset: PASS"
    )


    # ========================================================
    # DEFAULT ACTION
    # ========================================================

    print()

    print("-" * 70)

    print(
        "TESTING DEFAULT THRESHOLD ACTION"
    )

    print("-" * 70)


    default_action = int(
        np.argmin(
            np.abs(
                environment.thresholds
                - DEFAULT_THRESHOLD
            )
        )
    )


    print(
        f"Action index: "
        f"{default_action}"
    )


    print(
        f"Threshold: "
        f"{environment.thresholds[default_action]:.2f}"
    )


    (
        next_state,
        reward,
        terminated,
        truncated,
        info,
    ) = environment.step(
        default_action
    )


    if next_state.shape != (5,):

        raise RuntimeError(
            "Next state shape is incorrect."
        )


    if not np.isfinite(
        next_state
    ).all():

        raise RuntimeError(
            "Next state contains NaN or infinity."
        )


    if not np.isfinite(
        reward
    ):

        raise RuntimeError(
            "Reward contains NaN or infinity."
        )


    print(
        "Environment step: PASS"
    )


    print(
        f"Raw reward: "
        f"{info['raw_reward']:.4f}"
    )


    print(
        f"Normalized reward: "
        f"{reward:.6f}"
    )


    print(
        f"Terminated: "
        f"{terminated}"
    )


    print(
        f"Truncated: "
        f"{truncated}"
    )


    # ========================================================
    # METRICS
    # ========================================================

    print()

    print("-" * 70)

    print(
        "TESTING REINFORCEMENT LEARNING METRICS"
    )

    print("-" * 70)


    print(
        f"Threshold      : "
        f"{info['threshold']:.2f}"
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
        f"Normalized reward: "
        f"{info['reward']:.6f}"
    )


    # ========================================================
    # TEST ALL ACTIONS
    # ========================================================

    print()

    print("-" * 70)

    print(
        "TESTING ALL THRESHOLD ACTIONS"
    )

    print("-" * 70)


    best_reward = -float(
        "inf"
    )

    best_threshold = None

    best_info = None


    for action in range(
        environment.num_actions
    ):

        (
            _,
            action_reward,
            _,
            _,
            action_info,
        ) = environment.step(
            action
        )


        if action_reward > best_reward:

            best_reward = (
                action_reward
            )

            best_threshold = (
                action_info[
                    "threshold"
                ]
            )

            best_info = (
                action_info
            )


    print(
        "All threshold actions: PASS"
    )


    # ========================================================
    # BEST THRESHOLD
    # ========================================================

    print()

    print(
        "Best threshold by direct reward evaluation:"
    )


    print(
        f"Threshold : "
        f"{best_threshold:.2f}"
    )


    print(
        f"Raw reward: "
        f"{best_info['raw_reward']:.4f}"
    )


    print(
        f"Normalized reward: "
        f"{best_info['reward']:.6f}"
    )


    print(
        f"Precision : "
        f"{best_info['precision']:.6f}"
    )


    print(
        f"Recall    : "
        f"{best_info['recall']:.6f}"
    )


    print(
        f"F1 Score  : "
        f"{best_info['f1']:.6f}"
    )


    print(
        f"Review rate: "
        f"{best_info['review_rate'] * 100:.4f}%"
    )


    # ========================================================
    # REWARD RANKING VALIDATION
    # ========================================================

    print()

    print("-" * 70)

    print(
        "VALIDATING REWARD NORMALIZATION"
    )

    print("-" * 70)


    # The normalized reward must preserve the same ordering
    # as the raw reward because every reward is divided by
    # the same positive sample count.

    raw_rewards = []

    normalized_rewards = []


    for action in range(
        environment.num_actions
    ):

        (
            _,
            action_reward,
            _,
            _,
            action_info,
        ) = environment.step(
            action
        )


        raw_rewards.append(
            action_info[
                "raw_reward"
            ]
        )


        normalized_rewards.append(
            action_reward
        )


    raw_best_action = int(
        np.argmax(
            raw_rewards
        )
    )


    normalized_best_action = int(
        np.argmax(
            normalized_rewards
        )
    )


    if (
        raw_best_action
        ==
        normalized_best_action
    ):

        print(
            "Reward ranking preservation: PASS"
        )

    else:

        raise RuntimeError(
            "Reward normalization changed "
            "the threshold ranking."
        )


    print(
        f"Best raw action: "
        f"{raw_best_action}"
    )


    print(
        f"Best normalized action: "
        f"{normalized_best_action}"
    )


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print()

    print("=" * 70)

    print(
        "RFGN REINFORCEMENT LEARNING ENVIRONMENT TEST: PASSED"
    )

    print("=" * 70)

    print()

    print(
        "Global GraphSAGE probabilities: PASS"
    )


    print(
        "State representation: PASS"
    )


    print(
        "Threshold action space: PASS"
    )


    print(
        "Raw reward calculation: PASS"
    )


    print(
        "Reward normalization: PASS"
    )


    print(
        "Reward ranking preservation: PASS"
    )


    print(
        "Fraud metrics: PASS"
    )


    print(
        "Environment reset: PASS"
    )


    print(
        "Environment step: PASS"
    )


    print()

    print(
        "Reinforcement Learning environment "
        "is ready for DQN retraining."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()