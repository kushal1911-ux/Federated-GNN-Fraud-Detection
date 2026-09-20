from pathlib import Path
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GRAPH_DATA_DIR = DATA_DIR / "graphs"

MODELS_DIR = PROJECT_ROOT / "models"
GNN_MODEL_DIR = MODELS_DIR / "gnn"
FEDERATED_MODEL_DIR = MODELS_DIR / "federated"
RL_MODEL_DIR = MODELS_DIR / "reinforcement"

TRAINING_DIR = PROJECT_ROOT / "training"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
TESTS_DIR = PROJECT_ROOT / "tests"

SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
LOG_DIR = PROJECT_ROOT / "logs"


# ============================================================
# DATASET
# ============================================================

DATASET_NAME = "IEEE-CIS"

# These are the expected processed client files.
CLIENT_DATASET_FILES = {
    1: PROCESSED_DATA_DIR / "ieee_dataset_1.csv",
    2: PROCESSED_DATA_DIR / "ieee_dataset_2.csv",
    3: PROCESSED_DATA_DIR / "ieee_dataset_3.csv",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42


# ============================================================
# COMPUTE DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_CUDA = torch.cuda.is_available()

GPU_NAME = (
    torch.cuda.get_device_name(0)
    if USE_CUDA
    else "CPU"
)


# ============================================================
# DATA SPLITS
# ============================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

assert abs(
    TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0
) < 1e-9


# ============================================================
# GRAPH CONFIGURATION
# ============================================================

INPUT_FEATURES = 431

NUM_CLASSES = 2

NUM_CLIENTS = 3


# ============================================================
# GRAPH NEURAL NETWORK
# ============================================================

GNN_HIDDEN_FEATURES = 128

GNN_OUTPUT_FEATURES = NUM_CLASSES

GNN_LEARNING_RATE = 0.001

GNN_WEIGHT_DECAY = 1e-4

GNN_EPOCHS = 50


# ============================================================
# FEDERATED LEARNING
# ============================================================

FEDERATED_ROUNDS = 10

FEDERATED_LOCAL_EPOCHS = 5

FEDERATED_LEARNING_RATE = GNN_LEARNING_RATE


# ============================================================
# REINFORCEMENT LEARNING
# ============================================================

RL_STATE_SIZE = 8

RL_ACTION_SIZE = 3

RL_HIDDEN_SIZE = 128

RL_LEARNING_RATE = 0.0003

RL_GAMMA = 0.95

RL_INITIAL_EPSILON = 1.0

RL_MIN_EPSILON = 0.05

RL_EPSILON_DECAY = 0.995

RL_BATCH_SIZE = 64

RL_REPLAY_CAPACITY = 100000

RL_TARGET_UPDATE_FREQUENCY = 100


# ============================================================
# FRAUD DETECTION ACTIONS
# ============================================================

ACTION_LEGITIMATE = 0
ACTION_FRAUD = 1
ACTION_REVIEW = 2

ACTION_NAMES = {
    ACTION_LEGITIMATE: "LEGITIMATE",
    ACTION_FRAUD: "FRAUD",
    ACTION_REVIEW: "REVIEW",
}


# ============================================================
# EVALUATION TARGETS
# ============================================================

TARGET_RECALL_AT_1 = 0.969

TARGET_F1 = 0.925

TARGET_ACCURACY = 0.939


# ============================================================
# RANKING EVALUATION
# ============================================================

RECALL_AT_K_VALUES = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
]


# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = "INFO"

LOG_FILE = LOG_DIR / "rfgn.log"


# ============================================================
# MODEL FILES
# ============================================================

GLOBAL_GNN_MODEL_PATH = (
    SAVED_MODELS_DIR / "global_graphsage.pth"
)

DQN_MODEL_PATH = (
    SAVED_MODELS_DIR / "dqn_fraud_agent.pth"
)


# ============================================================
# DIRECTORY INITIALIZATION
# ============================================================

DIRECTORIES_TO_CREATE = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    GRAPH_DATA_DIR,
    GNN_MODEL_DIR,
    FEDERATED_MODEL_DIR,
    RL_MODEL_DIR,
    TRAINING_DIR,
    EVALUATION_DIR,
    TESTS_DIR,
    SAVED_MODELS_DIR,
    LOG_DIR,
]


def create_project_directories():
    """
    Create all required project directories.
    """

    for directory in DIRECTORIES_TO_CREATE:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

def get_config_summary():
    """
    Return the important configuration values.
    """

    return {
        "project_root": str(PROJECT_ROOT),
        "dataset": DATASET_NAME,
        "random_seed": RANDOM_SEED,
        "device": str(DEVICE),
        "gpu": GPU_NAME,
        "input_features": INPUT_FEATURES,
        "num_classes": NUM_CLASSES,
        "num_clients": NUM_CLIENTS,
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": TEST_RATIO,
        "gnn_hidden_features": GNN_HIDDEN_FEATURES,
        "gnn_learning_rate": GNN_LEARNING_RATE,
        "gnn_epochs": GNN_EPOCHS,
        "federated_rounds": FEDERATED_ROUNDS,
        "rl_state_size": RL_STATE_SIZE,
        "rl_action_size": RL_ACTION_SIZE,
        "rl_gamma": RL_GAMMA,
        "target_recall_at_1": TARGET_RECALL_AT_1,
        "target_f1": TARGET_F1,
        "target_accuracy": TARGET_ACCURACY,
    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RFGN CONFIGURATION")
    print("=" * 70)

    create_project_directories()

    summary = get_config_summary()

    for key, value in summary.items():
        print(f"{key:25}: {value}")

    print("=" * 70)
    print("CONFIGURATION: PASS")
    print("=" * 70)