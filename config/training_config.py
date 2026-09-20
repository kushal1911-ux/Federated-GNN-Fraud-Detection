# ============================================================
# RFGN TRAINING CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# GraphSAGE
# ------------------------------------------------------------

INPUT_DIM = 814

HIDDEN_DIM = 128

OUTPUT_DIM = 2

DROPOUT = 0.30


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 100

EARLY_STOPPING_PATIENCE = 12

MIN_DELTA = 1e-4


# ------------------------------------------------------------
# Class imbalance
# ------------------------------------------------------------

NUM_CLASSES = 2

LEGITIMATE_CLASS = 0

FRAUD_CLASS = 1


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

RANDOM_SEED = 42


# ------------------------------------------------------------
# Model paths
# ------------------------------------------------------------

LOCAL_MODEL_DIR = (
    "saved_models/local"
)

CLIENT_1_MODEL_PATH = (
    "saved_models/local/client_1_graphsage.pt"
)

CLIENT_2_MODEL_PATH = (
    "saved_models/local/client_2_graphsage.pt"
)

CLIENT_3_MODEL_PATH = (
    "saved_models/local/client_3_graphsage.pt"
)


# ------------------------------------------------------------
# Training logs
# ------------------------------------------------------------

TRAINING_LOG_DIR = (
    "logs/training"
)


# ------------------------------------------------------------
# Validation / threshold configuration
# ------------------------------------------------------------

# Recall@1% means evaluating the highest-risk
# 1% of transactions according to model score.

RECALL_AT_PERCENT = 1.0


# ------------------------------------------------------------
# Configuration display
# ------------------------------------------------------------

def get_training_config():

    return {
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": OUTPUT_DIM,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,
        "min_delta": MIN_DELTA,
        "num_classes": NUM_CLASSES,
        "random_seed": RANDOM_SEED,
        "recall_at_percent":
            RECALL_AT_PERCENT,
    }


if __name__ == "__main__":

    print()
    print("=" * 70)
    print("RFGN TRAINING CONFIGURATION")
    print("=" * 70)

    config = get_training_config()

    for key, value in config.items():

        print(
            f"{key:<30}: {value}"
        )

    print()
    print("Training configuration: READY")
    print()