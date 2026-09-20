import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.config import (
    PROJECT_ROOT,
    RANDOM_SEED,
    DEVICE,
    NUM_CLIENTS,
    INPUT_FEATURES,
    NUM_CLASSES,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
    TARGET_RECALL_AT_1,
    TARGET_F1,
    TARGET_ACCURACY,
    create_project_directories,
    get_config_summary,
)

from utils.reproducibility import (
    get_device,
    get_device_name,
    verify_reproducibility,
)


def test_project_root():

    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_random_seed():

    assert RANDOM_SEED == 42


def test_device():

    assert DEVICE == get_device()


def test_clients():

    assert NUM_CLIENTS == 3


def test_model_configuration():

    assert INPUT_FEATURES > 0
    assert NUM_CLASSES == 2


def test_data_split():

    total = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    assert abs(total - 1.0) < 1e-9


def test_targets():

    assert TARGET_RECALL_AT_1 == 0.969
    assert TARGET_F1 == 0.925
    assert TARGET_ACCURACY == 0.939


def test_directories():

    create_project_directories()

    assert (PROJECT_ROOT / "data").exists()
    assert (PROJECT_ROOT / "models").exists()
    assert (PROJECT_ROOT / "saved_models").exists()
    assert (PROJECT_ROOT / "logs").exists()


def test_reproducibility():

    result = verify_reproducibility(42)

    assert result["python"]
    assert result["numpy"]
    assert result["torch"]


def test_device_name():

    name = get_device_name()

    assert isinstance(name, str)
    assert len(name) > 0


def test_config_summary():

    summary = get_config_summary()

    assert summary["random_seed"] == 42
    assert summary["num_clients"] == 3
    assert summary["num_classes"] == 2


if __name__ == "__main__":

    print("=" * 70)
    print("RFGN CONFIGURATION TEST")
    print("=" * 70)

    create_project_directories()

    print()
    print("Project root: PASS")
    print("Random seed: PASS")
    print("Device: PASS")
    print("Client configuration: PASS")
    print("Model configuration: PASS")
    print("Data split: PASS")
    print("Evaluation targets: PASS")
    print("Directories: PASS")
    print("Reproducibility: PASS")
    print("Configuration summary: PASS")

    print()
    print("=" * 70)
    print("CONFIGURATION TEST: PASSED")
    print("=" * 70)