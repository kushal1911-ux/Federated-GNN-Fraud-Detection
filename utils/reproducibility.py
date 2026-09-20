import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible experiments.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic CUDA behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Return CUDA device when available, otherwise CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_device_name() -> str:
    """
    Return the active compute device name.
    """

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)

    return "CPU"


def verify_reproducibility(seed: int = 42) -> dict:
    """
    Verify deterministic random number generation.
    """

    set_seed(seed)

    python_value_1 = random.random()
    numpy_value_1 = np.random.rand()
    torch_value_1 = torch.rand(1).item()

    set_seed(seed)

    python_value_2 = random.random()
    numpy_value_2 = np.random.rand()
    torch_value_2 = torch.rand(1).item()

    return {
        "python": python_value_1 == python_value_2,
        "numpy": numpy_value_1 == numpy_value_2,
        "torch": torch_value_1 == torch_value_2,
    }


if __name__ == "__main__":

    print("=" * 70)
    print("RFGN REPRODUCIBILITY VERIFICATION")
    print("=" * 70)

    seed = 42

    set_seed(seed)

    print(f"Random seed: {seed}")

    print(f"Device: {get_device()}")

    print(f"Device name: {get_device_name()}")

    verification = verify_reproducibility(seed)

    print()
    print("Python reproducibility:", "PASS" if verification["python"] else "FAIL")
    print("NumPy reproducibility:", "PASS" if verification["numpy"] else "FAIL")
    print("PyTorch reproducibility:", "PASS" if verification["torch"] else "FAIL")

    all_passed = all(verification.values())

    print()

    if all_passed:
        print("REPRODUCIBILITY: PASS")
    else:
        print("REPRODUCIBILITY: FAIL")
        raise RuntimeError(
            "Reproducibility verification failed."
        )

    print("=" * 70)