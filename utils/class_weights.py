import torch


# ============================================================
# RFGN CLASS WEIGHT CALCULATION
# ============================================================

def calculate_class_weights(
    labels,
    num_classes=2,
):
    """
    Calculate balanced class weights.

    Weight for each class:

        total_samples
        ----------------------------
        num_classes * class_samples

    This gives the minority fraud class
    greater influence during training.
    """

    if not isinstance(
        labels,
        torch.Tensor
    ):
        labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    labels = labels.long()

    if labels.numel() == 0:
        raise ValueError(
            "Labels tensor is empty."
        )

    class_counts = torch.bincount(
        labels,
        minlength=num_classes
    ).float()

    if torch.any(
        class_counts == 0
    ):
        raise ValueError(
            "At least one class has zero samples."
        )

    total_samples = (
        class_counts.sum()
    )

    weights = (
        total_samples
        / (
            num_classes
            * class_counts
        )
    )

    return weights


def print_class_distribution(
    labels,
):
    """
    Display class counts and percentages.
    """

    if not isinstance(
        labels,
        torch.Tensor
    ):
        labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    total = labels.numel()

    class_0 = int(
        (labels == 0).sum().item()
    )

    class_1 = int(
        (labels == 1).sum().item()
    )

    print()
    print(
        "Class distribution:"
    )

    print(
        f"  Legitimate (0): "
        f"{class_0:,} "
        f"({class_0 / total * 100:.4f}%)"
    )

    print(
        f"  Fraud (1):      "
        f"{class_1:,} "
        f"({class_1 / total * 100:.4f}%)"
    )

    print(
        f"  Total:          "
        f"{total:,}"
    )


def create_weighted_loss(
    labels,
    device,
):
    """
    Create weighted CrossEntropyLoss.
    """

    weights = calculate_class_weights(
        labels
    )

    weights = weights.to(
        device
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=weights
    )

    return criterion, weights


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("RFGN CLASS WEIGHT TEST")
    print("=" * 70)

    # Example distribution similar to
    # the RFGN training dataset.

    labels = torch.tensor(
        [0] * 100
        + [1] * 10,
        dtype=torch.long
    )

    print_class_distribution(
        labels
    )

    weights = calculate_class_weights(
        labels
    )

    print()
    print(
        "Calculated class weights:"
    )

    print(
        f"  Legitimate (0): "
        f"{weights[0].item():.6f}"
    )

    print(
        f"  Fraud (1):      "
        f"{weights[1].item():.6f}"
    )

    assert (
        weights[1]
        > weights[0]
    )

    print()
    print(
        "Fraud class receives higher weight: PASS"
    )

    print()
    print(
        "CLASS WEIGHT TEST: PASSED"
    )

    print()