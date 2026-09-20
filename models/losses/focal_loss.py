import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RFGN FOCAL LOSS
# ============================================================
#
# Purpose:
#   Improve learning on highly imbalanced fraud-detection data
#   by reducing the contribution of easy, correctly classified
#   examples and focusing training on difficult examples.
#
# This module is designed for binary/multi-class classification
# using raw logits.
#
# Input:
#   logits -> [N, C]
#   targets -> [N]
#
# Output:
#   scalar loss
# ============================================================


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for classification.

    Formula:

        FL(pt) = -alpha * (1 - pt)^gamma * log(pt)

    where:

        pt    = probability assigned to the true class
        alpha = class weighting factor
        gamma = focusing parameter

    Parameters
    ----------
    alpha : float or None
        Optional weighting factor.

        If None:
            no additional alpha weighting is applied.

        If a float is supplied:
            the factor is applied to the target class.

        If a list/tuple/tensor is supplied:
            each class receives its corresponding weight.

    gamma : float
        Focusing parameter.

        gamma = 0:
            equivalent to standard cross entropy.

        Higher gamma:
            puts more emphasis on difficult examples.

    reduction : str
        One of:
            "mean"
            "sum"
            "none"
    """

    def __init__(
        self,
        alpha=None,
        gamma=2.0,
        reduction="mean",
    ):

        super().__init__()

        if gamma < 0:

            raise ValueError(
                "gamma must be >= 0."
            )

        if reduction not in {
            "mean",
            "sum",
            "none",
        }:

            raise ValueError(
                "reduction must be "
                "'mean', 'sum', or 'none'."
            )

        self.gamma = float(
            gamma
        )

        self.reduction = reduction

        if alpha is None:

            self.alpha = None

        elif isinstance(
            alpha,
            (list, tuple)
        ):

            self.alpha = torch.tensor(
                alpha,
                dtype=torch.float32
            )

        elif isinstance(
            alpha,
            torch.Tensor
        ):

            self.alpha = alpha.float()

        else:

            self.alpha = float(
                alpha
            )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        logits,
        targets,
    ):

        if logits.ndim != 2:

            raise ValueError(
                "logits must have shape [N, C]."
            )

        if targets.ndim != 1:

            raise ValueError(
                "targets must have shape [N]."
            )

        if logits.shape[0] != targets.shape[0]:

            raise ValueError(
                "Number of logits and targets "
                "must match."
            )

        if logits.shape[0] == 0:

            raise ValueError(
                "Empty batch is not supported."
            )

        targets = targets.long()

        # ----------------------------------------------------
        # Cross entropy per sample
        # ----------------------------------------------------

        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none"
        )

        # ----------------------------------------------------
        # Probability of the true class
        # ----------------------------------------------------

        pt = torch.exp(
            -ce_loss
        )

        # ----------------------------------------------------
        # Focusing factor
        # ----------------------------------------------------

        focal_factor = (
            1.0 - pt
        ).pow(
            self.gamma
        )

        loss = (
            focal_factor
            * ce_loss
        )

        # ----------------------------------------------------
        # Alpha weighting
        # ----------------------------------------------------

        if self.alpha is not None:

            if isinstance(
                self.alpha,
                torch.Tensor
            ):

                alpha = self.alpha.to(
                    device=logits.device,
                    dtype=logits.dtype
                )

                if alpha.numel() != logits.shape[1]:

                    raise ValueError(
                        "Alpha tensor must contain "
                        "one weight per class."
                    )

                sample_alpha = (
                    alpha[targets]
                )

            else:

                sample_alpha = torch.full_like(
                    loss,
                    float(self.alpha)
                )

            loss = (
                sample_alpha
                * loss
            )

        # ----------------------------------------------------
        # Reduction
        # ----------------------------------------------------

        if self.reduction == "mean":

            return loss.mean()

        if self.reduction == "sum":

            return loss.sum()

        return loss


# ============================================================
# BINARY FOCAL LOSS
# ============================================================

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss using logits.

    This implementation is provided for binary classification
    experiments where the model produces a single logit.

    Input:
        logits  -> [N] or [N, 1]
        targets -> [N] or [N, 1]

    Parameters
    ----------
    alpha : float or None
        Weight applied to positive examples.

    gamma : float
        Focusing parameter.

    reduction : str
        "mean", "sum", or "none"
    """

    def __init__(
        self,
        alpha=None,
        gamma=2.0,
        reduction="mean",
    ):

        super().__init__()

        if gamma < 0:

            raise ValueError(
                "gamma must be >= 0."
            )

        if reduction not in {
            "mean",
            "sum",
            "none",
        }:

            raise ValueError(
                "reduction must be "
                "'mean', 'sum', or 'none'."
            )

        self.alpha = (
            None
            if alpha is None
            else float(alpha)
        )

        self.gamma = float(
            gamma
        )

        self.reduction = reduction

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        logits,
        targets,
    ):

        logits = logits.float()

        targets = targets.float()

        if logits.ndim == 2:

            if logits.shape[1] != 1:

                raise ValueError(
                    "BinaryFocalLoss expects "
                    "[N] or [N, 1] logits."
                )

            logits = logits.squeeze(
                dim=1
            )

        if targets.ndim == 2:

            if targets.shape[1] != 1:

                raise ValueError(
                    "BinaryFocalLoss expects "
                    "[N] or [N, 1] targets."
                )

            targets = targets.squeeze(
                dim=1
            )

        if logits.ndim != 1:

            raise ValueError(
                "logits must have shape [N] or [N, 1]."
            )

        if targets.ndim != 1:

            raise ValueError(
                "targets must have shape [N] or [N, 1]."
            )

        if logits.shape[0] != targets.shape[0]:

            raise ValueError(
                "Number of logits and targets "
                "must match."
            )

        if not torch.all(
            (targets == 0)
            | (targets == 1)
        ):

            raise ValueError(
                "Binary targets must contain "
                "only 0 and 1."
            )

        # ----------------------------------------------------
        # Binary cross entropy
        # ----------------------------------------------------

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        # ----------------------------------------------------
        # Probability assigned to true class
        # ----------------------------------------------------

        probabilities = torch.sigmoid(
            logits
        )

        pt = torch.where(
            targets == 1,
            probabilities,
            1.0 - probabilities
        )

        # ----------------------------------------------------
        # Focal factor
        # ----------------------------------------------------

        focal_factor = (
            1.0 - pt
        ).pow(
            self.gamma
        )

        loss = (
            focal_factor
            * bce
        )

        # ----------------------------------------------------
        # Positive-class weighting
        # ----------------------------------------------------

        if self.alpha is not None:

            alpha_factor = torch.where(
                targets == 1,
                torch.full_like(
                    targets,
                    self.alpha
                ),
                torch.full_like(
                    targets,
                    1.0 - self.alpha
                )
            )

            loss = (
                alpha_factor
                * loss
            )

        # ----------------------------------------------------
        # Reduction
        # ----------------------------------------------------

        if self.reduction == "mean":

            return loss.mean()

        if self.reduction == "sum":

            return loss.sum()

        return loss


# ============================================================
# SELF TEST
# ============================================================

def run_self_test():

    print()
    print("=" * 70)
    print("RFGN FOCAL LOSS SELF TEST")
    print("=" * 70)

    torch.manual_seed(42)

    # --------------------------------------------------------
    # Test 1: Multi-class input
    # --------------------------------------------------------

    logits = torch.tensor(
        [
            [3.0, 0.2],
            [0.1, 2.5],
            [1.5, 0.8],
            [0.2, 1.1],
            [4.0, 0.1],
            [0.4, 2.2],
        ],
        dtype=torch.float32
    )

    targets = torch.tensor(
        [
            0,
            1,
            0,
            1,
            0,
            1,
        ],
        dtype=torch.long
    )

    print()
    print("Multi-class Focal Loss")

    focal_loss = FocalLoss(
        alpha=[0.25, 0.75],
        gamma=2.0,
        reduction="mean"
    )

    loss = focal_loss(
        logits,
        targets
    )

    print(
        "Loss:",
        f"{loss.item():.8f}"
    )

    assert torch.isfinite(
        loss
    ).item()

    print(
        "Finite loss: PASS"
    )

    assert loss.item() >= 0

    print(
        "Non-negative loss: PASS"
    )

    # --------------------------------------------------------
    # Test 2: Per-sample loss
    # --------------------------------------------------------

    focal_none = FocalLoss(
        alpha=[0.25, 0.75],
        gamma=2.0,
        reduction="none"
    )

    individual_loss = focal_none(
        logits,
        targets
    )

    print()
    print(
        "Per-sample loss shape:",
        tuple(individual_loss.shape)
    )

    assert individual_loss.shape == (
        logits.shape[0],
    )

    print(
        "Per-sample output: PASS"
    )

    # --------------------------------------------------------
    # Test 3: Sum reduction
    # --------------------------------------------------------

    focal_sum = FocalLoss(
        alpha=[0.25, 0.75],
        gamma=2.0,
        reduction="sum"
    )

    sum_loss = focal_sum(
        logits,
        targets
    )

    print()
    print(
        "Sum loss:",
        f"{sum_loss.item():.8f}"
    )

    assert torch.isfinite(
        sum_loss
    ).item()

    print(
        "Sum reduction: PASS"
    )

    # --------------------------------------------------------
    # Test 4: Gamma zero
    # --------------------------------------------------------
    #
    # When gamma = 0:
    #
    # (1 - pt)^0 = 1
    #
    # Therefore focal loss becomes
    # alpha-weighted cross entropy.
    # --------------------------------------------------------

    focal_gamma_zero = FocalLoss(
        alpha=None,
        gamma=0.0,
        reduction="mean"
    )

    focal_zero_loss = focal_gamma_zero(
        logits,
        targets
    )

    ce_loss = F.cross_entropy(
        logits,
        targets
    )

    print()
    print(
        "Gamma=0 Focal Loss:",
        f"{focal_zero_loss.item():.8f}"
    )

    print(
        "Cross Entropy:",
        f"{ce_loss.item():.8f}"
    )

    assert torch.allclose(
        focal_zero_loss,
        ce_loss,
        atol=1e-6
    )

    print(
        "Gamma=0 equivalence: PASS"
    )

    # --------------------------------------------------------
    # Test 5: Binary Focal Loss
    # --------------------------------------------------------

    binary_logits = torch.tensor(
        [
            3.0,
            -2.0,
            0.5,
            -0.5,
            4.0,
            -3.0,
        ],
        dtype=torch.float32
    )

    binary_targets = torch.tensor(
        [
            1,
            0,
            1,
            0,
            1,
            0,
        ],
        dtype=torch.float32
    )

    binary_focal = BinaryFocalLoss(
        alpha=0.75,
        gamma=2.0,
        reduction="mean"
    )

    binary_loss = binary_focal(
        binary_logits,
        binary_targets
    )

    print()
    print(
        "Binary Focal Loss:",
        f"{binary_loss.item():.8f}"
    )

    assert torch.isfinite(
        binary_loss
    ).item()

    assert binary_loss.item() >= 0

    print(
        "Binary Focal Loss: PASS"
    )

    # --------------------------------------------------------
    # Test 6: Gradient propagation
    # --------------------------------------------------------

    gradient_logits = torch.tensor(
        [
            [1.0, -1.0],
            [-0.5, 0.8],
            [0.2, 0.4],
            [2.0, -0.2],
        ],
        dtype=torch.float32,
        requires_grad=True
    )

    gradient_targets = torch.tensor(
        [
            0,
            1,
            1,
            0,
        ],
        dtype=torch.long
    )

    gradient_loss = focal_loss(
        gradient_logits,
        gradient_targets
    )

    gradient_loss.backward()

    assert (
        gradient_logits.grad
        is not None
    )

    assert torch.isfinite(
        gradient_logits.grad
    ).all().item()

    print()
    print(
        "Gradient propagation: PASS"
    )

    # --------------------------------------------------------
    # Test 7: Invalid shapes
    # --------------------------------------------------------

    invalid_shape_detected = False

    try:

        focal_loss(
            torch.randn(5),
            targets
        )

    except ValueError:

        invalid_shape_detected = True

    assert invalid_shape_detected

    print(
        "Invalid shape validation: PASS"
    )

    # --------------------------------------------------------
    # Test 8: Invalid targets
    # --------------------------------------------------------

    binary_invalid_detected = False

    try:

        binary_focal(
            binary_logits,
            torch.tensor(
                [
                    0,
                    1,
                    2,
                    0,
                    1,
                    0,
                ],
                dtype=torch.float32
            )
        )

    except ValueError:

        binary_invalid_detected = True

    assert binary_invalid_detected

    print(
        "Invalid binary target validation: PASS"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RFGN FOCAL LOSS: PASS")
    print("=" * 70)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_self_test()