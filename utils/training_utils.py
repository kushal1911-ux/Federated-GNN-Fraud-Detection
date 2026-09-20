# ============================================================
# RFGN TRAINING UTILITIES
# ============================================================

import os
import torch


# ============================================================
# OPTIMIZER
# ============================================================

def create_optimizer(
    model,
    learning_rate=0.001,
    weight_decay=1e-4,
):
    """
    Create AdamW optimizer for GraphSAGE.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return optimizer


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

def create_scheduler(
    optimizer,
    patience=4,
    factor=0.5,
    min_lr=1e-6,
):
    """
    Reduce learning rate when validation loss
    stops improving.
    """

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )

    return scheduler


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:

    def __init__(
        self,
        patience=12,
        min_delta=1e-4,
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def step(
        self,
        validation_loss,
    ):

        validation_loss = float(
            validation_loss
        )

        # First validation result
        if self.best_loss is None:

            self.best_loss = validation_loss
            self.counter = 0

            return False

        # Improvement
        if validation_loss < (
            self.best_loss - self.min_delta
        ):

            self.best_loss = validation_loss
            self.counter = 0

        # No improvement
        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.should_stop = True

        return self.should_stop


# ============================================================
# CHECKPOINT SAVING
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    validation_loss,
    path,
    scheduler=None,
):
    """
    Save a complete training checkpoint.
    """

    directory = os.path.dirname(
        path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict":
            model.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict(),
        "validation_loss":
            float(validation_loss),
    }

    if scheduler is not None:

        checkpoint[
            "scheduler_state_dict"
        ] = scheduler.state_dict()

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# MODEL CHECKPOINT LOADING
# ============================================================

def load_checkpoint(
    model,
    optimizer,
    path,
    scheduler=None,
    device="cpu",
):
    """
    Load a previously saved checkpoint.
    """

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    if (
        scheduler is not None
        and "scheduler_state_dict"
        in checkpoint
    ):

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

    return checkpoint


# ============================================================
# CURRENT LEARNING RATE
# ============================================================

def get_learning_rate(
    optimizer,
):

    return optimizer.param_groups[0][
        "lr"
    ]


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("RFGN TRAINING UTILITIES TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Dummy model
    # --------------------------------------------------------

    model = torch.nn.Linear(
        10,
        2
    )

    print()
    print("Dummy model created: PASS")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = create_optimizer(
        model,
        learning_rate=0.001,
        weight_decay=1e-4,
    )

    print(
        "AdamW optimizer: PASS"
    )

    assert (
        get_learning_rate(
            optimizer
        )
        == 0.001
    )

    print(
        "Learning rate: PASS"
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = create_scheduler(
        optimizer,
        patience=4,
        factor=0.5,
        min_lr=1e-6,
    )

    print(
        "Learning-rate scheduler: PASS"
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        patience=3,
        min_delta=1e-4,
    )

    losses = [
        1.0,
        0.8,
        0.7,
    ]

    for loss in losses:

        stopped = early_stopping.step(
            loss
        )

        assert stopped is False

    print(
        "Early stopping: PASS"
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    test_path = os.path.join(
        "logs",
        "training",
        "test_checkpoint.pt",
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        validation_loss=0.7,
        path=test_path,
    )

    assert os.path.exists(
        test_path
    )

    print(
        "Checkpoint saving: PASS"
    )

    # --------------------------------------------------------
    # Checkpoint loading
    # --------------------------------------------------------

    model2 = torch.nn.Linear(
        10,
        2
    )

    optimizer2 = create_optimizer(
        model2,
        learning_rate=0.001,
        weight_decay=1e-4,
    )

    scheduler2 = create_scheduler(
        optimizer2
    )

    checkpoint = load_checkpoint(
        model=model2,
        optimizer=optimizer2,
        scheduler=scheduler2,
        path=test_path,
        device="cpu",
    )

    assert (
        checkpoint["epoch"]
        == 1
    )

    print(
        "Checkpoint loading: PASS"
    )

    # --------------------------------------------------------
    # Cleanup test checkpoint
    # --------------------------------------------------------

    if os.path.exists(
        test_path
    ):

        os.remove(
            test_path
        )

    print(
        "Checkpoint cleanup: PASS"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING UTILITIES TEST: PASSED")
    print("=" * 70)
    print()