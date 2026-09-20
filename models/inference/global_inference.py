"""
RFGN - Global GraphSAGE Inference

Loads the authoritative Flower/FedAvg global GraphSAGE model
and performs inference on a PyTorch Geometric graph.

This module is responsible ONLY for GraphSAGE inference.

It does NOT:
    - train the model
    - modify model weights
    - apply the DQN threshold
    - store database results

Input:
    PyG graph
    x shape = [N, 769]

Output:
    GraphSAGE logits
    Legitimate probability
    Fraud probability
    Predicted class
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# GLOBAL MODEL PATH
# ============================================================

GLOBAL_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "global"
    / "global_graphsage_flower.pt"
)


# ============================================================
# EXPECTED MODEL CONFIGURATION
# ============================================================

EXPECTED_INPUT_FEATURES = 769
EXPECTED_HIDDEN_FEATURES = 128
EXPECTED_OUTPUT_CLASSES = 2
EXPECTED_DROPOUT = 0.3
EXPECTED_PARAMETER_COUNT = 230146


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# IMPORT GRAPH SAGE
# ============================================================

try:
    from models.gnn.graphsage_model import GraphSAGE
except ImportError as exc:
    raise ImportError(
        "Unable to import GraphSAGE from "
        "models.gnn.graphsage_model"
    ) from exc


# ============================================================
# GLOBAL GRAPHSAGE INFERENCE
# ============================================================

class GlobalGraphSAGEInference:
    """
    Loads the trained global GraphSAGE model and performs
    inference on real-time graph data.
    """

    def __init__(
        self,
        model_path: Path = GLOBAL_MODEL_PATH,
        device: torch.device = DEVICE,
    ) -> None:

        self.model_path = Path(model_path)
        self.device = device

        self.model = self._load_model()

        self._validate_model()

        self.model.eval()

    # ========================================================
    # CREATE EXACT RFGN GRAPHSAGE
    # ========================================================

    def _create_model(self) -> torch.nn.Module:
        """
        Create the exact GraphSAGE architecture used by RFGN.

        IMPORTANT:
        The existing GraphSAGE class uses positional arguments,
        so do not use in_channels/hidden_channels/out_channels
        keyword arguments here.
        """

        model = GraphSAGE(
            EXPECTED_INPUT_FEATURES,
            EXPECTED_HIDDEN_FEATURES,
            EXPECTED_OUTPUT_CLASSES,
            EXPECTED_DROPOUT,
        )

        return model

    # ========================================================
    # LOAD GLOBAL MODEL
    # ========================================================

    def _load_model(self) -> torch.nn.Module:

        if not self.model_path.exists():

            raise FileNotFoundError(
                "Global GraphSAGE model not found:\n"
                f"{self.model_path}"
            )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        # ----------------------------------------------------
        # Case 1: complete PyTorch model
        # ----------------------------------------------------

        if isinstance(
            checkpoint,
            torch.nn.Module,
        ):

            model = checkpoint.to(
                self.device
            )

            return model

        # ----------------------------------------------------
        # Checkpoint must otherwise be a dictionary
        # ----------------------------------------------------

        if not isinstance(
            checkpoint,
            dict,
        ):

            raise ValueError(
                "Unsupported global model checkpoint format."
            )

        # ----------------------------------------------------
        # Locate state dictionary
        # ----------------------------------------------------

        state_dict = None

        for key in (
            "state_dict",
            "model_state_dict",
            "parameters",
        ):

            if key in checkpoint:

                candidate = checkpoint[key]

                if isinstance(
                    candidate,
                    dict,
                ):

                    state_dict = candidate
                    break

        # ----------------------------------------------------
        # Direct state_dict checkpoint
        # ----------------------------------------------------

        if state_dict is None:

            if all(
                isinstance(
                    value,
                    torch.Tensor,
                )
                for value in checkpoint.values()
            ):

                state_dict = checkpoint

        if state_dict is None:

            raise ValueError(
                "Could not find GraphSAGE state_dict "
                "inside global checkpoint."
            )

        # ----------------------------------------------------
        # Create exact architecture
        # ----------------------------------------------------

        model = self._create_model()

        # ----------------------------------------------------
        # Remove DataParallel prefix if present
        # ----------------------------------------------------

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("module."):

                key = key[
                    len("module.") :
                ]

            cleaned_state_dict[key] = value

        # ----------------------------------------------------
        # Load parameters
        # ----------------------------------------------------

        missing_keys, unexpected_keys = (
            model.load_state_dict(
                cleaned_state_dict,
                strict=False,
            )
        )

        if missing_keys:

            raise ValueError(
                "Missing GraphSAGE parameters:\n"
                + "\n".join(
                    str(key)
                    for key in missing_keys
                )
            )

        if unexpected_keys:

            raise ValueError(
                "Unexpected GraphSAGE parameters:\n"
                + "\n".join(
                    str(key)
                    for key in unexpected_keys
                )
            )

        model = model.to(
            self.device
        )

        return model

    # ========================================================
    # MODEL VALIDATION
    # ========================================================

    def _validate_model(self) -> None:

        # ----------------------------------------------------
        # Parameter count
        # ----------------------------------------------------

        parameter_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

        if parameter_count != EXPECTED_PARAMETER_COUNT:

            raise ValueError(
                "Unexpected GraphSAGE parameter count.\n"
                f"Expected: {EXPECTED_PARAMETER_COUNT}\n"
                f"Actual:   {parameter_count}"
            )

        # ----------------------------------------------------
        # Parameter numerical safety
        # ----------------------------------------------------

        for name, parameter in (
            self.model.named_parameters()
        ):

            if torch.isnan(
                parameter
            ).any():

                raise ValueError(
                    f"NaN detected in model parameter: {name}"
                )

            if torch.isinf(
                parameter
            ).any():

                raise ValueError(
                    "Infinity detected in model parameter: "
                    f"{name}"
                )

    # ========================================================
    # FORWARD
    # ========================================================

    @torch.no_grad()
    def forward(
        self,
        graph: Any,
    ) -> torch.Tensor:
        """
        Run GraphSAGE forward inference.

        Returns:
            Tensor with shape [N, 2]
        """

        if graph is None:

            raise ValueError(
                "Graph cannot be None."
            )

        if not hasattr(
            graph,
            "x",
        ):

            raise ValueError(
                "Graph does not contain x."
            )

        if not hasattr(
            graph,
            "edge_index",
        ):

            raise ValueError(
                "Graph does not contain edge_index."
            )

        x = graph.x.to(
            self.device
        )

        edge_index = graph.edge_index.to(
            self.device
        )

        # ----------------------------------------------------
        # Validate x
        # ----------------------------------------------------

        if x.ndim != 2:

            raise ValueError(
                "Graph x must have shape [N, F]."
            )

        if x.shape[1] != EXPECTED_INPUT_FEATURES:

            raise ValueError(
                "Graph feature dimension mismatch.\n"
                f"Expected: {EXPECTED_INPUT_FEATURES}\n"
                f"Actual:   {x.shape[1]}"
            )

        # ----------------------------------------------------
        # Validate edge_index
        # ----------------------------------------------------

        if edge_index.ndim != 2:

            raise ValueError(
                "edge_index must have shape [2, E]."
            )

        if edge_index.shape[0] != 2:

            raise ValueError(
                "edge_index must have shape [2, E]."
            )

        # ----------------------------------------------------
        # Numerical safety
        # ----------------------------------------------------

        if torch.isnan(x).any():

            raise ValueError(
                "NaN detected in graph features."
            )

        if torch.isinf(x).any():

            raise ValueError(
                "Infinity detected in graph features."
            )

        # ----------------------------------------------------
        # GraphSAGE inference
        # ----------------------------------------------------

        logits = self.model(
            x,
            edge_index,
        )

        # ----------------------------------------------------
        # Validate output
        # ----------------------------------------------------

        if not isinstance(
            logits,
            torch.Tensor,
        ):

            raise ValueError(
                "GraphSAGE did not return a tensor."
            )

        if logits.ndim != 2:

            raise ValueError(
                "GraphSAGE output must have shape [N, 2]."
            )

        if logits.shape[0] != x.shape[0]:

            raise ValueError(
                "Output node count does not match input."
            )

        if logits.shape[1] != EXPECTED_OUTPUT_CLASSES:

            raise ValueError(
                "GraphSAGE output class count mismatch.\n"
                f"Expected: {EXPECTED_OUTPUT_CLASSES}\n"
                f"Actual:   {logits.shape[1]}"
            )

        if torch.isnan(logits).any():

            raise ValueError(
                "NaN detected in GraphSAGE logits."
            )

        if torch.isinf(logits).any():

            raise ValueError(
                "Infinity detected in GraphSAGE logits."
            )

        return logits

    # ========================================================
    # PREDICT
    # ========================================================

    @torch.no_grad()
    def predict(
        self,
        graph: Any,
    ) -> Dict[str, Any]:
        """
        Run GraphSAGE and convert logits into probabilities.

        Class convention:
            0 = Legitimate
            1 = Fraud
        """

        logits = self.forward(
            graph
        )

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        legitimate_probability = (
            probabilities[:, 0]
        )

        fraud_probability = (
            probabilities[:, 1]
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1,
        )

        # ----------------------------------------------------
        # Probability validation
        # ----------------------------------------------------

        if torch.isnan(
            probabilities
        ).any():

            raise ValueError(
                "NaN detected in probabilities."
            )

        if torch.isinf(
            probabilities
        ).any():

            raise ValueError(
                "Infinity detected in probabilities."
            )

        if torch.any(
            probabilities < 0
        ):

            raise ValueError(
                "Negative probability detected."
            )

        if torch.any(
            probabilities > 1
        ):

            raise ValueError(
                "Probability greater than 1 detected."
            )

        probability_sum = probabilities.sum(
            dim=1
        )

        if not torch.allclose(
            probability_sum,
            torch.ones_like(
                probability_sum
            ),
            atol=1e-5,
        ):

            raise ValueError(
                "Class probabilities do not sum to 1."
            )

        return {
            "logits":
                logits.cpu(),

            "probabilities":
                probabilities.cpu(),

            "legitimate_probability":
                legitimate_probability.cpu(),

            "fraud_probability":
                fraud_probability.cpu(),

            "predicted_class":
                predicted_class.cpu(),
        }

    # ========================================================
    # SINGLE TRANSACTION
    # ========================================================

    @torch.no_grad()
    def predict_single(
        self,
        graph: Any,
    ) -> Dict[str, Any]:
        """
        Run inference for a graph containing one node.
        """

        if graph.x.shape[0] != 1:

            raise ValueError(
                "predict_single() requires exactly one node."
            )

        result = self.predict(
            graph
        )

        fraud_probability = float(
            result[
                "fraud_probability"
            ][0].item()
        )

        legitimate_probability = float(
            result[
                "legitimate_probability"
            ][0].item()
        )

        predicted_class = int(
            result[
                "predicted_class"
            ][0].item()
        )

        logits = [
            float(value)
            for value in result[
                "logits"
            ][0].tolist()
        ]

        probabilities = [
            float(value)
            for value in result[
                "probabilities"
            ][0].tolist()
        ]

        return {
            "fraud_probability":
                fraud_probability,

            "legitimate_probability":
                legitimate_probability,

            "predicted_class":
                predicted_class,

            "logits":
                logits,

            "probabilities":
                probabilities,
        }

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:

        parameter_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

        return {
            "module":
                "GlobalGraphSAGEInference",

            "status":
                "healthy",

            "model_path":
                str(
                    self.model_path
                ),

            "device":
                str(
                    self.device
                ),

            "input_features":
                EXPECTED_INPUT_FEATURES,

            "hidden_features":
                EXPECTED_HIDDEN_FEATURES,

            "output_classes":
                EXPECTED_OUTPUT_CLASSES,

            "dropout":
                EXPECTED_DROPOUT,

            "parameter_count":
                parameter_count,

            "eval_mode":
                not self.model.training,
        }


# ============================================================
# SELF TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 70)
    print(
        "RFGN GLOBAL GRAPHSAGE INFERENCE SELF-TEST"
    )
    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Model loading
    # --------------------------------------------------------

    try:

        inference = GlobalGraphSAGEInference()

        print(
            "Global model loading  : PASS"
        )

    except Exception as exc:

        print(
            "Global model loading  : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Health check
    # --------------------------------------------------------

    health = inference.health_check()

    if health[
        "status"
    ] != "healthy":

        print(
            "Model health          : FAIL"
        )

        return False

    print(
        "Model health          : PASS"
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    if health[
        "parameter_count"
    ] != EXPECTED_PARAMETER_COUNT:

        print(
            "Parameter count       : FAIL"
        )

        print(
            f"Expected: {EXPECTED_PARAMETER_COUNT}"
        )

        print(
            f"Actual:   {health['parameter_count']}"
        )

        return False

    print(
        "Parameter count       : PASS"
    )

    # --------------------------------------------------------
    # PyG test graph
    # --------------------------------------------------------

    try:

        from torch_geometric.data import Data

        x = torch.randn(
            (
                1,
                EXPECTED_INPUT_FEATURES,
            ),
            dtype=torch.float32,
        )

        edge_index = torch.empty(
            (
                2,
                0,
            ),
            dtype=torch.long,
        )

        graph = Data(
            x=x,
            edge_index=edge_index,
        )

        print(
            "Test graph creation  : PASS"
        )

    except Exception as exc:

        print(
            "Test graph creation  : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    try:

        logits = inference.forward(
            graph
        )

        print(
            "Forward pass          : PASS"
        )

    except Exception as exc:

        print(
            "Forward pass          : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Output shape
    # --------------------------------------------------------

    if logits.shape != (
        1,
        EXPECTED_OUTPUT_CLASSES,
    ):

        print(
            "Logit shape           : FAIL"
        )

        print(
            f"Actual shape: {tuple(logits.shape)}"
        )

        return False

    print(
        "Logit shape           : PASS"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        result = inference.predict_single(
            graph
        )

        print(
            "Probability inference : PASS"
        )

    except Exception as exc:

        print(
            "Probability inference : FAIL"
        )

        print(
            f"Error: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Probability range
    # --------------------------------------------------------

    fraud_probability = result[
        "fraud_probability"
    ]

    legitimate_probability = result[
        "legitimate_probability"
    ]

    if not (
        0.0
        <= fraud_probability
        <= 1.0
    ):

        print(
            "Probability range     : FAIL"
        )

        return False

    if not (
        0.0
        <= legitimate_probability
        <= 1.0
    ):

        print(
            "Probability range     : FAIL"
        )

        return False

    print(
        "Probability range     : PASS"
    )

    # --------------------------------------------------------
    # Probability sum
    # --------------------------------------------------------

    probability_sum = (
        fraud_probability
        + legitimate_probability
    )

    if abs(
        probability_sum - 1.0
    ) > 1e-5:

        print(
            "Probability sum       : FAIL"
        )

        print(
            f"Sum: {probability_sum}"
        )

        return False

    print(
        "Probability sum       : PASS"
    )

    # --------------------------------------------------------
    # Predicted class
    # --------------------------------------------------------

    predicted_class = result[
        "predicted_class"
    ]

    if predicted_class not in (
        0,
        1,
    ):

        print(
            "Predicted class       : FAIL"
        )

        return False

    print(
        "Predicted class       : PASS"
    )

    # --------------------------------------------------------
    # NaN / Infinity safety
    # --------------------------------------------------------

    if any(
        not torch.isfinite(
            torch.tensor(
                value,
                dtype=torch.float32,
            )
        ).item()
        for value in result[
            "probabilities"
        ]
    ):

        print(
            "NaN / Infinity safety : FAIL"
        )

        return False

    print(
        "NaN / Infinity safety : PASS"
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print()

    print(
        f"Fraud probability     : "
        f"{fraud_probability:.6f}"
    )

    print(
        f"Legitimate probability: "
        f"{legitimate_probability:.6f}"
    )

    print(
        f"Predicted class       : "
        f"{predicted_class}"
    )

    print()

    print("=" * 70)
    print(
        "GLOBAL GRAPHSAGE INFERENCE: PASS"
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