# V2 GraphSAGE training on Colab GPU

This guide runs the existing full-graph V2 experiment on a GPU. It does not use neighbor sampling: the trainer builds a normalized sparse CSR mean-adjacency and applies the existing GraphSAGE linear layers to sparse neighbor aggregation. This avoids the edge-by-feature message tensor while keeping the same client graphs, node features, labels, split, model parameters, and weighted FedAvg protocol.

## Runtime and dependencies

Use a Colab GPU runtime with **Python 3.12** and preferably at least **16 GB GPU memory**. Check the assigned GPU before installing:

```bash
nvidia-smi
python --version
```

Install the pinned CUDA 12.8 PyTorch wheel and remaining dependencies:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install torch-geometric==2.7.0 flwr==1.38.0 numpy==2.3.5 pandas==3.0.1
```

Confirm CUDA is visible to PyTorch:

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
```

The documented PyTorch 2.8 wheel matrix includes CUDA 12.8 builds; PyG documents the 2.8 series with CUDA 12.8. If the selected Colab image cannot run this wheel, use a PyTorch CUDA build supported by that runtime and its matching PyG-compatible installation. See the [PyTorch version installation guide](https://docs.pytorch.org/get-started/previous-versions/) and [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Files to transfer before training

Preserve this repository-relative layout in Colab. Transfer these files/directories:

- `training/train_v2_federated_graphsage.py`
- `tests/build_v2_client_graphs.py` (feature encoding and validation graph helper imported by the trainer)
- `models/gnn/graphsage_model.py`
- `data/processed/v2/feature_manifest.json`
- `data/processed/v2/splits/validation.csv`
- `data/processed/v2/graphs/client_1/graph.npz`
- `data/processed/v2/graphs/client_2/graph.npz`
- `data/processed/v2/graphs/client_3/graph.npz`

The trainer does not need TEST data. It uses only those three TRAIN client graphs and the VALIDATION split. Do not put TEST files in the training workspace.

## Train

From the repository root in Colab:

```bash
python training/train_v2_federated_graphsage.py --rounds 5 --local-epochs 1 --lr 0.0005 --weight-decay 0.0001 --grad-clip 1.0
```

The script prints each client training loss, per-round validation loss and metrics, aggregation completion, and round runtime. It selects the frozen global state by validation PR-AUC. Outputs are written to:

- `saved_models/v2/global_graphsage_v2.pt`
- `saved_models/v2/training_metrics.json`

Download these outputs after training. The `saved_models/` directory is git-ignored.

## Held-out TEST evaluation (after freezing the model)

Only after training and model selection are finished, transfer:

- `tests/evaluate_v2_heldout_test.py`
- `training/train_v2_federated_graphsage.py` (shared metric function)
- `models/gnn/graphsage_model.py`
- `data/processed/v2/graphs/heldout_test/graph.npz`
- `saved_models/v2/global_graphsage_v2.pt`

Then run from the repository root:

```bash
python tests/evaluate_v2_heldout_test.py
```

The held-out graph contains TEST rows only and already uses the TRAIN-fitted feature manifest and normalization. Evaluation writes `saved_models/v2/heldout_test_metrics.json`. Do not use TEST results to select training rounds or tune the model.
