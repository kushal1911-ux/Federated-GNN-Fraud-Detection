# RFGN

Reinforced Federated Graph Network for Fraud Detection.

## Objective

Build a privacy-aware fraud detection system using:

- Graph Neural Networks
- Federated Learning
- Reinforcement Learning

## Primary Evaluation Targets

- Recall@1% > 96.9%
- F1-score > 92.5%
- Overall Accuracy > 93.9%

All reported metrics must be obtained from actual evaluation results.

## Project Structure

- config/ — configuration
- data/ — datasets and graph data
- models/ — GNN, federated learning and reinforcement learning
- training/ — training pipelines
- evaluation/ — evaluation scripts
- tests/ — validation tests
- utils/ — utility functions
- logs/ — training and evaluation logs
- saved_models/ — trained model checkpoints

## V2 chronological pipeline

V2 uses the local IEEE-CIS `train_transaction.csv` and `train_identity.csv` files. `TransactionID` joins the sources; the merged data is sorted by `TransactionDT, TransactionID` and divided contiguously into 70% TRAIN, 15% VALIDATION, and 15% TEST. The three deterministic non-IID clients are assigned from `ProductCD` strata (W/H, C/R, and S); labels do not determine membership.

Each client graph uses bounded entity-neighbor edges and up to five prior temporal neighbors within one hour. `isFraud`, `TransactionID`, and `TransactionDT` are excluded from node features; labels are retained separately. Numeric feature selection, categorical vocabularies, common feature order, and z-score normalization are fitted from TRAIN only. The resulting feature dimension is recorded in `data/processed/v2/feature_manifest.json`.

Reproduction commands (run from the repository root with pandas/numpy installed):

```powershell
python tests/build_v2_dataset.py
python tests/build_v2_clients.py
python tests/build_v2_client_graphs.py
python tests/align_v2_training_graphs.py
python tests/validate_v2_pipeline.py
python tests/build_v2_heldout_test_graph.py
python training/train_v2_federated_graphsage.py --rounds 5 --local-epochs 1
python tests/evaluate_v2_heldout_test.py
```

Federated aggregation is sample-count-weighted FedAvg over three client GraphSAGE updates; model selection uses VALIDATION PR-AUC. Held-out TEST is evaluated only with the frozen selected model. Recall@1% ranks TEST transactions by fraud probability, selects `ceil(0.01 * N)` transactions, and divides captured fraud by all TEST fraud. No V2 performance result is available until the training and held-out evaluation commands complete successfully.

Limitations: the supplied training CSV is the only labeled source, the temporal holdout may reflect distribution shift, ProductCD partition sizes are intentionally unequal, graph edges use selected observed entity fields and a fixed one-hour window, and this pipeline does not establish real-world deployment performance or privacy guarantees by itself. Training requires PyTorch and PyTorch Geometric; Flower is not required because the trainer performs weighted FedAvg directly.
