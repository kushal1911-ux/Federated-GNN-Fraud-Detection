"""Evaluate a frozen V2 global model on TEST only; never fit artifacts here."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; V2=ROOT/"data/processed/v2"; MODEL=ROOT/"saved_models/v2/global_graphsage_v2.pt"
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"training"))
from train_v2_federated_graphsage import ranking_metrics

def main() -> None:
    if not MODEL.is_file(): raise FileNotFoundError(f"Frozen V2 model missing: run training after installing torch and torch-geometric: {MODEL}")
    try:
        import torch
        from models.gnn.graphsage_model import GraphSAGE
    except ImportError as e: raise SystemExit(f"Evaluation dependency unavailable: {e}")
    graph_path=V2/"graphs/heldout_test/graph.npz"
    if not graph_path.is_file(): raise FileNotFoundError(f"Build held-out graph first: {graph_path}")
    ckpt=torch.load(MODEL,map_location="cpu",weights_only=False); model=GraphSAGE(input_dim=ckpt["input_dim"],hidden_dim=128,output_dim=2,dropout=.30); model.load_state_dict(ckpt["state_dict"]); model.eval()
    with np.load(graph_path) as g: x=torch.tensor(g["x"],dtype=torch.float32); edges=torch.tensor(g["edge_index"],dtype=torch.long); y=g["y"].astype(np.int64)
    with torch.no_grad(): p=torch.softmax(model(x,edges),dim=1)[:,1].numpy()
    metrics=ranking_metrics(y,p); result={"test_transaction_count":len(y),**metrics,
        "recall_at_1_percent_protocol":"descending fraud probability; top_k=ceil(0.01*N); captured fraud / total fraud",
        "model":"frozen global_graphsage_v2.pt","feature_manifest":"TRAIN-fitted feature_manifest.json"}
    out=ROOT/"saved_models/v2"; (out/"heldout_test_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
