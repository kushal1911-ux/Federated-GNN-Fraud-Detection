"""Build bounded entity and temporal graphs for V2 TRAIN clients."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data/processed/v2"
CAT_CANDIDATES = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType"]
RELATIONSHIPS = ["card1", "card2", "card5", "addr1", "P_emaildomain", "R_emaildomain", "DeviceInfo"]
MAX_VALUE_GROUP = 5000
NEIGHBORS = 5
TIME_WINDOW = 3600

def feature_spec(train: pd.DataFrame) -> dict:
    excluded = {"TransactionID", "TransactionDT", "isFraud"}
    numeric = [c for c in train.select_dtypes(include=[np.number]).columns if c not in excluded]
    cats = [c for c in CAT_CANDIDATES if c in train.columns]
    vocab = {}
    for c in cats:
        vocab[c] = sorted(train[c].dropna().astype(str).unique().tolist())
    return {"numeric": numeric, "categorical_vocab": vocab,
            "feature_order": numeric + [f"{c}=={v}" for c in cats for v in vocab[c]]}

def encode(df: pd.DataFrame, spec: dict) -> np.ndarray:
    blocks = []
    for c in spec["numeric"]:
        blocks.append(pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)[:, None])
    for c, values in spec["categorical_vocab"].items():
        s = df[c].astype("string")
        vals = np.asarray(values, dtype=object)
        a = np.zeros((len(df), len(values)), dtype=np.float32)
        lookup = {v:i for i,v in enumerate(values)}
        for row, value in enumerate(s):
            idx = lookup.get(str(value)) if pd.notna(value) else None
            if idx is not None: a[row, idx] = 1.0
        blocks.append(a)
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(df), 0), dtype=np.float32)

def build_edges(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    n = len(df); src=[]; dst=[]; counts={"entity":0,"temporal":0}
    for col in RELATIONSHIPS:
        if col not in df: continue
        groups = df.loc[df[col].notna(), [col, "TransactionDT"]].copy()
        groups["node"] = np.flatnonzero(df[col].notna().to_numpy())
        groups["key"] = groups[col].astype(str)
        groups.sort_values(["key", "TransactionDT", "node"], kind="mergesort", inplace=True)
        for _, g in groups.groupby("key", sort=False):
            nodes=g.node.to_numpy(dtype=np.int64)
            if len(nodes)>MAX_VALUE_GROUP: continue
            for j in range(1,len(nodes)):
                for k in range(max(0,j-NEIGHBORS),j): src.extend((int(nodes[j]),int(nodes[k]))); dst.extend((int(nodes[k]),int(nodes[j])))
        counts["entity"] += len(src)
    order=np.lexsort((df.TransactionID.to_numpy(),df.TransactionDT.to_numpy()))
    times=df.TransactionDT.to_numpy()
    for pos,node in enumerate(order):
        k=pos-1
        linked=0
        while k>=0 and linked<NEIGHBORS and times[node]-times[order[k]]<=TIME_WINDOW:
            other=int(order[k]); src.extend((int(node),other)); dst.extend((other,int(node))); linked+=1; k-=1
    counts["temporal"]=len(src)-counts["entity"]
    return np.asarray([src,dst],dtype=np.int64), {"edge_counts_directed":counts,"relationship_columns":RELATIONSHIPS,
        "relationship_group_cap":MAX_VALUE_GROUP,"neighbors_per_entity":NEIGHBORS,"temporal_window_seconds":TIME_WINDOW}

def save_graph(df: pd.DataFrame, out: Path, spec: dict, graph_name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    x=encode(df,spec); edges,edge_meta=build_edges(df); y=df.isFraud.to_numpy(dtype=np.int64)
    np.savez_compressed(out/"graph.npz", x=x, y=y, edge_index=edges,
        TransactionID=df.TransactionID.to_numpy(dtype=np.int64), TransactionDT=df.TransactionDT.to_numpy(dtype=np.int64))
    meta={"graph":graph_name,"num_nodes":len(df),"num_features":x.shape[1],"edge_index_shape":list(edges.shape),
          "feature_order":spec["feature_order"],"excluded_from_features":["isFraud","TransactionID","TransactionDT"],
          "label_used_for_edges":False,"label_used_for_features":False,"feature_spec_fit_on":"TRAIN only",**edge_meta}
    (out/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")

def main() -> None:
    train=pd.read_csv(V2/"splits/train.csv",low_memory=False); spec=feature_spec(train)
    graphroot=V2/"graphs"; graphroot.mkdir(parents=True,exist_ok=True)
    (graphroot/"feature_spec.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")
    for cid in (1,2,3):
        df=pd.read_csv(V2/f"clients/client_{cid}/train.csv",low_memory=False)
        save_graph(df,graphroot/f"client_{cid}",spec,f"client_{cid}_train")
        print(f"client_{cid}: {len(df):,} nodes, {spec and len(spec['feature_order'])} features")

if __name__ == "__main__": main()
