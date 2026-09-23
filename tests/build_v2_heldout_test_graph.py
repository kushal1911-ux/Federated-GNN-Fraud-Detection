"""Build TEST-only graph using V2 TRAIN-fitted feature and scale artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_v2_client_graphs import encode, build_edges

ROOT=Path(__file__).resolve().parents[1]; V2=ROOT/"data/processed/v2"

def main() -> None:
    fm=json.loads((V2/"feature_manifest.json").read_text(encoding="utf-8"))
    spec={"numeric":fm["numeric_features"],"categorical_vocab":fm["categorical_vocab"],"feature_order":fm["feature_order"]}
    df=pd.read_csv(V2/"splits/test.csv",low_memory=False)
    x=encode(df,spec); mean=np.asarray(fm["mean"],dtype=np.float64); std=np.asarray(fm["std"],dtype=np.float64)
    if x.shape[1]!=fm["feature_dimension"]: raise ValueError("TEST feature shape differs from TRAIN manifest")
    x=((x-mean)/std).astype(np.float32); edges,edge_meta=build_edges(df)
    out=V2/"graphs/heldout_test"; out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"graph.npz",x=x,y=df.isFraud.to_numpy(dtype=np.int64),edge_index=edges,
        TransactionID=df.TransactionID.to_numpy(dtype=np.int64),TransactionDT=df.TransactionDT.to_numpy(dtype=np.int64))
    meta={"graph":"chronological_test_only","num_nodes":len(df),"num_features":x.shape[1],"feature_order":fm["feature_order"],
          "excluded_from_features":["isFraud","TransactionID","TransactionDT"],"label_used_for_edges":False,
          "label_used_for_features":False,"normalization_reused_from":"TRAIN feature manifest","fit_on_test":False,
          "TransactionDT_min":int(df.TransactionDT.min()),"TransactionDT_max":int(df.TransactionDT.max()),**edge_meta}
    (out/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(f"Built TEST-only graph: {len(df):,} rows, {x.shape[1]} features")

if __name__ == "__main__": main()
