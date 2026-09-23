"""Required V2 leakage, split, partition, and feature compatibility checks."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; V2=ROOT/"data/processed/v2"

def main() -> None:
    splits={k:pd.read_csv(V2/f"splits/{k}.csv",usecols=["TransactionID","TransactionDT","isFraud"]) for k in ("train","validation","test")}
    idsets={k:set(v.TransactionID) for k,v in splits.items()}
    for a,b in (("train","validation"),("train","test"),("validation","test")):
        assert not idsets[a]&idsets[b], f"ID overlap: {a}/{b}"
    assert splits["train"].TransactionDT.max()<=splits["validation"].TransactionDT.min()
    assert splits["validation"].TransactionDT.max()<=splits["test"].TransactionDT.min()
    clientsets=[]; client_total=0
    for cid in (1,2,3):
        c=pd.read_csv(V2/f"clients/client_{cid}/train.csv",usecols=["TransactionID"])
        clientsets.append(set(c.TransactionID)); client_total+=len(c)
    assert all(not(clientsets[i]&clientsets[j]) for i in range(3) for j in range(i+1,3)), "Client ID overlap"
    assert set().union(*clientsets)==idsets["train"] and client_total==len(splits["train"]), "Client rows do not exactly cover TRAIN"
    fm=json.loads((V2/"feature_manifest.json").read_text(encoding="utf-8"))
    assert not ({"isFraud","TransactionID","TransactionDT"}&set(fm["feature_order"])), "Forbidden model feature present"
    assert fm["fit_scope"]=="all V2 TRAIN rows across three clients only"
    assert fm["train_row_count"]==len(splits["train"])
    feature_order=fm["feature_order"]; dims=set()
    for cid in (1,2,3):
        meta=json.loads((V2/f"graphs/client_{cid}/metadata.json").read_text(encoding="utf-8"))
        assert meta["label_used_for_edges"] is False and meta["label_used_for_features"] is False
        assert meta["excluded_from_features"]==["isFraud","TransactionID","TransactionDT"]
        assert meta["feature_order"]==feature_order, f"Feature order differs for client_{cid}"
        assert meta["normalization_fitted_on"]=="TRAIN clients only"
        with np.load(V2/f"graphs/client_{cid}/graph.npz") as g:
            dims.add(g["x"].shape[1]); assert g["x"].shape[1]==len(feature_order)
            assert len(g["y"])==len(g["TransactionID"])
    assert len(dims)==1 and next(iter(dims))==len(feature_order)
    # TEST is checked to be absent from training IDs and artifacts are explicitly tagged train-only.
    assert not (set().union(*clientsets)&idsets["test"]), "TEST IDs found in clients"
    assert fm["fit_scope"].endswith("TRAIN rows across three clients only")
    assert fm["feature_dimension"]==len(feature_order)
    print(json.dumps({"status":"PASSED","checks":10,"split_rows":{k:len(v) for k,v in splits.items()},
        "client_rows":client_total,"feature_dimension":len(feature_order)},indent=2))

if __name__ == "__main__": main()
