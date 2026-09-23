"""Fit shared feature order and scaling on TRAIN clients only."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; V2=ROOT/"data/processed/v2"; GRAPH=V2/"graphs"

def main() -> None:
    spec=json.loads((GRAPH/"feature_spec.json").read_text(encoding="utf-8"))
    arrays=[]; n_total=0; sums=None; squares=None
    id_hash=hashlib.sha256()
    for cid in (1,2,3):
        with np.load(GRAPH/f"client_{cid}/graph.npz") as g:
            x=g["x"].astype(np.float64); ids=g["TransactionID"]
            if x.shape[1]!=len(spec["feature_order"]): raise ValueError("Feature order/dimension mismatch")
            sums=x.sum(0) if sums is None else sums+x.sum(0)
            squares=(x*x).sum(0) if squares is None else squares+(x*x).sum(0)
            n_total+=len(x); id_hash.update(ids.tobytes()); arrays.append((cid,x))
    mean=sums/n_total; var=np.maximum(squares/n_total-mean*mean,0); std=np.sqrt(var); std[std<1e-12]=1.0
    for cid,x in arrays:
        with np.load(GRAPH/f"client_{cid}/graph.npz") as g:
            data={k:g[k] for k in g.files}
        data["x"]=((x-mean)/std).astype(np.float32)
        np.savez_compressed(GRAPH/f"client_{cid}/graph.npz",**data)
        meta=json.loads((GRAPH/f"client_{cid}/metadata.json").read_text(encoding="utf-8"))
        meta.update({"normalization":"z-score", "normalization_fitted_on":"TRAIN clients only", "is_normalized":True})
        (GRAPH/f"client_{cid}/metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    manifest={"version":"v2","feature_order":spec["feature_order"],"feature_dimension":len(spec["feature_order"]),
        "numeric_features":spec["numeric"],"categorical_vocab":spec["categorical_vocab"],"mean":mean.tolist(),"std":std.tolist(),
        "normalization":"z-score; population mean/std; zero/near-zero std replaced by 1",
        "fit_scope":"all V2 TRAIN rows across three clients only","train_row_count":n_total,
        "train_id_sha256_in_client_order":id_hash.hexdigest(),"excluded_columns":["isFraud","TransactionID","TransactionDT"],
        "feature_selection":"all numeric TRAIN columns except identifiers/time/label, plus fixed low-cardinality categorical one-hot vocab fitted on TRAIN"}
    (V2/"feature_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Aligned {n_total:,} train nodes at dimension {manifest['feature_dimension']}")

if __name__ == "__main__": main()
