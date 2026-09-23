"""Partition V2 TRAIN deterministically by ProductCD, without labels."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/processed/v2/splits/train.csv"
OUT = ROOT / "data/processed/v2/clients"
SEED = 42
# Intentional covariate skew: W/H, C/R, and S assigned to separate clients.
PRODUCT_CLIENT = {"W": 1, "H": 1, "C": 2, "R": 2, "S": 3}

def main() -> None:
    df = pd.read_csv(TRAIN, low_memory=False)
    if "ProductCD" not in df:
        raise ValueError("ProductCD is required for documented non-IID partition")
    missing = df.ProductCD.isna()
    assigned = df.ProductCD.map(PRODUCT_CLIENT)
    # Rare/unseen and missing values are deterministically balanced by ID hash.
    fallback = ((df.TransactionID.astype("uint64") * 11400714819323198485 + SEED) % 3 + 1).astype("int8")
    assigned = assigned.fillna(pd.Series(fallback, index=df.index)).astype("int8")
    assigned.loc[missing] = fallback.loc[missing]
    manifests = {}
    id_sets = []
    for cid in (1, 2, 3):
        part = df.loc[assigned == cid].copy()
        directory = OUT / f"client_{cid}"
        directory.mkdir(parents=True, exist_ok=True)
        part.to_csv(directory / "train.csv", index=False)
        id_sets.append(set(part.TransactionID))
        manifests[f"client_{cid}"] = {
            "row_count": len(part), "ProductCD_distribution": {str(k): int(v) for k, v in part.ProductCD.value_counts(dropna=False).items()},
            "fraud_count": int(part.isFraud.sum()), "fraud_rate": float(part.isFraud.mean())
        }
    allids = set(df.TransactionID)
    union = set().union(*id_sets)
    overlaps = sum(len(id_sets[i] & id_sets[j]) for i in range(3) for j in range(i+1, 3))
    if union != allids or overlaps or sum(map(len, id_sets)) != len(df):
        raise ValueError("Client assignment is not an exact disjoint partition of TRAIN")
    manifest = {"partition_method": "ProductCD strata: W/H->1, C/R->2, S->3; null/unmapped rows assigned by stable TransactionID multiplicative hash modulo 3",
                "seed": SEED, "label_used_for_membership": False, "train_row_count": len(df), "clients": manifests,
                "verification": {"all_train_ids_covered": True, "overlap_count": overlaps,
                                 "each_train_id_exactly_once": True, "client_count_sum": sum(map(len, id_sets))}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__": main()
