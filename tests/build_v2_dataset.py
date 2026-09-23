"""Build the additive V2 chronologically split IEEE-CIS dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "v2" / "splits"
TX_FILE = RAW / "train_transaction.csv"
ID_FILE = RAW / "train_identity.csv"
ID, TIME, LABEL = "TransactionID", "TransactionDT", "isFraud"


def main() -> None:
    for path in (TX_FILE, ID_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)
    tx = pd.read_csv(TX_FILE, low_memory=False)
    identity = pd.read_csv(ID_FILE, low_memory=False)
    tx_rows, id_rows = len(tx), len(identity)
    if tx[ID].duplicated().any() or identity[ID].duplicated().any():
        raise ValueError("TransactionID must be unique in each source")
    if tx[[ID, TIME, LABEL]].isna().any().any():
        raise ValueError("Required key/time/label columns contain nulls")
    merged = tx.merge(identity, on=ID, how="left", validate="one_to_one", sort=False)
    if len(merged) != tx_rows:
        raise ValueError("Left merge changed transaction row count")
    # Normalize invalid numeric sentinels while preserving nulls for train-fitted imputation.
    numeric = merged.select_dtypes(include=[np.number]).columns
    merged[numeric] = merged[numeric].replace([np.inf, -np.inf], np.nan)
    merged = merged.sort_values([TIME, ID], kind="mergesort").reset_index(drop=True)
    n = len(merged)
    n_train, n_val = int(n * .70), int(n * .15)
    splits = {"train": merged.iloc[:n_train],
              "validation": merged.iloc[n_train:n_train + n_val],
              "test": merged.iloc[n_train + n_val:]}
    OUT.mkdir(parents=True, exist_ok=True)
    boundaries = {}
    fraud = {}
    for name, frame in splits.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)
        boundaries[name] = {"rows": len(frame), "TransactionDT_min": int(frame[TIME].min()),
                            "TransactionDT_max": int(frame[TIME].max()),
                            "TransactionID_min": int(frame[ID].min()), "TransactionID_max": int(frame[ID].max())}
        count = int(frame[LABEL].sum())
        fraud[name] = {"count": count, "rate": count / len(frame)}
    ids = [set(f[ID]) for f in splits.values()]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Chronological splits overlap on TransactionID")
    if not (splits["train"][TIME].max() <= splits["validation"][TIME].min()
            and splits["validation"][TIME].max() <= splits["test"][TIME].min()):
        raise ValueError("Chronological split order failed")
    manifest = {
        "version": "v2", "source_files": [str(TX_FILE.relative_to(ROOT)), str(ID_FILE.relative_to(ROOT))],
        "source_row_counts": {"train_transaction.csv": tx_rows, "train_identity.csv": id_rows},
        "merged_row_count": len(merged), "merged_column_count": len(merged.columns),
        "split_row_counts": {k: len(v) for k, v in splits.items()}, "fraud_counts_and_rates": fraud,
        "TransactionDT_boundaries": boundaries, "random_seeds": None,
        "preprocessing": ["left one-to-one merge on TransactionID", "numeric +/-inf converted to null",
                          "stable chronological sort by TransactionDT, TransactionID",
                          "contiguous 70/15/15 row split; no randomization", "raw features retained for train-fitted V2 processing"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
