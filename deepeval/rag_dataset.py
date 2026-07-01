#!/usr/bin/env python3
"""RAG demo dataset for the scorers — data lives in rag_dataset.jsonl.

`rag_dataset.jsonl` is a standalone, directly-uploadable dataset (one JSON object
per line: input / expected / metadata / tags), so you can load it into Braintrust
with NO Python:

    bt datasets create "DeepEval RAG demo dataset" --file rag_dataset.jsonl -p "<Project>"

...or via the Braintrust UI's dataset import. `upload()` / `python rag_dataset.py`
does the same through the SDK. `RAG_ROWS` exposes the rows for the local Eval()
runner (the reference answer is `metadata.reference_output`, surfaced as `output`).

The rows are intentionally mixed — faithful, hallucinated, and noisy/missing
retrieval — so every metric produces a meaningful, non-trivial score.
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(_HERE, "rag_dataset.jsonl")


def _load_rows():
    rows = []
    with open(DATASET_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            md = rec.get("metadata", {})
            rows.append({
                "input": rec["input"],
                "expected": rec["expected"],
                "output": md.get("reference_output", ""),  # reference answer for local runs
                "metadata": md,
                "tags": rec.get("tags", []),
            })
    return rows


RAG_ROWS = _load_rows()


def upload(project_name="RAG-Scorers-Demo", dataset_name="DeepEval RAG demo dataset"):
    """Create/replace the Braintrust dataset from rag_dataset.jsonl; return record ids."""
    import braintrust

    org = os.environ.get("BRAINTRUST_ORG_NAME", "My Org")
    braintrust.login(org_name=org)
    ds = braintrust.init_dataset(project=project_name, name=dataset_name)
    existing = list(ds.fetch())
    for rec in existing:
        ds.delete(rec["id"])
    if existing:
        ds.flush()
    ids = []
    for row in RAG_ROWS:
        ids.append(ds.insert(
            input=row["input"],
            expected=row["expected"],
            metadata=row["metadata"],
            tags=row["tags"],
        ))
    ds.flush()
    print(f"Uploaded {len(RAG_ROWS)} rows to '{dataset_name}' (cleared {len(existing)} old rows).")
    return ids


if __name__ == "__main__":
    upload()
