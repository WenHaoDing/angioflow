"""Shared master-CSV leaderboard logic for eval_wss.py / eval_joint.py.

Each eval script's run_summary.json (one row of metrics + hyperparameters per run,
already written at <ckpt_dir>/eval/run_summary.json) gets upserted into a single
master CSV at the checkpoint ROOT -- one file per script (they have different
metric/arg columns; kept as separate files rather than sheets since plain csv has
no sheet concept), so many runs accumulate into one spreadsheet. A run_name already
present gets its row REPLACED (never duplicated), and if a run's run_summary.json
already exists on disk, its numbers are reused as-is rather than re-running the
evaluation.
"""
from __future__ import annotations

import csv
import json
import os


def load_existing_run_summary(out_root):
    path = os.path.join(out_root, "run_summary.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def upsert_leaderboard(row, path):
    """Insert or update one run's row in the master CSV, keyed by run_name. Unions
    columns across runs (different models/architectures produce different arg_*
    keys; older rows get blanks for columns they don't have). Returns "added" or
    "updated"."""
    fieldnames, existing = [], []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            existing = list(reader)
    for k in row:
        if k not in fieldnames:
            fieldnames.append(k)

    status = "added"
    for i, r in enumerate(existing):
        if r.get("run_name") == row["run_name"]:
            existing[i] = row
            status = "updated"
            break
    if status == "added":
        existing.append(row)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in existing:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    return status
