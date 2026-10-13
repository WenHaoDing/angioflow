"""Scan a checkpoints directory for eval/summary.json files and print a comparison table.

Usage:
    python -m angioflow.training.compare_checkpoints [--root DIR] [--sort COLUMN]
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/media/yaplab2/wd8tb/wenhao/datasets/angioflow_checkpoints")
    parser.add_argument("--sort", default="angioflow_mean_vel_rel_l2")
    args = parser.parse_args()

    rows = []
    for summary_path in sorted(Path(args.root).glob("*/eval/summary.json")):
        run_name = summary_path.parent.parent.name
        d = json.load(open(summary_path))
        row = {"run": run_name, "epoch": d.get("epoch"), "n_cases": len(d.get("cases", []))}
        row.update({k: v for k, v in d.items() if k not in ("cases", "checkpoint", "epoch")})
        rows.append(row)

    if not rows:
        print(f"No eval/summary.json found under {args.root}")
        return

    df = pd.DataFrame(rows)
    if args.sort in df.columns:
        df = df.sort_values(args.sort)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
