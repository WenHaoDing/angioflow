"""Remove CFD-anomaly cases from whding123/AneuG-Flow's transient_data/ on the Hub.

Case list comes from angioflowv2_merged/_meta/cfd_anomalies.json -- the same
exclusion list extract_wss_dataset.py's discover() uses locally -- filtered to
--version (cfd_anomalies.json also carries AngioFlow rows, which don't apply to this repo).

    python -m angioflow.huggingface.remove_aneug_flow_anomalies                  # dry run
    python -m angioflow.huggingface.remove_aneug_flow_anomalies --execute --yes  # apply
"""
from __future__ import annotations

import argparse
import json
import os

from angioflow.data.dataset_tags import normalize_tag
from angioflow.huggingface.remove_cases_from_dataset import remove_cases_from_dataset

REPO_ID = "whding123/AneuG-Flow"
PREFIX = "transient_data"
ANOMALIES_PATH = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged/_meta/cfd_anomalies.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", default=REPO_ID)
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--anomalies", default=ANOMALIES_PATH)
    p.add_argument("--version", default="aneug_flow", help="only remove anomalies of this dataset")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--execute", action="store_true", help="actually delete (needs --yes too)")
    p.add_argument("--yes", action="store_true", help="confirm --execute; without it, --execute still only prints")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.anomalies) as f:
        rows = json.load(f)
    # cfd_anomalies.json predates the rename and still says "v1"/"v2"
    cases = sorted({r["cid"] for r in rows if normalize_tag(r["version"]) == args.version})
    print(f"[plan] {len(cases)} {args.version} case(s) flagged as anomalies in {args.anomalies}")

    remove_cases_from_dataset(repo_id=args.repo_id, prefix=args.prefix, cases=cases,
                              token=args.token, execute=args.execute, yes=args.yes)


if __name__ == "__main__":
    main()
