"""Recover the FULL original name of every AneuX case folder in angioflowv2_merged,
using ONLY the AneuX geometry folder -- no GHD fits, no assembled/AneuX, no other
script's output.

    /media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/AneuX_stable

holds all 323 AneuX cases under their original names. A short id in angioflowv2_merged
is matched to whichever AneuX_stable folder name simplify_name() (angioflow/data/
rename_geometry_folder.py) maps down to it. Collisions -- several full names
simplifying to the same short id -- are reported and never guessed at.

VERIFICATION IS SELF-CONTAINED, computed here from scratch, not borrowed from any
other tool's output. AneuX_stable/<full_name>/merged_reconstruction_remeshed.obj is
raw geometry in MILLIMETRES, already in the SAME FRAME as the CFD wall
(angioflowv2_merged/<case>/wall_data.pt's `pos`, in METRES) -- no rotation, no
translation, no GHD reconstruction involved (checked: C0002, median distance to the
CFD wall 0.025 mm with a straight metre->millimetre conversion). So for every
candidate rename, this script loads both, measures the one-directional nearest-
neighbour distance (geometry -> CFD wall, since the geometry file can be a slightly
larger patch), and accepts the match only if it actually overlaps:

    median_mm <= --accept_median_mm (default 0.30)
    coverage  >= --accept_coverage  (default 0.90, fraction of geometry points
                                      within --tol_mm of the CFD wall)

A rename that fails this is flagged NOT VERIFIED and reported, never renamed
silently as if it were confirmed.

Dry run by default. --execute --yes also updates angioflowv2_merged/_meta/
case_registry.xlsx's "case" column (AneuX rows only, backed up first) and any
matching files in angioflowv2_merged/_sanity/.

    python -m angioflow.data.retrieve_aneux_full_name                  # dry run
    python -m angioflow.data.retrieve_aneux_full_name --execute --yes  # do it
"""
from __future__ import annotations

import argparse
import collections
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.join(REPO, "angioflow")):
    if p not in sys.path:
        sys.path.insert(0, p)

from angioflow.data.rename_geometry_folder import simplify_name  # noqa: E402

MERGED_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged"
GEOMETRY_ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/AneuX_stable"
REGISTRY_PATH = os.path.join(MERGED_ROOT, "_meta", "case_registry.xlsx")
GEOMETRY_OBJ = "merged_reconstruction_remeshed.obj"
M_TO_MM = 1000.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merged_root", default=MERGED_ROOT)
    p.add_argument("--geometry_root", default=GEOMETRY_ROOT)
    p.add_argument("--registry", default=REGISTRY_PATH)
    p.add_argument("--tol_mm", type=float, default=0.5)
    p.add_argument("--accept_median_mm", type=float, default=0.30)
    p.add_argument("--accept_coverage", type=float, default=0.90)
    p.add_argument("--no_verify", action="store_true",
                   help="skip the geometry check, match by name only (fast, not recommended)")
    p.add_argument("--execute", action="store_true", help="actually rename (needs --yes too)")
    p.add_argument("--yes", action="store_true", help="confirm --execute; without it, --execute still only prints")
    return p.parse_args()


def full_name_candidates(geometry_root):
    """simplified name -> [full_name, ...] from the AneuX geometry folder."""
    cands = collections.defaultdict(list)
    for n in sorted(os.listdir(geometry_root)):
        if os.path.isdir(os.path.join(geometry_root, n)):
            cands[simplify_name(n)].append(n)
    return cands


def load_cfd_wall_mm(case_dir):
    import torch
    w = torch.load(os.path.join(case_dir, "wall_data.pt"), map_location="cpu", weights_only=False)
    wall = w["wall"] if "wall" in w else w
    return np.asarray(wall["pos"].numpy(), np.float64) * M_TO_MM


def load_geometry_mm(geometry_root, full_name):
    import trimesh
    obj = os.path.join(geometry_root, full_name, GEOMETRY_OBJ)
    if not os.path.isfile(obj):
        return None
    return np.asarray(trimesh.load(obj, process=False).vertices, np.float64)


def verify(geometry_root, full_name, case_dir, tol_mm):
    """One-directional: does the AneuX_stable geometry actually sit on THIS case's
    own CFD wall? Returns None if either file is missing."""
    from scipy.spatial import cKDTree
    pts = load_geometry_mm(geometry_root, full_name)
    if pts is None or not os.path.isfile(os.path.join(case_dir, "wall_data.pt")):
        return None
    wall = load_cfd_wall_mm(case_dir)
    d, _ = cKDTree(wall).query(pts)
    return {"median_mm": float(np.median(d)), "p90_mm": float(np.percentile(d, 90)),
            "max_mm": float(d.max()), "coverage": float((d <= tol_mm).mean())}


def plan(args):
    """AneuX-only: cases are selected via the CFD registry's dataset column."""
    import pandas as pd

    reg = pd.read_excel(args.registry)
    reg["case"] = reg["case"].astype(str)
    aneux_cases = sorted(set(reg.loc[reg["dataset"] == "AneuX", "case"]) &
                         {d for d in os.listdir(args.merged_root)
                          if os.path.isdir(os.path.join(args.merged_root, d))})

    cands = full_name_candidates(args.geometry_root)
    all_full_names = {n for v in cands.values() for n in v}

    def _fmt(cand, m):
        if m is None:
            return f"{cand}: no data"
        return f"{cand}: median {m['median_mm']:.3f} mm, coverage {100 * m['coverage']:.0f}%"

    def resolve_by_geometry(name, candidates, note):
        """Run the geometry check on every candidate and accept the one that
        actually sits on this case's CFD wall. Returns a row dict, either "ok"
        (exactly one candidate passed) or "ambiguous"/"no_candidate" (zero or
        several passed) -- geometry decides, nothing here guesses."""
        case_dir = os.path.join(args.merged_root, name)
        checked = [(cand, verify(args.geometry_root, cand, case_dir, args.tol_mm)) for cand in candidates]
        passing = [(cand, m) for cand, m in checked if m is not None
                  and m["median_mm"] <= args.accept_median_mm and m["coverage"] >= args.accept_coverage]
        detail = "; ".join(_fmt(cand, m) for cand, m in checked)
        if len(passing) == 1:
            full, m = passing[0]
            rejected = "; ".join(cand for cand, _ in checked if cand != full)
            return {"current": name, "full_name": full, "status": "ok", **m,
                    "verify": f"VERIFIED ({note}, geometry chose 1 of {len(candidates)}) "
                    f"median {m['median_mm']:.3f} mm, coverage {100 * m['coverage']:.0f}%"
                    f"{' -- rejected: ' + rejected if rejected else ''}"}
        status = "no_candidate" if not passing else "ambiguous"
        return {"current": name, "full_name": "", "status": status,
                "candidates": "; ".join(candidates),
                "tiebreak": (f"{note}: " +
                            ("none of the candidates' geometry matches this case's CFD wall"
                             if not passing else
                             f"{len(passing)} candidates ALL pass (genuinely ambiguous)")
                            + f" -- {detail}")}

    rows = []
    t0 = time.time()
    for i, name in enumerate(aneux_cases):
        if all_full_names and name in all_full_names:
            full, status = name, "already_full"
        else:
            c = cands.get(name, [])
            if not c:
                # no exact simplify_name() match -- try candidates whose simplified
                # name merely STARTS WITH this short id (simplify_name only strips
                # "_cutN"; a trailing disambiguator like "C0028b_cut2" -> "C0028b"
                # is missed by exact matching but still a plausible full name for
                # the short id "C0028"). Confirmed for real: case_registry.xlsx
                # independently records case=C0028, original_name=C0028b.
                if args.no_verify:
                    rows.append({"current": name, "full_name": "", "status": "no_candidate"})
                    continue
                prefix = [full for key, names in cands.items() if key.upper().startswith(name.upper())
                         for full in names]
                if not prefix:
                    rows.append({"current": name, "full_name": "", "status": "no_candidate"})
                    continue
                rows.append(resolve_by_geometry(name, prefix, "prefix fallback"))
                continue
            if len(c) > 1:
                # several names simplify to this exact short id -- let geometry
                # break the tie rather than picking one string match by guesswork
                if args.no_verify:
                    rows.append({"current": name, "full_name": "", "status": "ambiguous",
                                "candidates": "; ".join(c)})
                    continue
                rows.append(resolve_by_geometry(name, c, "exact name collision"))
                continue
            full, status = c[0], "ok"

        row = {"current": name, "full_name": full, "status": status}
        if args.no_verify:
            row["verify"] = "SKIPPED (--no_verify)"
        else:
            m = verify(args.geometry_root, full, os.path.join(args.merged_root, name), args.tol_mm)
            if m is None:
                row["verify"] = "NOT VERIFIED (geometry .obj or wall_data.pt missing)"
            else:
                row.update(m)
                ok = m["median_mm"] <= args.accept_median_mm and m["coverage"] >= args.accept_coverage
                row["verify"] = (f"VERIFIED median {m['median_mm']:.3f} mm, "
                                f"coverage {100 * m['coverage']:.0f}%" if ok else
                                f"NOT VERIFIED median {m['median_mm']:.3f} mm, "
                                f"coverage {100 * m['coverage']:.0f}%")
                if not ok:
                    row["status"] = "geometry_mismatch"
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(aneux_cases)}] verified ({time.time() - t0:.0f}s)", flush=True)
    return rows, cands


def safety_check(rows, all_current_names):
    """Defence in depth on top of os.rename's own no-overwrite guard: refuse to even
    ATTEMPT a rename whose destination is unsafe, rather than rely on that guard
    catching it live. Nothing here is destructive; it only downgrades a row's status
    to a non-renameable one and explains why. Checks:

      dest_collision   two proposed renames share the same destination -- renaming
                       both would make the second overwrite the first
      dest_is_current  the destination is another AneuX case's CURRENT folder name
                       (a chain: renaming A->B before B moves out of the way would
                       otherwise be silently skipped by the exists-check, which is
                       safe but means an ordering-dependent partial result) -- we
                       surface it instead of leaving it to chance
    """
    ok = [r for r in rows if r["status"] == "ok"]
    dests = collections.Counter(r["full_name"] for r in ok)
    dup_dests = {d for d, n in dests.items() if n > 1}
    for r in ok:
        if r["full_name"] in dup_dests:
            r["status"], r["safety"] = "dest_collision", \
                f"destination shared by {dests[r['full_name']]} proposed renames"
    ok = [r for r in rows if r["status"] == "ok"]
    for r in ok:
        if r["full_name"] in all_current_names and r["full_name"] != r["current"]:
            r["status"], r["safety"] = "dest_is_current", \
                f"destination '{r['full_name']}' is another case's CURRENT folder name"
    return rows


def write_name_mapping(rows, cands, merged_root):
    """SHORT ID -> FULL NAME, keyed by simplify_name() output (e.g. "ANSYS09",
    "SNF365"), not by any folder's current name -- reusable for renaming OTHER
    folders of the same AneuX dataset (a different mirror, a future incoming
    batch, ...), regardless of what angioflowv2_merged happens to be named.

    That "not by current name" distinction matters here specifically: once a
    case has been renamed to its full name in THIS dataset, its short id no
    longer appears anywhere in angioflowv2_merged, so a mapping built from
    `rows[*]["current"]` alone would forget it. Every simplify_name() key with
    exactly one AneuX_stable candidate is therefore added unconditionally --
    that correspondence is a fact about AneuX_stable, independent of any
    rename history.

    For the rare collision keys (several full names simplifying to one short
    id, e.g. "SNF365"), simplify_name() alone cannot resolve them; only a live
    geometry check against a real CFD case can, which requires a folder still
    carrying that exact short name. So this file is MERGED, not overwritten:
    a collision resolved in an earlier run is kept even if no current folder
    can re-derive it, and only a collision resolved again in THIS run
    (rows[*]["status"] == "ok" with current == the short id) can add one.
    Anything still unresolved is listed separately with the reason, and is
    explicitly NOT safe to rename from this file alone.
    """
    import json

    path = os.path.join(merged_root, "_meta", "aneux_name_mapping.json")
    existing = {"mapping": {}, "unresolved": {}}
    if os.path.isfile(path):
        with open(path) as fh:
            existing = json.load(fh)

    mapping = dict(existing.get("mapping", {}))
    unresolved = dict(existing.get("unresolved", {}))

    for key, names in cands.items():
        if len(names) == 1:
            mapping[key] = names[0]                      # unambiguous: always safe, always kept fresh
    resolved_now = {r["current"]: r["full_name"] for r in rows
                    if r["status"] == "ok" and r["full_name"]}
    mapping.update(resolved_now)
    unresolved = {k: v for k, v in unresolved.items() if k not in mapping}  # since resolved
    for r in rows:
        if r["status"] in ("no_candidate", "ambiguous", "geometry_mismatch",
                           "dest_collision", "dest_is_current") and r["current"] not in mapping:
            unresolved[r["current"]] = r.get("tiebreak") or r.get("safety") or r["status"]

    out = {"mapping": mapping, "unresolved": unresolved,
          "note": "short id (simplify_name output) -> full AneuX case name (AneuX_stable folder "
                  "name). Unique matches come from simplify_name() alone; collisions were broken "
                  "by a geometry check against a real CFD case at the time they were resolved. "
                  "This file is merged across runs, so an entry survives even after the "
                  "corresponding angioflowv2_merged folder has itself been renamed to its full "
                  "name. 'unresolved' entries have no confirmed full name -- do not rename from "
                  "those without checking them yourself."}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    return path, len(mapping), len(unresolved)


def main():
    args = parse_args()
    rows, cands = plan(args)
    rows = safety_check(rows, {r["current"] for r in rows})
    by_status = collections.Counter(r["status"] for r in rows)
    print(f"\n[scan] {len(rows)} AneuX case folder(s) in {args.merged_root}")
    print(f"       name source: {args.geometry_root}")
    print(f"       geometry check: {'SKIPPED' if args.no_verify else GEOMETRY_OBJ + ' vs wall_data.pt'}")
    print(f"       {dict(by_status)}\n")

    renameable = [r for r in rows if r["status"] in ("ok",)]
    mismatched = [r for r in rows if r["status"] == "geometry_mismatch"]
    unsafe = [r for r in rows if r["status"] in ("dest_collision", "dest_is_current")]
    print(f"proposed renames (geometry OK + safety OK): {len(renameable)}")
    print(f"name matched but geometry FAILED the check (not renamed): {len(mismatched)}")
    if unsafe:
        print(f"BLOCKED by safety check (not renamed) -- {len(unsafe)}:")
        for r in unsafe:
            print(f"   {r['current']:20s} -> {r['full_name']:34s} {r['safety']}")
    print()

    print(f"{'current':22s} -> {'full name':40s} {'verification'}")
    print("-" * 110)
    for r in renameable + mismatched:
        print(f"{r['current']:22s} -> {r['full_name']:40s} {r['verify']}")

    amb = [r for r in rows if r["status"] == "ambiguous"]
    if amb:
        print(f"\nAMBIGUOUS (not renamed, needs a human decision) -- {len(amb)}:")
        for r in amb:
            print(f"   {r['current']:20s} candidates: {r['candidates']}")
            if r.get("tiebreak"):
                print(f"      {r['tiebreak']}")

    no_cand = [r for r in rows if r["status"] == "no_candidate"]
    if no_cand:
        print(f"\nno full name found in {os.path.basename(args.geometry_root)} for "
              f"{len(no_cand)} folder(s) (kept as-is):")
        for r in no_cand:
            print(f"   {r['current']:20s}" + (f"   {r['tiebreak']}" if r.get("tiebreak") else ""))

    already = [r for r in rows if r["status"] == "already_full"]
    print(f"\nalready full-named: {len(already)}")

    map_path, n_mapped, n_unresolved = write_name_mapping(rows, cands, args.merged_root)
    print(f"[map ] short -> full name mapping ({n_mapped} entries, {n_unresolved} unresolved) -> {map_path}")

    if not args.execute:
        print("\n[dry run only] pass --execute --yes to actually rename.")
        return
    if not args.yes:
        print("\n[dry run only] --execute given without --yes -- not touching anything.")
        return

    # ---- apply --------------------------------------------------------------
    import datetime
    import json

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.merged_root, "_meta", f"rename_aneux_{stamp}.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = {"started": stamp, "merged_root": args.merged_root, "geometry_root": args.geometry_root,
           "accept_median_mm": args.accept_median_mm, "accept_coverage": args.accept_coverage,
           "renamed": [], "skipped_target_exists": [], "failed": []}

    def _save_log():
        # written after every case (not just at the end), so a crash or a kill
        # partway through still leaves a record of exactly what completed
        with open(log_path, "w") as fh:
            json.dump(log, fh, indent=2)

    dest_set = [r["full_name"] for r in renameable]
    assert len(dest_set) == len(set(dest_set)), \
        "refusing to execute: duplicate destinations slipped past safety_check()"
    print(f"\n[execute] renaming {len(renameable)} folder(s) ... log -> {log_path}")
    renamed = {}
    for r in renameable:
        src = os.path.join(args.merged_root, r["current"])
        dst = os.path.join(args.merged_root, r["full_name"])
        entry = {"current": r["current"], "full_name": r["full_name"],
                 "median_mm": r.get("median_mm"), "coverage": r.get("coverage"),
                 "verify": r.get("verify")}
        if os.path.exists(dst):
            print(f"  [skip] {r['current']} -> {r['full_name']}: target already exists")
            log["skipped_target_exists"].append(entry)
            _save_log()
            continue
        try:
            os.rename(src, dst)
        except OSError as exc:
            print(f"  [FAIL] {r['current']} -> {r['full_name']}: {exc}")
            entry["error"] = str(exc)
            log["failed"].append(entry)
            _save_log()
            continue
        renamed[r["current"]] = r["full_name"]
        log["renamed"].append(entry)
        _save_log()
        print(f"  [ok] {r['current']} -> {r['full_name']}")

    sanity_dir = os.path.join(args.merged_root, "_sanity")
    sanity_renamed = []
    if os.path.isdir(sanity_dir):
        import glob
        for old, new in renamed.items():
            for f in glob.glob(os.path.join(sanity_dir, old)) + \
                     glob.glob(os.path.join(sanity_dir, f"{old}_*")) + \
                     glob.glob(os.path.join(sanity_dir, f"{old}.*")):
                suffix = os.path.basename(f)[len(old):]
                nf = os.path.join(sanity_dir, new + suffix)
                if not os.path.exists(nf):
                    os.rename(f, nf)
                    sanity_renamed.append({"from": f, "to": nf})
    log["sanity_renamed"] = sanity_renamed

    log["registry_updated"] = False
    if renamed and os.path.isfile(args.registry):
        import pandas as pd
        backup = args.registry + f".bak.{stamp}"
        shutil.copy(args.registry, backup)
        df = pd.read_excel(args.registry)
        df["case"] = df["case"].astype(str).map(lambda c: renamed.get(c, c))
        df.to_excel(args.registry, index=False)
        log["registry_updated"] = True
        log["registry_backup"] = backup
        print(f"\n[registry] updated {args.registry} ({len(renamed)} case ids), backup at {backup}")

    log["finished"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_log()
    print(f"\n[done] {len(renamed)} renamed, {len(log['skipped_target_exists'])} skipped "
          f"(target existed), {len(log['failed'])} failed.")
    print(f"[log] {log_path}")


if __name__ == "__main__":
    main()
