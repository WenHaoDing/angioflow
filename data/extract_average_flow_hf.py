# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.4.0",
#     "numpy<2.4",
#     "huggingface_hub>=1.8",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Hugging Face Job: time-averaged blood flow of the ImperialNHS AngioFlow CFD cases.

A self-contained uv script -- submit it from your machine and everything runs on Hugging
Face (nothing is downloaded locally, no repo clone; the case lists it needs are embedded
below):

    hf jobs uv run --flavor cpu-upgrade --timeout 6h -s HF_TOKEN -d \\
        data/extract_average_flow_hf.py --num_jobs 4 --job_index 0 --execute --yes
    (repeat with --job_index 1, 2, 3 for 4 parallel Jobs)

Reads whding123/AngioFlow cfd_data/<case>/ -- only the 261 ImperialNHS cases of the case
registry (embedded, IMPERIALNHS_CASES; 256 are on the Hub with blood_data.pt + the
registration; the Hub folder also holds AneuX cases and nothing in it tells them apart
reliably; --all_cases to process every case) -- and uploads to the same dataset:

    average_flow/<case>/average_flow.pt           the 3D mesh + the cycle-averaged velocity
    average_flow/<case>/wall_mesh_registration.pt  copied unchanged (wall surface, same frame)

It only ADDS files under average_flow/ (never deletes or overwrites; cases already there
are skipped), and refuses an --out_prefix that is, contains or lies inside the source
cfd_data/. --out_repo <dataset> writes to a different dataset instead.

average_flow.pt (a dict):
    pos         [N, 3]  float32   interior mesh nodes, metres (blood_data.pt "pos";
                                  "pos_frame" is identical and not kept)
    cells       [C, 4]  int32     tetrahedra (node indices into pos; int64 in the source,
                                  stored int32 -- exact, N < 2^31)
    vel_mean    [N, 3]  float32   time-averaged velocity over all exported frames, m/s
                                  (accumulated in float64)
    speed_mean  [N]     float32   time-averaged |velocity|, m/s
    meta        dict              case, n_frames, units, source repo/path/LFS sha256,
                                  first_cycle_suspect (see below), created

The frames are uniform in time over one cardiac cycle (80 per case), so the plain mean
over frames is the cycle average. NOTE: some cases were exported from the FIRST simulated
cycle, which still holds the start-up transient, so their average is biased;
meta["first_cycle_suspect"] marks the 216 AngioFlow cases flagged by
debugging/detect_first_cycle_cases.py (embedded, FIRST_CYCLE_SUSPECTS; 173 are ImperialNHS).

On the Job, each 3.4-3.6 GB blood_data.pt is fetched to the Job's own disk (--scratch),
only x/y/z_velocity + pos + cells are read, and it is deleted before the next case.
Results are uploaded in commits of --commit_every cases; a rerun resumes (skips cases
already uploaded). --num_jobs N --job_index i splits the cases (crc32(case) % N == i).
Without --execute --yes nothing is uploaded: --max_cases (default 1) cases are written to
--scratch. --source_dir reads a local mirror instead of the Hub (local test with
`uv run data/extract_average_flow_hf.py --source_dir <mirror> --max_cases 1`).
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
import zlib

import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from huggingface_hub.hf_api import RepoFile

SOURCE_REPO = "whding123/AngioFlow"
PREFIX = "cfd_data"
OUT_PREFIX = "average_flow"
REPO_TYPE = "dataset"
BLOOD, REG, OUT_NAME = "blood_data.pt", "wall_mesh_registration.pt", "average_flow.pt"
VEL_KEYS = ("x_velocity", "y_velocity", "z_velocity")
MAX_RETRIES, RETRY_DELAY = 5, 10


def with_retry(fn, what):
    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except (RepositoryNotFoundError, EntryNotFoundError):
            raise
        except Exception as e:
            print(f"[retry] {what}: attempt {attempt}/{MAX_RETRIES} failed: {e}", flush=True)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(max(delay, 60) if "429" in str(e) else delay)
            delay *= 2


def list_tree(api, repo, prefix):
    """{case: {file name: RepoFile}} for <prefix>/<case>/<file> (top-level files only)."""
    def _list():
        try:
            return list(api.list_repo_tree(repo, path_in_repo=prefix, recursive=True, repo_type=REPO_TYPE))
        except (RepositoryNotFoundError, EntryNotFoundError):
            return []
    cases = {}
    for f in with_retry(_list, f"list {repo}/{prefix}"):
        parts = f.path.split("/")
        if isinstance(f, RepoFile) and len(parts) == 3:
            cases.setdefault(parts[1], {})[parts[2]] = f
    return cases


def average_case(blood_path):
    """(pos, cells, vel_mean, speed_mean, n_frames) from one blood_data.pt (mmap: only
    the keys used are read)."""
    b = torch.load(blood_path, weights_only=False, mmap=True)
    pos = b["pos"].float().contiguous()
    cells = b["cells"]
    if int(cells.max()) >= 2 ** 31 or int(cells.min()) < 0 or int(cells.max()) >= pos.shape[0]:
        raise ValueError(f"cells index range {int(cells.min())}..{int(cells.max())} vs {pos.shape[0]} nodes")
    T = b[VEL_KEYS[0]].shape[0]
    vel_sum = torch.zeros(pos.shape[0], 3, dtype=torch.float64)
    speed_sum = torch.zeros(pos.shape[0], dtype=torch.float64)
    comps = [b[k] for k in VEL_KEYS]                       # each [T, N, 1], memory-mapped
    for t in range(T):
        v = torch.cat([c[t] for c in comps], dim=-1).double()   # [N, 3]
        vel_sum += v
        speed_sum += v.norm(dim=-1)
    if not torch.isfinite(vel_sum).all():
        raise ValueError("non-finite velocity")
    return pos, cells.to(torch.int32).contiguous(), (vel_sum / T).float(), (speed_sum / T).float(), T


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source_repo", default=SOURCE_REPO)
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--out_repo", default=None, help="default: --source_repo")
    p.add_argument("--out_prefix", default=OUT_PREFIX)
    p.add_argument("--source_dir", default=None,
                   help="read <source_dir>/<case>/ locally instead of from the Hub (local test)")
    p.add_argument("--scratch", default="/tmp/average_flow")
    p.add_argument("--case_list", default=None,
                   help="file with the cases to process (default: the embedded ImperialNHS list)")
    p.add_argument("--all_cases", action="store_true", help="every case on the Hub, not just ImperialNHS")
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--max_cases", type=int, default=None, help="default 1 without --execute")
    p.add_argument("--num_jobs", type=int, default=1)
    p.add_argument("--job_index", type=int, default=0)
    p.add_argument("--commit_every", type=int, default=10, help="cases per upload commit")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()
    upload = args.execute and args.yes
    out_repo = args.out_repo or args.source_repo
    # never write into the source data: the output folder must be a separate, non-root
    # folder, neither the source prefix, inside it, nor containing it
    out_p, src_p = args.out_prefix.strip("/"), args.prefix.strip("/")
    if out_repo == args.source_repo and (not out_p or out_p == src_p or out_p.startswith(src_p + "/")
                                         or src_p.startswith(out_p + "/")):
        raise SystemExit(f"[abort] --out_prefix '{args.out_prefix}' would write into the source data "
                         f"'{args.prefix}/' of {out_repo} -- use a separate folder (default {OUT_PREFIX})")
    api = HfApi(token=args.token)
    suspects = set(FIRST_CYCLE_SUSPECTS.split())
    print(f"[job] JOB_ID={os.environ.get('JOB_ID', '(local)')} -> {out_repo}/{args.out_prefix}/", flush=True)

    if args.source_dir:
        cases = sorted(c for c in os.listdir(args.source_dir)
                       if os.path.exists(os.path.join(args.source_dir, c, BLOOD)))
        src_files = {}
    else:
        src_files = list_tree(api, args.source_repo, args.prefix)
        cases = sorted(c for c, fs in src_files.items() if BLOOD in fs and REG in fs)
        missing = sorted(c for c, fs in src_files.items() if not (BLOOD in fs and REG in fs))
        if missing:
            print(f"[warn] {len(missing)} case(s) lack {BLOOD} or {REG}, skipped: {missing[:10]}")
    if not args.all_cases:
        wanted = set(open(args.case_list).read().split()) if args.case_list else set(IMPERIALNHS_CASES.split())
        print(f"[cases] {len(wanted)} wanted ({args.case_list or 'embedded ImperialNHS list'}); "
              f"{len(wanted & set(cases))} available, missing: {sorted(wanted - set(cases))}")
        cases = [c for c in cases if c in wanted]
    if args.cases:
        cases = [c for c in cases if c in set(args.cases)]
    cases = [c for c in cases if zlib.crc32(c.encode()) % args.num_jobs == args.job_index]
    done = {c for c, fs in list_tree(api, out_repo, args.out_prefix).items()
            if OUT_NAME in fs and REG in fs} if upload else set()
    todo = [c for c in cases if c not in done]
    limit = args.max_cases if args.max_cases is not None else (None if upload else 1)
    todo = todo[:limit]
    print(f"[plan] job {args.job_index}/{args.num_jobs}: {len(cases)} case(s), {len(done & set(cases))} "
          f"already on {out_repo}/{args.out_prefix}, {len(todo)} to process | upload: {upload}", flush=True)

    os.makedirs(args.scratch, exist_ok=True)
    staged, failed = [], {}

    def flush():
        if not staged:
            return
        ops = []
        for c in staged:
            for name in (OUT_NAME, REG):
                ops.append(CommitOperationAdd(path_in_repo=f"{args.out_prefix}/{c}/{name}",
                                              path_or_fileobj=os.path.join(args.scratch, "out", c, name)))
        with_retry(lambda: api.create_commit(repo_id=out_repo, repo_type=REPO_TYPE, operations=ops,
                                             commit_message=f"average_flow: {len(staged)} case(s) "
                                                            f"({staged[0]} .. {staged[-1]})"),
                   f"upload {len(staged)} case(s)")
        print(f"[upload] {len(staged)} case(s) -> {out_repo}/{args.out_prefix}/", flush=True)
        for c in staged:
            shutil.rmtree(os.path.join(args.scratch, "out", c), ignore_errors=True)
        staged.clear()

    for i, c in enumerate(todo, 1):
        t0 = time.time()
        dl = os.path.join(args.scratch, "dl")
        try:
            if args.source_dir:
                blood, reg = os.path.join(args.source_dir, c, BLOOD), os.path.join(args.source_dir, c, REG)
                sha = None
            else:
                fetch = lambda name: with_retry(lambda: hf_hub_download(
                    args.source_repo, f"{args.prefix}/{c}/{name}", repo_type=REPO_TYPE, token=args.token,
                    local_dir=dl), f"fetch {c}/{name}")
                blood, reg = fetch(BLOOD), fetch(REG)
                lfs = src_files[c][BLOOD].lfs
                sha = getattr(lfs, "sha256", None) if lfs else None
            t_fetch = time.time() - t0
            pos, cells, vel_mean, speed_mean, T = average_case(blood)
            out_dir = os.path.join(args.scratch, "out", c)
            os.makedirs(out_dir, exist_ok=True)
            meta = dict(case=c, n_frames=T, average="mean over all frames (uniform in time, one cycle)",
                        units=dict(pos="m", vel_mean="m/s", speed_mean="m/s"),
                        source_repo=args.source_repo, source_path=f"{args.prefix}/{c}/{BLOOD}",
                        source_sha256=sha, first_cycle_suspect=c in suspects,
                        created=time.strftime("%Y-%m-%d %H:%M:%S"), job_id=os.environ.get("JOB_ID"))
            torch.save(dict(pos=pos, cells=cells, vel_mean=vel_mean, speed_mean=speed_mean, meta=meta),
                       os.path.join(out_dir, OUT_NAME))
            shutil.copyfile(reg, os.path.join(out_dir, REG))
            if not args.source_dir:
                shutil.rmtree(dl, ignore_errors=True)        # free the 3.5 GB before the next case
            print(f"[ok] [{i}/{len(todo)}] {c}: N {pos.shape[0]} C {cells.shape[0]} T {T} | "
                  f"mean |v_mean| {float(vel_mean.norm(dim=-1).mean()):.3f} m/s | "
                  f"fetch {t_fetch:.0f}s total {time.time() - t0:.0f}s"
                  + (" | first-cycle suspect" if meta["first_cycle_suspect"] else ""), flush=True)
        except Exception as e:
            failed[c] = repr(e)
            print(f"[fail] {c}: {e!r}", flush=True)
            shutil.rmtree(dl, ignore_errors=True)
            continue
        if upload:
            staged.append(c)
            if len(staged) >= args.commit_every:
                flush()
    if upload:
        flush()
    else:
        print(f"[no upload] outputs in {os.path.join(args.scratch, 'out')} -- pass --execute --yes to upload")
    print(f"[done] {len(todo) - len(failed)} ok, {len(failed)} failed" + (f": {failed}" if failed else ""))


# --- embedded case lists (uv Jobs upload only this file) -------------------------------
# IMPERIALNHS_CASES: the case registry's ImperialNHS cases (angioflowv2_merged/_meta/case_registry.xlsx)
# FIRST_CYCLE_SUSPECTS: debugging/file_lists/angioflow_first_cycle_suspects.txt
IMPERIALNHS_CASES = """
0HRrQPIFOr_aneurysm1 0WM2CpNLPz_aneurysm1 0WM2CpNLPz_aneurysm2 0YC7yhYVd5_aneurysm1
0ZafRQZ51s_aneurysm1 0ZafRQZ51s_aneurysm2 0bjVkRg2pM_aneurysm1 0m5cVmE8R1_aneurysm1
184D0jB3dS_aneurysm1 1AICKuP53m_aneurysm1 1FtKxiai9X_aneurysm1 1KPkS2Uocy_aneurysm1
1NndJJrTjS_aneurysm1 2uw5ITsPNj_aneurysm1 33aRAgScjC_aneurysm1 33aRAgScjC_aneurysm2
34cX31boJl_aneurysm1 3AZ6NRxph7_aneurysm1 3uErkiItxd_aneurysm1 4Kg5vZ4wzI_aneurysm1
4N00Dar9XN_aneurysm1 4hTp3oln7B_aneurysm1 4lNuZOEFjC_aneurysm1 5M9lbzxsQn_aneurysm1
5y61ssrDxn_aneurysm1 5ywqV955hJ_aneurysm1 6CEUwdLVY7_aneurysm1 6CEUwdLVY7_aneurysm2
6GpXbvnIHl_aneurysm1 6ND9t8FGqc_aneurysm1 6RKCruGy1L_aneurysm1 6RKCruGy1L_aneurysm2
6RKCruGy1L_aneurysm3 6jilh4kjKC_aneurysm1 72rwT0u5nJ_aneurysm1 72rwT0u5nJ_aneurysm2
7TFJw4nWkT_aneurysm1 7UdKnGmnSl_aneurysm1 7UdKnGmnSl_aneurysm2 7WAaHVj4Ph_aneurysm1
7WAaHVj4Ph_aneurysm2 7kb6NlsoYG_aneurysm1 7kb6NlsoYG_aneurysm2 7oc9yUpOpK_aneurysm1
7oc9yUpOpK_aneurysm2 7xgWsj2Zyy_aneurysm1 85ysHiRLZo_aneurysm1 8GR5WKJm5p_aneurysm1
8QGRdoo5w7_aneurysm1 8a4gezE7C7_aneurysm1 8jD72Wn56o_aneurysm1 92CNYwYfxY_aneurysm1
9u0NcIKDhI_aneurysm1 AEjGIUDIR0_aneurysm1 AVTJ0tTFOC_aneurysm1 AnHmgCxWr8_aneurysm1
AnHmgCxWr8_aneurysm2 AoQOqiHhfc_aneurysm1 AwTkB4fFo5_aneurysm1 B2o6VFmWVW_aneurysm1
BBHJDCAXOc_aneurysm1 BCwbjsbsql_aneurysm1 BcXeCx3voF_aneurysm1 BcxvP7fCKc_aneurysm1
BmsAIvOmyi_aneurysm1 CFV81HMHqU_aneurysm1 CPH5rxEbyy_aneurysm1 CPH5rxEbyy_aneurysm2
DEKKRPCeEs_aneurysm1 DdLdNXx4Ha_aneurysm1 E1IkhBc9Ob_aneurysm1 E9qZXSZFkq_aneurysm1
EKM3GE5uJy_aneurysm1 ELB4vZfIax_aneurysm1 ETdbJM4C7h_aneurysm1 ETdbJM4C7h_aneurysm2
EerGeccCJO_aneurysm1 F3C7xUIbYX_aneurysm1 F6VVfHU20n_aneurysm1 FSMAfOehzu_aneurysm1
G7XzfBWOwd_aneurysm1 G7XzfBWOwd_aneurysm2 GQNJfE8a2B_aneurysm2 GdGHoFVI2R_aneurysm1
GfGaEQELI0_aneurysm1 GfGaEQELI0_aneurysm2 GfGaEQELI0_aneurysm3 GtJPG36fHo_aneurysm1
H0ZO7MsTcK_aneurysm1 HMhvLDdLeW_aneurysm1 HOntnbCoYW_aneurysm1 HUfWo4Tb22_aneurysm1
HUfWo4Tb22_aneurysm2 HZTGU6EAEz_aneurysm1 HjYE3vqBnn_aneurysm1 HkxQLQS02d_aneurysm1
I4vL5p57t4_aneurysm1 I4vL5p57t4_mxf9_aneurysm1 I4vL5p57t4_z19o_aneurysm1 IBP7G5ag7l_aneurysm1
J5ux89ZZKC_aneurysm1 JFtZYj7T3t_aneurysm1 Jc0kx4I6ue_aneurysm1 KZ1T8EkshQ_aneurysm1
Kq3x0YFqqG_aneurysm1 LDvVm8wwnZ_aneurysm1 LKyKFYLRnu_aneurysm1 LNwsEEXDJd_aneurysm1
LpB12Yjn9j_aneurysm1 MKTVUV4PYh_aneurysm1 Mm988eVmYt_aneurysm1 MsIF6SVJAb_aneurysm1
N2k0WrXF8P_aneurysm1 NnimNakgLG_aneurysm1 NvBz7T8I5h_aneurysm2 O391k17MTW_aneurysm1
OT59KQMzHU_aneurysm1 OWVb3sYh9i_aneurysm1 P8zX0D9puy_aneurysm1 P8zX0D9puy_aneurysm2
PWaEBPxO1x_aneurysm1 Pl7X2uNWem_aneurysm1 Pztke2rFUB_aneurysm1 QEONoTMnF3_aneurysm1
QFC8asCLko_aneurysm1 QIH3neKXZG_aneurysm1 Qh3VMN5hFO_aneurysm1 QsLUrELCDq_aneurysm1
QsLUrELCDq_aneurysm2 Qui9M1ncb0_aneurysm1 RH1qo9HH2P_aneurysm1 RLsnAHGQtx_aneurysm1
RcbTSbBFsL_aneurysm1 RcbTSbBFsL_aneurysm2 RcbTSbBFsL_aneurysm3 ReCexteoIH_aneurysm1
ReCexteoIH_aneurysm2 RkQXo2wzr7_aneurysm1 RlJkB9JXCD_aneurysm1 SURRrgDljJ_aneurysm1
SURRrgDljJ_aneurysm2 SUx0YKpq4K_aneurysm1 Tf1AKPy0kY_aneurysm1 TiF2teqOxw_aneurysm1
U8Pxm8MQSd_aneurysm1 UEuEc57rXz_aneurysm1 UEuU7EVKjf_aneurysm1 V0UeCxjtJc_aneurysm1
V6dLCCpAFd_aneurysm1 VbC0oFd7nx_aneurysm1 VbMgxonVZy_aneurysm1 VjSecjMW7L_aneurysm1
VyKbIURnoD_aneurysm1 W1GD4X6con_aneurysm1 X0YyCGSNsH_aneurysm1 XkBm8Strus_aneurysm2
XkBm8Strus_aneurysm3 XxWinm78X4_aneurysm1 YzNWocbVXH_aneurysm1 ZjPWBHzE8Q_aneurysm1
a0i4VV2kKG_aneurysm1 aaOmjpxYTZ_aneurysm1 aucywHRA8H_aneurysm1 bPyKGh0SGQ_aneurysm1
bPyKGh0SGQ_aneurysm2 bRJ2RUOn2G_aneurysm1 blXmg4ZQvI_aneurysm1 bmkieogL9h_aneurysm1
cTevqTzOcY_aneurysm2 ckNAGjVSzH_aneurysm1 d1Zz8k1U5r_aneurysm1 dJrNvTqn2c_aneurysm1
divtqcYefj_aneurysm1 dkn3xg4stb_aneurysm1 dnk9JaE2fV_aneurysm1 eLNFbBQbJ0_aneurysm1
eM6xqoqoo1_aneurysm1 ekAU1rnb85_aneurysm1 ephJnDiiaz_aneurysm1 euHpN3eKdM_aneurysm1
f9nPCr633w_aneurysm1 fL8IO9dss5_aneurysm1 fPl4CB4npr_aneurysm2 feRpFIokI2_aneurysm2
feRpFIokI2_aneurysm3 fjXCdEOEMh_aneurysm1 gTfMyJqXQp_aneurysm1 gfH8m5H8B1_aneurysm1
gjaVYZ6ifJ_aneurysm1 htpUBQgrql_aneurysm2 i69sYpgXky_aneurysm1 iEgZYkrR0y_aneurysm1
iQsrvhNKjH_aneurysm1 iw8NFvcAKs_aneurysm1 joCX3Lfukw_aneurysm1 k8M8eBuy4l_aneurysm1
kZ1xSqn9iB_aneurysm1 kZ1xSqn9iB_dsxf_aneurysm1 kj9ygVbap7_aneurysm1 lFnV0JnxJv_aneurysm1
li6nKBMI9B_aneurysm1 lsdXovnVrM_aneurysm1 lsdXovnVrM_aneurysm2 lzW4AVuotE_aneurysm1
m0z3ee3JP7_aneurysm1 mfWNIzdKlh_aneurysm1 nNyeHcJwuY_aneurysm1 nUbO1B2S1v_aneurysm1
nVl2UoKaxY_aneurysm1 nfiwuPvfPU_aneurysm1 nfiwuPvfPU_aneurysm2 nszCwhJiNN_aneurysm1
o7cb5hR1rV_aneurysm1 oUiXl6WLr1_aneurysm1 oaoQ29WgOs_aneurysm1 of9ElE5xae_aneurysm1
p9jX0QvqGX_aneurysm1 pBlNGeQpPz_aneurysm1 pNi4EfNLfi_aneurysm1 piPOQz5Qkr_aneurysm1
psZtYn1Eg8_aneurysm1 q0DCEKN8Yz_aneurysm1 q5aHWgC04b_aneurysm1 qHBtIl5Ft4_aneurysm2
qXdeQXeju1_aneurysm1 qXdeQXeju1_aneurysm2 qZGjujyA9f_aneurysm1 qmcoiXe2Ze_aneurysm1
qmcoiXe2Ze_aneurysm2 r2EVkwBdXy_aneurysm1 rASk5uvMt2_aneurysm1 rCHyvghmVr_aneurysm1
rDHhjMGZ6K_aneurysm1 rYkp9DJRZq_aneurysm1 rsxrGg1ACS_aneurysm1 sAZ6eNswyc_aneurysm1
sZoUSg9dKz_aneurysm1 sdfgnnRzJL_aneurysm1 sdfgnnRzJL_aneurysm2 sjJsMVFx89_aneurysm1
tYZ0IT9avC_aneurysm1 uAFn03Y0Gp_aneurysm1 uhALUddPnS_aneurysm1 v2hrDDjiip_aneurysm2
v6TrlhwP7M_aneurysm1 vJS7ISxFri_aneurysm1 vbfoaHqAT8_aneurysm1 wFs1QbldbE_aneurysm1
wTUS1Sz4S4_aneurysm1 wiYQO6BmRS_aneurysm1 wqXrhCmudF_aneurysm1 wvHGUqVal5_aneurysm1
x0tZoieDWj_aneurysm1 y3GooHGaVq_aneurysm3 y3GooHGaVq_aneurysm5 yFTGojohJ4_aneurysm1
yHqeJVY84k_aneurysm1 yUJN2uVA3D_aneurysm1 yUJN2uVA3D_aneurysm2 ziSACL5Z7M_aneurysm1
ztUSO0doTz_aneurysm1
"""

FIRST_CYCLE_SUSPECTS = """
0HRrQPIFOr_aneurysm1 0WM2CpNLPz_aneurysm1 0WM2CpNLPz_aneurysm2 0YC7yhYVd5_aneurysm1
0ZafRQZ51s_aneurysm1 0ZafRQZ51s_aneurysm2 0bjVkRg2pM_aneurysm1 0m5cVmE8R1_aneurysm1
184D0jB3dS_aneurysm1 1AICKuP53m_aneurysm1 1FtKxiai9X_aneurysm1 1KPkS2Uocy_aneurysm1
1NndJJrTjS_aneurysm1 2uw5ITsPNj_aneurysm1 33aRAgScjC_aneurysm1 33aRAgScjC_aneurysm2
34cX31boJl_aneurysm1 3AZ6NRxph7_aneurysm1 3uErkiItxd_aneurysm1 4Kg5vZ4wzI_aneurysm1
4N00Dar9XN_aneurysm1 4hTp3oln7B_aneurysm1 4lNuZOEFjC_aneurysm1 5M9lbzxsQn_aneurysm1
5y61ssrDxn_aneurysm1 5ywqV955hJ_aneurysm1 6CEUwdLVY7_aneurysm1 6CEUwdLVY7_aneurysm2
6GpXbvnIHl_aneurysm1 6ND9t8FGqc_aneurysm1 6RKCruGy1L_aneurysm1 6RKCruGy1L_aneurysm2
6RKCruGy1L_aneurysm3 6jilh4kjKC_aneurysm1 72rwT0u5nJ_aneurysm1 72rwT0u5nJ_aneurysm2
7TFJw4nWkT_aneurysm1 7UdKnGmnSl_aneurysm1 7UdKnGmnSl_aneurysm2 7WAaHVj4Ph_aneurysm1
7WAaHVj4Ph_aneurysm2 7kb6NlsoYG_aneurysm1 7kb6NlsoYG_aneurysm2 7oc9yUpOpK_aneurysm1
7oc9yUpOpK_aneurysm2 7xgWsj2Zyy_aneurysm1 85ysHiRLZo_aneurysm1 8GR5WKJm5p_aneurysm1
8QGRdoo5w7_aneurysm1 8a4gezE7C7_aneurysm1 8jD72Wn56o_aneurysm1 92CNYwYfxY_aneurysm1
9u0NcIKDhI_aneurysm1 AEjGIUDIR0_aneurysm1 AVTJ0tTFOC_aneurysm1 AnHmgCxWr8_aneurysm1
AnHmgCxWr8_aneurysm2 AoQOqiHhfc_aneurysm1 AwTkB4fFo5_aneurysm1 B2o6VFmWVW_aneurysm1
BBHJDCAXOc_aneurysm1 BCwbjsbsql_aneurysm1 BcXeCx3voF_aneurysm1 BcxvP7fCKc_aneurysm1
BmsAIvOmyi_aneurysm1 C0013_cut2 C0064_cut2 C0065_cut2 C0067_cut2 C0071_cut2 C0074a_cut2 C0085_cut2
C0086_cut2 C0088a_cut2 C0088b_cut2 C0089_cut2 C0090_cut2 C0097_cut2 CFV81HMHqU_aneurysm1
CPH5rxEbyy_aneurysm2 DEKKRPCeEs_aneurysm1 DdLdNXx4Ha_aneurysm1 E1IkhBc9Ob_aneurysm1
E9qZXSZFkq_aneurysm1 EKM3GE5uJy_aneurysm1 ELB4vZfIax_aneurysm1 ETdbJM4C7h_aneurysm1
ETdbJM4C7h_aneurysm2 EerGeccCJO_aneurysm1 F3C7xUIbYX_aneurysm1 F6VVfHU20n_aneurysm1
FSMAfOehzu_aneurysm1 G7XzfBWOwd_aneurysm1 G7XzfBWOwd_aneurysm2 GQNJfE8a2B_aneurysm2
GdGHoFVI2R_aneurysm1 GfGaEQELI0_aneurysm1 GfGaEQELI0_aneurysm2 GfGaEQELI0_aneurysm3
GtJPG36fHo_aneurysm1 H0ZO7MsTcK_aneurysm1 HMhvLDdLeW_aneurysm1 HOntnbCoYW_aneurysm1
HUfWo4Tb22_aneurysm1 HUfWo4Tb22_aneurysm2 HZTGU6EAEz_aneurysm1 HjYE3vqBnn_aneurysm1
HkxQLQS02d_aneurysm1 I4vL5p57t4_aneurysm1 I4vL5p57t4_mxf9_aneurysm1 I4vL5p57t4_z19o_aneurysm1
IBP7G5ag7l_aneurysm1 J5ux89ZZKC_aneurysm1 JFtZYj7T3t_aneurysm1 Jc0kx4I6ue_aneurysm1
KZ1T8EkshQ_aneurysm1 Kq3x0YFqqG_aneurysm1 LDvVm8wwnZ_aneurysm1 LKyKFYLRnu_aneurysm1
LNwsEEXDJd_aneurysm1 LpB12Yjn9j_aneurysm1 MKTVUV4PYh_aneurysm1 Mm988eVmYt_aneurysm1
MsIF6SVJAb_aneurysm1 N2k0WrXF8P_aneurysm1 NnimNakgLG_aneurysm1 NvBz7T8I5h_aneurysm2
O391k17MTW_aneurysm1 OT59KQMzHU_aneurysm1 OWVb3sYh9i_aneurysm1 P8zX0D9puy_aneurysm1
P8zX0D9puy_aneurysm2 PWaEBPxO1x_aneurysm1 Pl7X2uNWem_aneurysm1 QEONoTMnF3_aneurysm1
QFC8asCLko_aneurysm1 QIH3neKXZG_aneurysm1 QsLUrELCDq_aneurysm1 QsLUrELCDq_aneurysm2
Qui9M1ncb0_aneurysm1 RH1qo9HH2P_aneurysm1 RLsnAHGQtx_aneurysm1 RcbTSbBFsL_aneurysm1
RcbTSbBFsL_aneurysm2 RcbTSbBFsL_aneurysm3 ReCexteoIH_aneurysm1 RkQXo2wzr7_aneurysm1 SNF00000063_cut2
SNF00000148_cut2 SNF00000152_cut2 SNF00000261_cut2 SURRrgDljJ_aneurysm1 SUx0YKpq4K_aneurysm1
Tf1AKPy0kY_aneurysm1 TiF2teqOxw_aneurysm1 UEuEc57rXz_aneurysm1 UEuU7EVKjf_aneurysm1
UPF_P0038.00_ID2_cut1 UPF_P0084.00_ID1_cut1 UPF_P0098.00_ID2_cut1 UPF_P0188.00_ID1_cut1
UPF_P0207.01_ID1_cut1 UPF_P0229.00_ID1_cut1 UPF_P0232.01_ID1_cut1 UPF_P0272.00_ID1_cut1
UPF_P0277.00_ID1_cut1 UPF_P0280.00_ID2_cut1 UPF_P0286.00_ID1_cut1 UPF_P0311.00_ID1_cut1
USFD_0002_cut1 USFD_0006_cut1 USFD_0011_cut1 USFD_0019_cut1 USFD_0021_cut1 USFD_0025_cut1
USFD_0026_cut1 USFD_0035_cut1 USFD_0051_cut1 USFD_0059_cut1 VbMgxonVZy_aneurysm1
VyKbIURnoD_aneurysm1 W1GD4X6con_aneurysm1 ZjPWBHzE8Q_aneurysm1 a0i4VV2kKG_aneurysm1
blXmg4ZQvI_aneurysm1 dJrNvTqn2c_aneurysm1 divtqcYefj_aneurysm1 dnk9JaE2fV_aneurysm1
fL8IO9dss5_aneurysm1 fPl4CB4npr_aneurysm2 fjXCdEOEMh_aneurysm1 gTfMyJqXQp_aneurysm1
iEgZYkrR0y_aneurysm1 iw8NFvcAKs_aneurysm1 k8M8eBuy4l_aneurysm1 lzW4AVuotE_aneurysm1
m0z3ee3JP7_aneurysm1 nUbO1B2S1v_aneurysm1 nfiwuPvfPU_aneurysm2 oUiXl6WLr1_aneurysm1
p171_EgAWBAIcAQIbABQREwULFwEL_1_cut2 p366_EQ8RAAkWHAISHAcBEQAXBBsb_cut2
p389_HQAAFRkWExMHDAsOABUHCBQK_1_cut2 p407_FxQbDRwNBh8GDAoDFxUJEBwU_cut2 pBlNGeQpPz_aneurysm1
psZtYn1Eg8_aneurysm1 q5aHWgC04b_aneurysm1 qmcoiXe2Ze_aneurysm2 r2EVkwBdXy_aneurysm1
rASk5uvMt2_aneurysm1 sZoUSg9dKz_aneurysm1 sdfgnnRzJL_aneurysm2 v6TrlhwP7M_aneurysm1
vJS7ISxFri_aneurysm1 yHqeJVY84k_aneurysm1 ziSACL5Z7M_aneurysm1
"""

if __name__ == "__main__":
    main()
