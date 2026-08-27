"""Pool-size ablation: does metadata-aware retrieval need the full ~190k database?

Holds the query budget fixed at 4000 and shrinks the candidate pool by random
subsampling, then measures leave-one-target-out ASR for Static Top-4000 and
Adaptive Transferability. If ASR saturates well before the full pool, the gain
is not an artifact of dataset scale.
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
from pathlib import Path
import sys, time
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jailbreak_retrieval.utils.data import prepare_target_dataset  # noqa: E402
from jailbreak_retrieval.utils.methods import select_static_top, deterministic_seed  # noqa: E402
from jailbreak_retrieval.hparam_sensitivity_suite import (  # noqa: E402
    get_data, select_adaptive_transferability_p, PAPER_W,
)

OUT = REPO_ROOT / "jailbreak_retrieval_params_grid_search" / "pool_size"
PREP: dict = {}
BUDGET = 4000
SIZES = [6000, 12000, 25000, 50000, 0]  # 0 = full pool
BATCH = 25


def run_task(task):
    size, target, method = task
    p = PREP[target]
    pdf, mcols = p["prepared_df"], p["model_cols"]
    n = len(pdf)
    use = n if size == 0 else min(size, n)
    sub = pdf.sample(n=use, random_state=deterministic_seed("poolsize", target, size)) \
             .reset_index(drop=True) if use < n else pdf
    budget = min(BUDGET, len(sub))
    if method == "static_top":
        sel = select_static_top(sub, mcols, BATCH, budget, target)
    else:
        sel = select_adaptive_transferability_p(sub, mcols, BATCH, budget, target, w=PAPER_W)
    # Report the ACTUAL pool size used (use), not the requested size. When the
    # requested size exceeds the real pool, `use` == full-pool size, which is the
    # same row the size=0 (full) task produces -- deduped in main() so the
    # saturation curve has exactly one row per real pool size.
    return (use, target, method,
            float(sel["target_success"].mean()), use, budget)


def main():
    global PREP
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="both", choices=["both", "mistral"])
    ap.add_argument("--data-path", default=None,
                    help="Path to a pre-built response-level parquet (overrides --data / "
                         "cache; fully standalone). Env: JAILBREAK_RETRIEVAL_DATA.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df, targets = get_data(args.data, data_path=args.data_path)
    print("[prep] preparing per-target datasets...", flush=True)
    for t in targets:
        pr = prepare_target_dataset(df, targets, t)
        PREP[t] = {"prepared_df": pr.prepared_df, "model_cols": pr.model_cols}
    # One task per DISTINCT real pool size per target. When a requested size
    # exceeds a target's pool it collapses to the full pool; dedupe so full-pool
    # is not double-run (once via size>=n, once via size=0).
    tasks = []
    for t in targets:
        n = len(PREP[t]["prepared_df"])
        seen_actual = set()
        for s in SIZES:
            actual = n if s == 0 else min(s, n)
            if actual in seen_actual:
                continue
            seen_actual.add(actual)
            for m in ("static_top", "adaptive_transfer"):
                tasks.append((s, t, m))
    print(f"[run] {len(tasks)} tasks", flush=True)
    ctx = mp.get_context("fork")
    rows = []
    with ctx.Pool(min(90, mp.cpu_count() - 4)) as pool:
        done = 0
        for r in pool.imap_unordered(run_task, tasks, chunksize=1):
            rows.append(r); done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"[run] {done}/{len(tasks)}", flush=True)
    res = pd.DataFrame(rows, columns=["pool_size", "target", "method", "asr", "used", "budget"])
    res.to_csv(OUT / "pool_size_per_target.csv", index=False)
    agg = res.groupby(["method", "pool_size"]).agg(
        mean_asr=("asr", "mean"), sd_asr=("asr", "std")).reset_index()
    agg["mean_asr"] = (100 * agg["mean_asr"]).round(1)
    agg["sd_asr"] = (100 * agg["sd_asr"]).round(1)
    agg.to_csv(OUT / "pool_size_summary.csv", index=False)
    print("\n===== POOL-SIZE ABLATION (ASR %, budget 4000) =====", flush=True)
    for m in ("static_top", "adaptive_transfer"):
        sub = agg[agg.method == m].sort_values("pool_size")
        print(f"\n{m}:", flush=True)
        for _, r in sub.iterrows():
            print(f"  pool~{int(r.pool_size):>6d}  ASR={r.mean_asr:.1f}  SD={r.sd_asr:.1f}", flush=True)
    print(f"\n[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()
