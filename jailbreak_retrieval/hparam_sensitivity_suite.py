"""Hyperparameter sensitivity / grid-search suite for the reviewer rebuttal.

Reviewer objection: the coefficients in Eq. 3 (beta_1..3, the Adaptive Baseline
ranking score) and Eq. 4 (w_1..4, the Adaptive Transferability seed score) are
"set by hand" with "no grid search and no sensitivity analysis", and the chosen
values "are not justified".

Best-practice response, on the PAPER'S ACTUAL METHODS
(select_adaptive_baseline / select_adaptive_transferability), under the same
strict leave-one-target-out (LOTO) protocol, the same [25,50,100,200] batch
schedule, and the same 4000-query budget used for Table 3:

  A. reproduce -> reproduce Table 3 (all 5 methods) as a validity check.
  B. beta      -> full grid + one-at-a-time sensitivity of Eq. 3 coefficients.
  C. w         -> full grid + one-at-a-time sensitivity of Eq. 4 coefficients.
  D. eq5       -> sensitivity of Eq. 5 annealing constants (alpha4, c, decay).
  E. nested    -> nested LOTO hyperparameter selection: for each held-out target
                  pick the best config using ONLY the other 7 targets, evaluate
                  on the held-out target; compare oracle-per-fold tuning against
                  the fixed hand-set values.

Deterministic. Raw frame cached to parquet (parsed once). Config x target tasks
run in parallel across CPU cores (multiprocessing, fork -> COW-shared data).
"""
from __future__ import annotations

import argparse
import ast
from itertools import product
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jailbreak_retrieval.utils.data import (  # noqa: E402
    BASE_FEATURES,
    HF_DATASET,
    TRANSFER_FEATURES,
    load_deepseek_response_dataframe,
    load_response_frame,
    load_response_frame_from_hf,
    load_response_level_dataframe,
    prepare_target_dataset,
)
from jailbreak_retrieval.utils.methods import (  # noqa: E402
    select_adaptive_baseline,
    select_adaptive_transferability,
    select_baseline_random,
    select_static_top,
    select_weighted_success_sampling,
    _transfer_prior_score,
)

PAPER_BETA = (0.45, 0.25, 0.30)          # Eq. 3
PAPER_W = (0.40, 0.30, 0.15, 0.15)       # Eq. 4
PAPER_ALPHA4 = 0.35                      # Eq. 5
PAPER_C = 0.06                           # Eq. 5
PAPER_DECAY = (0.28, 0.20)               # Eq. 5 alpha1 decay bounds

# Cache dir for the parsed response frame. Env-overridable; defaults to a
# relative ./cache under the package root (no absolute / scratch paths in code).
CACHE = Path(os.environ.get("JAILBREAK_RETRIEVAL_CACHE", REPO_ROOT / "cache"))
OUT = REPO_ROOT / "jailbreak_retrieval_params_grid_search" / "rebuttal_suite"

# Populated in the parent before the pool forks; inherited by workers via COW.
PREP: dict = {}
BUDGET = 4000


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# Parameterized paper methods (D8 cleanup). The canonical implementations live
# in jailbreak_retrieval.utils.methods and now accept the Eq.3/4/5 coefficients as
# arguments (defaults = paper values), so these are thin delegators kept for the
# public names imported by feature_ablation.py / pool_size_sweep.py. They are
# numerically identical to the canonical methods at the paper coefficients, so
# Table-3 parity is exact.
# --------------------------------------------------------------------------
def select_adaptive_baseline_p(prepared_df, model_cols, batch_size, budget,
                               target_model_id, beta=PAPER_BETA):
    return select_adaptive_baseline(prepared_df, model_cols, batch_size, budget,
                                    target_model_id, beta=beta)


def _transfer_prior_score_p(pool_df, w):
    return _transfer_prior_score(pool_df, w)


def select_adaptive_transferability_p(prepared_df, model_cols, batch_size, budget,
                                      target_model_id, w=PAPER_W,
                                      alpha4=PAPER_ALPHA4, c_ucb=PAPER_C,
                                      decay=PAPER_DECAY):
    return select_adaptive_transferability(prepared_df, model_cols, batch_size,
                                           budget, target_model_id, w=w,
                                           alpha4=alpha4, c_ucb=c_ucb, decay=decay)


# --------------------------------------------------------------------------
# Data + caching
# --------------------------------------------------------------------------
def _load_both_mutator_deepseek():
    """Exact Table-3 data pool: Gemma-3 + Mistral-Nemo mutator attacks, DeepSeek
    judge, harmful success mode. Reproduces Table 3 (Adaptive Transferability
    89.83, Adaptive Baseline 86.93, Static Top 83.82, ...).

    Uses the vendored loader (utils/analysis.py) so no external repo is imported;
    it still reads raw sources from LLM_EVAL_ROOT when actually rebuilding."""
    return load_deepseek_response_dataframe(success_mode="harmful")


def get_data(data_source="both", data_path=None):
    """Return (response_df, target_models).

    Priority:
      1. ``data_path`` (or env JAILBREAK_RETRIEVAL_DATA) -> read that pre-built
         response-level parquet directly (fully standalone, offline).
      2. cached parquet under CACHE (env JAILBREAK_RETRIEVAL_CACHE, default ./cache).
      3. if LLM_EVAL_ROOT is set -> rebuild from raw via the vendored loaders.
      4. default -> build from the published Hugging Face dataset (DeepSeek judge),
         caching the frame under CACHE for reuse.
    """
    data_path = data_path or os.environ.get("JAILBREAK_RETRIEVAL_DATA")
    if data_path:
        log(f"[data] loading pre-built response frame from {data_path}")
        df, targets = load_response_frame(data_path)
        log(f"[data] source=parquet rows={len(df)} targets={len(targets)}")
        return df, targets

    CACHE.mkdir(parents=True, exist_ok=True)
    tag = "both_deepseek" if data_source == "both" else "mistral_deepseek"
    parq = CACHE / f"response_df_{tag}.parquet"
    tgt = CACHE / f"targets_{tag}.txt"
    if parq.exists() and tgt.exists():
        log(f"[data] loading cached response_df ({tag}) from {parq}")
        df = pd.read_parquet(parq)
        targets = [t for t in tgt.read_text().split("\n") if t]
    elif os.environ.get("LLM_EVAL_ROOT"):
        # Optional: rebuild from the original raw sources (authors only).
        log(f"[data] building response_df ({tag}) from raw (one-time, minutes)...")
        t0 = time.time()
        if data_source == "both":
            df, targets = _load_both_mutator_deepseek()
        else:
            df, targets = load_response_level_dataframe()
        log(f"[data] built in {time.time()-t0:.0f}s; caching to parquet")
        df.to_parquet(parq)
        tgt.write_text("\n".join(targets))
    else:
        # Default standalone path: read from the published Hugging Face dataset.
        log(f"[data] loading from Hugging Face dataset {HF_DATASET} (gated; caching to {CACHE})...")
        df, targets = load_response_frame_from_hf(data_source=data_source, cache_dir=CACHE)
    log(f"[data] source={tag} rows={len(df)} targets={len(targets)}")
    return df, targets


# --------------------------------------------------------------------------
# Parallel task execution. A task = (key, kind, param, target, batches).
# Worker returns rows of final ASR per batch size. PREP/BUDGET are inherited.
# --------------------------------------------------------------------------
def run_task(task):
    key, kind, param, target, batches = task
    p = PREP[target]
    budget = min(BUDGET, len(p["prepared_df"]))
    pdf, mcols = p["prepared_df"], p["model_cols"]
    rows = []
    for bs in batches:
        if kind == "random":
            sel = select_baseline_random(pdf, mcols, bs, budget, target)
        elif kind == "weighted":
            sel = select_weighted_success_sampling(pdf, mcols, bs, budget, target)
        elif kind == "static":
            sel = select_static_top(pdf, mcols, bs, budget, target)
        elif kind == "beta":
            sel = select_adaptive_baseline_p(pdf, mcols, bs, budget, target, beta=param)
        elif kind == "w":
            sel = select_adaptive_transferability_p(pdf, mcols, bs, budget, target, w=param)
        elif kind == "eq5":
            sel = select_adaptive_transferability_p(pdf, mcols, bs, budget, target,
                                                    w=PAPER_W, **param)
        else:
            raise ValueError(kind)
        rows.append((key, kind, target, bs, float(sel["target_success"].mean()), len(sel)))
    return rows


def run_parallel(tasks, nproc):
    log(f"[run] {len(tasks)} tasks on {nproc} procs")
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=nproc) as pool:
        results = []
        done = 0
        for res in pool.imap_unordered(run_task, tasks, chunksize=1):
            results.extend(res)
            done += 1
            if done % 25 == 0 or done == len(tasks):
                log(f"[run] {done}/{len(tasks)} tasks ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(results, columns=["key", "kind", "target", "batch_size",
                                        "final_asr", "n"])
    return df


# --------------------------------------------------------------------------
# Config grids (same neighborhood the user already searched; simplex-valid).
# --------------------------------------------------------------------------
def beta_configs():
    b1s = [0.35, 0.40, 0.45, 0.50, 0.55]
    b2s = [0.15, 0.20, 0.25, 0.30, 0.35]
    out = set()
    for b1, b2 in product(b1s, b2s):
        b3 = round(1.0 - b1 - b2, 6)
        if b3 < -1e-9:
            continue
        out.add((round(b1, 4), round(b2, 4), round(max(0.0, b3), 4)))
    out.add(PAPER_BETA)
    return sorted(out)


def w_configs():
    w1s = [0.30, 0.35, 0.40, 0.45, 0.50]
    w2s = [0.20, 0.25, 0.30, 0.35, 0.40]
    w3s = [0.10, 0.15, 0.20]
    out = set()
    for w1, w2, w3 in product(w1s, w2s, w3s):
        w4 = round(1.0 - w1 - w2 - w3, 6)
        if w4 < -1e-9:
            continue
        out.add((round(w1, 4), round(w2, 4), round(w3, 4), round(max(0.0, w4), 4)))
    out.add(PAPER_W)
    return sorted(out)


def eq5_configs():
    g = []
    for a4 in [0.25, 0.30, 0.35, 0.40, 0.45]:
        g.append((f"alpha4={a4}", {"alpha4": a4}))
    for cc in [0.0, 0.03, 0.06, 0.12]:
        g.append((f"c={cc}", {"c_ucb": cc}))
    for hi in [0.24, 0.28, 0.32]:
        g.append((f"decay_hi={hi}", {"decay": (hi, 0.20)}))
    return g


def target_level(df_res):
    """Mean final ASR over the batch schedule, per (key,target)."""
    return df_res.groupby(["key", "kind", "target"])["final_asr"].mean().reset_index()


# --------------------------------------------------------------------------
def main():
    global PREP, BUDGET, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="ABCDE")
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--batches", default="25,50,100,200")
    ap.add_argument("--nproc", type=int, default=min(120, mp.cpu_count() - 4))
    ap.add_argument("--data", default="both", choices=["both", "mistral"],
                    help="'both' = Gemma-3+Mistral-Nemo pool w/ DeepSeek judge (Table-3 config); "
                         "'mistral' = Mistral-Nemo-only pool.")
    ap.add_argument("--data-path", default=None,
                    help="Path to a pre-built response-level parquet (overrides --data / "
                         "cache; fully standalone). Env: JAILBREAK_RETRIEVAL_DATA.")
    args = ap.parse_args()
    BUDGET = args.budget
    batches = tuple(int(x) for x in args.batches.split(","))
    stages = set(args.stages.upper())
    if args.data == "mistral":
        OUT = OUT.parent / "rebuttal_suite_mistral_only"
    OUT.mkdir(parents=True, exist_ok=True)

    df, targets = get_data(args.data, data_path=args.data_path)
    log("[prep] preparing per-target datasets (cached in RAM before fork)...")
    t0 = time.time()
    for t in targets:
        p = prepare_target_dataset(df, targets, t)
        PREP[t] = {"prepared_df": p.prepared_df, "model_cols": p.model_cols}
    log(f"[prep] done in {time.time()-t0:.0f}s")

    log("\n[inventory] per-target pool / budget / hidden ASR:")
    inv = []
    for t in targets:
        pdf = PREP[t]["prepared_df"]
        b = min(BUDGET, len(pdf))
        hidden = 100 * pdf["target_success"].mean()
        inv.append({"target": t, "pool": len(pdf), "budget": b, "hidden_asr_pct": hidden})
        log(f"  {t:40s} pool={len(pdf):6d} budget={b:5d} hidden_ASR={hidden:.1f}%")
    pd.DataFrame(inv).to_csv(OUT / "inventory.csv", index=False)

    # Build the full task list up front (dedupe heavy configs across stages).
    tasks = []
    if "A" in stages:
        for kind in ["random", "weighted", "static"]:
            for t in targets:
                tasks.append((f"A::{kind}", kind, None, t, batches))
        # Stage A reproduces Table 3 = ALL 5 methods. The two adaptive methods
        # must be scheduled at the paper coefficients even when B/C/E are not
        # requested, otherwise A_reproduce_table3.csv silently drops them. The
        # (key, target) dedupe below collapses any overlap with B/C/E.
        for t in targets:
            tasks.append((f"beta::{PAPER_BETA}", "beta", PAPER_BETA, t, batches))
            tasks.append((f"w::{PAPER_W}", "w", PAPER_W, t, batches))
    beta_cfgs = beta_configs()
    w_cfgs = w_configs()
    if "B" in stages or "E" in stages:
        for c in beta_cfgs:
            for t in targets:
                tasks.append((f"beta::{c}", "beta", c, t, batches))
    if "C" in stages or "E" in stages:
        for c in w_cfgs:
            for t in targets:
                tasks.append((f"w::{c}", "w", c, t, batches))
    if "D" in stages:
        for label, kw in eq5_configs():
            for t in targets:
                tasks.append((f"eq5::{label}", "eq5", kw, t, batches))
    # dedupe
    seen = set(); uniq = []
    for tk in tasks:
        sig = (tk[0], tk[3])
        if sig in seen:
            continue
        seen.add(sig); uniq.append(tk)
    tasks = uniq

    res = run_parallel(tasks, args.nproc)
    res.to_csv(OUT / "raw_task_asr.csv", index=False)
    tl = target_level(res)
    tl.to_csv(OUT / "target_level_asr.csv", index=False)

    # ---- Stage A report ----
    if "A" in stages:
        log("\n===== STAGE A: reproduce Table 3 (mean over 8 targets & batch schedule) =====")
        names = {"A::random": "Random Sampling", "A::weighted": "Weighted Success",
                 "A::static": "Static Top-4000",
                 f"beta::{PAPER_BETA}": "Adaptive Baseline (Eq.3 paper)",
                 f"w::{PAPER_W}": "Adaptive Transferability (Eq.4/5 paper)"}
        arows = []
        for key, label in names.items():
            sub = tl[tl.key == key]
            if sub.empty:
                continue
            arows.append({"method": label, "mean_asr": 100 * sub.final_asr.mean(),
                          "sd_asr": 100 * sub.final_asr.std()})
        adf = pd.DataFrame(arows)
        rand = adf.loc[adf.method == "Random Sampling", "mean_asr"]
        base = float(rand.iloc[0]) if len(rand) else 0.0
        adf["delta_vs_random"] = adf["mean_asr"] - base
        adf.to_csv(OUT / "A_reproduce_table3.csv", index=False)
        for _, r in adf.iterrows():
            log(f"[A] {r.method:42s} {r.mean_asr:5.2f} +/- {r.sd_asr:5.2f}   d_rand={r.delta_vs_random:+.2f}")

    # ---- Stage B report ----
    if "B" in stages:
        log("\n===== STAGE B: Eq. 3 (beta) grid =====")
        b = tl[tl.kind == "beta"].groupby("key").agg(
            mean_asr=("final_asr", "mean"), sd_asr=("final_asr", "std")).reset_index()
        b[["beta_1", "beta_2", "beta_3"]] = pd.DataFrame(
            [ast.literal_eval(k.split("::")[1]) for k in b.key], index=b.index)
        b["is_paper"] = b.key == f"beta::{PAPER_BETA}"
        b = b.sort_values("mean_asr", ascending=False).reset_index(drop=True)
        b.to_csv(OUT / "B_beta_grid.csv", index=False)
        pr = b[b.is_paper].iloc[0]
        rank = int((b.mean_asr > pr.mean_asr).sum()) + 1
        log(f"[B] configs={len(b)}  paper beta={PAPER_BETA} mean={100*pr.mean_asr:.2f} "
            f"rank={rank}/{len(b)}  best={100*b.mean_asr.max():.2f} "
            f"worst={100*b.mean_asr.min():.2f}  spread={100*(b.mean_asr.max()-b.mean_asr.min()):.2f}pp "
            f"std_across_cfgs={100*b.mean_asr.std():.3f}pp")

    # ---- Stage C report ----
    if "C" in stages:
        log("\n===== STAGE C: Eq. 4 (w) grid =====")
        c = tl[tl.kind == "w"].groupby("key").agg(
            mean_asr=("final_asr", "mean"), sd_asr=("final_asr", "std")).reset_index()
        c[["w1", "w2", "w3", "w4"]] = pd.DataFrame(
            [ast.literal_eval(k.split("::")[1]) for k in c.key], index=c.index)
        c["is_paper"] = c.key == f"w::{PAPER_W}"
        c = c.sort_values("mean_asr", ascending=False).reset_index(drop=True)
        c.to_csv(OUT / "C_w_grid.csv", index=False)
        pr = c[c.is_paper].iloc[0]
        rank = int((c.mean_asr > pr.mean_asr).sum()) + 1
        log(f"[C] configs={len(c)}  paper w={PAPER_W} mean={100*pr.mean_asr:.2f} "
            f"rank={rank}/{len(c)}  best={100*c.mean_asr.max():.2f} "
            f"worst={100*c.mean_asr.min():.2f}  spread={100*(c.mean_asr.max()-c.mean_asr.min()):.2f}pp "
            f"std_across_cfgs={100*c.mean_asr.std():.3f}pp")

    # ---- Stage D report ----
    if "D" in stages:
        log("\n===== STAGE D: Eq. 5 annealing sensitivity =====")
        d = tl[tl.kind == "eq5"].groupby("key").agg(
            mean_asr=("final_asr", "mean"), sd_asr=("final_asr", "std")).reset_index()
        d["label"] = d.key.str.split("::").str[1]
        d = d.sort_values("label").reset_index(drop=True)
        d.to_csv(OUT / "D_eq5_sensitivity.csv", index=False)
        for _, r in d.iterrows():
            log(f"[D] {r.label:16s} mean={100*r.mean_asr:.2f} sd={100*r.sd_asr:.2f}")

    # ---- Stage E: nested LOTO selection ----
    if "E" in stages:
        log("\n===== STAGE E: nested LOTO hyperparameter selection =====")
        for label, kind, paper_cfg in [("beta", "beta", PAPER_BETA), ("w", "w", PAPER_W)]:
            sub = tl[tl.kind == kind]
            # matrix cfg_key -> {target -> asr}
            mat = {}
            for key, g in sub.groupby("key"):
                mat[key] = dict(zip(g.target, g.final_asr))
            cfg_keys = list(mat.keys())
            paper_key = f"{kind}::{paper_cfg}"
            nrows = []
            for held in targets:
                others = [t for t in targets if t != held]
                best_key = max(cfg_keys,
                               key=lambda k: np.mean([mat[k][o] for o in others]))
                nrows.append({"held_out": held, "selected_cfg": best_key.split("::")[1],
                              "nested_asr": mat[best_key][held],
                              "paper_asr": mat[paper_key][held]})
            nd = pd.DataFrame(nrows)
            nd.to_csv(OUT / f"E_nested_{label}.csv", index=False)
            gain = 100 * (nd.nested_asr.mean() - nd.paper_asr.mean())
            log(f"[E:{label}] nested-select mean ASR={100*nd.nested_asr.mean():.2f}  "
                f"fixed-paper mean ASR={100*nd.paper_asr.mean():.2f}  "
                f"per-fold-tuning gain={gain:+.2f}pp")

    log(f"\n[done] outputs in {OUT}")


if __name__ == "__main__":
    main()
