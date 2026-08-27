"""Metadata feature ablation for the reviewer rebuttal.

Reviewer: "conduct a metadata feature ablation ... compare retrieval using only
historical success, only risk category, only attack style, only severity, only
partial compliance, only source-model transfer profile, and combinations ...
This would demonstrate whether 'metadata-aware retrieval' adds value beyond
historical success ranking or simply selecting attacks from stronger source
models."

We isolate each metadata signal as a stand-alone ranker: score every candidate
attack by that signal, select the top-budget attacks, and measure ASR on the
held-out target under the same leave-one-target-out (LOTO) protocol, attack
pool, and budget as Table 3. We then add the combinations and the full method.

All numbers use budget=4000 and batch=25 (the Table-3 headline setting), so the
'historical success only' ranker equals Static Top-4000 and lands at the Table-3
value (83.8).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jailbreak_retrieval.utils.data import prepare_target_dataset  # noqa: E402
from jailbreak_retrieval.utils.methods import (  # noqa: E402
    select_baseline_random,
    select_model_mixture,
    select_static_top,
)
from jailbreak_retrieval.hparam_sensitivity_suite import (  # noqa: E402
    get_data,
    select_adaptive_baseline_p,
    select_adaptive_transferability_p,
    _transfer_prior_score_p,
    PAPER_BETA,
    PAPER_W,
)

OUT = REPO_ROOT / "jailbreak_retrieval_params_grid_search" / "feature_ablation"
PREP: dict = {}
BUDGET = 4000
BATCH = 25

# reviewer-facing label + how the signal is built
# kind: "static" (rank by a score column) or "online" (registry-style method)
ABLATIONS = [
    ("random",            "Random (floor)",                       "online"),
    ("hist_success",      "Historical success only (mu0)",        "static"),
    ("strict_rate",       "Strict-unsafe rate only",              "static"),
    ("partial_rate",      "Partial-compliance only",              "static"),
    ("severity",          "Judge severity only",                  "static"),
    ("risk_category",     "Risk category only",                   "static"),
    ("attack_style",      "Attack style only",                    "static"),
    ("source_model",      "Source-model only",                    "static"),
    ("categorical_all",   "All categorical metadata",             "static"),
    ("metadata_combined", "All success/severity metadata (Eq.4 static)", "static"),
    ("model_mixture",     "Source-model transfer profile (learned)",     "online"),
    ("adaptive_baseline", "Adaptive Baseline (Eq.3, online)",     "online"),
    ("adaptive_transfer", "Adaptive Transferability (Eq.4/5, online)",   "online"),
]

CATEGORICAL = ["source_target_model", "attack_style_tag", "risk_category_tag", "planner_action"]


def _group_prior(pool_df, feature):
    """Each candidate scored by the mean train (visible-model) success rate of
    its own feature value -> a leave-target-out prior from that feature alone."""
    return pool_df.groupby(feature, dropna=False)["train_success_rate"].transform("mean")


def static_score(pool_df, key):
    if key == "hist_success":
        return pool_df["prior_mean"]
    if key == "strict_rate":
        return pool_df["strict_success_rate"]
    if key == "partial_rate":
        return pool_df["partial_success_rate"]
    if key == "severity":
        return pool_df["mean_judge_severity"]
    if key == "risk_category":
        return _group_prior(pool_df, "risk_category_tag")
    if key == "attack_style":
        return _group_prior(pool_df, "attack_style_tag")
    if key == "source_model":
        return _group_prior(pool_df, "source_target_model")
    if key == "categorical_all":
        s = sum(_group_prior(pool_df, f) for f in CATEGORICAL) / len(CATEGORICAL)
        return s
    if key == "metadata_combined":
        return _transfer_prior_score_p(pool_df, PAPER_W)
    raise ValueError(key)


def select_static(pool_df, key, budget):
    # Feature-neutral tiebreak (attack_id only). Using prior_mean as a secondary
    # key would let historical success leak into every single-feature ranker:
    # low-cardinality features (10 risk categories, 7 source models) tie the
    # entire budget at one group score, so the tiebreak decides the whole
    # selection. attack_id is arbitrary w.r.t. success, so each feature is
    # isolated cleanly.
    score = static_score(pool_df, key)
    ranked = pool_df.assign(_score=score).sort_values(
        ["_score", "attack_id"], ascending=[False, True])
    return ranked.head(budget)


def run_task(task):
    key, kind, target = task
    p = PREP[target]
    pdf, mcols = p["prepared_df"], p["model_cols"]
    budget = min(BUDGET, len(pdf))
    if kind == "static":
        if key == "hist_success":
            # 'Historical success only' must EQUAL Static Top-4000 (docstring/caption
            # claim 83.8). Use the identical sort/tiebreak as select_static_top
            # (_sort_pool: prior_mean, then strict_rate, train_success_count,
            # max_risk_severity_score, attack_id) rather than the feature-neutral
            # attack_id-only tiebreak used to isolate the other single-feature rankers.
            sel = select_static_top(pdf, mcols, BATCH, budget, target)
        else:
            sel = select_static(pdf, key, budget)
    elif key == "random":
        sel = select_baseline_random(pdf, mcols, BATCH, budget, target)
    elif key == "model_mixture":
        sel = select_model_mixture(pdf, mcols, BATCH, budget, target)
    elif key == "adaptive_baseline":
        sel = select_adaptive_baseline_p(pdf, mcols, BATCH, budget, target, beta=PAPER_BETA)
    elif key == "adaptive_transfer":
        sel = select_adaptive_transferability_p(pdf, mcols, BATCH, budget, target, w=PAPER_W)
    else:
        raise ValueError(key)
    return (key, target, float(sel["target_success"].mean()), len(sel))


def main():
    global PREP
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="both", choices=["both", "mistral"])
    ap.add_argument("--data-path", default=None,
                    help="Path to a pre-built response-level parquet (overrides --data / "
                         "cache; fully standalone). Env: JAILBREAK_RETRIEVAL_DATA.")
    ap.add_argument("--nproc", type=int, default=min(120, mp.cpu_count() - 4))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df, targets = get_data(args.data, data_path=args.data_path)
    print("[prep] preparing per-target datasets...", flush=True)
    t0 = time.time()
    for t in targets:
        pr = prepare_target_dataset(df, targets, t)
        PREP[t] = {"prepared_df": pr.prepared_df, "model_cols": pr.model_cols}
    print(f"[prep] done in {time.time()-t0:.0f}s", flush=True)

    tasks = [(key, kind, t) for key, _lab, kind in ABLATIONS for t in targets]
    print(f"[run] {len(tasks)} tasks on {args.nproc} procs", flush=True)
    ctx = mp.get_context("fork")
    rows = []
    with ctx.Pool(args.nproc) as pool:
        done = 0
        for r in pool.imap_unordered(run_task, tasks, chunksize=1):
            rows.append(r); done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"[run] {done}/{len(tasks)}", flush=True)
    res = pd.DataFrame(rows, columns=["key", "target", "asr", "n"])
    res.to_csv(OUT / "ablation_per_target.csv", index=False)

    labels = {k: lab for k, lab, _ in ABLATIONS}
    order = {k: i for i, (k, _l, _kd) in enumerate(ABLATIONS)}
    agg = res.groupby("key").agg(mean_asr=("asr", "mean"), sd_asr=("asr", "std")).reset_index()
    agg["label"] = agg.key.map(labels)
    agg["order"] = agg.key.map(order)
    hist = float(agg.loc[agg.key == "hist_success", "mean_asr"].iloc[0])
    agg["delta_vs_hist"] = 100 * (agg["mean_asr"] - hist)
    agg["mean_asr"] *= 100; agg["sd_asr"] *= 100
    agg = agg.sort_values("order").reset_index(drop=True)
    agg[["label", "mean_asr", "sd_asr", "delta_vs_hist"]].to_csv(OUT / "ablation_summary.csv", index=False)

    print("\n===== METADATA FEATURE ABLATION (ASR %, LOTO, budget 4000, batch 25) =====", flush=True)
    print(f"{'feature set':46s} {'ASR':>6s} {'SD':>6s} {'d vs hist':>10s}", flush=True)
    for _, r in agg.iterrows():
        print(f"{r.label:46s} {r.mean_asr:6.1f} {r.sd_asr:6.1f} {r.delta_vs_hist:+10.1f}", flush=True)

    # LaTeX
    tex = ["\\begin{table}[t]\\centering\\small", "\\begin{tabular}{lcc}\\toprule",
           "Retrieval signal & Mean ASR $\\pm$ SD & $\\Delta$ vs.\\ hist.\\\\\\midrule"]
    for _, r in agg.iterrows():
        d = "--" if r.key == "hist_success" else f"{r.delta_vs_hist:+.1f}"
        tex.append(f"{r.label} & {r.mean_asr:.1f} $\\pm$ {r.sd_asr:.1f} & {d}\\\\")
    tex += ["\\bottomrule\\end{tabular}",
            "\\caption{Metadata feature ablation under the Table-3 protocol (LOTO, "
            "8 held-out targets, budget 4000). Each single-feature ranker scores candidates "
            "by that signal alone and selects the top 4000. `Historical success only' equals "
            "Static Top-4000.}", "\\label{tab:ablation}\\end{table}"]
    (OUT / "ablation_table.tex").write_text("\n".join(tex))
    print(f"\n[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
