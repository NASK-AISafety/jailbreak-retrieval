# Jailbreak Retrieval — metadata-aware reuse of adversarial attacks

This package implements the **jailbreak retrieval** experiments from the paper: given a
large pool of previously-generated jailbreak attacks with per-attack transfer
metadata, select a budget-limited subset for a *held-out* target model so as to
maximise attack success rate (ASR). All experiments use a strict
**leave-one-target-out (LOTO)** protocol, a fixed **4000-query budget**, and a
`[25, 50, 100, 200]` batch schedule (the Table-3 headline setting).

The retrieval signals compared are:

- **Random** (floor) and **Weighted-success sampling** (weak baselines).
- **Static Top-4000** — rank by historical (Bayesian-smoothed) success only.
- **Adaptive Baseline** (Eq. 3) — online posterior over base features + source models.
- **Adaptive Transferability** (Eq. 4/5) — adds transfer features + a UCB exploration
  bonus with an annealing schedule.

## Package layout

```
jailbreak-retrieval/             # repo root
  README.md · LICENSE · requirements.txt · .gitignore
  jailbreak_retrieval/           # the importable package
    __init__.py
    feature_ablation.py          # metadata feature ablation (paper ablation table)
    pool_size_sweep.py           # pool-size ablation at fixed budget
    hparam_sensitivity_suite.py  # Table 3 reproduction (Stage A) + Eq.3/4/5 grids & sensitivity (B–E)
    utils/
      __init__.py
      analysis.py    # vendored-verbatim raw-data loaders (no external repo dependency)
      data.py        # load + LOTO-prepare the response-level table (+ HF / parquet loaders)
      methods.py     # the selection methods (Static/Adaptive/etc.) + METHOD_REGISTRY
      benchmark.py   # run_candidate_benchmark: full sweep + CSV outputs (library API)
      reporting.py   # matplotlib/seaborn summary plots (library API)
```

The `feature_ablation.py`, `pool_size_sweep.py`, and `hparam_sensitivity_suite.py`
scripts each add the repo root to `sys.path` and import `jailbreak_retrieval.utils.*`,
so they can be run directly.

## Install

```bash
git clone https://github.com/NASK-AISafety/jailbreak-retrieval
cd jailbreak-retrieval
python -m pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `pyarrow`, `matplotlib`, `seaborn`, and
`datasets` / `huggingface_hub` (for the default Hugging Face data path).

## Input data

The selection code operates on a **response-level table**: one row per
(attack, target-model) trial, with the judge label plus attack metadata
(source model, attack style, risk category, planner action, severity).
`jailbreak_retrieval/utils/data.py:prepare_target_dataset` turns this into the
per-target leave-one-target-out (LOTO) training pool.

By default the scripts build this table directly from the published dataset
[`NASK-PIB/a-art-jailbreaks`](https://huggingface.co/datasets/NASK-PIB/a-art-jailbreaks),
joining its `transfer_scores` and `attacks` configs. The dataset is gated:
request access and authenticate first (`huggingface-cli login`, or set the
`HF_TOKEN` environment variable). The built frame is cached under the cache dir
(env `JAILBREAK_RETRIEVAL_CACHE`, default `./cache`) so later runs skip the download.

```bash
python jailbreak_retrieval/feature_ablation.py        # reads the HF dataset by default
```

### Use a local parquet instead

To run fully offline, point `--data-path` (or env `JAILBREAK_RETRIEVAL_DATA`) at a
pre-built response-level parquet — one row per (attack, target) trial containing
`attack_id`, `target_model_id`, `source_target_model`, the metadata tags, and
either `target_success` or the raw `harmful_attack_succeeded` /
`strict_attack_succeeded` columns (from which `target_success` / `judge_severity`
are derived identically):

```bash
python jailbreak_retrieval/feature_ablation.py --data-path /path/to/response_df.parquet
```

### Rebuild from raw sources (optional)

Setting `LLM_EVAL_ROOT` to a local copy of the original raw sources rebuilds the
frame from scratch; omit it to use the HF dataset. No absolute or scratch paths
are hardcoded anywhere in the code — all data, cache, and root locations are CLI
args or environment variables.

## Judge provenance of the published numbers

The published jailbreak retrieval ASR numbers are graded by the
DeepSeek-R1-Distill-Qwen-32B judge used throughout the retrieval evaluation
(e.g. Random 61.3 / Historical 83.5 / Adaptive Transferability 89.8). The default
Hugging Face data path reproduces this DeepSeek-judged, harmful-success pool.

## Reproduce the tables

Run from the directory that contains the `jailbreak_retrieval/` package (outputs are
written under `jailbreak_retrieval_params_grid_search/`). The scripts read the HF
dataset by default; append `--data-path /path/to/response_df.parquet` to any
command to use a local parquet offline instead.

```bash
# Table 3 reproduction (validity check: ALL 5 methods, LOTO, budget 4000):
python jailbreak_retrieval/hparam_sensitivity_suite.py --stages A
#   -> jailbreak_retrieval_params_grid_search/rebuttal_suite/A_reproduce_table3.csv  (5 rows)

# Hyperparameter grids + sensitivity (Eq.3 beta, Eq.4 w, Eq.5 annealing, nested LOTO):
python jailbreak_retrieval/hparam_sensitivity_suite.py --stages BCDE
#   -> rebuttal_suite/{B_beta_grid,C_w_grid,D_eq5_sensitivity,E_nested_beta,E_nested_w}.csv

# Metadata feature ablation:
python jailbreak_retrieval/feature_ablation.py
#   -> jailbreak_retrieval_params_grid_search/feature_ablation/{ablation_per_target,ablation_summary}.csv
#      + ablation_table.tex

# Pool-size ablation (does retrieval need the full ~190k pool?):
python jailbreak_retrieval/pool_size_sweep.py
#   -> jailbreak_retrieval_params_grid_search/pool_size/{pool_size_per_target,pool_size_summary}.csv
```

Notes:
- Data source precedence: `--data-path` (env `JAILBREAK_RETRIEVAL_DATA`, a local
  parquet) > `LLM_EVAL_ROOT` raw rebuild (if set) > the HF dataset (default).
- The frame built from the HF dataset is cached under the cache dir
  (env `JAILBREAK_RETRIEVAL_CACHE`, default `./cache`); later runs reload the cache.
  With `--data-path` the parquet is read directly and no cache is written.
- All methods are deterministic (fixed seeds derived from the target id).
- Runs use `multiprocessing` (`fork`) across CPU cores; pass `--nproc` to bound it.
  CPU-only — no GPU or model serving is required.

## Citation

If you use this code or the accompanying dataset, please cite:

> M. Kowalczyk, M. Sendera, S. Cygert. *Data-Driven Red-Teaming: Reusing
> Adversarial Attacks via Metadata-Aware Retrieval.* Findings of EMNLP 2026,
> October 2026.

```bibtex
@inproceedings{kowalczyk2026aart,
  title     = {Data-Driven Red-Teaming: Reusing Adversarial Attacks via Metadata-Aware Retrieval},
  author    = {Kowalczyk, Mateusz and Sendera, Marcin and Cygert, Sebastian},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026},
  month     = oct
}
```

## License

Licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE).
