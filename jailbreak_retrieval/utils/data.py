from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import pandas as pd

# Raw-data loaders are vendored into utils/analysis.py so this package imports
# and runs standalone (no dependency on the external llm_evaluation repo). The
# vendored code only reaches for the external repo when *rebuilding* a frame
# from raw sources; the LLM_EVAL_ROOT it uses is env-overridable. The standalone
# path (load_response_frame, below) reads a pre-built response-level parquet and
# touches none of it.
from .analysis import (
    DEFAULT_ATTACKS_PATH,
    DEFAULT_JUDGE_ROOT,
    DEFAULT_TARGET_RESPONSES_ROOT,
    LLM_EVAL_ROOT,
    build_analysis_artifacts,
    load_deepseek_response_dataframe,  # noqa: F401  (re-exported for callers)
)

ROOT_DIR = LLM_EVAL_ROOT
ATTACKS_PATH = DEFAULT_ATTACKS_PATH
TARGET_ROOT = DEFAULT_TARGET_RESPONSES_ROOT
JUDGE_ROOT = DEFAULT_JUDGE_ROOT
SUCCESS_MODE = "harmful"
TOTAL_BUDGET = 4000
BATCH_SIZES = [25, 50, 100, 200]
FOCUS_TARGETS = [
    "mistralai/Mistral-Nemo-Instruct-2407",
    "openai/gpt-oss-20b",
]
BASE_FEATURES = [
    "source_target_model",
    "attack_style_tag",
    "risk_category_tag",
    "planner_action",
]
TRANSFER_FEATURES = [
    "train_success_count",
    "strict_success_count",
    "partial_success_count",
    "max_judge_severity",
    "success_profile",
]
PAIR_FEATURES = [
    ("source_target_model", "attack_style_tag"),
    ("source_target_model", "risk_category_tag"),
    ("attack_style_tag", "risk_category_tag"),
]


@dataclass(frozen=True)
class PreparedTargetData:
    target_model_id: str
    prepared_df: pd.DataFrame
    model_cols: list[str]


def load_response_level_dataframe(
    success_mode: str = SUCCESS_MODE,
) -> tuple[pd.DataFrame, list[str]]:
    artifacts = build_analysis_artifacts(
        attacks_path=ATTACKS_PATH,
        target_root=TARGET_ROOT,
        judge_root=JUDGE_ROOT,
    )
    response_df = artifacts.merged.copy()
    response_df = response_df.loc[
        response_df["judge_label_known"] & response_df["target_status"].eq("success")
    ].copy()
    response_df["harmful_target_success"] = (
        response_df["harmful_attack_succeeded"].eq(True).astype(int)
    )
    response_df["strict_target_success"] = (
        response_df["strict_attack_succeeded"].eq(True).astype(int)
    )
    response_df["judge_severity"] = 2 * response_df["strict_target_success"] + (
        response_df["harmful_target_success"] - response_df["strict_target_success"]
    )
    if success_mode == "harmful":
        response_df["target_success"] = response_df["harmful_target_success"]
    elif success_mode == "strict":
        response_df["target_success"] = response_df["strict_target_success"]
    else:
        raise ValueError(f"Unsupported success_mode: {success_mode}")

    target_models = sorted(response_df["target_model_id"].dropna().unique().tolist())
    return response_df, target_models


# Default published dataset (gated) that the retrieval scripts read by default.
HF_DATASET = "NASK-PIB/a-art-jailbreaks"


def _derive_success_columns(
    response_df: pd.DataFrame,
    success_mode: str = SUCCESS_MODE,
    source: str = "frame",
) -> tuple[pd.DataFrame, list[str]]:
    """Keep known-judge successful-target rows and (re)compute ``target_success``
    for ``success_mode``. Shared by the parquet and Hugging Face loaders so both
    derive numbers identically."""
    if "judge_label_known" in response_df.columns and "target_status" in response_df.columns:
        response_df = response_df.loc[
            response_df["judge_label_known"] & response_df["target_status"].eq("success")
        ].copy()
    else:
        response_df = response_df.copy()

    if {"harmful_attack_succeeded", "strict_attack_succeeded"}.issubset(response_df.columns):
        response_df["harmful_target_success"] = (
            response_df["harmful_attack_succeeded"].eq(True).astype(int)
        )
        response_df["strict_target_success"] = (
            response_df["strict_attack_succeeded"].eq(True).astype(int)
        )
        response_df["judge_severity"] = 2 * response_df["strict_target_success"] + (
            response_df["harmful_target_success"] - response_df["strict_target_success"]
        )
        if success_mode == "harmful":
            response_df["target_success"] = response_df["harmful_target_success"]
        elif success_mode == "strict":
            response_df["target_success"] = response_df["strict_target_success"]
        else:
            raise ValueError(f"Unsupported success_mode: {success_mode}")
    elif "target_success" not in response_df.columns:
        raise ValueError(
            f"Response frame ({source}) has neither the raw judge-label columns "
            "(harmful_attack_succeeded/strict_attack_succeeded) nor a precomputed "
            "target_success column; cannot derive success labels."
        )

    target_models = sorted(response_df["target_model_id"].dropna().unique().tolist())
    return response_df, target_models


def load_response_frame(
    path: str | Path,
    success_mode: str = SUCCESS_MODE,
) -> tuple[pd.DataFrame, list[str]]:
    """Standalone loader: read a pre-built *response-level* frame from parquet.

    ``path`` points at a response-level parquet (one row per (attack, target)
    trial), e.g. a frame cached from a previous run. Rows are filtered and the
    ``target_success`` labels are (re)computed for ``success_mode``.
    """
    return _derive_success_columns(pd.read_parquet(path), success_mode, source=str(path))


# Maps the ``--data`` selector to the mutator whose attacks form the pool.
_DATA_SOURCE_MUTATORS = {
    "both": None,  # both canonical mutators (gemma-3-12b + Mistral-Nemo)
    "mistral": "mistralai/Mistral-Nemo-Instruct-2407",
}


def load_response_frame_from_hf(
    dataset: str = HF_DATASET,
    success_mode: str = SUCCESS_MODE,
    data_source: str = "both",
    judge: str = "deepseek",
    cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the response-level frame directly from the published dataset.

    Joins the ``transfer_scores`` config (one row per (attack, target) with the
    per-judge verdicts) with ``attacks`` (attack metadata), and derives the
    success labels from the requested ``judge`` (default ``deepseek`` — the judge
    behind the published retrieval numbers) exactly as the raw loader does:
    ``strict`` = label UNSAFE; ``harmful`` = label in {PARTIAL, UNSAFE}. Rows
    whose judge label is missing are dropped (``judge_label_known``).

    ``data_source`` selects the mutator pool (``both`` or ``mistral``). The
    dataset is gated: authenticate first (``huggingface-cli login`` or set the
    ``HF_TOKEN`` environment variable). If ``cache_dir`` is given, the built
    frame is cached there (keyed on dataset/judge/data_source) so later runs
    skip the download.
    """
    if data_source not in _DATA_SOURCE_MUTATORS:
        raise ValueError(
            f"Unsupported data_source: {data_source!r} (use 'both' or 'mistral')."
        )

    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe = dataset.replace("/", "__")
        cache_file = cache_dir / f"response_df_hf_{safe}_{judge}_{data_source}.parquet"
        if cache_file.exists():
            return _derive_success_columns(
                pd.read_parquet(cache_file), success_mode, source=str(cache_file)
            )

    from datasets import load_dataset

    scores = load_dataset(dataset, "transfer_scores", split="train").to_pandas()
    attacks = load_dataset(dataset, "attacks", split="train").to_pandas()

    mutator = _DATA_SOURCE_MUTATORS[data_source]
    if mutator is not None and "mutator_model_id" in attacks.columns:
        attacks = attacks.loc[attacks["mutator_model_id"] == mutator]

    meta_cols = [
        "attack_id",
        "source_target_model",
        "attack_style_tag",
        "risk_category_tag",
        "planner_action",
    ]
    attacks = attacks[[c for c in meta_cols if c in attacks.columns]]
    # inner join so the data_source (mutator) filter propagates to the trials
    df = scores.merge(attacks, on="attack_id", how="inner")

    label_col, sev_col = f"{judge}_label", f"{judge}_severity"
    if label_col not in df.columns:
        raise ValueError(f"Dataset has no '{label_col}' column for judge={judge!r}.")
    label = df[label_col]
    # Mirror the raw loader's mapping (utils/analysis.py): a label that is not one
    # of SAFE/PARTIAL/UNSAFE is marked unknown and dropped in _derive_success_columns.
    df["strict_attack_succeeded"] = label.map({"UNSAFE": True, "PARTIAL": False, "SAFE": False})
    df["harmful_attack_succeeded"] = label.map({"UNSAFE": True, "PARTIAL": True, "SAFE": False})
    df["judge_label_known"] = label.isin(["SAFE", "PARTIAL", "UNSAFE"])
    df["risk_severity_score"] = df[sev_col] if sev_col in df.columns else 0.0
    df["target_status"] = "success"

    if cache_file is not None:
        df.to_parquet(cache_file, index=False)
    return _derive_success_columns(df, success_mode, source=f"{dataset}:{judge}:{data_source}")


def prepare_target_dataset(
    response_df: pd.DataFrame,
    target_models: list[str],
    target_model_id: str,
) -> PreparedTargetData:
    train_df = response_df.loc[
        response_df["source_target_model"].ne(target_model_id)
        & response_df["target_model_id"].ne(target_model_id)
    ].copy()
    eval_df = response_df.loc[
        response_df["source_target_model"].ne(target_model_id)
        & response_df["target_model_id"].eq(target_model_id),
        ["attack_id", "target_success"],
    ].drop_duplicates(subset=["attack_id"])

    summary_df = (
        train_df.groupby("attack_id", dropna=False)
        .agg(
            source_target_model=("source_target_model", "first"),
            attack_style_tag=("attack_style_tag", "first"),
            risk_category_tag=("risk_category_tag", "first"),
            planner_action=("planner_action", "first"),
            train_model_count=("target_model_id", "nunique"),
            train_success_count=("target_success", "sum"),
            train_success_rate=("target_success", "mean"),
            strict_success_count=("strict_target_success", "sum"),
            strict_success_rate=("strict_target_success", "mean"),
            mean_judge_severity=("judge_severity", "mean"),
            max_judge_severity=("judge_severity", "max"),
            max_risk_severity_score=("risk_severity_score", "max"),
        )
        .reset_index()
    )
    model_wide_df = train_df.pivot_table(
        index="attack_id",
        columns="target_model_id",
        values="target_success",
        aggfunc="first",
    )
    model_wide_df = model_wide_df.rename(
        columns={column: f"model__{column}" for column in model_wide_df.columns}
    ).reset_index()

    prepared_df = summary_df.merge(model_wide_df, on="attack_id", how="left").merge(
        eval_df,
        on="attack_id",
        how="inner",
    )
    prepared_df = prepared_df.loc[
        prepared_df["train_model_count"].eq(len(target_models) - 1)
    ].copy()
    prepared_df["prior_mean"] = (prepared_df["train_success_count"] + 1.0) / (
        prepared_df["train_model_count"] + 2.0
    )
    prepared_df["partial_success_count"] = (
        prepared_df["train_success_count"] - prepared_df["strict_success_count"]
    )
    prepared_df["partial_success_rate"] = (
        prepared_df["partial_success_count"] / prepared_df["train_model_count"]
    )
    prepared_df["success_profile"] = (
        "harmful="
        + prepared_df["train_success_count"].astype(str)
        + "|strict="
        + prepared_df["strict_success_count"].astype(str)
    )

    model_cols = [
        column for column in prepared_df.columns if column.startswith("model__")
    ]
    for column in model_cols:
        prepared_df[column] = prepared_df[column].fillna(0).astype(int)

    for left, right in PAIR_FEATURES:
        prepared_df[f"{left}__{right}"] = (
            prepared_df[left].astype(str) + "||" + prepared_df[right].astype(str)
        )
    prepared_df["model_pattern"] = (
        prepared_df[model_cols].astype(str).agg("".join, axis=1)
    )

    return PreparedTargetData(
        target_model_id=target_model_id,
        prepared_df=prepared_df,
        model_cols=model_cols,
    )
