"""Vendored raw-data loaders for the jailbreak retrieval release.

To make this package import and run *standalone* (no dependency on the external
``llm_evaluation`` repository), the functions that build the response-level
DataFrame from the raw attack / target-response / judge files are copied here
**verbatim** from their original locations, so the numbers they produce are
byte-for-byte identical to the published run:

  * ``open_text`` / ``iter_text_lines`` / ``iter_jsonl``
        <- llm_evaluation/judge_existing_responses/io_utils.py
  * ``build_analysis_artifacts`` (+ helpers) and the ``DEFAULT_*`` paths
        <- llm_evaluation/scripts/analyze_deepseek_transferability.py
  * ``load_combined_*`` / ``collect_judge_dirs`` (+ helpers) and the config
    constants (``ATTACKS_PATHS`` etc.)
        <- llm_evaluation/notebooks/gemma_and_mistral_attacks/combined_jailbreak_retrieval_data.py
  * ``build_attack_inventory`` / ``load_deepseek_response_dataframe`` (+ helper)
        <- llm_evaluation/notebooks/combined_transferability_paper_helpers.py

Only the hard-coded repository root is de-hardcoded: it is read from the
``LLM_EVAL_ROOT`` environment variable (default: the original box path). This
matters *only* when rebuilding a frame from raw sources; the standalone path
(loading a pre-built response-level parquet) does not touch any of it.

The raw source files themselves are NOT shipped with this release. To rebuild a
frame from raw you must have the ``llm_evaluation`` repo's data available and
point ``LLM_EVAL_ROOT`` at it. Otherwise, load a pre-built response-level
parquet (see ``utils/data.py:load_response_frame`` and the ``--data-path`` /
``JAILBREAK_RETRIEVAL_DATA`` options on the scripts).
"""
from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import subprocess
from contextlib import contextmanager  # noqa: F401  (kept for verbatim parity)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, TextIO

import pandas as pd

# Repository root for the OPTIONAL raw-rebuild path. No default: set the
# LLM_EVAL_ROOT environment variable only if you are rebuilding frames from the
# original raw sources. The standard data path is the Hugging Face dataset
# (see utils/data.py:load_response_frame_from_hf), which needs none of this.
LLM_EVAL_ROOT = Path(os.environ.get("LLM_EVAL_ROOT", ""))


# ==========================================================================
# Vendored verbatim from judge_existing_responses/io_utils.py
# ==========================================================================
def open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def iter_text_lines(path: Path, logger: Optional[logging.Logger] = None) -> Iterator[str]:
    if path.suffix != ".gz":
        with path.open("r", encoding="utf-8") as handle:
            yield from handle
        return

    yielded_lines = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                yielded_lines += 1
                yield line
        return
    except (EOFError, gzip.BadGzipFile, OSError, ValueError) as exc:
        if logger is not None:
            logger.warning(
                "Falling back to partial gzip recovery for %s after read failure (%s)",
                path,
                exc,
            )

    process = subprocess.Popen(
        ["gzip", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    try:
        for line_number, line in enumerate(process.stdout, start=1):
            if line_number <= yielded_lines:
                continue
            yield line
    finally:
        stderr_output = ""
        if process.stderr is not None:
            stderr_output = process.stderr.read().strip()
        return_code = process.wait()
        if return_code != 0 and logger is not None:
            message = stderr_output or f"gzip exited with code {return_code}"
            logger.warning("Recovered partial gzip stream from %s (%s)", path, message)


def iter_jsonl(path: Path) -> Iterator[tuple[int, Dict[str, Any]]]:
    for line_number, line in enumerate(iter_text_lines(path), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        yield line_number, json.loads(stripped)


# ==========================================================================
# Vendored verbatim from scripts/analyze_deepseek_transferability.py
# (ROOT_DIR replaced by LLM_EVAL_ROOT; otherwise identical)
# ==========================================================================
DEFAULT_ATTACKS_PATH = LLM_EVAL_ROOT / "datasets" / "mistral_nemo_successful_attacks_combined.jsonl"
DEFAULT_TARGET_RESPONSES_ROOT = LLM_EVAL_ROOT / "outputs" / "mistral_nemo_mutator_h100_eval_demo" / "target_responses"
DEFAULT_JUDGE_ROOT = (
    LLM_EVAL_ROOT
    / "llm_evaluation_outputs"
    / "h100_eval_demo_deepseek_r1_qwen_32b_judge_all_targets_consolidated"
    / "judge_evaluations"
    / "judge_model=deepseek_r1_distill_qwen_32b_judge"
)
DEFAULT_OUTPUT_DIR = LLM_EVAL_ROOT / "analysis_outputs" / "deepseek_r1_transferability"


@dataclass(frozen=True)
class AnalysisArtifacts:
    attacks: pd.DataFrame
    targets: pd.DataFrame
    judges: pd.DataFrame
    merged: pd.DataFrame
    model_summary: pd.DataFrame
    strict_transfer_rate_matrix: pd.DataFrame
    harmful_transfer_rate_matrix: pd.DataFrame
    strict_transfer_count_matrix: pd.DataFrame
    harmful_transfer_count_matrix: pd.DataFrame
    transfer_attempt_count_matrix: pd.DataFrame


def iter_jsonl_gz_records(root: Path) -> Iterator[dict]:
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.jsonl.gz"))
    for path in paths:
        try:
            for _, payload in iter_jsonl(path):
                yield payload
        except (EOFError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Skipping corrupted shard: {path} ({exc})", file=__import__("sys").stderr)


def iter_jsonl_records(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def load_attack_metadata(attacks_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for row in iter_jsonl_records(attacks_path):
        rows.append(
            {
                "attack_id": row.get("attack_id"),
                "source_target_model": row.get("target_model_full_name"),
                "risk_category_tag": row.get("risk_category_tag"),
                "attack_style_tag": row.get("attack_style_tag"),
                "planner_action": row.get("planner_action"),
                "dataset_source": row.get("dataset_source"),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["attack_id"])


def load_target_responses(target_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for row in iter_jsonl_gz_records(target_root):
        rows.append(
            {
                "response_run_id": row.get("response_run_id"),
                "attack_id": row.get("attack_id"),
                "target_model_id": row.get("target_model_id"),
                "target_model_revision": row.get("target_model_revision"),
                "target_status": row.get("status"),
                "response_text": row.get("response_text"),
                "timestamp_utc": row.get("timestamp_utc"),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["response_run_id"])


def normalize_violated_categories(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def count_true(series: pd.Series) -> int:
    return int(series.eq(True).sum())


def load_judge_evaluations(judge_root: Path) -> pd.DataFrame:
    latest_rows: dict[str, dict] = {}
    for row in iter_jsonl_gz_records(judge_root):
        judge_run_id = row.get("judge_run_id")
        if isinstance(judge_run_id, str):
            latest_rows[judge_run_id] = row

    rows: list[dict] = []
    for row in latest_rows.values():
        label = row.get("label")
        if isinstance(label, str):
            label = label.upper()
        rows.append(
            {
                "judge_run_id": row.get("judge_run_id"),
                "response_run_id": row.get("response_run_id"),
                "attack_id": row.get("attack_id"),
                "judge_status": row.get("status"),
                "judge_label": label,
                "judge_verdict_code": row.get("judge_verdict_code"),
                "risk_severity_score": row.get("risk_severity_score"),
                "violated_categories": normalize_violated_categories(row.get("violated_categories")),
                "judge_reasoning": row.get("judge_reasoning"),
                "raw_judge_output": row.get("raw_judge_output"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["strict_attack_succeeded"] = df["judge_label"].map({"UNSAFE": True, "SAFE": False, "PARTIAL": False})
    df["harmful_attack_succeeded"] = df["judge_label"].map({"UNSAFE": True, "SAFE": False, "PARTIAL": True})
    df["judge_label_known"] = df["judge_label"].isin(["SAFE", "PARTIAL", "UNSAFE"])
    return df


def build_analysis_artifacts(
    attacks_path: Path = DEFAULT_ATTACKS_PATH,
    target_root: Path = DEFAULT_TARGET_RESPONSES_ROOT,
    judge_root: Path = DEFAULT_JUDGE_ROOT,
) -> AnalysisArtifacts:
    attacks = load_attack_metadata(attacks_path)
    targets = load_target_responses(target_root)
    judges = load_judge_evaluations(judge_root)

    merged = targets.merge(judges, on=["response_run_id", "attack_id"], how="inner")
    merged = merged.merge(attacks, on="attack_id", how="left")
    merged = merged[merged["target_status"] == "success"].copy()

    if merged.empty:
        empty = pd.DataFrame()
        return AnalysisArtifacts(attacks, targets, judges, merged, empty, empty, empty, empty, empty, empty)

    model_summary = (
        merged.groupby("target_model_id", dropna=False)
        .agg(
            attempts=("response_run_id", "count"),
            judged_attempts=("judge_label_known", "sum"),
            safe_count=("judge_label", lambda s: int((s == "SAFE").sum())),
            partial_count=("judge_label", lambda s: int((s == "PARTIAL").sum())),
            unsafe_count=("judge_label", lambda s: int((s == "UNSAFE").sum())),
            strict_successes=("strict_attack_succeeded", count_true),
            harmful_successes=("harmful_attack_succeeded", count_true),
            mean_risk_severity_score=("risk_severity_score", "mean"),
            median_risk_severity_score=("risk_severity_score", "median"),
        )
        .reset_index()
    )
    model_summary["strict_attack_success_rate"] = (
        model_summary["strict_successes"] / model_summary["judged_attempts"].replace(0, pd.NA)
    )
    model_summary["harmful_attack_success_rate"] = (
        model_summary["harmful_successes"] / model_summary["judged_attempts"].replace(0, pd.NA)
    )
    model_summary["strict_attack_success_pct"] = model_summary["strict_attack_success_rate"] * 100.0
    model_summary["harmful_attack_success_pct"] = model_summary["harmful_attack_success_rate"] * 100.0
    model_summary = model_summary.sort_values("harmful_attack_success_rate", ascending=False, na_position="last")

    known = merged[merged["judge_label_known"]].copy()

    strict_transfer_rate_matrix = pd.pivot_table(
        known,
        index="source_target_model",
        columns="target_model_id",
        values="strict_attack_succeeded",
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)
    harmful_transfer_rate_matrix = pd.pivot_table(
        known,
        index="source_target_model",
        columns="target_model_id",
        values="harmful_attack_succeeded",
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)
    strict_transfer_count_matrix = pd.pivot_table(
        known,
        index="source_target_model",
        columns="target_model_id",
        values="strict_attack_succeeded",
        aggfunc="sum",
    ).sort_index(axis=0).sort_index(axis=1)
    harmful_transfer_count_matrix = pd.pivot_table(
        known,
        index="source_target_model",
        columns="target_model_id",
        values="harmful_attack_succeeded",
        aggfunc="sum",
    ).sort_index(axis=0).sort_index(axis=1)
    transfer_attempt_count_matrix = pd.pivot_table(
        known,
        index="source_target_model",
        columns="target_model_id",
        values="response_run_id",
        aggfunc="count",
    ).sort_index(axis=0).sort_index(axis=1)

    return AnalysisArtifacts(
        attacks=attacks,
        targets=targets,
        judges=judges,
        merged=merged,
        model_summary=model_summary,
        strict_transfer_rate_matrix=strict_transfer_rate_matrix,
        harmful_transfer_rate_matrix=harmful_transfer_rate_matrix,
        strict_transfer_count_matrix=strict_transfer_count_matrix,
        harmful_transfer_count_matrix=harmful_transfer_count_matrix,
        transfer_attempt_count_matrix=transfer_attempt_count_matrix,
    )


# ==========================================================================
# Vendored verbatim from
# notebooks/gemma_and_mistral_attacks/combined_jailbreak_retrieval_data.py
# (ROOT replaced by LLM_EVAL_ROOT; otherwise identical)
# ==========================================================================
CONSENSUS_SEVERITY = {
    "consensus_safe": 0,
    "mixed_partial": 1,
    "consensus_partial": 2,
    "disputed_unsafe": 3,
    "majority_unsafe": 4,
    "consensus_unsafe": 5,
}

DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
GEMMA_MODEL_ID = "google/gemma-4-31B-it"

ATTACKS_PATHS = {
    "mistral": LLM_EVAL_ROOT / "datasets" / "mistral_nemo_successful_attacks_combined.jsonl",
    "gemma": LLM_EVAL_ROOT / "datasets" / "gemma_3_successful_attacks_combined.jsonl",
}
TARGET_RESPONSE_ROOTS = {
    "mistral": LLM_EVAL_ROOT / "outputs" / "mistral_nemo_mutator_h100_eval_demo" / "target_responses",
    "gemma": LLM_EVAL_ROOT / "outputs" / "gemma_3_mutator_h100_eval_demo" / "target_responses",
}
LLM_EVALUATION_OUTPUTS = LLM_EVAL_ROOT / "llm_evaluation_outputs"
ALLOWED_ATTACK_BASENAMES = {path.name for path in ATTACKS_PATHS.values()}


def _iter_manifests() -> Iterable[tuple[dict, Path]]:
    for manifest_path in sorted(LLM_EVALUATION_OUTPUTS.glob("*/manifest.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        yield payload, manifest_path.parent


def _resolve_judge_dir(payload: dict, manifest_dir: Path, judge_model_id: str) -> Path | None:
    judge_models = payload.get("judge_models") or []
    served_model_name = None
    for judge_model in judge_models:
        if judge_model.get("model_id") == judge_model_id:
            served_model_name = judge_model.get("served_model_name")
            break
    judge_evaluations_dir = manifest_dir / "judge_evaluations"
    if served_model_name:
        candidate = judge_evaluations_dir / f"judge_model={str(served_model_name).replace('/', '__')}"
        if candidate.exists():
            return candidate
    if judge_evaluations_dir.exists():
        children = [child for child in judge_evaluations_dir.iterdir() if child.is_dir()]
        if len(children) == 1:
            return children[0]
    return None


def collect_judge_dirs(judge_model_id: str) -> list[Path]:
    matches: list[tuple[str, str, Path]] = []
    for payload, manifest_dir in _iter_manifests():
        source_attack_path = payload.get("source_attack_jsonl_path") or ""
        source_attack_name = Path(source_attack_path).name if source_attack_path else ""
        if source_attack_name not in ALLOWED_ATTACK_BASENAMES:
            continue
        judge_models = payload.get("judge_models") or []
        if not any(judge_model.get("model_id") == judge_model_id for judge_model in judge_models):
            continue
        judge_dir = _resolve_judge_dir(payload, manifest_dir, judge_model_id)
        if judge_dir is None or not judge_dir.exists():
            continue
        matches.append((
            str(payload.get("created_at_utc") or ""),
            str(payload.get("judge_run_experiment_id") or manifest_dir.name),
            judge_dir,
        ))
    ordered_dirs: list[Path] = []
    seen: set[Path] = set()
    for _, _, judge_dir in sorted(matches):
        if judge_dir not in seen:
            seen.add(judge_dir)
            ordered_dirs.append(judge_dir)
    return ordered_dirs


def load_combined_attack_metadata() -> pd.DataFrame:
    frames = [load_attack_metadata(path) for path in ATTACKS_PATHS.values()]
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["attack_id"])


def load_combined_target_responses() -> pd.DataFrame:
    frames = [load_target_responses(root) for root in TARGET_RESPONSE_ROOTS.values()]
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["response_run_id"])


def load_combined_judge_evaluations(judge_model_id: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for order, judge_dir in enumerate(collect_judge_dirs(judge_model_id)):
        frame = load_judge_evaluations(judge_dir)
        if frame.empty:
            continue
        frame = frame.copy()
        if "judge_label_known" in frame.columns:
            frame = frame.loc[frame["judge_label_known"].eq(True)].copy()
        if "judge_label" in frame.columns:
            frame = frame.loc[
                frame["judge_label"].isin(["SAFE", "PARTIAL", "UNSAFE"])
            ].copy()
        if frame.empty:
            continue
        frame["_source_order"] = order
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["_source_order"]).drop_duplicates(
        subset=["response_run_id"], keep="last"
    )
    return combined.drop(columns=["_source_order"])


# ==========================================================================
# Vendored verbatim from notebooks/combined_transferability_paper_helpers.py
# ==========================================================================
def build_attack_inventory() -> pd.DataFrame:
    attack_df = load_combined_attack_metadata().copy()
    attack_source_rows = []
    for attack_source, path in ATTACKS_PATHS.items():
        source_df = pd.read_json(path, lines=True)
        if "attack_id" not in source_df.columns:
            continue
        attack_source_rows.append(
            source_df[["attack_id"]].drop_duplicates().assign(attack_source=attack_source)
        )
    attack_source_map = pd.concat(attack_source_rows, ignore_index=True).drop_duplicates(
        subset=["attack_id"]
    )
    return attack_df.merge(attack_source_map, on="attack_id", how="left")


def _build_single_judge_response_dataframe(
    judge_model_id: str,
    success_mode: str,
) -> tuple[pd.DataFrame, list[str]]:
    targets_df = load_combined_target_responses()
    attacks_df = build_attack_inventory()
    judge_df = load_combined_judge_evaluations(judge_model_id)
    response_df = targets_df.merge(
        judge_df,
        on=["response_run_id", "attack_id"],
        how="inner",
    )
    response_df = response_df.merge(attacks_df, on="attack_id", how="left")
    response_df = response_df.loc[
        response_df["target_status"].eq("success")
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


def load_deepseek_response_dataframe(success_mode: str = "harmful") -> tuple[pd.DataFrame, list[str]]:
    return _build_single_judge_response_dataframe(DEEPSEEK_MODEL_ID, success_mode)


def load_gemma_response_dataframe(success_mode: str = "harmful") -> tuple[pd.DataFrame, list[str]]:
    return _build_single_judge_response_dataframe(GEMMA_MODEL_ID, success_mode)
