from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import FOCUS_TARGETS, prepare_target_dataset
from .methods import METHOD_REGISTRY

METHOD_LABELS = {
    "baseline_random": "Random baseline",
    "weighted_success_sampling": "Weighted success sampling",
    "static_top": "Static top-4000",
    "adaptive_baseline": "Adaptive baseline",
    "adaptive_transferability": "Adaptive transferability",
    "model_mixture": "Model mixture",
    "contextual_ucb": "Contextual UCB",
    "hybrid_mixture_ucb": "Hybrid mixture + UCB",
}


def build_learning_curve_rows(
    selected_df: pd.DataFrame,
    method: str,
    batch_size: int,
    target_model_id: str,
    budget: int,
) -> list[dict]:
    checkpoints = list(range(batch_size, budget + 1, batch_size))
    if not checkpoints:
        # batch_size > budget (e.g. a tiny held-out pool): a single checkpoint at
        # the whole budget. Avoids IndexError on checkpoints[-1].
        checkpoints = [budget]
    elif checkpoints[-1] != budget:
        checkpoints.append(budget)
    return [
        {
            "target_model_id": target_model_id,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "batch_size": batch_size,
            "attacks_seen": attacks_seen,
            "cumulative_asr": float(
                selected_df["target_success"].iloc[:attacks_seen].mean()
            ),
        }
        for attacks_seen in checkpoints
    ]


def run_candidate_benchmark(
    response_df: pd.DataFrame,
    target_models: list[str],
    candidate_method_name: str,
    batch_sizes: list[int],
    total_budget: int,
    output_dir: Path,
    comparison_methods: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    methods = comparison_methods or [
        "baseline_random",
        "static_top",
        "adaptive_baseline",
        candidate_method_name,
    ]
    final_rows: list[dict] = []
    learning_curve_rows: list[dict] = []
    inventory_rows: list[dict] = []

    for target_model_id in target_models:
        prepared = prepare_target_dataset(
            response_df=response_df,
            target_models=target_models,
            target_model_id=target_model_id,
        )
        prepared_df = prepared.prepared_df
        model_cols = prepared.model_cols
        budget = min(total_budget, len(prepared_df))
        inventory_rows.append(
            {
                "target_model_id": target_model_id,
                "candidate_attacks": len(prepared_df),
                "budget": budget,
                "mean_hidden_target_asr": float(prepared_df["target_success"].mean()),
                "mean_prior_mean": float(prepared_df["prior_mean"].mean()),
            }
        )

        for batch_size in batch_sizes:
            for method in methods:
                selected_df = METHOD_REGISTRY[method](
                    prepared_df=prepared_df,
                    model_cols=model_cols,
                    batch_size=batch_size,
                    budget=budget,
                    target_model_id=target_model_id,
                )
                final_rows.append(
                    {
                        "target_model_id": target_model_id,
                        "batch_size": batch_size,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "final_asr": float(selected_df["target_success"].mean()),
                        "selected_attacks": len(selected_df),
                    }
                )
                learning_curve_rows.extend(
                    build_learning_curve_rows(
                        selected_df=selected_df,
                        method=method,
                        batch_size=batch_size,
                        target_model_id=target_model_id,
                        budget=budget,
                    )
                )

    inventory_df = pd.DataFrame(inventory_rows)
    final_results_df = pd.DataFrame(final_rows)
    learning_curves_df = pd.DataFrame(learning_curve_rows)

    aggregate_summary_df = final_results_df.groupby(
        ["batch_size", "method", "method_label"], as_index=False
    ).agg(
        mean_asr=("final_asr", "mean"),
        std_asr=("final_asr", "std"),
        min_asr=("final_asr", "min"),
        max_asr=("final_asr", "max"),
    )
    comparison_df = (
        aggregate_summary_df.pivot(
            index="batch_size", columns="method", values="mean_asr"
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    comparison_df["candidate_minus_top_pp"] = 100.0 * (
        comparison_df[candidate_method_name] - comparison_df["static_top"]
    )
    comparison_df["candidate_minus_adaptive_pp"] = 100.0 * (
        comparison_df[candidate_method_name] - comparison_df["adaptive_baseline"]
    )
    comparison_df["candidate_minus_random_pp"] = 100.0 * (
        comparison_df[candidate_method_name] - comparison_df["baseline_random"]
    )

    per_target_summary_df = (
        final_results_df.pivot_table(
            index=["target_model_id", "batch_size"],
            columns="method",
            values="final_asr",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    per_target_summary_df["candidate_minus_top_pp"] = 100.0 * (
        per_target_summary_df[candidate_method_name]
        - per_target_summary_df["static_top"]
    )
    per_target_summary_df["candidate_minus_adaptive_pp"] = 100.0 * (
        per_target_summary_df[candidate_method_name]
        - per_target_summary_df["adaptive_baseline"]
    )
    per_target_summary_df["candidate_minus_random_pp"] = 100.0 * (
        per_target_summary_df[candidate_method_name]
        - per_target_summary_df["baseline_random"]
    )

    focus_summary_df = (
        per_target_summary_df.loc[
            per_target_summary_df["target_model_id"].isin(FOCUS_TARGETS)
        ]
        .sort_values(["target_model_id", "batch_size"])
        .reset_index(drop=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_df.to_csv(output_dir / "target_inventory.csv", index=False)
    final_results_df.to_csv(output_dir / "final_results.csv", index=False)
    learning_curves_df.to_csv(output_dir / "learning_curves.csv", index=False)
    aggregate_summary_df.to_csv(output_dir / "aggregate_summary.csv", index=False)
    comparison_df.to_csv(output_dir / "comparison_summary.csv", index=False)
    per_target_summary_df.to_csv(output_dir / "per_target_summary.csv", index=False)
    focus_summary_df.to_csv(output_dir / "focus_targets_summary.csv", index=False)

    return {
        "inventory_df": inventory_df,
        "final_results_df": final_results_df,
        "learning_curves_df": learning_curves_df,
        "aggregate_summary_df": aggregate_summary_df,
        "comparison_df": comparison_df,
        "per_target_summary_df": per_target_summary_df,
        "focus_summary_df": focus_summary_df,
        "candidate_method_name": pd.DataFrame(
            [{"candidate_method_name": candidate_method_name}]
        ),
    }
