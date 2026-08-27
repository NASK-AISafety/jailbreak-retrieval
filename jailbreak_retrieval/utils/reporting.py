from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .benchmark import METHOD_LABELS
from .data import FOCUS_TARGETS


def plot_candidate_summary(
    aggregate_summary_df: pd.DataFrame,
    per_target_summary_df: pd.DataFrame,
    learning_curves_df: pd.DataFrame,
    candidate_method_name: str,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    candidate_label = METHOD_LABELS[candidate_method_name]
    heatmap_df = per_target_summary_df.pivot(
        index="target_model_id",
        columns="batch_size",
        values="candidate_minus_top_pp",
    )
    heatmap_df = heatmap_df.loc[sorted(heatmap_df.index)]

    focus_curve_df = learning_curves_df.loc[
        learning_curves_df["target_model_id"].isin(FOCUS_TARGETS)
    ].copy()
    mean_curve_df = learning_curves_df.groupby(
        ["batch_size", "method", "method_label", "attacks_seen"], as_index=False
    ).agg(mean_cumulative_asr=("cumulative_asr", "mean"))

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0.0,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title(f"{candidate_label} minus static top (pp)")
    axes[0, 0].set_xlabel("Batch size")
    axes[0, 0].set_ylabel("Held-out target")

    sns.barplot(
        data=aggregate_summary_df,
        x="batch_size",
        y="mean_asr",
        hue="method_label",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Mean ASR across all 8 held-out targets")
    axes[0, 1].set_xlabel("Batch size")
    axes[0, 1].set_ylabel("Mean ASR at 4000 attacks")
    axes[0, 1].legend(title="Method", fontsize=10)

    candidate_curve = mean_curve_df.loc[
        mean_curve_df["method"].eq(candidate_method_name)
        & mean_curve_df["batch_size"].eq(100)
    ]
    adaptive_curve = mean_curve_df.loc[
        mean_curve_df["method"].eq("adaptive_baseline")
        & mean_curve_df["batch_size"].eq(100)
    ]
    top_curve = mean_curve_df.loc[
        mean_curve_df["method"].eq("static_top") & mean_curve_df["batch_size"].eq(25)
    ]
    axes[1, 0].plot(
        candidate_curve["attacks_seen"],
        candidate_curve["mean_cumulative_asr"],
        linewidth=2,
        label=f"{candidate_label} (batch=100)",
    )
    axes[1, 0].plot(
        adaptive_curve["attacks_seen"],
        adaptive_curve["mean_cumulative_asr"],
        linewidth=2,
        linestyle="--",
        label="Adaptive baseline (batch=100)",
    )
    axes[1, 0].plot(
        top_curve["attacks_seen"],
        top_curve["mean_cumulative_asr"],
        linewidth=2,
        linestyle=":",
        label="Static top",
    )
    axes[1, 0].set_title("Mean learning curves across all targets")
    axes[1, 0].set_xlabel("Attacks seen")
    axes[1, 0].set_ylabel("Mean cumulative ASR")
    axes[1, 0].legend(fontsize=10)

    focus_plot_df = focus_curve_df.loc[
        focus_curve_df["method"].isin(
            [candidate_method_name, "adaptive_baseline", "static_top"]
        )
        & focus_curve_df["batch_size"].isin([25, 100])
    ].copy()
    sns.lineplot(
        data=focus_plot_df,
        x="attacks_seen",
        y="cumulative_asr",
        hue="method_label",
        style="target_model_id",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Focus targets: Mistral-Nemo and GPT-OSS")
    axes[1, 1].set_xlabel("Attacks seen")
    axes[1, 1].set_ylabel("Cumulative ASR")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.20)
    plt.show()
