from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np
import pandas as pd

from .data import BASE_FEATURES, PAIR_FEATURES, TRANSFER_FEATURES

SelectionFn = Callable[[pd.DataFrame, list[str], int, int, str], pd.DataFrame]


def deterministic_seed(*parts: object) -> int:
    value = 0
    for part in parts:
        text = str(part)
        for char in text:
            value = (value * 131 + ord(char)) % (2**32)
    return value


def posterior_mean(
    successes: int, total: int, prior_mean: float, pseudo_count: float
) -> float:
    return (successes + pseudo_count * prior_mean) / (total + pseudo_count)


def fit_ridge_weights(
    X: np.ndarray, y: np.ndarray, alpha: float = 4.0
) -> tuple[float, np.ndarray]:
    X_aug = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    reg = alpha * np.eye(X_aug.shape[1])
    reg[0, 0] = 0.0
    beta = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)
    intercept = float(beta[0])
    weights = np.maximum(beta[1:], 0.0)
    if float(weights.sum()) == 0.0:
        weights = np.full_like(weights, 1.0 / len(weights))
    else:
        weights = weights / weights.sum()
    return intercept, weights


def _sort_pool(pool_df: pd.DataFrame, score_col: str, n: int) -> pd.DataFrame:
    return (
        pool_df.sort_values(
            [
                score_col,
                "prior_mean",
                "strict_success_rate",
                "train_success_count",
                "max_risk_severity_score",
                "attack_id",
            ],
            ascending=[False, False, False, False, False, True],
        )
        .head(n)
        .copy()
    )


def _transfer_prior_score(
    pool_df: pd.DataFrame,
    w: tuple[float, float, float, float] = (0.40, 0.30, 0.15, 0.15),  # Eq. 4
) -> pd.Series:
    w1, w2, w3, w4 = w
    return (
        w1 * pool_df["prior_mean"]
        + w2 * pool_df["strict_success_rate"]
        + w3 * pool_df["partial_success_rate"]
        + w4 * (pool_df["mean_judge_severity"] / 2.0)
    )


def _weighted_sample_without_replacement(
    pool_df: pd.DataFrame,
    weights: pd.Series,
    n: int,
    seed: int,
) -> pd.DataFrame:
    positive_mask = weights.gt(0)
    positive_df = pool_df.loc[positive_mask].copy()
    zero_df = pool_df.loc[~positive_mask].copy()
    selected_parts: list[pd.DataFrame] = []
    remaining = min(n, len(pool_df))

    if remaining == 0:
        return pool_df.head(0).copy()

    if not positive_df.empty:
        take_positive = min(remaining, len(positive_df))
        selected_parts.append(
            positive_df.sample(
                n=take_positive,
                weights=weights.loc[positive_df.index],
                random_state=seed,
                replace=False,
            )
        )
        remaining -= take_positive

    if remaining > 0 and not zero_df.empty:
        selected_parts.append(
            zero_df.sample(
                n=min(remaining, len(zero_df)),
                random_state=seed + 1,
                replace=False,
            )
        )

    return pd.concat(selected_parts, ignore_index=True)


def select_baseline_random(
    prepared_df: pd.DataFrame,
    model_cols: list[str],
    batch_size: int,
    budget: int,
    target_model_id: str,
) -> pd.DataFrame:
    del model_cols, batch_size
    budget = min(budget, len(prepared_df))
    return prepared_df.sample(
        n=budget,
        random_state=deterministic_seed("baseline_random", target_model_id),
    ).reset_index(drop=True)


def select_static_top(
    prepared_df: pd.DataFrame,
    model_cols: list[str],
    batch_size: int,
    budget: int,
    target_model_id: str,
) -> pd.DataFrame:
    del model_cols, batch_size, target_model_id
    budget = min(budget, len(prepared_df))
    return _sort_pool(prepared_df, "prior_mean", budget).reset_index(drop=True)


def select_weighted_success_sampling(
    prepared_df: pd.DataFrame,
    model_cols: list[str],
    batch_size: int,
    budget: int,
    target_model_id: str,
) -> pd.DataFrame:
    del model_cols, batch_size
    budget = min(budget, len(prepared_df))
    weights = prepared_df["train_success_rate"].astype(float)
    return _weighted_sample_without_replacement(
        pool_df=prepared_df,
        weights=weights,
        n=budget,
        seed=deterministic_seed("weighted_success_sampling", target_model_id),
    ).reset_index(drop=True)


def select_adaptive_baseline(
    prepared_df: pd.DataFrame,
    model_cols: list[str],
    batch_size: int,
    budget: int,
    target_model_id: str,
    beta: tuple[float, float, float] = (0.45, 0.25, 0.30),  # Eq. 3 paper coefficients
) -> pd.DataFrame:
    del target_model_id
    b1, b2, b3 = beta
    budget = min(budget, len(prepared_df))
    pool_df = prepared_df.copy()
    global_prior = float(pool_df["prior_mean"].mean())
    feature_priors = {
        feature: pool_df.groupby(feature, dropna=False)["train_success_rate"]
        .mean()
        .to_dict()
        for feature in BASE_FEATURES
    }
    model_priors = {}
    for column in model_cols:
        for value in (0, 1):
            mean_value = pool_df.loc[pool_df[column].eq(value), "prior_mean"].mean()
            model_priors[(column, value)] = (
                global_prior if pd.isna(mean_value) else float(mean_value)
            )

    feature_stats = {feature: {} for feature in BASE_FEATURES}
    model_stats = {(column, value): [0, 0] for column in model_cols for value in (0, 1)}
    selected_parts: list[pd.DataFrame] = []
    observed_count = 0
    while observed_count < budget:
        current_batch_size = min(batch_size, budget - observed_count)
        if observed_count == 0:
            batch_df = _sort_pool(pool_df, "prior_mean", current_batch_size)
        else:
            feature_score = pd.Series(0.0, index=pool_df.index)
            for feature in BASE_FEATURES:
                stats = feature_stats[feature]
                feature_score += (
                    pool_df[feature]
                    .map(
                        lambda value: posterior_mean(
                            stats.get(value, [0, 0])[0],
                            stats.get(value, [0, 0])[1],
                            float(feature_priors[feature].get(value, global_prior)),
                            12.0,
                        )
                    )
                    .astype(float)
                )
            feature_score /= len(BASE_FEATURES)

            model_score = pd.Series(0.0, index=pool_df.index)
            for column in model_cols:
                model_score += pool_df[column].map(
                    lambda value: posterior_mean(
                        model_stats[(column, int(value))][0],
                        model_stats[(column, int(value))][1],
                        model_priors[(column, int(value))],
                        10.0,
                    )
                )
            model_score /= max(len(model_cols), 1)
            pool_df = pool_df.assign(
                score=b1 * pool_df["prior_mean"]
                + b2 * feature_score
                + b3 * model_score,
            )
            batch_df = _sort_pool(pool_df, "score", current_batch_size)

        selected_parts.append(batch_df)
        observed_count += len(batch_df)
        for row in batch_df.to_dict("records"):
            label = int(row["target_success"])
            for feature in BASE_FEATURES:
                stats = feature_stats[feature].setdefault(row[feature], [0, 0])
                stats[0] += label
                stats[1] += 1
            for column in model_cols:
                key = (column, int(row[column]))
                model_stats[key][0] += label
                model_stats[key][1] += 1
        pool_df = pool_df.loc[~pool_df["attack_id"].isin(batch_df["attack_id"])].copy()

    return pd.concat(selected_parts, ignore_index=True)


def select_adaptive_transferability(
    prepared_df: pd.DataFrame,
    model_cols: list[str],
    batch_size: int,
    budget: int,
    target_model_id: str,
    w: tuple[float, float, float, float] = (0.40, 0.30, 0.15, 0.15),  # Eq. 4
    alpha4: float = 0.35,          # Eq. 5 transfer weight
    c_ucb: float = 0.06,           # Eq. 5 UCB exploration constant
    decay: tuple[float, float] = (0.28, 0.20),  # Eq. 5 prior-weight anneal bounds
) -> pd.DataFrame:
    del target_model_id
    d_hi, d_lo = decay
    budget = min(budget, len(prepared_df))
    pool_df = prepared_df.copy()
    pool_df["transfer_prior_score"] = _transfer_prior_score(pool_df, w)
    global_prior = float(pool_df["transfer_prior_score"].mean())

    feature_priors = {
        feature: pool_df.groupby(feature, dropna=False)["train_success_rate"]
        .mean()
        .to_dict()
        for feature in BASE_FEATURES
    }
    transfer_priors = {
        feature: pool_df.groupby(feature, dropna=False)["train_success_rate"]
        .mean()
        .to_dict()
        for feature in TRANSFER_FEATURES
    }
    model_priors = {}
    for column in model_cols:
        for value in (0, 1):
            mean_value = pool_df.loc[
                pool_df[column].eq(value), "transfer_prior_score"
            ].mean()
            model_priors[(column, value)] = (
                global_prior if pd.isna(mean_value) else float(mean_value)
            )

    feature_stats = {feature: {} for feature in BASE_FEATURES}
    transfer_stats = {feature: {} for feature in TRANSFER_FEATURES}
    model_stats = {(column, value): [0, 0] for column in model_cols for value in (0, 1)}

    selected_parts: list[pd.DataFrame] = []
    observed_count = 0
    while observed_count < budget:
        current_batch_size = min(batch_size, budget - observed_count)
        if observed_count == 0:
            batch_df = _sort_pool(pool_df, "transfer_prior_score", current_batch_size)
        else:
            learning_fraction = min(1.0, observed_count / max(batch_size * 6, 1))

            feature_score = pd.Series(0.0, index=pool_df.index)
            for feature in BASE_FEATURES:
                stats = feature_stats[feature]
                feature_score += (
                    pool_df[feature]
                    .map(
                        lambda value: posterior_mean(
                            stats.get(value, [0, 0])[0],
                            stats.get(value, [0, 0])[1],
                            float(feature_priors[feature].get(value, global_prior)),
                            12.0,
                        )
                    )
                    .astype(float)
                )
            feature_score /= len(BASE_FEATURES)

            transfer_score = pd.Series(0.0, index=pool_df.index)
            transfer_bonus = pd.Series(0.0, index=pool_df.index)
            for feature in TRANSFER_FEATURES:
                stats = transfer_stats[feature]
                transfer_score += (
                    pool_df[feature]
                    .map(
                        lambda value: posterior_mean(
                            stats.get(value, [0, 0])[0],
                            stats.get(value, [0, 0])[1],
                            float(transfer_priors[feature].get(value, global_prior)),
                            8.0,
                        )
                    )
                    .astype(float)
                )
                transfer_bonus += (
                    pool_df[feature]
                    .map(
                        lambda value: 1.0 / math.sqrt(stats.get(value, [0, 0])[1] + 1.0)
                    )
                    .astype(float)
                )
            transfer_score /= len(TRANSFER_FEATURES)
            transfer_bonus /= len(TRANSFER_FEATURES)

            model_score = pd.Series(0.0, index=pool_df.index)
            for column in model_cols:
                model_score += pool_df[column].map(
                    lambda value: posterior_mean(
                        model_stats[(column, int(value))][0],
                        model_stats[(column, int(value))][1],
                        model_priors[(column, int(value))],
                        10.0,
                    )
                )
            model_score /= max(len(model_cols), 1)

            w_prior = d_hi - (d_hi - d_lo) * learning_fraction
            w_feature = 0.17 + 0.03 * learning_fraction
            w_model = 0.20 + 0.05 * learning_fraction
            w_transfer = alpha4

            pool_df = pool_df.assign(
                score=(
                    w_prior * pool_df["transfer_prior_score"]
                    + w_feature * feature_score
                    + w_model * model_score
                    + w_transfer * transfer_score
                    + c_ucb * transfer_bonus
                )
            )
            batch_df = _sort_pool(pool_df, "score", current_batch_size)

        selected_parts.append(batch_df)
        observed_count += len(batch_df)
        for row in batch_df.to_dict("records"):
            label = int(row["target_success"])
            for feature in BASE_FEATURES:
                stats = feature_stats[feature].setdefault(row[feature], [0, 0])
                stats[0] += label
                stats[1] += 1
            for feature in TRANSFER_FEATURES:
                stats = transfer_stats[feature].setdefault(row[feature], [0, 0])
                stats[0] += label
                stats[1] += 1
            for column in model_cols:
                key = (column, int(row[column]))
                model_stats[key][0] += label
                model_stats[key][1] += 1
        pool_df = pool_df.loc[~pool_df["attack_id"].isin(batch_df["attack_id"])].copy()

    return pd.concat(selected_parts, ignore_index=True)


def select_model_mixture(
    prepared_df: pd.DataFrame,
    model_cols: list[str],
    batch_size: int,
    budget: int,
    target_model_id: str,
) -> pd.DataFrame:
    del target_model_id
    budget = min(budget, len(prepared_df))
    pool_df = prepared_df.copy()
    selected_parts: list[pd.DataFrame] = []
    observed_count = 0
    while observed_count < budget:
        current_batch_size = min(batch_size, budget - observed_count)
        if observed_count == 0:
            batch_df = select_static_top(
                pool_df, model_cols, batch_size, current_batch_size, ""
            )
        else:
            observed_df = pd.concat(selected_parts, ignore_index=True)
            X_obs = observed_df[model_cols].to_numpy(dtype=float)
            y_obs = observed_df["target_success"].to_numpy(dtype=float)
            intercept, weights = fit_ridge_weights(X_obs, y_obs)
            X_pool = pool_df[model_cols].to_numpy(dtype=float)
            mixture_score = pd.Series(intercept + X_pool @ weights, index=pool_df.index)
            pool_df = pool_df.assign(
                score=0.65 * mixture_score + 0.35 * pool_df["prior_mean"],
            )
            batch_df = (
                pool_df.sort_values(
                    [
                        "score",
                        "prior_mean",
                        "train_success_count",
                        "max_risk_severity_score",
                        "attack_id",
                    ],
                    ascending=[False, False, False, False, True],
                )
                .head(current_batch_size)
                .copy()
            )

        selected_parts.append(batch_df)
        observed_count += len(batch_df)
        pool_df = pool_df.loc[~pool_df["attack_id"].isin(batch_df["attack_id"])].copy()

    return pd.concat(selected_parts, ignore_index=True)


def select_contextual_ucb(
    prepared_df: pd.DataFrame,
    model_cols: list[str],
    batch_size: int,
    budget: int,
    target_model_id: str,
) -> pd.DataFrame:
    del model_cols, target_model_id
    budget = min(budget, len(prepared_df))
    pool_df = prepared_df.copy()
    interaction_features = [f"{left}__{right}" for left, right in PAIR_FEATURES]
    all_features = list(BASE_FEATURES) + interaction_features
    global_prior = float(pool_df["prior_mean"].mean())
    feature_priors = {
        feature: pool_df.groupby(feature, dropna=False)["prior_mean"].mean().to_dict()
        for feature in all_features
    }
    feature_stats = {feature: {} for feature in all_features}
    selected_parts: list[pd.DataFrame] = []
    observed_count = 0
    while observed_count < budget:
        current_batch_size = min(batch_size, budget - observed_count)
        if observed_count == 0:
            batch_df = select_static_top(
                pool_df, [], batch_size, current_batch_size, ""
            )
        else:
            score = 0.35 * pool_df["prior_mean"].copy()
            bonus = pd.Series(0.0, index=pool_df.index)
            for feature in all_features:
                stats = feature_stats[feature]
                means = (
                    pool_df[feature]
                    .map(
                        lambda value: posterior_mean(
                            stats.get(value, [0, 0])[0],
                            stats.get(value, [0, 0])[1],
                            float(feature_priors[feature].get(value, global_prior)),
                            10.0,
                        )
                    )
                    .astype(float)
                )
                confidence = (
                    pool_df[feature]
                    .map(
                        lambda value: 1.0 / math.sqrt(stats.get(value, [0, 0])[1] + 1.0)
                    )
                    .astype(float)
                )
                weight = 0.18 if feature in BASE_FEATURES else 0.07
                score += weight * means
                bonus += 0.04 * confidence
            pool_df = pool_df.assign(score=score + bonus)
            batch_df = (
                pool_df.sort_values(
                    [
                        "score",
                        "prior_mean",
                        "train_success_count",
                        "max_risk_severity_score",
                        "attack_id",
                    ],
                    ascending=[False, False, False, False, True],
                )
                .head(current_batch_size)
                .copy()
            )

        selected_parts.append(batch_df)
        observed_count += len(batch_df)
        for row in batch_df.to_dict("records"):
            label = int(row["target_success"])
            for feature in all_features:
                stats = feature_stats[feature].setdefault(row[feature], [0, 0])
                stats[0] += label
                stats[1] += 1
        pool_df = pool_df.loc[~pool_df["attack_id"].isin(batch_df["attack_id"])].copy()

    return pd.concat(selected_parts, ignore_index=True)


def select_hybrid_mixture_ucb(
    prepared_df: pd.DataFrame,
    model_cols: list[str],
    batch_size: int,
    budget: int,
    target_model_id: str,
) -> pd.DataFrame:
    del target_model_id
    budget = min(budget, len(prepared_df))
    pool_df = prepared_df.copy()
    interaction_features = [f"{left}__{right}" for left, right in PAIR_FEATURES]
    all_features = list(BASE_FEATURES) + interaction_features
    global_prior = float(pool_df["prior_mean"].mean())
    feature_priors = {
        feature: pool_df.groupby(feature, dropna=False)["prior_mean"].mean().to_dict()
        for feature in all_features
    }
    feature_stats = {feature: {} for feature in all_features}
    selected_parts: list[pd.DataFrame] = []
    observed_count = 0
    while observed_count < budget:
        current_batch_size = min(batch_size, budget - observed_count)
        if observed_count == 0:
            batch_df = select_static_top(
                pool_df, model_cols, batch_size, current_batch_size, ""
            )
        else:
            observed_df = pd.concat(selected_parts, ignore_index=True)
            X_obs = observed_df[model_cols].to_numpy(dtype=float)
            y_obs = observed_df["target_success"].to_numpy(dtype=float)
            intercept, weights = fit_ridge_weights(X_obs, y_obs)
            X_pool = pool_df[model_cols].to_numpy(dtype=float)
            mixture_score = pd.Series(intercept + X_pool @ weights, index=pool_df.index)

            feature_score = pd.Series(0.0, index=pool_df.index)
            bonus = pd.Series(0.0, index=pool_df.index)
            for feature in all_features:
                stats = feature_stats[feature]
                means = (
                    pool_df[feature]
                    .map(
                        lambda value: posterior_mean(
                            stats.get(value, [0, 0])[0],
                            stats.get(value, [0, 0])[1],
                            float(feature_priors[feature].get(value, global_prior)),
                            10.0,
                        )
                    )
                    .astype(float)
                )
                confidence = (
                    pool_df[feature]
                    .map(
                        lambda value: 1.0 / math.sqrt(stats.get(value, [0, 0])[1] + 1.0)
                    )
                    .astype(float)
                )
                weight = 0.16 if feature in BASE_FEATURES else 0.06
                feature_score += weight * means
                bonus += 0.03 * confidence

            pool_df = pool_df.assign(
                score=0.20 * pool_df["prior_mean"]
                + 0.50 * mixture_score
                + 0.30 * feature_score
                + bonus,
            )
            batch_df = (
                pool_df.sort_values(
                    [
                        "score",
                        "prior_mean",
                        "train_success_count",
                        "max_risk_severity_score",
                        "attack_id",
                    ],
                    ascending=[False, False, False, False, True],
                )
                .head(current_batch_size)
                .copy()
            )

        selected_parts.append(batch_df)
        observed_count += len(batch_df)
        for row in batch_df.to_dict("records"):
            label = int(row["target_success"])
            for feature in all_features:
                stats = feature_stats[feature].setdefault(row[feature], [0, 0])
                stats[0] += label
                stats[1] += 1
        pool_df = pool_df.loc[~pool_df["attack_id"].isin(batch_df["attack_id"])].copy()

    return pd.concat(selected_parts, ignore_index=True)


METHOD_REGISTRY: dict[str, SelectionFn] = {
    "baseline_random": select_baseline_random,
    "weighted_success_sampling": select_weighted_success_sampling,
    "static_top": select_static_top,
    "adaptive_baseline": select_adaptive_baseline,
    "adaptive_transferability": select_adaptive_transferability,
    "model_mixture": select_model_mixture,
    "contextual_ucb": select_contextual_ucb,
    "hybrid_mixture_ucb": select_hybrid_mixture_ucb,
}
