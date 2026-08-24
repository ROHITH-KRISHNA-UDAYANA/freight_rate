"""Tune native-categorical LightGBM without touching the final holdout early.

Workflow:
1. Fit candidate configurations on January-July and early-stop on August.
2. Freeze each objective/target variant and refit on January-August for its
   single September-October evaluation.
3. If LightGBM wins holdout MAE, refit on all labeled data using the already
   selected iteration count and regenerate the official prediction artifacts.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from .lightgbm_model import fit_lightgbm_model
from .modeling import TARGET, fit_freight_model, regression_metrics
from .paths import (
    DATA_DIR,
    LOGS_DIR,
    METADATA_DIR,
    METRICS_DIR,
    ROOT,
    ensure_output_directories,
)
from .train import Tee, split_by_date


NUM_LEAVES = (63, 100, 127, 150)
LEARNING_RATES = (0.01, 0.02, 0.03)
MIN_CHILD_SAMPLES = (10, 12, 15, 20)
VARIANTS = (("l1", "raw"), ("l1", "log"), ("l2", "raw"), ("l2", "log"))
EARLY_STOPPING_ROUNDS = 100
MAX_ESTIMATORS = 5_000
HOLDOUT_LABEL = "2025-09-01 to 2025-10-31"


def base_config(objective: str) -> dict[str, Any]:
    return {
        "objective": objective,
        "num_leaves": 127,
        "learning_rate": 0.02,
        "min_child_samples": 12,
        "max_depth": -1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "cat_smooth": 10.0,
        "cat_l2": 10.0,
        "include_lane": False,
        "num_threads": 0,
    }


def config_key(feature_set: str, target_transform: str, config: dict[str, Any]) -> str:
    stable = {key: value for key, value in config.items() if key != "num_threads"}
    return json.dumps(
        {"feature_set": feature_set, "target_transform": target_transform, **stable},
        sort_keys=True,
    )


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "num_threads"}


def tune_feature_set(
    inner_train: pd.DataFrame,
    tuning: pd.DataFrame,
    feature_set: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Screen four variants, then tune the August winner in a focused search."""

    cache: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    tuning_target = tuning[TARGET].to_numpy(dtype=float)

    def trial(
        objective: str,
        target_transform: str,
        config: dict[str, Any],
        stage: str,
    ) -> dict[str, Any]:
        key = config_key(feature_set, target_transform, config)
        if key in cache:
            return cache[key]
        started = time.perf_counter()
        model = fit_lightgbm_model(
            inner_train,
            feature_set=feature_set,
            config=config,
            target_transform=target_transform,
            evaluation=tuning,
            num_boost_round=MAX_ESTIMATORS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        metrics = regression_metrics(tuning_target, model.predict(tuning))
        row: dict[str, Any] = {
            "feature_set": feature_set,
            "objective": objective,
            "target_transform": target_transform,
            "stage": stage,
            **public_config(config),
            "best_iteration": model.best_iteration,
            "august_mae": metrics["mae"],
            "august_rmse": metrics["rmse"],
            "august_mape_percent": metrics["mape_percent"],
            "august_r2": metrics["r2"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        cache[key] = row
        results.append(row)
        print(
            f"{feature_set:6s} {stage:13s} {objective}/{target_transform:3s} "
            f"leaves={config['num_leaves']:3d} lr={config['learning_rate']:.2f} "
            f"child={config['min_child_samples']:2d} iter={model.best_iteration:4d} "
            f"Aug MAE={metrics['mae']:.2f} MAPE={metrics['mape_percent']:.3f}% "
            f"({row['elapsed_seconds']:.1f}s)",
            flush=True,
        )
        return row

    # Objective and target-transform comparison at the same representative
    # configuration. Holdout remains completely untouched here.
    frozen: dict[tuple[str, str], dict[str, Any]] = {}
    screen_rows: list[dict[str, Any]] = []
    for objective, target_transform in VARIANTS:
        config = base_config(objective)
        row = trial(objective, target_transform, config, "screen")
        screen_rows.append(row)
        frozen[(objective, target_transform)] = {
            "config": public_config(config),
            "best_iteration": int(row["best_iteration"]),
            "august_metrics": {
                "mae": row["august_mae"],
                "rmse": row["august_rmse"],
                "mape_percent": row["august_mape_percent"],
                "r2": row["august_r2"],
            },
        }

    screen_winner = min(
        screen_rows,
        key=lambda row: (row["august_mae"], row["august_mape_percent"], row["august_rmse"]),
    )
    winning_variant = (screen_winner["objective"], screen_winner["target_transform"])
    objective, target_transform = winning_variant
    print(
        f"{feature_set} screen winner: {objective}/{target_transform} "
        f"at August MAE={screen_winner['august_mae']:.2f}",
        flush=True,
    )

    # Search the requested leaf/learning-rate neighborhood with the central
    # min-child setting, then refine min-child around the two best structures.
    grid_rows: list[dict[str, Any]] = []
    for num_leaves in NUM_LEAVES:
        for learning_rate in LEARNING_RATES:
            config = base_config(objective)
            config.update(
                {
                    "num_leaves": num_leaves,
                    "learning_rate": learning_rate,
                    "min_child_samples": 12,
                }
            )
            grid_rows.append(trial(objective, target_transform, config, "leaf_lr_grid"))

    best_structures: list[tuple[int, float]] = []
    for row in sorted(grid_rows, key=lambda item: item["august_mae"]):
        structure = (int(row["num_leaves"]), float(row["learning_rate"]))
        if structure not in best_structures:
            best_structures.append(structure)
        if len(best_structures) == 2:
            break
    for num_leaves, learning_rate in best_structures:
        for min_child_samples in MIN_CHILD_SAMPLES:
            config = base_config(objective)
            config.update(
                {
                    "num_leaves": num_leaves,
                    "learning_rate": learning_rate,
                    "min_child_samples": min_child_samples,
                }
            )
            trial(objective, target_transform, config, "child_refine")

    best_so_far = min(
        [row for row in results if (row["objective"], row["target_transform"]) == winning_variant],
        key=lambda row: (row["august_mae"], row["august_mape_percent"]),
    )
    best_config = {
        key: best_so_far[key]
        for key in base_config(objective)
        if key != "num_threads"
    }
    best_config["num_threads"] = 0

    # Verify whether stochastic row/column sampling helps. The full-data
    # configuration is retained unless a sampled version wins on August.
    for subsample, colsample in [(0.9, 1.0), (1.0, 0.9), (0.9, 0.9), (0.8, 0.8)]:
        config = dict(best_config)
        config["subsample"] = subsample
        config["colsample_bytree"] = colsample
        trial(objective, target_transform, config, "sampling_check")

    # Directly verify explicit lane categorization and categorical smoothing.
    config = dict(best_config)
    config["include_lane"] = True
    trial(objective, target_transform, config, "lane_check")
    for cat_smooth, cat_l2 in [(5.0, 10.0), (20.0, 10.0), (10.0, 5.0), (10.0, 20.0)]:
        config = dict(best_config)
        config["cat_smooth"] = cat_smooth
        config["cat_l2"] = cat_l2
        trial(objective, target_transform, config, "category_reg")

    tuned_rows = [
        row
        for row in results
        if (row["objective"], row["target_transform"]) == winning_variant
    ]
    tuned_winner = min(
        tuned_rows,
        key=lambda row: (row["august_mae"], row["august_mape_percent"], row["august_rmse"]),
    )
    tuned_config = {
        key: tuned_winner[key]
        for key in base_config(objective)
        if key != "num_threads"
    }
    frozen[winning_variant] = {
        "config": tuned_config,
        "best_iteration": int(tuned_winner["best_iteration"]),
        "august_metrics": {
            "mae": tuned_winner["august_mae"],
            "rmse": tuned_winner["august_rmse"],
            "mape_percent": tuned_winner["august_mape_percent"],
            "r2": tuned_winner["august_r2"],
        },
    }
    print(
        f"Frozen {feature_set} winner before holdout: {objective}/{target_transform}, "
        f"iteration={tuned_winner['best_iteration']}, config={tuned_config}",
        flush=True,
    )
    return results, frozen


def outlier_feature_audit(inner_train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare predictable features for extreme residual-ratio labels."""

    baseline = fit_freight_model(
        inner_train,
        feature_set="full",
        family="huber",
        alpha=1.0,
        delta=1.0,
    )
    baseline_prediction = baseline.predict(inner_train)
    ratio = inner_train[TARGET].to_numpy(dtype=float) / baseline_prediction
    outlier = ratio >= 2.5
    clean = baseline.cleaner.transform(inner_train)
    rows: list[dict[str, Any]] = []

    numeric_columns = [
        "distance",
        "weight",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "market_index",
        "quote_signal",
    ]
    for column in numeric_columns:
        outlier_values = clean.loc[outlier, column].astype(float)
        normal_values = clean.loc[~outlier, column].astype(float)
        pooled_std = float(np.sqrt((outlier_values.var() + normal_values.var()) / 2.0))
        standardized_difference = (
            float((outlier_values.mean() - normal_values.mean()) / pooled_std)
            if pooled_std
            else 0.0
        )
        ks = ks_2samp(outlier_values, normal_values, alternative="two-sided", method="auto")
        rows.append(
            {
                "feature": column,
                "feature_type": "numeric",
                "outlier_summary": float(outlier_values.mean()),
                "normal_summary": float(normal_values.mean()),
                "standardized_mean_difference": standardized_difference,
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "total_variation_distance": np.nan,
                "chi_square_pvalue": np.nan,
                "cramers_v": np.nan,
            }
        )

    for column in ["pickup", "delivery", "equipment"]:
        outlier_distribution = clean.loc[outlier, column].value_counts(normalize=True)
        normal_distribution = clean.loc[~outlier, column].value_counts(normalize=True)
        levels = outlier_distribution.index.union(normal_distribution.index)
        outlier_aligned = outlier_distribution.reindex(levels, fill_value=0.0)
        normal_aligned = normal_distribution.reindex(levels, fill_value=0.0)
        contingency = pd.crosstab(outlier, clean[column])
        chi_square, chi_pvalue, _, _ = chi2_contingency(contingency)
        denominator = len(clean) * min(
            contingency.shape[0] - 1, contingency.shape[1] - 1
        )
        rows.append(
            {
                "feature": column,
                "feature_type": "categorical",
                "outlier_summary": float(outlier_aligned.max()),
                "normal_summary": float(normal_aligned.max()),
                "standardized_mean_difference": np.nan,
                "ks_statistic": np.nan,
                "ks_pvalue": np.nan,
                "total_variation_distance": float(
                    0.5 * np.abs(outlier_aligned - normal_aligned).sum()
                ),
                "chi_square_pvalue": float(chi_pvalue),
                "cramers_v": float(np.sqrt(chi_square / denominator))
                if denominator
                else 0.0,
            }
        )

    audit = pd.DataFrame(rows)
    summary = {
        "baseline": "January-July robust Huber ridge, evaluated in-sample only for label audit",
        "outlier_definition": "posted_rate / baseline_prediction >= 2.5",
        "outlier_rows": int(outlier.sum()),
        "outlier_percent": float(outlier.mean() * 100.0),
        "outlier_ratio_min": float(ratio[outlier].min()),
        "outlier_ratio_max": float(ratio[outlier].max()),
        "maximum_absolute_numeric_smd": float(
            audit.loc[audit["feature_type"].eq("numeric"), "standardized_mean_difference"]
            .abs()
            .max()
        ),
        "maximum_categorical_cramers_v": float(
            audit.loc[audit["feature_type"].eq("categorical"), "cramers_v"].max()
        ),
        "interpretation": (
            "Extreme labels are rare. Numeric distribution tests find no meaningful "
            "separation; categorical effect sizes are also tiny even where a sparse "
            "chi-square test detects a delivery mix difference. Their extreme magnitude "
            "is not learnable from the supplied predictors."
        ),
    }
    return audit, summary


def add_r2_to_existing(comparison: pd.DataFrame, holdout_target: np.ndarray) -> pd.DataFrame:
    result = comparison.copy()
    variance = float(np.mean((holdout_target - holdout_target.mean()) ** 2))
    if "r2" not in result:
        result["r2"] = 1.0 - result["rmse"].astype(float) ** 2 / variance
    else:
        missing = result["r2"].isna()
        result.loc[missing, "r2"] = 1.0 - result.loc[missing, "rmse"].astype(float) ** 2 / variance
    return result


def main() -> None:
    ensure_output_directories()
    train = pd.read_csv(DATA_DIR / "train_test.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    template = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")
    december_template = pd.read_csv(DATA_DIR / "december_chart_inputs_template.csv")
    inner_train, tuning, holdout = split_by_date(train)
    model_train = pd.concat([inner_train, tuning], ignore_index=True)
    holdout_target = holdout[TARGET].to_numpy(dtype=float)

    existing_comparison = pd.read_csv(METRICS_DIR / "model_comparison.csv")
    if existing_comparison["model"].astype(str).str.startswith("lightgbm_").any():
        raise RuntimeError("LightGBM rows already exist; refusing to duplicate holdout evaluations")

    original_stdout = sys.stdout
    with (LOGS_DIR / "lgbm_training.log").open("w", encoding="utf-8", newline="\n") as log:
        sys.stdout = Tee(original_stdout, log)
        try:
            print("SPOTTER NATIVE-CATEGORICAL LIGHTGBM TUNING", flush=True)
            print(
                f"inner_train={len(inner_train):,}; August={len(tuning):,}; "
                f"untouched holdout={len(holdout):,}",
                flush=True,
            )
            print(
                "All configuration choices below use August dollar MAE only. "
                "Sep-Oct is evaluated once per frozen variant after tuning ends.",
                flush=True,
            )

            outlier_audit, outlier_summary = outlier_feature_audit(inner_train)
            print(f"Outlier audit: {json.dumps(outlier_summary, sort_keys=True)}", flush=True)
            raw_correlation = float(train["distance"].corr(train[TARGET]))
            log_correlation = float(np.log(train["distance"]).corr(np.log(train[TARGET])))
            print(
                f"Distance/target correlation: raw={raw_correlation:.6f}; "
                f"log-log={log_correlation:.6f}",
                flush=True,
            )

            all_tuning_rows: list[dict[str, Any]] = []
            frozen_by_feature_set: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
            for feature_set in ["full", "common"]:
                rows, frozen = tune_feature_set(inner_train, tuning, feature_set)
                all_tuning_rows.extend(rows)
                frozen_by_feature_set[feature_set] = frozen

            # Tuning is now over. Each frozen objective/target variant gets one
            # and only one holdout evaluation, with no subsequent adjustments.
            holdout_rows: list[dict[str, Any]] = []
            fitted_holdout: dict[tuple[str, str, str], Any] = {}
            print("\nFROZEN SEP-OCT HOLDOUT EVALUATIONS", flush=True)
            for feature_set in ["full", "common"]:
                for objective, target_transform in VARIANTS:
                    specification = frozen_by_feature_set[feature_set][
                        (objective, target_transform)
                    ]
                    config = dict(specification["config"])
                    config["num_threads"] = 0
                    model = fit_lightgbm_model(
                        model_train,
                        feature_set=feature_set,
                        config=config,
                        target_transform=target_transform,
                        evaluation=None,
                        num_boost_round=int(specification["best_iteration"]),
                    )
                    prediction = model.predict(holdout)
                    metrics = regression_metrics(holdout_target, prediction)
                    name = f"lightgbm_{objective}_{target_transform}_{feature_set}"
                    parameters = {
                        **public_config(config),
                        "target_transform": target_transform,
                        "best_iteration": int(specification["best_iteration"]),
                        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                        "max_estimators": MAX_ESTIMATORS,
                    }
                    row = {
                        "model": name,
                        "feature_set": feature_set,
                        "selection_split": HOLDOUT_LABEL,
                        **metrics,
                        "tuned_parameters": json.dumps(parameters, sort_keys=True),
                    }
                    holdout_rows.append(row)
                    fitted_holdout[(feature_set, objective, target_transform)] = model
                    print(
                        f"{name}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, "
                        f"MAPE={metrics['mape_percent']:.3f}%, R2={metrics['r2']:.6f}",
                        flush=True,
                    )

            existing_comparison = add_r2_to_existing(existing_comparison, holdout_target)
            lightgbm_comparison = pd.DataFrame(holdout_rows)
            comparison = pd.concat(
                [existing_comparison, lightgbm_comparison], ignore_index=True
            ).sort_values(["mae", "mape_percent", "rmse"], ignore_index=True)

            # Model selection must remain an August decision. Looking at the
            # holdout table to choose a feature set or target transform would
            # leak Sep-Oct information into the submitted model. Each lookup
            # below therefore uses the frozen August MAE, then merely attaches
            # that already-selected candidate's one-time holdout result.
            selected_by_feature_set: dict[str, tuple[str, str, dict[str, Any]]] = {}
            for feature_set in ["full", "common"]:
                (objective, target_transform), specification = min(
                    frozen_by_feature_set[feature_set].items(),
                    key=lambda item: (
                        item[1]["august_metrics"]["mae"],
                        item[1]["august_metrics"]["mape_percent"],
                        item[1]["august_metrics"]["rmse"],
                    ),
                )
                selected_by_feature_set[feature_set] = (
                    objective,
                    target_transform,
                    specification,
                )

            primary_feature_set = min(
                selected_by_feature_set,
                key=lambda feature_set: (
                    selected_by_feature_set[feature_set][2]["august_metrics"]["mae"],
                    selected_by_feature_set[feature_set][2]["august_metrics"][
                        "mape_percent"
                    ],
                ),
            )
            primary_objective, primary_target, primary_spec = selected_by_feature_set[
                primary_feature_set
            ]
            full_objective, full_target, full_spec = selected_by_feature_set["full"]
            common_objective, common_target, common_spec = selected_by_feature_set["common"]

            def holdout_row(feature_set: str, objective: str, target: str) -> pd.Series:
                model_name = f"lightgbm_{objective}_{target}_{feature_set}"
                return lightgbm_comparison.loc[
                    lightgbm_comparison["model"].eq(model_name)
                ].iloc[0]

            selected_lightgbm_full = holdout_row("full", full_objective, full_target)
            selected_lightgbm_common = holdout_row(
                "common", common_objective, common_target
            )
            selected_lightgbm_primary = holdout_row(
                primary_feature_set, primary_objective, primary_target
            )
            previous_best = existing_comparison.sort_values(["mae", "mape_percent"]).iloc[0]
            lightgbm_wins = float(selected_lightgbm_primary["mae"]) < float(
                previous_best["mae"]
            )
            lightgbm_mape_wins = float(
                selected_lightgbm_primary["mape_percent"]
            ) < float(
                existing_comparison["mape_percent"].min()
            )
            print(
                f"\nBest prior={previous_best['model']} MAE={previous_best['mae']:.3f}; "
                f"August-selected LightGBM={selected_lightgbm_primary['model']} "
                f"MAE={selected_lightgbm_primary['mae']:.3f}; wins_MAE={lightgbm_wins}; "
                f"wins_MAPE={lightgbm_mape_wins}",
                flush=True,
            )

            full_config = dict(full_spec["config"])
            full_config["num_threads"] = 0
            final_full_model = fit_lightgbm_model(
                train,
                feature_set="full",
                config=full_config,
                target_transform=full_target,
                evaluation=None,
                num_boost_round=int(full_spec["best_iteration"]),
            )

            common_config = dict(common_spec["config"])
            common_config["num_threads"] = 0
            final_common_model = fit_lightgbm_model(
                train,
                feature_set="common",
                config=common_config,
                target_transform=common_target,
                evaluation=None,
                num_boost_round=int(common_spec["best_iteration"]),
            )
            if primary_feature_set == "common":
                final_primary_model = final_common_model
                primary_config = common_config
            else:
                final_primary_model = final_full_model
                primary_config = full_config

            if lightgbm_wins:
                validation_prediction = final_primary_model.predict(validation)
                if list(template.columns) != ["load_id", "predicted_rate"]:
                    raise ValueError("Prediction template schema is not load_id,predicted_rate")
                if not template["load_id"].equals(validation["load_id"]):
                    raise ValueError("Prediction template IDs do not match validation order")
                submission = template.copy()
                submission["predicted_rate"] = np.round(validation_prediction, 2)
                submission.to_csv(ROOT / "validation_predictions.csv", index=False)

                december_output = december_template.copy()
                december_output["predicted_rate"] = np.round(
                    final_common_model.predict(december_template), 2
                )
                december_output.to_csv(DATA_DIR / "december_chart_inputs.csv", index=False)
                print(
                    f"Regenerated final outputs with LightGBM: validation range "
                    f"${submission.predicted_rate.min():,.2f}-${submission.predicted_rate.max():,.2f}; "
                    f"December range ${december_output.predicted_rate.min():,.2f}-"
                    f"${december_output.predicted_rate.max():,.2f}.",
                    flush=True,
                )
            else:
                print("LightGBM did not beat prior MAE; official prediction files were left unchanged.")

            selection_path = METADATA_DIR / "model_selection.json"
            with selection_path.open("r", encoding="utf-8") as handle:
                selection = json.load(handle)
            experiment = {
                "august_selected_full_model": str(selected_lightgbm_full["model"]),
                "august_selected_full_parameters": {
                    **public_config(full_config),
                    "target_transform": full_target,
                    "best_iteration": int(full_spec["best_iteration"]),
                },
                "august_selected_full_holdout_metrics": {
                    key: float(selected_lightgbm_full[key])
                    for key in ["mae", "rmse", "mape_percent", "r2"]
                },
                "august_selected_common_model": str(selected_lightgbm_common["model"]),
                "august_selected_common_parameters": {
                    **public_config(common_config),
                    "target_transform": common_target,
                    "best_iteration": int(common_spec["best_iteration"]),
                },
                "august_selected_common_holdout_metrics": {
                    key: float(selected_lightgbm_common[key])
                    for key in ["mae", "rmse", "mape_percent", "r2"]
                },
                "primary_selection_basis": "lowest August MAE before holdout evaluation",
                "beats_prior_mae": bool(lightgbm_wins),
                "beats_prior_mape": bool(lightgbm_mape_wins),
                "holdout_evaluations_per_variant": 1,
            }
            selection["lightgbm_experiment"] = experiment
            if lightgbm_wins:
                selection.update(
                    {
                        "primary_model": str(selected_lightgbm_primary["model"]),
                        "primary_parameters": {
                            **public_config(primary_config),
                            "target_transform": primary_target,
                            "best_iteration": int(primary_spec["best_iteration"]),
                        },
                        "primary_holdout_metrics": {
                            key: float(selected_lightgbm_primary[key])
                            for key in ["mae", "rmse", "mape_percent", "r2"]
                        },
                        "december_model": str(selected_lightgbm_common["model"]),
                        "december_parameters": experiment[
                            "august_selected_common_parameters"
                        ],
                        "december_holdout_metrics": experiment[
                            "august_selected_common_holdout_metrics"
                        ],
                    }
                )

            # Write outputs only after every holdout score is final and frozen.
            comparison.to_csv(
                METRICS_DIR / "model_comparison.csv", index=False, float_format="%.6f"
            )
            pd.DataFrame(all_tuning_rows).to_csv(
                METRICS_DIR / "lgbm_tuning_results.csv", index=False, float_format="%.6f"
            )
            outlier_audit.to_csv(
                METRICS_DIR / "lgbm_outlier_feature_comparison.csv",
                index=False,
                float_format="%.6f",
            )
            final_primary_model.feature_importance().to_csv(
                METRICS_DIR / "lgbm_feature_importance.csv",
                index=False,
                float_format="%.8f",
            )
            with selection_path.open("w", encoding="utf-8") as handle:
                json.dump(selection, handle, indent=2)
            with (METADATA_DIR / "lgbm_verification.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "raw_distance_rate_correlation": raw_correlation,
                        "log_distance_log_rate_correlation": log_correlation,
                        "outlier_audit": outlier_summary,
                        "split": {
                            "inner_train": "2025-01-01 to 2025-07-31",
                            "early_stopping": "2025-08-01 to 2025-08-31",
                            "holdout": HOLDOUT_LABEL,
                        },
                    },
                    handle,
                    indent=2,
                )

            if lightgbm_wins:
                selected_holdout_model = fitted_holdout[
                    (primary_feature_set, primary_objective, primary_target)
                ]
                selected_prediction = selected_holdout_model.predict(holdout)
                holdout_month = pd.to_datetime(holdout["date"]).dt.to_period("M")
                monthly_rows = []
                for month in sorted(holdout_month.unique()):
                    mask = holdout_month.eq(month).to_numpy()
                    monthly_rows.append(
                        {
                            "model": str(selected_lightgbm_primary["model"]),
                            "month": str(month),
                            "rows": int(mask.sum()),
                            **regression_metrics(
                                holdout_target[mask], selected_prediction[mask]
                            ),
                        }
                    )
                pd.DataFrame(monthly_rows).to_csv(
                    METRICS_DIR / "holdout_monthly_metrics.csv",
                    index=False,
                    float_format="%.6f",
                )

            print("\nFINAL MODEL TABLE", flush=True)
            print(
                comparison[
                    ["model", "feature_set", "mae", "rmse", "mape_percent", "r2"]
                ].to_string(index=False, float_format=lambda value: f"{value:,.4f}"),
                flush=True,
            )
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
