"""Benchmark models, select on a chronological holdout, and create outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .modeling import (
    TARGET,
    LaneResidualAdjustment,
    fit_freight_model,
    regression_metrics,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
TUNING_START = pd.Timestamp("2025-08-01")
HOLDOUT_START = pd.Timestamp("2025-09-01")
FINAL_VALIDATION_START = pd.Timestamp("2025-11-01")
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1_000.0, 1_500.0, 2_500.0, 5_000.0)
HUBER_DELTAS = (1.0, 1.5, 2.0)
LANE_SMOOTHING_VALUES = (5.0, 15.0, 40.0)


class Tee:
    def __init__(self, *streams: object) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def split_by_date(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(frame["date"], errors="raise")
    inner_train = frame.loc[dates < TUNING_START].copy()
    tuning = frame.loc[(dates >= TUNING_START) & (dates < HOLDOUT_START)].copy()
    holdout = frame.loc[dates >= HOLDOUT_START].copy()
    if holdout.empty or pd.to_datetime(holdout["date"]).nunique() != 61:
        raise ValueError("Expected a 61-day September-October holdout")
    return inner_train, tuning, holdout


def baseline_predictions(name: str, train: pd.DataFrame, predict: pd.DataFrame) -> np.ndarray:
    if name == "global_median":
        return np.full(len(predict), float(train[TARGET].median()))
    if name == "median_rate_per_mile":
        median_rate_per_mile = float((train[TARGET] / train["distance"]).median())
        return predict["distance"].to_numpy(dtype=float) * median_rate_per_mile
    raise ValueError(f"Unknown baseline: {name}")


def tune_candidate(
    name: str,
    inner_train: pd.DataFrame,
    tuning: pd.DataFrame,
    feature_set: str,
    family: str,
    use_lane_adjustment: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    target = tuning[TARGET].to_numpy(dtype=float)
    trials: list[tuple[dict[str, float], dict[str, float]]] = []
    deltas = HUBER_DELTAS if family == "huber" else (1.5,)
    for alpha in ALPHAS:
        for delta in deltas:
            # Fit the expensive base estimator once per alpha/delta pair. Lane
            # smoothing changes only the lightweight residual lookup.
            model = fit_freight_model(
                inner_train,
                feature_set=feature_set,
                family=family,
                alpha=alpha,
                delta=delta,
                lane_smoothing=None,
            )
            if use_lane_adjustment:
                clean_inner = model.cleaner.transform(inner_train)
                clean_tuning = model.cleaner.transform(tuning)
                inner_base = model.regressor.predict(model.features.transform(clean_inner))
                tuning_base = model.regressor.predict(model.features.transform(clean_tuning))
                smoothings: tuple[float | None, ...] = LANE_SMOOTHING_VALUES
            else:
                clean_inner = None
                clean_tuning = None
                inner_base = None
                tuning_base = model.predict(tuning)
                smoothings = (None,)
            for smoothing in smoothings:
                if smoothing is None:
                    prediction = tuning_base
                else:
                    adjustment = LaneResidualAdjustment(smoothing).fit(
                        clean_inner,
                        inner_train[TARGET].to_numpy(dtype=float),
                        inner_base,
                    )
                    prediction = tuning_base + adjustment.predict(clean_tuning)
                    prediction = np.maximum(prediction, model.lower_bound)
                metrics = regression_metrics(target, prediction)
                params = {"alpha": alpha, "delta": delta}
                if smoothing is not None:
                    params["lane_smoothing"] = smoothing
                trials.append((params, metrics))
    best = min(trials, key=lambda item: (item[1]["mae"], item[1]["rmse"], item[1]["mape_percent"]))
    print(
        f"Tuned {name}: parameters={best[0]}, "
        f"August MAE={best[1]['mae']:.2f}, RMSE={best[1]['rmse']:.2f}, "
        f"MAPE={best[1]['mape_percent']:.2f}%"
    )
    return best


def cleaning_summary(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, frame in [("train_test", train), ("validation", validation)]:
        weight = pd.to_numeric(frame["weight"], errors="coerce")
        market = pd.to_numeric(frame["market_index"], errors="coerce")
        rows.extend(
            [
                {
                    "dataset": name,
                    "issue": "missing_weight",
                    "affected_rows": int(weight.isna().sum()),
                    "treatment": "equipment-specific training median, then global training median",
                },
                {
                    "dataset": name,
                    "issue": "negative_weight",
                    "affected_rows": int((weight < 0).sum()),
                    "treatment": "absolute value (verified sign-flip error pattern)",
                },
                {
                    "dataset": name,
                    "issue": "missing_market_index",
                    "affected_rows": int(market.isna().sum()),
                    "treatment": "same-day batch median, then training-global median",
                },
            ]
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train_test.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    template = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")
    december = pd.read_csv(DATA_DIR / "december_chart_inputs.csv")
    inner_train, tuning, holdout = split_by_date(train)
    model_train = pd.concat([inner_train, tuning], ignore_index=True)

    original_stdout = sys.stdout
    with (OUTPUT_DIR / "training.log").open("w", encoding="utf-8", newline="\n") as log:
        sys.stdout = Tee(original_stdout, log)
        try:
            print("SPOTTER FREIGHT-RATE MODEL TRAINING")
            print(
                "Chronological partitions: "
                f"inner train={len(inner_train):,} rows through 2025-07-31; "
                f"tuning={len(tuning):,} rows in August; "
                f"final holdout={len(holdout):,} rows across Sep-Oct (61 days)."
            )
            print(
                "The 61-day holdout matches the 61-day Nov-Dec final inference horizon and avoids "
                "using future rows to predict the past."
            )

            candidates = [
                {
                    "name": "engineered_ridge_full",
                    "feature_set": "full",
                    "family": "ridge",
                    "lane": False,
                },
                {
                    "name": "engineered_log_ridge_full",
                    "feature_set": "full",
                    "family": "log_ridge",
                    "lane": False,
                },
                {
                    "name": "robust_huber_full",
                    "feature_set": "full",
                    "family": "huber",
                    "lane": False,
                },
                {
                    "name": "robust_huber_lane_full",
                    "feature_set": "full",
                    "family": "huber",
                    "lane": True,
                },
                {
                    "name": "engineered_ridge_common",
                    "feature_set": "common",
                    "family": "ridge",
                    "lane": False,
                },
                {
                    "name": "robust_huber_common",
                    "feature_set": "common",
                    "family": "huber",
                    "lane": False,
                },
                {
                    "name": "robust_huber_lane_common",
                    "feature_set": "common",
                    "family": "huber",
                    "lane": True,
                },
            ]

            tuned: dict[str, dict[str, object]] = {}
            for candidate in candidates:
                params, tuning_metrics = tune_candidate(
                    candidate["name"],
                    inner_train,
                    tuning,
                    candidate["feature_set"],
                    candidate["family"],
                    candidate["lane"],
                )
                tuned[candidate["name"]] = {
                    **candidate,
                    "params": params,
                    "tuning_metrics": tuning_metrics,
                }

            results: list[dict[str, object]] = []
            holdout_target = holdout[TARGET].to_numpy(dtype=float)
            for baseline in ["global_median", "median_rate_per_mile"]:
                metrics = regression_metrics(
                    holdout_target, baseline_predictions(baseline, model_train, holdout)
                )
                results.append(
                    {
                        "model": baseline,
                        "feature_set": "baseline",
                        "selection_split": "2025-09-01 to 2025-10-31",
                        **metrics,
                        "tuned_parameters": "{}",
                    }
                )

            fitted_holdout_models: dict[str, object] = {}
            for name, specification in tuned.items():
                params = specification["params"]
                model = fit_freight_model(
                    model_train,
                    feature_set=specification["feature_set"],
                    family=specification["family"],
                    alpha=params["alpha"],
                    delta=params["delta"],
                    lane_smoothing=params.get("lane_smoothing"),
                )
                fitted_holdout_models[name] = model
                metrics = regression_metrics(holdout_target, model.predict(holdout))
                results.append(
                    {
                        "model": name,
                        "feature_set": specification["feature_set"],
                        "selection_split": "2025-09-01 to 2025-10-31",
                        **metrics,
                        "tuned_parameters": json.dumps(params, sort_keys=True),
                    }
                )

            comparison = pd.DataFrame(results).sort_values(
                ["mae", "rmse", "mape_percent"], ignore_index=True
            )
            comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, float_format="%.6f")
            print("\nFINAL CHRONOLOGICAL HOLDOUT RESULTS")
            print(
                comparison[["model", "feature_set", "mae", "rmse", "mape_percent", "tuned_parameters"]]
                .to_string(index=False, float_format=lambda value: f"{value:,.3f}")
            )

            advanced = comparison[~comparison["feature_set"].eq("baseline")]
            primary_name = str(advanced.iloc[0]["model"])
            common = advanced[advanced["feature_set"].eq("common")]
            december_name = str(common.iloc[0]["model"])
            print(f"\nSelected primary model by lowest holdout MAE: {primary_name}")
            print(f"Selected December-compatible model by lowest holdout MAE: {december_name}")

            primary_holdout_prediction = fitted_holdout_models[primary_name].predict(holdout)
            monthly_rows: list[dict[str, object]] = []
            holdout_month = pd.to_datetime(holdout["date"]).dt.to_period("M")
            for month in sorted(holdout_month.unique()):
                mask = holdout_month.eq(month).to_numpy()
                monthly_rows.append(
                    {
                        "month": str(month),
                        "rows": int(mask.sum()),
                        **regression_metrics(
                            holdout_target[mask], primary_holdout_prediction[mask]
                        ),
                    }
                )
            pd.DataFrame(monthly_rows).to_csv(
                OUTPUT_DIR / "holdout_monthly_metrics.csv", index=False, float_format="%.6f"
            )

            primary_spec = tuned[primary_name]
            primary_params = primary_spec["params"]
            primary_model = fit_freight_model(
                train,
                feature_set=primary_spec["feature_set"],
                family=primary_spec["family"],
                alpha=primary_params["alpha"],
                delta=primary_params["delta"],
                lane_smoothing=primary_params.get("lane_smoothing"),
            )
            validation_prediction = primary_model.predict(validation)

            if list(template.columns) != ["load_id", "predicted_rate"]:
                raise ValueError("Prediction template schema is not load_id,predicted_rate")
            if not template["load_id"].equals(validation["load_id"]):
                raise ValueError("Template IDs do not exactly match validation IDs and order")
            submission = template.copy()
            submission["predicted_rate"] = np.round(validation_prediction, 2)
            submission.to_csv(ROOT / "validation_predictions.csv", index=False)

            december_spec = tuned[december_name]
            december_params = december_spec["params"]
            december_model = fit_freight_model(
                train,
                feature_set="common",
                family=december_spec["family"],
                alpha=december_params["alpha"],
                delta=december_params["delta"],
                lane_smoothing=december_params.get("lane_smoothing"),
            )
            december_output = december.copy()
            december_output["predicted_rate"] = np.round(december_model.predict(december_output), 2)
            december_output.to_csv(DATA_DIR / "december_chart_inputs.csv", index=False)

            cleaning = cleaning_summary(train, validation)
            cleaning.to_csv(OUTPUT_DIR / "data_cleaning_summary.csv", index=False)
            split_summary = pd.DataFrame(
                [
                    {
                        "partition": "inner_train",
                        "start_date": pd.to_datetime(inner_train["date"]).min().date(),
                        "end_date": pd.to_datetime(inner_train["date"]).max().date(),
                        "rows": len(inner_train),
                        "purpose": "fit candidates during hyperparameter tuning",
                    },
                    {
                        "partition": "tuning",
                        "start_date": pd.to_datetime(tuning["date"]).min().date(),
                        "end_date": pd.to_datetime(tuning["date"]).max().date(),
                        "rows": len(tuning),
                        "purpose": "select regularization and robustness parameters",
                    },
                    {
                        "partition": "holdout",
                        "start_date": pd.to_datetime(holdout["date"]).min().date(),
                        "end_date": pd.to_datetime(holdout["date"]).max().date(),
                        "rows": len(holdout),
                        "purpose": "unbiased final model-family comparison",
                    },
                ]
            )
            split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

            selection = {
                "selection_metric": "MAE",
                "primary_model": primary_name,
                "primary_parameters": primary_params,
                "december_model": december_name,
                "december_parameters": december_params,
                "primary_holdout_metrics": comparison.set_index("model").loc[
                    primary_name, ["mae", "rmse", "mape_percent"]
                ].to_dict(),
                "december_holdout_metrics": comparison.set_index("model").loc[
                    december_name, ["mae", "rmse", "mape_percent"]
                ].to_dict(),
            }
            with (OUTPUT_DIR / "model_selection.json").open("w", encoding="utf-8") as handle:
                json.dump(selection, handle, indent=2)

            print(
                f"\nWrote {len(submission):,} validation predictions; range "
                f"${submission['predicted_rate'].min():,.2f} to ${submission['predicted_rate'].max():,.2f}."
            )
            print(
                f"Wrote {len(december_output):,} December predictions; range "
                f"${december_output['predicted_rate'].min():,.2f} to "
                f"${december_output['predicted_rate'].max():,.2f}."
            )
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
