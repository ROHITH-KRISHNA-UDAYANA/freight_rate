"""Native-categorical LightGBM models for the freight-rate assessment.

This module intentionally bypasses the linear model's one-hot ``FeatureBuilder``.
LightGBM receives pandas categorical columns directly, allowing tree paths to
learn pickup/delivery/equipment interactions and repeated-lane effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd

from .modeling import FreightCleaner, TARGET


FeatureSet = Literal["full", "common"]
TargetTransform = Literal["raw", "log"]
Objective = Literal["l1", "l2"]


class LightGBMFeatureBuilder:
    """Create native categorical and compact numeric tree features."""

    BASE_CATEGORICAL = ("pickup", "delivery", "equipment")

    def __init__(self, feature_set: FeatureSet, include_lane: bool = True) -> None:
        if feature_set not in {"full", "common"}:
            raise ValueError("feature_set must be 'full' or 'common'")
        self.feature_set = feature_set
        self.include_lane = include_lane

    @property
    def categorical_features(self) -> list[str]:
        columns = list(self.BASE_CATEGORICAL)
        if self.include_lane:
            columns.append("lane")
        return columns

    @staticmethod
    def _lane(frame: pd.DataFrame) -> pd.Series:
        return frame["pickup"].astype(str) + " -> " + frame["delivery"].astype(str)

    def fit(self, frame: pd.DataFrame) -> "LightGBMFeatureBuilder":
        self.category_levels: dict[str, list[str]] = {}
        for column in self.BASE_CATEGORICAL:
            self.category_levels[column] = sorted(frame[column].astype(str).unique().tolist())
        if self.include_lane:
            self.category_levels["lane"] = sorted(self._lane(frame).unique().tolist())
        prepared = self._prepare(frame)
        self.feature_columns = prepared.columns.tolist()
        return self

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        dates = pd.to_datetime(frame["date"], errors="raise")
        day_of_week = dates.dt.dayofweek.astype(float)
        day_of_year = dates.dt.dayofyear.astype(float)
        distance = frame["distance"].astype(float)
        weight = frame["weight"].astype(float)

        values: dict[str, pd.Series | np.ndarray] = {
            "pickup": frame["pickup"].astype(str),
            "delivery": frame["delivery"].astype(str),
            "equipment": frame["equipment"].astype(str),
            "distance": distance,
            "log_distance": np.log(distance),
            "sqrt_distance": np.sqrt(distance),
            "weight": weight,
            "log_weight": np.log(weight),
            "distance_x_weight": distance * weight / 1_000_000.0,
            "weight_was_missing": frame["weight_was_missing"].astype(float),
            "weight_was_negative": frame["weight_was_negative"].astype(float),
            "date_ordinal": (dates - pd.Timestamp("2025-01-01")).dt.days.astype(float),
            "day_of_week": day_of_week,
            "day_of_year": day_of_year,
            "weekday_sin": np.sin(2.0 * np.pi * day_of_week / 7.0),
            "weekday_cos": np.cos(2.0 * np.pi * day_of_week / 7.0),
            "annual_sin": np.sin(2.0 * np.pi * day_of_year / 365.25),
            "annual_cos": np.cos(2.0 * np.pi * day_of_year / 365.25),
        }
        if self.include_lane:
            values["lane"] = self._lane(frame)

        if self.feature_set == "full":
            market = frame["market_index"].astype(float)
            quote = frame["quote_signal"].astype(float)
            values.update(
                {
                    "pickup_lat": frame["pickup_lat"].astype(float),
                    "pickup_lon": frame["pickup_lon"].astype(float),
                    "delivery_lat": frame["delivery_lat"].astype(float),
                    "delivery_lon": frame["delivery_lon"].astype(float),
                    "market_index": market,
                    "market_index_was_missing": frame["market_index_was_missing"].astype(float),
                    "quote_signal": quote,
                    "distance_x_market_index": distance * market,
                    "distance_x_quote_signal": distance * quote,
                    "distance_x_market_x_quote": distance * market * quote,
                }
            )

        prepared = pd.DataFrame(values, index=frame.index)
        for column in self.categorical_features:
            levels = self.category_levels[column]
            # Categories absent from the fit period become LightGBM missing
            # values. Full-feature models can still generalize using coordinates.
            prepared[column] = pd.Categorical(prepared[column], categories=levels)
        return prepared

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "feature_columns"):
            raise RuntimeError("LightGBMFeatureBuilder must be fit before transform")
        return self._prepare(frame)[self.feature_columns]

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


def _target_values(values: np.ndarray, transform: TargetTransform) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if transform == "log":
        if (values <= 0).any():
            raise ValueError("Log target requires positive posted_rate values")
        return np.log(values)
    return values


def _inverse_target(values: np.ndarray, transform: TargetTransform) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.exp(values) if transform == "log" else values


def dollar_mae_eval(transform: TargetTransform):
    """Return an early-stopping metric measured in original rate dollars."""

    def evaluate(prediction: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
        actual = _inverse_target(dataset.get_label(), transform)
        predicted = _inverse_target(prediction, transform)
        return "mae_dollars", float(np.mean(np.abs(predicted - actual))), False

    return evaluate


@dataclass
class FittedLightGBMModel:
    cleaner: FreightCleaner
    features: LightGBMFeatureBuilder
    booster: lgb.Booster
    target_transform: TargetTransform
    best_iteration: int
    lower_bound: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        clean = self.cleaner.transform(frame)
        matrix = self.features.transform(clean)
        prediction = self.booster.predict(matrix, num_iteration=self.best_iteration)
        return np.maximum(_inverse_target(prediction, self.target_transform), self.lower_bound)

    def feature_importance(self) -> pd.DataFrame:
        gain = self.booster.feature_importance(
            importance_type="gain", iteration=self.best_iteration
        ).astype(float)
        split = self.booster.feature_importance(
            importance_type="split", iteration=self.best_iteration
        ).astype(int)
        total_gain = float(gain.sum())
        return (
            pd.DataFrame(
                {
                    "feature": self.booster.feature_name(),
                    "gain": gain,
                    "gain_percent": gain / total_gain * 100.0 if total_gain else 0.0,
                    "split_count": split,
                }
            )
            .sort_values(["gain", "split_count"], ascending=False, ignore_index=True)
        )


def lightgbm_parameters(config: dict[str, Any]) -> dict[str, Any]:
    objective = config["objective"]
    subsample = float(config.get("subsample", 1.0))
    return {
        "objective": "regression_l1" if objective == "l1" else "regression_l2",
        "metric": "None",
        "boosting_type": "gbdt",
        "learning_rate": float(config["learning_rate"]),
        "num_leaves": int(config["num_leaves"]),
        "min_data_in_leaf": int(config["min_child_samples"]),
        "max_depth": -1,
        "feature_fraction": float(config.get("colsample_bytree", 1.0)),
        "bagging_fraction": subsample,
        "bagging_freq": 1 if subsample < 1.0 else 0,
        "cat_smooth": float(config.get("cat_smooth", 10.0)),
        "cat_l2": float(config.get("cat_l2", 10.0)),
        "min_data_per_group": 10,
        "max_cat_threshold": 64,
        "max_cat_to_onehot": 4,
        "verbosity": -1,
        "seed": 42,
        "feature_fraction_seed": 42,
        "bagging_seed": 42,
        "data_random_seed": 42,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": int(config.get("num_threads", 0)),
    }


def fit_lightgbm_model(
    train: pd.DataFrame,
    feature_set: FeatureSet,
    config: dict[str, Any],
    target_transform: TargetTransform,
    evaluation: pd.DataFrame | None = None,
    num_boost_round: int = 5_000,
    early_stopping_rounds: int = 100,
) -> FittedLightGBMModel:
    """Fit with optional early stopping on an explicitly supplied later slice."""

    cleaner = FreightCleaner()
    clean_train = cleaner.fit_transform(train)
    features = LightGBMFeatureBuilder(
        feature_set=feature_set,
        include_lane=bool(config.get("include_lane", True)),
    )
    train_matrix = features.fit_transform(clean_train)
    train_target_raw = clean_train[TARGET].to_numpy(dtype=float)
    train_target = _target_values(train_target_raw, target_transform)
    train_dataset = lgb.Dataset(
        train_matrix,
        label=train_target,
        categorical_feature=features.categorical_features,
        free_raw_data=False,
    )

    valid_sets: list[lgb.Dataset] | None = None
    valid_names: list[str] | None = None
    callbacks: list[Any] = [lgb.log_evaluation(period=0)]
    if evaluation is not None:
        clean_evaluation = cleaner.transform(evaluation)
        evaluation_matrix = features.transform(clean_evaluation)
        evaluation_target = _target_values(
            clean_evaluation[TARGET].to_numpy(dtype=float), target_transform
        )
        evaluation_dataset = lgb.Dataset(
            evaluation_matrix,
            label=evaluation_target,
            reference=train_dataset,
            categorical_feature=features.categorical_features,
            free_raw_data=False,
        )
        valid_sets = [evaluation_dataset]
        valid_names = ["august"]
        callbacks.append(
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds,
                first_metric_only=True,
                verbose=False,
            )
        )

    booster = lgb.train(
        params=lightgbm_parameters(config),
        train_set=train_dataset,
        num_boost_round=int(num_boost_round),
        valid_sets=valid_sets,
        valid_names=valid_names,
        feval=dollar_mae_eval(target_transform),
        callbacks=callbacks,
    )
    best_iteration = int(booster.best_iteration or num_boost_round)
    lower_bound = max(1.0, float(np.quantile(train_target_raw, 0.001)) * 0.25)
    return FittedLightGBMModel(
        cleaner=cleaner,
        features=features,
        booster=booster,
        target_transform=target_transform,
        best_iteration=best_iteration,
        lower_bound=lower_bound,
    )
