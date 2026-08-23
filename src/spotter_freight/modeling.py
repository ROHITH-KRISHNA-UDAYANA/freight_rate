"""Explicit linear freight-rate baselines built with pandas and NumPy.

These implementations keep the original ridge and robust baselines reviewable
and dependency-light. Native-categorical gradient boosting lives separately in
``lightgbm_model.py`` so it cannot accidentally enter this one-hot feature path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TARGET = "posted_rate"


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return holdout error metrics, including coefficient of determination."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual
    total_variation = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape_percent": float(np.mean(np.abs(errors) / np.maximum(np.abs(actual), 1e-9)) * 100.0),
        "r2": float(1.0 - np.sum(errors**2) / total_variation) if total_variation else np.nan,
    }


class FreightCleaner:
    """Apply train-fitted repairs to the observed data-quality defects.

    * Negative weights are sign errors: their absolute-value distribution is
      nearly identical to the valid positive distribution, so taking ``abs``
      recovers information that dropping or median-imputing would discard.
    * Missing weights are imputed with the training median for the same
      equipment type (then the global median as a defensive fallback).
    * Missing market indices are filled with the same-day median computed from
      the feature batch. This macro signal is date-dependent and is observed on
      almost every other load that day. A training-global median remains as a
      fallback for a hypothetical entirely-missing day.
    * Missingness/correction flags are retained so models can learn whether a
      repair carries residual information.
    """

    def fit(self, frame: pd.DataFrame) -> "FreightCleaner":
        positive_weight = pd.to_numeric(frame["weight"], errors="coerce").abs()
        positive_weight = positive_weight.where(positive_weight > 0)
        weight_frame = pd.DataFrame(
            {"equipment": frame["equipment"].astype("string"), "weight": positive_weight}
        )
        self.weight_by_equipment = weight_frame.groupby("equipment")["weight"].median().to_dict()
        self.weight_global = float(positive_weight.median())

        market = pd.to_numeric(frame["market_index"], errors="coerce") if "market_index" in frame else None
        self.market_global = float(market[market > 0].median()) if market is not None else 1.0

        self.numeric_fallbacks: dict[str, float] = {}
        for column in ["distance", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
            if column in frame:
                values = pd.to_numeric(frame[column], errors="coerce")
                self.numeric_fallbacks[column] = float(values.median())
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "weight_global"):
            raise RuntimeError("FreightCleaner must be fit before transform")
        result = frame.copy()

        for column in ["pickup", "delivery", "equipment"]:
            if column in result:
                result[column] = result[column].astype("string").str.strip().fillna("__MISSING__")
                result.loc[result[column].eq(""), column] = "__MISSING__"

        raw_weight = pd.to_numeric(result["weight"], errors="coerce")
        result["weight_was_missing"] = raw_weight.isna().astype(float)
        result["weight_was_negative"] = (raw_weight < 0).astype(float)
        repaired_weight = raw_weight.abs().where(raw_weight.abs() > 0)
        equipment_median = result["equipment"].map(self.weight_by_equipment).astype(float)
        result["weight"] = repaired_weight.fillna(equipment_median).fillna(self.weight_global)

        if "market_index" in result:
            raw_market = pd.to_numeric(result["market_index"], errors="coerce")
            raw_market = raw_market.where(raw_market > 0)
            result["market_index_was_missing"] = raw_market.isna().astype(float)
            parsed_dates = pd.to_datetime(result["date"], errors="coerce")
            same_day_median = raw_market.groupby(parsed_dates).transform("median")
            result["market_index"] = raw_market.fillna(same_day_median).fillna(self.market_global)

        for column, fallback in self.numeric_fallbacks.items():
            if column not in result:
                continue
            values = pd.to_numeric(result[column], errors="coerce")
            if column == "distance":
                values = values.where(values > 0)
            result[f"{column}_was_invalid"] = values.isna().astype(float)
            result[column] = values.fillna(fallback)

        parsed_dates = pd.to_datetime(result["date"], errors="coerce")
        if parsed_dates.isna().any():
            # No invalid dates exist in the supplied files. Raising is safer
            # than inventing a calendar value if a future input breaks schema.
            raise ValueError(f"Found {int(parsed_dates.isna().sum())} invalid dates")
        result["date"] = parsed_dates

        if "quote_signal" in result:
            quote = pd.to_numeric(result["quote_signal"], errors="coerce")
            if quote.isna().any() or (quote <= 0).any():
                raise ValueError("quote_signal must be positive and complete")
            result["quote_signal"] = quote
        return result

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


class FeatureBuilder:
    """Train-fitted nonlinear numeric and one-hot categorical feature map."""

    COMMON_CATEGORICAL = ("pickup", "delivery", "equipment")

    def __init__(self, feature_set: str) -> None:
        if feature_set not in {"full", "common"}:
            raise ValueError("feature_set must be 'full' or 'common'")
        self.feature_set = feature_set

    def fit(self, frame: pd.DataFrame) -> "FeatureBuilder":
        self.categories = {
            column: sorted(frame[column].astype(str).unique().tolist())
            for column in self.COMMON_CATEGORICAL
        }
        distance = frame["distance"].astype(float)
        self.distance_knots = np.unique(distance.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).to_numpy())
        numeric = self._numeric_features(frame)
        self.numeric_columns = numeric.columns.tolist()
        self.numeric_means = numeric.mean(axis=0)
        std = numeric.std(axis=0, ddof=0)
        self.numeric_stds = std.mask(std < 1e-12, 1.0)
        return self

    def _numeric_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        distance = frame["distance"].astype(float)
        weight = frame["weight"].astype(float)
        dates = pd.to_datetime(frame["date"])
        date_ordinal = (dates - pd.Timestamp("2025-01-01")).dt.days.astype(float)
        day_of_week = dates.dt.dayofweek.astype(float)
        day_of_year = dates.dt.dayofyear.astype(float)

        features: dict[str, pd.Series | np.ndarray] = {
            "distance": distance,
            "sqrt_distance": np.sqrt(distance),
            "log_distance": np.log(distance),
            "weight": weight,
            "log_weight": np.log(weight),
            "distance_x_weight": distance * weight / 1_000_000.0,
            "date_ordinal": date_ordinal,
            "weekday_sin": np.sin(2.0 * np.pi * day_of_week / 7.0),
            "weekday_cos": np.cos(2.0 * np.pi * day_of_week / 7.0),
            "annual_sin": np.sin(2.0 * np.pi * day_of_year / 365.25),
            "annual_cos": np.cos(2.0 * np.pi * day_of_year / 365.25),
            "weight_was_missing": frame["weight_was_missing"].astype(float),
            "weight_was_negative": frame["weight_was_negative"].astype(float),
        }
        for knot in self.distance_knots:
            features[f"distance_hinge_{knot:.3f}"] = np.maximum(distance - knot, 0.0)

        for equipment in self.categories["equipment"]:
            indicator = frame["equipment"].astype(str).eq(equipment).astype(float)
            features[f"distance_x_equipment_{equipment}"] = distance * indicator

        if self.feature_set == "full":
            market = frame["market_index"].astype(float)
            quote = frame["quote_signal"].astype(float)
            features.update(
                {
                    "pickup_lat": frame["pickup_lat"].astype(float),
                    "pickup_lon": frame["pickup_lon"].astype(float),
                    "delivery_lat": frame["delivery_lat"].astype(float),
                    "delivery_lon": frame["delivery_lon"].astype(float),
                    "market_index": market,
                    "market_index_squared": market**2,
                    "market_index_was_missing": frame["market_index_was_missing"].astype(float),
                    "quote_signal": quote,
                    "quote_signal_squared": quote**2,
                    "distance_x_market_index": distance * market,
                    "distance_x_quote_signal": distance * quote,
                    "distance_x_market_x_quote": distance * market * quote,
                    "market_x_quote": market * quote,
                }
            )
        return pd.DataFrame(features, index=frame.index).astype(float)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "numeric_columns"):
            raise RuntimeError("FeatureBuilder must be fit before transform")
        numeric = self._numeric_features(frame)[self.numeric_columns]
        standardized = ((numeric - self.numeric_means) / self.numeric_stds).to_numpy(dtype=float)
        blocks = [np.ones((len(frame), 1), dtype=float), standardized]
        for column in self.COMMON_CATEGORICAL:
            values = frame[column].astype(str).to_numpy()
            category_block = np.column_stack(
                [(values == category).astype(float) for category in self.categories[column]]
            )
            blocks.append(category_block)
        return np.column_stack(blocks)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


def _ridge_solution(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    if sample_weight is None:
        gram = features.T @ features
        rhs = features.T @ target
    else:
        weights = np.asarray(sample_weight, dtype=float)
        gram = features.T @ (features * weights[:, None])
        rhs = features.T @ (target * weights)
    penalty = np.eye(features.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0  # Never penalize the intercept.
    return np.linalg.solve(gram + penalty, rhs)


class RidgeRegressor:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def fit(self, features: np.ndarray, target: np.ndarray) -> "RidgeRegressor":
        self.coefficients = _ridge_solution(features, np.asarray(target, dtype=float), self.alpha)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=float) @ self.coefficients


class HuberRidgeRegressor(RidgeRegressor):
    """Ridge regression fitted by Huber iteratively reweighted least squares."""

    def __init__(self, alpha: float, delta: float = 1.5, max_iter: int = 20) -> None:
        super().__init__(alpha)
        self.delta = float(delta)
        self.max_iter = int(max_iter)

    def fit(self, features: np.ndarray, target: np.ndarray) -> "HuberRidgeRegressor":
        target = np.asarray(target, dtype=float)
        coefficients = _ridge_solution(features, target, self.alpha)
        for _ in range(self.max_iter):
            residual = target - features @ coefficients
            center = np.median(residual)
            scale = 1.4826 * np.median(np.abs(residual - center))
            if not np.isfinite(scale) or scale < 1e-9:
                break
            cutoff = self.delta * scale
            absolute = np.abs(residual - center)
            weights = np.ones_like(absolute)
            outside = absolute > cutoff
            weights[outside] = cutoff / absolute[outside]
            updated = _ridge_solution(features, target, self.alpha, weights)
            relative_change = np.linalg.norm(updated - coefficients) / max(
                np.linalg.norm(coefficients), 1e-12
            )
            coefficients = updated
            if relative_change < 1e-7:
                break
        self.coefficients = coefficients
        return self


class LogRidgeRegressor(RidgeRegressor):
    """Ridge model on log-rate, which is naturally robust to upper outliers."""

    def fit(self, features: np.ndarray, target: np.ndarray) -> "LogRidgeRegressor":
        target = np.asarray(target, dtype=float)
        if (target <= 0).any():
            raise ValueError("Log target model requires positive rates")
        super().fit(features, np.log(target))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.exp(super().predict(features))


@dataclass
class LaneResidualAdjustment:
    """Smoothed, outlier-resistant residual correction for repeated lanes."""

    smoothing: float = 15.0

    @staticmethod
    def _keys(frame: pd.DataFrame) -> pd.Series:
        return frame["pickup"].astype(str) + " -> " + frame["delivery"].astype(str)

    def fit(
        self,
        frame: pd.DataFrame,
        actual: np.ndarray,
        base_prediction: np.ndarray,
    ) -> "LaneResidualAdjustment":
        residual = np.asarray(actual, dtype=float) - np.asarray(base_prediction, dtype=float)
        center = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - center)))
        # Clipping prevents the handful of apparent target anomalies from
        # contaminating a lane lookup shared by otherwise normal loads.
        clipped = np.clip(residual, center - 3.0 * scale, center + 3.0 * scale)
        stats = pd.DataFrame({"lane": self._keys(frame), "residual": clipped}).groupby("lane")[
            "residual"
        ].agg(["sum", "count"])
        self.adjustments = (stats["sum"] / (stats["count"] + self.smoothing)).to_dict()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self._keys(frame).map(self.adjustments).fillna(0.0).to_numpy(dtype=float)


@dataclass
class FittedFreightModel:
    cleaner: FreightCleaner
    features: FeatureBuilder
    regressor: RidgeRegressor
    lane_adjustment: LaneResidualAdjustment | None
    lower_bound: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        clean = self.cleaner.transform(frame)
        prediction = self.regressor.predict(self.features.transform(clean))
        if self.lane_adjustment is not None:
            prediction = prediction + self.lane_adjustment.predict(clean)
        return np.maximum(prediction, self.lower_bound)


def fit_freight_model(
    frame: pd.DataFrame,
    feature_set: str,
    family: str,
    alpha: float,
    delta: float = 1.5,
    lane_smoothing: float | None = None,
) -> FittedFreightModel:
    cleaner = FreightCleaner()
    clean = cleaner.fit_transform(frame)
    features = FeatureBuilder(feature_set)
    matrix = features.fit_transform(clean)
    target = clean[TARGET].to_numpy(dtype=float)
    if family == "ridge":
        regressor: RidgeRegressor = RidgeRegressor(alpha)
    elif family == "huber":
        regressor = HuberRidgeRegressor(alpha, delta=delta)
    elif family == "log_ridge":
        regressor = LogRidgeRegressor(alpha)
    else:
        raise ValueError(f"Unknown model family: {family}")
    regressor.fit(matrix, target)
    base_prediction = regressor.predict(matrix)
    adjustment = None
    if lane_smoothing is not None:
        adjustment = LaneResidualAdjustment(lane_smoothing).fit(clean, target, base_prediction)
    lower_bound = max(1.0, float(np.quantile(target, 0.001)) * 0.25)
    return FittedFreightModel(cleaner, features, regressor, adjustment, lower_bound)


def choose_best_parameter(
    results: Iterable[tuple[dict[str, float], dict[str, float]]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Select hyperparameters by MAE, with RMSE then MAPE as tie-breakers."""

    return min(results, key=lambda item: (item[1]["mae"], item[1]["rmse"], item[1]["mape_percent"]))
