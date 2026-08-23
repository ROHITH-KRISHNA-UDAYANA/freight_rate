"""Reproducible exploratory analysis for the Spotter freight-rate data.

The script deliberately prints every audit section to stdout and mirrors the
same text to ``output/eda/exploration.log``.  This makes the decisions in the
modeling pipeline traceable to observed data rather than assumptions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output" / "eda"
TARGET = "posted_rate"
ID_COLUMN = "load_id"
DATE_COLUMN = "date"


class Tee:
    """Write identical text to the console and a persistent audit log."""

    def __init__(self, *streams: object) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def section(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def haversine_miles(frame: pd.DataFrame) -> np.ndarray:
    """Return great-circle miles implied by the four coordinate columns."""

    lat1 = np.radians(frame["pickup_lat"].to_numpy(dtype=float))
    lon1 = np.radians(frame["pickup_lon"].to_numpy(dtype=float))
    lat2 = np.radians(frame["delivery_lat"].to_numpy(dtype=float))
    lon2 = np.radians(frame["delivery_lon"].to_numpy(dtype=float))
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    return 3958.7613 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def audit_frame(name: str, frame: pd.DataFrame) -> None:
    section(f"FILE AUDIT: {name}")
    print(f"shape: {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    print(f"columns: {frame.columns.tolist()}")
    print("dtypes:")
    print(frame.dtypes.to_string())
    print("\nfirst 3 rows:")
    print(frame.head(3).to_string(index=False))

    missing = pd.DataFrame(
        {
            "missing_count": frame.isna().sum(),
            "missing_percent": frame.isna().mean().mul(100.0),
        }
    )
    print("\nmissing values:")
    print(missing.to_string(float_format=lambda value: f"{value:.4f}"))

    text_columns = frame.select_dtypes(include=["object", "string"]).columns
    if len(text_columns):
        blank_counts = {
            column: int(frame[column].astype("string").str.strip().eq("").sum())
            for column in text_columns
        }
        print(f"\nblank/whitespace-only strings: {blank_counts}")

    numeric_columns = frame.select_dtypes(include=np.number).columns
    if len(numeric_columns):
        numeric = frame[numeric_columns].to_numpy(dtype=float)
        print(f"non-finite numeric values (excluding NaN): {int(np.isinf(numeric).sum())}")
        print("\nnumeric summary:")
        print(
            frame[numeric_columns]
            .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            .T.to_string(float_format=lambda value: f"{value:,.4f}")
        )

    print(f"\nexact duplicate rows: {int(frame.duplicated().sum()):,}")
    if ID_COLUMN in frame:
        print(f"missing IDs: {int(frame[ID_COLUMN].isna().sum()):,}")
        print(f"duplicate IDs: {int(frame[ID_COLUMN].duplicated().sum()):,}")


def audit_dates(name: str, frame: pd.DataFrame) -> pd.Series:
    parsed = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    section(f"DATE AUDIT: {name}")
    print(f"invalid dates: {int(parsed.isna().sum()):,}")
    print(f"minimum date: {parsed.min()}")
    print(f"maximum date: {parsed.max()}")
    print(f"unique dates: {parsed.nunique():,}")
    print(f"input is chronological: {bool(parsed.is_monotonic_increasing)}")
    daily_counts = frame.assign(_date=parsed).groupby("_date", dropna=False).size()
    print(
        "rows per date: "
        f"min={daily_counts.min():,}, median={daily_counts.median():,.1f}, "
        f"mean={daily_counts.mean():,.2f}, max={daily_counts.max():,}"
    )
    monthly = parsed.dt.to_period("M").value_counts().sort_index()
    print("rows per month:")
    print(monthly.to_string())
    return parsed


def audit_invalid_values(name: str, frame: pd.DataFrame) -> None:
    section(f"VALIDITY CHECKS: {name}")
    checks: dict[str, int] = {}
    positive_columns = ["distance", "weight", "market_index", "quote_signal", TARGET]
    for column in positive_columns:
        if column in frame:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            checks[f"{column}: non-numeric/missing"] = int(numeric.isna().sum())
            checks[f"{column}: <= 0"] = int((numeric <= 0).sum())
    coordinate_bounds = {
        "pickup_lat": (-90.0, 90.0),
        "delivery_lat": (-90.0, 90.0),
        "pickup_lon": (-180.0, 180.0),
        "delivery_lon": (-180.0, 180.0),
    }
    for column, (lower, upper) in coordinate_bounds.items():
        if column in frame:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            checks[f"{column}: outside [{lower:g}, {upper:g}]"] = int(
                ((numeric < lower) | (numeric > upper)).sum()
            )
    for label, count in checks.items():
        print(f"{label}: {count:,}")


def categorical_overlap(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in ["pickup", "delivery", "equipment"]:
        train_values = set(train[column].dropna().astype(str))
        validation_values = set(validation[column].dropna().astype(str))
        unseen = sorted(validation_values - train_values)
        rows.append(
            {
                "column": column,
                "train_unique": len(train_values),
                "validation_unique": len(validation_values),
                "shared_unique": len(train_values & validation_values),
                "validation_only_count": len(unseen),
                "validation_only_values": " | ".join(unseen) if unseen else "",
            }
        )
    return pd.DataFrame(rows)


def numeric_shift(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    shared = [
        column
        for column in train.select_dtypes(include=np.number).columns
        if column in validation and column != TARGET
    ]
    rows: list[dict[str, object]] = []
    for column in shared:
        train_values = train[column].astype(float)
        validation_values = validation[column].astype(float)
        train_std = float(train_values.std(ddof=0))
        rows.append(
            {
                "feature": column,
                "train_mean": train_values.mean(),
                "validation_mean": validation_values.mean(),
                "train_median": train_values.median(),
                "validation_median": validation_values.median(),
                "standardized_mean_shift": (
                    (validation_values.mean() - train_values.mean()) / train_std if train_std else np.nan
                ),
                "validation_below_train_min": int((validation_values < train_values.min()).sum()),
                "validation_above_train_max": int((validation_values > train_values.max()).sum()),
            }
        )
    return pd.DataFrame(rows)


def save_plots(train: pd.DataFrame, parsed_dates: pd.Series) -> None:
    daily = train.assign(_date=parsed_dates).groupby("_date", as_index=False)[TARGET].agg(["mean", "median"])
    figure, axis = plt.subplots(figsize=(10.5, 4.8), dpi=160)
    axis.plot(daily["_date"], daily["mean"], label="Daily mean", color="#064A56", linewidth=1.8)
    axis.plot(daily["_date"], daily["median"], label="Daily median", color="#E8505B", linewidth=1.4)
    axis.set(title="Observed posted rate over development period", ylabel="Posted rate ($)", xlabel="Date")
    axis.grid(axis="y", color="#D9E2E4", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "target_over_time.png", bbox_inches="tight")
    plt.close(figure)

    equipment = train.groupby("equipment", as_index=False)[TARGET].agg(["mean", "median", "count"])
    figure, axis = plt.subplots(figsize=(7.4, 4.6), dpi=160)
    positions = np.arange(len(equipment))
    axis.bar(positions - 0.18, equipment["mean"], width=0.36, label="Mean", color="#064A56")
    axis.bar(positions + 0.18, equipment["median"], width=0.36, label="Median", color="#E8505B")
    axis.set_xticks(positions, equipment["equipment"])
    axis.set(title="Posted rate by equipment", ylabel="Posted rate ($)")
    axis.grid(axis="y", color="#D9E2E4", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "target_by_equipment.png", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        "train_test.csv": pd.read_csv(DATA_DIR / "train_test.csv"),
        "validation.csv": pd.read_csv(DATA_DIR / "validation.csv"),
        "validation_predictions_template.csv": pd.read_csv(
            DATA_DIR / "validation_predictions_template.csv"
        ),
        # Audit the untouched shell so rerunning exploration after training does
        # not reinterpret generated predictions as source data.
        "december_chart_inputs.csv": pd.read_csv(DATA_DIR / "december_chart_inputs_template.csv"),
    }

    original_stdout = sys.stdout
    with (OUTPUT_DIR / "exploration.log").open("w", encoding="utf-8", newline="\n") as log:
        sys.stdout = Tee(original_stdout, log)
        try:
            print("SPOTTER FREIGHT-RATE DATA AUDIT")
            print("All counts and summaries below are computed directly from the supplied CSV files.")
            for name, frame in sources.items():
                audit_frame(name, frame)

            train = sources["train_test.csv"]
            validation = sources["validation.csv"]
            template = sources["validation_predictions_template.csv"]
            december = sources["december_chart_inputs.csv"]

            section("COLUMN AVAILABILITY")
            all_columns = sorted(set().union(*(frame.columns for frame in sources.values())))
            availability = pd.DataFrame(
                {
                    name: [column in frame.columns for column in all_columns]
                    for name, frame in sources.items()
                },
                index=all_columns,
            )
            availability.index.name = "column"
            print(availability.to_string())
            availability.to_csv(OUTPUT_DIR / "column_availability.csv")
            print(f"\ntrain-only versus validation: {sorted(set(train) - set(validation))}")
            print(f"validation-only versus train: {sorted(set(validation) - set(train))}")
            print(
                "December inputs absent from train feature set: "
                f"{sorted((set(train) - {TARGET, ID_COLUMN}) - set(december))}"
            )

            train_dates = audit_dates("train_test.csv", train)
            validation_dates = audit_dates("validation.csv", validation)
            december_dates = audit_dates("december_chart_inputs.csv", december)

            section("ORDERING AND ID INTEGRITY")
            expected_train_ids = pd.Series(
                [f"TR-{index:06d}" for index in range(1, len(train) + 1)], dtype="object"
            )
            expected_validation_ids = pd.Series(
                [f"TE-{index:06d}" for index in range(1, len(validation) + 1)], dtype="object"
            )
            print(f"train IDs are complete and sequential: {bool(train[ID_COLUMN].equals(expected_train_ids))}")
            print(
                "validation IDs are complete and sequential: "
                f"{bool(validation[ID_COLUMN].equals(expected_validation_ids))}"
            )
            print(
                "template IDs match validation order exactly: "
                f"{bool(template[ID_COLUMN].equals(validation[ID_COLUMN]))}"
            )
            print(
                "template and validation ID sets match: "
                f"{set(template[ID_COLUMN]) == set(validation[ID_COLUMN])}"
            )

            for name, frame in [("train_test.csv", train), ("validation.csv", validation)]:
                audit_invalid_values(name, frame)

            section("INVALID-VALUE PATTERN EVIDENCE")
            for name, frame in [("train", train), ("validation", validation)]:
                weight = pd.to_numeric(frame["weight"], errors="coerce")
                negative_absolute = weight[weight < 0].abs()
                valid_positive = weight[weight > 0]
                print(f"{name} absolute negative-weight quantiles:")
                print(
                    negative_absolute.quantile([0.0, 0.01, 0.25, 0.50, 0.75, 0.99, 1.0]).to_string(
                        float_format=lambda value: f"{value:,.2f}"
                    )
                )
                print(f"{name} valid positive-weight quantiles:")
                print(
                    valid_positive.quantile([0.0, 0.01, 0.25, 0.50, 0.75, 0.99, 1.0]).to_string(
                        float_format=lambda value: f"{value:,.2f}"
                    )
                )
                missing_by_month = (
                    frame.assign(_month=pd.to_datetime(frame["date"]).dt.to_period("M"))
                    .groupby("_month")["market_index"]
                    .apply(lambda values: int(values.isna().sum()))
                )
                print(f"{name} missing market_index by month:")
                print(missing_by_month.to_string())
            weight_status = np.select(
                [train["weight"].isna(), train["weight"] < 0],
                ["missing", "negative"],
                default="valid",
            )
            print("development target by weight-quality status:")
            print(
                train.assign(weight_status=weight_status)
                .groupby("weight_status")[TARGET]
                .agg(["count", "mean", "median", "max"])
                .to_string(float_format=lambda value: f"{value:,.2f}")
            )

            section("TARGET-TAIL AUDIT")
            target_quantiles = train[TARGET].quantile([0.0, 0.01, 0.50, 0.95, 0.99, 0.995, 1.0])
            print(target_quantiles.to_string(float_format=lambda value: f"{value:,.2f}"))
            print("ten largest posted rates (retained because they are positive and not provably invalid):")
            print(
                train.nlargest(10, TARGET)[
                    ["load_id", "pickup", "delivery", "distance", "equipment", "date", TARGET]
                ].to_string(index=False)
            )

            section("DUPLICATION BEYOND LOAD ID")
            train_feature_columns = [column for column in validation.columns if column != ID_COLUMN]
            print(
                "train duplicate feature rows (ignoring load_id and target): "
                f"{int(train.duplicated(subset=train_feature_columns).sum()):,}"
            )
            print(
                "train duplicate feature+target rows (ignoring load_id): "
                f"{int(train.duplicated(subset=train_feature_columns + [TARGET]).sum()):,}"
            )
            print(
                "validation duplicate feature rows (ignoring load_id): "
                f"{int(validation.duplicated(subset=train_feature_columns).sum()):,}"
            )

            section("LANE AND CATEGORY COVERAGE")
            train_lanes = set(zip(train["pickup"], train["delivery"]))
            validation_lanes = set(zip(validation["pickup"], validation["delivery"]))
            print(f"unique train lanes: {len(train_lanes):,}")
            print(f"unique validation lanes: {len(validation_lanes):,}")
            print(f"shared lanes: {len(train_lanes & validation_lanes):,}")
            print(f"validation-only lanes: {len(validation_lanes - train_lanes):,}")
            print(
                "validation rows on unseen lanes: "
                f"{int((~pd.Series(list(zip(validation['pickup'], validation['delivery']))).isin(train_lanes)).sum()):,}"
            )
            overlap = categorical_overlap(train, validation)
            print(overlap.to_string(index=False))
            overlap.to_csv(OUTPUT_DIR / "categorical_overlap.csv", index=False)

            section("NUMERIC DISTRIBUTION SHIFT: TRAIN VS VALIDATION")
            shift = numeric_shift(train, validation)
            print(shift.to_string(index=False, float_format=lambda value: f"{value:,.5f}"))
            shift.to_csv(OUTPUT_DIR / "numeric_distribution_shift.csv", index=False)

            section("COORDINATE AND DISTANCE CONSISTENCY")
            for name, frame in [("train", train), ("validation", validation)]:
                great_circle = haversine_miles(frame)
                ratio = frame["distance"].to_numpy(dtype=float) / np.maximum(great_circle, 1e-9)
                print(f"{name} great-circle distance comparison:")
                print(
                    pd.Series(ratio, name="reported/great_circle ratio")
                    .describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
                    .to_string(float_format=lambda value: f"{value:,.4f}")
                )
                print(f"{name} rows with reported distance below great-circle distance: {int((ratio < 1).sum()):,}")
                print(
                    f"{name} Pearson correlation (reported vs great-circle): "
                    f"{np.corrcoef(frame['distance'], great_circle)[0, 1]:.6f}"
                )
            city_coordinate_issues: list[str] = []
            for city_column, lat_column, lon_column in [
                ("pickup", "pickup_lat", "pickup_lon"),
                ("delivery", "delivery_lat", "delivery_lon"),
            ]:
                counts = (
                    pd.concat(
                        [
                            train[[city_column, lat_column, lon_column]],
                            validation[[city_column, lat_column, lon_column]],
                        ],
                        ignore_index=True,
                    )
                    .drop_duplicates()
                    .groupby(city_column)
                    .size()
                )
                inconsistent = counts[counts > 1]
                city_coordinate_issues.extend(inconsistent.index.astype(str).tolist())
                print(
                    f"{city_column}: cities mapping to more than one coordinate pair: "
                    f"{len(inconsistent):,}"
                )
            print(f"coordinate-inconsistent city labels: {sorted(set(city_coordinate_issues))}")

            section("TARGET ASSOCIATIONS")
            numeric_columns = train.select_dtypes(include=np.number).columns
            pearson = train[numeric_columns].corr(method="pearson")[TARGET].rename("pearson")
            spearman = train[numeric_columns].corr(method="spearman")[TARGET].rename("spearman")
            correlations = pd.concat([pearson, spearman], axis=1).sort_values(
                "pearson", key=lambda series: series.abs(), ascending=False
            )
            print(correlations.to_string(float_format=lambda value: f"{value:.6f}"))
            correlations.to_csv(OUTPUT_DIR / "numeric_target_correlations.csv")
            train[numeric_columns].corr(method="pearson").to_csv(OUTPUT_DIR / "numeric_correlation_matrix.csv")

            for column in ["equipment", "pickup", "delivery"]:
                summary = (
                    train.groupby(column, dropna=False)[TARGET]
                    .agg(["count", "mean", "median", "std"])
                    .sort_values("count", ascending=False)
                )
                print(f"\nposted_rate by {column}:")
                print(summary.to_string(float_format=lambda value: f"{value:,.2f}"))
                summary.to_csv(OUTPUT_DIR / f"target_by_{column}.csv")

            section("DERIVED TARGET DIAGNOSTICS")
            rate_per_mile = train[TARGET] / train["distance"]
            print("posted_rate / distance summary:")
            print(
                rate_per_mile.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string(
                    float_format=lambda value: f"{value:,.4f}"
                )
            )
            print(
                f"correlation of rate_per_mile with distance: "
                f"{rate_per_mile.corr(train['distance']):.6f}"
            )
            print(
                f"correlation of rate_per_mile with market_index: "
                f"{rate_per_mile.corr(train['market_index']):.6f}"
            )
            print(
                f"correlation of rate_per_mile with quote_signal: "
                f"{rate_per_mile.corr(train['quote_signal']):.6f}"
            )
            daily_target = train.assign(_date=train_dates).groupby("_date")[TARGET].mean()
            date_ordinal = pd.Series(daily_target.index.map(pd.Timestamp.toordinal), index=daily_target.index)
            print(
                "correlation of daily mean posted_rate with calendar time: "
                f"{daily_target.corr(date_ordinal):.6f}"
            )

            section("TIME-PERIOD RELATIONSHIP")
            print(f"train ends before validation starts: {train_dates.max() < validation_dates.min()}")
            print(f"gap between train end and validation start: {(validation_dates.min() - train_dates.max()).days - 1} days")
            print(
                "December chart dates overlap validation period: "
                f"{december_dates.min() >= validation_dates.min() and december_dates.max() <= validation_dates.max()}"
            )
            print(
                "CONCLUSION FOR SPLITTING: because rows are chronological and final inference is on strictly "
                "later dates, a chronological holdout is the closest honest simulation of deployment."
            )

            save_plots(train, train_dates)
            section("ARTIFACTS WRITTEN")
            for path in sorted(OUTPUT_DIR.iterdir()):
                print(path.relative_to(ROOT))
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
