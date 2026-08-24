# Spotter Freight Rate ML Assessment

This repository is a reproducible solution to the
Spotter Machine Learning Engineer take-home. It audits the supplied freight
data, uses a deployment-aligned chronological holdout, benchmarks multiple
model families, produces all 12,000 required predictions, and passes the
official scorer.

## Result

The model selected before looking at September-October is a native-categorical
LightGBM with an L1 objective and log target. It improved the former ridge
winner by 6.1% on MAE and 14.2% on MAPE:

| Model | Selection status | MAE | RMSE | MAPE | R2 |
|---|---|---:|---:|---:|---:|
| LightGBM L1/log, common columns | **Selected on August** | $109.76 | $634.36 | 4.75% | 0.8272 |
| LightGBM L1/raw, common columns | Diagnostic variant | $109.40 | $634.33 | 4.75% | 0.8272 |
| Engineered ridge, full columns | Former winner | $116.89 | $635.98 | 5.54% | 0.8263 |
| Median rate-per-mile | Baseline | $256.95 | $684.25 | 11.63% | 0.7990 |

The raw-target diagnostic happened to score $0.36 better on holdout MAE, but it
was not substituted after the fact: the log-target configuration had already
won on August. See `output/metrics/model_comparison.csv` for all 17 evaluated
variants and `docs/report.md` for the full reasoning.

## Prerequisites and setup

- Python 3.11
- Poetry 2.x

Install Poetry if necessary, configure an in-project environment, install the
locked dependencies, and verify imports:

```powershell
py -m pip install --user "poetry>=2.1,<3"
poetry config virtualenvs.in-project true --local
poetry install
poetry run python -c "import lightgbm, matplotlib, numpy, pandas, scipy; print('Environment OK')"
```

If the Poetry executable is not immediately found after installation, reopen
the terminal or add Python's user `Scripts` directory to `PATH`.

`requirements.txt` is the official scorer's dependency list, not a ceiling on
model-development libraries. Poetry retains those constraints and adds
LightGBM `>=4.6,<5` for native categorical splits and robust gradient boosting,
plus SciPy `>=1.10,<2` for the explicit distribution tests in the outlier audit;
the lock currently resolves LightGBM 4.7.0. The original ridge/Huber baselines
remain explicit NumPy implementations. CatBoost, XGBoost, and scikit-learn are
not present in this checkout's tracked dependency file or comparison artifact,
so this repository does not attribute unverified scores to them.

## Reproduce the analysis and predictions

Run from the repository root:

```powershell
poetry run spotter-explore
poetry run spotter-train
poetry run spotter-tune-lightgbm
```

The first training command recreates the original nine baselines. The
LightGBM command then searches on August, evaluates each frozen variant once on
September-October, appends eight rows, and regenerates the selected outputs.
It takes several minutes and intentionally refuses to append duplicate
LightGBM rows; rerun `train` first for a clean end-to-end reproduction. Random
sampling is seeded and LightGBM deterministic mode is enabled.

Run the official scorer exactly as follows:

```powershell
poetry run python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv --output-dir output/figures
```

Expected success output begins with:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
```

Run the automated tests:

```powershell
poetry run python -m unittest discover -s tests -v
```

## Repository layout

```text
data/                       Supplied data and completed December input file
docs/                       Assessment brief, report, and Loom script
src/spotter_freight/        Exploration, cleaning, features, models, tuning pipeline
tests/                      Standard-library regression and schema tests
output/eda/                 Exploratory tables, plots, and audit log
output/figures/             Official scorer-generated chart
output/logs/                Reproducible training and validation logs
output/metadata/            Selected-model and verification JSON
output/metrics/             Model, split, quality, and importance tables
validation_predictions.csv Required 12,000-row submission
score.py                    Official validator and chart generator
```

## Design in brief

The 48,000 development rows are already chronological from January through
October 2025, while the final 12,000 rows are the immediately following 61 days
in November and December. Accordingly, January-July is used for inner fitting,
August for hyperparameter tuning, and the final 61 days (September-October) as
an untouched model-family holdout. After selection, the winning model is refit
on all 48,000 labeled rows.

LightGBM receives pickup, delivery, and equipment as pandas categorical columns
rather than the original one-hot linear design. The December chart file lacks
six fields present in development and final validation data, so the common-
column LightGBM was tuned separately without fabricated `market_index`,
`quote_signal`, or coordinate values. It also won the August comparison against
the full feature set and is therefore the submission model as well.
