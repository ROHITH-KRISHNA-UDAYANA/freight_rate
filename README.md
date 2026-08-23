# Spotter Freight Rate ML Assessment

This repository is a reproducible, dependency-constrained solution to the
Spotter Machine Learning Engineer take-home. It audits the supplied freight
data, uses a deployment-aligned chronological holdout, benchmarks multiple
model families, produces all 12,000 required predictions, and passes the
official scorer.

## Result

The selected full-feature ridge model achieved the best untouched September-
October holdout result:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Engineered ridge (selected) | $116.89 | $635.98 | 5.54% |
| Median rate-per-mile baseline | $256.95 | $684.25 | 11.63% |
| Global median baseline | $1,148.92 | $1,569.42 | 70.15% |

See `output/model_comparison.csv` for all nine evaluated model variants and
`report.md` for the rationale and limitations.

## Prerequisites and setup

- Python 3.11
- Poetry 2.x

Install Poetry if necessary, configure an in-project environment, install the
locked dependencies, and verify imports:

```powershell
py -m pip install --user "poetry>=2.1,<3"
poetry config virtualenvs.in-project true --local
poetry install
poetry run python -c "import matplotlib, numpy, pandas; print('Environment OK')"
```

If the Poetry executable is not immediately found after installation, reopen
the terminal or add Python's user `Scripts` directory to `PATH`.

Dependencies intentionally use the exact constraints from `requirements.txt`:
Matplotlib `>=3.8,<4`, NumPy `>=1.26,<3`, and pandas `>=2.0,<3`.

## Reproduce the analysis and predictions

Run from the repository root:

```powershell
poetry run python -m src.spotter_freight.explore
poetry run python -m src.spotter_freight.train
```

The training command deterministically retunes the candidate models, rewrites
`validation_predictions.csv`, and fills `data/december_chart_inputs.csv` from
the preserved `data/december_chart_inputs_template.csv` shell.
There is no random split or stochastic estimator.

Run the official scorer exactly as follows:

```powershell
poetry run python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv --output-dir output
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
src/spotter_freight/        Exploration, cleaning, features, models, pipeline
tests/                      Standard-library regression and schema tests
output/                     EDA logs, comparison metrics, scorer chart
validation_predictions.csv Required 12,000-row submission
report.md                   Panel-facing technical report
loom_script.md              Timed 2-3 minute walkthrough script
score.py                    Official validator and chart generator
```

## Design in brief

The 48,000 development rows are already chronological from January through
October 2025, while the final 12,000 rows are the immediately following 61 days
in November and December. Accordingly, January-July is used for inner fitting,
August for hyperparameter tuning, and the final 61 days (September-October) as
an untouched model-family holdout. After selection, the winning model is refit
on all 48,000 labeled rows.

The December chart file lacks six features present in development and final
validation data. A separately validated common-feature model is therefore used
for that chart rather than fabricating unavailable `market_index`,
`quote_signal`, or coordinate values.
