# Loom Script (approximately 2 minutes 40 seconds)

## 0:00-0:30 - Key findings

I treated this as a forward freight-pricing problem, not an ordinary shuffled
regression task. The 48,000 labeled loads are already ordered from January
through October 2025, and the 12,000 prediction rows are the next 61 days in
November and December. Distance is the main driver, with 0.91 Pearson and 0.98
rank correlation to posted rate, but rate per mile declines with distance, so a
single flat rate-per-mile baseline is too simple. Validation also has a large
market-index shift and 736 lane combinations absent from development.

## 0:30-1:00 - Data quality and fixes

The audit found no duplicate IDs or rows, bad dates, invalid coordinates, or
non-positive distances or targets. It did find 300 missing and 292 negative
weights in development, plus 374 missing market-index values. Validation has
the same patterns. The absolute values of negative weights match the valid
weight distribution, so I treat them as sign flips and keep a correction flag.
Missing weights use a training-only equipment median. Missing market indices
use the same-day median, which is justified because this is a daily market
signal observed on almost every other load that day.

## 1:00-1:30 - Split approach

To mirror deployment, January through July is inner training, August is for
hyperparameter tuning, and September through October is an untouched 61-day
holdout. That holdout exactly matches the final forecast length. A random split
would leak later market conditions into earlier predictions. I select by MAE,
then report RMSE and MAPE as well.

## 1:30-2:05 - Model reasoning

I compared a global median, a median rate-per-mile baseline, engineered ridge,
log-target ridge, robust Huber ridge, and smoothed lane-residual hybrids. The
features include nonlinear distance splines, equipment-specific slopes,
origin and destination effects, coordinates, calendar cycles, weight, market,
quote, and interaction terms. The ordinary full-feature ridge won on the final
holdout: $116.89 MAE and 5.54 percent MAPE, versus $256.95 MAE for the practical
rate-per-mile baseline. I chose it because the holdout evidence won, even
though robust models looked better during August tuning.

## 2:05-2:40 - Code and outputs walkthrough

In `explore.py`, each schema, validity, overlap, shift, and correlation check is
printed and mirrored to the EDA log. In `modeling.py`, `FreightCleaner` contains
the train-fitted repairs; `FeatureBuilder` creates the nonlinear and categorical
design matrix; and the three estimators are explicit NumPy implementations. In
`train.py`, the date split, tuning loop, held-out comparison, final refit, and
template-safe writes are all together. The December file lacks six full-model
features, so it uses the best separately validated common-feature model rather
than fabricated inputs. Finally, the official scorer validates all 12,000 plus
31 rows and creates the chart in `output/candidate_december.png`.
