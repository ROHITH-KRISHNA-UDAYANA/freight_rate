# Loom Script (approximately 2 minutes 50 seconds)

## 0:00-0:30 — Key findings

I treated this as a forward freight-pricing problem, not a shuffled regression
task. The 48,000 labeled loads run from January through October, and the 12,000
prediction rows are the next 61 days. Distance is the main driver: its raw
correlation with rate is point nine one, but the log-log correlation is point
nine seven, so the pricing process looks more multiplicative than additive.
There are also 64 origins, 64 destinations, and just over 4,000 observed lane
combinations, which makes category interactions important.

## 0:30-1:00 — Data quality and fixes

The audit found no duplicate IDs, bad dates, invalid coordinates, or
non-positive distances or targets. It did find 300 missing and 292 negative
weights in development, plus 374 missing market-index values. Negative weights
match the valid weight distribution after taking absolute value, so I repair
them as sign flips and preserve a flag. Missing weights use a training-only
equipment median, while missing market indices use the same-day median.

I also audited extreme target residuals. About point six percent of the fitting
rows are two and a half to five point three times their robust-baseline estimate.
Numeric distribution differences are small, and categorical effect sizes are
tiny, so the available features cannot explain their magnitude.

## 1:00-1:30 — Split approach

January through July is inner training, August is the only tuning and early-
stopping slice, and September through October is an untouched 61-day holdout.
That matches the length and direction of the final November-December forecast.
I froze each configuration before scoring the holdout, and I did not switch to
a variant just because its final number looked better.

## 1:30-2:15 — Model reasoning

I kept the original median, ridge, log-ridge, and robust Huber comparisons, then
added LightGBM. Unlike the linear one-hot path, LightGBM receives pickup,
delivery, and equipment as native pandas categories, so tree paths can learn
lane-like interactions. An explicit lane category actually hurt August, so I
left it out.

The objective test was decisive: L1 beat squared error for both raw and log
targets because those rare extreme labels otherwise dominate the gradients.
Log target also won the August screen, consistent with the stronger log-log
relationship. The selected model uses 63 leaves, a point zero two learning
rate, 80 percent row and feature sampling, and 4,083 trees. It scored $109.76
MAE, $634.36 RMSE, 4.75 percent MAPE, and point eight two seven two R-squared.
That beats the former ridge on both MAE and MAPE. Competitive-model RMSE stays
within about nine dollars because the same unpredictable extremes dominate
squared error, so RMSE was reported but was not the optimization target.

## 2:15-2:50 — Code and outputs walkthrough

In `explore.py`, every schema, validity, overlap, shift, and correlation check
is logged. `modeling.py` preserves the original train-fitted cleaner and linear
baselines. `lightgbm_model.py` builds the native categorical matrices and keeps
common versus full feature schemas explicit. `tune_lightgbm.py` owns the fixed
date split, August search, frozen holdout evaluation, final refit, comparison
table, and gain importance.

The December file has no coordinates, market index, or quote signal, so I
tuned its common-column model separately instead of inventing placeholders.
That same feature set also won August overall. Finally, the tests assert the
categorical and submission schemas, and the official scorer validates all
12,000 prediction rows plus the 31-day December file and creates the chart.
