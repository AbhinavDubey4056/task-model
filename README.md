# Software Feature Shipment Delay Prediction

Predicts the delay (in days) for a planned software feature shipment, and
from that, the actual expected delivery window.

## Project structure

```
data/            raw dataset
notebooks/eda.py exploratory data analysis (stats, correlations, plots)
src/data_prep.py loading, cleaning, feature engineering, train/test split
src/train.py     baseline models + tuned gradient boosting model
src/predict.py   inference: predict delay + next shipment date for new input
models/          saved model (.joblib) and metrics (.json)
```

## Setup

```bash
pip install -r requirements.txt
```

**Important:** `models/` starts empty. You must run `src/train.py` yourself
before `predict.py` or `interactive.py` will work — the trained model is
always regenerated in *your* environment, matching *your* scikit-learn/
XGBoost versions. Never copy a `.joblib` file trained elsewhere into this
folder; a model pickled with a different scikit-learn version than the one
loading it can silently misbehave (you'll see an `InconsistentVersionWarning`
if this happens — retrain locally if you ever do).

## Run

```bash
python notebooks/eda.py    # prints stats, saves plots to notebooks/eda_outputs/
python src/train.py        # trains baseline + tuned model, saves both
python src/predict.py      # example prediction for a new shipment
python src/interactive.py  # ask about a feature one question at a time
```

## Web frontend

```bash
cd src
python3 app.py
```
Then open http://localhost:5000 in a browser. `app.py` is a small Flask
API (`/api/predict`) that wraps the trained model and also serves
`frontend/index.html` — a single self-contained HTML/CSS/JS page with a
form. The browser never touches the model directly (it can't — `.joblib`
is a Python object); it sends form values to `/api/predict` as JSON and
displays whatever the real model returns.

`interactive.py` gives a conversational feel — it asks one question at a
time and validates each answer — but every prediction still comes from the
trained regression model in `predict.py`, not a language model guessing.
It only knows about *attributes* of a feature (team size, complexity,
dependencies, etc.), not feature names — this dataset has no mapping from
a real feature name to its attributes, so "when is the login redesign
shipping" isn't answerable without knowing that feature's numbers first.

## Testing

```bash
python3 -m unittest discover -s tests -v
```

Covers: no missing values, all expected feature columns present, the
feature matrix stays full-rank (regression test for the rank-deficiency bug
found during development — see below), the train/test split is
chronological with no overlap, and `predict.py` returns sane, directionally
correct output (higher `feature_complexity` → longer predicted delay).

## Data notes

- 1,300 rows, 10 features, **no missing values** — no imputation was needed.
- `planned_shipment_date` is spread almost uniformly from 2021 to 2045
  (~51-54 rows/year), which indicates the dates in this dataset are
  synthetically generated rather than a real historical log. Calendar
  features (`month`, `day_of_week`, `quarter`) are still engineered for
  completeness, but they carry little real signal here as a result.
- IQR outlier checks flagged `holidays_in_sprint` heavily, but that column
  is a binary flag (0/1) — the "outliers" are just the minority class, not
  data quality issues.

## Key finding

Correlation with `delay_days`:

| Feature | Correlation |
|---|---|
| feature_complexity | 0.82 |
| num_blockers | 0.41 |
| num_dependencies | 0.27 |
| holidays_in_sprint | 0.17 |
| past_avg_delay_days | 0.15 |
| estimated_bug_count | 0.14 |
| priority_encoded | 0.01 |
| sprint_length_weeks | -0.02 |
| team_size | -0.02 |

The relationship between `feature_complexity` and delay is close to linear,
so **plain linear regression outperformed a tuned gradient boosting model**
on this dataset:

| Model | MAE | RMSE |
|---|---|---|
| Baseline (median) | 4.19 | 5.28 |
| Baseline (ridge regression) | **0.88** | **1.08** |
| Tuned gradient boosting | 0.96-1.00 | 1.18-1.23 |

Gradient boosting models earn their keep on nonlinear interactions; when a
target is largely explained by one near-linear driver, the simpler model
wins and is also easier to explain to stakeholders. `train.py` still fits
and tunes the gradient boosting model (as the assignment names XGBoost
explicitly), but the honest recommendation — and the one worth defending in
the video — is to ship the linear model, or an ensemble of the two, and
treat the tree model's feature importances as a validation check on the
linear model's coefficients rather than the production model itself.

**Note on the baseline model:** an early version of the engineered features
included `num_blockers + num_dependencies` as a "friction" feature — an
exact linear combination of two columns already in the dataset, which made
the design matrix singular (rank-deficient) and caused OLS to throw
divide-by-zero/overflow warnings. The fix was two-fold: drop the redundant
feature at the source (`data_prep.py`), and use Ridge instead of plain
`LinearRegression` for the baseline, since L2 regularization is robust to
near-singular designs in general (worth doing even after fixing this
particular case, since new engineered features could reintroduce the same
issue). This kind of bug is easy to introduce silently when engineering
interaction features — it's exactly the "problem-solving with imperfect
data" the assignment asks about, so it's worth including in your video.

`team_size`, `sprint_length_weeks`, and `priority_encoded` show ~0
correlation with delay and add mostly noise; they were kept as model inputs
for completeness but are candidates to drop in a v2.

## Model backend

`train.py` uses XGBoost if installed (`pip install xgboost`), matching the
model named in the assignment. If XGBoost isn't available in an environment,
it falls back to scikit-learn's `HistGradientBoostingRegressor` so the
pipeline still runs end-to-end.

## With more time

- Try an ElasticNet/Ridge model to see if regularizing the linear model
  closes the gap further, and try a simple linear + tree ensemble.
- If real (non-synthetic) shipment dates were available, revisit calendar
  features — month-end/quarter-end crunch effects are common in real
  engineering orgs and were not testable on this data.
- Collect confidence intervals on predictions (e.g., quantile regression)
  rather than a single point estimate, since "next delivery window" implies
  a range, not just a date.

## Note on a platform-specific warning (Apple Silicon)

On M-series Macs, NumPy's Accelerate BLAS backend throws spurious
`RuntimeWarning: divide by zero / overflow / invalid value encountered in
matmul` on ordinary matrix multiplications — including completely benign
ones (NumPy's own bug reports reproduce it with `np.identity(15) @
np.identity(15)`; see
[#28790](https://github.com/numpy/numpy/issues/28790) and
[#29820](https://github.com/numpy/numpy/issues/29820)). It is unrelated to
this project's data, features, or model. `train.py` filters this specific
warning message after confirming independently that the feature matrix is
full rank and results are bit-identical with or without it firing. On
Linux/Windows or Intel Macs this warning likely won't appear at all.
