# Smart Energy Analytics — Forecasting and Vulnerability Profiling

This repository builds an hourly, weather-aligned dataset from multiple sources (Ausgrid NSW CSV and LCL London Parquet), engineers leakage-safe features, trains time-aware models, explains temperature sensitivity, generates extensive diagnostics/visuals (>30), and renders a LaTeX report.

Project map
- scripts/
  - 00_unify_datasets.py — Unify multi-source inputs into a single CSV with region label (NSW/LCL). Enforces hourly alignment and retains rows with weather present only.
  - 01_build_features.py — Leakage-safe features per building: lags, rolling means, calendar, weather, region_id; defines y_next (t+1).
  - 02_train_models.py — Time-aware splits from timestamps; trains Naive, RandomForest, HistGradientBoosting; saves metrics, models, tables.
  - 03_explain_segments.py — Computes per-building temperature elasticity via ICE slopes; segments buildings (High/Moderate/Low).
  - 04_make_plots.py — Core figures (feature importance, pred vs actual, residual patterns, heatmaps, usage vs temp).
  - 04a_extra_plots.py — Expanded suite (>30 figs): missingness, distributions, calibration, PDP/ICE, model comparison, timeline of splits, pred-vs-true density and by region, error CDF, clustering, PCA, worst buildings, etc.
  - 05_render_report.py — Generates LaTeX report (report/paper.tex) integrating methods, results, extended EDA, model comparison, program usage, and conclusions.
  - predict.py — Executable CLI to predict next-hour usage for each building from a usage file (+ optional weather file).
- data/
  - ausgrid_with_weather_normalized.csv — Example source (NSW).
  - lcl_with_weather_normalized.parquet — Example source (LCL).
  - processed/ — Engineered features (.csv, .parquet) and samples.
- outputs/
  - figures/ — All PNG figures produced by 04_make_plots.py and 04a_extra_plots.py (now >30).
  - metrics/metrics.json — MAE/RMSE for Naive, RF, HGBR on val/test with time cutoffs.
  - tables/ — Supporting CSV/JSON tables (feature importance, per-building MAE, preds sample, elasticity summary, data dictionary).
  - vulnerability/building_elasticity.csv — Elasticity and segments per building.
- report/
  - paper.tex — LaTeX source (Tectonic-friendly).
  - README.md — How to compile on Overleaf or locally.

Environment setup
1) Create and activate a virtual environment
   python3 -m venv .venv
   source .venv/bin/activate
2) Install dependencies
   python -m pip install -r requirements.txt

Pipeline quickstart
- Unify inputs (CSV + Parquet → single CSV with region)
  source .venv/bin/activate && python scripts/00_unify_datasets.py \
    --ausgrid data/ausgrid_with_weather_normalized.csv \
    --lcl data/lcl_with_weather_normalized.parquet \
    --out data/combined_with_weather_normalized.csv

- Build features (leakage-safe, region-aware)
  source .venv/bin/activate && python scripts/01_build_features.py \
    --input data/combined_with_weather_normalized.csv \
    --out data/processed/features

- Train models (time-aware)
  source .venv/bin/activate && python scripts/02_train_models.py \
    --features data/processed/features.csv

- Explain and segment vulnerability
  source .venv/bin/activate && python scripts/03_explain_segments.py

- Generate figures (base + extended)
  source .venv/bin/activate && python scripts/04_make_plots.py
  source .venv/bin/activate && python scripts/04a_extra_plots.py

- Render LaTeX report
  source .venv/bin/activate && python scripts/05_render_report.py
  (Open report/paper.tex locally or upload to Overleaf)

Prediction CLI (scripts/predict.py)
- Purpose: Given a usage file (CSV/Parquet) and a region, predict next-hour usage per building.
- Minimal input columns in --input:
  - building_name, full_timestamp, usage_kwh_norm OR usage_kwh
  - Optional: apparent_temperature_norm OR apparent_temperature, precipitation, is_day, is_weekend, is_holiday
- Optional --weather CSV:
  - time/full_timestamp, apparent_temperature_norm (or apparent_temperature), precipitation (optional), is_day (optional)
- Region: --region NSW|LCL (used when no region column is present)

Examples:
1) Predict from Ausgrid CSV:
   source .venv/bin/activate && python scripts/predict.py \
     --input data/ausgrid_with_weather_normalized.csv \
     --region NSW \
     --out outputs/tables/predictions_next_hour.csv

2) Predict with an external weather file:
   source .venv/bin/activate && python scripts/predict.py \
     --input my_usage.csv \
     --region LCL \
     --weather my_weather.csv \
     --out outputs/tables/predictions_next_hour.csv

What the program produces
- Accurate next-hour forecasts validated with chronological splits (MAE/RMSE across models).
- Model selection rationale and comparison charts (Naive vs RF vs HGBR).
- Extensive, readable figures that interpret the model’s performance, patterns, and risks.
- Per-building temperature elasticity and segments that support targeted interventions.
- A Tectonic-friendly LaTeX report that narrates the process from data acquisition and scraping to conclusions.

Notes on design choices
- Leakage-safe features using per-building groupby + shift/rolling (no future leakage).
- Region-aware modeling via region_id, optional weather lags.
- Strict chronological splits (no random mixing) and streaming ingestion for scalability.
- Avoid exogenous imputation: we drop hours with missing weather to preserve sensitivity analyses.
