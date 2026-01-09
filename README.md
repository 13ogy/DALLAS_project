# Smart Energy Analytics — Forecasting and Vulnerability Profiling

* Project map
- scripts/
  - 00_unify_datasets.py — Unify multi-source inputs into a single CSV with region label (NSW/LCL). Enforces hourly alignment and retains rows with weather present only. 
  => The original datasets aren't submitted due to size > 20GB.
  - 01_build_features.py — Features per building: lags (1h, 24h, 168h), rolling means, calendar and global temp z-score.
  - 02_train_models.py
    - Chronological splits from timestamps 
    - trains Naive, Seasonal-24, Weekly-168, RF, HGBR models 
    - evaluates on full test set, computes test importance and skill scores.
  - 02b_shap_analysis.py — SHAP analysis for model interpretability on a test sample.
  - 02c_residual_diagnostics.py — Residual diagnostics on full test set.
  - 03_explain_segments.py — Computes per-building temperature elasticity.
  - 04_make_plots.py — Core figures.
  - 04a_extra_plots.py — Expanded suite of figures
  - predict.py — Executable to predict next-hour usage for each building from a usage file and an optional weather file.

- data/
  We kept only sample of final dataset with engineered features to reduce size
  - processed/ — Engineered features samples.

- outputs/
  - figures/ — All PNG figures.
  - metrics/metrics.json — MAE/RMSE for Naive, RF, HGBR on val/test with time cutoffs.
  - tables/ — Supporting CSV/JSON tables (feature importance, per-building MAE, preds sample, elasticity summary, data dictionary).
  - vulnerability/building_elasticity.csv — Elasticity and segments per building.

- report/
  - paper.tex — LaTeX source.
  - papr.pdf

Environment setup
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt

Pipeline of execution
- Unifying inputs
  source .venv/bin/activate && python scripts/00_unify_datasets.py \
    --ausgrid data/ausgrid_with_weather_normalized.csv \
    --lcl data/lcl_with_weather_normalized.parquet \
    --out data/combined_with_weather_normalized.csv

- Build features
  source .venv/bin/activate && python scripts/01_build_features.py \
    --input data/combined_with_weather_normalized.csv \
    --out data/processed/features

- Training models
  source .venv/bin/activate && python scripts/02_train_models.py \
    --features data/processed/features.csv \
    --max-train 500000 --max-val 100000 --max-test -1

- SHAP analysis
  source .venv/bin/activate && python scripts/02b_shap_analysis.py

- Residual diagnostics
  source .venv/bin/activate && python scripts/02c_residual_diagnostics.py

- Explaining and segment vulnerability
  source .venv/bin/activate && python scripts/03_explain_segments.py

- Generating figures
  source .venv/bin/activate && python scripts/04_make_plots.py
  source .venv/bin/activate && python scripts/04a_extra_plots.py

- Rendering LaTeX report
  cd report && tectonic paper.tex

Prediction CLI (scripts/predict.py)
- Given a usage file (CSV or Parquet) and a region, predict next-hour usage per building.
- Minimal input --input:
  - building_name, full_timestamp, usage_kwh_norm OR usage_kwh
  - Optional: apparent_temperature_norm OR apparent_temperature, precipitation, is_day, is_weekend, is_holiday
- Optional --weather CSV:
  - time/full_timestamp, apparent_temperature_norm (or apparent_temperature), precipitation (optional), is_day (optional)
- Region: --region NSW|LCL (used when no region column is present)

  Prediction example with an external weather file:
   source .venv/bin/activate && python scripts/predict.py \
     --input my_usage.csv \
     --region LCL \
     --weather my_weather.csv \
     --out outputs/tables/predictions_next_hour.csv
