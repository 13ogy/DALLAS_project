#!/usr/bin/env python3
"""
Predict next-hour usage per building from an input usage file and region.

Usage examples:
  python3 scripts/predict.py --input data/ausgrid_with_weather_normalized.csv --region NSW
  python3 scripts/predict.py --input my_usage.csv --region LCL --weather my_weather.csv
  python3 scripts/predict.py --input my_usage.parquet --region NSW --model outputs/models/hgbr.joblib

Inputs:
- --input: CSV or Parquet with at least:
    - building_name (str)
    - full_timestamp (datetime, any minute; we will floor to hour)
    - usage_kwh_norm OR usage_kwh
    - Optional: apparent_temperature_norm OR apparent_temperature, precipitation, is_day, is_weekend, is_holiday
- --region: Region label for all rows if region column is missing (NSW or LCL)
- --weather: Optional CSV with hourly weather to merge if input lacks weather.
    The script expects columns (case-insensitive/fuzzy):
      - time or full_timestamp
      - apparent_temperature_norm (preferred) or apparent_temperature (we normalize)
      - precipitation (optional)
      - is_day (optional)
    Weather is aggregated to one row per hour and merged by floored hour.
- --model: Optional explicit model path. If not provided, the script will:
    1) Inspect outputs/metrics/metrics.json to decide HGBR vs RF
    2) Load the respective joblib file from outputs/models/
- --out: Output CSV path (default: outputs/tables/predictions_next_hour.csv)

Behavior:
- Computes leakage-safe features from the input history per building:
  lag_1h, lag_24h, rollmean_24h (24-hour mean excluding current hour),
  calendar fields, weather, flags, and region_id.
- Uses the most recent hour (per building) as feature row to predict y_next at t+1.
- If a required feature is unavailable for a building (e.g., insufficient history),
  that building is skipped and reported.

Outputs:
- CSV with columns: building_name, timestamp_last, timestamp_next, y_pred_next, and features used for audit.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import joblib


# -------------------------- Utilities -------------------------- #

def ensure_dirs():
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)


def per_building_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    min_v = s.min()
    max_v = s.max()
    denom = max(max_v - min_v, 1e-6)
    return (s - min_v) / denom


def pick_first(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    # contains
    for k in lower_map.keys():
        if any(c in k for c in candidates):
            return lower_map[k]
    return None


def load_weather_df(path: str) -> pd.DataFrame:
    """
    Load weather CSV and return hourly-aggregated with:
      timestamp_hour, apparent_temperature_norm, precipitation, is_day
    """
    w = pd.read_csv(path)
    w.columns = [str(c).strip().lower() for c in w.columns]

    tcol = pick_first(list(w.columns), ["full_timestamp", "timestamp", "time", "datetime"])
    if tcol is None:
        raise ValueError("Weather file must contain a 'time' or 'full_timestamp' column.")

    atn_col = pick_first(list(w.columns), ["apparent_temperature_norm"])
    at_col = pick_first(list(w.columns), ["apparent_temperature"])
    pr_col = pick_first(list(w.columns), ["precipitation", "precip", "rain"])
    day_col = pick_first(list(w.columns), ["is_day"])

    w["time_parsed"] = pd.to_datetime(w[tcol], errors="coerce")
    w = w.dropna(subset=["time_parsed"]).copy()
    w["timestamp_hour"] = w["time_parsed"].dt.floor("h")

    if atn_col is not None:
        w["apparent_temperature_norm"] = pd.to_numeric(w[atn_col], errors="coerce")
    elif at_col is not None:
        tmp = pd.to_numeric(w[at_col], errors="coerce")
        # global normalization for weather input
        mn, mx = tmp.min(), tmp.max()
        denom = max(mx - mn, 1e-6)
        w["apparent_temperature_norm"] = (tmp - mn) / denom
    else:
        raise ValueError("Weather file must contain apparent_temperature_norm or apparent_temperature.")

    if pr_col is not None:
        w["precipitation"] = pd.to_numeric(w[pr_col], errors="coerce")
    else:
        w["precipitation"] = 0.0

    if day_col is not None:
        w["is_day"] = pd.to_numeric(w[day_col], errors="coerce").fillna(0).astype(np.int8)
    else:
        # derive simple day/night if missing
        h = w["timestamp_hour"].dt.hour
        w["is_day"] = ((h >= 6) & (h <= 18)).astype(np.int8)

    wh = (
        w.groupby("timestamp_hour", as_index=False)
         .agg({
             "apparent_temperature_norm": "mean",
             "precipitation": "sum",
             "is_day": "max"
         })
    )
    return wh


def derive_flags(df: pd.DataFrame, region: str) -> pd.DataFrame:
    # is_weekend if missing
    if "is_weekend" not in df.columns:
        dow = df["full_timestamp"].dt.dayofweek
        df["is_weekend"] = (dow >= 5).astype(np.int8)
    else:
        df["is_weekend"] = pd.to_numeric(df["is_weekend"], errors="coerce").fillna(0).astype(np.int8)

    # is_holiday if missing: use 'holidays' lib, else default 0
    if "is_holiday" not in df.columns:
        try:
            import holidays as hlib
            years = sorted(set(df["full_timestamp"].dt.year.tolist()))
            if region.upper() == "NSW":
                hol = hlib.AU(state="NSW", years=years)
            else:
                hol = hlib.GB(years=years)
            df["is_holiday"] = df["full_timestamp"].dt.date.isin(set(hol.keys())).astype(np.int8)
        except Exception:
            df["is_holiday"] = 0
    else:
        df["is_holiday"] = pd.to_numeric(df["is_holiday"], errors="coerce").fillna(0).astype(np.int8)

    return df


def region_to_id(region: str) -> int:
    # Stabilize mapping consistent with sorted(["LCL","NSW"]) => {"LCL":0,"NSW":1}
    return 0 if region.upper() == "LCL" else 1


def load_best_model_and_feature_order(explicit_model: Optional[str]) -> Tuple[object, List[str]]:
    """
    Decide which model to load and what feature order to use.
    Feature order priority:
      1) outputs/tables/feature_importance_permutation_val.csv (column 'feature' in file order)
      2) fallback to candidate feature list
    """
    # Decide model
    model_path = None
    if explicit_model:
        model_path = explicit_model
    else:
        metrics_path = Path("outputs/metrics/metrics.json")
        best = None
        if metrics_path.exists():
            try:
                with open(metrics_path, "r") as f:
                    m = json.load(f)
                mae_rf = m.get("RF", {}).get("val", {}).get("MAE", None)
                mae_hg = m.get("HGBR", {}).get("val", {}).get("MAE", None)
                if mae_hg is not None and (mae_rf is None or mae_hg <= mae_rf):
                    best = "hgbr"
                else:
                    best = "rf"
            except Exception:
                best = None
        # Fallback: prefer HGBR if present, else RF
        if best is None:
            if Path("outputs/models/hgbr.joblib").exists():
                best = "hgbr"
            elif Path("outputs/models/rf.joblib").exists():
                best = "rf"
        if best is None:
            raise FileNotFoundError("No trained model found. Expected outputs/models/hgbr.joblib or rf.joblib.")
        model_path = f"outputs/models/{best}.joblib"

    model = joblib.load(model_path)

    # Feature order
    feat_path = Path("outputs/tables/feature_importance_permutation_val.csv")
    if feat_path.exists():
        try:
            df = pd.read_csv(feat_path)
            if "feature" in df.columns:
                order = df["feature"].tolist()
                return model, order
        except Exception:
            pass

    # Fallback candidate order (same as training candidate list)
    candidate_features = [
        "lag_1h", "lag_24h", "rollmean_24h",
        "hour", "day_of_week", "month",
        "apparent_temperature_norm", "temp_lag_1h", "temp_lag_24h",
        "precipitation",
        "is_day", "is_weekend", "is_holiday",
        "region_id"
    ]
    return model, candidate_features


# -------------------------- Core logic -------------------------- #

def build_features_for_prediction(
    usage_df: pd.DataFrame,
    region_label: str,
    weather_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Return one feature row per building for the most recent hour available,
    to predict y_next at t+1.
    """
    df = usage_df.copy()
    # Standardize columns
    cols = [str(c) for c in df.columns]
    ts_col = pick_first(cols, ["full_timestamp", "timestamp", "time", "datetime"])
    if ts_col is None:
        raise ValueError("Input must contain 'full_timestamp' (or similar) column.")

    bcol = pick_first(cols, ["building_name", "meter_id", "site", "household", "zone substation"])
    if bcol is None:
        raise ValueError("Input must contain a building identifier (e.g., building_name).")

    df["building_name"] = df[bcol].astype(str)
    df["full_timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=["full_timestamp"]).copy()
    df["timestamp_hour"] = df["full_timestamp"].dt.floor("h")

    # Usage normalization
    usage_norm_col = pick_first(cols, ["usage_kwh_norm", "consumption_norm", "kwh_norm"])
    usage_kwh_col = pick_first(cols, ["usage_kwh", "kwh", "consumption_kwh", "consumption"])
    if usage_norm_col is not None:
        df["usage_kwh_norm"] = pd.to_numeric(df[usage_norm_col], errors="coerce")
    elif usage_kwh_col is not None:
        tmp = pd.to_numeric(df[usage_kwh_col], errors="coerce")
        # global normalize (compat); per-building renorm next
        mn, mx = tmp.min(), tmp.max()
        denom = max(mx - mn, 1e-6)
        df["usage_kwh_norm"] = (tmp - mn) / denom
    else:
        raise ValueError("Input must contain usage_kwh_norm or usage_kwh.")

    # Weather presence?
    atn_in_input = pick_first(cols, ["apparent_temperature_norm"]) is not None
    pr_in_input = pick_first(cols, ["precipitation", "precip", "rain"]) is not None
    day_in_input = pick_first(cols, ["is_day"]) is not None

    if not atn_in_input and weather_df is None:
        # Cannot run model without weather. Provide a conservative fallback.
        # We set apparent_temperature_norm=0.5 (mid), precipitation=0.0, derive is_day.
        print("Warning: No weather found in input and no --weather provided. Using fallback values.")
        df["apparent_temperature_norm"] = 0.5
        df["precipitation"] = 0.0
        df["is_day"] = df["timestamp_hour"].dt.hour.between(6, 18).astype(np.int8)
    else:
        if weather_df is not None:
            wh = weather_df.copy()
        else:
            # Assemble weather from input columns
            wh = df[["timestamp_hour"]].copy()
            wh["apparent_temperature_norm"] = pd.to_numeric(df[pick_first(cols, ["apparent_temperature_norm"])], errors="coerce")
            wh["precipitation"] = pd.to_numeric(
                df[pick_first(cols, ["precipitation", "precip", "rain"])], errors="coerce"
            ) if pr_in_input else 0.0
            if day_in_input:
                wh["is_day"] = pd.to_numeric(df[pick_first(cols, ["is_day"])], errors="coerce").fillna(0).astype(np.int8)
            else:
                h = df["timestamp_hour"].dt.hour
                wh["is_day"] = ((h >= 6) & (h <= 18)).astype(np.int8)
            wh = (
                wh.groupby("timestamp_hour", as_index=False)
                  .agg({"apparent_temperature_norm": "mean", "precipitation": "sum", "is_day": "max"})
            )
        # Drop conflicting columns before merge to avoid _x/_y suffixes
        for c in ["apparent_temperature_norm", "precipitation", "is_day"]:
            if c in df.columns:
                df = df.drop(columns=[c])

        # Merge on floored hour
        df = df.merge(wh, on="timestamp_hour", how="left", validate="m:1")

    # Ensure weather columns exist after merge (fallbacks if missing)
    if "apparent_temperature_norm" not in df.columns:
        print("Warning: no apparent_temperature_norm after weather merge; using fallback 0.5")
        df["apparent_temperature_norm"] = 0.5
    if "precipitation" not in df.columns:
        df["precipitation"] = 0.0
    if "is_day" not in df.columns:
        h = df["timestamp_hour"].dt.hour
        df["is_day"] = ((h >= 6) & (h <= 18)).astype(np.int8)

    # Keep only rows with weather present and exactly on the hour
    is_on_hour = df["full_timestamp"] == df["timestamp_hour"]
    df = df[is_on_hour & df["apparent_temperature_norm"].notna()].copy()

    # Derive flags (weekend/holiday) if missing
    df = derive_flags(df, region_label)

    # Region fields
    df["region"] = region_label
    df["region_id"] = region_to_id(region_label)

    # Sort and recompute per-building normalization (usage_pb)
    df = df.sort_values(["building_name", "full_timestamp"]).reset_index(drop=True)
    df["usage_pb"] = (
        df.groupby("building_name", group_keys=False)["usage_kwh_norm"]
          .apply(per_building_minmax)
          .astype(np.float32)
    )

    # Calendar
    df["hour"] = df["full_timestamp"].dt.hour.astype(np.int8)
    df["day_of_week"] = df["full_timestamp"].dt.dayofweek.astype(np.int8)
    df["month"] = df["full_timestamp"].dt.month.astype(np.int8)

    # Group to compute leakage-safe dynamics
    g = df.groupby("building_name", group_keys=False)
    df["lag_1h"] = g["usage_pb"].shift(1).astype(np.float32)
    df["lag_24h"] = g["usage_pb"].shift(24).astype(np.float32)
    df["rollmean_24h"] = g["usage_pb"].apply(lambda s: s.shift(1).rolling(window=24, min_periods=24).mean()).astype(np.float32)

    # Temperature lags
    df["temp_lag_1h"] = g["apparent_temperature_norm"].shift(1).astype(np.float32)
    df["temp_lag_24h"] = g["apparent_temperature_norm"].shift(24).astype(np.float32)

    # For each building, take the most recent hour row with all required key features
    req = ["lag_1h", "lag_24h", "rollmean_24h", "apparent_temperature_norm"]
    df_valid = df.dropna(subset=[c for c in req if c in df.columns]).copy()

    # Get last row per building
    idx = df_valid.groupby("building_name")["full_timestamp"].idxmax()
    last = df_valid.loc[idx].copy()
    last = last.sort_values("building_name").reset_index(drop=True)

    # timestamp_next = last_hour + 1h
    last["timestamp_last"] = last["full_timestamp"]
    last["timestamp_next"] = last["timestamp_last"] + pd.Timedelta(hours=1)

    return last


def main():
    parser = argparse.ArgumentParser(description="Predict next-hour usage per building from an input usage file and region.")
    parser.add_argument("--input", type=str, required=True, help="Path to input usage file (CSV or Parquet).")
    parser.add_argument("--region", type=str, default="NSW", help="Region label to assign if missing (NSW or LCL).")
    parser.add_argument("--weather", type=str, default=None, help="Optional path to weather CSV to merge.")
    parser.add_argument("--model", type=str, default=None, help="Optional explicit model path (joblib).")
    parser.add_argument("--out", type=str, default="outputs/tables/predictions_next_hour.csv", help="Output CSV path.")
    args = parser.parse_args()

    ensure_dirs()

    # Load model and feature order
    model, feature_order = load_best_model_and_feature_order(args.model)

    # Load input
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    if in_path.suffix.lower() == ".parquet":
        try:
            df_in = pd.read_parquet(in_path)
        except Exception as e:
            raise RuntimeError(f"Reading parquet failed. Ensure 'pyarrow' is installed. ({e})")
    else:
        df_in = pd.read_csv(in_path)

    # Optional weather
    wh = None
    if args.weather:
        wh = load_weather_df(args.weather)

    # Build features
    feats = build_features_for_prediction(df_in, args.region, weather_df=wh)

    if feats.empty:
        raise RuntimeError("No valid rows to predict (insufficient history or missing required features).")

    # Assemble X in the expected feature order
    for c in feature_order:
        if c not in feats.columns:
            # fill missing optional columns with 0
            feats[c] = 0.0
    X = feats[feature_order].to_numpy(dtype=np.float32)

    # Predict
    yhat = model.predict(X)

    out_cols = ["building_name", "timestamp_last", "timestamp_next"]
    out_df = feats.copy()
    out_df["y_pred_next"] = yhat.astype(np.float32)
    # Keep a compact audit: features used (optional; can be large)
    audit_cols = feature_order
    out_df = out_df[out_cols + ["y_pred_next"] + audit_cols]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # Summary
    summary = {
        "n_predicted": int(out_df.shape[0]),
        "region": args.region,
        "model_path": str(args.model) if args.model else "(auto)",
        "feature_order_source": "feature_importance_permutation_val.csv" if Path("outputs/tables/feature_importance_permutation_val.csv").exists() else "fallback_candidate_list",
        "output_csv": str(out_path)
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
