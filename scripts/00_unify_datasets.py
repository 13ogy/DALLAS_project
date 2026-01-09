#!/usr/bin/env python3
"""
Inputs:
- data/ausgrid_with_weather_normalized.csv
- data/lcl_with_weather_normalized.parquet

Output:
- data/combined_with_weather_normalized.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import holidays
except Exception as e:
    holidays = None


def ensure_dirs():
    Path("data").mkdir(parents=True, exist_ok=True)


def min_max_normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    min_v = s.min()
    max_v = s.max()
    denom = max(max_v - min_v, 1e-9)
    if pd.isna(min_v) or pd.isna(max_v):
        return pd.Series(np.nan, index=s.index)
    return (s - min_v) / denom


def pick_first(df: pd.DataFrame, candidates) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    # exact then contains
    for c in candidates:
        if c in cols:
            return cols[c]
    for key in cols.keys():
        if any(c in key for c in candidates):
            return cols[key]
    return None


def standardize_frame(df: pd.DataFrame, region: str) -> pd.DataFrame:
    # Lower-case helper mapping
    cols_map = {c.lower(): c for c in df.columns}

    # --- Timestamp ---
    ts_col = pick_first(df, ["full_timestamp"])
    if ts_col is None:
        # try 'time' or 'timestamp'
        ts_col = pick_first(df, ["timestamp", "time", "datetime"])
    if ts_col is None:
        raise ValueError("No timestamp column found (expected 'full_timestamp' or similar).")
    df["full_timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=["full_timestamp"]).copy()

    # Keeping only hourly rows
    on_hour = df["full_timestamp"].dt.floor("h") == df["full_timestamp"]
    df = df.loc[on_hour].copy()

    # building_name / id
    bld_col = pick_first(df, ["building_name", "building_id", "meter_id", "site", "household", "zone substation", "id", "customer_id", "customer"])
    if bld_col is None:
        raise ValueError("No building identifier column found.")
    df["building_name"] = df[bld_col].astype(str)

    # usage normalization
    usage_norm_col = pick_first(df, [
        "usage_kwh_norm", "consumption_norm", "kwh_norm",
        "usage_kwh_normalised", "usage_kwh_normalized",
        "usage_normalised", "usage_normalized",
        "usage_norm", "normalized_usage", "normalised_usage"
    ])
    usage_kwh_col = pick_first(df, ["usage_kwh", "kwh", "consumption_kwh", "consumption"])
    if usage_norm_col is not None:
        df["usage_kwh_norm"] = pd.to_numeric(df[usage_norm_col], errors="coerce")
    elif usage_kwh_col is not None:
        tmp = pd.to_numeric(df[usage_kwh_col], errors="coerce")
        # Global min-max
        df["usage_kwh_norm"] = min_max_normalize(tmp)
    else:
        raise ValueError("No usage column found (expected usage_kwh_norm or usage_kwh).")

    # --- Weather ---
    atn_col = pick_first(df, ["apparent_temperature_norm", "normalized_temperature", "temperature_norm", "temp_norm"])
    at_col = pick_first(df, ["apparent_temperature", "temperature", "temp"])
    if atn_col is not None:
        df["apparent_temperature_norm"] = pd.to_numeric(df[atn_col], errors="coerce")
        if at_col is None:
            df["apparent_temperature"] = pd.NA
        else:
            df["apparent_temperature"] = pd.to_numeric(df[at_col], errors="coerce")
    else:
        if at_col is None:
            raise ValueError("No temperature column found (apparent_temperature_norm or apparent_temperature).")
        df["apparent_temperature"] = pd.to_numeric(df[at_col], errors="coerce")

    pr_col = pick_first(df, ["precipitation", "rain", "precip"])
    if pr_col is not None:
        df["precipitation"] = pd.to_numeric(df[pr_col], errors="coerce")
    else:
        df["precipitation"] = np.nan

    day_col = pick_first(df, ["is_day"])
    if day_col is not None:
        df["is_day"] = pd.to_numeric(df[day_col], errors="coerce").fillna(0).astype(np.int8)
    else:
        # Fallback: 6:00-18:00 treated as day
        hours = df["full_timestamp"].dt.hour
        df["is_day"] = ((hours >= 6) & (hours <= 18)).astype(np.int8)

    wk_col = pick_first(df, ["is_weekend"])
    if wk_col is not None:
        df["is_weekend"] = pd.to_numeric(df[wk_col], errors="coerce").fillna(0).astype(np.int8)
    else:
        dow = df["full_timestamp"].dt.dayofweek
        df["is_weekend"] = (dow >= 5).astype(np.int8)

    hol_col = pick_first(df, ["is_holiday"])
    if hol_col is not None:
        df["is_holiday"] = pd.to_numeric(df[hol_col], errors="coerce").fillna(0).astype(np.int8)
    else:
        if holidays is None:
            df["is_holiday"] = 0
        else:
            dates = df["full_timestamp"].dt.date
            if region.upper() == "NSW":
                hol = holidays.AU(state="NSW", years=sorted(set(df["full_timestamp"].dt.year.tolist())))
            else:
                # LCL -> Great Britain holidays
                hol = holidays.GB(years=sorted(set(df["full_timestamp"].dt.year.tolist())))
            df["is_holiday"] = dates.isin(set(hol.keys())).astype(np.int8)

    # dropping rows without usage or temperature
    temp_filter_col = "apparent_temperature" if "apparent_temperature" in df.columns else "apparent_temperature_norm"
    df = df.dropna(subset=["usage_kwh_norm", temp_filter_col]).copy()

    # normalized usage
    df["usage_kwh_norm"] = df["usage_kwh_norm"].clip(0.0, 1.0)
    if "apparent_temperature_norm" in df.columns:
        df["apparent_temperature_norm"] = df["apparent_temperature_norm"].clip(0.0, 1.0)

    # Region
    df["region"] = region

    keep_cols = [
        "building_name", "full_timestamp",
        "usage_kwh_norm",
        "apparent_temperature_norm", "precipitation", "is_day",
        "is_weekend", "is_holiday",
        "region"
    ]
    # Keep any additional columns too (but order preferred first)
    cols = [c for c in keep_cols if c in df.columns] + [c for c in df.columns if c not in keep_cols]
    out = df[cols].copy()

    return out


def load_source(path: Path, region: str) -> pd.DataFrame:
    if not path.exists():
        print(f"Warning: source not found: {path}")
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(f"Reading parquet failed for {path}. Ensure 'pyarrow' is installed. ({e})")
    else:
        df = pd.read_csv(path)
    return standardize_frame(df, region=region)


def main():
    parser = argparse.ArgumentParser(description="Unify multiple normalized sources into one CSV with region field.")
    parser.add_argument("--ausgrid", type=str, default="data/ausgrid_with_weather_normalized.csv",
                        help="Path to Ausgrid CSV (normalized)")
    parser.add_argument("--lcl", type=str, default="data/lcl_with_weather_normalized.parquet",
                        help="Path to LCL Parquet (normalized)")
    parser.add_argument("--out", type=str, default="data/combined_with_weather_normalized.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    ensure_dirs()

    df_au = load_source(Path(args.ausgrid), region="NSW")
    df_lcl = load_source(Path(args.lcl), region="LCL")

    sources = [d for d in [df_au, df_lcl] if not d.empty]
    if not sources:
        raise RuntimeError("No valid sources were loaded. Check input paths.")

    combined = pd.concat(sources, ignore_index=True)
    combined = combined.sort_values(["region", "building_name", "full_timestamp"]).reset_index(drop=True)

    # Diagnostics
    n_rows = len(combined)
    n_bld = combined["building_name"].nunique() if "building_name" in combined.columns else 0
    regions = combined["region"].value_counts(dropna=False).to_dict() if "region" in combined.columns else {}

    out_path = Path(args.out)
    combined.to_csv(out_path, index=False)

    print("Unification complete.")
    print(f"Rows: {n_rows:,}")
    print(f"Unique buildings: {n_bld}")
    print(f"By region: {regions}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
