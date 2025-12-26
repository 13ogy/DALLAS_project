#!/usr/bin/env python3
"""
Build leakage-safe features and next-hour target from a unified CSV:
data/combined_with_weather_normalized.csv (default; can accept single-source CSV)

Outputs:
- data/processed/features.parquet (if parquet engine available) and features.csv
- outputs/tables/data_dictionary.md (description of columns)
- Console diagnostics (rows kept, buildings, date range)

Notes:
- Recomputes per-building MinMax on usage_kwh_norm to mitigate global scaling bias.
- Adds calendar features, lag_1h, lag_24h, rollmean_24h (24-hour mean excluding current hour).
- Defines target y_next = next-hour usage per building.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd


SEASON_MAP = {
    12: "Summer", 1: "Summer", 2: "Summer",
    3: "Autumn", 4: "Autumn", 5: "Autumn",
    6: "Winter", 7: "Winter", 8: "Winter",
    9: "Spring", 10: "Spring", 11: "Spring"
}


def ensure_dirs():
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)


def per_building_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    min_v = s.min()
    max_v = s.max()
    denom = max(max_v - min_v, 1e-6)
    return (s - min_v) / denom


def build_features(input_csv: str, output_base: str = "data/processed/features") -> dict:
    ensure_dirs()

    print(f"Reading input CSV: {input_csv}")
    df = pd.read_csv(input_csv, parse_dates=["full_timestamp"])

    # Basic sort and dtypes
    df = df.sort_values(["building_name", "full_timestamp"]).reset_index(drop=True)

    # Coerce numeric columns
    for c in ["usage_kwh_norm", "apparent_temperature_norm", "precipitation"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Flags to ints
    for c in ["is_day", "is_holiday", "is_weekend"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int8)

    # Drop rows without timestamps or required numeric cols
    df = df.dropna(subset=["full_timestamp", "usage_kwh_norm", "apparent_temperature_norm"]).copy()

    # Region handling (categorical + encoded)
    if "region" not in df.columns:
        df["region"] = "NSW"
    df["region"] = df["region"].astype(str)
    region_map = {name: idx for idx, name in enumerate(sorted(df["region"].unique()))}
    df["region_id"] = df["region"].map(region_map).astype(np.int8)

    # Recompute per-building MinMax on usage_kwh_norm (approximate per-building normalization)
    # Keep original for reference
    df["usage_kwh_norm_orig"] = df["usage_kwh_norm"].astype(np.float32)
    df["usage_pb"] = (
        df.groupby("building_name", group_keys=False)["usage_kwh_norm_orig"]
        .apply(per_building_minmax)
        .astype(np.float32)
    )

    # Calendar features
    df["hour"] = df["full_timestamp"].dt.hour.astype(np.int8)
    df["day_of_week"] = df["full_timestamp"].dt.dayofweek.astype(np.int8)
    df["month"] = df["full_timestamp"].dt.month.astype(np.int8)
    df["season"] = df["month"].map(SEASON_MAP)

    # Group for leakage-safe dynamics
    g = df.groupby("building_name", group_keys=False)

    # Lags and rolling (exclude current hour via shift(1))
    df["lag_1h"] = g["usage_pb"].shift(1).astype(np.float32)
    df["lag_24h"] = g["usage_pb"].shift(24).astype(np.float32)
    # Rolling 24h mean of prior hours
    df["rollmean_24h"] = (
        g["usage_pb"].apply(lambda s: s.shift(1).rolling(window=24, min_periods=24).mean())
    ).astype(np.float32)

    # Temperature lags (leakage-safe, per building)
    if "apparent_temperature_norm" in df.columns:
        df["temp_lag_1h"] = g["apparent_temperature_norm"].shift(1).astype(np.float32)
        df["temp_lag_24h"] = g["apparent_temperature_norm"].shift(24).astype(np.float32)

    # Target: next-hour usage per building
    df["y_next"] = g["usage_pb"].shift(-1).astype(np.float32)

    # Feature selection and cleaning
    required_cols = ["lag_1h", "lag_24h", "rollmean_24h", "y_next",
                     "apparent_temperature_norm"]
    df = df.dropna(subset=[c for c in required_cols if c in df.columns]).copy()

    # Downcast numerics to save space
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    for c in float_cols:
        df[c] = df[c].astype(np.float32)

    # Order columns (keep ID/time first)
    preferred = [
        "building_name", "full_timestamp",
        "region", "region_id",
        "usage_kwh_norm_orig", "usage_pb",
        "apparent_temperature_norm", "temp_lag_1h", "temp_lag_24h",
        "precipitation", "is_day",
        "is_holiday", "is_weekend",
        "hour", "day_of_week", "month", "season",
        "lag_1h", "lag_24h", "rollmean_24h",
        "y_next"
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    # Diagnostics
    n_rows = len(df)
    n_bld = df["building_name"].nunique()
    tmin, tmax = df["full_timestamp"].min(), df["full_timestamp"].max()
    print(f"Rows after feature build: {n_rows:,}")
    print(f"Unique buildings: {n_bld}")
    print(f"Time range: {tmin} -> {tmax}")

    # Write outputs
    base = Path(output_base)
    parquet_path = base.with_suffix(".parquet")
    csv_path = base.with_suffix(".csv")
    sample_path = base.with_name(base.name + "_sample.csv")

    # Try Parquet, fallback to CSV only
    wrote_parquet = False
    try:
        df.to_parquet(parquet_path, index=False)
        wrote_parquet = True
        print(f"Wrote {parquet_path}")
    except Exception as e:
        print(f"Parquet write failed ({e}); will write CSV only.")

    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    df.head(5000).to_csv(sample_path, index=False)
    print(f"Wrote sample (first 5k rows) to {sample_path}")

    # Data dictionary
    data_dict = {
        "description": "Leakage-safe features and next-hour target per building (hourly).",
        "ids": {
            "building_name": "String identifier of the building/site",
            "full_timestamp": "Timestamp (hourly)",
            "region": "Region label (e.g., NSW, LCL)",
            "region_id": "Integer-encoded region"
        },
        "targets": {
            "y_next": "Next-hour normalized usage per building (usage_pb shifted -1)"
        },
        "usage_fields": {
            "usage_kwh_norm_orig": "Original normalized usage from source CSV (likely global scaling)",
            "usage_pb": "Per-building MinMax re-normalized usage (0-1 within building)"
        },
        "weather": {
            "apparent_temperature_norm": "Normalized apparent temperature (global, from source)",
            "precipitation": "Hourly precipitation (sum)",
            "is_day": "1 if daylight hour else 0",
            "temp_lag_1h": "Apparent temperature (normalized) previous hour",
            "temp_lag_24h": "Apparent temperature (normalized) same hour previous day"
        },
        "calendar": {
            "hour": "Hour of day [0..23]",
            "day_of_week": "Day of week [0=Mon..6=Sun]",
            "month": "Month [1..12]",
            "season": "Season (AU mapping)"
        },
        "flags": {
            "is_holiday": "1 if NSW public holiday else 0",
            "is_weekend": "1 if Saturday/Sunday else 0"
        },
        "dynamics": {
            "lag_1h": "Usage (usage_pb) previous hour",
            "lag_24h": "Usage (usage_pb) same hour previous day",
            "rollmean_24h": "Mean of last 24 hours of usage_pb (excluding current hour)"
        },
        "stats": {
            "rows": int(n_rows),
            "unique_buildings": int(n_bld),
            "time_start": str(tmin),
            "time_end": str(tmax),
            "wrote_parquet": wrote_parquet,
            "csv_path": str(csv_path),
            "parquet_path": str(parquet_path) if wrote_parquet else None
        }
    }
    dd_path = Path("outputs/tables/data_dictionary.json")
    dd_md_path = Path("outputs/tables/data_dictionary.md")
    dd_path.write_text(json.dumps(data_dict, indent=2))

    # Also write a readable markdown dictionary
    md_lines = [
        "# Data Dictionary - Features Dataset",
        "",
        "## IDs",
        "- building_name: String identifier of the building/site",
        "- full_timestamp: Timestamp (hourly)",
        "- region: Region label (e.g., NSW, LCL)",
        "- region_id: Integer-encoded region",
        "",
        "## Targets",
        "- y_next: Next-hour normalized usage per building (usage_pb shifted -1)",
        "",
        "## Usage fields",
        "- usage_kwh_norm_orig: Original normalized usage from source CSV (likely global scaling)",
        "- usage_pb: Per-building MinMax re-normalized usage (0-1 within building)",
        "",
        "## Weather",
        "- apparent_temperature_norm: Normalized apparent temperature (global, from source)",
        "- precipitation: Hourly precipitation (sum)",
        "- is_day: 1 if daylight hour else 0",
        "- temp_lag_1h: Apparent temperature (normalized) previous hour",
        "- temp_lag_24h: Apparent temperature (normalized) same hour previous day",
        "",
        "## Calendar",
        "- hour: Hour of day [0..23]",
        "- day_of_week: Day of week [0=Mon..6=Sun]",
        "- month: Month [1..12]",
        "- season: Season (AU mapping)",
        "",
        "## Flags",
        "- is_holiday: 1 if NSW public holiday else 0",
        "- is_weekend: 1 if Saturday/Sunday else 0",
        "",
        "## Dynamics",
        "- lag_1h: Usage (usage_pb) previous hour",
        "- lag_24h: Usage (usage_pb) same hour previous day",
        "- rollmean_24h: Mean of last 24 hours of usage_pb (excluding current hour)",
        "",
        "## Stats",
        f"- rows: {n_rows}",
        f"- unique_buildings: {n_bld}",
        f"- time_start: {tmin}",
        f"- time_end: {tmax}",
        f"- wrote_parquet: {wrote_parquet}",
        f"- csv_path: {csv_path}",
        f"- parquet_path: {parquet_path if wrote_parquet else 'N/A'}",
        ""
    ]
    dd_md_path.write_text("\n".join(md_lines))
    print(f"Wrote {dd_path} and {dd_md_path}")

    return data_dict


def main():
    parser = argparse.ArgumentParser(description="Build leakage-safe features and next-hour target from single CSV.")
    parser.add_argument("--input", type=str, default="data/combined_with_weather_normalized.csv",
                        help="Path to input CSV")
    parser.add_argument("--out", type=str, default="data/processed/features",
                        help="Output base path without extension")
    args = parser.parse_args()

    info = build_features(args.input, args.out)
    # Print compact summary
    print(json.dumps({"rows": info["stats"]["rows"],
                      "unique_buildings": info["stats"]["unique_buildings"],
                      "csv": info["stats"]["csv_path"],
                      "parquet": info["stats"]["parquet_path"]}, indent=2))


if __name__ == "__main__":
    main()
