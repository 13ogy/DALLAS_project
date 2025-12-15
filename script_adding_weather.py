import pandas as pd
import numpy as np
from pathlib import Path

USAGE_FILE = "all_buildings_merged_standardized.csv"
WEATHER_FILE = "australia_nsw_weather.csv"
OUTPUT_FILE = "all_buildings_hourly_with_weather_normalized.csv"

def min_max_normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    min_v = s.min()
    max_v = s.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series(0.0, index=s.index)
    return (s - min_v) / (max_v - min_v)

def find_weather_header_row(path: str) -> int:
    preview = pd.read_csv(path, header=None, nrows=20)
    for i in range(len(preview)):
        row_vals = preview.iloc[i].astype(str).str.lower().tolist()
        if any(v.strip() == "time" for v in row_vals):
            return i
    return 3  # fallback (skip first 3 metadata rows)

def load_and_prepare_weather(path: str) -> pd.DataFrame:
    header_row = find_weather_header_row(path)
    w = pd.read_csv(path, header=header_row)
    w.columns = [str(c).strip().lower() for c in w.columns]

    # Pick required columns by fuzzy matching (handles unit suffixes)
    def pick(cols_needed):
        sel = []
        for c in cols_needed:
            matches = [k for k in w.columns if c in k]
            if not matches:
                return None
            sel.append(matches[0])
        return sel

    cols = pick(["time", "apparent_temperature", "precipitation", "is_day"])
    if cols is None:
        raise ValueError(f"Could not find required weather columns in: {list(w.columns)}")
    time_col, temp_col, precip_col, isday_col = cols

    w = w[[time_col, temp_col, precip_col, isday_col]].copy()
    w.rename(
        columns={
            time_col: "time",
            temp_col: "apparent_temperature",
            precip_col: "precipitation",
            isday_col: "is_day",
        },
        inplace=True,
    )

    w["time"] = pd.to_datetime(w["time"], errors="coerce")
    w = w.dropna(subset=["time"])
    w["timestamp_hour"] = w["time"].dt.floor("H")

    # Ensure numeric types
    for c in ["apparent_temperature", "precipitation", "is_day"]:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w["is_day"] = w["is_day"].fillna(0).astype(int)

    # Aggregate to one row per hour if needed
    w = (
        w.groupby("timestamp_hour", as_index=False)
        .agg({
            "apparent_temperature": "mean",
            "precipitation": "sum",
            "is_day": "max"
        })
    )

    # Normalize temperature before merging (global min-max)
    w["apparent_temperature_norm"] = min_max_normalize(w["apparent_temperature"])
    return w

def load_and_prepare_usage(path: str) -> pd.DataFrame:
    u = pd.read_csv(path)
    u["full_timestamp"] = pd.to_datetime(u["full_timestamp"], errors="coerce")
    u = u.dropna(subset=["full_timestamp"])
    u["timestamp_hour"] = u["full_timestamp"].dt.floor("H")

    u["usage_kwh"] = pd.to_numeric(u["usage_kwh"], errors="coerce")
    u = u.dropna(subset=["usage_kwh"])

    # Normalize usage before merging (global min-max)
    u["usage_kwh_norm"] = min_max_normalize(u["usage_kwh"])
    return u

def main():
    for f in [USAGE_FILE, WEATHER_FILE]:
        if not Path(f).exists():
            raise FileNotFoundError(f"Required file not found: {f}")

    weather = load_and_prepare_weather(WEATHER_FILE)
    usage = load_and_prepare_usage(USAGE_FILE)

    # Merge on floored hour (many usage rows per hour -> one weather row per hour)
    merged = usage.merge(
        weather,
        on="timestamp_hour",
        how="left",
        validate="m:1"
    )

    # Keep only hourly rows with weather present:
    # - full_timestamp exactly on the hour
    # - weather data exists (use normalized temp as indicator)
    is_on_hour = merged["full_timestamp"].dt.floor("H") == merged["full_timestamp"]
    has_weather = merged["apparent_temperature_norm"].notna()
    filtered = merged[is_on_hour & has_weather].copy()

    # Drop columns per request:
    # - season
    # - timestamp_hour
    # - usage_kwh (keep only usage_kwh_norm)
    # - apparent_temperature (keep only apparent_temperature_norm)
    cols_to_drop = [c for c in ["season", "timestamp_hour", "usage_kwh", "apparent_temperature"] if c in filtered.columns]
    filtered.drop(columns=cols_to_drop, inplace=True)

    # Optional: reorder columns
    preferred_order = [
        "building_name", "full_timestamp",
        "usage_kwh_norm",
        "apparent_temperature_norm",
        "precipitation", "is_day",
        "is_holiday", "is_weekend",
    ]
    existing_order = [c for c in preferred_order if c in filtered.columns]
    filtered = filtered[existing_order + [c for c in filtered.columns if c not in existing_order]]

    filtered.to_csv(OUTPUT_FILE, index=False)

    # Diagnostics
    total_before = len(merged)
    total_hourly = is_on_hour.sum()
    total_after = len(filtered)
    print(f"Total rows before: {total_before:,}")
    print(f"Rows exactly on the hour: {total_hourly:,}")
    print(f"Rows kept (hourly with weather): {total_after:,}")
    print(f"Wrote: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
