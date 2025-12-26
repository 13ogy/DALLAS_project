# Data Dictionary - Features Dataset

## IDs
- building_name: String identifier of the building/site
- full_timestamp: Timestamp (hourly)
- region: Region label (e.g., NSW, LCL)
- region_id: Integer-encoded region

## Targets
- y_next: Next-hour normalized usage per building (usage_pb shifted -1)

## Usage fields
- usage_kwh_norm_orig: Original normalized usage from source CSV (likely global scaling)
- usage_pb: Per-building MinMax re-normalized usage (0-1 within building)

## Weather
- apparent_temperature_norm: Normalized apparent temperature (global, from source)
- precipitation: Hourly precipitation (sum)
- is_day: 1 if daylight hour else 0
- temp_lag_1h: Apparent temperature (normalized) previous hour
- temp_lag_24h: Apparent temperature (normalized) same hour previous day

## Calendar
- hour: Hour of day [0..23]
- day_of_week: Day of week [0=Mon..6=Sun]
- month: Month [1..12]
- season: Season (AU mapping)

## Flags
- is_holiday: 1 if NSW public holiday else 0
- is_weekend: 1 if Saturday/Sunday else 0

## Dynamics
- lag_1h: Usage (usage_pb) previous hour
- lag_24h: Usage (usage_pb) same hour previous day
- rollmean_24h: Mean of last 24 hours of usage_pb (excluding current hour)

## Stats
- rows: 74498698
- unique_buildings: 3835
- time_start: 2007-01-02 00:00:00
- time_end: 2022-04-30 22:00:00
- wrote_parquet: True
- csv_path: data/processed/features.csv
- parquet_path: data/processed/features.parquet
