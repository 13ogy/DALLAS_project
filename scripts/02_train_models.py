#!/usr/bin/env python3
"""
Train baselines and tree models on the single-CSV features with time-aware splits.

Inputs:
- data/processed/features.csv  (from scripts/01_build_features.py)

Outputs:
- outputs/models/rf.joblib, outputs/models/hgbr.joblib
- outputs/metrics/metrics.json (MAE/RMSE for Naive, RF, HGBR on Val/Test)
- outputs/tables/feature_importance_permutation_val.csv
- outputs/tables/per_building_mae_test.csv
- outputs/tables/preds_sample_test.csv (small sample for plotting)

Design:
- Time-aware split via approximate timeline percentiles on full_timestamp (reservoir sampling for cutoffs).
- Stream the large CSV in chunks; collect up to configurable max rows per split to bound memory.
- Baseline 0: Naive (predict y_hat = lag_1h).
- Models: RandomForestRegressor, HistGradientBoostingRegressor (sklearn).

CLI:
  python3 scripts/02_train_models.py --features data/processed/features.csv --max-train 500000 --max-val 100000 --max-test 100000
"""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import random
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error


RNG = np.random.default_rng(42)


def ensure_dirs():
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)


def reservoir_time_cutoffs(path: str,
                           train_frac: float = 0.8,
                           val_frac: float = 0.1,
                           sample_size: int = 2_000_000,
                           chunksize: int = 1_000_000) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Compute approximate time cutoffs (t_train_end, t_val_end) using reservoir sampling
    over the full_timestamp column to estimate the 80th and 90th percentiles.
    Robust implementation with a fixed-size reservoir to avoid index errors.
    """
    print(f"Estimating time cutoffs from {path} using reservoir sampling...")
    k = int(sample_size)
    if k <= 0:
        raise ValueError("sample_size must be positive")

    # Fixed-size reservoir and counters
    reservoir = np.empty(k, dtype="int64")
    filled = 0
    seen = 0

    usecols = ["full_timestamp"]
    for chunk in pd.read_csv(path, usecols=usecols, parse_dates=["full_timestamp"],
                             chunksize=chunksize, low_memory=False):
        ts = chunk["full_timestamp"].dropna().astype("int64").to_numpy()  # ns since epoch
        if ts.size == 0:
            continue

        for v in ts:
            if filled < k:
                reservoir[filled] = v
                filled += 1
                seen += 1
                continue

            seen += 1
            # Draw j in [0, seen-1]; replace if j < k
            j = int(RNG.integers(0, seen))
            if j < k:
                reservoir[j] = v

    # Use only the filled part (in case dataset smaller than k)
    if filled == 0:
        raise RuntimeError("No timestamps found to compute cutoffs.")

    sample = reservoir[:filled]
    # Compute quantiles
    q80 = np.quantile(sample, train_frac)
    q90 = np.quantile(sample, train_frac + val_frac)
    t_train_end = pd.to_datetime(int(q80))
    t_val_end = pd.to_datetime(int(q90))
    print(f"Estimated cutoffs: train_end={t_train_end}, val_end={t_val_end}")
    return t_train_end, t_val_end


def collect_split_samples(path: str,
                          t_train_end: pd.Timestamp,
                          t_val_end: pd.Timestamp,
                          max_rows: Dict[str, int],
                          feature_cols: List[str],
                          target_col: str = "y_next",
                          keep_cols_extra: List[str] | None = None,
                          chunksize: int = 500_000) -> Dict[str, pd.DataFrame]:
    """
    Stream through the CSV once, collecting up to max_rows[split] rows for each split.
    Splits:
      - train: full_timestamp <= t_train_end
      - val: t_train_end < full_timestamp <= t_val_end
      - test: full_timestamp > t_val_end
    """
    keep_cols_extra = keep_cols_extra or []
    usecols = ["full_timestamp", target_col] + feature_cols + keep_cols_extra
    got = {"train": 0, "val": 0, "test": 0}
    out = {"train": [], "val": [], "test": []}

    for chunk in pd.read_csv(path, usecols=usecols, parse_dates=["full_timestamp"],
                             chunksize=chunksize, low_memory=False):
        # Drop rows with missing target or key features
        chunk = chunk.dropna(subset=[target_col, "lag_1h", "lag_24h", "rollmean_24h"])
        # Assign split
        ts = chunk["full_timestamp"]
        mask_train = ts <= t_train_end
        mask_val = (ts > t_train_end) & (ts <= t_val_end)
        mask_test = ts > t_val_end

        for split, mask in (("train", mask_train), ("val", mask_val), ("test", mask_test)):
            need = max_rows.get(split, 0) - got[split]
            if need <= 0:
                continue
            sub = chunk.loc[mask, :]
            if sub.empty:
                continue
            if len(sub) > need:
                # random sample to fill the remainder
                sub = sub.sample(n=need, random_state=42)
            out[split].append(sub)
            got[split] += len(sub)

        # stop early if all filled
        if all(got[s] >= max_rows.get(s, 0) for s in got):
            break

    # Concatenate
    result = {}
    for split in ("train", "val", "test"):
        if out[split]:
            df = pd.concat(out[split], ignore_index=True)
        else:
            df = pd.DataFrame(columns=usecols)
        result[split] = df

        print(f"{split}: collected {len(df):,} rows")

    return result


def downcast_types(df: pd.DataFrame, cols_float: List[str], cols_int: List[str]) -> pd.DataFrame:
    for c in cols_float:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    for c in cols_int:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int8)
    return df


def evaluate_split(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    return {"MAE": mae, "RMSE": rmse}


def main():
    parser = argparse.ArgumentParser(description="Train models with time-aware splits from features CSV.")
    parser.add_argument("--features", type=str, default="data/processed/features.csv")
    parser.add_argument("--max-train", type=int, default=500_000)
    parser.add_argument("--max-val", type=int, default=100_000)
    parser.add_argument("--max-test", type=int, default=100_000)
    parser.add_argument("--sample-size", type=int, default=2_000_000, help="Reservoir sampling size for time cutoffs")
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    ensure_dirs()

    features_path = args.features
    if not Path(features_path).exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    # Define features (detected from header to be robust to optional columns)
    header = pd.read_csv(features_path, nrows=0).columns.tolist()
    candidate_features = [
        "lag_1h", "lag_24h", "rollmean_24h",
        "hour", "day_of_week", "month",
        "apparent_temperature_norm", "temp_lag_1h", "temp_lag_24h",
        "precipitation",
        "is_day", "is_weekend", "is_holiday",
        "region_id"
    ]
    feature_cols = [c for c in candidate_features if c in header]
    target_col = "y_next"
    keep_extra = ["building_name"]  # for per-building metrics; also keep timestamp for preds sample
    keep_extra.append("full_timestamp")

    # Estimate time cutoffs
    t_train_end, t_val_end = reservoir_time_cutoffs(
        features_path, train_frac=0.8, val_frac=0.1, sample_size=args.sample_size, chunksize=args.chunksize
    )

    # Collect split samples
    max_rows = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    splits = collect_split_samples(
        features_path, t_train_end, t_val_end, max_rows, feature_cols, target_col, keep_cols_extra=keep_extra,
        chunksize=args.chunksize
    )

    # Prepare datasets and types
    for split_name, df in splits.items():
        # Downcast floats/ints
        int_candidates = ["hour", "day_of_week", "month", "is_day", "is_weekend", "is_holiday", "region_id"]
        int_present = [c for c in int_candidates if c in df.columns]
        float_present = [c for c in (feature_cols + [target_col]) if c in df.columns]
        splits[split_name] = downcast_types(
            df, cols_float=float_present, cols_int=int_present
        )

    # Baseline 0: Naive (y_hat = lag_1h)
    metrics = {"cutoffs": {"train_end": str(t_train_end), "val_end": str(t_val_end)}, "Naive": {}, "RF": {}, "HGBR": {}}

    for split in ("val", "test"):
        df = splits[split]
        if len(df) == 0:
            metrics["Naive"][split] = {"MAE": None, "RMSE": None}
            continue
        y_true = df[target_col].to_numpy()
        if "lag_1h" in df.columns:
            y_pred_naive = df["lag_1h"].to_numpy()
            metrics["Naive"][split] = evaluate_split(y_true, y_pred_naive)
        else:
            metrics["Naive"][split] = {"MAE": None, "RMSE": None}
        print(f"Naive {split}: {metrics['Naive'][split]}")

    # Train RandomForestRegressor
    feat_X = feature_cols
    df_train = splits["train"]
    X_train = df_train[feat_X].to_numpy(dtype=np.float32)
    y_train = df_train[target_col].to_numpy(dtype=np.float32)

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, n_jobs=-1, random_state=42, min_samples_leaf=5, max_features="sqrt"
    )
    print("Training RandomForestRegressor...")
    rf.fit(X_train, y_train)
    joblib.dump(rf, "outputs/models/rf.joblib")
    print("Saved outputs/models/rf.joblib")

    # Evaluate RF
    for split in ("val", "test"):
        df = splits[split]
        if len(df) == 0:
            metrics["RF"][split] = {"MAE": None, "RMSE": None}
            continue
        X = df[feat_X].to_numpy(dtype=np.float32)
        y = df[target_col].to_numpy(dtype=np.float32)
        y_pred = rf.predict(X)
        metrics["RF"][split] = evaluate_split(y, y_pred)
        print(f"RF {split}: {metrics['RF'][split]}")

    # Train HistGradientBoostingRegressor
    hgbr = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        l2_regularization=0.0,
        random_state=42
    )
    print("Training HistGradientBoostingRegressor...")
    hgbr.fit(X_train, y_train)
    joblib.dump(hgbr, "outputs/models/hgbr.joblib")
    print("Saved outputs/models/hgbr.joblib")

    # Evaluate HGBR
    for split in ("val", "test"):
        df = splits[split]
        if len(df) == 0:
            metrics["HGBR"][split] = {"MAE": None, "RMSE": None}
            continue
        X = df[feat_X].to_numpy(dtype=np.float32)
        y = df[target_col].to_numpy(dtype=np.float32)
        y_pred = hgbr.predict(X)
        metrics["HGBR"][split] = evaluate_split(y, y_pred)
        print(f"HGBR {split}: {metrics['HGBR'][split]}")

    # Permutation importance on validation (HGBR)
    df_val = splits["val"]
    if len(df_val) > 0:
        Xv = df_val[feat_X].to_numpy(dtype=np.float32)
        yv = df_val[target_col].to_numpy(dtype=np.float32)
        print("Computing permutation importance on validation (HGBR, n_repeats=3, single-threaded)...")
        # Use single-threaded to avoid joblib/loky resource tracker issues on some Python/macOS combos
        pi = permutation_importance(
            hgbr, Xv, yv,
            n_repeats=3,
            random_state=42,
            scoring="neg_mean_absolute_error",
            n_jobs=1
        )
        imp_df = pd.DataFrame({"feature": feat_X, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std})
        imp_df.sort_values("importance_mean", ascending=False, inplace=True)
        imp_df.to_csv("outputs/tables/feature_importance_permutation_val.csv", index=False)
        print("Wrote outputs/tables/feature_importance_permutation_val.csv")

    # Per-building MAE on test for best model (choose best by Val MAE)
    def pick_best_model(val_scores: Dict[str, Dict[str, float]]) -> str:
        # Compare RF vs HGBR by Val MAE
        mae_rf = val_scores.get("RF", {}).get("MAE", math.inf)
        mae_hg = val_scores.get("HGBR", {}).get("MAE", math.inf)
        if mae_hg <= mae_rf:
            return "HGBR"
        return "RF"

    best = pick_best_model({"RF": metrics["RF"].get("val", {}), "HGBR": metrics["HGBR"].get("val", {})})
    print(f"Best model by Val MAE: {best}")

    df_test = splits["test"]
    per_bld = []
    preds_sample = []
    if len(df_test) > 0:
        Xte = df_test[feat_X].to_numpy(dtype=np.float32)
        yte = df_test[target_col].to_numpy(dtype=np.float32)
        if best == "HGBR":
            yhat = hgbr.predict(Xte)
        else:
            yhat = rf.predict(Xte)

        df_tmp = pd.DataFrame({
            "building_name": df_test["building_name"].values,
            "full_timestamp": df_test["full_timestamp"].values,
            "y_true": yte,
            "y_pred": yhat
        })
        # Per-building MAE
        perf = df_tmp.groupby("building_name").apply(lambda g: mean_absolute_error(g["y_true"], g["y_pred"])).reset_index(name="MAE")
        perf.sort_values("MAE", inplace=True)
        perf.to_csv("outputs/tables/per_building_mae_test.csv", index=False)
        print("Wrote outputs/tables/per_building_mae_test.csv")

        # Save a small sample for plotting
        preds_sample_df = df_tmp.sample(n=min(5000, len(df_tmp)), random_state=42)
        preds_sample_df.to_csv("outputs/tables/preds_sample_test.csv", index=False)
        print("Wrote outputs/tables/preds_sample_test.csv")

    # Save metrics
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    with open("outputs/metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Wrote outputs/metrics/metrics.json")

    # Final summary
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
