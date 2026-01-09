#!/usr/bin/env python3
"""
Inputs:
- data/processed/features.csv

Outputs:
- outputs/models/rf.joblib, outputs/models/hgbr.joblib
- outputs/metrics/metrics.json (MAE/RMSE for Naive, RF, HGBR on Val/Test)
- outputs/tables/feature_importance_permutation_val.csv
- outputs/tables/per_building_mae_test.csv
- outputs/tables/preds_sample_test.csv (small sample for plotting)
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


def chronological_time_cutoffs(path: str,
                               train_frac: float = 0.8,
                               val_frac: float = 0.1,
                               chunksize: int = 1_000_000) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Computes exact chronological time cutoffs.
    """
    print(f"Computing exact chronological cutoffs from {path}...")

    # Collect all timestamps (sorted)
    all_ts = []
    usecols = ["full_timestamp"]
    for chunk in pd.read_csv(path, usecols=usecols, parse_dates=["full_timestamp"],
                             chunksize=chunksize, low_memory=False):
        ts = chunk["full_timestamp"].dropna()
        all_ts.extend(ts.tolist())

    if not all_ts:
        raise RuntimeError("No timestamps found to compute cutoffs.")

    all_ts.sort()
    n = len(all_ts)
    train_end_idx = int(train_frac * n)
    val_end_idx = int((train_frac + val_frac) * n)

    t_train_end = all_ts[train_end_idx]
    t_val_end = all_ts[val_end_idx]

    print(f"Exact cutoffs: train_end={t_train_end}, val_end={t_val_end} (n={n:,})")
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
        # Dropping rows with missing target or key features
        chunk = chunk.dropna(subset=[target_col, "lag_1h", "lag_24h", "rollmean_24h"])
        # Assigning split
        ts = chunk["full_timestamp"]
        mask_train = ts <= t_train_end
        mask_val = (ts > t_train_end) & (ts <= t_val_end)
        mask_test = ts > t_val_end

        for split, mask in (("train", mask_train), ("val", mask_val), ("test", mask_test)):
            sub = chunk.loc[mask, :]
            if sub.empty:
                continue
            cap = max_rows.get(split, 0)
            if cap and cap > 0:
                need = cap - got[split]
                if need <= 0:
                    continue
                if len(sub) > need:
                    # random sample to fill the remainder
                    sub = sub.sample(n=need, random_state=42)
            # For cap <= 0 we treat it as unlimited: append all rows
            out[split].append(sub)
            got[split] += len(sub)

        # stop early only when all capped splits are filled
        capped = [s for s in got if max_rows.get(s, 0) and max_rows.get(s, 0) > 0]
        if capped and all(got[s] >= max_rows.get(s, 0) for s in capped):
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
        "lag_1h", "lag_24h", "lag_168h", "rollmean_24h",
        "hour", "day_of_week", "month",
        "temp_z", "temp_z_lag_1h", "temp_z_lag_24h",  # Prioritize temp_z
        "apparent_temperature_norm", "temp_lag_1h", "temp_lag_24h",
        "precipitation",
        "is_day", "is_weekend", "is_holiday",
        "region_id"
    ]
    feature_cols = [c for c in candidate_features if c in header]
    target_col = "y_next"
    keep_extra = ["building_name"]
    keep_extra.append("full_timestamp")

    # Estimate time cutoffs
    t_train_end, t_val_end = chronological_time_cutoffs(
        features_path, train_frac=0.8, val_frac=0.1, chunksize=args.chunksize
    )

    # Collect split samples
    max_rows = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    splits = collect_split_samples(
        features_path, t_train_end, t_val_end, max_rows, feature_cols, target_col, keep_cols_extra=keep_extra,
        chunksize=args.chunksize
    )

    for split_name, df in splits.items():
        # Downcast floats/ints
        int_candidates = ["hour", "day_of_week", "month", "is_day", "is_weekend", "is_holiday", "region_id"]
        int_present = [c for c in int_candidates if c in df.columns]
        float_present = [c for c in (feature_cols + [target_col]) if c in df.columns]
        splits[split_name] = downcast_types(
            df, cols_float=float_present, cols_int=int_present
        )

    # Baselines: Naive (lag_1h), Seasonal-24 (lag_24h), Weekly-168 (lag_168h)
    metrics = {"cutoffs": {"train_end": str(t_train_end), "val_end": str(t_val_end)}, "Naive": {}, "Seasonal-24": {}, "Weekly-168": {}, "RF": {}, "HGBR": {}}

    for split in ("val", "test"):
        df = splits[split]
        if len(df) == 0:
            for baseline in ["Naive", "Seasonal-24", "Weekly-168"]:
                metrics[baseline][split] = {"MAE": None, "RMSE": None}
            continue
        y_true = df[target_col].to_numpy()

        # Naive (lag_1h)
        if "lag_1h" in df.columns:
            y_pred_naive = df["lag_1h"].to_numpy()
            metrics["Naive"][split] = evaluate_split(y_true, y_pred_naive)
        else:
            metrics["Naive"][split] = {"MAE": None, "RMSE": None}
        print(f"Naive {split}: {metrics['Naive'][split]}")

        # Seasonal-24 (lag_24h)
        if "lag_24h" in df.columns:
            y_pred_seasonal = df["lag_24h"].to_numpy()
            metrics["Seasonal-24"][split] = evaluate_split(y_true, y_pred_seasonal)
        else:
            metrics["Seasonal-24"][split] = {"MAE": None, "RMSE": None}
        print(f"Seasonal-24 {split}: {metrics['Seasonal-24'][split]}")

        # Weekly-168 (lag_168h)
        if "lag_168h" in df.columns:
            y_pred_weekly = df["lag_168h"].to_numpy()
            metrics["Weekly-168"][split] = evaluate_split(y_true, y_pred_weekly)
        else:
            metrics["Weekly-168"][split] = {"MAE": None, "RMSE": None}
        print(f"Weekly-168 {split}: {metrics['Weekly-168'][split]}")

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

    # Compute skill scores
    def compute_skill_scores(metrics_dict):
        for split in ("val", "test"):
            naive_mae = metrics_dict.get("Naive", {}).get(split, {}).get("MAE")
            if naive_mae and naive_mae > 0:
                for model in ["RF", "HGBR"]:
                    model_mae = metrics_dict.get(model, {}).get(split, {}).get("MAE")
                    if model_mae is not None:
                        skill = 1.0 - (model_mae / naive_mae)
                        metrics_dict[model][split]["Skill"] = float(skill)

    compute_skill_scores(metrics)

    # Permutation importance on validation for HGBR
    df_val = splits["val"]
    if len(df_val) > 0:
        Xv = df_val[feat_X].to_numpy(dtype=np.float32)
        yv = df_val[target_col].to_numpy(dtype=np.float32)
        print("Computing permutation importance on validation (HGBR, n_repeats=3, single-threaded)...")

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

    # Permutation importance on test
    df_test = splits["test"]
    if len(df_test) > 0:
        Xte = df_test[feat_X].to_numpy(dtype=np.float32)
        yte = df_test[target_col].to_numpy(dtype=np.float32)
        print("Computing permutation importance on test (HGBR, n_repeats=3, single-threaded)...")
        pi_test = permutation_importance(
            hgbr, Xte, yte,
            n_repeats=3,
            random_state=42,
            scoring="neg_mean_absolute_error",
            n_jobs=1
        )
        imp_test_df = pd.DataFrame({"feature": feat_X, "importance_mean": pi_test.importances_mean, "importance_std": pi_test.importances_std})
        imp_test_df.sort_values("importance_mean", ascending=False, inplace=True)
        imp_test_df.to_csv("outputs/tables/feature_importance_permutation_test.csv", index=False)
        print("Wrote outputs/tables/feature_importance_permutation_test.csv")

    # Per-building MAE on test
    def pick_best_model(val_scores: Dict[str, Dict[str, float]]) -> str:
        # Comparing RF vs HGBR by Val MAE
        mae_rf = val_scores.get("RF", {}).get("MAE", math.inf)
        mae_hg = val_scores.get("HGBR", {}).get("MAE", math.inf)
        if mae_hg <= mae_rf:
            return "HGBR"
        return "RF"

    best = pick_best_model({"RF": metrics["RF"].get("val", {}), "HGBR": metrics["HGBR"].get("val", {})})
    print(f"Best model by Val MAE: {best}")

    df_test = splits["test"]
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
