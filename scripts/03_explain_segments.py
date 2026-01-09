#!/usr/bin/env python3
"""
Inputs:
- outputs/metrics/metrics.json
- outputs/models/hgbr.joblib or outputs/models/rf.joblib
- data/processed/features.csv

Outputs:
- outputs/figures/elasticity_distribution.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CANDIDATE_FEATURES = [
    "lag_1h", "lag_24h", "rollmean_24h",
    "hour", "day_of_week", "month",
    "temp_z", "temp_z_lag_1h", "temp_z_lag_24h",
    "precipitation",
    "is_day", "is_weekend", "is_holiday",
    "region_id"
]
TEMP_COL = "temp_z"


def ensure_dirs():
    Path("outputs/vulnerability").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)


def load_best_model_and_cutoffs(metrics_path: str) -> Dict:
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    # Pick best by Val MAE
    best = "HGBR"
    try:
        mae_rf = metrics.get("RF", {}).get("val", {}).get("MAE", float("inf"))
        mae_hg = metrics.get("HGBR", {}).get("val", {}).get("MAE", float("inf"))
        best = "HGBR" if (mae_hg is not None and mae_hg <= (mae_rf if mae_rf is not None else float("inf"))) else "RF"
    except Exception:
        best = "HGBR"
    t_val_end = pd.to_datetime(metrics["cutoffs"]["val_end"])
    model_path = f"outputs/models/{'hgbr' if best == 'HGBR' else 'rf'}.joblib"
    model = joblib.load(model_path)
    return {"best": best, "val_end": t_val_end, "model": model}


def collect_test_samples(features_csv: str,
                         t_val_end: pd.Timestamp,
                         feature_cols: List[str],
                         max_per_building: int = 200,
                         chunksize: int = 500_000,
                         rng_seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Stream features.csv and collect up to max_per_building rows per building from the test period.
    """
    rng = np.random.default_rng(rng_seed)
    keep_cols = ["building_name", "full_timestamp"] + feature_cols
    samples: Dict[str, List[pd.DataFrame]] = {}
    counts: Dict[str, int] = {}

    for chunk in pd.read_csv(features_csv, usecols=keep_cols, parse_dates=["full_timestamp"],
                             chunksize=chunksize, low_memory=False):
        # Test split
        chunk = chunk[chunk["full_timestamp"] > t_val_end]
        if chunk.empty:
            continue
        # Drop rows with missing key features
        chunk = chunk.dropna(subset=feature_cols)
        if chunk.empty:
            continue

        # Group by building and sample remainder needed
        for bld, g in chunk.groupby("building_name"):
            need = max_per_building - counts.get(bld, 0)
            if need <= 0:
                continue
            if len(g) > need:
                g = g.sample(n=need, random_state=42)
            samples.setdefault(bld, []).append(g)
            counts[bld] = counts.get(bld, 0) + len(g)

    # Concatenate per building
    out: Dict[str, pd.DataFrame] = {}
    for bld, parts in samples.items():
        dfb = pd.concat(parts, ignore_index=True)
        out[bld] = dfb
    return out


def compute_building_elasticity(model,
                                df_bld: pd.DataFrame,
                                feature_cols: List[str],
                                temp_col: str,
                                grid: np.ndarray) -> Dict[str, float]:
    """
    Computing elasticity metrics per building
    """
    if df_bld.empty or temp_col not in df_bld.columns:
        return {"elasticity": np.nan, "elasticity_cool": np.nan, "elasticity_heat": np.nan}

    X_base = df_bld[feature_cols].dropna().to_numpy(dtype=np.float32)
    if X_base.size == 0:
        return {"elasticity": np.nan, "elasticity_cool": np.nan, "elasticity_heat": np.nan}

    temp_idx = feature_cols.index(temp_col)
    # Split grid around the building's typical temperature (median)
    med_t = float(df_bld[temp_col].median(skipna=True)) if df_bld[temp_col].notna().any() else 0.0
    grid_cool = grid[grid <= med_t]
    grid_heat = grid[grid >= med_t]
    if grid_cool.size < 2:
        grid_cool = grid[: max(2, grid.size // 2)]
    if grid_heat.size < 2:
        grid_heat = grid[-max(2, grid.size // 2):]

    slopes_all = []
    slopes_cool = []
    slopes_heat = []

    for i in range(X_base.shape[0]):
        row = X_base[i].copy()

        # Predict across the full grid
        Xg = np.tile(row, (grid.size, 1))
        Xg[:, temp_idx] = grid
        yhat = model.predict(Xg)
        slopes_all.append(np.polyfit(grid, yhat, deg=1)[0])

        # Cool-side slope
        Xc = np.tile(row, (grid_cool.size, 1))
        Xc[:, temp_idx] = grid_cool
        yhat_c = model.predict(Xc)
        slopes_cool.append(np.polyfit(grid_cool, yhat_c, deg=1)[0])

        # Heat-side slope
        Xh = np.tile(row, (grid_heat.size, 1))
        Xh[:, temp_idx] = grid_heat
        yhat_h = model.predict(Xh)
        slopes_heat.append(np.polyfit(grid_heat, yhat_h, deg=1)[0])

    return {
        "elasticity": float(np.median(slopes_all)) if slopes_all else np.nan,
        "elasticity_cool": float(np.median(slopes_cool)) if slopes_cool else np.nan,
        "elasticity_heat": float(np.median(slopes_heat)) if slopes_heat else np.nan,
    }


def segment_scores(scores: pd.Series) -> pd.Series:
    """
    Segment into High (top 25%), Moderate (middle 50%), Low (bottom 25%).
    """
    q25 = scores.quantile(0.25)
    q75 = scores.quantile(0.75)
    def label(v):
        if pd.isna(v):
            return "Unknown"
        if v >= q75:
            return "High"
        if v <= q25:
            return "Low"
        return "Moderate"
    return scores.apply(label)


def main():
    parser = argparse.ArgumentParser(description="Compute temperature elasticity and building segments from trained model.")
    parser.add_argument("--features", type=str, default="data/processed/features.csv")
    parser.add_argument("--metrics", type=str, default="outputs/metrics/metrics.json")
    parser.add_argument("--max-per-building", type=int, default=200)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--grid-points", type=int, default=11)
    args = parser.parse_args()

    ensure_dirs()

    info = load_best_model_and_cutoffs(args.metrics)
    model = info["model"]
    t_val_end = info["val_end"]
    best_name = "HGBR" if "HistGradientBoosting" in model.__class__.__name__ else "RF"
    print(f"Using model: {best_name}; test cutoff full_timestamp > {t_val_end}")

    # Determine feature order consistent with training
    header = pd.read_csv(args.features, nrows=0).columns.tolist()
    feature_cols = [c for c in CANDIDATE_FEATURES if c in header]

    samples = collect_test_samples(
        args.features, t_val_end, feature_cols,
        max_per_building=args.max_per_building,
        chunksize=args.chunksize
    )
    n_bld = len(samples)
    print(f"Collected test samples for {n_bld} buildings")

    # Temperature grid based on observed temp_z in test samples (p5..p95)
    all_t = []
    for dfb in samples.values():
        if TEMP_COL in dfb.columns:
            all_t.append(dfb[TEMP_COL].dropna().to_numpy())
    if all_t:
        arr = np.concatenate(all_t)
        lo, hi = np.nanpercentile(arr, [5, 95])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = -3.0, 3.0
    else:
        lo, hi = -3.0, 3.0
    grid = np.linspace(lo, hi, num=max(3, args.grid_points), dtype=np.float32)

    results = []
    for bld, dfb in samples.items():
        metrics = compute_building_elasticity(model, dfb, feature_cols, TEMP_COL, grid)
        results.append({
            "building_name": bld,
            "elasticity": metrics.get("elasticity"),
            "elasticity_cool": metrics.get("elasticity_cool"),
            "elasticity_heat": metrics.get("elasticity_heat"),
            "n_rows_used": int(len(dfb))
        })

    res_df = pd.DataFrame(results)
    res_df.sort_values("elasticity", inplace=True, na_position="last")


    # Plot distribution
    fig_path = "outputs/figures/elasticity_distribution.png"
    plt.figure(figsize=(8, 5))
    vals = res_df["elasticity"].dropna().to_numpy()
    plt.hist(vals, bins=40, color="#3b7ddd", edgecolor="white")
    plt.axvline(np.median(vals) if vals.size else 0.0, color="black", linestyle="--", label="Median")
    plt.title("Building Temperature Elasticity Distribution (dy/dtemp)")
    plt.xlabel("Elasticity (slope of prediction vs temperature)")
    plt.ylabel("Count of buildings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
