#!/usr/bin/env python3
"""
Compute temperature elasticity per building and segment vulnerability using the trained best model.

Inputs:
- outputs/metrics/metrics.json  (to get cutoffs and best model)
- outputs/models/hgbr.joblib or outputs/models/rf.joblib
- data/processed/features.csv   (engineered features with y_next)

Outputs:
- outputs/vulnerability/building_elasticity.csv  (building_name, elasticity, segment, n_rows_used)
- outputs/figures/elasticity_distribution.png    (histogram of elasticity)
- outputs/tables/elasticity_summary.json         (high-level stats)

Method:
- Use the test period defined by metrics.json (full_timestamp > val_end).
- For each building, sample up to N rows from the test set (default 200).
- For each sampled row, vary apparent_temperature_norm over a fixed grid [0..1] (11 points),
  holding other features constant, and predict with the trained model.
- For that row, estimate slope dy/dtemp via linear fit of predictions vs temperature grid.
- Building elasticity = median slope across the building's sampled rows.
- Segment by quantiles: High (top 25%), Moderate (middle 50%), Low (bottom 25%).

Note:
- Works with the feature set used in training in scripts/02_train_models.py.
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
    "apparent_temperature_norm", "temp_lag_1h", "temp_lag_24h",
    "precipitation",
    "is_day", "is_weekend", "is_holiday",
    "region_id"
]
TEMP_COL = "apparent_temperature_norm"


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


def compute_building_elasticity(model, df_bld: pd.DataFrame,
                                feature_cols: List[str],
                                temp_col: str,
                                grid: np.ndarray) -> float:
    """
    Compute median slope dy/dtemp for a building using per-row ICE linear slope.
    """
    if df_bld.empty:
        return np.nan

    X_base = df_bld[feature_cols].to_numpy(dtype=np.float32)
    temp_idx = feature_cols.index(temp_col)
    slopes = []

    # For each row, predict across temperature grid and fit a line y ~ temp
    for i in range(X_base.shape[0]):
        row = X_base[i].copy()
        X_grid = np.tile(row, (grid.size, 1))
        X_grid[:, temp_idx] = grid
        # Predict
        yhat = model.predict(X_grid)
        # Robust linear fit slope
        slope = np.polyfit(grid, yhat, deg=1)[0]
        slopes.append(slope)

    if len(slopes) == 0:
        return np.nan
    return float(np.median(slopes))


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

    # Temperature grid in [0,1]
    grid = np.linspace(0.0, 1.0, num=max(3, args.grid_points), dtype=np.float32)

    results = []
    for bld, dfb in samples.items():
        elast = compute_building_elasticity(model, dfb, feature_cols, TEMP_COL, grid)
        results.append({"building_name": bld, "elasticity": elast, "n_rows_used": int(len(dfb))})

    res_df = pd.DataFrame(results)
    res_df.sort_values("elasticity", inplace=True, na_position="last")

    # Segment
    res_df["segment"] = segment_scores(res_df["elasticity"])

    # Save CSV
    out_csv = "outputs/vulnerability/building_elasticity.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

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

    # Summary JSON
    summary = {
        "n_buildings": int(len(res_df)),
        "n_with_scores": int(res_df["elasticity"].notna().sum()),
        "median_elasticity": float(res_df["elasticity"].median(skipna=True)) if len(res_df) else None,
        "q25_elasticity": float(res_df["elasticity"].quantile(0.25)) if len(res_df) else None,
        "q75_elasticity": float(res_df["elasticity"].quantile(0.75)) if len(res_df) else None,
        "segments_count": res_df["segment"].value_counts(dropna=False).to_dict()
    }
    with open("outputs/tables/elasticity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote outputs/tables/elasticity_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
