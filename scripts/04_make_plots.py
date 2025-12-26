#!/usr/bin/env python3
"""
Generate evaluation and EDA plots from produced artifacts.

Inputs:
- outputs/tables/feature_importance_permutation_val.csv
- outputs/tables/preds_sample_test.csv
- data/processed/features.csv

Outputs (in outputs/figures/):
- feature_importance_bar.png
- pred_vs_actual_scatter.png
- pred_vs_actual_timeseries.png
- residual_by_hour.png
- error_heatmap_hour_month.png
- usage_vs_temp_scatter.png
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


FIG_DIR = Path("outputs/figures")
TAB_DIR = Path("outputs/tables")
DATA_FEATURES = Path("data/processed/features.csv")
FI_CSV = TAB_DIR / "feature_importance_permutation_val.csv"
PREDS_SAMPLE_CSV = TAB_DIR / "preds_sample_test.csv"


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def feature_importance_bar():
    if not FI_CSV.exists():
        print(f"Skip feature_importance_bar: {FI_CSV} not found")
        return
    fi = pd.read_csv(FI_CSV)
    fi = fi.sort_values("importance_mean", ascending=True)
    plt.figure(figsize=(7, 4.5))
    plt.barh(fi["feature"], fi["importance_mean"], xerr=fi["importance_std"], color="#3b7ddd", alpha=0.85)
    plt.xlabel("Permutation Importance (val, higher is more important)")
    plt.tight_layout()
    out = FIG_DIR / "feature_importance_bar.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


def preds_based_plots():
    if not PREDS_SAMPLE_CSV.exists():
        print(f"Skip preds_based_plots: {PREDS_SAMPLE_CSV} not found")
        return
    df = pd.read_csv(PREDS_SAMPLE_CSV, parse_dates=["full_timestamp"])
    if df.empty:
        print("Skip preds_based_plots: empty preds sample")
        return

    # Pred vs Actual scatter
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(df["y_true"], df["y_pred"], s=6, alpha=0.25, color="#2ca02c", edgecolors="none")
    lims = [0, 1]
    plt.plot(lims, lims, "--", color="k", linewidth=1)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("True (y_next)")
    plt.ylabel("Predicted")
    plt.title("Predicted vs True (Test sample)")
    plt.tight_layout()
    out1 = FIG_DIR / "pred_vs_actual_scatter.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"Wrote {out1}")

    # Residual by hour-of-day
    tmp = df.copy()
    tmp["hour"] = tmp["full_timestamp"].dt.hour
    tmp["abs_err"] = (tmp["y_true"] - tmp["y_pred"]).abs()
    g = tmp.groupby("hour")["abs_err"].mean().reset_index()
    plt.figure(figsize=(7, 3.5))
    plt.plot(g["hour"], g["abs_err"], marker="o", color="#d62728")
    plt.xticks(range(0, 24, 1))
    plt.xlabel("Hour of day")
    plt.ylabel("Mean Absolute Error")
    plt.title("MAE by Hour (Test sample)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = FIG_DIR / "residual_by_hour.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Wrote {out2}")

    # Error heatmap hour x month
    tmp["month"] = tmp["full_timestamp"].dt.month
    heat = tmp.assign(err=(tmp["y_true"] - tmp["y_pred"]).abs()).groupby(["hour", "month"])["err"].mean().unstack("month")
    plt.figure(figsize=(8, 4))
    sns.heatmap(heat, cmap="mako", cbar_kws={"label": "Mean Abs Error"})
    plt.title("Error heatmap (hour x month) - Test sample")
    plt.xlabel("Month")
    plt.ylabel("Hour")
    plt.tight_layout()
    out3 = FIG_DIR / "error_heatmap_hour_month.png"
    plt.savefig(out3, dpi=150)
    plt.close()
    print(f"Wrote {out3}")

    # Predicted vs Actual time series for 3 buildings
    # Pick top-3 buildings by count in sample
    top_blds = df["building_name"].value_counts().head(3).index.tolist()
    plt.figure(figsize=(10, 6))
    for i, b in enumerate(top_blds, 1):
        sub = df[df["building_name"] == b].sort_values("full_timestamp").copy()
        # Downsample long series for visibility
        if len(sub) > 1000:
            sub = sub.iloc[:: max(1, len(sub)//1000)]
        ax = plt.subplot(len(top_blds), 1, i)
        ax.plot(sub["full_timestamp"], sub["y_true"], label="True", color="#1f77b4", linewidth=1.0)
        ax.plot(sub["full_timestamp"], sub["y_pred"], label="Pred", color="#ff7f0e", linewidth=1.0, alpha=0.9)
        ax.set_title(f"Building: {b}")
        if i == len(top_blds):
            ax.set_xlabel("Timestamp")
        ax.set_ylabel("Usage (norm)")
        if i == 1:
            ax.legend()
    plt.tight_layout()
    out4 = FIG_DIR / "pred_vs_actual_timeseries.png"
    plt.savefig(out4, dpi=150)
    plt.close()
    print(f"Wrote {out4}")


def usage_vs_temp_scatter(max_points: int = 100_000, chunksize: int = 500_000):
    """
    Build a usage vs temperature scatter from features.csv without loading full file.
    """
    if not DATA_FEATURES.exists():
        print(f"Skip usage_vs_temp_scatter: {DATA_FEATURES} not found")
        return

    used = 0
    parts = []
    usecols = ["usage_pb", "apparent_temperature_norm"]
    for chunk in pd.read_csv(DATA_FEATURES, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk = chunk.dropna(subset=usecols)
        if chunk.empty:
            continue
        need = max_points - used
        if need <= 0:
            break
        take = min(need, len(chunk))
        parts.append(chunk.sample(n=take, random_state=42))
        used += take
        if used >= max_points:
            break

    if not parts:
        print("Skip usage_vs_temp_scatter: no data assembled")
        return

    df = pd.concat(parts, ignore_index=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(df["apparent_temperature_norm"], df["usage_pb"], s=4, alpha=0.15, color="#9467bd", edgecolors="none")
    plt.xlabel("Apparent temperature (norm)")
    plt.ylabel("Usage (per-building norm)")
    plt.title("Usage vs Temperature (random subsample)")
    plt.tight_layout()
    out = FIG_DIR / "usage_vs_temp_scatter.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


def main():
    ensure_dirs()
    feature_importance_bar()
    preds_based_plots()
    usage_vs_temp_scatter()


if __name__ == "__main__":
    main()
