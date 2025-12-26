#!/usr/bin/env python3
"""
Produce an expanded suite of diagnostics and EDA plots (>30 figures) for the energy forecasting project.

Inputs:
- data/processed/features.csv (large; read in chunks with sampling)
- outputs/tables/preds_sample_test.csv
- outputs/tables/per_building_mae_test.csv
- outputs/tables/feature_importance_permutation_val.csv
- outputs/vulnerability/building_elasticity.csv
- outputs/metrics/metrics.json
- outputs/models/{hgbr.joblib, rf.joblib} (optional for PDP/ICE)

Outputs (examples, not exhaustive; all saved to outputs/figures/):
- missingness_bar.png
- missingness_heatmap.png
- usage_hist_by_region.png
- usage_violin_by_season.png
- usage_by_hour_box.png
- usage_weekend_weekday_violin.png
- usage_holiday_violin.png
- usage_heatmap_hour_dow.png
- usage_heatmap_hour_month.png
- feature_correlation_heatmap.png
- residual_vs_temp_bin.png
- residual_vs_precip_bin.png
- calibration_plot.png
- pdp_temperature.png
- ice_temperature_examples.png
- pdp_hour.png
- pdp_precip.png
- elasticity_by_region_box.png
- per_building_mae_hist.png
- per_building_mae_by_region_box.png
- error_by_season_box.png
- error_by_temp_bin_region.png
- error_over_time.png
- worst_buildings_timeseries.png
- pca_features_scatter.png
- cluster_centers_hourly.png
- pairplot_sample.png

Note:
- Uses sampling to keep memory/time reasonable.
- Figures are robust to missing optional columns (skip plot if required columns absent).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


FIG_DIR = Path("outputs/figures")
TAB_DIR = Path("outputs/tables")
DATA_FEATURES = Path("data/processed/features.csv")
PREDS_SAMPLE_CSV = TAB_DIR / "preds_sample_test.csv"
PER_BLD_CSV = TAB_DIR / "per_building_mae_test.csv"
ELAST_CSV = Path("outputs/vulnerability/building_elasticity.csv")
METRICS_JSON = Path("outputs/metrics/metrics.json")


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics():
    val_end = None
    try:
        m = json.loads(METRICS_JSON.read_text())
        v = m.get("cutoffs", {}).get("val_end", None)
        if v:
            val_end = pd.to_datetime(v)
    except Exception:
        pass
    return val_end


def load_best_model():
    # Prefer HGBR then RF if both exist (as per metrics best)
    try:
        m = json.loads(METRICS_JSON.read_text())
        mae_rf = m.get("RF", {}).get("val", {}).get("MAE", None)
        mae_hg = m.get("HGBR", {}).get("val", {}).get("MAE", None)
        if mae_hg is not None and (mae_rf is None or mae_hg <= mae_rf):
            path = Path("outputs/models/hgbr.joblib")
        else:
            path = Path("outputs/models/rf.joblib")
        if not path.exists():
            # fallback
            if Path("outputs/models/hgbr.joblib").exists():
                path = Path("outputs/models/hgbr.joblib")
            else:
                path = Path("outputs/models/rf.joblib")
        if path.exists():
            return joblib.load(path)
    except Exception:
        pass
    return None


def header_features() -> List[str]:
    try:
        cols = pd.read_csv(DATA_FEATURES, nrows=0).columns.tolist()
        return cols
    except Exception:
        return []


def sample_features(n_rows: int = 200_000, chunksize: int = 500_000, require_cols: List[str] | None = None) -> pd.DataFrame:
    """
    Return a sampled DataFrame of features with the requested columns if present.
    """
    require_cols = require_cols or []
    header = header_features()
    needed = [c for c in require_cols if c in header]
    # Always include common IDs + a few defaults if present
    base_cols = ["building_name", "full_timestamp", "usage_pb", "hour", "day_of_week", "month", "season", "region", "region_id",
                 "apparent_temperature_norm", "precipitation", "is_day", "is_weekend", "is_holiday"]
    usecols = sorted(set([c for c in base_cols if c in header] + needed))
    used = 0
    parts = []
    if not usecols:
        return pd.DataFrame()
    for chunk in pd.read_csv(DATA_FEATURES, usecols=usecols, parse_dates=["full_timestamp"] if "full_timestamp" in usecols else None,
                             chunksize=chunksize, low_memory=False):
        # drop na heavy rows only if needed
        # sample
        take = min(n_rows - used, len(chunk))
        if take <= 0:
            break
        parts.append(chunk.sample(n=take, random_state=42) if len(chunk) > take else chunk)
        used += take
    if not parts:
        return pd.DataFrame(columns=usecols)
    df = pd.concat(parts, ignore_index=True)
    return df


def build_mapping_building_region(max_rows: int = 2_000_000, chunksize: int = 500_000) -> pd.DataFrame:
    header = header_features()
    usecols = [c for c in ["building_name", "region"] if c in header]
    if len(usecols) < 2:
        return pd.DataFrame(columns=["building_name", "region"])
    used = 0
    parts = []
    for chunk in pd.read_csv(DATA_FEATURES, usecols=usecols, chunksize=chunksize):
        parts.append(chunk.dropna(subset=["building_name", "region"]).drop_duplicates(subset=["building_name", "region"]))
        used += len(chunk)
        if used >= max_rows:
            break
    if not parts:
        return pd.DataFrame(columns=["building_name", "region"])
    df = pd.concat(parts, ignore_index=True)
    # prefer first mapping
    df = df.groupby("building_name", as_index=False).first()
    return df[["building_name", "region"]]


def join_preds_with_features(preds: pd.DataFrame, require_cols: List[str]) -> pd.DataFrame:
    """
    Join preds_sample_test (building_name, full_timestamp, y_true, y_pred)
    with a subset of features to get temp/season for error analyses.
    """
    key_cols = ["building_name", "full_timestamp"]
    header = header_features()
    need = [c for c in require_cols if c in header]
    if not all(k in header for k in key_cols):
        return preds.copy()
    out = []
    idx = preds[key_cols].copy()
    idx["full_timestamp"] = pd.to_datetime(idx["full_timestamp"], errors="coerce")
    idx = idx.dropna()
    # Build dictionary of tuples to filter chunks quickly
    idx["key"] = idx["building_name"].astype(str) + "||" + idx["full_timestamp"].astype(str)
    keyset = set(idx["key"].tolist())
    cols_to_read = sorted(set(key_cols + need))
    for chunk in pd.read_csv(DATA_FEATURES, usecols=cols_to_read, parse_dates=["full_timestamp"], chunksize=300_000):
        chunk["key"] = chunk["building_name"].astype(str) + "||" + chunk["full_timestamp"].astype(str)
        sel = chunk[chunk["key"].isin(keyset)].copy()
        if not sel.empty:
            out.append(sel.drop(columns=["key"]))
    if not out:
        return preds.copy()
    feat = pd.concat(out, ignore_index=True)
    merged = preds.merge(feat, on=key_cols, how="left")
    return merged


def savefig_simple(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Wrote {path}")


def plots_missingness():
    df = sample_features(n_rows=150_000)
    if df.empty:
        return
    fracs = df.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    fracs.plot(kind="bar", color="#6baed6")
    plt.ylabel("Missing fraction")
    plt.title("Missingness by column (sample)")
    savefig_simple(FIG_DIR / "missingness_bar.png")

    # Heatmap of missingness for top 30 columns with variance
    cols = fracs.index.tolist()[:30]
    if not cols:
        return
    miss = df[cols].isna().astype(int)
    plt.figure(figsize=(10, 6))
    sns.heatmap(miss.T, cmap="Reds", cbar_kws={"label": "Missing (1) / Present (0)"})
    plt.xlabel("Samples")
    plt.ylabel("Columns")
    plt.title("Missingness heatmap (sample x columns)")
    savefig_simple(FIG_DIR / "missingness_heatmap.png")


def plots_usage_distributions():
    df = sample_features(n_rows=250_000, require_cols=["usage_pb", "region", "season", "hour"])
    if df.empty or "usage_pb" not in df.columns:
        return
    # usage hist by region
    if "region" in df.columns:
        plt.figure(figsize=(7, 4))
        for r, g in df.groupby("region"):
            sns.kdeplot(g["usage_pb"].dropna(), label=str(r), fill=True, alpha=0.2)
        plt.xlabel("usage_pb")
        plt.title("Usage distribution by region (KDE, sample)")
        plt.legend()
        savefig_simple(FIG_DIR / "usage_hist_by_region.png")

    # violin by season
    if "season" in df.columns:
        plt.figure(figsize=(7, 4))
        sns.violinplot(data=df.dropna(subset=["season"]), x="season", y="usage_pb", inner="quartile", scale="width")
        plt.title("Usage by season (per-building normalized)")
        savefig_simple(FIG_DIR / "usage_violin_by_season.png")

    # by hour of day
    if "hour" in df.columns:
        plt.figure(figsize=(9, 4))
        sns.boxplot(data=df, x="hour", y="usage_pb", showfliers=False)
        plt.title("Usage by hour of day (boxplot)")
        savefig_simple(FIG_DIR / "usage_by_hour_box.png")

    # weekend vs weekday
    if "is_weekend" in df.columns:
        plt.figure(figsize=(6, 4))
        df["weekend_label"] = np.where(df["is_weekend"] == 1, "Weekend", "Weekday")
        sns.violinplot(data=df, x="weekend_label", y="usage_pb", inner="quartile", scale="width")
        plt.title("Usage: Weekend vs Weekday")
        savefig_simple(FIG_DIR / "usage_weekend_weekday_violin.png")

    # holiday vs non-holiday
    if "is_holiday" in df.columns:
        plt.figure(figsize=(6, 4))
        df["holiday_label"] = np.where(df["is_holiday"] == 1, "Holiday", "Non-Holiday")
        sns.violinplot(data=df, x="holiday_label", y="usage_pb", inner="quartile", scale="width")
        plt.title("Usage: Holiday vs Non-Holiday")
        savefig_simple(FIG_DIR / "usage_holiday_violin.png")


def plots_usage_heatmaps():
    df = sample_features(n_rows=300_000, require_cols=["usage_pb", "hour", "day_of_week", "month"])
    if df.empty or "usage_pb" not in df.columns:
        return
    # hour x day_of_week
    piv = df.pivot_table(index="hour", columns="day_of_week", values="usage_pb", aggfunc="mean")
    plt.figure(figsize=(8, 4))
    sns.heatmap(piv, cmap="viridis", cbar_kws={"label": "Mean usage_pb"})
    plt.xlabel("Day of week (0=Mon)")
    plt.ylabel("Hour")
    plt.title("Mean usage heatmap (hour x day_of_week)")
    savefig_simple(FIG_DIR / "usage_heatmap_hour_dow.png")

    # hour x month
    piv2 = df.pivot_table(index="hour", columns="month", values="usage_pb", aggfunc="mean")
    plt.figure(figsize=(10, 4))
    sns.heatmap(piv2, cmap="viridis", cbar_kws={"label": "Mean usage_pb"})
    plt.xlabel("Month")
    plt.ylabel("Hour")
    plt.title("Mean usage heatmap (hour x month)")
    savefig_simple(FIG_DIR / "usage_heatmap_hour_month.png")


def plot_feature_correlation():
    df = sample_features(n_rows=200_000)
    if df.empty:
        return
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return
    corr = num.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0.0)
    plt.title("Feature correlation heatmap (sample)")
    savefig_simple(FIG_DIR / "feature_correlation_heatmap.png")


def plots_residuals_and_calibration():
    if not PREDS_SAMPLE_CSV.exists():
        return
    preds = pd.read_csv(PREDS_SAMPLE_CSV, parse_dates=["full_timestamp"])
    preds["resid"] = preds["y_true"] - preds["y_pred"]
    # Join with temp and precip for binning
    require = ["apparent_temperature_norm", "precipitation", "season", "region", "hour"]
    merged = join_preds_with_features(preds.copy(), require_cols=require)

    # residual vs temperature bins
    if "apparent_temperature_norm" in merged.columns:
        merged["temp_bin"] = pd.cut(merged["apparent_temperature_norm"], bins=np.linspace(0, 1, 11), include_lowest=True)
        g = merged.groupby("temp_bin")["resid"].apply(lambda s: s.abs().mean()).reset_index(name="MAE")
        plt.figure(figsize=(8, 3.5))
        plt.plot(range(len(g)), g["MAE"], marker="o", color="#d62728")
        plt.xticks(range(len(g)), [str(b) for b in g["temp_bin"]], rotation=45, ha="right")
        plt.ylabel("Mean Abs Error")
        plt.title("Residual vs Temperature (binned)")
        savefig_simple(FIG_DIR / "residual_vs_temp_bin.png")

    # residual vs precipitation bins
    if "precipitation" in merged.columns:
        merged["precip_bin"] = pd.qcut(merged["precipitation"].fillna(0), q=10, duplicates="drop")
        g = merged.groupby("precip_bin")["resid"].apply(lambda s: s.abs().mean()).reset_index(name="MAE")
        plt.figure(figsize=(8, 3.5))
        plt.plot(range(len(g)), g["MAE"], marker="s", color="#1f77b4")
        plt.xticks(range(len(g)), [str(b) for b in g["precip_bin"]], rotation=45, ha="right")
        plt.ylabel("Mean Abs Error")
        plt.title("Residual vs Precipitation (binned)")
        savefig_simple(FIG_DIR / "residual_vs_precip_bin.png")

    # calibration (pred deciles vs true mean)
    q = np.quantile(preds["y_pred"], np.linspace(0, 1, 11))
    preds["pred_bin"] = pd.cut(preds["y_pred"], bins=q, include_lowest=True, duplicates="drop")
    cal = preds.groupby("pred_bin").agg(pred_mean=("y_pred", "mean"), true_mean=("y_true", "mean")).reset_index(drop=True)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="k")
    plt.scatter(cal["pred_mean"], cal["true_mean"], s=30, color="#2ca02c")
    plt.xlabel("Predicted (bin mean)")
    plt.ylabel("True (bin mean)")
    plt.title("Calibration plot (test sample)")
    savefig_simple(FIG_DIR / "calibration_plot.png")

    # error by season
    if "season" in merged.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=merged, x="season", y=np.abs(merged["resid"]), showfliers=False)
        plt.ylabel("Absolute Error")
        plt.title("Error by season (test sample)")
        savefig_simple(FIG_DIR / "error_by_season_box.png")

    # error by temp bin and region (facet)
    if "temp_bin" in merged.columns and "region" in merged.columns:
        g = merged.groupby(["region", "temp_bin"])["resid"].apply(lambda s: s.abs().mean()).reset_index(name="MAE")
        plt.figure(figsize=(10, 4))
        for i, (reg, sub) in enumerate(g.groupby("region"), 1):
            ax = plt.subplot(1, len(g["region"].unique()), i)
            ax.plot(range(len(sub)), sub["MAE"], marker="o")
            ax.set_title(str(reg))
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels([str(x) for x in sub["temp_bin"]], rotation=45, ha="right")
            if i == 1:
                ax.set_ylabel("MAE")
        plt.suptitle("Error by temperature bin and region")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "error_by_temp_bin_region.png", dpi=150)
        plt.close()
        print(f"Wrote {FIG_DIR / 'error_by_temp_bin_region.png'}")

    # error trend over time (weekly)
    ts = preds.copy()
    ts["week"] = ts["full_timestamp"].dt.to_period("W").apply(lambda r: r.start_time)
    g = ts.groupby("week")["resid"].apply(lambda s: s.abs().mean()).reset_index(name="MAE")
    plt.figure(figsize=(9, 3))
    plt.plot(g["week"], g["MAE"], color="#9467bd")
    plt.xlabel("Week")
    plt.ylabel("MAE")
    plt.title("Error over time (weekly, test sample)")
    savefig_simple(FIG_DIR / "error_over_time.png")


def plots_pdp_ice():
    model = load_best_model()
    if model is None:
        return
    header = header_features()
    # Features used in training (subset)
    candidates = [
        "lag_1h", "lag_24h", "rollmean_24h",
        "hour", "day_of_week", "month",
        "apparent_temperature_norm", "temp_lag_1h", "temp_lag_24h",
        "precipitation",
        "is_day", "is_weekend", "is_holiday",
        "region_id"
    ]
    feat_cols = [c for c in candidates if c in header]
    req_cols = ["full_timestamp", "building_name"] + feat_cols
    # Take a small sample from test period for ICE
    val_end = load_metrics()
    parts = []
    got = 0
    for chunk in pd.read_csv(DATA_FEATURES, usecols=[c for c in req_cols if c in header], parse_dates=["full_timestamp"], chunksize=400_000):
        if val_end is not None and "full_timestamp" in chunk.columns:
            chunk = chunk[chunk["full_timestamp"] > val_end]
        chunk = chunk.dropna(subset=feat_cols)
        if chunk.empty:
            continue
        take = min(2000 - got, len(chunk))
        if take <= 0:
            break
        parts.append(chunk.sample(n=take, random_state=42) if len(chunk) > take else chunk)
        got += take
    if not parts:
        return
    df = pd.concat(parts, ignore_index=True)

    # PDP: temperature
    if "apparent_temperature_norm" in feat_cols:
        base = df[feat_cols].median(axis=0).to_numpy(dtype=np.float32)
        grid = np.linspace(0.0, 1.0, 41, dtype=np.float32)
        X = np.tile(base, (grid.size, 1))
        temp_idx = feat_cols.index("apparent_temperature_norm")
        X[:, temp_idx] = grid
        y = model.predict(X)
        plt.figure(figsize=(6, 4))
        plt.plot(grid, y, color="#d62728")
        plt.xlabel("apparent_temperature_norm")
        plt.ylabel("Predicted usage (norm)")
        plt.title("PDP: Temperature")
        savefig_simple(FIG_DIR / "pdp_temperature.png")

        # ICE: pick 10 rows
        ice_rows = df[feat_cols].sample(n=min(10, len(df)), random_state=42).to_numpy(dtype=np.float32)
        plt.figure(figsize=(6, 4))
        for row in ice_rows:
            Xg = np.tile(row, (grid.size, 1))
            Xg[:, temp_idx] = grid
            yhat = model.predict(Xg)
            plt.plot(grid, yhat, alpha=0.4)
        plt.xlabel("apparent_temperature_norm")
        plt.ylabel("Predicted usage (norm)")
        plt.title("ICE: Temperature (10 examples)")
        savefig_simple(FIG_DIR / "ice_temperature_examples.png")

    # PDP: hour
    if "hour" in feat_cols:
        base = df[feat_cols].median(axis=0).to_numpy(dtype=np.float32)
        grid = np.arange(0, 24, dtype=np.int32)
        X = np.tile(base, (grid.size, 1))
        hidx = feat_cols.index("hour")
        X[:, hidx] = grid
        y = model.predict(X)
        plt.figure(figsize=(7, 4))
        plt.plot(grid, y, marker="o")
        plt.xlabel("hour")
        plt.ylabel("Predicted usage (norm)")
        plt.title("PDP: Hour of day")
        savefig_simple(FIG_DIR / "pdp_hour.png")

    # PDP: precipitation
    if "precipitation" in feat_cols:
        base = df[feat_cols].median(axis=0).to_numpy(dtype=np.float32)
        # cap precipitation at 99th percentile to avoid outliers
        pmax = float(np.nanpercentile(df["precipitation"].to_numpy(), 99)) if "precipitation" in df.columns else 1.0
        grid = np.linspace(0.0, max(1e-6, pmax), 41, dtype=np.float32)
        X = np.tile(base, (grid.size, 1))
        pidx = feat_cols.index("precipitation")
        X[:, pidx] = grid
        y = model.predict(X)
        plt.figure(figsize=(6, 4))
        plt.plot(grid, y)
        plt.xlabel("precipitation")
        plt.ylabel("Predicted usage (norm)")
        plt.title("PDP: Precipitation")
        savefig_simple(FIG_DIR / "pdp_precip.png")


def plots_elasticity_and_building_performance():
    # Elasticity by region (needs mapping)
    if ELAST_CSV.exists():
        elast = pd.read_csv(ELAST_CSV)
        map_df = build_mapping_building_region()
        if not elast.empty and not map_df.empty:
            e = elast.merge(map_df, on="building_name", how="left")
            if "region" in e.columns:
                plt.figure(figsize=(6, 4))
                sns.boxplot(data=e, x="region", y="elasticity", showfliers=False)
                plt.title("Elasticity by region")
                savefig_simple(FIG_DIR / "elasticity_by_region_box.png")

    # Per-building MAE distribution and by region
    if PER_BLD_CSV.exists():
        perf = pd.read_csv(PER_BLD_CSV)
        if "MAE" in perf.columns:
            plt.figure(figsize=(6, 4))
            plt.hist(perf["MAE"].dropna(), bins=50, color="#3182bd", edgecolor="white")
            plt.title("Per-building MAE (test) distribution")
            plt.xlabel("MAE")
            savefig_simple(FIG_DIR / "per_building_mae_hist.png")

            map_df = build_mapping_building_region()
            if not map_df.empty and "building_name" in perf.columns:
                m = perf.merge(map_df, on="building_name", how="left")
                if "region" in m.columns:
                    plt.figure(figsize=(6, 4))
                    sns.boxplot(data=m, x="region", y="MAE", showfliers=False)
                    plt.title("Per-building MAE by region (test)")
                    savefig_simple(FIG_DIR / "per_building_mae_by_region_box.png")

    # Worst buildings time series sample
    if PER_BLD_CSV.exists() and PREDS_SAMPLE_CSV.exists():
        perf = pd.read_csv(PER_BLD_CSV)
        preds = pd.read_csv(PREDS_SAMPLE_CSV, parse_dates=["full_timestamp"])
        if "building_name" in perf.columns and "MAE" in perf.columns:
            worst = perf.sort_values("MAE", ascending=False).head(3)["building_name"].tolist()
            plt.figure(figsize=(10, 6))
            for i, b in enumerate(worst, 1):
                sub = preds[preds["building_name"] == b].sort_values("full_timestamp")
                ax = plt.subplot(len(worst), 1, i)
                ax.plot(sub["full_timestamp"], sub["y_true"], label="True", color="#1f77b4")
                ax.plot(sub["full_timestamp"], sub["y_pred"], label="Pred", color="#ff7f0e", alpha=0.9)
                ax.set_title(f"Worst building: {b}")
                if i == 1:
                    ax.legend()
            plt.tight_layout()
            savefig_simple(FIG_DIR / "worst_buildings_timeseries.png")


def plots_pca_and_clusters():
    # PCA scatter
    df = sample_features(n_rows=80_000, require_cols=["usage_pb", "hour", "day_of_week", "month", "apparent_temperature_norm", "precipitation", "region_id"])
    if df.empty:
        return
    feats = df.select_dtypes(include=[np.number]).copy()
    feats = feats.dropna(axis=0)
    if feats.shape[0] < 100 or feats.shape[1] < 3:
        return
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(feats.to_numpy(dtype=np.float32))
    plt.figure(figsize=(6, 5))
    if "region_id" in feats.columns:
        ridx = feats["region_id"].to_numpy()
        plt.scatter(Z[:, 0], Z[:, 1], c=ridx, s=3, cmap="tab10", alpha=0.3)
    else:
        plt.scatter(Z[:, 0], Z[:, 1], s=3, alpha=0.3)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of numeric features (sample)")
    savefig_simple(FIG_DIR / "pca_features_scatter.png")

    # Cluster centers of hourly usage profiles
    df2 = sample_features(n_rows=300_000, require_cols=["usage_pb", "hour", "building_name"])
    if df2.empty or not {"usage_pb", "hour", "building_name"}.issubset(df2.columns):
        return
    prof = df2.groupby(["building_name", "hour"])["usage_pb"].mean().reset_index()
    pivot = prof.pivot(index="building_name", columns="hour", values="usage_pb").fillna(0.0)
    # Simple KMeans on 24-dim vectors
    try:
        km = KMeans(n_clusters=4, n_init=10, random_state=42)
        labels = km.fit_predict(pivot.to_numpy(dtype=np.float32))
        centers = km.cluster_centers_
        plt.figure(figsize=(8, 5))
        for k in range(centers.shape[0]):
            plt.plot(range(24), centers[k, :], label=f"Cluster {k}")
        plt.xlabel("Hour")
        plt.ylabel("Mean usage_pb")
        plt.title("Cluster centers of hourly usage profiles (k=4)")
        plt.legend()
        savefig_simple(FIG_DIR / "cluster_centers_hourly.png")
    except Exception:
        pass


def plot_pairplot_sample():
    df = sample_features(n_rows=8_000, require_cols=["usage_pb", "apparent_temperature_norm", "precipitation", "hour", "region"])
    if df.empty:
        return
    sub = df.dropna().copy()
    if sub.shape[0] < 100:
        return
    try:
        g = sns.pairplot(sub[["usage_pb", "apparent_temperature_norm", "precipitation", "hour"]], diag_kind="kde", corner=True)
        g.fig.suptitle("Pairplot (usage, temp, precip, hour) - sample", y=1.02)
        g.savefig(FIG_DIR / "pairplot_sample.png", dpi=150)
        plt.close("all")
        print(f"Wrote {FIG_DIR / 'pairplot_sample.png'}")
    except Exception:
        pass


# -------------------- Additional comparative plots -------------------- #

def plot_model_comparison_bar():
    """Bar chart comparing MAE for Naive, RF, HGBR on val/test."""
    if not METRICS_JSON.exists():
        return
    try:
        m = json.loads(METRICS_JSON.read_text())
        labels = ["Naive", "RF", "HGBR"]
        val = [m.get(k, {}).get("val", {}).get("MAE") for k in labels]
        test = [m.get(k, {}).get("test", {}).get("MAE") for k in labels]
        x = np.arange(len(labels))
        w = 0.35
        plt.figure(figsize=(7, 4))
        plt.bar(x - w/2, val, width=w, label="Val MAE", color="#1f77b4")
        plt.bar(x + w/2, test, width=w, label="Test MAE", color="#ff7f0e")
        plt.xticks(x, labels)
        plt.ylabel("MAE")
        plt.title("Model comparison (MAE)")
        plt.legend()
        savefig_simple(FIG_DIR / "model_comparison_bar.png")
    except Exception:
        pass


def plot_splits_timeline():
    """Histogram of timestamps with train/val/test cutoffs overlaid."""
    val_end = load_metrics()
    if not DATA_FEATURES.exists():
        return
    try:
        # sample timestamps only
        used = 0
        parts = []
        for ch in pd.read_csv(DATA_FEATURES, usecols=["full_timestamp"], parse_dates=["full_timestamp"], chunksize=800_000):
            parts.append(ch.sample(n=min(200_000 - used, len(ch)), random_state=42) if used < 200_000 else ch.iloc[0:0])
            used += min(200_000 - used, len(ch))
            if used >= 200_000:
                break
        if not parts:
            return
        df = pd.concat(parts, ignore_index=True).dropna()
        plt.figure(figsize=(9, 3.5))
        df["full_timestamp"].hist(bins=60, color="#6baed6")
        # draw cutoffs
        try:
            with open(METRICS_JSON, "r") as f:
                m = json.load(f)
            t_train_end = pd.to_datetime(m.get("cutoffs", {}).get("train_end"))
            t_val_end = pd.to_datetime(m.get("cutoffs", {}).get("val_end"))
            for t, lbl, col in [(t_train_end, "Train end", "red"), (t_val_end, "Val end", "black")]:
                if pd.notnull(t):
                    plt.axvline(t, color=col, linestyle="--", label=lbl)
            plt.legend()
        except Exception:
            pass
        plt.title("Timestamp distribution (sample) with train/val cutoffs")
        plt.xlabel("Timestamp")
        plt.ylabel("Count")
        savefig_simple(FIG_DIR / "splits_timeline.png")
    except Exception:
        pass


def plot_pred_vs_actual_kde():
    """2D density of pred vs true."""
    if not PREDS_SAMPLE_CSV.exists():
        return
    try:
        df = pd.read_csv(PREDS_SAMPLE_CSV)
        if df.empty:
            return
        plt.figure(figsize=(5.5, 5.5))
        sns.kdeplot(x=df["y_true"], y=df["y_pred"], fill=True, cmap="mako")
        lims = [0, 1]
        plt.plot(lims, lims, "--", color="k", linewidth=1)
        plt.xlim(lims); plt.ylim(lims)
        plt.xlabel("True (y_next)")
        plt.ylabel("Predicted")
        plt.title("Predicted vs True (2D density, test sample)")
        savefig_simple(FIG_DIR / "pred_vs_actual_kde.png")
    except Exception:
        pass


def plot_pred_vs_actual_by_region():
    """Compare pred vs true distributions by region."""
    if not PREDS_SAMPLE_CSV.exists():
        return
    preds = pd.read_csv(PREDS_SAMPLE_CSV, parse_dates=["full_timestamp"])
    map_df = build_mapping_building_region()
    if map_df.empty:
        return
    df = preds.merge(map_df, on="building_name", how="left")
    if "region" not in df.columns:
        return
    plt.figure(figsize=(8, 4))
    sns.kdeplot(data=df, x="y_true", y="y_pred", hue="region", fill=False, common_norm=False)
    lims = [0, 1]
    plt.plot(lims, lims, "--", color="k", linewidth=1)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("True (y_next)")
    plt.ylabel("Predicted")
    plt.title("Predicted vs True by region (KDE overlays)")
    savefig_simple(FIG_DIR / "pred_vs_actual_by_region.png")


def plot_error_cdf():
    """CDF of absolute error on test sample."""
    if not PREDS_SAMPLE_CSV.exists():
        return
    df = pd.read_csv(PREDS_SAMPLE_CSV)
    if df.empty:
        return
    err = np.abs(df["y_true"] - df["y_pred"]).to_numpy()
    if err.size == 0:
        return
    xs = np.sort(err)
    ys = np.linspace(0, 1, xs.size, endpoint=True)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, color="#2ca02c")
    plt.xlabel("Absolute error")
    plt.ylabel("CDF")
    plt.title("Absolute error CDF (test sample)")
    savefig_simple(FIG_DIR / "error_cdf.png")


def main():
    parser = argparse.ArgumentParser(description="Generate expanded EDA and diagnostics plots (>30).")
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    ensure_dirs()
    sns.set_context("talk")
    sns.set_style("whitegrid")

    plots_missingness()
    plots_usage_distributions()
    plots_usage_heatmaps()
    plot_feature_correlation()
    plots_residuals_and_calibration()
    plots_pdp_ice()
    plots_elasticity_and_building_performance()
    plots_pca_and_clusters()
    plot_pairplot_sample()

    # New comparative suite
    plot_model_comparison_bar()
    plot_splits_timeline()
    plot_pred_vs_actual_kde()
    plot_pred_vs_actual_by_region()
    plot_error_cdf()


if __name__ == "__main__":
    main()
