#!/usr/bin/env python3
"""
Inputs:
- outputs/models/hgbr.joblib or outputs/models/rf.joblib
- outputs/metrics/metrics.json
- data/processed/features.csv

Outputs:
- outputs/figures/shap_summary.png (beeswarm plot)
- outputs/figures/shap_waterfall_example.png (individual prediction explanation)
- outputs/tables/shap_importance_summary.csv
"""

import argparse
import json
from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def ensure_dirs():
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)


def load_best_model_and_metrics(metrics_path: str) -> tuple:
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

    model_path = f"outputs/models/{'hgbr' if best == 'HGBR' else 'rf'}.joblib"
    model = joblib.load(model_path)

    return model, best


def sample_test_data(features_csv: str, metrics_path: str, sample_size: int = 10000, chunksize: int = 500_000) -> pd.DataFrame:
    """
    Sample a stratified subset from test period.
    """
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    t_val_end = pd.to_datetime(metrics["cutoffs"]["val_end"])

    header = pd.read_csv(features_csv, nrows=0).columns.tolist()
    candidate_features = [
        "lag_1h", "lag_24h", "lag_168h", "rollmean_24h",
        "hour", "day_of_week", "month",
        "temp_z", "temp_z_lag_1h", "temp_z_lag_24h",
        "apparent_temperature_norm", "temp_lag_1h", "temp_lag_24h",
        "precipitation", "is_day", "is_weekend", "is_holiday", "region_id"
    ]
    feature_cols = [c for c in candidate_features if c in header]
    target_col = "y_next"
    keep_cols = ["building_name", "full_timestamp", target_col] + feature_cols

    # Collecting test data
    test_data = []
    for chunk in pd.read_csv(features_csv, usecols=keep_cols, parse_dates=["full_timestamp"],
                             chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk["full_timestamp"] > t_val_end]
        if not chunk.empty:
            test_data.append(chunk)

    if not test_data:
        raise RuntimeError("No test data found")

    df_test = pd.concat(test_data, ignore_index=True)

    # Stratified sample by building
    if len(df_test) <= sample_size:
        return df_test

    # Sample proportionally by building
    building_counts = df_test["building_name"].value_counts()
    sample_per_building = max(1, sample_size // len(building_counts))

    sampled = []
    for bld in building_counts.index:
        bld_data = df_test[df_test["building_name"] == bld]
        n_take = min(len(bld_data), sample_per_building)
        sampled.append(bld_data.sample(n=n_take, random_state=42))

    df_sample = pd.concat(sampled, ignore_index=True)

    # If still over, random sample down
    if len(df_sample) > sample_size:
        df_sample = df_sample.sample(n=sample_size, random_state=42)

    return df_sample


def main():
    parser = argparse.ArgumentParser(description="Compute SHAP analysis for model interpretability")
    parser.add_argument("--features", type=str, default="data/processed/features.csv")
    parser.add_argument("--metrics", type=str, default="outputs/metrics/metrics.json")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of test samples for SHAP")
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    ensure_dirs()

    # Load best model
    model, model_name = load_best_model_and_metrics(args.metrics)
    print(f"Using model: {model_name}")

    # Sample test data
    df_sample = sample_test_data(args.features, args.metrics, args.sample_size, args.chunksize)
    print(f"Sampled {len(df_sample)} rows from test set")

    # Define features
    header = pd.read_csv(args.features, nrows=0).columns.tolist()
    candidate_features = [
        "lag_1h", "lag_24h", "lag_168h", "rollmean_24h",
        "hour", "day_of_week", "month",
        "temp_z", "temp_z_lag_1h", "temp_z_lag_24h",
        "apparent_temperature_norm", "temp_lag_1h", "temp_lag_24h",
        "precipitation", "is_day", "is_weekend", "is_holiday", "region_id"
    ]
    feature_cols = [c for c in candidate_features if c in header and c in df_sample.columns]

    X_sample = df_sample[feature_cols].to_numpy(dtype=np.float32)

    # Creating SHAP explainer
    explainer = shap.Explainer(model)

    # Computing SHAP values
    if len(X_sample) > 5000:
        idx = np.random.choice(len(X_sample), size=5000, replace=False)
        X_shap = X_sample[idx]
    else:
        X_shap = X_sample

    # Use the new unified interface to obtain an Explanation object
    explanation = explainer(X_shap, check_additivity=False)
    shap_values = explanation.values

    # Summary plot (beeswarm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_cols, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig("outputs/figures/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Wrote outputs/figures/shap_summary.png")

    # Waterfall plot for one example prediction
    if shap_values.shape[0] > 0:
        example_idx = 0  # First example
        # Use the new API: pass a single-row Explanation
        shap.plots.waterfall(explanation[example_idx], max_display=15)
        plt.tight_layout()
        plt.savefig("outputs/figures/shap_waterfall_example.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Wrote outputs/figures/shap_waterfall_example.png")

    # Importance summary
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap
    })
    importance_df.sort_values("mean_abs_shap", ascending=False, inplace=True)
    importance_df.to_csv("outputs/tables/shap_importance_summary.csv", index=False)
    print("Wrote outputs/tables/shap_importance_summary.csv")

    print("SHAP analysis complete")


if __name__ == "__main__":
    main()

