#!/usr/bin/env python3
"""

Inputs:
- outputs/models/hgbr.joblib or outputs/models/rf.joblib
- outputs/metrics/metrics.json
- data/processed/features.csv

Outputs:
- outputs/figures/residuals_vs_fitted.png
"""

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def ensure_dirs():
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)


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

    return model, best, metrics


def stream_test_predictions(features_csv: str, model, feature_cols: list, metrics: dict, chunksize: int = 500_000, sample_size: int = 100000) -> tuple:
    """
    Stream through test data, sample up to sample_size rows, and collect predictions/residuals.
    Returns: (residuals, fitted_values, timestamps)
    """
    t_val_end = pd.to_datetime(metrics["cutoffs"]["val_end"])
    target_col = "y_next"

    test_chunks = []
    total_collected = 0

    for chunk in pd.read_csv(features_csv, usecols=["full_timestamp", target_col] + feature_cols,
                             parse_dates=["full_timestamp"], chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk["full_timestamp"] > t_val_end]
        if chunk.empty:
            continue

        chunk = chunk.dropna(subset=[target_col] + feature_cols)
        if chunk.empty:
            continue

        # Sample from chunk if needed
        need = sample_size - total_collected
        if need <= 0:
            break
        n_take = min(len(chunk), need)
        sampled = chunk.sample(n=n_take, random_state=42) if n_take < len(chunk) else chunk

        test_chunks.append(sampled)
        total_collected += len(sampled)

        if total_collected >= sample_size:
            break

    if not test_chunks:
        return np.array([]), np.array([]), []

    df_test = pd.concat(test_chunks, ignore_index=True)

    X = df_test[feature_cols].to_numpy(dtype=np.float32)
    y_true = df_test[target_col].to_numpy(dtype=np.float32)

    y_pred = model.predict(X)
    res = y_true - y_pred

    return res, y_pred, df_test["full_timestamp"].tolist()




def plot_residuals_vs_fitted(residuals: np.ndarray, fitted: np.ndarray):
    """Plot residuals vs fitted values to check heteroscedasticity."""
    plt.figure(figsize=(8, 6))
    plt.scatter(fitted, residuals, alpha=0.3, s=1)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted Values")
    plt.tight_layout()
    plt.savefig("outputs/figures/residuals_vs_fitted.png", dpi=150)
    plt.close()
    print("Wrote outputs/figures/residuals_vs_fitted.png")






def main():
    parser = argparse.ArgumentParser(description="Compute residual diagnostics and prediction intervals")
    parser.add_argument("--features", type=str, default="data/processed/features.csv")
    parser.add_argument("--metrics", type=str, default="outputs/metrics/metrics.json")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--sample-size", type=int, default=100000, help="Sample size from test set for diagnostics")
    parser.add_argument("--max-lag", type=int, default=50, help="Max lag for ACF/PACF")
    args = parser.parse_args()

    ensure_dirs()

    # Load model and metrics
    model, model_name, metrics = load_best_model_and_metrics(args.metrics)
    print(f"Using model: {model_name}")

    # Define features matching the trained model's expected input (14 features with positive importance)
    feature_cols = [
        "lag_1h", "lag_24h", "lag_168h", "rollmean_24h",
        "hour", "day_of_week", "month",
        "temp_z", "temp_z_lag_1h", "temp_z_lag_24h",
        "precipitation", "is_day", "is_weekend", "is_holiday"
    ]

    # Stream predictions on sampled test set
    residuals, fitted, timestamps = stream_test_predictions(
        args.features, model, feature_cols, metrics, args.chunksize, args.sample_size
    )

    print(f"Collected {len(residuals)} test predictions")

    if len(residuals) == 0:
        print("No test data found, skipping diagnostics")
        return

    # Residuals vs fitted
    plot_residuals_vs_fitted(residuals, fitted)

    print("Residual diagnostics complete")


if __name__ == "__main__":
    main()
