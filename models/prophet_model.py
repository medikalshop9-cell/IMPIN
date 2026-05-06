"""
IMPIN — Prophet Forecaster
==========================
Bayesian structural time-series with weekly seasonality disabled (monthly data).
Regressors: log_ghsusd, brent_lag6.

Train/test split matches ARIMAX:
  Train: 2019-08 → 2022-06  (35 obs)
  Test:  2022-07 → 2023-07  (13 obs)

Outputs:
  models/results/prophet_metrics.csv  — test set metrics
  outputs/plots/12a_prophet_forecast.png
  outputs/plots/12b_prophet_components.png
"""

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pathlib import Path

from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

RANDOM_SEED = 42
TRAIN_END = "2022-06"
TEST_START = "2022-07"
TEST_END = "2023-07"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "historical_panel.parquet"
RESULTS = ROOT / "models" / "results"
PLOTS = ROOT / "outputs" / "plots"
RESULTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))


def main():
    print("=" * 60)
    print("IMPIN — Prophet Forecaster")
    print("=" * 60)

    # ── load data ──
    raw = pd.read_parquet(DATA)
    df = raw.dropna(subset=["wfp_food_index", "ghsusd", "brent"]).copy()
    df = df.sort_values("year_month").reset_index(drop=True)

    # Prophet requires columns named 'ds' (datetime) and 'y' (target)
    df["ds"] = pd.to_datetime(df["year_month"])
    df["y"] = df["wfp_food_index"]
    df["log_ghsusd"] = np.log(df["ghsusd"])
    df["brent_lag6"] = df["brent"].shift(6)
    df = df.dropna(subset=["brent_lag6"])

    print(f"\nClean panel: {len(df)} rows  ({df['year_month'].min()} → {df['year_month'].max()})")

    # ── train / test split ──
    train = df[df["ds"] <= pd.Timestamp(f"{TRAIN_END}-30")]
    test = df[(df["ds"] >= pd.Timestamp(f"{TEST_START}-01")) &
              (df["ds"] <= pd.Timestamp(f"{TEST_END}-31"))]

    print(f"Train: {len(train)} obs  |  Test: {len(test)} obs")

    # ── fit model ──
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.3,
    )
    model.add_regressor("log_ghsusd")
    model.add_regressor("brent_lag6")

    model.fit(train[["ds", "y", "log_ghsusd", "brent_lag6"]])

    # ── predict on test ──
    future = test[["ds", "log_ghsusd", "brent_lag6"]].copy()
    forecast = model.predict(future)

    y_true = test["y"].values
    y_pred = forecast["yhat"].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    da = directional_accuracy(y_true, y_pred)

    # naive baseline
    naive_pred = np.concatenate([[y_true[0]], y_true[:-1]])
    naive_rmse = np.sqrt(mean_squared_error(y_true, naive_pred))
    naive_da = directional_accuracy(y_true, naive_pred)

    print(f"\nProphet:")
    print(f"  RMSE:    {rmse:.2f}  (naive: {naive_rmse:.2f})")
    print(f"  MAE:     {mae:.2f}")
    print(f"  MAPE:    {mape:.1f}%")
    print(f"  Dir Acc: {da:.1%}  (naive: {naive_da:.1%})")

    # ── save metrics ──
    metrics = pd.DataFrame([{
        "model": "Prophet",
        "test_rmse": round(rmse, 4),
        "test_mae": round(mae, 4),
        "test_mape_pct": round(mape, 4),
        "dir_acc": round(da, 4),
        "train_obs": len(train),
        "test_obs": len(test),
    }])
    metrics.to_csv(RESULTS / "prophet_metrics.csv", index=False)
    print(f"\nMetrics saved → models/results/prophet_metrics.csv")

    # ── plot 12a: forecast vs actuals ──
    fig, ax = plt.subplots(figsize=(12, 5))

    # training actuals
    ax.plot(train["ds"], train["y"], color="#aaaaaa", lw=1.5, label="Train (actual)")

    # test actuals
    ax.plot(test["ds"], y_true, color="#1f77b4", lw=2, marker="o", ms=5, label="Test (actual)")

    # Prophet forecast + CI
    ax.plot(forecast["ds"], forecast["yhat"], color="#9467bd", ls="--", lw=2, label="Prophet forecast")
    ax.fill_between(
        forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
        color="#9467bd", alpha=0.2, label="80% CI"
    )

    ax.axvline(pd.Timestamp(TEST_START), color="grey", ls=":", lw=1, label="Train/Test split")
    ax.set_title("WFP Food Index — Prophet Forecast vs Actuals", fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("WFP Food Index (base 2019-08=100)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path_12a = PLOTS / "12a_prophet_forecast.png"
    fig.savefig(path_12a, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {path_12a}")

    # ── plot 12b: Prophet components ──
    # extend future to end of test period for component plot
    future_full = model.make_future_dataframe(periods=len(test), freq="MS")
    future_full = future_full.merge(
        df[["ds", "log_ghsusd", "brent_lag6"]], on="ds", how="left"
    )
    future_full[["log_ghsusd", "brent_lag6"]] = (
        future_full[["log_ghsusd", "brent_lag6"]].ffill().bfill()
    )
    forecast_full = model.predict(future_full)

    try:
        fig_comp = model.plot_components(forecast_full)
        path_12b = PLOTS / "12b_prophet_components.png"
        fig_comp.savefig(path_12b, dpi=150, bbox_inches="tight")
        plt.close(fig_comp)
        print(f"Plot saved → {path_12b}")
    except Exception as e:
        print(f"Component plot skipped: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<18} {'RMSE':>8} {'MAE':>8} {'MAPE%':>8} {'DirAcc':>8}")
    print("-" * 58)
    print(f"{'Naive':18} {naive_rmse:8.2f} {'—':>8} {'—':>8} {naive_da:8.1%}")
    print(f"{'Prophet':18} {rmse:8.2f} {mae:8.2f} {mape:8.1f} {da:8.1%}")
    print("=" * 60)
    print("\nARIMAX(1,1,0) reference: RMSE=47.15  DirAcc=83.3%")


if __name__ == "__main__":
    main()
