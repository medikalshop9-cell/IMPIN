"""
IMPIN — ML Forecasters: XGBoost + Random Forest
================================================
Train/test split matches ARIMAX exactly:
  Train: 2019-08 → 2022-06  (35 obs after lag features created)
  Test:  2022-07 → 2023-07  (13 obs)

Features per row t:
  wfp_lag1, wfp_lag2, wfp_lag3   — lagged target (levels, not diff)
  log_ghsusd_t                   — exchange rate (log)
  brent_lag6                     — Brent crude 6 months prior
  month_1 … month_11             — month dummies (month 12 = reference)

Target: wfp_food_index (levels — tree models handle non-stationarity fine)

Outputs:
  models/results/ml_comparison.csv   — per-model metrics
  outputs/plots/11a_ml_forecast.png  — forecast vs actuals
  outputs/plots/11b_ml_features.png  — XGBoost feature importance
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

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


# ── helpers ──────────────────────────────────────────────────────────────────

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of steps where predicted and actual direction agree."""
    if len(y_true) < 2:
        return float("nan")
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(actual_dir == pred_dir))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features, log exchange rate, brent lag 6, and month dummies.
    Returns a copy with only complete rows (no NaN in feature columns).
    """
    d = df.copy()
    d = d.sort_values("year_month").reset_index(drop=True)

    # lag features of target
    d["wfp_lag1"] = d["wfp_food_index"].shift(1)
    d["wfp_lag2"] = d["wfp_food_index"].shift(2)
    d["wfp_lag3"] = d["wfp_food_index"].shift(3)

    # log exchange rate
    d["log_ghsusd"] = np.log(d["ghsusd"])

    # brent at lag 6
    d["brent_lag6"] = d["brent"].shift(6)

    # month dummies (drop month 12 as reference)
    d["month"] = pd.to_datetime(d["year_month"]).dt.month
    for m in range(1, 12):
        d[f"month_{m}"] = (d["month"] == m).astype(int)
    d = d.drop(columns=["month"])

    # drop rows with NaN in any feature column (first 6 rows will have NaN)
    feature_cols = (
        ["wfp_lag1", "wfp_lag2", "wfp_lag3", "log_ghsusd", "brent_lag6"]
        + [f"month_{m}" for m in range(1, 12)]
    )
    d = d.dropna(subset=feature_cols + ["wfp_food_index"])
    return d, feature_cols


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("IMPIN — ML Forecasters (XGBoost + Random Forest)")
    print("=" * 60)

    # ── load data ──
    raw = pd.read_parquet(DATA)
    raw = raw.dropna(subset=["wfp_food_index", "ghsusd", "brent"])
    print(f"\nClean panel: {len(raw)} rows  ({raw['year_month'].min()} → {raw['year_month'].max()})")

    df, feature_cols = build_features(raw)
    print(f"After lag features: {len(df)} rows")

    X = df[feature_cols].values
    y = df["wfp_food_index"].values
    dates = pd.to_datetime(df["year_month"])

    # ── train / test split ──
    train_mask = dates <= pd.Period(TRAIN_END, "M").to_timestamp(how="E")
    test_mask = (dates >= pd.Period(TEST_START, "M").to_timestamp()) & (
        dates <= pd.Period(TEST_END, "M").to_timestamp(how="E")
    )

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    dates_test = dates[test_mask]

    print(f"\nTrain: {train_mask.sum()} obs  |  Test: {test_mask.sum()} obs")

    # ── models ──
    models = {
        "XGBoost": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
        ),
    }

    # naive persistence baseline (y_pred[t] = y_true[t-1])
    naive_pred = np.concatenate([[y_test[0]], y_test[:-1]])
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
    naive_mae = mean_absolute_error(y_test, naive_pred)
    naive_da = directional_accuracy(y_test, naive_pred)

    results = []
    predictions = {"Naive": naive_pred}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred

        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        mape = float(np.mean(np.abs((y_test - pred) / y_test)) * 100)
        da = directional_accuracy(y_test, pred)

        results.append(
            {
                "model": name,
                "test_rmse": round(rmse, 4),
                "test_mae": round(mae, 4),
                "test_mape_pct": round(mape, 4),
                "dir_acc": round(da, 4),
                "train_obs": int(train_mask.sum()),
                "test_obs": int(test_mask.sum()),
            }
        )

        print(f"\n{name}:")
        print(f"  RMSE:    {rmse:.2f}  (naive: {naive_rmse:.2f})")
        print(f"  MAE:     {mae:.2f}  (naive: {naive_mae:.2f})")
        print(f"  MAPE:    {mape:.1f}%")
        print(f"  Dir Acc: {da:.1%}  (naive: {naive_da:.1%})")

        joblib.dump(model, RESULTS / f"{name.lower().replace(' ', '_')}_model.pkl")
        print(f"  Saved → models/results/{name.lower().replace(' ', '_')}_model.pkl")

    # ── save results ──
    results_df = pd.DataFrame(results)
    out_csv = RESULTS / "ml_comparison.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nResults saved → {out_csv}")

    # ── plot 11a: forecast vs actuals ──
    fig, ax = plt.subplots(figsize=(12, 5))

    # full actual series for context
    ax.plot(
        pd.to_datetime(df["year_month"][train_mask]),
        y_train,
        color="#aaaaaa",
        lw=1.5,
        label="Train (actual)",
    )
    ax.plot(
        dates_test,
        y_test,
        color="#1f77b4",
        lw=2.0,
        marker="o",
        ms=5,
        label="Test (actual)",
    )

    colours = {"XGBoost": "#d62728", "Random Forest": "#2ca02c", "Naive": "#ff7f0e"}
    linestyles = {"XGBoost": "--", "Random Forest": "-.", "Naive": ":"}
    for model_name, pred in predictions.items():
        ax.plot(
            dates_test,
            pred,
            color=colours[model_name],
            ls=linestyles[model_name],
            lw=1.8,
            label=model_name,
        )

    ax.axvline(pd.Timestamp(TEST_START), color="grey", ls=":", lw=1, label="Train/Test split")
    ax.set_title("WFP Food Index — ML Model Forecasts vs Actuals", fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("WFP Food Index (base 2019-08=100)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path_11a = PLOTS / "11a_ml_forecast.png"
    fig.savefig(path_11a, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {path_11a}")

    # ── plot 11b: XGBoost feature importance ──
    xgb_model = models["XGBoost"]
    importances = xgb_model.feature_importances_
    feat_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feat_df["feature"], feat_df["importance"], color="#d62728", alpha=0.8)
    ax.set_title("XGBoost Feature Importances", fontsize=13)
    ax.set_xlabel("Importance (mean decrease in loss)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path_11b = PLOTS / "11b_ml_features.png"
    fig.savefig(path_11b, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {path_11b}")

    # ── summary ──
    print("\n" + "=" * 60)
    print("SUMMARY — Test Set Performance")
    print("=" * 60)
    print(f"{'Model':<18} {'RMSE':>8} {'MAE':>8} {'MAPE%':>8} {'DirAcc':>8}")
    print("-" * 58)
    print(f"{'Naive':18} {naive_rmse:8.2f} {naive_mae:8.2f} {'—':>8} {naive_da:8.1%}")
    for r in results:
        print(
            f"{r['model']:18} {r['test_rmse']:8.2f} {r['test_mae']:8.2f}"
            f" {r['test_mape_pct']:8.1f} {r['dir_acc']:8.1%}"
        )
    print("=" * 60)
    print("\nARIMAX(1,1,0) reference: RMSE=47.15  DirAcc=83.3%")


if __name__ == "__main__":
    main()
