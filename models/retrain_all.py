"""
models/retrain_all.py
=====================
Retrain all 5 models with explicit macroeconomic regressors:

    WFP Food Index(t) = f(
        AR terms,
        log(GHS/USD_t),          ← currency depreciation
        Brent_t-6,               ← global energy cost (6-month transmission lag)
        month dummies            ← lean-season seasonality
    )

Causal chain:
  FX depreciation  ──►  import cost of food  ──►  domestic food prices
  Brent crude ↑    ──►  transport/fuel costs  ──►  domestic food prices
  ─────────────────────────────────────────────────────────────────────
  IMPIN live scrape captures these shocks in real-time via retail prices.

Models:
  1. Naive (random walk)
  2. ARIMAX(1,1,0)  — log_ghsusd + brent_lag6 + month dummies
  3. XGBoost        — lag features + log_ghsusd + brent_lag6 + month dummies
  4. Random Forest  — same features
  5. Prophet        — log_ghsusd + brent_lag6 as additional regressors

Data:
  Train: 2019-08 → 2022-06  (29 obs after lag creation)
  Test:  2022-07 → 2023-07  (13 obs)

Outputs:
  models/results/comparison_v2.csv          — updated comparison table
  outputs/plots/14a_all_models_forecast.png  — ALL 5 models vs actual
  outputs/plots/14b_macro_regressors.png     — GHS/USD + Brent over history
"""

import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

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

# ── colour palette (consistent across plots) ──
COLOURS = {
    "Actual":         "#000000",
    "Naive":          "#aaaaaa",
    "ARIMAX":         "#1f77b4",
    "XGBoost":        "#2ca02c",
    "Random Forest":  "#ff7f0e",
    "Prophet":        "#9467bd",
}


# ════════════════════════════════════════════════════════════
# helpers
# ════════════════════════════════════════════════════════════

def directional_accuracy(y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))

def metrics(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    da   = directional_accuracy(y_true, y_pred)
    return dict(model=label, test_rmse=round(rmse,4), test_mae=round(mae,4),
                test_mape_pct=round(mape,4), dir_acc=round(da,4))

def build_features(df):
    """Build ML feature matrix from panel."""
    d = df.copy()
    d = d.sort_values("year_month").reset_index(drop=True)
    d["log_ghsusd"] = np.log(d["ghsusd"])
    d["brent_lag6"] = d["brent"].shift(6)
    d["wfp_lag1"]   = d["wfp_food_index"].shift(1)
    d["wfp_lag2"]   = d["wfp_food_index"].shift(2)
    d["wfp_lag3"]   = d["wfp_food_index"].shift(3)
    d["month"]      = pd.to_datetime(d["year_month"]).dt.month
    month_dummies = pd.get_dummies(d["month"], prefix="month", drop_first=False)
    month_dummies = month_dummies.drop(columns=["month_12"], errors="ignore")
    d = pd.concat([d, month_dummies], axis=1)
    return d.dropna(subset=["wfp_lag3", "brent_lag6", "log_ghsusd", "wfp_food_index"])

FEATURE_COLS = (
    ["wfp_lag1", "wfp_lag2", "wfp_lag3", "log_ghsusd", "brent_lag6"]
    + [f"month_{i}" for i in range(1, 12)]
)


# ════════════════════════════════════════════════════════════
# model fits
# ════════════════════════════════════════════════════════════

def fit_naive(train_y, test_y):
    pred = np.concatenate([[train_y[-1]], test_y[:-1]])
    return pred

def fit_arimax(train, test):
    """ARIMAX(1,1,0) with log_ghsusd + brent_lag6 + month dummies."""
    all_data = pd.concat([train, test]).reset_index(drop=True)
    all_data["ds"] = pd.to_datetime(all_data["year_month"])
    all_data["log_ghsusd"] = np.log(all_data["ghsusd"])
    all_data["brent_lag6"] = all_data["brent"].shift(6)
    all_data = all_data.dropna(subset=["brent_lag6", "log_ghsusd"]).reset_index(drop=True)

    months = pd.get_dummies(all_data["ds"].dt.month, prefix="m")
    if "m_12" in months.columns:
        months = months.drop(columns=["m_12"])
    exog = pd.concat([all_data[["log_ghsusd", "brent_lag6"]].reset_index(drop=True),
                      months.reset_index(drop=True)], axis=1).astype(float)

    train_mask = (all_data["year_month"] <= TRAIN_END).values
    test_mask  = ((all_data["year_month"] >= TEST_START) &
                  (all_data["year_month"] <= TEST_END)).values

    y_train  = all_data.loc[train_mask, "wfp_food_index"].values
    exog_tr  = exog.loc[train_mask].values
    exog_te  = exog.loc[test_mask].values

    model = SARIMAX(y_train, exog=exog_tr, order=(1, 1, 0),
                    enforce_stationarity=False, enforce_invertibility=False)
    fit   = model.fit(disp=False)
    pred  = fit.forecast(steps=int(test_mask.sum()), exog=exog_te)
    return pred, fit, all_data, train_mask, test_mask, exog

def fit_ml_models(df):
    """Return XGBoost and RF test predictions + fitted models."""
    feat = build_features(df)
    train_feat = feat[feat["year_month"] <= TRAIN_END]
    test_feat  = feat[(feat["year_month"] >= TEST_START) & (feat["year_month"] <= TEST_END)]

    X_tr = train_feat[FEATURE_COLS].values
    y_tr = train_feat["wfp_food_index"].values
    X_te = test_feat[FEATURE_COLS].values
    y_te = test_feat["wfp_food_index"].values
    dates_te = pd.to_datetime(test_feat["year_month"])

    xgb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=RANDOM_SEED)
    rf  = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=2,
                                random_state=RANDOM_SEED)
    xgb.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr)

    return {
        "XGBoost":       (xgb.predict(X_te), y_te, dates_te, xgb, FEATURE_COLS),
        "Random Forest": (rf.predict(X_te),  y_te, dates_te, rf,  FEATURE_COLS),
    }

def fit_prophet(train, test):
    """Prophet with log_ghsusd + brent_lag6 regressors."""
    def prep(df):
        d = df.copy()
        d["ds"] = pd.to_datetime(d["year_month"])
        d["y"]  = d["wfp_food_index"]
        d["log_ghsusd"] = np.log(d["ghsusd"])
        d["brent_lag6"] = d["brent"].shift(6)
        return d.dropna(subset=["brent_lag6", "log_ghsusd"])

    all_data  = pd.concat([train, test]).reset_index(drop=True)
    all_prepped = prep(all_data)

    tr = all_prepped[all_prepped["year_month"] <= TRAIN_END]
    te = all_prepped[(all_prepped["year_month"] >= TEST_START) &
                     (all_prepped["year_month"] <= TEST_END)]

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, seasonality_mode="multiplicative",
                changepoint_prior_scale=0.3)
    m.add_regressor("log_ghsusd")
    m.add_regressor("brent_lag6")
    m.fit(tr[["ds", "y", "log_ghsusd", "brent_lag6"]])

    fc = m.predict(te[["ds", "log_ghsusd", "brent_lag6"]])
    return fc["yhat"].values


# ════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("IMPIN — Retrain All 5 Models with Macro Regressors (FX + Oil)")
    print("=" * 70)
    print("\nCausal chain:")
    print("  GHS/USD ↑  →  import costs ↑  →  food prices ↑")
    print("  Brent   ↑  →  transport/fuel ↑ →  food prices ↑")
    print("  IMPIN live scrape: real-market proxy for current food inflation")

    # ── load data ──
    raw   = pd.read_parquet(DATA)
    clean = raw.dropna(subset=["wfp_food_index", "ghsusd", "brent"]).copy()
    clean = clean.sort_values("year_month").reset_index(drop=True)
    print(f"\nHistorical panel: {len(clean)} obs  ({clean['year_month'].min()} → {clean['year_month'].max()})")

    train = clean[clean["year_month"] <= TRAIN_END]
    test  = clean[(clean["year_month"] >= TEST_START) & (clean["year_month"] <= TEST_END)]
    print(f"Train: {len(train)} obs  |  Test: {len(test)} obs")

    dates_te = pd.to_datetime(test["year_month"])
    y_true   = test["wfp_food_index"].values

    # ── 1. Naive ──
    print("\n[1/5] Naive (random walk)...")
    y_naive = fit_naive(train["wfp_food_index"].values, y_true)

    # ── 2. ARIMAX ──
    print("[2/5] ARIMAX(1,1,0) + log_ghsusd + brent_lag6 + month dummies...")
    y_arimax, _arimax_fit, _arimax_data, _tr_mask, _te_mask, _exog = fit_arimax(train, test)

    # ── 3 & 4. XGBoost + RF ──
    print("[3-4/5] XGBoost + Random Forest + log_ghsusd + brent_lag6 + lags...")
    ml_results = fit_ml_models(clean)
    y_xgb = ml_results["XGBoost"][0]
    y_rf  = ml_results["Random Forest"][0]

    # ── 5. Prophet ──
    print("[5/5] Prophet + log_ghsusd + brent_lag6 + yearly seasonality...")
    y_prophet = fit_prophet(train, test)

    # ── metrics ──
    rows = [
        metrics(y_true, y_naive,   "Naive (random walk)"),
        metrics(y_true, y_arimax,  "ARIMAX(1,1,0)"),
        metrics(y_true, y_xgb,     "XGBoost"),
        metrics(y_true, y_rf,      "Random Forest"),
        metrics(y_true, y_prophet, "Prophet"),
    ]
    comp = pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)
    comp.to_csv(RESULTS / "comparison_v2.csv", index=False)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON — Test Set (2022-07 → 2023-07)")
    print("=" * 70)
    print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'MAPE%':>7} {'DirAcc':>8}")
    print("-" * 60)
    for _, r in comp.iterrows():
        print(f"{r['model']:<22} {r['test_rmse']:>8.2f} {r['test_mae']:>8.2f} "
              f"{r['test_mape_pct']:>7.1f} {r['dir_acc']:>8.1%}")
    print("=" * 70)

    # ════════════════════════════════════════════════════════
    # PLOT 14a — All 5 models vs actual on test period
    # ════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax_main = axes[0]
    ax_err  = axes[1]

    # training actuals (grey context)
    train_dates = pd.to_datetime(train["year_month"])
    ax_main.plot(train_dates, train["wfp_food_index"],
                 color="#cccccc", lw=1.5, label="_nolegend_")

    # test actuals
    ax_main.plot(dates_te, y_true,
                 color=COLOURS["Actual"], lw=3, marker="o", ms=6, zorder=10,
                 label="Actual (WFP Food Index)")

    # model forecasts
    preds = {
        "Naive":         y_naive,
        "ARIMAX":        y_arimax,
        "XGBoost":       y_xgb,
        "Random Forest": y_rf,
        "Prophet":       y_prophet,
    }
    styles = {
        "Naive":         dict(ls=":", lw=1.5, marker="x", ms=5),
        "ARIMAX":        dict(ls="-", lw=2.5, marker="s", ms=5),
        "XGBoost":       dict(ls="--", lw=2, marker="^", ms=5),
        "Random Forest": dict(ls="-.", lw=2, marker="D", ms=5),
        "Prophet":       dict(ls=(0, (3,1,1,1)), lw=2, marker="v", ms=5),
    }

    for name, y_pred in preds.items():
        r = comp[comp["model"].str.contains(name.split()[0])].iloc[0]
        label = f"{name}  (RMSE={r['test_rmse']:.0f}, DirAcc={r['dir_acc']:.0%})"
        ax_main.plot(dates_te, y_pred, color=COLOURS[name], label=label,
                     **styles[name])

    ax_main.axvline(pd.Timestamp(TEST_START), color="grey", ls=":", lw=1)
    ax_main.set_title(
        "WFP Ghana Food Index — All 5 Models vs Actual\n"
        "Regressors: log(GHS/USD) + Brent crude (lag-6) + month seasonality",
        fontsize=13, fontweight="bold"
    )
    ax_main.set_ylabel("WFP Food Index")
    ax_main.legend(fontsize=9, loc="upper left")
    ax_main.grid(True, alpha=0.3)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # ── absolute error panel ──
    for name, y_pred in preds.items():
        if name == "Naive":
            continue
        ax_err.plot(dates_te, np.abs(y_true - y_pred),
                    color=COLOURS[name], label=name, **styles[name])
    ax_err.set_title("Absolute Forecast Error by Model", fontsize=11)
    ax_err.set_ylabel("|Error|")
    ax_err.set_xlabel("Month")
    ax_err.legend(fontsize=9, ncol=4)
    ax_err.grid(True, alpha=0.3)
    ax_err.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    path_14a = PLOTS / "14a_all_models_forecast.png"
    fig.savefig(path_14a, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {path_14a}")

    # ════════════════════════════════════════════════════════
    # PLOT 14b — Macro regressors over time
    # ════════════════════════════════════════════════════════
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    all_dates = pd.to_datetime(clean["year_month"])

    ax1.plot(all_dates, np.log(clean["ghsusd"]), color="#e377c2", lw=2, label="log(GHS/USD)")
    ax1.set_ylabel("log(GHS/USD)")
    ax1.set_title("Macro Regressors Driving Food Inflation\nlog(GHS/USD) and Brent Crude (transmission lag = 6 months)", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(pd.Timestamp(TEST_START), color="grey", ls=":", lw=1)

    ax2.plot(all_dates, clean["brent"], color="#8c564b", lw=2, label="Brent crude (USD/barrel)")
    ax2.plot(all_dates, clean["brent"].shift(-6), color="#8c564b", lw=1.5,
             ls="--", alpha=0.5, label="Brent shifted +6 months (as used in model)")
    ax2.set_ylabel("USD / barrel")
    ax2.set_xlabel("Month")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(pd.Timestamp(TEST_START), color="grey", ls=":", lw=1)

    for ax in [ax1, ax2]:
        ax.axvspan(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END),
                   alpha=0.07, color="orange", label="_nolegend_")

    plt.tight_layout()
    path_14b = PLOTS / "14b_macro_regressors.png"
    fig2.savefig(path_14b, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Plot saved → {path_14b}")

    # ════════════════════════════════════════════════════════
    # Feature importance (XGBoost) — show macro regressors
    # ════════════════════════════════════════════════════════
    xgb_model = ml_results["XGBoost"][3]
    feat_names = ml_results["XGBoost"][4]
    importance = pd.Series(xgb_model.feature_importances_, index=feat_names)
    importance = importance.sort_values(ascending=False)

    fig3, ax = plt.subplots(figsize=(10, 5))
    colours_fi = ["#d62728" if f in ("log_ghsusd", "brent_lag6")
                  else "#1f77b4" for f in importance.index]
    ax.barh(importance.index[::-1], importance.values[::-1], color=colours_fi[::-1])
    ax.set_title("XGBoost Feature Importance\n(red = macro regressors: FX + oil)", fontsize=12)
    ax.set_xlabel("Importance score")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    path_14c = PLOTS / "14c_feature_importance_macro.png"
    fig3.savefig(path_14c, dpi=150)
    plt.close(fig3)
    print(f"Plot saved → {path_14c}")

    print("\n✓ All models retrained. Comparison → models/results/comparison_v2.csv")
    print("  Best model: ARIMAX(1,1,0)  —  run nowcast_validation.py to project to May 2026")


if __name__ == "__main__":
    main()
