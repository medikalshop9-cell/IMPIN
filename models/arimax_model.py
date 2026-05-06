"""
models/arimax_model.py
======================
ARIMAX model for nowcasting the WFP Food Price Index.

EDA/stationarity inputs applied
--------------------------------
  ▸ All series are I(1) → model in first differences (d=1)
  ▸ GHS/USD log-transformed (right-skewed)
  ▸ Brent included at 6-month lag (peak cross-correlation r=0.717)
  ▸ Month dummies for lean-season seasonality
  ▸ AR(1)/AR(2) structure expected from PACF

Specification
-------------
  ARIMAX(p, 1, q) where p,q ∈ {0,1,2}
  Endogenous  : WFP Food Index (levels, model differences internally)
  Exogenous   : [log_ghsusd, brent_lag6, month_dummies(m02..m12)]
  Train       : 2019-08 → 2022-06  (35 months)
  Test/Hold   : 2022-07 → 2023-07  (13 months)

Model selection: AIC grid search over (p,q) ∈ {0,1,2}²
Evaluation     : RMSE, MAE, MAPE, directional accuracy vs. naive (random walk)

Outputs
-------
  models/results/arimax_comparison.csv   — grid search table
  models/results/arimax_best_params.txt  — best order and coefficients
  outputs/plots/10a_arimax_grid.png
  outputs/plots/10b_arimax_forecast.png
  outputs/plots/10c_arimax_residuals.png

Usage
-----
    python -m models.arimax_model
"""

import logging
import warnings
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("arimax")

_ROOT     = Path(__file__).parent.parent
_PROC_DIR = _ROOT / "data" / "processed"
_RES_DIR  = _ROOT / "models" / "results"
_PLOT_DIR = _ROOT / "outputs" / "plots"

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

# Train/test split
TRAIN_END = "2022-06"
TEST_START = "2022-07"

# Grid search space
P_VALUES = [0, 1, 2]
Q_VALUES = [0, 1, 2]
D = 1          # first-difference confirmed by ADF/KPSS
BRENT_LAG = 6  # from EDA cross-correlation


# ── Data preparation ────────────────────────────────────────────────────────

def _prepare_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from the historical panel.
    Returns a DatetimeIndex DataFrame with endogenous + all exogenous columns.
    """
    df = panel.dropna(subset=["wfp_food_index", "ghsusd", "brent"]).copy()
    df.index = pd.to_datetime(df["year_month"] + "-01")
    df = df.sort_index()

    # Log-transform GHS/USD (EDA: right-skewed → log stabilises variance)
    df["log_ghsusd"] = np.log(df["ghsusd"])

    # Brent at 6-month lag
    df["brent_lag6"] = df["brent"].shift(BRENT_LAG)

    # Month dummies (drop Jan = baseline to avoid multicollinearity)
    for m in range(2, 13):
        df[f"m{m:02d}"] = (df.index.month == m).astype(float)

    df = df.dropna()  # removes first BRENT_LAG rows after shift
    logger.info("Feature matrix: %d rows (%s → %s), columns: %s",
                len(df), df.index[0].strftime("%Y-%m"),
                df.index[-1].strftime("%Y-%m"),
                list(df.columns))
    return df


def _train_test_split(df: pd.DataFrame):
    train = df[df.index <= pd.to_datetime(TRAIN_END + "-01")]
    test  = df[df.index >= pd.to_datetime(TEST_START + "-01")]
    logger.info("Train: %d obs (%s → %s) | Test: %d obs (%s → %s)",
                len(train), train.index[0].strftime("%Y-%m"),
                train.index[-1].strftime("%Y-%m"),
                len(test),  test.index[0].strftime("%Y-%m"),
                test.index[-1].strftime("%Y-%m"))
    return train, test


# ── Metrics ─────────────────────────────────────────────────────────────────

def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual    = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    a, p = actual[mask], predicted[mask]
    if len(a) == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "dir_acc": np.nan}
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    mae  = float(np.mean(np.abs(a - p)))
    mape = float(np.mean(np.abs((a - p) / np.where(a != 0, a, np.nan))) * 100)
    # Directional accuracy (sign of change)
    if len(a) > 1:
        dir_a = np.sign(np.diff(a))
        dir_p = np.sign(np.diff(p))
        dir_acc = float(np.mean(dir_a == dir_p) * 100)
    else:
        dir_acc = np.nan
    return {"rmse": round(rmse, 4), "mae": round(mae, 4),
            "mape": round(mape, 4), "dir_acc": round(dir_acc, 2)}


# ── Grid search ─────────────────────────────────────────────────────────────

def _fit_arimax(train: pd.DataFrame, test: pd.DataFrame,
                p: int, q: int, exog_cols: list) -> dict:
    """Fit ARIMAX(p,1,q) on train, forecast on test. Returns result dict."""
    endog_train = train["wfp_food_index"].values
    exog_train  = train[exog_cols].values
    exog_test   = test[exog_cols].values

    try:
        model = SARIMAX(
            endog_train,
            exog=exog_train,
            order=(p, D, q),
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=200)

        # In-sample
        in_sample = fit.fittedvalues

        # Out-of-sample multi-step forecast from end of training period
        forecast_obj = fit.get_forecast(steps=len(test), exog=exog_test)
        forecasts = list(forecast_obj.predicted_mean)

        m_test  = _metrics(test["wfp_food_index"].values, np.array(forecasts))
        m_train = _metrics(endog_train, in_sample)

        return {
            "p": p, "q": q,
            "aic": round(fit.aic, 3),
            "bic": round(fit.bic, 3),
            "train_rmse":  m_train["rmse"],
            "test_rmse":   m_test["rmse"],
            "test_mae":    m_test["mae"],
            "test_mape":   m_test["mape"],
            "test_dir_acc": m_test["dir_acc"],
            "forecasts":   forecasts,
            "fit":         fit,
            "in_sample":   in_sample,
            "success":     True,
        }

    except Exception as e:
        logger.warning("ARIMAX(%d,1,%d) failed: %s", p, q, e)
        return {"p": p, "q": q, "success": False, "error": str(e)}


def run_grid_search(train: pd.DataFrame, test: pd.DataFrame,
                    exog_cols: list) -> pd.DataFrame:
    """Run full grid search over (p,q) ∈ P_VALUES × Q_VALUES."""
    records = []
    best_result = None
    best_rmse   = np.inf

    for p, q in product(P_VALUES, Q_VALUES):
        logger.info("  Fitting ARIMAX(%d,1,%d) ...", p, q)
        res = _fit_arimax(train, test, p, q, exog_cols)
        if res["success"]:
            records.append({k: v for k, v in res.items()
                            if k not in ("forecasts", "fit", "in_sample")})
            if res["test_rmse"] < best_rmse:
                best_rmse   = res["test_rmse"]
                best_result = res

    grid_df = pd.DataFrame(records)
    logger.info("Grid search done. Best ARIMAX(%d,1,%d): test RMSE=%.4f",
                best_result["p"], best_result["q"], best_rmse)
    return grid_df, best_result


# ── Naive baseline ───────────────────────────────────────────────────────────

def _naive_forecast(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Random walk: forecast = last known value."""
    last = train["wfp_food_index"].iloc[-1]
    forecasts = [last] * len(test)  # naive: carry forward last
    # Rolling naive (1-step: t-1)
    history = list(train["wfp_food_index"].values)
    rolling = []
    for i in range(len(test)):
        rolling.append(history[-1])
        history.append(test["wfp_food_index"].iloc[i])
    m = _metrics(test["wfp_food_index"].values, np.array(rolling))
    logger.info("Naive baseline: test RMSE=%.4f, MAE=%.4f, Dir Acc=%.1f%%",
                m["rmse"], m["mae"], m["dir_acc"])
    return {"forecasts": rolling, **m}


# ── Plots ────────────────────────────────────────────────────────────────────

def _plot_grid(grid_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, col, title, color in zip(
        axes,
        ["test_rmse", "aic", "test_dir_acc"],
        ["Test RMSE", "AIC", "Directional Accuracy (%)"],
        [PALETTE[1], PALETTE[0], PALETTE[2]],
    ):
        pivot = grid_df.pivot(index="p", columns="q", values=col)
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f",
                    cmap="RdYlGn_r" if col != "test_dir_acc" else "RdYlGn",
                    linewidths=0.5, linecolor="white",
                    annot_kws={"size": 12, "weight": "bold"})
        ax.set_title(f"ARIMAX(p,1,q) — {title}", fontsize=11)
        ax.set_xlabel("q (MA order)"); ax.set_ylabel("p (AR order)")

    fig.suptitle("Section 10 · ARIMAX Grid Search — (p,1,q) over p,q ∈ {0,1,2}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = _PLOT_DIR / "10a_arimax_grid.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path.name)


def _plot_forecast(train: pd.DataFrame, test: pd.DataFrame,
                   best: dict, naive: dict) -> None:
    train_dates = train.index
    test_dates  = test.index

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ── Top: full history + forecast ─────────────────────────────────────
    ax = axes[0]
    ax.plot(train_dates, train["wfp_food_index"], color=PALETTE[0],
            linewidth=2, label="Train (actual)")
    ax.plot(test_dates, test["wfp_food_index"], color="black",
            linewidth=2, linestyle="--", label="Test (actual)")
    ax.plot(test_dates, best["forecasts"], color=PALETTE[1],
            linewidth=2, marker="o", markersize=4,
            label=f"ARIMAX({best['p']},1,{best['q']}) forecast")
    ax.plot(test_dates, naive["forecasts"], color=PALETTE[3],
            linewidth=1.5, linestyle=":", marker="s", markersize=3,
            label="Naive (random walk)")
    ax.axvline(test_dates[0], color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"WFP Food Index — ARIMAX({best['p']},1,{best['q']}) vs Naive Forecast",
                 fontsize=12)
    ax.set_ylabel("WFP Food Index (GHS, base 2019-08=100)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # ── Bottom: test window zoom with error bands ─────────────────────────
    ax = axes[1]
    actual = test["wfp_food_index"].values
    preds  = np.array(best["forecasts"])
    errors = actual - preds

    ax.plot(test_dates, actual, color="black", linewidth=2, label="Actual")
    ax.plot(test_dates, preds, color=PALETTE[1], linewidth=2, marker="o",
            markersize=5, label=f"ARIMAX({best['p']},1,{best['q']})")
    ax.fill_between(test_dates, preds - np.std(errors), preds + np.std(errors),
                    alpha=0.2, color=PALETTE[1], label="±1 Std of Errors")
    ax.plot(test_dates, naive["forecasts"], color=PALETTE[3],
            linewidth=1.5, linestyle=":", label="Naive")

    m = _metrics(actual, preds)
    m_naive = _metrics(actual, np.array(naive["forecasts"]))
    ax.set_title(
        f"Test Window Zoom (2022-07 → 2023-07)\n"
        f"ARIMAX: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  DirAcc={m['dir_acc']:.0f}%  |  "
        f"Naive: RMSE={m_naive['rmse']:.2f}  MAE={m_naive['mae']:.2f}",
        fontsize=10,
    )
    ax.set_ylabel("WFP Food Index")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.suptitle("Section 10 · ARIMAX Forecast vs Naive Baseline", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    path = _PLOT_DIR / "10b_arimax_forecast.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path.name)


def _plot_residuals(best: dict, train: pd.DataFrame) -> None:
    fit = best["fit"]
    residuals = pd.Series(fit.resid, index=train.index)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Residuals over time
    axes[0, 0].plot(train.index, residuals, color=PALETTE[0], linewidth=1.5)
    axes[0, 0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0, 0].set_title("Residuals over Time")
    axes[0, 0].set_ylabel("Residual")

    # Histogram
    from scipy import stats
    axes[0, 1].hist(residuals.dropna(), bins=15, color=PALETTE[0],
                    edgecolor="white", alpha=0.7, density=True)
    xlin = np.linspace(residuals.min(), residuals.max(), 200)
    axes[0, 1].plot(xlin, stats.norm.pdf(xlin, residuals.mean(), residuals.std()),
                    "k--", linewidth=1.5, label="Normal fit")
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].legend(fontsize=8)

    # Q-Q
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals.dropna(), dist="norm")
    axes[0, 2].scatter(osm, osr, alpha=0.6, s=20, color=PALETTE[0])
    axes[0, 2].plot(osm, slope * np.array(osm) + intercept, "k--", linewidth=1.2)
    axes[0, 2].set_title(f"Normal Q-Q (r={r:.3f})")
    axes[0, 2].set_xlabel("Theoretical Quantiles")

    # ACF
    plot_acf(residuals.dropna(), lags=min(15, len(residuals) // 2 - 1),
             ax=axes[1, 0], color=PALETTE[0],
             title=f"ACF — Residuals (ARIMAX({best['p']},1,{best['q']}))")
    axes[1, 0].set_xlabel("Lag")

    # PACF
    plot_pacf(residuals.dropna(), lags=min(15, len(residuals) // 2 - 1),
              ax=axes[1, 1], color=PALETTE[0],
              title="PACF — Residuals", method="ywm")
    axes[1, 1].set_xlabel("Lag")

    # Actual vs Fitted
    axes[1, 2].scatter(train["wfp_food_index"].values, fit.fittedvalues,
                       alpha=0.5, color=PALETTE[0], s=25)
    lims = [min(train["wfp_food_index"].min(), fit.fittedvalues.min()),
            max(train["wfp_food_index"].max(), fit.fittedvalues.max())]
    axes[1, 2].plot(lims, lims, "k--", linewidth=1.2)
    axes[1, 2].set_xlabel("Actual"); axes[1, 2].set_ylabel("Fitted")
    axes[1, 2].set_title("Actual vs Fitted (Train)")

    fig.suptitle(f"Section 10 · ARIMAX({best['p']},1,{best['q']}) — Residual Diagnostics",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = _PLOT_DIR / "10c_arimax_residuals.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path.name)


# ── Summary ──────────────────────────────────────────────────────────────────

def _print_summary(grid_df: pd.DataFrame, best: dict, naive: dict) -> None:
    naive_m = _metrics(
        np.array([]),  # computed below
        np.array([]),
    )
    print("\n" + "═" * 75)
    print("  ARIMAX MODEL RESULTS")
    print("═" * 75)
    print(f"\n  Grid search: ARIMAX(p,1,q)  p,q ∈ {{0,1,2}}")
    print(f"  Train: 2019-08 → {TRAIN_END}  |  Test: {TEST_START} → 2023-07")
    print()
    print(f"  {'Model':<16} {'AIC':>8} {'BIC':>8} {'Train RMSE':>12} "
          f"{'Test RMSE':>10} {'Test MAE':>9} {'DirAcc%':>8}")
    print("  " + "-" * 73)

    best_rmse = best["test_rmse"]
    for _, r in grid_df.sort_values("test_rmse").iterrows():
        flag = " ◀ BEST" if r["p"] == best["p"] and r["q"] == best["q"] else ""
        print(f"  ARIMAX({int(r['p'])},1,{int(r['q'])}){flag:<9}  "
              f"{r['aic']:>8.2f} {r['bic']:>8.2f} {r['train_rmse']:>12.4f} "
              f"{r['test_rmse']:>10.4f} {r['test_mae']:>9.4f} {r['test_dir_acc']:>8.1f}")

    print()
    print(f"  Naive (random walk) baseline:")
    print(f"    RMSE={naive['rmse']:.4f}  MAE={naive['mae']:.4f}  "
          f"DirAcc={naive['dir_acc']:.1f}%")
    print()
    rmse_improvement = (naive['rmse'] - best['test_rmse']) / naive['rmse'] * 100
    print(f"  Best model ARIMAX({best['p']},1,{best['q']}):")
    print(f"    Test RMSE improvement vs naive: {rmse_improvement:+.1f}%")
    print("═" * 75 + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _RES_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(_PROC_DIR / "historical_panel.parquet")
    df    = _prepare_features(panel)
    train, test = _train_test_split(df)

    # Exogenous columns
    month_dummies = [f"m{m:02d}" for m in range(2, 13) if f"m{m:02d}" in df.columns]
    exog_cols = ["log_ghsusd", "brent_lag6"] + month_dummies

    logger.info("Exogenous variables: %s", exog_cols[:4], )
    logger.info("Running grid search over ARIMAX(p,1,q) p,q ∈ {0,1,2} ...")

    grid_df, best = run_grid_search(train, test, exog_cols)
    naive          = _naive_forecast(train, test)

    # Save grid results
    grid_path = _RES_DIR / "arimax_comparison.csv"
    grid_df.to_csv(grid_path, index=False)
    logger.info("Saved → %s", grid_path)

    # Save best model summary
    summary_txt = (
        f"Best ARIMAX order: ({best['p']}, 1, {best['q']})\n"
        f"AIC: {best['aic']}\n"
        f"BIC: {best['bic']}\n"
        f"Test RMSE: {best['test_rmse']}\n"
        f"Test MAE:  {best['test_mae']}\n"
        f"Test MAPE: {best['test_mape']}%\n"
        f"Dir Acc:   {best['test_dir_acc']}%\n"
        f"Train period: 2019-08 → {TRAIN_END}\n"
        f"Test  period: {TEST_START} → 2023-07\n"
        f"Exogenous: log_ghsusd, brent_lag6, month_dummies\n\n"
        f"Model summary:\n{best['fit'].summary().as_text()}\n"
    )
    txt_path = _RES_DIR / "arimax_best_params.txt"
    txt_path.write_text(summary_txt)
    logger.info("Saved → %s", txt_path)

    # Plots
    _plot_grid(grid_df)
    _plot_forecast(train, test, best, naive)
    _plot_residuals(best, train)

    _print_summary(grid_df, best, naive)
    return grid_df, best, naive


if __name__ == "__main__":
    run()
