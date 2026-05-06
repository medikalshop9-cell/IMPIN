"""
analysis/stationarity.py
=========================
Formal stationarity tests on the IMPIN historical series, applying
EDA-informed transformations before Granger causality and ARIMAX.

Tests applied
-------------
  ADF  — Augmented Dickey-Fuller  (H₀: unit root present → non-stationary)
  KPSS — Kwiatkowski-Phillips-Schmidt-Shin  (H₀: series is stationary)

For each variable four versions are tested:
  (a) raw level
  (b) log-transformed level  (GHS/USD is strongly right-skewed)
  (c) first-difference of raw
  (d) first-difference of log  (= log-return)

Outputs
-------
  outputs/stationarity_report.csv   — full results table
  outputs/plots/08_stationarity.png — panel of test-statistic plots
  Console summary with integration orders confirmed

Usage
-----
    python -m analysis.stationarity
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stationarity")

_ROOT     = Path(__file__).parent.parent
_PROC_DIR = _ROOT / "data" / "processed"
_OUT_DIR  = _ROOT / "outputs"
_PLOT_DIR = _ROOT / "outputs" / "plots"

PALETTE = {"wfp_food_index": "#1f77b4", "ghsusd": "#d62728", "brent": "#2ca02c"}
SERIES_LABELS = {
    "wfp_food_index": "WFP Food Index",
    "ghsusd":         "GHS/USD (implied)",
    "brent":          "Brent Crude",
}

# ADF significance thresholds (MacKinnon critical values are returned by adfuller)
# KPSS critical values at 5%: 0.463 (trend), 0.146 (constant)
KPSS_CV_5PCT = {"c": 0.463, "ct": 0.146}


def _adf_test(series: pd.Series, regression: str = "c") -> dict:
    """Run ADF test. regression='c' (constant) or 'ct' (constant+trend)."""
    result = adfuller(series.dropna(), regression=regression, autolag="AIC")
    return {
        "stat":    round(result[0], 4),
        "p_value": round(result[1], 4),
        "lags":    int(result[2]),
        "cv_1pct": round(result[4]["1%"], 4),
        "cv_5pct": round(result[4]["5%"], 4),
        "reject_H0_5pct": result[1] < 0.05,   # True → stationary
    }


def _kpss_test(series: pd.Series, regression: str = "c") -> dict:
    """Run KPSS test. regression='c' (level) or 'ct' (trend)."""
    try:
        stat, p_value, lags, crit = kpss(series.dropna(), regression=regression, nlags="auto")
        cv_5pct = crit.get("5%", np.nan)
        return {
            "stat":    round(stat, 4),
            "p_value": round(p_value, 4),
            "lags":    int(lags),
            "cv_5pct": round(cv_5pct, 4),
            "reject_H0_5pct": stat > cv_5pct,  # True → non-stationary (reject stationarity)
        }
    except Exception as e:
        return {"stat": np.nan, "p_value": np.nan, "lags": 0,
                "cv_5pct": np.nan, "reject_H0_5pct": None, "error": str(e)}


def _interpret(adf_reject: bool, kpss_reject: bool) -> str:
    """
    Combine ADF and KPSS conclusions.
    ADF H₀: unit root (non-stationary) → reject = stationary
    KPSS H₀: stationary              → reject = non-stationary
    """
    if adf_reject and not kpss_reject:
        return "STATIONARY ✓"
    elif not adf_reject and kpss_reject:
        return "NON-STATIONARY ✗"
    elif adf_reject and kpss_reject:
        return "UNCERTAIN (trend-stationary?)"
    else:
        return "UNCERTAIN (weak evidence)"


def run_all_tests(overlap: pd.DataFrame) -> pd.DataFrame:
    """
    Run ADF + KPSS on four transformations of each series.
    Returns a tidy DataFrame with all results.
    """
    cols = ["wfp_food_index", "ghsusd", "brent"]
    records = []

    for col in cols:
        raw = overlap[col].dropna()
        log = np.log(raw)
        d1  = raw.diff().dropna()
        dlog = log.diff().dropna()

        transforms = {
            "Level (raw)":          raw,
            "Level (log)":          log,
            "First diff (raw)":     d1,
            "First diff (log-ret)": dlog,
        }

        for tfm_name, series in transforms.items():
            adf  = _adf_test(series, regression="c")
            kpss_ = _kpss_test(series, regression="c")
            conclusion = _interpret(adf["reject_H0_5pct"], kpss_["reject_H0_5pct"])

            records.append({
                "variable":     col,
                "label":        SERIES_LABELS[col],
                "transform":    tfm_name,
                "n_obs":        len(series),
                "adf_stat":     adf["stat"],
                "adf_p":        adf["p_value"],
                "adf_lags":     adf["lags"],
                "adf_cv5":      adf["cv_5pct"],
                "adf_stationary": adf["reject_H0_5pct"],
                "kpss_stat":    kpss_["stat"],
                "kpss_p":       kpss_["p_value"],
                "kpss_cv5":     kpss_["cv_5pct"],
                "kpss_nonstationary": kpss_["reject_H0_5pct"],
                "conclusion":   conclusion,
            })

    return pd.DataFrame(records)


def plot_transformations(overlap: pd.DataFrame, pdf_path=None) -> None:
    """Plot raw vs log vs diff vs log-diff for each series."""
    cols = ["wfp_food_index", "ghsusd", "brent"]
    dates = pd.to_datetime(overlap["year_month"] + "-01")

    fig, axes = plt.subplots(len(cols), 4, figsize=(20, 13))

    for row, col in enumerate(cols):
        raw  = overlap[col].values
        log_ = np.log(raw)
        d1   = np.diff(raw); d_dates = dates.iloc[1:]
        dlog = np.diff(log_)

        color = PALETTE[col]
        label = SERIES_LABELS[col]

        series_data = [
            (dates,   raw,   "Level (raw)"),
            (dates,   log_,  "Level (log)"),
            (d_dates, d1,    "First Diff (raw)"),
            (d_dates, dlog,  "First Diff (log-ret)"),
        ]

        for col_idx, (x, y, title) in enumerate(series_data):
            ax = axes[row, col_idx]
            ax.plot(x, y, color=color, linewidth=1.5)
            ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
            if row == 0:
                ax.set_title(title, fontsize=9, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)

    fig.suptitle("Section 8 · Stationarity Transformations — Raw / Log / Diff / Log-Diff",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = _PLOT_DIR / "08_stationarity_transforms.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path.name)


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "═" * 90)
    print("  STATIONARITY TEST RESULTS  (ADF + KPSS at 5% significance)")
    print("═" * 90)
    print(f"  {'Variable':<20} {'Transform':<24} {'ADF p':>7} {'KPSS p':>7}  {'Conclusion'}")
    print("  " + "-" * 86)
    for _, r in df.iterrows():
        adf_flag  = "✓" if r["adf_stationary"]  else "✗"
        kpss_flag = "✗" if r["kpss_nonstationary"] else "✓"
        print(f"  {r['label']:<20} {r['transform']:<24} "
              f"{r['adf_p']:>7.4f}{adf_flag} {r['kpss_p']:>7.4f}{kpss_flag}  "
              f"{r['conclusion']}")
    print("═" * 90)

    # Recommended integration orders
    print("\n  INTEGRATION ORDER RECOMMENDATIONS")
    print("  " + "-" * 50)
    for col in ["wfp_food_index", "ghsusd", "brent"]:
        level = df[(df["variable"] == col) & (df["transform"] == "Level (log)")].iloc[0]
        diff  = df[(df["variable"] == col) & (df["transform"] == "First diff (log-ret)")].iloc[0]
        level_stat = "stationary" if level["conclusion"].startswith("STAT") else "non-stationary"
        diff_stat  = "stationary" if diff["conclusion"].startswith("STAT") else "non-stationary"
        order = "I(1)" if not level["conclusion"].startswith("STAT") and \
                           diff["conclusion"].startswith("STAT") else "I(0) or check"
        print(f"  {SERIES_LABELS[col]:<22}  log-level={level_stat:<17}  "
              f"log-diff={diff_stat:<17}  → {order}")
    print()


def run() -> pd.DataFrame:
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel   = pd.read_parquet(_PROC_DIR / "historical_panel.parquet")
    overlap = panel.dropna(subset=["wfp_food_index", "ghsusd", "brent"]).copy()
    logger.info("Running stationarity tests on %d observations (2019-08 → 2023-07)", len(overlap))

    results = run_all_tests(overlap)
    plot_transformations(overlap)

    out_path = _OUT_DIR / "stationarity_report.csv"
    results.to_csv(out_path, index=False)
    logger.info("Saved → %s", out_path)

    print_summary(results)
    return results


if __name__ == "__main__":
    run()
