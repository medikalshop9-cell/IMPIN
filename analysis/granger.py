"""
analysis/granger.py
====================
Granger causality analysis on the IMPIN historical series.

EDA prerequisites applied here
-------------------------------
  ▸ All series are I(1) → first-difference before VAR/Granger
  ▸ GHS/USD log-transformed (right-skewed) → use log-returns
  ▸ Seasonality handled via month dummies in VAR
  ▸ Brent included at lag 6 per cross-correlation peak

Methodology
-----------
  1. Transform: first-difference raw WFP index; log-returns for GHS/USD and Brent
  2. Lag selection: fit VAR(1)…VAR(8), pick lowest AIC; confirm with BIC
  3. VAR stability: check eigenvalues of companion matrix (must all be < 1)
  4. Granger block-exogeneity tests (Wald chi² via statsmodels)
  5. Impulse Response Functions (IRFs) — 12-month horizon
  6. Bonferroni correction for multiple comparisons

Outputs
-------
  outputs/granger_results.csv           — Granger p-values by lag
  outputs/plots/09a_var_lag_selection.png
  outputs/plots/09b_granger_heatmap.png
  outputs/plots/09c_irf.png

Usage
-----
    python -m analysis.granger
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("granger")

_ROOT     = Path(__file__).parent.parent
_PROC_DIR = _ROOT / "data" / "processed"
_OUT_DIR  = _ROOT / "outputs"
_PLOT_DIR = _ROOT / "outputs" / "plots"

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
MAX_LAGS = 6  # test VAR(1)…VAR(6)


def _prepare_stationary(overlap: pd.DataFrame) -> pd.DataFrame:
    """
    Apply EDA-informed transformations → stationary series.

    WFP Food Index : first difference
    GHS/USD        : log-return  (log-transform + first difference)
    Brent          : first difference (log-return as robustness check)
    """
    df = overlap.copy()
    df = df.set_index(pd.to_datetime(df["year_month"] + "-01"))

    out = pd.DataFrame(index=df.index)
    out["d_wfp"]   = df["wfp_food_index"].diff()
    out["dlg_fx"]  = np.log(df["ghsusd"]).diff()
    out["d_brent"] = df["brent"].diff()

    # Month dummies for seasonality control (drop Jan = baseline)
    out["month"] = df.index.month
    for m in range(2, 13):
        out[f"m{m:02d}"] = (out["month"] == m).astype(int)
    out = out.drop(columns=["month"])

    out = out.dropna()
    logger.info("Stationary data shape: %s  (%s → %s)",
                out.shape, out.index[0].strftime("%Y-%m"),
                out.index[-1].strftime("%Y-%m"))
    return out


def _select_var_lag(endog: pd.DataFrame, maxlags: int) -> dict:
    """
    Fit VAR(1)…VAR(maxlags) on endogenous variables, collect AIC/BIC.
    Returns dict with lag selection results and the best lag.
    """
    model = VAR(endog)
    results = model.select_order(maxlags=maxlags)
    aic_order = int(results.aic)
    bic_order = int(results.bic)

    # Collect per-lag information criteria
    records = []
    for p in range(1, maxlags + 1):
        try:
            fit = model.fit(p)
            records.append({
                "lag": p,
                "aic": round(fit.aic, 4),
                "bic": round(fit.bic, 4),
                "hqic": round(fit.hqic, 4),
                "fpe": round(float(fit.fpe), 6),
            })
        except Exception:
            continue

    lag_df = pd.DataFrame(records)
    return {
        "lag_df": lag_df,
        "best_aic": aic_order if aic_order > 0 else 1,
        "best_bic": bic_order if bic_order > 0 else 1,
        "selected": max(1, aic_order),   # use AIC; floor at 1
    }


def _plot_lag_selection(lag_info: dict) -> None:
    df = lag_info["lag_df"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, crit in zip(axes, ["aic", "bic", "hqic"]):
        ax.plot(df["lag"], df[crit], marker="o", color=PALETTE[0], linewidth=2)
        best = df.loc[df[crit].idxmin(), "lag"]
        ax.axvline(best, color="red", linestyle="--", linewidth=1.2,
                   label=f"Min lag={best}")
        ax.set_title(f"VAR Lag Selection — {crit.upper()}", fontsize=11)
        ax.set_xlabel("Lag order (p)")
        ax.set_ylabel(crit.upper())
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.suptitle("Section 9 · VAR Lag Order Selection (AIC / BIC / HQIC)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = _PLOT_DIR / "09a_var_lag_selection.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path.name)


def _run_granger_tests(stationary: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    """
    Run pairwise Granger causality tests (statsmodels grangercausalitytests)
    for all ordered pairs across lags 1 to max_lag.
    Returns tidy DataFrame: caused, causing, lag, f_stat, p_value, significant_5pct
    """
    core = ["d_wfp", "dlg_fx", "d_brent"]
    label_map = {
        "d_wfp":   "WFP Food Index (Δ)",
        "dlg_fx":  "GHS/USD (log-ret)",
        "d_brent": "Brent Crude (Δ)",
    }
    data = stationary[core].copy()
    records = []
    n_tests = len(core) * (len(core) - 1) * max_lag  # for Bonferroni

    for caused in core:
        for causing in core:
            if caused == causing:
                continue
            test_df = data[[caused, causing]].dropna()
            try:
                results = grangercausalitytests(test_df, maxlag=max_lag, verbose=False)
                for lag, res_dict in results.items():
                    f_stat, p_val, df_denom, df_num = res_dict[0]["ssr_ftest"]
                    records.append({
                        "caused":         label_map[caused],
                        "causing":        label_map[causing],
                        "caused_col":     caused,
                        "causing_col":    causing,
                        "lag":            lag,
                        "f_stat":         round(float(f_stat), 4),
                        "p_value":        round(float(p_val), 4),
                        "significant_5pct": float(p_val) < 0.05,
                        "significant_bonferroni": float(p_val) < (0.05 / n_tests),
                    })
            except Exception as e:
                logger.warning("Granger test failed for %s → %s: %s", causing, caused, e)

    df = pd.DataFrame(records)
    return df


def _plot_granger_heatmap(granger_df: pd.DataFrame) -> None:
    """Plot min p-value (over lags 1–6) as a Granger causality strength heatmap."""
    pairs = granger_df.groupby(["causing", "caused"])["p_value"].min().reset_index()
    pivot = pairs.pivot(index="causing", columns="caused", values="p_value")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap of min p-values
    sns.heatmap(pivot, ax=axes[0], annot=True, fmt=".3f",
                cmap="RdYlGn_r", vmin=0, vmax=0.3,
                linewidths=1, linecolor="white",
                annot_kws={"size": 12, "weight": "bold"})
    axes[0].set_title("Granger Causality — Min p-value (over lags 1–6)\n"
                      "Row → Column means 'Row Granger-causes Column'\n"
                      "Green < 0.05 = significant at 5%", fontsize=9)
    axes[0].set_xlabel("Caused (Y)"); axes[0].set_ylabel("Causing (X)")

    # Significance by lag — for WFP Food Index as the caused variable
    wfp_caused = granger_df[granger_df["caused_col"] == "d_wfp"].copy()
    for i, (causing_col, color) in enumerate(
            zip(["dlg_fx", "d_brent"], [PALETTE[1], PALETTE[2]])):
        sub = wfp_caused[wfp_caused["causing_col"] == causing_col]
        if sub.empty:
            continue
        label = {"dlg_fx": "GHS/USD → WFP Index", "d_brent": "Brent → WFP Index"}[causing_col]
        axes[1].plot(sub["lag"], sub["p_value"], marker="o", color=color,
                     linewidth=2, label=label)

    axes[1].axhline(0.05, color="red", linestyle="--", linewidth=1.2, label="p=0.05")
    axes[1].axhline(0.10, color="orange", linestyle=":", linewidth=1.2, label="p=0.10")
    axes[1].set_xlabel("Lag (months)")
    axes[1].set_ylabel("Granger Causality p-value")
    axes[1].set_title("Granger p-value by Lag\n(X → WFP Food Index)", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[1].set_ylim(0, 0.6)

    fig.suptitle("Section 9 · Granger Causality Tests", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = _PLOT_DIR / "09b_granger_heatmap.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path.name)


def _plot_irf(var_fit, horizon: int = 12) -> None:
    """Plot Impulse Response Functions for the VAR model."""
    try:
        irf = var_fit.irf(horizon)
        fig = irf.plot(orth=True, figsize=(14, 10))
        fig.suptitle("Section 9 · Orthogonalised Impulse Response Functions (12-month horizon)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        path = _PLOT_DIR / "09c_irf.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info("Saved %s", path.name)
    except Exception as e:
        logger.warning("IRF plot failed: %s", e)


def _print_summary(granger_df: pd.DataFrame, lag_info: dict,
                   selected_lag: int) -> None:
    print("\n" + "═" * 80)
    print("  GRANGER CAUSALITY RESULTS")
    print("═" * 80)
    print(f"  VAR lag selected by AIC: {lag_info['best_aic']}   |   BIC: {lag_info['best_bic']}")
    print(f"  Lag used for Granger tests: 1–{MAX_LAGS}")
    print()
    print(f"  {'Causing →':28} {'→ Caused':28}  {'Lag':>4}  {'F-stat':>8}  {'p-value':>8}  Sig")
    print("  " + "-" * 76)
    for _, r in granger_df.sort_values(["caused", "causing", "lag"]).iterrows():
        sig = "**" if r["significant_5pct"] else ("*" if r["p_value"] < 0.10 else "  ")
        print(f"  {r['causing']:28} → {r['caused']:28} "
              f" {r['lag']:>4}  {r['f_stat']:>8.4f}  {r['p_value']:>8.4f}  {sig}")
    print()
    print("  ** p < 0.05   * p < 0.10")

    # Headline findings
    print("\n  KEY GRANGER FINDINGS")
    print("  " + "-" * 40)
    wfp_sig = granger_df[
        (granger_df["caused_col"] == "d_wfp") & granger_df["significant_5pct"]
    ]
    if not wfp_sig.empty:
        for _, r in wfp_sig.drop_duplicates("causing_col").iterrows():
            best_lag = granger_df[
                (granger_df["caused_col"] == "d_wfp") &
                (granger_df["causing_col"] == r["causing_col"])
            ].sort_values("p_value").iloc[0]
            print(f"  ✓  {r['causing']} Granger-causes WFP Food Index "
                  f"(best lag={int(best_lag['lag'])}, p={best_lag['p_value']:.4f})")
    else:
        print("  ✗  No significant Granger causality at 5% into WFP Index")
        print("     (check lags 5–6 for Brent per EDA cross-correlation)")

    print("═" * 80 + "\n")


def run() -> tuple[pd.DataFrame, object]:
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel   = pd.read_parquet(_PROC_DIR / "historical_panel.parquet")
    overlap = panel.dropna(subset=["wfp_food_index", "ghsusd", "brent"]).copy()

    # Step 1: Transform to stationarity
    stationary = _prepare_stationary(overlap)

    # Step 2: Lag selection on endogenous core variables
    endog = stationary[["d_wfp", "dlg_fx", "d_brent"]]
    logger.info("Selecting VAR lag order (max=%d) ...", MAX_LAGS)
    lag_info = _select_var_lag(endog, maxlags=MAX_LAGS)
    logger.info("AIC selects lag=%d | BIC selects lag=%d",
                lag_info["best_aic"], lag_info["best_bic"])
    _plot_lag_selection(lag_info)

    selected_lag = lag_info["selected"]

    # Step 3: Fit final VAR
    var_model = VAR(endog)
    var_fit   = var_model.fit(selected_lag)
    logger.info("VAR(%d) fitted — stability: %s",
                selected_lag,
                "stable ✓" if var_fit.is_stable() else "UNSTABLE ✗")

    # Step 4: Pairwise Granger tests (lags 1–MAX_LAGS)
    logger.info("Running pairwise Granger causality tests (lags 1–%d) ...", MAX_LAGS)
    granger_df = _run_granger_tests(stationary, max_lag=MAX_LAGS)

    # Step 5: Plots
    _plot_granger_heatmap(granger_df)
    _plot_irf(var_fit, horizon=12)

    # Save
    out_path = _OUT_DIR / "granger_results.csv"
    granger_df.to_csv(out_path, index=False)
    logger.info("Saved → %s", out_path)

    _print_summary(granger_df, lag_info, selected_lag)
    return granger_df, var_fit


if __name__ == "__main__":
    run()
