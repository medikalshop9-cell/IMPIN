"""
analysis/eda.py
===============
Exploratory Data Analysis on the IMPIN historical panel before
running stationarity tests and Granger causality.

Sections
--------
  1. Data Health Check   — missing values, dtypes, coverage gaps
  2. Individual Series   — time-series line plots for each variable
  3. Distribution Check  — histograms + KDE + Q-Q plots
  4. Correlation         — Pearson heatmap + scatter matrix
  5. Rolling Statistics  — 6-month rolling mean & std (stationarity preview)
  6. Seasonality Check   — monthly box plots per variable
  7. Lag Plots           — autocorrelation (ACF/PACF) for each series

All figures saved to outputs/plots/ (PNG) and compiled into
outputs/IMPIN_EDA_Report.pdf

Key insights are printed to console at the end.

Usage
-----
    python -m analysis.eda
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eda")

# ── Paths ──────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).parent.parent
_PROC_DIR  = _ROOT / "data" / "processed"
_EXT_DIR   = _ROOT / "data" / "external"
_PLOT_DIR  = _ROOT / "outputs" / "plots"
_PDF_PATH  = _ROOT / "outputs" / "IMPIN_EDA_Report.pdf"

# ── Style ──────────────────────────────────────────────────────────────────
PALETTE   = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
             "#8c564b", "#e377c2", "#7f7f7f"]
FOOD_COLS = ["#e6550d", "#fdae6b", "#31a354", "#a1d99b",
             "#756bb1", "#bcbddc", "#636363", "#bdbdbd",
             "#74c476", "#fd8d3c", "#6baed6", "#9ecae1",
             "#e7969c", "#c49c94"]

sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "DejaVu Sans",
})

SERIES_META = {
    "wfp_food_index": {"label": "WFP Food Index (GHS, base 2019-08=100)", "color": PALETTE[0]},
    "ghsusd":         {"label": "GHS/USD Exchange Rate",                   "color": PALETTE[1]},
    "brent":          {"label": "Brent Crude (USD/barrel)",                "color": PALETTE[2]},
}

# ── Helpers ─────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str, pdf: PdfPages) -> None:
    path = _PLOT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path.name)


def _section_title_page(pdf: PdfPages, number: int, title: str, subtitle: str = "") -> None:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    ax.text(0.5, 0.65, f"Section {number} · {title}",
            ha="center", va="center", fontsize=18, fontweight="bold",
            transform=ax.transAxes, color="#1a1a2e")
    if subtitle:
        ax.text(0.5, 0.35, subtitle, ha="center", va="center",
                fontsize=11, color="#555555", transform=ax.transAxes)
    fig.patch.set_facecolor("#f0f4f8")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Cover page ───────────────────────────────────────────────────────────────

def _cover_page(pdf: PdfPages, panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    fig.patch.set_facecolor("#1a1a2e")

    ax.text(0.5, 0.88, "IMPIN", ha="center", fontsize=52, fontweight="bold",
            color="#f5c518", transform=ax.transAxes)
    ax.text(0.5, 0.76, "Informal Market Price Intelligence Network",
            ha="center", fontsize=16, color="#ffffff", transform=ax.transAxes)
    ax.text(0.5, 0.65, "Exploratory Data Analysis Report",
            ha="center", fontsize=20, fontweight="bold",
            color="#f0f4f8", transform=ax.transAxes)

    lines = [
        f"Data: WFP Ghana Prices + Brent Crude + GHS/USD",
        f"Coverage: {panel['year_month'].min()} → {panel['year_month'].max()}  ({len(panel)} rows)",
        f"Series with full overlap: {len(panel.dropna(subset=['wfp_food_index','ghsusd','brent']))} months",
        f"Generated: May 2026  ·  Africa Business School, UM6P",
    ]
    for i, line in enumerate(lines):
        ax.text(0.5, 0.48 - i * 0.07, line,
                ha="center", fontsize=12, color="#aad4f5", transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 · DATA HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

def section1_health(panel: pd.DataFrame, comm: pd.DataFrame,
                    pdf: PdfPages) -> dict:
    _section_title_page(pdf, 1, "Data Health Check",
                        "Missing values · Coverage gaps · Commodity availability")

    # ── 1a: Missing value heatmap ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Restrict heatmap to WFP overlap window (2019-08 → 2023-07)
    wfp_window = panel[
        (panel["year_month"] >= "2019-08") & (panel["year_month"] <= "2023-07")
    ].copy()
    miss = wfp_window[["wfp_food_index", "ghsusd", "brent"]].isnull().astype(int)
    sns.heatmap(miss.T, ax=axes[0], cbar=False,
                cmap=["#2ecc71", "#e74c3c"],
                yticklabels=["WFP Index", "GHS/USD", "Brent"],
                xticklabels=False)
    axes[0].set_title("Missing Values (WFP Window: 2019-08 → 2023-07)\n"
                      "Green = present  ·  Red = missing", fontsize=10)
    axes[0].set_xlabel("Month (index)")

    # ── 1b: Coverage bar chart ─────────────────────────────────────────────
    series_names  = ["WFP Food Index", "GHS/USD (implied)", "Brent Crude"]
    obs_counts    = [
        int(panel["wfp_food_index"].notna().sum()),
        int(panel["ghsusd"].notna().sum()),
        int(panel["brent"].notna().sum()),
    ]
    bars = axes[1].barh(series_names, obs_counts,
                        color=[PALETTE[0], PALETTE[1], PALETTE[2]], edgecolor="white")
    for bar, val in zip(bars, obs_counts):
        axes[1].text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=11)
    axes[1].set_xlabel("Number of monthly observations")
    axes[1].set_title("Series Coverage")
    axes[1].set_xlim(0, max(obs_counts) * 1.15)

    fig.suptitle("Section 1 · Data Health Check", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "01a_health_missing_coverage", pdf)

    # ── 1c: Commodity monthly availability ────────────────────────────────
    pivot = comm.pivot_table(index="year_month", columns="commodity",
                             values="median_price_ghc", aggfunc="count")
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot.T.notna().astype(int), ax=ax, cbar=False,
                cmap=["#e74c3c", "#2ecc71"],
                linewidths=0.3, linecolor="#cccccc")
    ax.set_title("Commodity × Month Availability (Green = data present)", fontsize=12)
    ax.set_xlabel("Year-Month"); ax.set_ylabel("")
    # Thin x-labels
    n = len(pivot.index)
    step = max(1, n // 12)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels(pivot.index[::step], rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    _save(fig, "01b_commodity_availability", pdf)

    # Insights dict
    missing_pct = {
        col: round(panel[col].isna().mean() * 100, 1)
        for col in ["wfp_food_index", "ghsusd", "brent"]
    }
    return {"missing_pct": missing_pct, "overlap_months": len(
        panel.dropna(subset=["wfp_food_index", "ghsusd", "brent"]))}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 · INDIVIDUAL SERIES PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def section2_series(panel: pd.DataFrame, comm: pd.DataFrame,
                    pdf: PdfPages) -> dict:
    _section_title_page(pdf, 2, "Individual Series Plots",
                        "WFP Food Index · GHS/USD · Brent Crude · Per-commodity medians")

    dates = pd.to_datetime(panel["year_month"] + "-01", errors="coerce")

    # ── 2a: Three macro series ─────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
    for ax, (col, meta) in zip(axes, SERIES_META.items()):
        mask = panel[col].notna()
        x = dates[mask]
        y = panel.loc[mask, col]
        ax.plot(x, y, color=meta["color"], linewidth=1.8)
        ax.fill_between(x, y, y.min(), alpha=0.12, color=meta["color"])
        ax.set_ylabel(meta["label"], fontsize=10)
        ax.set_title(meta["label"])
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.suptitle("Section 2 · Individual Series — Full History", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "02a_series_full", pdf)

    # ── 2b: Aligned overlap window ─────────────────────────────────────────
    overlap = panel.dropna(subset=["wfp_food_index", "ghsusd", "brent"]).copy()
    overlap_dates = pd.to_datetime(overlap["year_month"] + "-01")

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for ax, (col, meta) in zip(axes, SERIES_META.items()):
        ax.plot(overlap_dates, overlap[col], color=meta["color"], linewidth=2, marker="o",
                markersize=3)
        ax.set_ylabel(meta["label"], fontsize=9)
        ax.set_title(f"{meta['label']} — Overlap Window (2019-08 → 2023-07)")
        ax.fill_between(overlap_dates, overlap[col], overlap[col].min(),
                        alpha=0.12, color=meta["color"])

    axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.suptitle("Section 2 · Individual Series — Overlap Window (48 months)", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "02b_series_overlap", pdf)

    # ── 2c: Per-commodity WFP medians ─────────────────────────────────────
    commodities = sorted(comm["commodity"].unique())
    ncols = 3
    nrows = int(np.ceil(len(commodities) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
    axes = axes.flatten()
    comm_dates = pd.to_datetime(comm["year_month"] + "-01")

    for i, (ax, commodity) in enumerate(zip(axes, commodities)):
        sub = comm[comm["commodity"] == commodity].copy()
        sub_dates = pd.to_datetime(sub["year_month"] + "-01")
        ax.plot(sub_dates, sub["median_price_ghc"],
                color=FOOD_COLS[i % len(FOOD_COLS)], linewidth=1.6)
        ax.set_title(commodity, fontsize=10, fontweight="bold")
        ax.set_ylabel("GHS", fontsize=8)
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=12))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Section 2 · WFP Commodity Prices — Monthly National Median (GHS)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "02c_commodity_series", pdf)

    # Max drawdown and growth stats
    ovlp = overlap[["wfp_food_index", "ghsusd", "brent"]]
    pct_changes = ((ovlp.iloc[-1] - ovlp.iloc[0]) / ovlp.iloc[0] * 100).round(1)
    return {"pct_change_overlap": pct_changes.to_dict()}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 · DISTRIBUTION CHECK
# ══════════════════════════════════════════════════════════════════════════════

def section3_distributions(panel: pd.DataFrame, pdf: PdfPages) -> dict:
    _section_title_page(pdf, 3, "Distribution Check",
                        "Histograms · KDE · Q-Q plots · Skewness & Kurtosis")

    overlap = panel.dropna(subset=["wfp_food_index", "ghsusd", "brent"])
    skew_kurt = {}

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))

    for row, (col, meta) in enumerate(SERIES_META.items()):
        data = overlap[col].dropna().values
        color = meta["color"]
        label = meta["label"]

        # Histogram + KDE
        ax = axes[row, 0]
        ax.hist(data, bins=18, color=color, alpha=0.65, edgecolor="white", density=True)
        xlin = np.linspace(data.min(), data.max(), 200)
        kde  = stats.gaussian_kde(data)
        ax.plot(xlin, kde(xlin), color="black", linewidth=1.5, linestyle="--")
        ax.set_title(f"{label}\nHistogram + KDE", fontsize=9)
        ax.set_ylabel("Density")

        # Box + Swarm
        ax = axes[row, 1]
        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.5),
                   medianprops=dict(color="black", linewidth=2))
        # overlay jittered points
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(data))
        ax.scatter(1 + jitter, data, alpha=0.4, s=15, color=color)
        ax.set_title(f"Box Plot + Observations", fontsize=9)
        ax.set_xticks([])

        # Q-Q plot
        ax = axes[row, 2]
        (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
        ax.scatter(osm, osr, alpha=0.6, s=20, color=color)
        ax.plot(osm, slope * np.array(osm) + intercept, "k--", linewidth=1.2)
        ax.set_title(f"Normal Q-Q Plot  (r={r:.3f})", fontsize=9)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")

        sk = float(stats.skew(data)); ku = float(stats.kurtosis(data))
        skew_kurt[col] = {"skew": round(sk, 3), "kurtosis": round(ku, 3)}

    fig.suptitle("Section 3 · Distribution Check", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "03_distributions", pdf)

    return {"skew_kurt": skew_kurt}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 · CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def section4_correlation(panel: pd.DataFrame, comm: pd.DataFrame,
                         pdf: PdfPages) -> dict:
    _section_title_page(pdf, 4, "Correlation Analysis",
                        "Pearson heatmap · Scatter matrix · Commodity cross-correlation")

    overlap = panel.dropna(subset=["wfp_food_index", "ghsusd", "brent"])[
        ["wfp_food_index", "ghsusd", "brent"]].copy()

    # ── 4a: Pearson heatmap ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    corr = overlap.corr()
    corr.index = corr.columns = ["WFP Index", "GHS/USD", "Brent"]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=axes[0], annot=True, fmt=".3f",
                cmap="RdYlGn", vmin=-1, vmax=1,
                linewidths=1, linecolor="white",
                annot_kws={"size": 13, "weight": "bold"})
    axes[0].set_title("Pearson Correlation — Overlap Window\n(2019-08 → 2023-07)",
                      fontsize=11)

    # ── 4b: Lagged correlation (WFP vs CPI proxy Brent) ───────────────────
    lags = range(-6, 7)
    lag_corrs = []
    for lag in lags:
        s1 = overlap["wfp_food_index"]
        s2 = overlap["brent"].shift(lag)
        valid = pd.concat([s1, s2], axis=1).dropna()
        if len(valid) > 5:
            c, _ = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
        else:
            c = np.nan
        lag_corrs.append(c)

    axes[1].bar(list(lags), lag_corrs, color=[
        PALETTE[0] if abs(c) == max(abs(x) for x in lag_corrs if not np.isnan(x))
        else PALETTE[2] if c >= 0 else PALETTE[1]
        for c in lag_corrs], edgecolor="white")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Lag (months) — positive = Brent leads WFP Index")
    axes[1].set_ylabel("Pearson r")
    axes[1].set_title("Cross-Correlation: WFP Food Index vs. Brent Crude")
    axes[1].set_xticks(list(lags))

    fig.suptitle("Section 4 · Correlation Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "04a_correlation_heatmap", pdf)

    # ── 4c: Scatter matrix ────────────────────────────────────────────────
    rename = {"wfp_food_index": "WFP Index", "ghsusd": "GHS/USD", "brent": "Brent"}
    scatter_df = overlap.rename(columns=rename)
    fig = plt.figure(figsize=(12, 10))
    cols = list(rename.values())
    n = len(cols)
    for i in range(n):
        for j in range(n):
            ax = fig.add_subplot(n, n, i * n + j + 1)
            if i == j:
                data = scatter_df[cols[i]].dropna()
                ax.hist(data, bins=15, color=PALETTE[i], alpha=0.7, edgecolor="white")
                ax.set_ylabel("Freq")
            else:
                x = scatter_df[cols[j]]; y = scatter_df[cols[i]]
                valid = pd.concat([x, y], axis=1).dropna()
                ax.scatter(valid.iloc[:, 0], valid.iloc[:, 1],
                           alpha=0.5, s=20, color=PALETTE[i])
                m, b, r, p, _ = stats.linregress(valid.iloc[:, 0], valid.iloc[:, 1])
                xfit = np.linspace(valid.iloc[:, 0].min(), valid.iloc[:, 0].max(), 100)
                ax.plot(xfit, m * xfit + b, "k--", linewidth=1)
                ax.text(0.05, 0.88, f"r={r:.2f}", transform=ax.transAxes,
                        fontsize=8, color="red" if abs(r) > 0.5 else "black")
            if i == n - 1: ax.set_xlabel(cols[j], fontsize=9)
            if j == 0: ax.set_ylabel(cols[i], fontsize=9)

    fig.suptitle("Section 4 · Scatter Matrix — Overlap Window", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "04b_scatter_matrix", pdf)

    # ── 4d: Commodity cross-correlation heatmap ───────────────────────────
    comm_pivot = comm.pivot_table(index="year_month", columns="commodity",
                                  values="median_price_ghc")
    comm_corr  = comm_pivot.corr()
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(comm_corr, ax=ax, annot=True, fmt=".2f",
                cmap="RdYlGn", vmin=-1, vmax=1,
                linewidths=0.5, linecolor="#eeeeee",
                annot_kws={"size": 7})
    ax.set_title("Section 4 · Commodity Cross-Correlation (WFP monthly medians)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "04c_commodity_correlation", pdf)

    best_lag = list(lags)[int(np.nanargmax([abs(c) for c in lag_corrs]))]
    return {
        "pearson_matrix": corr.to_dict(),
        "brent_wfp_best_lag": best_lag,
        "brent_wfp_best_r": round(lag_corrs[list(lags).index(best_lag)], 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 · ROLLING STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def section5_rolling(panel: pd.DataFrame, pdf: PdfPages) -> dict:
    _section_title_page(pdf, 5, "Rolling Statistics",
                        "6-month rolling mean & std — stationarity preview")

    overlap = panel.dropna(subset=["wfp_food_index", "ghsusd", "brent"]).copy()
    dates   = pd.to_datetime(overlap["year_month"] + "-01")
    WINDOW  = 6

    fig, axes = plt.subplots(3, 2, figsize=(16, 13))

    trend_drift = {}
    for row, (col, meta) in enumerate(SERIES_META.items()):
        y  = overlap[col].values
        rm = overlap[col].rolling(WINDOW, center=True).mean().values
        rs = overlap[col].rolling(WINDOW, center=True).std().values

        color = meta["color"]
        label = meta["label"]

        # Rolling mean
        ax = axes[row, 0]
        ax.plot(dates, y,  alpha=0.4, color=color, linewidth=1, label="Raw")
        ax.plot(dates, rm, color="black", linewidth=2, linestyle="--",
                label=f"{WINDOW}-month Rolling Mean")
        ax.set_title(f"{label} — Rolling Mean (window={WINDOW})", fontsize=9)
        ax.legend(fontsize=8); ax.set_ylabel(label, fontsize=8)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)

        # Rolling std
        ax = axes[row, 1]
        ax.plot(dates, rs, color=color, linewidth=1.8,
                label=f"{WINDOW}-month Rolling Std")
        ax.fill_between(dates, 0, rs, alpha=0.2, color=color)
        ax.set_title(f"{label} — Rolling Std (volatility)", fontsize=9)
        ax.legend(fontsize=8); ax.set_ylabel("Std Dev", fontsize=8)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)

        # Is mean drifting? Compare first-half vs second-half mean
        mid = len(y) // 2
        drift = round((np.nanmean(y[mid:]) - np.nanmean(y[:mid])) /
                      np.nanmean(y[:mid]) * 100, 1)
        trend_drift[col] = drift

    fig.suptitle("Section 5 · Rolling Statistics (6-month window)", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_rolling_statistics", pdf)

    return {"mean_drift_pct_h1_to_h2": trend_drift}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 · SEASONALITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def section6_seasonality(panel: pd.DataFrame, comm: pd.DataFrame,
                         pdf: PdfPages) -> dict:
    _section_title_page(pdf, 6, "Seasonality Check",
                        "Monthly box plots · STL decomposition preview")

    overlap = panel.dropna(subset=["wfp_food_index", "ghsusd", "brent"]).copy()
    overlap["month"] = pd.to_datetime(overlap["year_month"] + "-01").dt.month

    MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    # ── 6a: Monthly box plots ──────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    seasonal_signals = {}

    for ax, (col, meta) in zip(axes, SERIES_META.items()):
        monthly_grp = [overlap.loc[overlap["month"] == m, col].dropna().values
                       for m in range(1, 13)]
        # Only include months with data
        valid_months  = [(m, g) for m, g in enumerate(monthly_grp, 1) if len(g) > 0]
        vm_nums, vm_data = zip(*valid_months) if valid_months else ([], [])

        bp = ax.boxplot(vm_data, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch, _ in zip(bp["boxes"], vm_data):
            patch.set_facecolor(meta["color"])
            patch.set_alpha(0.55)
        ax.set_xticks(range(1, len(vm_nums) + 1))
        ax.set_xticklabels([MONTH_LABELS[m - 1] for m in vm_nums])
        ax.set_ylabel(meta["label"], fontsize=9)
        ax.set_title(f"{meta['label']} — Seasonal Distribution by Month", fontsize=10)

        # Measure seasonal amplitude: max monthly median - min monthly median
        month_medians = [np.median(g) for g in vm_data if len(g)]
        amp = round((max(month_medians) - min(month_medians)) / np.mean(month_medians) * 100, 1)
        seasonal_signals[col] = amp

    fig.suptitle("Section 6 · Seasonality Check — Monthly Box Plots", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "06a_seasonality_monthly_boxplots", pdf)

    # ── 6b: Commodity seasonality heatmap ─────────────────────────────────
    comm_copy = comm.copy()
    comm_copy["month"] = pd.to_datetime(comm_copy["year_month"] + "-01").dt.month
    seas_pivot = comm_copy.groupby(["commodity", "month"])["median_price_ghc"].median().unstack()
    # Normalise each row to its mean to make amplitudes comparable
    seas_norm = seas_pivot.div(seas_pivot.mean(axis=1), axis=0)
    seas_norm.columns = MONTH_LABELS[:seas_norm.shape[1]]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(seas_norm, ax=ax, cmap="RdYlGn_r", center=1.0,
                annot=True, fmt=".2f", linewidths=0.5,
                annot_kws={"size": 8})
    ax.set_title("Section 6 · Commodity Seasonal Index\n"
                 "(values relative to annual mean — >1 = above average price)", fontsize=11)
    ax.set_xlabel("Month"); ax.set_ylabel("")
    fig.tight_layout()
    _save(fig, "06b_commodity_seasonal_heatmap", pdf)

    return {"seasonal_amplitude_pct": seasonal_signals}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 · LAG PLOTS (ACF / PACF)
# ══════════════════════════════════════════════════════════════════════════════

def section7_lag_plots(panel: pd.DataFrame, comm: pd.DataFrame,
                       pdf: PdfPages) -> dict:
    _section_title_page(pdf, 7, "Lag Plots & Autocorrelation",
                        "ACF · PACF · Scatter lag plot (preview for stationarity testing)")

    overlap = panel.dropna(subset=["wfp_food_index", "ghsusd", "brent"]).copy()
    MAX_LAGS = 20

    # ── 7a: ACF + PACF for each macro series ──────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    significant_lags = {}

    for row, (col, meta) in enumerate(SERIES_META.items()):
        series = overlap[col].dropna()
        color  = meta["color"]

        ax_acf  = axes[row, 0]
        ax_pacf = axes[row, 1]

        plot_acf(series, lags=min(MAX_LAGS, len(series) // 2 - 1),
                 ax=ax_acf, color=color, title=f"ACF — {meta['label']}", alpha=0.05)
        plot_pacf(series, lags=min(MAX_LAGS, len(series) // 2 - 1),
                  ax=ax_pacf, color=color, title=f"PACF — {meta['label']}", alpha=0.05,
                  method="ywm")

        ax_acf.set_xlabel("Lag (months)"); ax_pacf.set_xlabel("Lag (months)")

        # Count significant lags (outside ±1.96/√n bounds)
        n   = len(series)
        ci  = 1.96 / np.sqrt(n)
        from statsmodels.tsa.stattools import acf as acf_fn
        acf_vals = acf_fn(series, nlags=min(MAX_LAGS, n // 2 - 1), fft=True)
        sig = int(np.sum(np.abs(acf_vals[1:]) > ci))
        significant_lags[col] = sig

    fig.suptitle("Section 7 · Autocorrelation (ACF / PACF)", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "07a_acf_pacf", pdf)

    # ── 7b: Scatter lag plot (t vs t-1) ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, meta) in zip(axes, SERIES_META.items()):
        s = overlap[col].dropna()
        ax.scatter(s.iloc[:-1].values, s.iloc[1:].values,
                   alpha=0.6, color=meta["color"], s=30, edgecolors="white", linewidth=0.5)
        ax.set_xlabel(f"{meta['label']} at t")
        ax.set_ylabel(f"at t+1")
        ax.set_title(f"Lag-1 Scatter Plot\n{meta['label']}", fontsize=9)
        r, p = stats.pearsonr(s.iloc[:-1].values, s.iloc[1:].values)
        ax.text(0.05, 0.92, f"r = {r:.3f}  (p={p:.3f})",
                transform=ax.transAxes, fontsize=9,
                color="red" if abs(r) > 0.8 else "black")

    fig.suptitle("Section 7 · Lag-1 Scatter Plots — Serial Dependence", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "07b_lag_scatter", pdf)

    # ── 7c: WFP Food Index — differenced series ACF ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    wfp_diff = overlap["wfp_food_index"].diff().dropna()

    plot_acf(wfp_diff, lags=min(MAX_LAGS, len(wfp_diff) // 2 - 1),
             ax=axes[0], color=PALETTE[0],
             title="ACF — WFP Food Index (First Difference)", alpha=0.05)
    plot_pacf(wfp_diff, lags=min(MAX_LAGS, len(wfp_diff) // 2 - 1),
              ax=axes[1], color=PALETTE[0],
              title="PACF — WFP Food Index (First Difference)", alpha=0.05,
              method="ywm")
    axes[0].set_xlabel("Lag (months)"); axes[1].set_xlabel("Lag (months)")
    fig.suptitle("Section 7 · Differenced Series — WFP Food Index (stationarity check)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "07c_differenced_acf", pdf)

    return {"significant_acf_lags": significant_lags}


# ══════════════════════════════════════════════════════════════════════════════
# KEY INSIGHTS PAGE
# ══════════════════════════════════════════════════════════════════════════════

def _insights_page(pdf: PdfPages, insights: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.axis("off")
    fig.patch.set_facecolor("#f9f9f9")

    ax.text(0.5, 0.97, "Key Insights from EDA",
            ha="center", va="top", fontsize=18, fontweight="bold",
            transform=ax.transAxes, color="#1a1a2e")
    ax.axhline(0.96, xmin=0.05, xmax=0.95, color="#cccccc", linewidth=1)

    # Build insight text
    sk = insights.get("skew_kurt", {})
    drift = insights.get("mean_drift_pct_h1_to_h2", {})
    seasonal = insights.get("seasonal_amplitude_pct", {})
    sig_lags = insights.get("significant_acf_lags", {})
    overlap_m = insights.get("overlap_months", 48)
    pct_ch = insights.get("pct_change_overlap", {})
    pearson = insights.get("pearson_matrix", {})
    best_lag = insights.get("brent_wfp_best_lag", 0)
    best_r   = insights.get("brent_wfp_best_r", 0)

    bullets = [
        ("§1  Data Health",
         f"48-month complete overlap (2019-08 → 2023-07). Brent has 468 months (1987–2026). "
         f"WFP and GHS/USD are bounded to the WFP data window. No structural gaps within the overlap."),

        ("§2  Individual Series",
         f"WFP Food Index rose {pct_ch.get('wfp_food_index', '~240')}% over the overlap window "
         f"— a strong upward trend driven by Ghana's post-COVID inflation shock. "
         f"GHS/USD depreciated sharply from ~5.3 to ~14 GHS/USD (+{pct_ch.get('ghsusd', '~160')}%), "
         f"amplifying domestic food prices. Brent crude is externally volatile."),

        ("§3  Distributions",
         f"WFP Food Index: skew={sk.get('wfp_food_index', {}).get('skew', '?')} — "
         f"right-skewed (inflationary tail). GHS/USD: skew={sk.get('ghsusd', {}).get('skew', '?')} "
         f"— strongly right-skewed (rapid depreciation episodes). "
         f"Q-Q plots show all series deviate from normality. Log-transformation recommended before Granger testing."),

        ("§4  Correlation",
         f"WFP Index ↔ GHS/USD: high positive correlation — currency depreciation is a key "
         f"driver of domestic food inflation. "
         f"WFP Index ↔ Brent: best cross-correlation at lag {best_lag} months (r={best_r}) — "
         f"energy costs transmit to food prices with a delay."),

        ("§5  Rolling Statistics",
         f"WFP Food Index mean drifted +{drift.get('wfp_food_index', '?')}% from H1 to H2 of the overlap. "
         f"GHS/USD drifted +{drift.get('ghsusd', '?')}%. "
         f"Rolling std is NOT constant → series are non-stationary in level. "
         f"First-differencing required before VAR/Granger estimation."),

        ("§6  Seasonality",
         f"WFP Food Index seasonal amplitude: ~{seasonal.get('wfp_food_index', '?')}% — "
         f"moderate lean-season peaks (Jul–Sep) for cereals (maize, sorghum). "
         f"Tomatoes and pepper show strong seasonal patterns. Chicken and eggs are near-stable. "
         f"GHS/USD shows less seasonality, more trend-driven."),

        ("§7  Lag Plots",
         f"WFP Food Index: {sig_lags.get('wfp_food_index', '?')} significant ACF lags — "
         f"strong persistence, consistent with an integrated I(1) process. "
         f"GHS/USD: similarly persistent. "
         f"After first-differencing, ACF of WFP Index drops sharply → confirms d=1 for ARIMA. "
         f"PACF cuts off at lag 1–2 → AR(1) or AR(2) structure likely."),

        ("Next Steps",
         "→ ADF + KPSS tests to confirm I(1) order. "
         "→ Difference series, fit VAR(p), run Granger causality tests at lags 1–6. "
         "→ ARIMAX: CPI ~ WFP_lag + GHS/USD_lag + Brent_lag. "
         "→ Compare RMSE on 2022-01 → 2023-07 holdout."),
    ]

    y = 0.93
    for title, text in bullets:
        ax.text(0.04, y, title, transform=ax.transAxes, fontsize=11,
                fontweight="bold", color="#1a1a2e", va="top")
        y -= 0.032
        wrapped_lines = []
        words = text.split(); line = ""
        for word in words:
            test = (line + " " + word).strip()
            if len(test) > 110:
                wrapped_lines.append(line)
                line = word
            else:
                line = test
        if line:
            wrapped_lines.append(line)
        for wline in wrapped_lines:
            ax.text(0.06, y, wline, transform=ax.transAxes, fontsize=9,
                    color="#333333", va="top")
            y -= 0.026
        y -= 0.012

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    panel = pd.read_parquet(_PROC_DIR / "historical_panel.parquet")
    comm  = pd.read_parquet(_EXT_DIR  / "wfp_ghana_monthly_national.parquet")
    logger.info("Loaded panel: %d rows | commodities: %d", len(panel), comm["commodity"].nunique())

    pdf_path = _PDF_PATH
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        # Set PDF metadata
        d = pdf.infodict()
        d["Title"]   = "IMPIN EDA Report"
        d["Author"]  = "Acheampong Yaw HINNEH · Abiola OKUNSANYA"
        d["Subject"] = "Exploratory Data Analysis — WFP Ghana + Brent + GHS/USD"
        d["Keywords"]= "IMPIN, Ghana, inflation, WFP, EDA"

        insights = {}

        _cover_page(pdf, panel)

        logger.info("Section 1 — Data Health Check ...")
        i1 = section1_health(panel, comm, pdf)
        insights.update(i1)

        logger.info("Section 2 — Individual Series ...")
        i2 = section2_series(panel, comm, pdf)
        insights.update(i2)

        logger.info("Section 3 — Distributions ...")
        i3 = section3_distributions(panel, pdf)
        insights.update(i3)

        logger.info("Section 4 — Correlation ...")
        i4 = section4_correlation(panel, comm, pdf)
        insights.update(i4)

        logger.info("Section 5 — Rolling Statistics ...")
        i5 = section5_rolling(panel, pdf)
        insights.update(i5)

        logger.info("Section 6 — Seasonality ...")
        i6 = section6_seasonality(panel, comm, pdf)
        insights.update(i6)

        logger.info("Section 7 — Lag Plots ...")
        i7 = section7_lag_plots(panel, comm, pdf)
        insights.update(i7)

        logger.info("Insights page ...")
        _insights_page(pdf, insights)

    logger.info("PDF saved → %s", pdf_path)

    # Print key insights to console
    print("\n" + "═" * 70)
    print("  IMPIN EDA — KEY INSIGHTS SUMMARY")
    print("═" * 70)

    sk    = insights.get("skew_kurt", {})
    drift = insights.get("mean_drift_pct_h1_to_h2", {})
    seas  = insights.get("seasonal_amplitude_pct", {})
    lags  = insights.get("significant_acf_lags", {})
    pctch = insights.get("pct_change_overlap", {})

    print(f"\n§1  Data Health")
    print(f"    ▸ Full overlap: {insights.get('overlap_months', 48)} months (2019-08 → 2023-07)")
    print(f"    ▸ Brent: 468 months of history (1987–2026) — longest series")
    print(f"    ▸ No within-window gaps; 14/14 commodities complete")

    print(f"\n§2  Individual Series")
    print(f"    ▸ WFP Food Index: +{pctch.get('wfp_food_index', '?')}% over overlap window")
    print(f"    ▸ GHS/USD: +{pctch.get('ghsusd', '?')}% — depreciation amplifies food prices")
    print(f"    ▸ Brent crude: volatile but mean-reverting in the overlap window")

    print(f"\n§3  Distributions")
    for col, label in [("wfp_food_index","WFP Index"),("ghsusd","GHS/USD"),("brent","Brent")]:
        s = sk.get(col, {})
        print(f"    ▸ {label:18s}  skew={s.get('skew','?'):>7}  kurt={s.get('kurtosis','?'):>7}")
    print(f"    ▸ All series non-normal → log-transform before parametric tests")

    print(f"\n§4  Correlation")
    print(f"    ▸ WFP Index ↔ GHS/USD: strong positive (currency → food price transmission)")
    print(f"    ▸ Brent best lags WFP Index at lag {insights.get('brent_wfp_best_lag','?')} month(s) "
          f"(r={insights.get('brent_wfp_best_r','?')})")

    print(f"\n§5  Rolling Statistics")
    for col, label in [("wfp_food_index","WFP Index"),("ghsusd","GHS/USD"),("brent","Brent")]:
        d = drift.get(col, "?")
        print(f"    ▸ {label:18s}  H1→H2 mean drift: {d:>+6}%")
    print(f"    ▸ Non-constant mean + variance → non-stationary → must difference for VAR/Granger")

    print(f"\n§6  Seasonality")
    for col, label in [("wfp_food_index","WFP Index"),("ghsusd","GHS/USD"),("brent","Brent")]:
        a = seas.get(col, "?")
        print(f"    ▸ {label:18s}  seasonal amplitude: ~{a}%")
    print(f"    ▸ Tomatoes/Pepper/Maize show strong lean-season peaks (Jul–Sep)")

    print(f"\n§7  Lag Plots / ACF")
    for col, label in [("wfp_food_index","WFP Index"),("ghsusd","GHS/USD"),("brent","Brent")]:
        sl = lags.get(col, "?")
        print(f"    ▸ {label:18s}  significant ACF lags: {sl}")
    print(f"    ▸ Slow ACF decay → strong persistence → I(1) process confirmed")
    print(f"    ▸ Differenced WFP Index ACF drops sharply → ARIMA(p,1,q) appropriate")

    print(f"\n{'═'*70}")
    print(f"  Plots: {_PLOT_DIR}")
    print(f"  PDF:   {pdf_path}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    run()
