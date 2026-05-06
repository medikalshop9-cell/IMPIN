"""
anomaly/detector.py
===================
IMPIN Anomaly Detection — two layers:

Layer 1 — Cross-sectional price outliers (scraped_combined.csv)
    For each CPI category, flag products whose price is anomalous using
    three complementary methods:
        • Z-score       : |z| > Z_THRESH  (default 2.5)  on log(1+price)
        • IQR fence     : price < Q1 - k*IQR  or  > Q3 + k*IQR  (k=2.0)
        • Isolation Forest: contamination=0.05 per category
    A product is FLAGGED if ≥ VOTE_MIN (2) of the 3 methods agree.
    Flagged products are excluded before computing the clean IMPIN index.

Layer 2 — WFP monthly time-series anomalies
    Detect structural breaks / outlier months in the 48-month WFP food
    index using Z-score on month-over-month log-returns (|z| > Z_THRESH).
    These are informational — they explain nowcast model instability.

Outputs
-------
    data/processed/anomaly_flags.csv        — per-product flag table
    data/processed/wfp_anomaly_months.csv   — flagged WFP months
    data/processed/impin_clean.csv          — IMPIN recalculated after
                                              removing flagged products
    outputs/plots/19a_price_outliers.png    — box-plots per category + flags
    outputs/plots/19b_wfp_anomalies.png     — WFP series with anomaly markers
    outputs/plots/19c_impin_clean.png       — clean vs raw IMPIN by category
    outputs/anomaly_report.csv              — legacy combined report
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────
Z_THRESH   = 2.5
IQR_K      = 2.0
IF_CONTAM  = 0.05
MIN_OBS    = 5
VOTE_MIN   = 2
RANDOM_SEED = 42

ROOT   = Path(__file__).resolve().parent.parent
SCRAPE = ROOT / "data" / "raw" / "scraped_combined.csv"
WFP_PATH = ROOT / "data" / "processed" / "macro_panel_live.parquet"
PROC   = ROOT / "data" / "processed"
PLOTS  = ROOT / "outputs" / "plots"
OUT_CSV = ROOT / "outputs" / "anomaly_report.csv"
PLOTS.mkdir(parents=True, exist_ok=True)
(ROOT / "outputs").mkdir(parents=True, exist_ok=True)

CATEGORY_COLOURS = {
    "Food & Beverages":          "#2ca02c",
    "General":                   "#1f77b4",
    "Household":                 "#ff7f0e",
    "Personal Care":             "#9467bd",
    "Clothing & Personal Care":  "#d62728",
}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 helpers
# ══════════════════════════════════════════════════════════════════════════════

def _zscore_flag(prices, thresh):
    if len(prices) < 3:
        return np.zeros(len(prices), dtype=bool)
    mu, sd = np.mean(prices), np.std(prices)
    if sd < 1e-9:
        return np.zeros(len(prices), dtype=bool)
    return np.abs((prices - mu) / sd) > thresh

def _iqr_flag(prices, k):
    if len(prices) < 3:
        return np.zeros(len(prices), dtype=bool)
    q1, q3 = np.percentile(prices, 25), np.percentile(prices, 75)
    iqr = q3 - q1
    if iqr < 1e-9:
        return np.zeros(len(prices), dtype=bool)
    return (prices < q1 - k * iqr) | (prices > q3 + k * iqr)

def _if_flag(prices, contam):
    if len(prices) < MIN_OBS:
        return np.zeros(len(prices), dtype=bool)
    X = np.log1p(prices).reshape(-1, 1)
    clf = IsolationForest(contamination=contam, random_state=RANDOM_SEED)
    return clf.fit_predict(X) == -1

def detect_price_outliers(df):
    df = df.copy()
    df["log_price"]   = np.log1p(df["price_ghc"].clip(lower=0))
    df["flag_zscore"] = False
    df["flag_iqr"]    = False
    df["flag_if"]     = False
    df["z_score"]     = 0.0
    df["anomaly_score"] = 0.0

    for cat, grp in df.groupby("cpi_category"):
        idx = grp.index
        lp  = grp["log_price"].values
        df.loc[idx, "flag_zscore"] = _zscore_flag(lp, Z_THRESH)
        df.loc[idx, "flag_iqr"]    = _iqr_flag(lp, IQR_K)
        df.loc[idx, "flag_if"]     = _if_flag(lp, IF_CONTAM)
        mu, sd = lp.mean(), lp.std()
        if sd > 1e-9:
            df.loc[idx, "z_score"] = (lp - mu) / sd
        # Isolation Forest anomaly score (per category, for legacy report)
        if len(grp) >= MIN_OBS:
            X = lp.reshape(-1, 1)
            iso = IsolationForest(contamination=IF_CONTAM, random_state=RANDOM_SEED, n_estimators=200)
            df.loc[idx, "anomaly_score"] = -iso.fit(X).score_samples(X)

    df["vote_count"] = (df["flag_zscore"].astype(int)
                      + df["flag_iqr"].astype(int)
                      + df["flag_if"].astype(int))
    df["is_flagged"] = df["vote_count"] >= VOTE_MIN
    return df

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 helpers
# ══════════════════════════════════════════════════════════════════════════════

def detect_wfp_anomalies(panel):
    wfp = panel.dropna(subset=["wfp_food_index"]).sort_values("year_month").copy()
    wfp["log_ret"] = np.log(wfp["wfp_food_index"] / wfp["wfp_food_index"].shift(1))
    wfp = wfp.dropna(subset=["log_ret"])
    mu, sd = wfp["log_ret"].mean(), wfp["log_ret"].std()
    wfp["z_ret"]   = (wfp["log_ret"] - mu) / sd
    wfp["flagged"] = wfp["z_ret"].abs() > Z_THRESH
    return wfp[["year_month", "wfp_food_index", "log_ret", "z_ret", "flagged"]]

# ══════════════════════════════════════════════════════════════════════════════
# IMPIN clean index
# ══════════════════════════════════════════════════════════════════════════════

def compute_impin(df, label):
    rows = []
    for cat, grp in df.groupby("cpi_category"):
        rows.append({"category": cat,
                     "median_price_ghc": grp["price_ghc"].median(),
                     "n_products": len(grp), "label": label})
    out = pd.DataFrame(rows)
    agg = pd.DataFrame([{"category": "All (equal-weight)",
                          "median_price_ghc": out["median_price_ghc"].mean(),
                          "n_products": out["n_products"].sum(), "label": label}])
    return pd.concat([out, agg], ignore_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_price_outliers(df):
    cats = sorted(df["cpi_category"].unique())
    fig, axes = plt.subplots(1, len(cats), figsize=(4 * len(cats), 6))
    if len(cats) == 1:
        axes = [axes]
    fig.suptitle(f"Layer 1 — Cross-Sectional Price Outlier Detection\n"
                 f"(Z>±{Z_THRESH}, IQR×{IQR_K}, IF={IF_CONTAM:.0%}; flagged if ≥{VOTE_MIN}/3 agree)",
                 fontweight="bold", fontsize=11)
    rng = np.random.default_rng(42)
    for ax, cat in zip(axes, cats):
        grp  = df[df["cpi_category"] == cat]
        col  = CATEGORY_COLOURS.get(cat, "#1f77b4")
        ax.boxplot(grp["log_price"], widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor=col, alpha=0.3),
                   medianprops=dict(color="black", lw=2),
                   flierprops=dict(marker="", linestyle="none"))
        norm_idx = grp[~grp["is_flagged"]]
        flag_idx = grp[grp["is_flagged"]]
        jn = rng.uniform(-0.15, 0.15, len(norm_idx))
        ax.scatter(np.ones(len(norm_idx)) + jn, norm_idx["log_price"],
                   color=col, alpha=0.35, s=14, zorder=3)
        if len(flag_idx):
            jf = rng.uniform(-0.12, 0.12, len(flag_idx))
            ax.scatter(np.ones(len(flag_idx)) + jf, flag_idx["log_price"],
                       color="#d62728", s=45, zorder=5, marker="x",
                       linewidths=1.8, label=f"Flagged ({len(flag_idx)})")
            ax.legend(fontsize=8)
        ax.set_title(f"{cat}\n(n={len(grp)}, flagged={grp['is_flagged'].sum()})", fontsize=9)
        ax.set_ylabel("log(1+price GHC)" if ax == axes[0] else "")
        ax.set_xticks([])
    plt.tight_layout()
    out = PLOTS / "19a_price_outliers.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out

def plot_wfp_anomalies(wfp_flags, panel):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle(f"Layer 2 — WFP Monthly Food Index: Time-Series Anomaly Detection\n"
                 f"(Z-score on MoM log-returns, threshold ±{Z_THRESH}σ)",
                 fontweight="bold", fontsize=11)
    wfp_all = panel.dropna(subset=["wfp_food_index"]).sort_values("year_month")
    ax1.plot(wfp_all["year_month"], wfp_all["wfp_food_index"],
             color="#1f77b4", lw=2, label="WFP Food Index")
    anom = wfp_flags[wfp_flags["flagged"]]
    if len(anom):
        ax1.scatter(anom["year_month"], anom["wfp_food_index"],
                    color="#d62728", s=90, zorder=6, marker="v",
                    label=f"Anomaly (n={len(anom)})")
    ax1.set_ylabel("WFP Food Index", fontsize=10)
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    bar_cols = wfp_flags["flagged"].map({True: "#d62728", False: "#1f77b4"})
    ax2.bar(wfp_flags["year_month"], wfp_flags["z_ret"], color=bar_cols, width=20, alpha=0.7)
    ax2.axhline( Z_THRESH, color="#d62728", ls="--", lw=1.2, alpha=0.8, label=f"+{Z_THRESH}σ")
    ax2.axhline(-Z_THRESH, color="#d62728", ls="--", lw=1.2, alpha=0.8, label=f"−{Z_THRESH}σ")
    ax2.axhline(0, color="#666", lw=0.8, alpha=0.4)
    ax2.set_ylabel("Z-score (MoM log-return)", fontsize=10)
    ax2.set_xlabel("Month", fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    out = PLOTS / "19b_wfp_anomalies.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out

def plot_impin_clean(raw_impin, clean_impin):
    cats = [c for c in raw_impin["category"].unique() if c != "All (equal-weight)"]
    all_cats = cats + ["All (equal-weight)"]
    fig, axes = plt.subplots(1, len(all_cats), figsize=(3.5 * len(all_cats), 5))
    fig.suptitle("Layer 1 Impact — Raw vs Clean IMPIN (after removing flagged products)",
                 fontweight="bold", fontsize=11)
    for ax, cat in zip(axes, all_cats):
        rr = raw_impin[raw_impin["category"] == cat]
        cr = clean_impin[clean_impin["category"] == cat]
        if rr.empty or cr.empty:
            ax.set_visible(False); continue
        r_val = float(rr["median_price_ghc"].values[0])
        c_val = float(cr["median_price_ghc"].values[0])
        r_n   = int(rr["n_products"].values[0])
        c_n   = int(cr["n_products"].values[0])
        pct   = (c_val - r_val) / r_val * 100 if r_val > 0 else 0
        col   = CATEGORY_COLOURS.get(cat, "#1f77b4")
        bars  = ax.bar(["Raw", "Clean"], [r_val, c_val],
                       color=[col, "#2ca02c"], alpha=0.75, width=0.5)
        ax.bar_label(bars, fmt="GHC %.0f", fontsize=8, padding=3)
        title = cat if cat != "All (equal-weight)" else "All categories"
        ax.set_title(f"{title}\n(n: {r_n}→{c_n}  |  Δ{pct:+.1f}%)", fontsize=8.5)
        ax.set_ylabel("Median price (GHC)" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    plt.tight_layout()
    out = PLOTS / "19c_impin_clean.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("IMPIN ANOMALY DETECTOR")
    print("=" * 72)

    # ── Load ──────────────────────────────────────────────────────────────────
    df     = pd.read_csv(SCRAPE)
    df     = df.dropna(subset=["price_ghc"]).copy()
    df["price_ghc"] = pd.to_numeric(df["price_ghc"], errors="coerce")
    df     = df[df["price_ghc"] > 0].copy()
    panel  = pd.read_parquet(WFP_PATH)
    panel["year_month"] = pd.to_datetime(panel["year_month"])
    print(f"Products loaded   : {len(df):,}  ({df['cpi_category'].nunique()} categories, "
          f"{df['source'].nunique()} sources)")
    print(f"WFP months loaded : {panel['wfp_food_index'].notna().sum()}")

    # ── Layer 1 ───────────────────────────────────────────────────────────────
    print("\n── Layer 1: Cross-Sectional Price Outliers ───────────────────────────")
    flagged_df = detect_price_outliers(df)
    n_flagged  = flagged_df["is_flagged"].sum()
    print(f"Products flagged  : {n_flagged} / {len(flagged_df)}  ({n_flagged/len(flagged_df):.1%})")
    print(f"\n{'Category':<30} {'Total':>6} {'Flagged':>8} {'%':>6}  {'Z / IQR / IF':>14}")
    print("-" * 72)
    for cat, grp in flagged_df.groupby("cpi_category"):
        fl = grp["is_flagged"].sum()
        print(f"  {cat:<28} {len(grp):>6} {fl:>8} {fl/len(grp):>6.1%}  "
              f"Z={grp['flag_zscore'].sum()} / IQR={grp['flag_iqr'].sum()} / IF={grp['flag_if'].sum()}")

    top = (flagged_df[flagged_df["is_flagged"]]
           .nlargest(10, "z_score")[["product_name","cpi_category","source","price_ghc","z_score","vote_count"]])
    if len(top):
        print(f"\nTop flagged products:")
        for _, r in top.iterrows():
            print(f"  {r['product_name'][:42]:<42}  GHC {r['price_ghc']:>9,.0f}  z={r['z_score']:+.2f}  votes={r['vote_count']}")

    # Legacy report (kept for dashboard compat)
    report_cols = ["product_name","source","cpi_category","price_ghc","unit",
                   "z_score","anomaly_score","flag_zscore","flag_iqr","flag_if","is_flagged"]
    flagged_df[report_cols].sort_values("anomaly_score", ascending=False).to_csv(OUT_CSV, index=False)
    flagged_df.drop(columns=["log_price"], errors="ignore").to_csv(PROC / "anomaly_flags.csv", index=False)
    print(f"\nSaved → {OUT_CSV}")
    print(f"Saved → {PROC/'anomaly_flags.csv'}")

    # ── Layer 2 ───────────────────────────────────────────────────────────────
    print("\n── Layer 2: WFP Monthly Time-Series Anomalies ────────────────────────")
    wfp_flags = detect_wfp_anomalies(panel)
    wfp_anom  = wfp_flags[wfp_flags["flagged"]]
    print(f"Anomalous months  : {len(wfp_anom)} / {len(wfp_flags)}")
    if len(wfp_anom):
        print(f"\n  {'Month':<10} {'WFP Index':>10} {'MoM%':>8} {'Z-score':>9}")
        print("  " + "-" * 42)
        for _, row in wfp_anom.iterrows():
            mom = (np.exp(row["log_ret"]) - 1) * 100
            print(f"  {row['year_month'].strftime('%Y-%m'):<10}  "
                  f"{row['wfp_food_index']:>10.1f}  {mom:>+7.1f}%  {row['z_ret']:>9.2f}σ")
    wfp_flags.to_csv(PROC / "wfp_anomaly_months.csv", index=False)
    print(f"\nSaved → {PROC/'wfp_anomaly_months.csv'}")

    # ── IMPIN clean ───────────────────────────────────────────────────────────
    print("\n── IMPIN Index: Raw vs Clean ─────────────────────────────────────────")
    raw_impin   = compute_impin(flagged_df, "raw")
    clean_impin = compute_impin(flagged_df[~flagged_df["is_flagged"]], "clean")
    merged      = raw_impin.merge(clean_impin, on="category", suffixes=("_raw","_clean"))
    print(f"\n  {'Category':<28} {'Raw median':>11} {'Clean median':>13} {'Δ%':>7} {'Removed':>8}")
    print("  " + "-" * 72)
    for _, r in merged.iterrows():
        delta = (r["median_price_ghc_clean"] - r["median_price_ghc_raw"]) / r["median_price_ghc_raw"] * 100
        n_rem = r["n_products_raw"] - r["n_products_clean"]
        print(f"  {r['category']:<28}  {r['median_price_ghc_raw']:>11.2f}  "
              f"{r['median_price_ghc_clean']:>13.2f}  {delta:>+6.1f}%  {n_rem:>8.0f}")
    clean_impin.to_csv(PROC / "impin_clean.csv", index=False)
    print(f"\nSaved → {PROC/'impin_clean.csv'}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n── Generating plots ──────────────────────────────────────────────────")
    print(f"Plot saved → {plot_price_outliers(flagged_df)}")
    print(f"Plot saved → {plot_wfp_anomalies(wfp_flags, panel)}")
    print(f"Plot saved → {plot_impin_clean(raw_impin, clean_impin)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    all_raw   = float(raw_impin[raw_impin["category"]=="All (equal-weight)"]["median_price_ghc"])
    all_clean = float(clean_impin[clean_impin["category"]=="All (equal-weight)"]["median_price_ghc"])
    delta_all = (all_clean - all_raw) / all_raw * 100
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Products flagged       : {n_flagged:>5}  ({n_flagged/len(flagged_df):.1%})")
    print(f"  WFP anomalous months   : {len(wfp_anom):>5}  ({len(wfp_anom)/len(wfp_flags):.1%})")
    print(f"  Raw IMPIN median price : {all_raw:>9.2f} GHC")
    print(f"  Clean IMPIN median     : {all_clean:>9.2f} GHC  (Δ{delta_all:+.1f}%)")
    if abs(delta_all) < 1.0:
        print(f"\n  ✓ Anomalies have minimal impact (<1%). IMPIN snapshot is robust.")
    elif delta_all < 0:
        print(f"\n  ~ Removing outliers LOWERS index by {abs(delta_all):.1f}% — flagged items were luxury/high-end.")
    else:
        print(f"\n  ~ Removing outliers RAISES index by {delta_all:.1f}% — flagged items were suspiciously cheap.")
    print("=" * 72)
    print("\nDONE.\n")


if __name__ == "__main__":
    main()
