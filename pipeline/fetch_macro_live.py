"""
pipeline/fetch_macro_live.py
============================
Fetch real observed macroeconomic regressors for IMPIN Layer 2:

  1. GHS/USD monthly average  — Yahoo Finance (USDGHS=X)
     Source: daily close prices, resampled to monthly mean
     Coverage: 2023-01 → present

  2. Brent crude monthly avg  — proxy_series.parquet (already real through 2026-04)
     No re-fetch needed; proxy_series was built from real daily data.

Outputs:
  data/external/ghsusd_live.parquet   — GHS/USD observed monthly (2019-08 → present)
  data/processed/macro_panel_live.parquet — full macro panel with real regressors

Run:
  python pipeline/fetch_macro_live.py
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXT        = ROOT / "data" / "external"
PROCESSED  = ROOT / "data" / "processed"
EXT.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

HIST_PNL = PROCESSED / "historical_panel.parquet"
PROXY    = EXT       / "proxy_series.parquet"
OUT_FX   = EXT       / "ghsusd_live.parquet"
OUT_PNL  = PROCESSED / "macro_panel_live.parquet"


def fetch_ghsusd_monthly() -> pd.DataFrame:
    """
    Fetch USD→GHS monthly averages from Yahoo Finance (USDGHS=X).
    Uses daily close prices resampled to monthly mean for robustness.
    Returns DataFrame with columns: year_month (MS), ghsusd
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Run: pip install yfinance --user")

    print("  Fetching USDGHS=X from Yahoo Finance …")
    t    = yf.Ticker("USDGHS=X")
    daily = t.history(start="2019-01-01", end="2026-06-01", interval="1d")

    if daily.empty:
        raise RuntimeError("Yahoo Finance returned empty data for USDGHS=X")

    monthly = (
        daily["Close"]
        .resample("MS")
        .mean()
        .reset_index()
    )
    monthly.columns = ["date", "ghsusd"]
    monthly["year_month"] = monthly["date"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    monthly = monthly[["year_month", "ghsusd"]].dropna()
    monthly = monthly.sort_values("year_month").reset_index(drop=True)

    print(f"  GHS/USD: {len(monthly)} monthly obs  "
          f"({monthly['year_month'].min().strftime('%Y-%m')} → "
          f"{monthly['year_month'].max().strftime('%Y-%m')})")
    print(f"  Range: min={monthly['ghsusd'].min():.3f}  max={monthly['ghsusd'].max():.3f}  "
          f"latest={monthly['ghsusd'].iloc[-1]:.4f} GHS/USD")
    return monthly


def build_live_panel(fx_live: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full macro panel (2019-08 → 2026-05) using:
    - WFP food index: from historical_panel.parquet (2019-08 → 2023-07)
    - GHS/USD:
        2019-08 → 2023-07: historical_panel (trusted Bank of Ghana data)
        2023-08 → 2026-05: Yahoo Finance daily average (USDGHS=X)
    - Brent: REAL from proxy_series.parquet through 2026-04, May-2026 held
    """
    # Load historical panel (has trusted GHS/USD through 2023-07 only)
    hist = pd.read_parquet(HIST_PNL)
    hist["year_month"] = pd.to_datetime(hist["year_month"])
    wfp  = hist[["year_month", "wfp_food_index"]].dropna(subset=["wfp_food_index"])
    # Cap historical GHS/USD at 2023-07 (after that the panel forward-fills 11.0, not real)
    hist_fx = (
        hist[["year_month", "ghsusd"]]
        .dropna(subset=["ghsusd"])
        .query("year_month <= '2023-07-01'")
        .copy()
    )

    # Load Brent from proxy_series (real through 2026-04)
    proxy = pd.read_parquet(PROXY)
    proxy["year_month"] = pd.to_datetime(proxy["year_month"])
    brent = proxy[["year_month", "brent"]].dropna(subset=["brent"])

    # Yahoo Finance GHS/USD: only use the gap period (2023-08 onwards)
    fx_gap = fx_live[fx_live["year_month"] >= "2023-08-01"].copy()

    # Combine: historical (trusted) + Yahoo (gap)
    fx_combined = pd.concat([hist_fx, fx_gap], ignore_index=True)
    fx_combined = fx_combined.drop_duplicates("year_month").sort_values("year_month")

    # Build full monthly date range
    full_dates = pd.date_range("2019-08-01", "2026-05-01", freq="MS")
    panel = pd.DataFrame({"year_month": full_dates})

    # Merge WFP food index
    panel = panel.merge(wfp[["year_month", "wfp_food_index"]], on="year_month", how="left")

    # Merge combined GHS/USD (historical + Yahoo gap)
    panel = panel.merge(fx_combined[["year_month", "ghsusd"]], on="year_month", how="left")

    # Merge Brent
    panel = panel.merge(brent[["year_month", "brent"]], on="year_month", how="left")

    # May-2026 Brent: hold last known value (Apr-2026 = 116.45) if missing
    if panel.loc[panel["year_month"] == "2026-05-01", "brent"].isna().all():
        last_brent = panel.dropna(subset=["brent"])["brent"].iloc[-1]
        panel.loc[panel["year_month"] == "2026-05-01", "brent"] = last_brent
        print(f"  May-2026 Brent: held at {last_brent:.2f} (last real: Apr-2026)")

    # Flag data quality
    panel["ghsusd_is_observed"] = panel["ghsusd"].notna()
    panel["brent_is_observed"]  = (panel["year_month"] <= pd.Timestamp("2026-04-01"))

    # Check for remaining NaN in GHS/USD (shouldn't happen if Yahoo has full coverage)
    missing_fx = panel["ghsusd"].isna().sum()
    if missing_fx > 0:
        print(f"  WARNING: {missing_fx} months still missing GHS/USD — filling forward")
        panel["ghsusd"] = panel["ghsusd"].ffill().bfill()

    return panel.reset_index(drop=True)


def main():
    print("=" * 65)
    print("IMPIN — Fetch Real Macro Regressors")
    print("=" * 65)

    # 1. Fetch GHS/USD
    print("\n[1] GHS/USD:")
    fx = fetch_ghsusd_monthly()
    fx.to_parquet(OUT_FX, index=False)
    print(f"  Saved → {OUT_FX.relative_to(ROOT)}")

    # 2. Build live panel
    print("\n[2] Building macro panel …")
    panel = build_live_panel(fx)

    obs_wfp = panel["wfp_food_index"].notna().sum()
    obs_fx  = panel["ghsusd_is_observed"].sum()
    obs_b   = panel["brent_is_observed"].sum()
    print(f"  Panel shape: {panel.shape}")
    print(f"  WFP food index:  {obs_wfp} observed rows  (2019-08 → 2023-07)")
    print(f"  GHS/USD:         {obs_fx} observed rows  (all real — Yahoo Finance)")
    print(f"  Brent:           {obs_b} observed rows  (proxy_series, real through 2026-04)")

    panel.to_parquet(OUT_PNL, index=False)
    print(f"\n  Saved → {OUT_PNL.relative_to(ROOT)}")

    # 3. Print boundary snapshot
    print("\n  GHS/USD comparison: bridge vs real (2023-07 to 2026-05)")
    print(f"  {'Month':<10}  {'Real (Yahoo)':>14}  {'Old bridge':>12}")
    bridge_start = 11.0
    bridge_end   = 16.5
    n_bridge = 35  # Aug-2023 to May-2026
    bridge_vals = dict(zip(
        pd.date_range("2023-08", "2026-05", freq="MS"),
        np.linspace(bridge_start, bridge_end, n_bridge)
    ))
    for _, row in panel[panel["year_month"] >= "2023-07"].iterrows():
        d    = row["year_month"]
        real = row["ghsusd"]
        old  = bridge_vals.get(d, row["ghsusd"])
        diff = real - old
        print(f"  {d.strftime('%Y-%m'):<10}  {real:>14.4f}  {old:>12.4f}  Δ={diff:+.2f}")


if __name__ == "__main__":
    main()
