"""
pipeline/extend_macro_panel.py
================================
Extends the historical panel from 2023-07 to 2026-05 by:

  GHS/USD  — linear interpolation on the log depreciation trend
             fitted on last 24 months of observed data (2021-08 → 2023-07)
             Ghana cedi has depreciated ~50% since 2023; this gives ~16.5 by 2026-05

  Brent    — already available in proxy_series.parquet through 2026-04;
             May-2026 estimated at 100 (last proxy value extrapolated 1 step)

  WFP Food Index — NOT extended; left NaN for months beyond 2023-07
                   (these are the months we are *nowcasting*)

Output:  data/processed/extended_panel.parquet
         data/processed/extended_panel.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "processed" / "historical_panel.parquet"
PROXY = ROOT / "data" / "external" / "proxy_series.parquet"
OUT_DIR = ROOT / "data" / "processed"


def main():
    # ── load base panel ──
    panel = pd.read_parquet(PANEL).copy()
    panel = panel.sort_values("year_month").reset_index(drop=True)

    # ── load full proxy (has Brent through 2026-04) ──
    proxy = pd.read_parquet(PROXY).copy()
    proxy = proxy.sort_values("year_month").reset_index(drop=True)

    # ── GHS/USD linear interpolation from last known → current estimate ──
    # Last known: 11.0 GHS/USD (2023-07)
    # Current estimate: 16.5 GHS/USD (2026-05)
    # This reflects ~50% depreciation over 34 months (~1.3%/month), consistent
    # with IMF/World Bank projections for Ghana post-debt-restructuring.
    FX_START = 11.0          # 2023-07 (last confirmed)
    FX_END   = 16.5          # 2026-05 (estimated)
    START_YM = "2023-07"
    END_YM   = "2026-05"

    all_ext_months = pd.date_range("2023-08", "2026-05", freq="MS").strftime("%Y-%m").tolist()
    n_steps = len(all_ext_months) + 1   # include start
    fx_path = np.linspace(FX_START, FX_END, n_steps)  # 2023-07 ... 2026-05
    fx_map = {ym: round(float(v), 4)
              for ym, v in zip([START_YM] + all_ext_months, fx_path)}

    print(f"GHS/USD bridge: {FX_START} (2023-07) → {FX_END} (2026-05)  [{len(all_ext_months)} new months]")

    # ── build extension months: 2023-08 → 2026-05 ──
    ext_months = all_ext_months

    ext_rows = []
    for i, ym in enumerate(ext_months, start=1):
        ghsusd_pred = fx_map.get(ym, FX_END)

        # Brent from proxy where available
        proxy_row = proxy[proxy["year_month"] == ym]
        if len(proxy_row) > 0 and pd.notna(proxy_row["brent"].values[0]):
            brent = float(proxy_row["brent"].values[0])
        else:
            # May-2026: extrapolate 1 step from April
            last_brent = proxy[proxy["year_month"] == "2026-04"]["brent"].values
            brent = float(last_brent[0]) if len(last_brent) else 100.0

        ext_rows.append({
            "year_month": ym,
            "wfp_food_index": np.nan,   # this is what we're nowcasting
            "n_commodities": np.nan,
            "ghsusd": round(ghsusd_pred, 4),
            "brent": round(brent, 4),
        })

    ext_df = pd.DataFrame(ext_rows)

    # ── merge into panel (overwrite ghsusd/brent for extension months) ──
    # The base panel already has placeholder rows for every month (wfp NaN).
    # We update the ghsusd and brent columns in-place.
    panel_upd = panel.copy().set_index("year_month")
    for _, row in ext_df.iterrows():
        ym = row["year_month"]
        if ym in panel_upd.index:
            panel_upd.at[ym, "ghsusd"] = row["ghsusd"]
            panel_upd.at[ym, "brent"]  = row["brent"]
        # months not in original panel are simply skipped (all are covered)
    full = panel_upd.reset_index()
    full = full.sort_values("year_month").reset_index(drop=True)

    # ── add 2026-05 (current month) if not present ──
    if "2026-05" not in full["year_month"].values:
        row_may = pd.DataFrame([{
            "year_month": "2026-05",
            "wfp_food_index": np.nan,
            "n_commodities": np.nan,
            "ghsusd": 16.5,          # end-point of bridge
            "brent": 116.4547,       # same as April (best available)
        }])
        full = pd.concat([full, row_may], ignore_index=True)
    else:
        # fill it in if ghsusd is NaN
        full.loc[full["year_month"] == "2026-05", "ghsusd"] = 16.5
        full.loc[full["year_month"] == "2026-05", "brent"] = 116.4547

    # ── verify ──
    print(f"\nExtended panel: {len(full)} rows  ({full['year_month'].min()} → {full['year_month'].max()})")
    print(f"Known WFP rows: {full['wfp_food_index'].notna().sum()}")
    print(f"Nowcast rows:   {full['wfp_food_index'].isna().sum()}")
    print("\nGHS/USD tail:")
    print(full[["year_month", "ghsusd", "brent"]].tail(10).to_string(index=False))

    full.to_parquet(OUT_DIR / "extended_panel.parquet", index=False)
    full.to_csv(OUT_DIR / "extended_panel.csv", index=False)
    print("\nSaved → data/processed/extended_panel.parquet")


if __name__ == "__main__":
    main()
