"""
models/walk_forward.py
======================
Walk-forward (expanding-window) backtest — IMPIN food price models

Why this matters
----------------
A single train/test split gives fragile metrics. Walk-forward evaluation
re-fits every model from scratch on each expanding window and makes a
TRUE 1-step-ahead out-of-sample prediction each month. This reveals:
  • real RMSE stability over time
  • whether XGB drift is structural or a one-off artifact
  • whether ARIMAX is consistently stable under regime shifts
  • whether the dynamic blend actually adds value vs its components

Protocol
--------
  Start : 2021-01  (minimum 17 months of training data before first prediction)
  Step  : 1 month ahead (no recursion → no multi-step accumulation error)
  End   : 2023-07  (last observed WFP data point)
  Total : 31 one-step-ahead predictions

Models
------
  Naive       — carry-forward last observed value
  ARIMAX      — (1,1,0) refit on expanding window, real exog at t
  XGBoost     — GBM on first-diff features, predict Δ → add to last level
  RF          — same architecture as XGBoost
  DynBlend    — inverse-variance combination of Naive+ARIMAX+XGBoost+RF,
                weights updated each step from last 6 prediction errors

Outputs
-------
  models/results/walkforward_results.csv
  outputs/plots/18a_walkforward.png   — actual vs each model over time
  outputs/plots/18b_walkforward_rmse.png — rolling 6-month RMSE + blend weights
"""

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pathlib import Path

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
DATA     = ROOT / "data" / "processed"
PLOTS    = ROOT / "outputs" / "plots"
RESULTS  = ROOT / "models" / "results"

PANEL_PATH = DATA / "macro_panel_live.parquet"

# ── Config ────────────────────────────────────────────────────────────────────
K_BRENT     = 7          # brent crude lag (months, from CCF analysis)
WFP_LAGS    = [1, 2, 3]  # AR lags for ML models
WF_START    = "2021-01"  # first month to predict (min training window = 17 mo)
RECENT_K    = 6          # rolling error window for dynamic blend weights
RANDOM_SEED = 42
ROLL_WIN    = 6          # window for rolling RMSE plot

COLOURS = {
    "Actual":   "#000000",
    "Naive":    "#aaaaaa",
    "ARIMAX":   "#1f77b4",
    "XGBoost":  "#2ca02c",
    "RF":       "#ff7f0e",
    "DynBlend": "#d62728",
}

# ── Stat helpers ──────────────────────────────────────────────────────────────
def _rmse(e):        return float(np.sqrt(np.mean(np.asarray(e) ** 2)))
def _mae(e):         return float(np.mean(np.abs(np.asarray(e))))
def _mape(a, p):     return float(np.mean(np.abs((np.asarray(a) - np.asarray(p)) / np.clip(np.asarray(a), 1e-6, None))) * 100)
def _dir_acc(a, p):
    """Directional accuracy: did forecast correctly call the sign of Δ?"""
    da = np.sign(np.diff(np.asarray(a))) == np.sign(np.asarray(p)[1:] - np.asarray(a)[:-1])
    return float(da.mean()) if len(da) else float("nan")

# ── Model predictors (1-step-ahead, strictly no look-ahead) ──────────────────

def pred_naive(train_wfp: pd.Series) -> float:
    return float(train_wfp.iloc[-1])


def pred_arimax(train_df: pd.DataFrame, panel_idx: pd.DataFrame, t_date: pd.Timestamp) -> float:
    """Refit ARIMAX(1,1,0) on expanding window, forecast 1 step ahead."""
    wfp = train_df["wfp_food_index"].values.astype(float)

    # brent_lag series: shift full panel by K_BRENT positions (monthly panel → months)
    brent_lags = panel_idx["brent"].shift(K_BRENT)
    ghsusd     = panel_idx["ghsusd"]

    tr_dates = list(train_df["year_month"])
    bl = brent_lags.reindex(tr_dates).values.astype(float)
    gs = ghsusd.reindex(tr_dates).values.astype(float)

    valid = ~(np.isnan(bl) | np.isnan(gs))
    if valid.sum() < 8:
        return pred_naive(train_df["wfp_food_index"])

    wfp_v  = wfp[valid]
    exog_v = np.column_stack([bl[valid], gs[valid]])

    # exog for the forecast step
    bl_fc = brent_lags.get(t_date, np.nan)
    gs_fc = ghsusd.get(t_date, np.nan)
    if np.isnan([bl_fc, gs_fc]).any():
        return pred_naive(train_df["wfp_food_index"])

    try:
        mdl = SARIMAX(wfp_v, exog=exog_v, order=(1, 1, 0), trend="n")
        fit = mdl.fit(disp=False)
        fc  = fit.forecast(steps=1, exog=np.array([[bl_fc, gs_fc]]))
        return float(fc.iloc[0])
    except Exception:
        return pred_naive(train_df["wfp_food_index"])


def pred_ml(train_df: pd.DataFrame, panel_idx: pd.DataFrame,
            t_date: pd.Timestamp, model_name: str) -> float:
    """
    Refit tree model on first-diff (ΔWFP) features each step.
    Predict delta, add to last observed level.
    1-step ahead → no recursive accumulation error.
    """
    d = train_df.copy()
    d["dwfp"] = d["wfp_food_index"].diff()

    brent_lags = panel_idx["brent"].shift(K_BRENT)
    ghsusd     = panel_idx["ghsusd"]

    tr_dates = list(d["year_month"])
    d["brent_lag"] = brent_lags.reindex(tr_dates).values
    d["ghsusd"]    = ghsusd.reindex(tr_dates).values
    d["log_ghsusd"] = np.log(d["ghsusd"].clip(lower=0.01))
    for lag in WFP_LAGS:
        d[f"wfp_lag{lag}"] = d["wfp_food_index"].shift(lag)

    feat_cols = ["brent_lag", "log_ghsusd"] + [f"wfp_lag{l}" for l in WFP_LAGS]
    d = d.dropna(subset=feat_cols + ["dwfp"])
    if len(d) < 6:
        return pred_naive(train_df["wfp_food_index"])

    X = d[feat_cols].values.astype(float)
    y = d["dwfp"].values.astype(float)

    if model_name == "XGBoost":
        mdl = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                         learning_rate=0.05, subsample=0.8,
                                         random_state=RANDOM_SEED)
    else:
        mdl = RandomForestRegressor(n_estimators=200, max_depth=4,
                                     random_state=RANDOM_SEED, n_jobs=-1)
    mdl.fit(X, y)

    # Features for forecast step t_date
    wfp_vals = train_df["wfp_food_index"].values
    last_wfp = float(wfp_vals[-1])
    lags = [float(wfp_vals[-i]) if len(wfp_vals) >= i else last_wfp for i in WFP_LAGS]

    bl_fc = brent_lags.get(t_date, np.nan)
    gs_fc = ghsusd.get(t_date, np.nan)
    if np.isnan([bl_fc, gs_fc]).any():
        return last_wfp

    x_pred = np.array([[bl_fc, np.log(max(gs_fc, 0.01))] + lags])
    delta  = float(mdl.predict(x_pred)[0])
    return last_wfp + delta


# ── Walk-forward loop ─────────────────────────────────────────────────────────

def run():
    panel = pd.read_parquet(PANEL_PATH)
    panel["year_month"] = pd.to_datetime(panel["year_month"])
    panel = panel.sort_values("year_month").reset_index(drop=True)
    panel_idx = panel.set_index("year_month")

    obs = (panel.dropna(subset=["wfp_food_index"])
                .sort_values("year_month")
                .reset_index(drop=True))

    pred_months = obs.loc[obs["year_month"] >= pd.Timestamp(WF_START), "year_month"].tolist()
    n_steps = len(pred_months)

    print("=" * 72)
    print(f"WALK-FORWARD BACKTEST  ({WF_START} → {pred_months[-1].strftime('%Y-%m')}, {n_steps} steps)")
    print(f"Training window: expanding from {obs['year_month'].iloc[0].strftime('%Y-%m')} + 1 month each step")
    print("=" * 72)
    print(f"{'Month':<10} {'Actual':>7} {'Naive':>7} {'ARIMAX':>7} {'XGB':>7} {'RF':>7} {'DynBlend':>9}")
    print("-" * 62)

    records  = []
    err_hist = {m: [] for m in ["Naive", "ARIMAX", "XGBoost", "RF"]}
    wt_hist  = []   # dynamic weights over time for plotting

    for i, t_date in enumerate(pred_months):
        train_df = obs[obs["year_month"] < t_date].copy()
        actual   = float(obs.loc[obs["year_month"] == t_date, "wfp_food_index"].values[0])

        p_n = pred_naive(train_df["wfp_food_index"])
        p_a = pred_arimax(train_df, panel_idx, t_date)
        p_x = pred_ml(train_df, panel_idx, t_date, "XGBoost")
        p_r = pred_ml(train_df, panel_idx, t_date, "RF")

        # ── Dynamic blend: inverse-variance weights from last RECENT_K errors ──
        # Key: weights are computed BEFORE seeing current actual → no leakage.
        # All 4 models included (1-step RF is fine; only long-horizon RF explodes).
        recent = {m: err_hist[m][-RECENT_K:] for m in ["Naive", "ARIMAX", "XGBoost", "RF"]}
        if all(len(v) >= 3 for v in recent.values()):
            ivs = {m: 1.0 / max(_rmse(v) ** 2, 1e-6) for m, v in recent.items()}
            tot = sum(ivs.values())
            w   = {m: ivs[m] / tot for m in ivs}
            p_b = w["Naive"] * p_n + w["ARIMAX"] * p_a + w["XGBoost"] * p_x + w["RF"] * p_r
        else:
            w   = {m: 0.25 for m in ["Naive", "ARIMAX", "XGBoost", "RF"]}
            p_b = (p_n + p_a + p_x + p_r) / 4.0

        wt_hist.append({"year_month": t_date, **{f"w_{m}": w[m] for m in w}})

        # Record errors AFTER prediction (used for next step's weights)
        for m, p in [("Naive", p_n), ("ARIMAX", p_a), ("XGBoost", p_x), ("RF", p_r)]:
            err_hist[m].append(actual - p)

        records.append({
            "year_month": t_date,
            "actual":     actual,
            "Naive":      p_n,
            "ARIMAX":     p_a,
            "XGBoost":    p_x,
            "RF":         p_r,
            "DynBlend":   p_b,
        })
        print(f"  {t_date.strftime('%Y-%m')}  {actual:>7.1f}  {p_n:>7.1f}  {p_a:>7.1f}  {p_x:>7.1f}  {p_r:>7.1f}  {p_b:>9.1f}")

    df  = pd.DataFrame(records)
    wdf = pd.DataFrame(wt_hist)
    df.to_csv(RESULTS / "walkforward_results.csv", index=False)

    # ── Summary metrics ───────────────────────────────────────────────────────
    models_list = ["Naive", "ARIMAX", "XGBoost", "RF", "DynBlend"]
    print("\n" + "=" * 68)
    print(f"WALK-FORWARD SUMMARY  ({n_steps} one-step-ahead predictions)")
    print("=" * 68)
    print(f"{'Model':<20} {'RMSE':>7} {'MAE':>7} {'MAPE%':>7} {'DirAcc':>8}")
    print("-" * 50)
    summary = {}
    for m in models_list:
        e   = df["actual"] - df[m]
        r   = _rmse(e)
        a   = _mae(e)
        mp  = _mape(df["actual"].values, df[m].values)
        da  = _dir_acc(df["actual"].values, df[m].values)
        summary[m] = {"rmse": r, "mae": a, "mape": mp, "dir_acc": da}
        print(f"  {m:<18} {r:>7.1f} {a:>7.1f} {mp:>7.1f} {da:>8.1%}")
    print("=" * 68)

    # ── Stability analysis (early vs late half) ───────────────────────────────
    mid = n_steps // 2
    print(f"\nSTABILITY  (early half: steps 1–{mid}  |  late half: steps {mid+1}–{n_steps})")
    print(f"{'Model':<20} {'RMSE early':>12} {'RMSE late':>10} {'Drift':>8}")
    print("-" * 55)
    for m in ["Naive", "ARIMAX", "XGBoost", "RF", "DynBlend"]:
        e_e = _rmse(df["actual"].values[:mid] - df[m].values[:mid])
        e_l = _rmse(df["actual"].values[mid:] - df[m].values[mid:])
        drift = e_l - e_e
        flag  = " ← drifts" if drift > 5 else (" ← improves" if drift < -5 else "")
        print(f"  {m:<18} {e_e:>12.1f} {e_l:>10.1f} {drift:>+8.1f}{flag}")

    # ── Blend effectiveness ───────────────────────────────────────────────────
    print("\nINSIGHTS:")
    best_base_rmse = min(summary[m]["rmse"] for m in ["Naive", "ARIMAX", "XGBoost", "RF"])
    blend_rmse     = summary["DynBlend"]["rmse"]
    if blend_rmse < best_base_rmse:
        pct = (best_base_rmse - blend_rmse) / best_base_rmse * 100
        print(f"  ✓ DynBlend BEATS best single model ({best_base_rmse:.1f}) → RMSE={blend_rmse:.1f}  [{pct:.1f}% improvement]")
    else:
        print(f"  ~ DynBlend ({blend_rmse:.1f}) does NOT beat best base ({best_base_rmse:.1f})")

    # XGB drift
    xe_e = _rmse(df["actual"].values[:mid] - df["XGBoost"].values[:mid])
    xe_l = _rmse(df["actual"].values[mid:] - df["XGBoost"].values[mid:])
    if xe_l > xe_e * 1.2:
        print(f"  ✗ XGBoost drift CONFIRMED (1-step): RMSE {xe_e:.1f} → {xe_l:.1f} in second half")
    else:
        print(f"  ✓ XGBoost 1-step drift NOT significant: RMSE {xe_e:.1f} → {xe_l:.1f}")

    # ARIMAX consistency
    ae_e = _rmse(df["actual"].values[:mid] - df["ARIMAX"].values[:mid])
    ae_l = _rmse(df["actual"].values[mid:] - df["ARIMAX"].values[mid:])
    print(f"  {'✓' if ae_l <= ae_e else '~'} ARIMAX 1-step: RMSE {ae_e:.1f} (early) → {ae_l:.1f} (late)")

    print(f"\n  NOTE: XGB had RMSE drift in the 34-step recursive NOWCAST because of")
    print(f"  delta accumulation error. In 1-step-ahead, no accumulation → stable.")
    print(f"  This explains the nowcast vs walk-forward gap.")

    # ── Plot 18a: actual vs predictions ───────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Walk-Forward Backtest — IMPIN Food Price Models\n"
                 f"({WF_START} → {pred_months[-1].strftime('%Y-%m')}, "
                 f"{n_steps} one-step-ahead predictions)",
                 fontweight="bold", fontsize=13)

    ax = axes[0]
    ax.plot(df["year_month"], df["actual"], color=COLOURS["Actual"],
            lw=2.5, zorder=10, label="Actual WFP")
    styles = {"Naive":    (":", 1.5, 0.9),
              "ARIMAX":   ("-.", 1.8, 0.85),
              "XGBoost":  ("--", 1.8, 0.85),
              "RF":       ("--", 1.4, 0.55),
              "DynBlend": ("-",  2.8, 1.0)}
    for m, (ls, lw, alpha) in styles.items():
        ax.plot(df["year_month"], df[m],
                color=COLOURS.get(m, "#333"), ls=ls, lw=lw, alpha=alpha, label=m)
    ax.set_ylabel("WFP Food Price Index (raw)", fontsize=10)
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # Shade early vs late for context
    mid_date = df["year_month"].iloc[mid]
    ax.axvline(mid_date, color="#888", ls=":", lw=1.0, alpha=0.6)
    ax.text(mid_date, ax.get_ylim()[0] + 5, "  mid", fontsize=8, color="#888")

    # ── Plot 18b: rolling RMSE + blend weights ─────────────────────────────────
    ax2 = axes[1]
    for m in ["Naive", "ARIMAX", "XGBoost", "RF", "DynBlend"]:
        sq_err = (df["actual"] - df[m]) ** 2
        r_rmse = sq_err.rolling(ROLL_WIN, min_periods=ROLL_WIN).mean().apply(np.sqrt)
        lw  = 2.8 if m == "DynBlend" else 1.5
        ls  = "-"  if m == "DynBlend" else "--"
        alpha = 1.0 if m in ("DynBlend", "ARIMAX", "XGBoost") else 0.55
        ax2.plot(df["year_month"], r_rmse,
                 color=COLOURS.get(m, "#333"), lw=lw, ls=ls, alpha=alpha, label=m)

    ax2.set_ylabel(f"Rolling {ROLL_WIN}-month RMSE", fontsize=10)
    ax2.set_xlabel("Month", fontsize=10)
    ax2.legend(ncol=3, fontsize=9, loc="upper left")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    out_a = PLOTS / "18a_walkforward.png"
    fig.savefig(out_a, dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 18b: dynamic blend weights over time ──────────────────────────────
    fig2, ax3 = plt.subplots(figsize=(14, 4))
    fig2.suptitle("DynBlend: adaptive weights over time\n"
                  "(driven by rolling inverse-variance of last 6 prediction errors)",
                  fontweight="bold", fontsize=11)
    weight_cols = [c for c in wdf.columns if c.startswith("w_")]
    wlabels = {"w_Naive": "Naive", "w_ARIMAX": "ARIMAX",
               "w_XGBoost": "XGBoost", "w_RF": "RF"}
    for wc in weight_cols:
        ax3.plot(wdf["year_month"], wdf[wc],
                 color=COLOURS.get(wlabels.get(wc, wc), "#333"),
                 lw=1.8, label=wlabels.get(wc, wc))
    ax3.set_ylabel("Weight in DynBlend", fontsize=10)
    ax3.set_xlabel("Month", fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.legend(ncol=4, fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    out_b = PLOTS / "18b_blend_weights.png"
    fig2.savefig(out_b, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nPlots saved →  {out_a}")
    print(f"               {out_b}")
    print(f"Results  →     {RESULTS / 'walkforward_results.csv'}")
    print("\nDONE.\n")


if __name__ == "__main__":
    run()
