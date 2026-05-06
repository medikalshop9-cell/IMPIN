"""
IMPIN Streamlit Dashboard
Ghana Food Price Index — Nowcast & Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
SCRAPE   = ROOT / "data" / "raw"    / "scraped_combined.csv"
ANOMALY  = ROOT / "data" / "processed" / "anomaly_flags.csv"
IMPIN_CL = ROOT / "data" / "processed" / "impin_clean.csv"
NOWCAST  = ROOT / "models" / "results" / "all_models_nowcast.csv"
METRICS  = ROOT / "models" / "results" / "retrain_live_metrics.csv"
WF_RES   = ROOT / "models" / "results" / "walkforward_results.csv"
MACRO    = ROOT / "data" / "processed" / "macro_panel_live.parquet"
PLOTS    = ROOT / "outputs" / "plots"

IMPIN_VAL = 100.0

# colour palette consistent with retrain_live.py
COLOURS = {
    "Actual":       "#000000",
    "Naive":        "#aaaaaa",
    "ARIMAX":       "#1f77b4",
    "ARIMAX+Boost": "#00CED1",
    "HorizonBlend": "#d62728",
    "XGBoost":      "#2ca02c",
    "RF":           "#ff7f0e",
    "Prophet":      "#9467bd",
    "IMPIN":        "#e377c2",
    "DynBlend":     "#8c564b",
}

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IMPIN Dashboard",
    page_icon="🇬🇭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🇬🇭 IMPIN — Ghana Food Price Index")
st.caption("Import Price Index · May 2026 snapshot · WFP food index base 2019-08 = 100")

# ── cached loaders ────────────────────────────────────────────────────────────
@st.cache_data
def load_scrape():
    return pd.read_csv(SCRAPE)

@st.cache_data
def load_anomaly():
    return pd.read_csv(ANOMALY)

@st.cache_data
def load_impin_clean():
    return pd.read_csv(IMPIN_CL)

@st.cache_data
def load_nowcast():
    df = pd.read_csv(NOWCAST, parse_dates=["year_month"])
    return df

@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS)

@st.cache_data
def load_wf():
    return pd.read_csv(WF_RES, parse_dates=["year_month"])

@st.cache_data
def load_macro():
    df = pd.read_parquet(MACRO)
    df["year_month"] = pd.to_datetime(df["year_month"])
    return df

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📦 IMPIN Live",
    "💰 Price Calculator",
    "📈 Nowcast",
    "🔬 Model Evaluation",
    "🌍 Macro Drivers",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMPIN Live
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    scrape    = load_scrape()
    anomaly   = load_anomaly()
    impin_cat = load_impin_clean()

    # headline metrics
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("IMPIN (May 2026)", f"{IMPIN_VAL:.1f}", "base = 100")
    col_b.metric("Products in index", f"{int(anomaly['is_flagged'].eq(False).sum()):,}")
    n_flagged = int(anomaly["is_flagged"].sum())
    col_c.metric("Removed (outliers)", f"{n_flagged}", f"{n_flagged/len(anomaly)*100:.1f}% excluded")
    col_d.metric("Scrape date", "2026-05-05")

    st.divider()

    # ── Live Price Index by category ──────────────────────────────────────────
    st.subheader("Live Price Index — May 2026 (base = 100)")
    st.caption(
        "May 2026 scraped prices are the index base. Each category shows its "
        "median observed price and product count after outlier removal."
    )

    cat_cols = st.columns(len(impin_cat))
    for i, row in impin_cat.iterrows():
        cat_cols[i].metric(
            row["category"],
            f"GHS {row['median_price_ghc']:.2f}",
            f"{int(row['n_products'])} products",
        )

    st.divider()

    # ── full product table ────────────────────────────────────────────────────
    st.subheader("All scraped commodities")

    filt_cat = st.selectbox(
        "Filter by category",
        ["All"] + sorted(anomaly["cpi_category"].unique().tolist()),
        key="live_cat_filter",
    )
    filt_status = st.radio(
        "Show",
        ["All products", "Included only", "Removed only"],
        horizontal=True,
        key="live_status_filter",
    )

    prod_df = anomaly.copy()
    prod_df["Status"] = prod_df["is_flagged"].map({False: "✓ Included", True: "⚠ Removed (outlier)"})
    if filt_cat != "All":
        prod_df = prod_df[prod_df["cpi_category"] == filt_cat]
    if filt_status == "Included only":
        prod_df = prod_df[prod_df["is_flagged"] == False]
    elif filt_status == "Removed only":
        prod_df = prod_df[prod_df["is_flagged"] == True]

    prod_disp = prod_df[[
        "product_name", "cpi_category", "source", "price_ghc", "z_score", "Status"
    ]].copy()
    prod_disp.columns = ["Product", "Category", "Source", "Price (GHS)", "Z-score", "Status"]
    prod_disp = prod_disp.sort_values(["Category", "Price (GHS)"]).reset_index(drop=True)

    def _colour_status(val):
        if "Removed" in str(val):
            return "color: #cc0000; font-weight: bold"
        return "color: #228B22"

    st.dataframe(
        prod_disp.style
            .map(_colour_status, subset=["Status"])
            .format({"Price (GHS)": "{:.2f}", "Z-score": "{:.2f}"}),
        use_container_width=True,
        height=420,
        hide_index=True,
    )
    st.caption(
        "Outliers flagged by ≥ 2/3 of: Z-score (|z|>2.5), IQR fence (k=2.0), "
        "Isolation Forest (contamination=5%). Removed products are excluded from "
        "the index calculation — impact on IMPIN ≈ 0.0%."
    )

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Products per source")
        src_counts = scrape["source"].value_counts().reset_index()
        src_counts.columns = ["Source", "Count"]
        fig_src = px.bar(
            src_counts, x="Count", y="Source", orientation="h",
            color="Count", color_continuous_scale="Blues",
            labels={"Count": "# products"},
            height=260,
        )
        fig_src.update_layout(
            coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title=None,
        )
        st.plotly_chart(fig_src, use_container_width=True)
    with right:
        st.subheader("Price distribution by category (log scale)")
        fig_box = px.box(
            scrape, x="cpi_category", y="price_ghc", log_y=True,
            color="cpi_category",
            labels={"cpi_category": "Category", "price_ghc": "Price (GHS, log)"},
            height=260,
        )
        fig_box.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Clean index vs raw — by category")
    img_clean = PLOTS / "19c_impin_clean.png"
    if img_clean.exists():
        st.image(str(img_clean), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Price Calculator
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    nowcast_pc = load_nowcast()
    macro_pc   = load_macro()
    anomaly_pc = load_anomaly()

    st.subheader("Future Price Estimator")
    st.info(
        "**How it works:** May 2026 scraped prices are the base (IMPIN = 100). "
        "The model forecasts how the index moves from that base. "
        "\n\n**Formula:** Estimated price = (Forecast index ÷ 100) × Current price today"
    )

    # product selector — clean products only
    clean_prods = anomaly_pc[anomaly_pc["is_flagged"] == False].copy()
    clean_prods = clean_prods.sort_values(["cpi_category", "product_name"])

    sel_cat = st.selectbox(
        "Category",
        sorted(clean_prods["cpi_category"].unique().tolist()),
        key="pc_cat",
    )
    cat_prods = clean_prods[clean_prods["cpi_category"] == sel_cat]
    sel_prod = st.selectbox(
        "Product",
        cat_prods["product_name"].tolist(),
        key="pc_prod",
    )
    current_price = float(cat_prods.loc[cat_prods["product_name"] == sel_prod, "price_ghc"].values[0])

    # forecast month selector
    forecast_months = nowcast_pc["year_month"].dt.strftime("%b %Y").tolist()
    sel_month_label = st.select_slider(
        "Forecast month",
        options=forecast_months,
        value=forecast_months[-1],
        key="pc_month",
    )
    sel_month_idx = forecast_months.index(sel_month_label)
    forecast_index = float(nowcast_pc.iloc[sel_month_idx]["blend_norm"])
    estimated_price = (forecast_index / 100.0) * current_price
    pct_change = (forecast_index - 100.0)

    st.divider()

    # ── result cards ──────────────────────────────────────────────────────────
    r1, r2, r3 = st.columns(3)
    r1.metric("Current price (May 2026)", f"GHS {current_price:.2f}")
    r2.metric(
        f"Estimated price ({sel_month_label})",
        f"GHS {estimated_price:.2f}",
        f"{pct_change:+.1f}% vs today",
        delta_color="inverse",
    )
    r3.metric("IMPIN forecast index", f"{forecast_index:.1f}", f"base = 100 (May 2026)")

    st.divider()

    # ── price trajectory chart ────────────────────────────────────────────────
    st.subheader(f"Price trajectory — {sel_prod}")

    prices_over_time = (nowcast_pc["blend_norm"] / 100.0) * current_price
    fig_pc = go.Figure()
    fig_pc.add_trace(go.Scatter(
        x=nowcast_pc["year_month"], y=prices_over_time,
        mode="lines", name="Estimated price",
        line=dict(color=COLOURS["HorizonBlend"], width=2),
        fill="tozeroy", fillcolor="rgba(214,39,40,0.07)",
    ))
    # today anchor
    fig_pc.add_trace(go.Scatter(
        x=[pd.Timestamp("2026-05-01")], y=[current_price],
        mode="markers", name="Today (May 2026)",
        marker=dict(symbol="circle", size=12, color=COLOURS["IMPIN"]),
    ))
    # selected month marker
    sel_ts = nowcast_pc.iloc[sel_month_idx]["year_month"]
    fig_pc.add_trace(go.Scatter(
        x=[sel_ts], y=[estimated_price],
        mode="markers", name=sel_month_label,
        marker=dict(symbol="star", size=14, color="#d62728"),
    ))
    fig_pc.update_layout(
        xaxis_title="Month",
        yaxis_title="Estimated price (GHS)",
        hovermode="x unified",
        margin=dict(t=20, b=30), height=380,
    )
    st.plotly_chart(fig_pc, use_container_width=True)

    # ── macro context (why) ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Why is the price changing?")

    sel_ts_pd = nowcast_pc.iloc[sel_month_idx]["year_month"]
    macro_row  = macro_pc[macro_pc["year_month"] == sel_ts_pd]
    macro_now  = macro_pc[macro_pc["year_month"] == pd.Timestamp("2026-05-01")]

    # Brent 6 months prior (the transmission lag)
    brent_lag_date = sel_ts_pd - pd.DateOffset(months=6)
    brent_lag_row  = macro_pc[macro_pc["year_month"] == brent_lag_date]

    ghs_now  = float(macro_now["ghsusd"].values[0])  if not macro_now.empty  else None
    ghs_then = float(macro_row["ghsusd"].values[0])  if not macro_row.empty  else None
    brent_lag_val = float(brent_lag_row["brent"].values[0]) if not brent_lag_row.empty else None
    brent_now_val = float(macro_pc[macro_pc["year_month"] == pd.Timestamp("2026-05-01")]["brent"].values[0]) \
                    if not macro_pc[macro_pc["year_month"] == pd.Timestamp("2026-05-01")].empty else None

    reason_parts = []
    if ghs_now and ghs_then:
        ghs_chg = ghs_then - ghs_now
        direction = "weaker" if ghs_chg > 0 else "stronger"
        reason_parts.append(
            f"**GHS/USD:** {ghs_now:.2f} today → {ghs_then:.2f} in {sel_month_label} "
            f"(cedi {direction} by {abs(ghs_chg):.2f})"
        )
    if brent_lag_val and brent_now_val:
        brent_chg = brent_lag_val - brent_now_val
        reason_parts.append(
            f"**Brent crude** 6 months before {sel_month_label}: "
            f"${brent_lag_val:.1f}/bbl (vs ${brent_now_val:.1f}/bbl today) — "
            f"{'higher' if brent_chg > 0 else 'lower'} oil → "
            f"{'higher' if brent_chg > 0 else 'lower'} import & transport costs"
        )

    expl_col1, expl_col2 = st.columns([1, 1])
    with expl_col1:
        st.markdown(
            f"**{sel_prod}**  \n"
            f"Today (May 2026): **GHS {current_price:.2f}**  \n"
            f"Forecast ({sel_month_label}): **GHS {estimated_price:.2f}** ({pct_change:+.1f}%)  \n\n"
            + "  \n".join(f"- {r}" for r in reason_parts)
        )
        st.caption(
            "Transmission channel: Oil price rises → transport & import costs rise "
            "(~6 months later) → traders raise food prices → WFP index rises → IMPIN rises."
        )
    with expl_col2:
        # mini GHS/USD chart with selected date highlighted
        fig_ghs_mini = go.Figure()
        ghs_obs = macro_pc[macro_pc["ghsusd_is_observed"] == True]
        ghs_fore = macro_pc[macro_pc["ghsusd_is_observed"] == False]
        fig_ghs_mini.add_trace(go.Scatter(
            x=ghs_obs["year_month"], y=ghs_obs["ghsusd"],
            name="GHS/USD (observed)", mode="lines",
            line=dict(color="#2ca02c", width=2),
        ))
        if not ghs_fore.empty:
            fig_ghs_mini.add_trace(go.Scatter(
                x=ghs_fore["year_month"], y=ghs_fore["ghsusd"],
                name="GHS/USD (projected)", mode="lines",
                line=dict(color="#2ca02c", width=1.5, dash="dash"),
            ))
        if ghs_then:
            fig_ghs_mini.add_trace(go.Scatter(
                x=[sel_ts_pd], y=[ghs_then],
                mode="markers", name=sel_month_label,
                marker=dict(symbol="star", size=12, color="#d62728"),
            ))
        fig_ghs_mini.update_layout(
            title="GHS/USD exchange rate",
            xaxis_title=None, yaxis_title="GHS per USD",
            margin=dict(t=40, b=20), height=280,
            showlegend=False, hovermode="x unified",
        )
        st.plotly_chart(fig_ghs_mini, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Nowcast
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    nowcast = load_nowcast()
    metrics = load_metrics()
    macro   = load_macro()

    # headline callouts
    blend_may26 = float(nowcast.loc[nowcast["year_month"] == "2026-05-01", "blend_norm"].values[-1])
    gap = blend_may26 - IMPIN_VAL
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HorizonBlend — May 2026", f"{blend_may26:.1f}", f"gap {gap:+.1f} vs IMPIN")
    c2.metric("IMPIN live anchor", f"{IMPIN_VAL:.1f}", "scraped May 2026")
    best = metrics.loc[metrics["test_rmse"].idxmin()]
    c3.metric("Best test RMSE", f"{best['test_rmse']:.1f}", best["model"])
    c4.metric("Nowcast horizon", "34 months", "Aug 2023 → May 2026")

    st.divider()

    # ── toggle ────────────────────────────────────────────────────────────────
    show_all_models = st.toggle("Show all models (examiner view)", value=False)

    if not show_all_models:
        st.subheader("IMPIN Nowcast — HorizonBlend (Aug 2023 → May 2026)")
        st.markdown(
            "> **HorizonBlend** combines three models — Naive, ARIMAX, and XGBoost — "
            "> weighted by recent forecast accuracy, with automatic adjustment for "
            "> forecast horizon so that short-run machine-learning signals dominate early "
            "> and stable statistical anchors dominate at long range."
        )
    else:
        st.subheader("Full nowcast — all models (Aug 2023 → May 2026)")

    # WFP actual
    wfp_actual = macro[macro["wfp_food_index"].notna()][["year_month", "wfp_food_index"]].copy()
    norm_base = float(wfp_actual.iloc[-1]["wfp_food_index"])
    wfp_actual["norm"] = wfp_actual["wfp_food_index"] / norm_base * 100.0

    fig = go.Figure()

    # historical actual (always shown)
    fig.add_trace(go.Scatter(
        x=wfp_actual["year_month"], y=wfp_actual["norm"],
        name="WFP Actual", mode="lines+markers",
        line=dict(color=COLOURS["Actual"], width=2),
        marker=dict(size=4),
    ))

    # IMPIN anchor (always shown)
    fig.add_trace(go.Scatter(
        x=[pd.Timestamp("2026-05-01")], y=[IMPIN_VAL],
        name="IMPIN (live scrape)", mode="markers",
        marker=dict(symbol="star", size=14, color=COLOURS["IMPIN"]),
    ))

    # HorizonBlend — always shown, thick red
    fig.add_trace(go.Scatter(
        x=nowcast["year_month"], y=nowcast["blend_norm"],
        name="HorizonBlend (γ=0.05)",
        mode="lines",
        line=dict(color=COLOURS["HorizonBlend"], width=2.5, dash="solid"),
    ))

    # other models — only when toggled on
    if show_all_models:
        other_models = [
            ("naive_norm",   "Naive",              COLOURS["Naive"],        1,  "dot"),
            ("arimax_norm",  "ARIMAX(1,1,0) lag7", COLOURS["ARIMAX"],       1,  "dash"),
            ("boost_norm",   "ARIMAX+Boost",        COLOURS["ARIMAX+Boost"],1,  "dash"),
            ("xgb_norm",     "XGBoost",             COLOURS["XGBoost"],     1,  "dash"),
            ("rf_norm",      "Random Forest",       COLOURS["RF"],          1,  "dash"),
            ("prophet_norm", "Prophet",             COLOURS["Prophet"],     1,  "dash"),
        ]
        for col, name, colour, width, dash in other_models:
            fig.add_trace(go.Scatter(
                x=nowcast["year_month"], y=nowcast[col],
                name=name, mode="lines",
                line=dict(color=colour, width=width, dash=dash),
            ))

    fig.add_vline(x=pd.Timestamp("2023-07-01").timestamp() * 1000,
                  line_dash="dot", line_color="grey",
                  annotation_text="Forecast start", annotation_position="top left")
    fig.add_hline(y=100, line_dash="dot", line_color=COLOURS["IMPIN"],
                  annotation_text="IMPIN=100", annotation_position="right")
    fig.update_layout(
        xaxis_title="Month", yaxis_title="Index (NORM_BASE_RAW = 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=40, b=40), height=520,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    if show_all_models:
        st.subheader("Static nowcast chart (full detail)")
        img_nc = PLOTS / "17c_full_nowcast.png"
        if img_nc.exists():
            st.image(str(img_nc), use_container_width=True)

    st.divider()

    # ── metrics table ──────────────────────────────────────────────────────────
    st.subheader("Test-period metrics  (Jul 2022 – Jul 2023, 13 steps)")

    # add nowcast final value column
    nowcast_last = nowcast[nowcast["year_month"] == "2026-05-01"].iloc[0]
    nc_vals = {
        "HorizonBlend (γ=0.05)": float(nowcast_last["blend_norm"]),
        "Naive":                  float(nowcast_last["naive_norm"]),
        "ARIMAX(1,1,0) lag7":    float(nowcast_last["arimax_norm"]),
        "ARIMAX+Boost":           float(nowcast_last["boost_norm"]),
        "XGBoost":                float(nowcast_last["xgb_norm"]),
        "Random Forest":          float(nowcast_last["rf_norm"]),
        "Prophet":                float(nowcast_last["prophet_norm"]),
    }
    disp_metrics = metrics.copy()
    disp_metrics["Nowcast May-26"] = disp_metrics["model"].map(nc_vals)
    disp_metrics["Gap vs IMPIN"]   = disp_metrics["Nowcast May-26"] - IMPIN_VAL
    disp_metrics.columns = ["Model", "RMSE", "MAE", "MAPE %", "Dir Acc", "Nowcast May-26", "Gap vs IMPIN"]
    disp_metrics["Dir Acc"] = (disp_metrics["Dir Acc"] * 100).round(1).astype(str) + "%"
    disp_metrics = disp_metrics.sort_values("RMSE").reset_index(drop=True)

    def _highlight_best(row):
        styles = [""] * len(row)
        if row["Model"].startswith("HorizonBlend"):
            styles = ["background-color: #fff0f0; font-weight: bold"] * len(row)
        return styles

    st.dataframe(
        disp_metrics.style
            .apply(_highlight_best, axis=1)
            .format({"RMSE": "{:.1f}", "MAE": "{:.1f}", "MAPE %": "{:.2f}",
                     "Nowcast May-26": "{:.1f}", "Gap vs IMPIN": "{:+.1f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "HorizonBlend decays XGBoost weight exponentially (γ=0.05) so the "
        "Naive + ARIMAX anchor dominates at long horizons, capping drift."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    wf = load_wf()
    metrics3 = load_metrics()

    st.subheader("Walk-forward backtest  (Jan 2021 – Jul 2023, 31 one-step-ahead predictions)")
    st.info(
        "**Key finding:** ARIMAX(1,1,0) is identical to Naive at 1-step-ahead (RMSE = 20.3 for both).  "
        "The 7-month Brent lag means the fuel signal at *t* contributes little to *t+1* WFP prediction.  "
        "Single-split directional accuracy (83%) was regime alignment, not generalizable."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    wf_rmse = {}
    for model in ["Naive", "ARIMAX", "XGBoost", "RF", "DynBlend"]:
        errs = wf["actual"] - wf[model]
        wf_rmse[model] = float(np.sqrt((errs**2).mean()))
    c1.metric("Naive RMSE",    f"{wf_rmse['Naive']:.1f}")
    c2.metric("ARIMAX RMSE",   f"{wf_rmse['ARIMAX']:.1f}", "= Naive")
    c3.metric("XGBoost RMSE",  f"{wf_rmse['XGBoost']:.1f}")
    c4.metric("RF RMSE",       f"{wf_rmse['RF']:.1f}")
    c5.metric("DynBlend RMSE", f"{wf_rmse['DynBlend']:.1f}")

    st.divider()

    left3, right3 = st.columns([3, 2])

    with left3:
        # interactive walk-forward chart
        fig_wf = go.Figure()
        fig_wf.add_trace(go.Scatter(
            x=wf["year_month"], y=wf["actual"],
            name="Actual", mode="lines+markers",
            line=dict(color=COLOURS["Actual"], width=2),
            marker=dict(size=5),
        ))
        wf_model_map = [
            ("Naive",    COLOURS["Naive"],     1, "dot"),
            ("ARIMAX",   COLOURS["ARIMAX"],     1, "dash"),
            ("XGBoost",  COLOURS["XGBoost"],    1, "dash"),
            ("RF",       COLOURS["RF"],          1, "dash"),
            ("DynBlend", COLOURS["DynBlend"],    2, "solid"),
        ]
        for model, colour, width, dash in wf_model_map:
            fig_wf.add_trace(go.Scatter(
                x=wf["year_month"], y=wf[model],
                name=model, mode="lines",
                line=dict(color=colour, width=width, dash=dash),
            ))
        fig_wf.update_layout(
            xaxis_title="Month", yaxis_title="WFP index (NORM_BASE_RAW=100)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=40, b=40), height=440,
            hovermode="x unified",
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    with right3:
        # walk-forward RMSE bar chart
        wf_df = pd.DataFrame({"Model": list(wf_rmse.keys()), "RMSE": list(wf_rmse.values())})
        wf_df = wf_df.sort_values("RMSE")
        fig_bar = px.bar(
            wf_df, x="RMSE", y="Model", orientation="h",
            color="RMSE", color_continuous_scale="Reds_r",
            height=300,
            labels={"RMSE": "1-step-ahead RMSE"},
        )
        fig_bar.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10), yaxis_title=None)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Walk-forward predictions (raw data)")
        st.dataframe(
            wf.rename(columns={"year_month": "Month", "actual": "Actual"})
              .style.format({"Actual": "{:.1f}", "Naive": "{:.1f}", "ARIMAX": "{:.1f}",
                             "XGBoost": "{:.1f}", "RF": "{:.1f}", "DynBlend": "{:.1f}"}),
            use_container_width=True,
            height=280,
            hide_index=True,
        )

    st.divider()
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.subheader("Test-period model comparison")
        img_cmp = PLOTS / "17b_model_comparison.png"
        if img_cmp.exists():
            st.image(str(img_cmp), use_container_width=True)
    with col_img2:
        st.subheader("Walk-forward actual vs models")
        img_wf = PLOTS / "18a_walkforward.png"
        if img_wf.exists():
            st.image(str(img_wf), use_container_width=True)

    st.subheader("HorizonBlend weight evolution over nowcast horizon")
    img_wts = PLOTS / "18b_blend_weights.png"
    if img_wts.exists():
        st.image(str(img_wts), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Macro Drivers
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    macro4 = load_macro()

    c1, c2, c3 = st.columns(3)
    wfp_last_known = macro4[macro4["wfp_food_index"].notna()]["wfp_food_index"].iloc[-1]
    wfp_last_date  = macro4[macro4["wfp_food_index"].notna()]["year_month"].iloc[-1].strftime("%b %Y")
    ghs_latest     = macro4["ghsusd"].iloc[-1]
    ghs_latest_date = macro4["year_month"].iloc[-1].strftime("%b %Y")
    brent_latest   = macro4["brent"].iloc[-1]
    c1.metric(f"WFP index (last known, {wfp_last_date})", f"{wfp_last_known:.1f}")
    c2.metric(f"GHS/USD ({ghs_latest_date})",             f"{ghs_latest:.2f}")
    c3.metric(f"Brent crude ({ghs_latest_date})",         f"${brent_latest:.1f}")

    st.divider()
    st.subheader("Macro drivers over time")

    # subplot: WFP, GHS/USD, Brent
    tab4a, tab4b, tab4c = st.tabs(["WFP Food Index", "GHS/USD Exchange Rate", "Brent Crude"])

    with tab4a:
        fig_wfp = go.Figure()
        obs = macro4[macro4["wfp_food_index"].notna()]
        fig_wfp.add_trace(go.Scatter(
            x=obs["year_month"], y=obs["wfp_food_index"],
            name="WFP food index (observed)", mode="lines+markers",
            line=dict(color="#1f77b4", width=2), marker=dict(size=5),
        ))
        fig_wfp.update_layout(
            xaxis_title="Month", yaxis_title="WFP food index",
            margin=dict(t=20, b=30), height=380, hovermode="x unified",
        )
        st.plotly_chart(fig_wfp, use_container_width=True)

    with tab4b:
        fig_ghs = go.Figure()
        obs_ghs  = macro4[macro4["ghsusd_is_observed"] == True]
        fore_ghs = macro4[macro4["ghsusd_is_observed"] == False]
        fig_ghs.add_trace(go.Scatter(
            x=obs_ghs["year_month"], y=obs_ghs["ghsusd"],
            name="GHS/USD (observed)", mode="lines+markers",
            line=dict(color="#2ca02c", width=2), marker=dict(size=4),
        ))
        if not fore_ghs.empty:
            fig_ghs.add_trace(go.Scatter(
                x=fore_ghs["year_month"], y=fore_ghs["ghsusd"],
                name="GHS/USD (projected)", mode="lines",
                line=dict(color="#2ca02c", width=1.5, dash="dash"),
            ))
        fig_ghs.update_layout(
            xaxis_title="Month", yaxis_title="GHS per USD",
            margin=dict(t=20, b=30), height=380, hovermode="x unified",
        )
        st.plotly_chart(fig_ghs, use_container_width=True)

    with tab4c:
        fig_brent = go.Figure()
        obs_b  = macro4[macro4["brent_is_observed"] == True]
        fore_b = macro4[macro4["brent_is_observed"] == False]
        fig_brent.add_trace(go.Scatter(
            x=obs_b["year_month"], y=obs_b["brent"],
            name="Brent (observed)", mode="lines+markers",
            line=dict(color="#ff7f0e", width=2), marker=dict(size=4),
        ))
        if not fore_b.empty:
            fig_brent.add_trace(go.Scatter(
                x=fore_b["year_month"], y=fore_b["brent"],
                name="Brent (projected)", mode="lines",
                line=dict(color="#ff7f0e", width=1.5, dash="dash"),
            ))
        fig_brent.update_layout(
            xaxis_title="Month", yaxis_title="Brent (USD/barrel)",
            margin=dict(t=20, b=30), height=380, hovermode="x unified",
        )
        st.plotly_chart(fig_brent, use_container_width=True)

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Lag analysis — CCF & PACF")
        img_lag = PLOTS / "17a_lag_analysis.png"
        if img_lag.exists():
            st.image(str(img_lag), use_container_width=True)
        st.caption(
            "Brent crude leads WFP food prices by **7 months** (CCF r = +0.434, p = 0.024).  "
            "Used as the ARIMAX exogenous regressor."
        )

    with col_r:
        st.subheader("WFP anomalous months (dual-layer detection)")
        img_wfp_anom = PLOTS / "19b_wfp_anomalies.png"
        if img_wfp_anom.exists():
            st.image(str(img_wfp_anom), use_container_width=True)
        st.caption(
            "Feb–Apr 2020: COVID-19 shock (+130%, −43%, +86% MoM).  "
            "All three months flagged at 2.79σ–3.88σ.  "
            "Retained in training — the model has survived a real black-swan event."
        )

    st.subheader("Macro panel — raw data")
    disp_macro = macro4.copy()
    disp_macro["year_month"] = disp_macro["year_month"].dt.strftime("%Y-%m")
    disp_macro.rename(columns={
        "year_month": "Month",
        "wfp_food_index": "WFP index",
        "ghsusd": "GHS/USD",
        "brent": "Brent ($/bbl)",
        "ghsusd_is_observed": "GHS observed",
        "brent_is_observed": "Brent observed",
    }, inplace=True)
    st.dataframe(
        disp_macro.style.format({
            "WFP index": "{:.2f}", "GHS/USD": "{:.4f}", "Brent ($/bbl)": "{:.2f}",
        }, na_rep="—"),
        use_container_width=True,
        height=360,
        hide_index=True,
    )
    st.caption("Rows after Jul-2023: WFP index = — (forecast period).  "
               "GHS/USD and Brent: real Yahoo Finance data up to May-2026.")
