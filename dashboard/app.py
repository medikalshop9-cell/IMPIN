"""
IMPIN Streamlit Dashboard  —  Ghana Food Price Index
Premium dark UI  ·  May 2026
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

COLOURS = {
    "Actual":       "#ffffff",
    "Naive":        "#888888",
    "ARIMAX":       "#4fa3e0",
    "ARIMAX+Boost": "#00CED1",
    "HorizonBlend": "#e05c4f",
    "XGBoost":      "#5cb85c",
    "RF":           "#f0a500",
    "Prophet":      "#b57bee",
    "IMPIN":        "#e377c2",
    "DynBlend":     "#c0856a",
}

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GH IMPIN",
    page_icon="🇬🇭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── base dark theme ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #1a1208 !important;
    color: #e8d5b0 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* hide default streamlit chrome */
#MainMenu, footer, [data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stDecoration"] {display:none!important;}
[data-testid="stSidebar"] {display:none!important;}
.block-container {padding: 0 2rem 2rem 2rem !important; max-width: 1400px; margin: auto;}

/* ── live badge ── */
.live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em;
    color: #e05c4f; text-transform: uppercase; margin-bottom: 0.4rem;
}
.live-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #e05c4f;
    animation: pulse 1.8s ease-in-out infinite;
}
@keyframes pulse {
    0%,100%{opacity:1;transform:scale(1)}
    50%{opacity:0.4;transform:scale(0.75)}
}

/* ── hero heading ── */
.hero-title {
    font-size: 2.6rem; font-weight: 800;
    color: #e8d5b0; line-height: 1.1; margin: 0.2rem 0;
}
.hero-sub {
    font-size: 0.82rem; color: #7a6a50; margin-top: 0.4rem;
}

/* ── metric cards (custom HTML) ── */
.metric-card {
    background: #221a0c;
    border: 1px solid #3a2e1a;
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
}
.metric-label {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #7a6a50; margin-bottom: 0.35rem;
}
.metric-value {
    font-size: 1.7rem; font-weight: 800; color: #e8d5b0; line-height: 1;
}
.metric-delta {
    font-size: 0.72rem; color: #7a6a50; margin-top: 0.25rem;
}
.metric-delta.up   {color: #5cb85c;}
.metric-delta.down {color: #e05c4f;}

/* ── category cards ── */
.cat-card {
    background: #221a0c;
    border: 1px solid #3a2e1a;
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    height: 100%;
}
.cat-card.aggregate {border-color: #6a3a2e;}
.cat-label {font-size: 0.62rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#7a6a50;}
.cat-count {font-size: 0.72rem; color:#b8a07a; float:right;}
.cat-icon  {font-size: 1.2rem; margin-bottom:0.4rem;}
.cat-price {font-size: 1.55rem; font-weight:800; color:#e8d5b0; margin-top:0.2rem;}

/* ── section headings ── */
.section-title {
    font-size: 1rem; font-weight: 700; color: #e8d5b0;
    border-left: 3px solid #e05c4f;
    padding-left: 0.75rem; margin: 1.5rem 0 0.75rem 0;
}

/* ── placeholder chart ── */
.chart-placeholder {
    background: #1e1508;
    border: 1px solid #2a2010;
    border-radius: 10px;
    height: 200px;
    display: flex; flex-direction:column;
    align-items: center; justify-content: center;
    color: #4a3e2a; font-size: 0.85rem;
}

/* ── footer ── */
.app-footer {
    display: flex; justify-content: space-between; align-items: center;
    border-top: 1px solid #2a2010;
    padding: 1.2rem 2rem; margin: 2rem -2rem 0 -2rem;
    font-size: 0.72rem; color: #4a3e2a;
}
.footer-links {display:flex; gap:1.5rem;}
.footer-links span {cursor:pointer; color:#5a4e3a;}
.footer-links span:hover {color:#b8a07a;}

/* ── streamlit native metric override ── */
[data-testid="stMetric"] {
    background: #221a0c !important;
    border: 1px solid #3a2e1a !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] p {
    color: #7a6a50 !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #e8d5b0 !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
}
[data-testid="stMetricDelta"] {color: #7a6a50 !important;}

/* ── info box ── */
[data-testid="stAlert"] {
    background: #2a1e0c !important;
    border-color: #e05c4f !important;
    border-radius: 8px !important;
    color: #e8d5b0 !important;
}

/* ── inputs ── */
[data-testid="stSelectbox"] > div > div {background: #221a0c !important; color: #e8d5b0 !important;}
label, [data-testid="stWidgetLabel"] p {color: #b8a07a !important;}
[data-testid="stSlider"] {background: none !important;}

/* ── dataframe ── */
[data-testid="stDataFrame"] {border: 1px solid #3a2e1a; border-radius: 8px; overflow: hidden;}

/* ── tab bar override ── */
[data-testid="stTabs"] button {
    color: #7a6a50 !important;
    background: none !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e8d5b0 !important;
    border-bottom-color: #e05c4f !important;
}

/* ── nav buttons ── */
div[data-testid="stHorizontalBlock"] > div > div > div > button {
    border: none !important;
    border-radius: 4px !important;
    font-size: 0.8rem !important;
}

/* ── divider ── */
hr {border-color: #2a2010 !important;}
</style>
""", unsafe_allow_html=True)

# ── plotly dark template ──────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1e1508",
    font=dict(color="#b8a07a", family="Inter, Segoe UI, sans-serif"),
    xaxis=dict(gridcolor="#2a2010", linecolor="#3a2e1a", zerolinecolor="#2a2010"),
    yaxis=dict(gridcolor="#2a2010", linecolor="#3a2e1a", zerolinecolor="#2a2010"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#b8a07a")),
    hoverlabel=dict(bgcolor="#2a1e0c", bordercolor="#3a2e1a", font=dict(color="#e8d5b0")),
)

def apply_dark(fig, **kwargs):
    fig.update_layout(**PLOTLY_LAYOUT, **kwargs)
    return fig

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
    return pd.read_csv(NOWCAST, parse_dates=["year_month"])

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

# ── navigation state ──────────────────────────────────────────────────────────
PAGES     = ["IMPIN Live", "Price Outlook", "Nowcast", "Model Evaluation", "Macro Drivers"]
PAGE_KEYS = ["live", "calc", "nowcast", "eval", "macro"]

if "page" not in st.session_state:
    st.session_state.page = "live"

# ── nav bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.nav-outer {
    display: flex; align-items: center; justify-content: space-between;
    background: #1a1208; border-bottom: 1px solid #3a2e1a;
    padding: 0.6rem 0; margin-bottom: 1.5rem;
}
.nav-logo-txt {
    font-size: 1.05rem; font-weight: 700; color: #e05c4f; letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)

nav_cols = st.columns([2, 7, 1])
with nav_cols[0]:
    st.markdown('<div class="nav-logo-txt">GH IMPIN</div>', unsafe_allow_html=True)
with nav_cols[1]:
    btn_cols = st.columns(len(PAGES))
    for col, label, key in zip(btn_cols, PAGES, PAGE_KEYS):
        with col:
            is_active = st.session_state.page == key
            if is_active:
                st.markdown(
                    f'<div style="text-align:center;background:#e8d5b0;color:#1a1208;'
                    f'border-radius:4px;padding:0.3rem 0.5rem;font-size:0.78rem;font-weight:700">{label}</div>',
                    unsafe_allow_html=True
                )
            else:
                if st.button(label, key=f"nav_{key}", use_container_width=True):
                    st.session_state.page = key
                    st.rerun()

st.markdown('<hr style="margin:0 0 1.5rem 0">', unsafe_allow_html=True)

page = st.session_state.page

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — IMPIN LIVE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "live":
    scrape    = load_scrape()
    anomaly   = load_anomaly()
    impin_cat = load_impin_clean()

    # hero
    st.markdown("""
    <div class="live-badge"><div class="live-dot"></div>LIVE MONITORING ACTIVE</div>
    <div class="hero-title">GH IMPIN &mdash; Ghana Food Price Index</div>
    <div class="hero-sub">Import Price Index &bull; May 2026 snapshot &bull; WFP food index base 2019-08 = 100</div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    n_flagged  = int(anomaly["is_flagged"].sum())
    n_included = int(anomaly["is_flagged"].eq(False).sum())

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">IMPIN May 2026</div>
          <div class="metric-value">{IMPIN_VAL:.1f}</div>
          <div class="metric-delta">&#8599; base = 100</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Products in index</div>
          <div class="metric-value">{n_included:,}</div>
          <div class="metric-delta">&nbsp;</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Removed Outliers</div>
          <div class="metric-value">{n_flagged}</div>
          <div class="metric-delta">&#9888; {n_flagged/len(anomaly)*100:.1f}% excluded</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Scrape Date</div>
          <div class="metric-value" style="font-size:1.3rem">2026-05-05</div>
          <div class="metric-delta">&nbsp;</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Live Price Index &mdash; May 2026 (base = 100)</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.78rem;color:#7a6a50;margin-bottom:1rem">May 2026 scraped prices are the index base. Each category shows its median observed price and product count after outlier removal.</p>', unsafe_allow_html=True)

    CAT_ICONS = {
        "Clothing & Personal Care": "&#128085;",
        "Food & Beverages": "&#127869;",
        "General": "&#128722;",
        "Household": "&#127968;",
        "Personal Care": "&#128134;",
    }

    cat_cols = st.columns(len(impin_cat) + 1)
    for i, row in impin_cat.iterrows():
        icon = CAT_ICONS.get(row["category"], "&#128230;")
        with cat_cols[i]:
            st.markdown(f"""<div class="cat-card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div class="cat-icon">{icon}</div>
                <div class="cat-count">&#8599; {int(row['n_products'])} products</div>
              </div>
              <div class="cat-label">{row['category']}</div>
              <div class="cat-price">GHS {row['median_price_ghc']:.2f}</div>
            </div>""", unsafe_allow_html=True)

    agg_price = float(scrape["price_ghc"].median())
    with cat_cols[-1]:
        st.markdown(f"""<div class="cat-card aggregate">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div class="cat-icon">&#128202;</div>
            <div style="background:#3a2010;border-radius:3px;padding:1px 5px;font-size:0.6rem;font-weight:700;color:#e05c4f">AGGREGATE</div>
          </div>
          <div style="font-size:0.62rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#7a6a50">ALL (Equal-Weight)</div>
          <div style="font-size:1.1rem;font-weight:700;color:#b8a07a;margin-top:0.4rem">&#8599; {n_included:,} products</div>
          <div style="font-size:1.3rem;font-weight:800;color:#e8d5b0">GHS {agg_price:.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""<div class="chart-placeholder">
      <div style="font-size:2rem;opacity:0.3">&#128202;</div>
      <div style="margin-top:0.5rem;font-weight:600">Temporal Trend Analysis Placeholder</div>
      <div style="font-size:0.72rem;margin-top:0.25rem">Market depth and volatility metrics pending next scrape</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">All scraped commodities</div>', unsafe_allow_html=True)

    filt_cat = st.selectbox("Filter by category",
        ["All"] + sorted(anomaly["cpi_category"].unique().tolist()), key="live_cat_filter")
    filt_status = st.radio("Show", ["All products","Included only","Removed only"],
                           horizontal=True, key="live_status_filter")

    prod_df = anomaly.copy()
    prod_df["Status"] = prod_df["is_flagged"].map({False: "✓ Included", True: "⚠ Removed (outlier)"})
    if filt_cat != "All":
        prod_df = prod_df[prod_df["cpi_category"] == filt_cat]
    if filt_status == "Included only":
        prod_df = prod_df[prod_df["is_flagged"] == False]
    elif filt_status == "Removed only":
        prod_df = prod_df[prod_df["is_flagged"] == True]

    prod_disp = prod_df[["product_name","cpi_category","source","price_ghc","z_score","Status"]].copy()
    prod_disp.columns = ["Product","Category","Source","Price (GHS)","Z-score","Status"]
    prod_disp = prod_disp.sort_values(["Category","Price (GHS)"]).reset_index(drop=True)

    def _colour_status(val):
        return "color:#cc4444;font-weight:bold" if "Removed" in str(val) else "color:#5cb85c"

    st.dataframe(
        prod_disp.style.map(_colour_status, subset=["Status"])
            .format({"Price (GHS)":"{:.2f}","Z-score":"{:.2f}"}),
        use_container_width=True, height=420, hide_index=True,
    )
    st.markdown('<p style="font-size:0.72rem;color:#4a3e2a">Outliers flagged by ≥ 2/3 of: Z-score (|z|>2.5), IQR fence (k=2.0), Isolation Forest (contamination=5%).</p>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Products per source</div>', unsafe_allow_html=True)
        src_counts = scrape["source"].value_counts().reset_index()
        src_counts.columns = ["Source","Count"]
        fig_src = px.bar(src_counts, x="Count", y="Source", orientation="h",
                         color="Count", color_continuous_scale=["#3a2010","#e05c4f"], height=260)
        fig_src.update_layout(coloraxis_showscale=False, margin=dict(l=10,r=10,t=10,b=10), yaxis_title=None)
        apply_dark(fig_src)
        st.plotly_chart(fig_src, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Price distribution by category (log)</div>', unsafe_allow_html=True)
        fig_box = px.box(scrape, x="cpi_category", y="price_ghc", log_y=True,
                         color="cpi_category",
                         color_discrete_sequence=["#e05c4f","#4fa3e0","#5cb85c","#f0a500","#b57bee"],
                         height=260)
        fig_box.update_layout(showlegend=False, margin=dict(t=10,b=10))
        apply_dark(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('<div class="section-title">Clean index vs raw &mdash; by category</div>', unsafe_allow_html=True)
    img_clean = PLOTS / "19c_impin_clean.png"
    if img_clean.exists():
        st.image(str(img_clean), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRICE CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "calc":
    nowcast_pc = load_nowcast()
    macro_pc   = load_macro()
    anomaly_pc = load_anomaly()

    st.markdown("""
    <div class="hero-title" style="font-size:2rem">&#128176; Price Outlook</div>
    <div class="hero-sub">Estimate future product prices from the IMPIN nowcast</div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    st.info(
        "**How it works:** May 2026 scraped prices are the base (IMPIN = 100). "
        "The model forecasts how the index moves from that base.\n\n"
        "**Formula:** Estimated price = (Forecast index ÷ 100) × Current price today"
    )

    clean_prods = anomaly_pc[anomaly_pc["is_flagged"] == False].copy()
    clean_prods = clean_prods.sort_values(["cpi_category","product_name"])

    c1, c2 = st.columns([2,3])
    with c1:
        sel_cat = st.selectbox("Category", sorted(clean_prods["cpi_category"].unique().tolist()), key="pc_cat")
    cat_prods = clean_prods[clean_prods["cpi_category"] == sel_cat]
    with c2:
        sel_prod = st.selectbox("Product", cat_prods["product_name"].tolist(), key="pc_prod")
    current_price = float(cat_prods.loc[cat_prods["product_name"] == sel_prod, "price_ghc"].values[0])

    forecast_months = nowcast_pc["year_month"].dt.strftime("%b %Y").tolist()
    sel_month_label = st.select_slider("Forecast month", options=forecast_months,
                                       value=forecast_months[-1], key="pc_month")
    sel_month_idx   = forecast_months.index(sel_month_label)
    forecast_index  = float(nowcast_pc.iloc[sel_month_idx]["blend_norm"])
    estimated_price = (forecast_index / 100.0) * current_price
    pct_change      = forecast_index - 100.0

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    r1.metric("Current price (May 2026)", f"GHS {current_price:.2f}")
    r2.metric(f"Estimated ({sel_month_label})", f"GHS {estimated_price:.2f}",
              f"{pct_change:+.1f}% vs today", delta_color="inverse")
    r3.metric("IMPIN forecast index", f"{forecast_index:.1f}", "base = 100")

    st.markdown('<div class="section-title">Price trajectory</div>', unsafe_allow_html=True)
    prices_over_time = (nowcast_pc["blend_norm"] / 100.0) * current_price
    fig_pc = go.Figure()
    fig_pc.add_trace(go.Scatter(x=nowcast_pc["year_month"], y=prices_over_time,
        mode="lines", name="Estimated price",
        line=dict(color=COLOURS["HorizonBlend"], width=2),
        fill="tozeroy", fillcolor="rgba(224,92,79,0.08)"))
    fig_pc.add_trace(go.Scatter(x=[pd.Timestamp("2026-05-01")], y=[current_price],
        mode="markers", name="Today", marker=dict(symbol="circle", size=12, color=COLOURS["IMPIN"])))
    sel_ts = nowcast_pc.iloc[sel_month_idx]["year_month"]
    fig_pc.add_trace(go.Scatter(x=[sel_ts], y=[estimated_price],
        mode="markers", name=sel_month_label, marker=dict(symbol="star", size=14, color="#e05c4f")))
    fig_pc.update_layout(xaxis_title="Month", yaxis_title="Estimated price (GHS)",
                         hovermode="x unified", margin=dict(t=20,b=30), height=360)
    apply_dark(fig_pc)
    st.plotly_chart(fig_pc, use_container_width=True)

    st.markdown('<div class="section-title">Why is the price changing?</div>', unsafe_allow_html=True)

    sel_ts_pd      = nowcast_pc.iloc[sel_month_idx]["year_month"]
    macro_row      = macro_pc[macro_pc["year_month"] == sel_ts_pd]
    macro_now      = macro_pc[macro_pc["year_month"] == pd.Timestamp("2026-05-01")]
    brent_lag_date = sel_ts_pd - pd.DateOffset(months=6)
    brent_lag_row  = macro_pc[macro_pc["year_month"] == brent_lag_date]
    may26_row      = macro_pc[macro_pc["year_month"] == pd.Timestamp("2026-05-01")]

    ghs_now       = float(macro_now["ghsusd"].values[0])      if not macro_now.empty      else None
    ghs_then      = float(macro_row["ghsusd"].values[0])      if not macro_row.empty      else None
    brent_lag_val = float(brent_lag_row["brent"].values[0])   if not brent_lag_row.empty  else None
    brent_now_val = float(may26_row["brent"].values[0])       if not may26_row.empty      else None

    reason_parts = []
    if ghs_now and ghs_then:
        direction = "weaker" if ghs_then > ghs_now else "stronger"
        reason_parts.append(f"**GHS/USD:** {ghs_now:.2f} today → {ghs_then:.2f} in {sel_month_label} (cedi {direction})")
    if brent_lag_val and brent_now_val:
        reason_parts.append(
            f"**Brent crude** 6 months before {sel_month_label}: ${brent_lag_val:.1f}/bbl "
            f"(vs ${brent_now_val:.1f}/bbl today) — "
            f"{'higher' if brent_lag_val > brent_now_val else 'lower'} oil → "
            f"{'higher' if brent_lag_val > brent_now_val else 'lower'} import costs"
        )

    expl_col1, expl_col2 = st.columns(2)
    with expl_col1:
        st.markdown(
            f"**{sel_prod}**  \n"
            f"Today: **GHS {current_price:.2f}**  \n"
            f"Forecast ({sel_month_label}): **GHS {estimated_price:.2f}** ({pct_change:+.1f}%)  \n\n"
            + "  \n".join(f"- {r}" for r in reason_parts)
        )
        st.caption("Oil price rises → transport & import costs rise (~6 months later) → traders raise food prices → WFP index rises → IMPIN rises.")
    with expl_col2:
        fig_ghs_mini = go.Figure()
        ghs_obs  = macro_pc[macro_pc["ghsusd_is_observed"] == True]
        ghs_fore = macro_pc[macro_pc["ghsusd_is_observed"] == False]
        fig_ghs_mini.add_trace(go.Scatter(x=ghs_obs["year_month"], y=ghs_obs["ghsusd"],
            name="GHS/USD (observed)", mode="lines", line=dict(color="#5cb85c", width=2)))
        if not ghs_fore.empty:
            fig_ghs_mini.add_trace(go.Scatter(x=ghs_fore["year_month"], y=ghs_fore["ghsusd"],
                name="Projected", mode="lines", line=dict(color="#5cb85c", width=1.5, dash="dash")))
        if ghs_then:
            fig_ghs_mini.add_trace(go.Scatter(x=[sel_ts_pd], y=[ghs_then],
                mode="markers", name=sel_month_label, marker=dict(symbol="star", size=12, color="#e05c4f")))
        fig_ghs_mini.update_layout(title="GHS/USD exchange rate", xaxis_title=None,
                                   yaxis_title="GHS per USD", margin=dict(t=40,b=20),
                                   height=260, showlegend=False, hovermode="x unified")
        apply_dark(fig_ghs_mini)
        st.plotly_chart(fig_ghs_mini, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — NOWCAST
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "nowcast":
    nowcast = load_nowcast()
    metrics = load_metrics()
    macro   = load_macro()

    st.markdown("""
    <div class="hero-title" style="font-size:2rem">&#128200; Nowcast</div>
    <div class="hero-sub">HorizonBlend forecast &bull; Aug 2023 &rarr; May 2026</div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    blend_may26 = float(nowcast.loc[nowcast["year_month"] == "2026-05-01", "blend_norm"].values[-1])
    gap  = blend_may26 - IMPIN_VAL
    best = metrics.loc[metrics["test_rmse"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HorizonBlend — May 2026", f"{blend_may26:.1f}", f"gap {gap:+.1f} vs IMPIN")
    c2.metric("IMPIN live anchor", f"{IMPIN_VAL:.1f}", "scraped May 2026")
    c3.metric("Best test RMSE", f"{best['test_rmse']:.1f}", best["model"])
    c4.metric("Nowcast horizon", "34 months", "Aug 2023 → May 2026")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    show_all_models = st.toggle("Show all models (examiner view)", value=False)

    if not show_all_models:
        st.markdown('<div class="section-title">IMPIN Nowcast &mdash; HorizonBlend</div>', unsafe_allow_html=True)
        st.markdown(
            "> **HorizonBlend** combines Naive, ARIMAX, and XGBoost — weighted by recent forecast accuracy, "
            "> with automatic adjustment so short-run ML signals dominate early and stable statistical anchors "
            "> dominate at long range."
        )
    else:
        st.markdown('<div class="section-title">Full nowcast &mdash; all models</div>', unsafe_allow_html=True)

    wfp_actual = macro[macro["wfp_food_index"].notna()][["year_month","wfp_food_index"]].copy()
    norm_base  = float(wfp_actual.iloc[-1]["wfp_food_index"])
    wfp_actual["norm"] = wfp_actual["wfp_food_index"] / norm_base * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wfp_actual["year_month"], y=wfp_actual["norm"],
        name="WFP Actual", mode="lines+markers",
        line=dict(color=COLOURS["Actual"], width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=[pd.Timestamp("2026-05-01")], y=[IMPIN_VAL],
        name="IMPIN (live)", mode="markers",
        marker=dict(symbol="star", size=14, color=COLOURS["IMPIN"])))
    fig.add_trace(go.Scatter(x=nowcast["year_month"], y=nowcast["blend_norm"],
        name="HorizonBlend (γ=0.05)", mode="lines",
        line=dict(color=COLOURS["HorizonBlend"], width=2.5)))

    if show_all_models:
        for col, name, colour, dash in [
            ("naive_norm",   "Naive",              COLOURS["Naive"],        "dot"),
            ("arimax_norm",  "ARIMAX(1,1,0) lag7", COLOURS["ARIMAX"],       "dash"),
            ("boost_norm",   "ARIMAX+Boost",        COLOURS["ARIMAX+Boost"],"dash"),
            ("xgb_norm",     "XGBoost",             COLOURS["XGBoost"],     "dash"),
            ("rf_norm",      "Random Forest",       COLOURS["RF"],          "dash"),
            ("prophet_norm", "Prophet",             COLOURS["Prophet"],     "dash"),
        ]:
            fig.add_trace(go.Scatter(x=nowcast["year_month"], y=nowcast[col],
                name=name, mode="lines", line=dict(color=colour, width=1, dash=dash)))

    fig.add_vline(x=pd.Timestamp("2023-07-01").timestamp()*1000,
                  line_dash="dot", line_color="#4a3e2a",
                  annotation_text="Forecast start", annotation_font_color="#7a6a50")
    fig.add_hline(y=100, line_dash="dot", line_color=COLOURS["IMPIN"],
                  annotation_text="IMPIN=100", annotation_position="right",
                  annotation_font_color="#e377c2")
    fig.update_layout(xaxis_title="Month", yaxis_title="Index (NORM_BASE_RAW = 100)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      margin=dict(t=40,b=40), height=500, hovermode="x unified")
    apply_dark(fig)
    st.plotly_chart(fig, use_container_width=True)

    if show_all_models:
        img_nc = PLOTS / "17c_full_nowcast.png"
        if img_nc.exists():
            st.markdown('<div class="section-title">Static nowcast chart (full detail)</div>', unsafe_allow_html=True)
            st.image(str(img_nc), use_container_width=True)

    st.markdown('<div class="section-title">Test-period metrics (Jul 2022 &ndash; Jul 2023, 13 steps)</div>', unsafe_allow_html=True)

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
    disp_metrics.columns = ["Model","RMSE","MAE","MAPE %","Dir Acc","Nowcast May-26","Gap vs IMPIN"]
    disp_metrics["Dir Acc"] = (disp_metrics["Dir Acc"]*100).round(1).astype(str) + "%"
    disp_metrics = disp_metrics.sort_values("RMSE").reset_index(drop=True)

    def _highlight_best(row):
        return ["background-color:#2a1508;font-weight:bold"] * len(row) \
               if row["Model"].startswith("HorizonBlend") else [""] * len(row)

    st.dataframe(
        disp_metrics.style.apply(_highlight_best, axis=1)
            .format({"RMSE":"{:.1f}","MAE":"{:.1f}","MAPE %":"{:.2f}",
                     "Nowcast May-26":"{:.1f}","Gap vs IMPIN":"{:+.1f}"}),
        use_container_width=True, hide_index=True,
    )
    st.caption("HorizonBlend decays XGBoost weight exponentially (γ=0.05) so the Naive + ARIMAX anchor dominates at long horizons, capping drift.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "eval":
    wf       = load_wf()
    metrics3 = load_metrics()

    st.markdown("""
    <div class="hero-title" style="font-size:2rem">&#128300; Model Evaluation</div>
    <div class="hero-sub">Walk-forward backtest &bull; Jan 2021 &ndash; Jul 2023 &bull; 31 one-step-ahead predictions</div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    st.info(
        "**Key finding:** ARIMAX(1,1,0) is identical to Naive at 1-step-ahead (RMSE = 20.3 for both). "
        "The 7-month Brent lag means the fuel signal at *t* contributes little to *t+1* WFP prediction."
    )

    wf_rmse = {}
    for model in ["Naive","ARIMAX","XGBoost","RF","DynBlend"]:
        errs = wf["actual"] - wf[model]
        wf_rmse[model] = float(np.sqrt((errs**2).mean()))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Naive RMSE",    f"{wf_rmse['Naive']:.1f}")
    c2.metric("ARIMAX RMSE",   f"{wf_rmse['ARIMAX']:.1f}", "= Naive")
    c3.metric("XGBoost RMSE",  f"{wf_rmse['XGBoost']:.1f}")
    c4.metric("RF RMSE",       f"{wf_rmse['RF']:.1f}")
    c5.metric("DynBlend RMSE", f"{wf_rmse['DynBlend']:.1f}")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    left3, right3 = st.columns([3,2])

    with left3:
        st.markdown('<div class="section-title">Walk-forward predictions</div>', unsafe_allow_html=True)
        fig_wf = go.Figure()
        fig_wf.add_trace(go.Scatter(x=wf["year_month"], y=wf["actual"],
            name="Actual", mode="lines+markers",
            line=dict(color=COLOURS["Actual"], width=2), marker=dict(size=5)))
        for model, colour, dash in [
            ("Naive",    COLOURS["Naive"],    "dot"),
            ("ARIMAX",   COLOURS["ARIMAX"],   "dash"),
            ("XGBoost",  COLOURS["XGBoost"],  "dash"),
            ("RF",       COLOURS["RF"],        "dash"),
            ("DynBlend", COLOURS["DynBlend"], "solid"),
        ]:
            fig_wf.add_trace(go.Scatter(x=wf["year_month"], y=wf[model],
                name=model, mode="lines", line=dict(color=colour, width=1.5, dash=dash)))
        fig_wf.update_layout(xaxis_title="Month", yaxis_title="WFP index",
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                             margin=dict(t=40,b=40), height=420, hovermode="x unified")
        apply_dark(fig_wf)
        st.plotly_chart(fig_wf, use_container_width=True)

    with right3:
        st.markdown('<div class="section-title">RMSE comparison</div>', unsafe_allow_html=True)
        wf_df = pd.DataFrame({"Model":list(wf_rmse.keys()), "RMSE":list(wf_rmse.values())}).sort_values("RMSE")
        fig_bar = px.bar(wf_df, x="RMSE", y="Model", orientation="h",
                         color="RMSE", color_continuous_scale=["#3a2010","#e05c4f"], height=260)
        fig_bar.update_layout(coloraxis_showscale=False, margin=dict(t=10,b=10), yaxis_title=None)
        apply_dark(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('<div class="section-title">Raw predictions</div>', unsafe_allow_html=True)
        st.dataframe(
            wf.rename(columns={"year_month":"Month","actual":"Actual"})
              .style.format({"Actual":"{:.1f}","Naive":"{:.1f}","ARIMAX":"{:.1f}",
                             "XGBoost":"{:.1f}","RF":"{:.1f}","DynBlend":"{:.1f}"}),
            use_container_width=True, height=280, hide_index=True,
        )

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown('<div class="section-title">Test-period model comparison</div>', unsafe_allow_html=True)
        img_cmp = PLOTS / "17b_model_comparison.png"
        if img_cmp.exists():
            st.image(str(img_cmp), use_container_width=True)
    with col_img2:
        st.markdown('<div class="section-title">Walk-forward actual vs models</div>', unsafe_allow_html=True)
        img_wf2 = PLOTS / "18a_walkforward.png"
        if img_wf2.exists():
            st.image(str(img_wf2), use_container_width=True)

    st.markdown('<div class="section-title">HorizonBlend weight evolution</div>', unsafe_allow_html=True)
    img_wts = PLOTS / "18b_blend_weights.png"
    if img_wts.exists():
        st.image(str(img_wts), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MACRO DRIVERS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "macro":
    macro4 = load_macro()

    st.markdown("""
    <div class="hero-title" style="font-size:2rem">&#127758; Macro Drivers</div>
    <div class="hero-sub">WFP index &bull; GHS/USD &bull; Brent crude &bull; lag analysis</div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    wfp_last_known  = macro4[macro4["wfp_food_index"].notna()]["wfp_food_index"].iloc[-1]
    wfp_last_date   = macro4[macro4["wfp_food_index"].notna()]["year_month"].iloc[-1].strftime("%b %Y")
    ghs_latest      = macro4["ghsusd"].iloc[-1]
    ghs_latest_date = macro4["year_month"].iloc[-1].strftime("%b %Y")
    brent_latest    = macro4["brent"].iloc[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric(f"WFP index ({wfp_last_date})",    f"{wfp_last_known:.1f}")
    c2.metric(f"GHS/USD ({ghs_latest_date})",     f"{ghs_latest:.2f}")
    c3.metric(f"Brent crude ({ghs_latest_date})", f"${brent_latest:.1f}")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Macro drivers over time</div>', unsafe_allow_html=True)

    tab4a, tab4b, tab4c = st.tabs(["WFP Food Index","GHS/USD Exchange Rate","Brent Crude"])

    with tab4a:
        obs = macro4[macro4["wfp_food_index"].notna()]
        fig_wfp = go.Figure()
        fig_wfp.add_trace(go.Scatter(x=obs["year_month"], y=obs["wfp_food_index"],
            name="WFP food index", mode="lines+markers",
            line=dict(color="#4fa3e0", width=2), marker=dict(size=5)))
        fig_wfp.update_layout(xaxis_title="Month", yaxis_title="WFP food index",
                              margin=dict(t=20,b=30), height=360, hovermode="x unified")
        apply_dark(fig_wfp)
        st.plotly_chart(fig_wfp, use_container_width=True)

    with tab4b:
        obs_ghs  = macro4[macro4["ghsusd_is_observed"] == True]
        fore_ghs = macro4[macro4["ghsusd_is_observed"] == False]
        fig_ghs = go.Figure()
        fig_ghs.add_trace(go.Scatter(x=obs_ghs["year_month"], y=obs_ghs["ghsusd"],
            name="GHS/USD (observed)", mode="lines+markers",
            line=dict(color="#5cb85c", width=2), marker=dict(size=4)))
        if not fore_ghs.empty:
            fig_ghs.add_trace(go.Scatter(x=fore_ghs["year_month"], y=fore_ghs["ghsusd"],
                name="Projected", mode="lines",
                line=dict(color="#5cb85c", width=1.5, dash="dash")))
        fig_ghs.update_layout(xaxis_title="Month", yaxis_title="GHS per USD",
                              margin=dict(t=20,b=30), height=360, hovermode="x unified")
        apply_dark(fig_ghs)
        st.plotly_chart(fig_ghs, use_container_width=True)

    with tab4c:
        obs_b  = macro4[macro4["brent_is_observed"] == True]
        fore_b = macro4[macro4["brent_is_observed"] == False]
        fig_brent = go.Figure()
        fig_brent.add_trace(go.Scatter(x=obs_b["year_month"], y=obs_b["brent"],
            name="Brent (observed)", mode="lines+markers",
            line=dict(color="#f0a500", width=2), marker=dict(size=4)))
        if not fore_b.empty:
            fig_brent.add_trace(go.Scatter(x=fore_b["year_month"], y=fore_b["brent"],
                name="Projected", mode="lines",
                line=dict(color="#f0a500", width=1.5, dash="dash")))
        fig_brent.update_layout(xaxis_title="Month", yaxis_title="Brent (USD/barrel)",
                                margin=dict(t=20,b=30), height=360, hovermode="x unified")
        apply_dark(fig_brent)
        st.plotly_chart(fig_brent, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-title">Lag analysis &mdash; CCF & PACF</div>', unsafe_allow_html=True)
        img_lag = PLOTS / "17a_lag_analysis.png"
        if img_lag.exists():
            st.image(str(img_lag), use_container_width=True)
        st.caption("Brent crude leads WFP food prices by **7 months** (CCF r = +0.434, p = 0.024). Used as the ARIMAX exogenous regressor.")
    with col_r:
        st.markdown('<div class="section-title">WFP anomalous months</div>', unsafe_allow_html=True)
        img_wfp_anom = PLOTS / "19b_wfp_anomalies.png"
        if img_wfp_anom.exists():
            st.image(str(img_wfp_anom), use_container_width=True)
        st.caption("Feb–Apr 2020: COVID-19 shock (+130%, −43%, +86% MoM). All three months flagged at 2.79σ–3.88σ. Retained in training.")

    st.markdown('<div class="section-title">Macro panel &mdash; raw data</div>', unsafe_allow_html=True)
    disp_macro = macro4.copy()
    disp_macro["year_month"] = disp_macro["year_month"].dt.strftime("%Y-%m")
    disp_macro.rename(columns={"year_month":"Month","wfp_food_index":"WFP index",
        "ghsusd":"GHS/USD","brent":"Brent ($/bbl)",
        "ghsusd_is_observed":"GHS observed","brent_is_observed":"Brent observed"}, inplace=True)
    st.dataframe(
        disp_macro.style.format({"WFP index":"{:.2f}","GHS/USD":"{:.4f}","Brent ($/bbl)":"{:.2f}"}, na_rep="—"),
        use_container_width=True, height=360, hide_index=True,
    )
    st.caption("Rows after Jul-2023: WFP index = — (forecast period). GHS/USD and Brent: real Yahoo Finance data up to May-2026.")

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  <div>
    <strong style="color:#5a4e3a">GH IMPIN</strong>
    <span style="margin-left:0.75rem">&bull; Institutional Grade Food Price Monitoring</span>
  </div>
  <div class="footer-links">
    <span>METHODOLOGY</span>
    <span>DATA EXPORT</span>
    <span>API ACCESS</span>
    <span>LEGAL</span>
  </div>
</div>
""", unsafe_allow_html=True)
