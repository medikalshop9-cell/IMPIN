"""
IMPIN — Streamlit Dashboard
============================
Three tabs:
  1. Live Index  — IMPIN scrape-based price snapshot + basket breakdown
  2. Forecast    — Historical WFP index + model comparison + best-model forecast
  3. Anomalies   — Flagged products table + z-score chart

Run:
  streamlit run app/dashboard.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── paths ──
ROOT = Path(__file__).resolve().parent.parent
SCRAPE = ROOT / "data" / "raw" / "scraped_combined.csv"
PANEL = ROOT / "data" / "processed" / "historical_panel.parquet"
ANOMALY = ROOT / "outputs" / "anomaly_report.csv"
COMPARISON = ROOT / "models" / "results" / "comparison.csv"
ARIMAX_CSV = ROOT / "models" / "results" / "arimax_comparison.csv"

st.set_page_config(
    page_title="IMPIN — Ghana Food Price Intelligence",
    page_icon="🇬🇭",
    layout="wide",
)

# ── sidebar ──
st.sidebar.title("IMPIN")
st.sidebar.markdown(
    "**I**nformal **M**arket **P**rice **I**ntelligence **N**etwork\n\n"
    "Real-time food price intelligence for Ghana."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Data sources**")
st.sidebar.markdown(
    "- Live scrape: Konzoom, Makola, Big Samps, Shopnaw, KiKUU\n"
    "- Historical: WFP VAM Ghana monthly\n"
    "- Exogenous: GHS/USD, Brent crude"
)


# ════════════════════════════════════════════════════════════
# helpers
# ════════════════════════════════════════════════════════════

@st.cache_data
def load_scrape():
    df = pd.read_csv(SCRAPE)
    df["price_ghc"] = pd.to_numeric(df["price_ghc"], errors="coerce")
    return df.dropna(subset=["price_ghc"])

@st.cache_data
def load_panel():
    df = pd.read_parquet(PANEL)
    df = df.dropna(subset=["wfp_food_index", "ghsusd", "brent"])
    df["ds"] = pd.to_datetime(df["year_month"])
    return df.sort_values("ds")

@st.cache_data
def load_anomaly():
    return pd.read_csv(ANOMALY)

@st.cache_data
def load_comparison():
    return pd.read_csv(COMPARISON)


# ════════════════════════════════════════════════════════════
# Tab definitions
# ════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["📊 Live Index", "🔮 Forecast", "⚠️ Anomalies"])

# ────────────────────────────────────────────────────────────
# TAB 1 — Live Index
# ────────────────────────────────────────────────────────────
with tab1:
    st.title("🇬🇭 IMPIN — Live Price Index Snapshot")
    st.markdown(
        "_IMPIN scrape-based price index (base Jan-2026 = 100), "
        "updated weekly from 5 online Ghanaian retailers._"
    )

    df_s = load_scrape()

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Products scraped", f"{len(df_s):,}")
    col2.metric("Sources", df_s["source"].nunique())
    col3.metric("Categories", df_s["cpi_category"].nunique())
    col4.metric("Median price (GHC)", f"{df_s['price_ghc'].median():,.0f}")

    st.markdown("---")

    # ── CPI basket weights ──
    BASKET = {
        "Food & Beverages": 42.5,
        "General": 40.5,
        "Household": 9.8,
        "Clothing & Personal Care": 7.2,
    }

    # count and median price per category
    stats = (
        df_s.groupby("cpi_category")["price_ghc"]
        .agg(count="count", median="median", mean="mean")
        .reset_index()
    )
    stats["basket_weight"] = stats["cpi_category"].map(BASKET).fillna(0)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Products per Source")
        src_counts = df_s["source"].value_counts().reset_index()
        src_counts.columns = ["source", "count"]
        fig_src = px.bar(
            src_counts, x="source", y="count",
            color="source",
            title="Products scraped per source",
            labels={"count": "# products"},
        )
        fig_src.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_src, use_container_width=True)

    with col_b:
        st.subheader("CPI Basket Weights")
        basket_df = pd.DataFrame(list(BASKET.items()), columns=["Category", "Weight (%)"])
        fig_basket = px.pie(
            basket_df, names="Category", values="Weight (%)",
            title="CPI basket composition",
            hole=0.35,
        )
        fig_basket.update_layout(height=350)
        st.plotly_chart(fig_basket, use_container_width=True)

    # ── price distribution per category ──
    st.subheader("Price Distribution by CPI Category")
    cats = sorted(df_s["cpi_category"].dropna().unique())
    sel_cat = st.selectbox("Select category", ["All"] + list(cats))
    plot_data = df_s if sel_cat == "All" else df_s[df_s["cpi_category"] == sel_cat]
    fig_box = px.box(
        plot_data, x="cpi_category", y="price_ghc",
        color="source", log_y=True,
        title=f"Price distribution ({'All categories' if sel_cat == 'All' else sel_cat})",
        labels={"price_ghc": "Price GHC (log scale)", "cpi_category": "Category"},
    )
    fig_box.update_layout(height=420)
    st.plotly_chart(fig_box, use_container_width=True)

    with st.expander("Category statistics"):
        st.dataframe(
            stats.rename(columns={
                "cpi_category": "Category", "count": "# products",
                "median": "Median GHC", "mean": "Mean GHC",
                "basket_weight": "Basket weight (%)"
            }).style.format({"Median GHC": "{:,.0f}", "Mean GHC": "{:,.0f}"}),
            use_container_width=True,
        )


# ────────────────────────────────────────────────────────────
# TAB 2 — Forecast
# ────────────────────────────────────────────────────────────
with tab2:
    st.title("🔮 WFP Food Index — Model Comparison & Forecast")
    st.markdown(
        "_Monthly WFP food commodity price index for Ghana (2019–2023). "
        "Nowcast quality assessed by RMSE and directional accuracy on a 13-month hold-out._"
    )

    df_p = load_panel()
    df_comp = load_comparison()

    # ── model comparison table ──
    st.subheader("Model Comparison (Test Set: 2022-07 → 2023-07)")

    comp_display = df_comp.copy()
    comp_display.columns = ["Model", "RMSE", "MAE", "MAPE (%)", "Dir Acc"]
    comp_display["Dir Acc"] = comp_display["Dir Acc"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—"
    )
    comp_display["MAPE (%)"] = comp_display["MAPE (%)"].map(
        lambda x: f"{x:.1f}" if pd.notna(x) else "—"
    )
    comp_display = comp_display.fillna("—")

    st.dataframe(comp_display, use_container_width=True, hide_index=True)

    st.caption(
        "**Best model:** ARIMAX(1,1,0) — 83.3% directional accuracy. "
        "Classical time-series outperforms ML on small monthly samples (n=29 train), "
        "consistent with published literature."
    )

    # ── RMSE bar chart ──
    fig_rmse = px.bar(
        df_comp[df_comp["test_rmse"] < 200],
        x="model", y="test_rmse",
        color="model",
        title="Test RMSE by Model (lower = better)",
        labels={"test_rmse": "RMSE", "model": "Model"},
    )
    fig_rmse.update_layout(showlegend=False, height=360)
    st.plotly_chart(fig_rmse, use_container_width=True)

    # ── historical time series ──
    st.subheader("Historical WFP Food Index (Ghana)")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=df_p["ds"], y=df_p["wfp_food_index"],
        mode="lines+markers", name="WFP Food Index",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=5),
    ))
    fig_hist.add_vrect(
        x0="2022-07-01", x1=df_p["ds"].max().isoformat(),
        fillcolor="orange", opacity=0.08,
        annotation_text="Test period", annotation_position="top left",
    )
    fig_hist.update_layout(
        title="WFP Ghana Food Commodity Price Index (monthly)",
        xaxis_title="Month",
        yaxis_title="WFP Food Index",
        height=400,
        hovermode="x unified",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── GHS/USD and Brent exogenous ──
    with st.expander("Exogenous regressors (GHS/USD, Brent crude)"):
        col1, col2 = st.columns(2)
        with col1:
            fig_fx = px.line(df_p, x="ds", y="ghsusd", title="GHS/USD Exchange Rate",
                             labels={"ghsusd": "GHS per USD", "ds": "Month"})
            st.plotly_chart(fig_fx, use_container_width=True)
        with col2:
            fig_brent = px.line(df_p, x="ds", y="brent", title="Brent Crude (USD/barrel)",
                                labels={"brent": "USD/barrel", "ds": "Month"})
            st.plotly_chart(fig_brent, use_container_width=True)


# ────────────────────────────────────────────────────────────
# TAB 3 — Anomalies
# ────────────────────────────────────────────────────────────
with tab3:
    st.title("⚠️ Price Anomaly Report")
    st.markdown(
        "_Anomalies detected using Isolation Forest (contamination=5%) "
        "and Z-score per CPI category (threshold |z|>2.5)._"
    )

    df_anom = load_anomaly()

    n_total = len(df_anom)
    n_flagged = df_anom["is_flagged"].sum()
    n_z = df_anom["z_flagged"].sum()
    n_iso = df_anom["iso_flagged"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total products", f"{n_total:,}")
    col2.metric("Flagged (either)", f"{n_flagged} ({n_flagged/n_total:.1%})")
    col3.metric("Z-score flags", str(n_z))
    col4.metric("Isolation Forest flags", str(n_iso))

    st.markdown("---")

    # ── filter controls ──
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        show_only_flagged = st.checkbox("Show only flagged products", value=True)
    with col_f2:
        cats = sorted(df_anom["cpi_category"].dropna().unique())
        sel_cats = st.multiselect("Filter by category", cats, default=cats)
    with col_f3:
        sources = sorted(df_anom["source"].dropna().unique())
        sel_srcs = st.multiselect("Filter by source", sources, default=sources)

    view = df_anom.copy()
    if show_only_flagged:
        view = view[view["is_flagged"]]
    if sel_cats:
        view = view[view["cpi_category"].isin(sel_cats)]
    if sel_srcs:
        view = view[view["source"].isin(sel_srcs)]

    view_display = view[[
        "product_name", "source", "cpi_category", "price_ghc",
        "z_score", "anomaly_score", "z_flagged", "iso_flagged",
    ]].sort_values("anomaly_score", ascending=False)

    st.subheader(f"Showing {len(view_display)} products")
    st.dataframe(
        view_display.style.format({
            "price_ghc": "{:,.0f}",
            "z_score": "{:.2f}",
            "anomaly_score": "{:.3f}",
        }).background_gradient(subset=["z_score", "anomaly_score"], cmap="Reds"),
        use_container_width=True,
        height=420,
    )

    # ── z-score scatter ──
    st.subheader("Z-score Distribution by Category")
    fig_z = px.scatter(
        df_anom, x="price_ghc", y="z_score",
        color="is_flagged",
        facet_col="cpi_category",
        facet_col_wrap=3,
        log_x=True,
        color_discrete_map={True: "#d62728", False: "#1f77b4"},
        hover_data=["product_name", "source"],
        title="Price vs Z-score (red = flagged)",
        labels={"price_ghc": "Price GHC (log)", "z_score": "Z-score", "is_flagged": "Flagged"},
    )
    fig_z.add_hline(y=2.5, line_dash="dash", line_color="red", opacity=0.5)
    fig_z.add_hline(y=-2.5, line_dash="dash", line_color="red", opacity=0.5)
    fig_z.update_layout(height=500)
    st.plotly_chart(fig_z, use_container_width=True)

    # ── top anomalies ──
    st.subheader("Top 20 Most Anomalous Products")
    top = df_anom.nlargest(20, "anomaly_score")[[
        "product_name", "source", "cpi_category", "price_ghc", "z_score", "anomaly_score",
    ]]
    fig_top = px.bar(
        top, x="anomaly_score", y="product_name",
        orientation="h",
        color="cpi_category",
        title="Top 20 by Isolation Forest anomaly score",
        labels={"anomaly_score": "Anomaly score", "product_name": "Product"},
    )
    fig_top.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_top, use_container_width=True)

    with st.expander("Download full anomaly report"):
        csv_bytes = df_anom.to_csv(index=False).encode()
        st.download_button(
            label="📥 Download anomaly_report.csv",
            data=csv_bytes,
            file_name="impin_anomaly_report.csv",
            mime="text/csv",
        )
