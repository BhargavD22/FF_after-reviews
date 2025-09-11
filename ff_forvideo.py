# -*- coding: utf-8 -*-
"""
Enhanced Streamlit Financial Forecast app
Replace your existing ff_forvideo.py with this file.
Dependencies:
 - streamlit
 - pandas
 - prophet
 - plotly
 - numpy
"""

import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import numpy as np
from prophet.plot import plot_components_plotly
import base64
import os

# --- CONFIGURATION ---
LOGO_PATH = "miracle-logo-dark.png"   # favicon + sidebar logo
CHAT_ICON_PATH = "miralogo.png"       # optional for floating chat icon (keeps previous overlay code if you need it)
CSV_FILE_PATH = "financial_forecast_modified.csv"

# Set Streamlit page config for wide layout + favicon
st.set_page_config(
    layout="wide",
    page_title="Financial Forecasting",
    page_icon=LOGO_PATH,
    initial_sidebar_state="expanded",
)

# --- helper functions ---
def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return ""

def safe_cagr(first_value, last_value, num_years):
    """Compute CAGR safely, returns 0 if invalid inputs."""
    try:
        if num_years <= 0 or first_value <= 0 or last_value <= 0:
            return 0.0
        return (last_value / first_value) ** (1.0 / num_years) - 1.0
    except Exception:
        return 0.0

# Encode images for display (used in sidebar)
encoded_logo = get_image_base64(LOGO_PATH)
encoded_chat_icon = get_image_base64(CHAT_ICON_PATH)

# --- CSS / styling ---
st.markdown(
    f"""
    <style>
        html {{ scroll-behavior: smooth; }}
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, .stApp {{ font-family: 'Inter', sans-serif; }}
        .stApp {{ background-color: #f6f8fb; color: #222; }}

        /* Section headings (icons already in titles) */
        h2, h3 {{ display:flex; align-items:center; gap:0.5rem; }}

        /* KPI card styles (brand-aligned gradients) */
        .kpi-container {{
            display:flex; flex-direction:column; justify-content:space-between;
            background: linear-gradient(135deg,#eef6ff,#f2f8ff);
            padding:1.25rem; border-radius:12px;
            box-shadow:0 6px 18px rgba(25,39,64,0.08);
            transition: transform 0.25s ease;
            height:100%;
        }}
        .kpi-container-historical {{ background: linear-gradient(135deg,#eaf4ff,#eef8ff); }}
        .kpi-container-forecasted {{ background: linear-gradient(135deg,#e9fbf0,#f7fff7); }}
        .kpi-container:hover {{ transform: translateY(-6px); }}

        .kpi-title {{ font-size:0.95rem; color:#354251; font-weight:600; margin:0; }}
        .kpi-value {{ font-size:1.8rem; font-weight:700; color:#0b6ef6; margin:0.2rem 0; }}
        .kpi-sub {{ font-size:0.85rem; color:#556370; margin:0; }}
        .kpi-delta {{ font-weight:600; margin-top:0.5rem; }}

        .positive-delta {{ color:#1f9d55; }}
        .negative-delta {{ color:#d45a5a; }}

        /* Sticky sidebar */
        section[data-testid="stSidebar"] > div:first-child {{
            position: sticky;
            top: 0;
            height: calc(100vh - 1rem);
            overflow-y: auto;
            padding-bottom: 1rem;
        }}

        /* Centered watermark */
        .watermark {{
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 13px;
            color: rgba(0,0,0,0.08);
            pointer-events: none;
            z-index: 9999;
        }}

        /* Dataframe radius */
        .stDataFrame table tbody tr th, .stDataFrame table tbody tr td {{
            border-radius: 6px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- App header & quick guide ---
st.title("📈 Financial Forecasting Dashboard")
st.markdown(
    "A **dynamic** dashboard to analyze historical revenue and generate forecasts using **Prophet**. "
    "Use the sidebar to control the forecast horizon, seasonality and what-if scenarios."
)

with st.expander("ℹ️ Quick Guide", expanded=False):
    st.markdown(
        """
        1. Upload/ensure your CSV (`ds` and `y` columns) is present in the repo as `financial_forecast_modified.csv`.
        2. Set forecast months and the what-if scenario in the sidebar.
        3. Explore Forecast, Model Performance and Insights tabs. Download results from Forecast tab.
        """
    )

# --- Sidebar controls ---
with st.sidebar:
    if encoded_logo:
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.write("**Logo not found** — expected at: " + LOGO_PATH)

    st.markdown("---")
    st.header("⚙️ Settings")
    forecast_months = st.slider("Forecast horizon (months):", min_value=1, max_value=60, value=36)
    forecast_period_days = forecast_months * 30
    confidence_interval = st.slider("Confidence Interval (%)", min_value=80, max_value=99, value=90) / 100.0
    st.markdown("**Seasonality**")
    weekly_seasonality = st.checkbox("Weekly seasonality", value=True)
    yearly_seasonality = st.checkbox("Yearly seasonality", value=True)
    st.markdown("---")
    st.header("🔮 What-if Scenario")
    what_if_enabled = st.checkbox("Enable what-if adjustment", value=True)
    what_if_change = st.number_input("Future revenue change (%)", value=0.0, step=0.5, min_value=-100.0, max_value=100.0)
    st.markdown("---")
    st.header("🔖 Bookmarks")
    st.markdown(
        """
        - [Core KPIs](#core-kpis)  
        - [Growth Metrics](#growth-metrics)  
        - [Historical Trends](#historical-trends)  
        - [Daily Revenue](#daily-revenue)  
        - [Forecast Table](#forecast-table)  
        - [Model Performance](#model-performance)
        """,
        unsafe_allow_html=True,
    )

# --- Data ingestion ---
if not os.path.exists(CSV_FILE_PATH):
    st.error(f"The required data file '{CSV_FILE_PATH}' was not found. Please add it to the app directory.")
    st.stop()

try:
    df = pd.read_csv(CSV_FILE_PATH)
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'])
except Exception as e:
    st.error(f"Unable to read csv file. Make sure it has 'ds' and 'y' columns. Error: {e}")
    st.stop()

# tabs layout: Forecast, Model Performance, Insights
tab_forecast, tab_perf, tab_insights = st.tabs(["📊 Forecast", "📈 Model Performance", "📚 Insights & Recommendations"])

# Pre-calc: holidays or anomalies for Prophet (example placeholders)
holidays_df = pd.DataFrame([
    {'holiday': 'Product Launch Spike', 'ds': pd.to_datetime('2022-07-15'), 'lower_window': -5, 'upper_window': 5},
    {'holiday': 'Supply Chain Dip', 'ds': pd.to_datetime('2023-11-20'), 'lower_window': -3, 'upper_window': 3},
])

# Fit model and produce forecast inside spinner so user sees loading indicator
with st.spinner("Training Prophet model and generating forecast..."):
    # add floor to prevent negative predictions
    df['floor'] = 0
    model = Prophet(weekly_seasonality=weekly_seasonality,
                    yearly_seasonality=yearly_seasonality,
                    holidays=holidays_df,
                    growth='linear',
                    interval_width=confidence_interval)
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_period_days)
    future['floor'] = 0
    forecast = model.predict(future)
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    # add what-if column
    forecast['yhat_what_if'] = forecast['yhat'] * (1.0 + float(what_if_change) / 100.0)
    # determine which column to use
    forecast_col = 'yhat_what_if' if (what_if_enabled and 'yhat_what_if' in forecast.columns) else 'yhat'

# ---------- FORECAST TAB ----------
with tab_forecast:
    st.header("📊 Forecast")

    # aggregation toggle
    agg_choice = st.radio("View revenue by:", options=["Day", "Month", "Quarter"], horizontal=True)

    # prepare plotting data depending on aggregation
    if agg_choice == "Month":
        df_plot = df.copy()
        df_plot['ds'] = df_plot['ds'].dt.to_period('M').dt.to_timestamp()
        forecast_plot = forecast.copy()
        forecast_plot['ds'] = forecast_plot['ds'].dt.to_period('M').dt.to_timestamp()
    elif agg_choice == "Quarter":
        df_plot = df.copy()
        df_plot['ds'] = df_plot['ds'].dt.to_period('Q').dt.to_timestamp()
        forecast_plot = forecast.copy()
        forecast_plot['ds'] = forecast_plot['ds'].dt.to_period('Q').dt.to_timestamp()
    else:
        df_plot = df.copy()
        forecast_plot = forecast.copy()

    # Build combined data (Historical + Forecast column) for certain UI elements
    historical_end_date = df['ds'].max()
    forecast_df_future = forecast[forecast['ds'] > historical_end_date].copy()

    # KPI calculations for showing on top
    total_hist = df['y'].sum()
    avg_hist = df['y'].mean()
    total_fore = forecast_df_future[forecast_col].sum() if not forecast_df_future.empty else 0.0
    avg_fore = forecast_df_future[forecast_col].mean() if not forecast_df_future.empty else 0.0
    delta_total_pct = ((total_fore - total_hist) / total_hist * 100.0) if total_hist != 0 else 0.0

    # CAGR historical
    first_rev = df.sort_values('ds').iloc[0]['y']
    last_rev = df.sort_values('ds').iloc[-1]['y']
    years_hist = (df['ds'].max() - df['ds'].min()).days / 365.25
    cagr_hist = safe_cagr(first_rev, last_rev, years_hist)

    # Forecast CAGR (from end of historical to end of forecast)
    if not forecast_df_future.empty:
        last_forecast_rev = forecast_df_future[forecast_col].iloc[-1]
        years_fore = (forecast_df_future['ds'].max() - df['ds'].max()).days / 365.25
        cagr_fore = safe_cagr(last_rev, last_forecast_rev, years_fore)
    else:
        cagr_fore = 0.0

    # KPI cards row
    st.markdown('<div id="core-kpis"></div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns([1.2, 1.2, 1.2, 1.2])
    with k1:
        st.markdown(
            f"""
            <div class="kpi-container kpi-container-historical">
                <p class="kpi-title">Total Historical Revenue</p>
                <p class="kpi-value">${total_hist:,.0f}</p>
                <p class="kpi-sub">Sum of past revenue</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-container kpi-container-forecasted">
                <p class="kpi-title">Total Forecasted Revenue ({forecast_months} mo)</p>
                <p class="kpi-value">${total_fore:,.0f}</p>
                <p class="kpi-sub">Projected sum for forecast window</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-container">
                <p class="kpi-title">Historical CAGR</p>
                <p class="kpi-value">{cagr_hist:.2%}</p>
                <p class="kpi-sub">Avg annual growth (historical)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-container">
                <p class="kpi-title">Forecasted CAGR</p>
                <p class="kpi-value">{cagr_fore:.2%}</p>
                <p class="kpi-sub">Avg annual growth (forecast)</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Historical chart (30-day moving average) + forecast overlay
    st.subheader("Historical & Forecasted Revenue")
    df_plot_sorted = df_plot.sort_values('ds')
    forecast_plot_sorted = forecast_plot.sort_values('ds')

    # moving average (applies only for daily; for month/quarter it's still meaningful)
    df_plot_sorted['moving_avg_30'] = df_plot_sorted['y'].rolling(window=30, min_periods=1).mean()
    # create figure
    fig = go.Figure()

    # Historical area + line
    fig.add_trace(go.Scatter(
        x=df_plot_sorted['ds'], y=df_plot_sorted['y'],
        mode='lines',
        name='Historical Revenue',
        line=dict(color='#0b6ef6', width=2),
        fill='tozeroy',
        fillcolor='rgba(11,110,246,0.06)',
        hovertemplate='Date: %{x}<br>Revenue: %{y:$,.2f}<extra></extra>'
    ))

    # historical moving average
    fig.add_trace(go.Scatter(
        x=df_plot_sorted['ds'], y=df_plot_sorted['moving_avg_30'],
        mode='lines',
        name='Moving Avg (30)',
        line=dict(color='#1f9d55', width=3),
        hovertemplate='Date: %{x}<br>Moving Avg: %{y:$,.2f}<extra></extra>'
    ))

    # Forecast line (dashed)
    fig.add_trace(go.Scatter(
        x=forecast_plot_sorted['ds'], y=forecast_plot_sorted[forecast_col],
        mode='lines',
        name='Forecasted Revenue',
        line=dict(color='#16a34a', width=2, dash='dot'),
        fill='tozeroy',
        fillcolor='rgba(22,163,74,0.04)',
        hovertemplate='Date: %{x}<br>Forecast: %{y:$,.2f}<extra></extra>'
    ))

    # Confidence interval shading for the forecast portion (use original yhat bounds)
    # Use only for the forecast region beyond historical_end_date if available
    fc_bounds = forecast[(forecast['ds'] > historical_end_date) & (forecast['yhat_upper'].notna())]
    if not fc_bounds.empty:
        fig.add_trace(go.Scatter(
            x=list(fc_bounds['ds']) + list(fc_bounds['ds'])[::-1],
            y=list(fc_bounds['yhat_upper']) + list(fc_bounds['yhat_lower'])[::-1],
            fill='toself',
            fillcolor='rgba(22,163,74,0.08)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name=f"{int(confidence_interval*100)}% Confidence Interval"
        ))

    # mark the forecast start
    fig.add_vline(x=historical_end_date, line_width=1, line_dash="dash", line_color="#ef4444")
    fig.add_annotation(
        x=historical_end_date,
        y=1.02,
        xref="x",
        yref="paper",
        text="Forecast begins",
        showarrow=False,
        font=dict(color="#ef4444")
    )

    fig.update_layout(
        title="Revenue: Historical vs Forecast",
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        hovermode="x unified",
        template="plotly_white",
        transition_duration=600
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # Forecast table + download
    st.subheader(f"🧾 {forecast_months}-Month Forecast Table")
    forecast_table = forecast[['ds', forecast_col, 'yhat_lower', 'yhat_upper']].rename(
        columns={'ds': 'Date', forecast_col: 'Predicted Revenue', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}
    )
    # show only forecast horizon rows
    table_to_show = forecast_table[forecast_table['Date'] > historical_end_date].head(forecast_period_days)
    st.dataframe(table_to_show.style.format({"Predicted Revenue": "${:,.2f}", "Lower Bound": "${:,.2f}", "Upper Bound": "${:,.2f}"}), height=300)

    csv = table_to_show.to_csv(index=False)
    st.download_button(f"⬇️ Download {forecast_months}-Month Forecast CSV", csv, f"forecast_{forecast_months}_months.csv", "text/csv")

# ---------- MODEL PERFORMANCE TAB ----------
with tab_perf:
    st.header("📈 Model Performance")

    # merge historical and in-sample predictions to compute error metrics
    merged_for_perf = pd.merge(df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='inner')
    if not merged_for_perf.empty:
        mae = np.mean(np.abs(merged_for_perf['y'] - merged_for_perf['yhat']))
        rmse = np.sqrt(np.mean((merged_for_perf['y'] - merged_for_perf['yhat']) ** 2))
        wape = (np.sum(np.abs(merged_for_perf['y'] - merged_for_perf['yhat'])) / np.sum(np.abs(merged_for_perf['y']))) * 100 if merged_for_perf['y'].abs().sum() != 0 else 0.0
        bias = np.mean(merged_for_perf['yhat'] - merged_for_perf['y'])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE (Mean Absolute Error)", f"${mae:,.2f}")
        c2.metric("RMSE", f"${rmse:,.2f}")
        c3.metric("WAPE", f"{wape:.2f}%")
        c4.metric("Forecast Bias", f"${bias:,.2f}")
    else:
        st.info("Not enough overlapping data to compute model performance metrics.")

    st.markdown("---")
    st.subheader("📉 Time Series Components")
    try:
        comp_fig = plot_components_plotly(model, forecast)
        # attempt to label axis
        for ax in comp_fig.select_yaxes():
            pass
        st.plotly_chart(comp_fig, use_container_width=True)
    except Exception as e:
        st.info("Could not render components plot: " + str(e))

# ---------- INSIGHTS & RECOMMENDATIONS TAB ----------
with tab_insights:
    st.header("📚 Insights & Recommendations")
    # derive insights again from the previously computed values
    # ensure forecast_df_future and totals exist
    total_historical = total_hist
    total_forecast = total_fore
    delta_pct = delta_total_pct
    hist_cagr = cagr_hist
    forecast_cagr = cagr_fore

    # Simple trend classification
    trend_text = "positive" if delta_pct > 0 else "negative" if delta_pct < 0 else "flat"
    trend_icon = "📈" if delta_pct > 0 else "📉" if delta_pct < 0 else "➡️"

    st.markdown(f"### {trend_icon} Summary")
    st.markdown(
        f"- **Total Historical Revenue:** ${total_historical:,.0f}\n"
        f"- **Total Forecasted Revenue ({forecast_months} mo):** ${total_forecast:,.0f}\n"
        f"- **Change vs Historical:** {delta_pct:.2f}%\n"
        f"- **Historical CAGR:** {hist_cagr:.2%}\n"
        f"- **Forecasted CAGR:** {forecast_cagr:.2%}\n"
    )

    st.markdown("### Recommendations")
    if delta_pct > 5:
        st.markdown(
            "- Forecast indicates meaningful positive growth — consider increasing investment in growth initiatives, hiring for capacity, and preparing inventory/operations for higher demand."
        )
    elif delta_pct < -5:
        st.markdown(
            "- Forecast suggests a contraction. Investigate drivers (seasonality, recent dips) and consider cost optimization, marketing to stabilize revenue, or hedging cashflow exposure."
        )
    else:
        st.markdown(
            "- Forecast is relatively stable. Maintain current strategy but monitor leading indicators and re-run forecasts frequently if new data arrives."
        )

    st.markdown("---")
    st.markdown("### Actionable next steps")
    st.markdown(
        """
        - Align budgets and headcount planning with forecast peaks.
        - Investigate anomalies flagged in historical data (e.g., product launches, supply dips).
        - Re-run forecasts monthly and after major events (promotions, product launches).
        """
    )

# --- Centered watermark / branding ---
st.markdown('<p class="watermark">Created by Gemini for Data Analytics</p>', unsafe_allow_html=True)


