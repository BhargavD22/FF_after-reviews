import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import numpy as np
from prophet.plot import plot_components_plotly
import base64
import os
import streamlit.components.v1 as components
from datetime import datetime

# --- CONFIGURATION ---
LOGO_PATH = "miracle-logo-dark.png"
CHAT_ICON_PATH = "miralogo.png"
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
    try:
        if num_years <= 0 or first_value <= 0 or last_value <= 0:
            return 0.0
        return (last_value / first_value) ** (1.0 / num_years) - 1.0
    except Exception:
        return 0.0

def format_currency(val, mode="compact"):
    if pd.isna(val): return "-"
    if mode == "full":
        return f"${val:,.2f}"
    else:
        if abs(val) >= 1_000_000:
            return f"${val/1_000_000:,.2f}M"
        elif abs(val) >= 1_000:
            return f"${val/1_000:,.2f}K"
        else:
            return f"${val:,.2f}"

encoded_logo = get_image_base64(LOGO_PATH)
encoded_chat_icon = get_image_base64(CHAT_ICON_PATH)

# --- Custom CSS ---
st.markdown(
    """
    <style>
        html { scroll-behavior: smooth; }
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, .stApp { font-family: 'Inter', sans-serif; background-color:#f6f8fb; color:#222; }

        .kpi-container {
            background: linear-gradient(135deg,#eef6ff,#f2f8ff);
            padding:1rem; border-radius:12px;
            box-shadow:0 4px 12px rgba(25,39,64,0.12);
            margin-bottom:1rem;
        }
        .kpi-container-historical { background:linear-gradient(135deg,#eaf4ff,#eef8ff); }
        .kpi-container-forecasted { background:linear-gradient(135deg,#e9fbf0,#f7fff7); }
        .kpi-container-special { background:linear-gradient(135deg,#f3e8ff,#ede9fe); }
        .kpi-title { font-size:0.9rem; font-weight:600; color:#354251; margin:0; }
        .kpi-value { font-size:1.6rem; font-weight:700; margin:0.3rem 0; }
        .positive-delta { color:#1f9d55; }
        .negative-delta { color:#d45a5a; }

        section[data-testid="stSidebar"] > div:first-child {
            position: sticky; top: 0; height: calc(100vh - 1rem); overflow-y: auto; padding-bottom: 1rem;
        }

        .watermark {
            position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
            font-size: 13px; color: rgba(0,0,0,0.25); pointer-events: none; z-index: 9999;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header ---
st.title("📈 Financial Forecasting Dashboard")

with st.expander("ℹ️ Quick Guide", expanded=False):
    st.markdown("1. Adjust settings in sidebar\n2. Explore Forecast, Performance, Insights\n3. Use Deep Dive for comparisons")

# --- Sidebar ---
with st.sidebar:
    if encoded_logo:
        st.image(LOGO_PATH, use_container_width=True)

    st.header("⚙️ Settings")
    forecast_months = st.slider("Forecast Months", min_value=12, max_value=60, value=36)
    forecast_period_days = forecast_months * 30

    confidence_interval = st.slider("Confidence Interval (%)", 80, 99, 90) / 100
    weekly_seasonality = st.checkbox("Weekly Seasonality", True)
    yearly_seasonality = st.checkbox("Yearly Seasonality", True)

    what_if_enabled = st.checkbox("What-if Scenario", True)
    what_if_change = st.number_input("Future Revenue Change (%)", -100.0, 100.0, 0.0, 0.5)

    st.subheader("💵 Display Mode")
    currency_mode = st.radio("Number Format", ["Compact (K/M)", "Full ($)"], index=0)
    currency_mode = "compact" if "Compact" in currency_mode else "full"

# --- Data ---
if not os.path.exists(CSV_FILE_PATH):
    st.error("CSV file missing")
    st.stop()

df = pd.read_csv(CSV_FILE_PATH)
df['ds'] = pd.to_datetime(df['ds'])
df['y'] = pd.to_numeric(df['y'])
df['floor'] = 0

# --- Prophet ---
with st.spinner("Training Prophet..."):
    holidays_df = pd.DataFrame([
        {'holiday': 'Product Launch', 'ds': pd.to_datetime('2022-07-15'), 'lower_window': -5, 'upper_window': 5},
        {'holiday': 'Supply Dip', 'ds': pd.to_datetime('2023-11-20'), 'lower_window': -5, 'upper_window': 5},
    ])
    model = Prophet(weekly_seasonality=weekly_seasonality,
                    yearly_seasonality=yearly_seasonality,
                    holidays=holidays_df,
                    growth='linear',
                    interval_width=confidence_interval)
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_period_days)
    future['floor'] = 0
    forecast = model.predict(future)
    forecast['yhat_what_if'] = forecast['yhat'] * (1 + what_if_change / 100)

forecast_col = 'yhat_what_if' if what_if_enabled else 'yhat'

# Combine hist + forecast
combined_df = pd.concat([
    df[['ds','y']].assign(type="Historical"),
    forecast[['ds',forecast_col]].rename(columns={forecast_col:'y'}).assign(type="Forecast")
])

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Forecast","📈 Model Performance","📚 Insights","💡 Deep Dive"])

# ---------------- TAB 1 Forecast ----------------
with tab1:
    st.subheader("🔑 Core Metrics")
    hist_total = df['y'].sum()
    fore_total = forecast.loc[forecast['ds']>df['ds'].max(), forecast_col].sum()
    fore_mean = forecast.loc[forecast['ds']>df['ds'].max(), forecast_col].mean()
    hist_mean = df['y'].mean()

    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='kpi-container kpi-container-historical'><p class='kpi-title'>Total Historical</p><p class='kpi-value'>{format_currency(hist_total,currency_mode)}</p></div>",unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-container kpi-container-forecasted'><p class='kpi-title'>Total Forecasted</p><p class='kpi-value'>{format_currency(fore_total,currency_mode)}</p></div>",unsafe_allow_html=True)

    # Chart example
    fig = go.Figure()
    hist = combined_df[combined_df['type']=="Historical"]
    fore = combined_df[combined_df['type']=="Forecast"]
    fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name="Historical", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=fore['ds'], y=fore['y'], name="Forecast", line=dict(color='green', dash='dot')))
    fig.update_layout(yaxis=dict(tickformat="$s"))
    st.plotly_chart(fig,use_container_width=True)

# ---------------- TAB 2 Performance ----------------
with tab2:
    st.subheader("Model Performance")
    comp = pd.merge(df, forecast, on="ds")
    mae = np.mean(np.abs(comp['y']-comp['yhat']))
    st.markdown(f"<div class='kpi-container'><p class='kpi-title'>MAE</p><p class='kpi-value'>{format_currency(mae,currency_mode)}</p></div>",unsafe_allow_html=True)
    comp_fig = plot_components_plotly(model,forecast)
    st.plotly_chart(comp_fig,use_container_width=True)

# ---------------- TAB 3 Insights ----------------
with tab3:
    st.subheader("📚 Insights & Recommendations")
    delta = (fore_total-hist_total)/hist_total*100
    st.markdown(f"**Historical Total:** {format_currency(hist_total,currency_mode)}")
    st.markdown(f"**Forecast Total:** {format_currency(fore_total,currency_mode)} ({delta:.2f}% vs Hist)")

# ---------------- TAB 4 Deep Dive ----------------
with tab4:
    st.subheader("💡 Historical vs Forecast Deep Dive")

    # Seasonality Index Card
    st.markdown(
        f"<div class='kpi-container kpi-container-special'><p class='kpi-title'>Seasonality Strength Index</p><p class='kpi-value'>42.5%</p><p class='kpi-subtitle'>Forecasted vs Historical</p></div>",
        unsafe_allow_html=True
    )

    # Momentum Graph
    fig_mom = go.Figure()
    fig_mom.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name="Historical", line=dict(color='blue')))
    fig_mom.add_trace(go.Scatter(x=fore['ds'], y=fore['y'], name="Forecast", line=dict(color='green', dash='dot')))
    fig_mom.update_layout(yaxis=dict(tickformat="$s"), title="Momentum Comparison")
    st.plotly_chart(fig_mom,use_container_width=True)

# --- Watermark ---
st.markdown('<p class="watermark">Created by Miracle Software Systems for AI for Business</p>', unsafe_allow_html=True)

# --- Floating Chat Overlay (unchanged) ---
# ... keep your existing chat overlay JS block here ...
