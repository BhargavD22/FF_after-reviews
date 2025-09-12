# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import numpy as np
from prophet.plot import plot_components_plotly
import base64
import os
import streamlit.components.v1 as components

# --- CONFIGURATION ---
LOGO_PATH = "miracle-logo-dark.png"

# Page settings
st.set_page_config(
    layout="wide",
    page_title="Financial Forecasting",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---
def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

def format_currency(value, compact=True):
    if compact:
        if abs(value) >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        elif abs(value) >= 1_000:
            return f"${value/1_000:.2f}K"
    return f"${value:,.2f}"

encoded_logo = get_image_base64(LOGO_PATH)

# --- Custom CSS ---
st.markdown(
    """
    <style>
        html, body, .stApp { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #f8f9fb; color: #333; }
        h1, h2, h3, h4 { color: #007bff; }

        /* KPI Card Styles */
        .kpi-container {
            display: flex; flex-direction: column;
            background: #fff;
            padding: 1.25rem; border-radius: 12px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.15);
            margin-bottom: 1rem; transition: transform 0.2s;
        }
        .kpi-container:hover { transform: translateY(-3px); }
        .kpi-title { font-size: 0.9rem; font-weight: 600; color: #555; margin-bottom: 0.5rem; }
        .kpi-value { font-size: 1.8rem; font-weight: 700; margin: 0.2rem 0; }
        .kpi-subtitle { font-size: 0.8rem; color: #777; }
        .kpi-delta { font-size: 0.9rem; font-weight: 600; margin-top: 0.5rem; }
        .positive-delta { color: #28a745; }
        .negative-delta { color: #dc3545; }

        .historical { border-left: 6px solid #007bff; }
        .forecasted { border-left: 6px solid #28a745; }
        .seasonality { border-left: 6px solid purple; background: #f9f1ff; }

        /* Date Input Styling */
        div[data-testid="stDateInput"] input {
            border-radius: 8px; border: 1px solid #ced4da;
            padding: 8px 12px; font-size: 0.95rem;
        }
        div[data-testid="stDateInput"] input:focus {
            border-color: #007bff; box-shadow: 0 0 0 0.2rem rgba(0,123,255,0.25);
        }

        /* Watermark */
        .watermark {
            position: fixed; bottom: 8px; right: 12px;
            font-size: 13px; color: rgba(0,0,0,0.35);
            pointer-events: none; z-index: 9999;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar ---
with st.sidebar:
    if encoded_logo:
        st.image(LOGO_PATH, use_container_width=True)
    st.header("⚙️ Settings")

    st.subheader("Forecast Period")
    forecast_months = st.slider("Select number of months to forecast:", min_value=12, max_value=60, value=36)
    forecast_days = forecast_months * 30

    st.subheader("Model Config")
    conf_int = st.slider("Confidence Interval (%)", 80, 99, 90) / 100
    weekly = st.checkbox("Weekly Seasonality", True)
    yearly = st.checkbox("Yearly Seasonality", True)

    st.subheader("What-if Scenario")
    what_if = st.checkbox("Apply Scenario", True)
    pct_change = st.number_input("Future Revenue Change (%)", -100.0, 100.0, 0.0, 0.5)

    st.subheader("Display")
    compact_numbers = st.checkbox("Compact Numbers (K/M)", True)

# --- Main Content ---
st.title("📈 Financial Forecasting Dashboard")
CSV_FILE = "financial_forecast_modified.csv"

if not os.path.exists(CSV_FILE):
    st.error(f"File '{CSV_FILE}' not found")
    st.stop()

# Load data
df = pd.read_csv(CSV_FILE)
df['ds'] = pd.to_datetime(df['ds'])
df['y'] = pd.to_numeric(df['y'])
df['floor'] = 0

# Prophet holidays
holidays = pd.DataFrame([
    {"holiday":"Product Launch Spike","ds":"2022-07-15","lower_window":-5,"upper_window":5},
    {"holiday":"Supply Chain Dip","ds":"2023-11-20","lower_window":-5,"upper_window":5}
])

# Fit model
m = Prophet(weekly_seasonality=weekly, yearly_seasonality=yearly, holidays=holidays, growth='linear')
m.fit(df)
future = m.make_future_dataframe(periods=forecast_days)
future['floor'] = 0
forecast = m.predict(future)
forecast['ds'] = pd.to_datetime(forecast['ds'])
forecast['yhat_what_if'] = forecast['yhat'] * (1 + pct_change/100)

forecast_col = "yhat_what_if" if what_if else "yhat"

# Combined df
combined = pd.concat([
    df[['ds','y']].assign(type="Historical"),
    forecast[['ds',forecast_col]].rename(columns={forecast_col:'y'}).assign(type="Forecast")
])

# Tabs
tab1, tab2 = st.tabs(["📊 Forecast", "📈 Model Performance"])

with tab1:
    st.subheader("🔑 Core KPIs")

    hist_total = df['y'].sum()
    hist_avg = df['y'].mean()
    fc_df = forecast[forecast['ds'] > df['ds'].max()]
    fc_total = fc_df[forecast_col].sum()
    fc_avg = fc_df[forecast_col].mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="kpi-container historical">
            <p class="kpi-title">Total Historical</p>
            <p class="kpi-value">{format_currency(hist_total, compact_numbers)}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-container forecasted">
            <p class="kpi-title">Total Forecasted</p>
            <p class="kpi-value">{format_currency(fc_total, compact_numbers)}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        seas_strength = round(fc_df[forecast_col].std() / fc_df[forecast_col].mean(),2)
        st.markdown(f"""
        <div class="kpi-container seasonality">
            <p class="kpi-title">Seasonality Index</p>
            <p class="kpi-value">{seas_strength}</p>
            <p class="kpi-subtitle">Std Dev / Mean</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Comparison chart
    st.subheader("📊 Historical vs Forecast Comparison")
    fig = go.Figure()
    hist = combined[combined['type']=="Historical"]
    fc = combined[combined['type']=="Forecast"]

    fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name="Historical", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=fc['ds'], y=fc['y'], name="Forecasted", line=dict(color="green")))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Revenue",
        yaxis=dict(tickprefix="$"),
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📊 Model Performance")
    merged = pd.merge(df, forecast, on="ds", how="inner")
    mae = np.mean(np.abs(merged['y'] - merged['yhat']))
    rmse = np.sqrt(np.mean((merged['y'] - merged['yhat'])**2))
    wape = np.sum(np.abs(merged['y']-merged['yhat'])) / np.sum(np.abs(merged['y']))*100
    bias = np.mean(merged['yhat'] - merged['y'])

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f"<div class='kpi-container'><p class='kpi-title'>MAE</p><p class='kpi-value'>{format_currency(mae, compact_numbers)}</p></div>", unsafe_allow_html=True)
    with col2: st.markdown(f"<div class='kpi-container'><p class='kpi-title'>RMSE</p><p class='kpi-value'>{format_currency(rmse, compact_numbers)}</p></div>", unsafe_allow_html=True)
    with col3: st.markdown(f"<div class='kpi-container'><p class='kpi-title'>WAPE</p><p class='kpi-value'>{wape:.2f}%</p></div>", unsafe_allow_html=True)
    with col4: st.markdown(f"<div class='kpi-container'><p class='kpi-title'>Bias</p><p class='kpi-value'>{format_currency(bias, compact_numbers)}</p></div>", unsafe_allow_html=True)

    st.subheader("📉 Components")
    comp = plot_components_plotly(m, forecast)
    st.plotly_chart(comp, use_container_width=True)

# --- Watermark ---
st.markdown('<p class="watermark">Created by Miracle Software Systems for AI for Business</p>', unsafe_allow_html=True)

# --- Floating Chat Overlay ---
CHAT_ICON_PATH = "miralogo.png"
try:
    with open(CHAT_ICON_PATH,"rb") as f: _CHAT_ICON_B64 = base64.b64encode(f.read()).decode()
except: _CHAT_ICON_B64 = ""

overlay_html = """
<script>
(function () {
  var host = window.parent.document.body;
  var div = document.createElement("div"); div.id="chat"; host.appendChild(div);
  var css = '.btn{position:fixed;bottom:18px;right:18px;width:56px;height:56px;border-radius:50%;background:#fff;border:2px solid #007bff;box-shadow:0 8px 24px rgba(0,0,0,.18);cursor:pointer;}';
  div.innerHTML='<style>'+css+'</style><button class="btn">💬</button>';
})();
</script>
"""
components.html(overlay_html, height=0, scrolling=False)
