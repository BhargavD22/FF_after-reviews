import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------
# Inject Custom CSS for a modern look
# ----------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
        color: #333333; /* Darker text */
    }
    .stButton>button {
        background-color: #4CAF50; /* Green button */
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #d3d3d3; /* Gray tab background */
        color: black;
        border-radius: 10px 10px 0 0;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff; /* White active tab */
        color: #000000;
        border-bottom: 3px solid #4CAF50; /* Green underline */
    }
    .stMetric > div[data-testid="stMetricLabel"] {
        font-weight: bold;
        color: #4CAF50;
    }
    h1 {
        color: #1a1a1a;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h2, h3 {
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stExpander {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background-color: #ffffff;
        padding: 20px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------
# Page Config
# ----------------------------------------
st.set_page_config(
    page_title="Financial Forecasting Dashboard",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Financial Forecasting Dashboard")
st.subheader("Revenue Projections & Analytics")

# ----------------------------------------
# Sidebar Controls
# ----------------------------------------
st.sidebar.header("⚙️ Configuration")
forecast_periods = st.sidebar.slider(
    "Forecast Horizon (Months):", 12, 24, 36
)
confidence_interval = st.sidebar.slider(
    "Confidence Interval", 0.80, 0.99, 0.95
)
weekly_seasonality = st.sidebar.checkbox("Include Weekly Seasonality", True)
yearly_seasonality = st.sidebar.checkbox("Include Yearly Seasonality", True)

# Placeholder dataset
@st.cache_data
def load_data():
    dates = pd.date_range(start="2021-01-01", end="2024-12-31", freq="D")
    revenue = np.random.randint(1000, 5000, len(dates)).astype(float)
    df = pd.DataFrame({"ds": dates, "y": revenue})
    return df

df = load_data()

# ----------------------------------------
# Prophet Forecasting
# ----------------------------------------
m = Prophet(
    interval_width=confidence_interval,
    weekly_seasonality=weekly_seasonality,
    yearly_seasonality=yearly_seasonality
)
m.fit(df)
future = m.make_future_dataframe(periods=forecast_periods * 30)
forecast = m.predict(future)

# ----------------------------------------
# Main Dashboard Content (Focus on Forecasting)
# ----------------------------------------
st.header("🔮 Revenue Forecast Outlook")
st.markdown("### Executive Summary")
st.write(
    f"This dashboard provides a **{forecast_periods}-month revenue forecast** based on historical data. "
    "The model projects future revenue trends, highlighting key metrics and seasonal patterns. "
    "Use the sidebar to adjust the forecast horizon and confidence level."
)

col1, col2, col3 = st.columns(3)

# Forecasted KPIs
with col1:
    forecasted_rev = forecast['yhat'][-forecast_periods*30:].sum()
    st.metric("Forecasted Revenue", f"${forecasted_rev:,.0f}")
with col2:
    avg_forecast_daily = forecast['yhat'][-forecast_periods*30:].mean()
    st.metric("Forecasted Daily Average", f"${avg_forecast_daily:,.2f}")
with col3:
    if len(forecast) > 1 and len(df) > 1:
        proj_cagr = ((forecast['yhat'].iloc[-1] / df['y'].iloc[0]) ** (1/(len(forecast)/365))) - 1
    else:
        proj_cagr = 0
    st.metric("Projected CAGR", f"{proj_cagr:.2%}")

# Main Forecast Chart
st.subheader("📅 Daily Revenue Forecast with Confidence Interval")
fig_forecast = plot_plotly(m, forecast)
fig_forecast.update_layout(height=500)
st.plotly_chart(fig_forecast, use_container_width=True)

# ----------------------------------------
# Expander for Historical Data Overview
# ----------------------------------------
with st.expander("📊 **View Historical Data Overview**", expanded=False):
    st.subheader("Historical Data Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        total_rev = df['y'].sum()
        st.metric("Total Historical Revenue", f"${total_rev:,.0f}")
    with col2:
        avg_daily_rev = df['y'].mean()
        st.metric("Average Daily Revenue", f"${avg_daily_rev:,.2f}")
    with col3:
        if len(df) > 1:
            hist_cagr = ((df['y'].iloc[-1] / df['y'].iloc[0]) ** (1/(len(df)/365))) - 1
        else:
            hist_cagr = 0
        st.metric("Historical CAGR", f"{hist_cagr:.2%}")

    st.markdown("---")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("#### Month-over-Month Growth (%)")
        df_monthly = df.copy()
        df_monthly['month'] = df_monthly['ds'].dt.to_period('M')
        monthly_sum = df_monthly.groupby('month')['y'].sum().reset_index()
        monthly_sum['month'] = monthly_sum['month'].dt.to_timestamp()
        monthly_sum['pct_change'] = monthly_sum['y'].pct_change() * 100
        fig_growth = px.line(
            monthly_sum,
            x='month',
            y='pct_change',
            markers=True
        )
        st.plotly_chart(fig_growth, use_container_width=True)
    with chart_col2:
        st.markdown("#### 30-Day Moving Average")
        df['30d_ma'] = df['y'].rolling(window=30).mean()
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Daily Revenue', line=dict(color='#888888')))
        fig_ma.add_trace(go.Scatter(x=df['ds'], y=df['30d_ma'], mode='lines',
                                    name='30-Day Moving Average', line=dict(color='#4CAF50', width=3)))
        st.plotly_chart(fig_ma, use_container_width=True)
    
# ----------------------------------------
# Expander for Model Evaluation
# ----------------------------------------
with st.expander("🔬 **View Model Evaluation & Components**", expanded=False):
    st.subheader("Model Evaluation & Components")
    
    df_eval = forecast.set_index('ds').join(df.set_index('ds'))
    df_eval = df_eval.dropna()

    col1, col2, col3 = st.columns(3)
    with col1:
        mae = np.mean(np.abs(df_eval['y'] - df_eval['yhat']))
        st.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}")
    with col2:
        rmse = np.sqrt(np.mean((df_eval['y'] - df_eval['yhat'])**2))
        st.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.2f}")
    with col3:
        wape = np.sum(np.abs(df_eval['y'] - df_eval['yhat'])) / np.sum(df_eval['y'])
        st.metric("WAPE", f"{wape:.2%}")

    st.markdown("---")

    st.subheader("Time Series Components")
    fig_components = plot_components_plotly(m, forecast)
    st.plotly_chart(fig_components, use_container_width=True)
