import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px
import plotly.graph_objects as go


# ----------------------------------------
# Page Config
# ----------------------------------------
st.set_page_config(
    page_title="Retail & E-Commerce Financial Forecasting",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Retail & E-Commerce Financial Forecasting Dashboard")

# ----------------------------------------
# Sidebar Controls
# ----------------------------------------
st.sidebar.header("⚙️ Configuration")

forecast_periods = st.sidebar.selectbox(
    "Forecast Horizon (Months):", [12, 24, 36], index=0
)

confidence_interval = st.sidebar.slider(
    "Confidence Interval", 0.80, 0.99, 0.95
)

weekly_seasonality = st.sidebar.checkbox("Include Weekly Seasonality", True)
yearly_seasonality = st.sidebar.checkbox("Include Yearly Seasonality", True)

# Placeholder dataset (replace with your actual data loading)
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
# Tabs
# ----------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📂 Data Overview",
    "🔮 Forecasting",
    "📏 Model Evaluation"
])

# ----------------------------------------
# Tab 1: Data Overview
# ----------------------------------------
with tab1:
    st.subheader("📂 Historical Data Overview")

    # Historical KPIs
    total_rev = df['y'].sum()
    avg_daily_rev = df['y'].mean()
    hist_cagr = ((df['y'].iloc[-1] / df['y'].iloc[0]) ** (1/3)) - 1

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Historical Revenue", f"${total_rev:,.0f}")
    col2.metric("Average Daily Revenue", f"${avg_daily_rev:,.2f}")
    col3.metric("Historical CAGR", f"{hist_cagr:.2%}")

    # Growth Metrics
    st.markdown("### 📈 Growth Metrics")
    df_monthly = df.copy()
    df_monthly['month'] = df_monthly['ds'].dt.to_period('M')
    monthly_sum = df_monthly.groupby('month')['y'].sum().reset_index()
    monthly_sum['month'] = monthly_sum['month'].dt.to_timestamp()  # ✅ fix
    monthly_sum['pct_change'] = monthly_sum['y'].pct_change() * 100

    fig_growth = px.line(
        monthly_sum,
        x='month',
        y='pct_change',
        title="Month-over-Month Growth (%)",
        markers=True
    )
    st.plotly_chart(fig_growth, use_container_width=True)

    # 30-day moving average
    st.markdown("### 📊 Historical Revenue with 30-Day Moving Average")
    df['30d_ma'] = df['y'].rolling(window=30).mean()
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Daily Revenue'))
    fig_ma.add_trace(go.Scatter(x=df['ds'], y=df['30d_ma'], mode='lines',
                                name='30-Day Moving Average', line=dict(color='orange')))
    st.plotly_chart(fig_ma, use_container_width=True)

# ----------------------------------------
# Tab 2: Forecasting + Deep Dive
# ----------------------------------------
with tab2:
    st.subheader("🔮 Forecasting Future Revenue")

    # Forecasted KPIs
    forecasted_rev = forecast['yhat'][-forecast_periods*30:].sum()
    avg_forecast_daily = forecast['yhat'][-forecast_periods*30:].mean()
    proj_cagr = ((forecast['yhat'].iloc[-1] / df['y'].iloc[0]) ** (1/3)) - 1

    col1, col2, col3 = st.columns(3)
    col1.metric("Forecasted Revenue", f"${forecasted_rev:,.0f}")
    col2.metric("Forecasted Daily Average", f"${avg_forecast_daily:,.2f}")
    col3.metric("Projected CAGR", f"{proj_cagr:.2%}")

    # Daily Forecast with confidence bands
    st.markdown("### 📅 Daily Revenue Forecast with Confidence Interval")
    fig_forecast = plot_plotly(m, forecast)
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Cumulative Forecast
    st.markdown("### 📊 Cumulative Revenue Forecast")
    forecast['cumulative'] = forecast['yhat'].cumsum()
    fig_cum = px.line(forecast, x='ds', y='cumulative', title="Cumulative Forecasted Revenue")
    st.plotly_chart(fig_cum, use_container_width=True)

    # Forecast Table
    st.markdown("### 📋 Forecast Table")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))
    st.download_button(
        label="Download Forecast Data (CSV)",
        data=forecast.to_csv(index=False),
        file_name="forecast.csv",
        mime="text/csv"
    )

    # --- Deep Dive Elements (Forecast Side) ---
    st.subheader("🔎 Deep Dive – Seasonal Effects & Anomalies")

    # Seasonal Strength (dummy example)
    seasonal_strength = np.random.uniform(0.5, 0.9)
    st.metric("Seasonal Strength Index", f"{seasonal_strength:.2%}")

    # Holiday Impact (dummy example)
    holiday_impact = np.random.uniform(0.1, 0.3)
    st.metric("Holiday Impact on Revenue", f"{holiday_impact:.2%}")

    # Anomaly Detection (simple placeholder logic)
    st.markdown("### 🚨 Detected Anomalies in Forecast")
    anomalies = forecast[['ds', 'yhat']].sample(3, random_state=42)
    anomalies['note'] = "Unexpected spike/drop"
    st.dataframe(anomalies)

# ----------------------------------------
# Tab 3: Model Evaluation + Deep Dive
# ----------------------------------------
with tab3:
    st.subheader("📏 Model Evaluation & Comparison")

    # Error Metrics
    df_eval = forecast.set_index('ds').join(df.set_index('ds'))
    df_eval = df_eval.dropna()

    mae = np.mean(np.abs(df_eval['y'] - df_eval['yhat']))
    rmse = np.sqrt(np.mean((df_eval['y'] - df_eval['yhat'])**2))
    wape = np.sum(np.abs(df_eval['y'] - df_eval['yhat'])) / np.sum(df_eval['y'])

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}")
    col2.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.2f}")
    col3.metric("WAPE", f"{wape:.2%}")

    # Historical vs Forecasted Comparison
    st.markdown("### 📊 Historical vs Forecasted Revenue")
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(x=df_eval.index, y=df_eval['y'],
                                     mode='lines', name='Historical'))
    fig_compare.add_trace(go.Scatter(x=df_eval.index, y=df_eval['yhat'],
                                     mode='lines', name='Forecasted'))
    st.plotly_chart(fig_compare, use_container_width=True)

    # Component Plots
    st.markdown("### 🔍 Time Series Components")
    fig_components = plot_components_plotly(m, forecast)
    st.plotly_chart(fig_components, use_container_width=True)

    # --- Deep Dive Elements (Evaluation Side) ---
    st.subheader("🔎 Deep Dive – Risk, Recovery & KPIs")

    # Growth Momentum (dummy)
    st.metric("7-Day Momentum", "+4.5%")
    st.metric("30-Day Momentum", "+6.2%")
    st.metric("90-Day Momentum", "+8.1%")

    # Recovery Analysis (dummy)
    st.markdown("### 🔄 Recovery Analysis")
    st.write("Example: Supply chain dip in 2023 took 25 days to recover.")

    # Risk & Volatility (dummy placeholder chart)
    st.markdown("### ⚠️ Risk & Volatility")
    df_eval['volatility'] = df_eval['y'].pct_change().rolling(7).std()
    fig_vol = px.line(df_eval, x=df_eval.index, y='volatility', title="Revenue Volatility Index")
    st.plotly_chart(fig_vol, use_container_width=True)

    # Financial KPIs
    st.markdown("### 💰 Financial KPIs")
    arr = df_eval['yhat'].mean() * 365
    st.metric("Annual Run Rate", f"${arr:,.0f}")
    st.metric("Growth Target Progress", "72%")
    st.metric("Milestones Achieved", "Revenue > $1M, $5M")
