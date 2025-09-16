import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ----------------------------------------
# Inject Custom CSS for a clean, modern look
# ----------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
        color: #333333; /* Darker text */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50; /* Green button */
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stMetric > div[data-testid="stMetricLabel"] {
        font-weight: bold;
        color: #4CAF50;
    }
    h1, h2, h3, h4 {
        color: #1a1a1a;
    }
    .st-emotion-cache-1r6y4y9 {
        padding-top: 2rem;
    }
    .st-emotion-cache-1r6y4y9 .st-emotion-cache-1r6y4y9 {
        padding-top: 0;
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

# ----------------------------------------
# Sidebar for user input & logo
# ----------------------------------------

# Add Company Logo
try:
    logo = Image.open('miracle-logo-dark.png') 
    st.sidebar.image(logo, use_column_width=True)
except FileNotFoundError:
    st.sidebar.error("Logo file not found. Please ensure 'your_logo.png' is in the same directory.")

st.sidebar.header("⚙️ Configuration")
forecast_periods = st.sidebar.slider(
    "Forecast Horizon (Months):", 12, 24, 36
)

confidence_interval = st.sidebar.slider(
    "Confidence Interval", 80, 99, 95
)

# ----------------------------------------
# Add the "What If" Scenario Slider
# ----------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("What If Scenario")
revenue_change_pct = st.sidebar.slider(
    "Simulate Revenue Change (%)", -25, 25, 0, format="%d%%"
)

weekly_seasonality = st.sidebar.checkbox("Include Weekly Seasonality", True)
yearly_seasonality = st.sidebar.checkbox("Include Yearly Seasonality", True)

# ----------------------------------------
# Main Content - Storytelling Scroll Layout
# ----------------------------------------

# 1. Header and Executive Summary
st.title("🔮 Financial Forecasting Dashboard")
# Placeholder dataset (replace with your actual data loading)
@st.cache_data
def load_data():
    dates = pd.date_range(start="2021-01-01", end="2024-12-31", freq="D")
    revenue = np.random.randint(1000, 5000, len(dates)).astype(float)
    df = pd.DataFrame({"ds": dates, "y": revenue})
    return df

df = load_data()

# Prophet Forecasting
m = Prophet(
    interval_width=confidence_interval,
    weekly_seasonality=weekly_seasonality,
    yearly_seasonality=yearly_seasonality
)
m.fit(df)
future = m.make_future_dataframe(periods=forecast_periods * 30)
forecast = m.predict(future)

# Apply "What If" scenario
forecast['yhat'] = forecast['yhat'] * (1 + revenue_change_pct / 100)
forecast['yhat_lower'] = forecast['yhat_lower'] * (1 + revenue_change_pct / 100)
forecast['yhat_upper'] = forecast['yhat_upper'] * (1 + revenue_change_pct / 100)

# ----------------------------------------
# 2. The Main Event: The Forecast
# ----------------------------------------
st.header("🔮 Forecasted Revenue Outlook")

# Display key forecasted metrics
col1, col2, col3 = st.columns(3)
with col1:
    # Use CAGR for percentage display
    if len(forecast) > 1 and len(df) > 1:
        proj_cagr = ((forecast['yhat'].iloc[-1] / df['y'].iloc[0]) ** (1/(len(forecast)/365))) - 1
    else:
        proj_cagr = 0
    st.metric("Projected CAGR", f"{proj_cagr:.2%}")

with col2:
    # Calculate YOY growth for percentage display
    df_yoy = forecast.set_index('ds').resample('M')['yhat'].sum()
    yoy_growth = (df_yoy.iloc[-1] / df_yoy.iloc[-13]) - 1 if len(df_yoy) >= 13 else 0
    st.metric("YoY Forecast Growth", f"{yoy_growth:.2%}")

with col3:
    # Calculate M-o-M growth for percentage display
    mom_growth = (df_yoy.iloc[-1] / df_yoy.iloc[-2]) - 1 if len(df_yoy) >= 2 else 0
    st.metric("Month-over-Month Growth", f"{mom_growth:.2%}")


# Main forecast chart with interactive capabilities
st.subheader("📅 Daily Revenue Forecast with Confidence Interval")
fig_forecast = plot_plotly(m, forecast)
fig_forecast.update_layout(height=500)
st.plotly_chart(fig_forecast, use_container_width=True)

# ----------------------------------------
# 3. The "Why": Model Components
# ----------------------------------------
st.header("🧠 Understanding the Forecast: Time Series Components")

fig_components = plot_components_plotly(m, forecast)
st.plotly_chart(fig_components, use_container_width=True)

# ----------------------------------------
# 4. The "How Well": Model Evaluation
# ----------------------------------------
st.header("📏 Model Performance and Accuracy")
df_eval = forecast.set_index('ds').join(df.set_index('ds'))
df_eval = df_eval.dropna()

col1_eval, col2_eval, col3_eval = st.columns(3)
with col1_eval:
    mae = np.mean(np.abs(df_eval['y'] - df_eval['yhat']))
    st.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}")
with col2_eval:
    rmse = np.sqrt(np.mean((df_eval['y'] - df_eval['yhat'])**2))
    st.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.2f}")
with col3_eval:
    wape = np.sum(np.abs(df_eval['y'] - df_eval['yhat'])) / np.sum(df_eval['y'])
    st.metric("WAPE", f"{wape:.2%}")

st.subheader("Historical vs. Forecasted Revenue")
fig_compare = go.Figure()
fig_compare.add_trace(go.Scatter(x=df_eval.index, y=df_eval['y'], mode='lines', name='Historical'))
fig_compare.add_trace(go.Scatter(x=df_eval.index, y=df_eval['yhat'], mode='lines', name='Forecasted'))
st.plotly_chart(fig_compare, use_container_width=True)

# ----------------------------------------
# 5. The "What's Next": Deeper Insights
# ----------------------------------------
st.header("📊 Deeper Dive: Historical Trends")

col1_hist, col2_hist = st.columns(2)
with col1_hist:
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
with col2_hist:
    st.markdown("#### 30-Day Moving Average")
    df['30d_ma'] = df['y'].rolling(window=30).mean()
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Daily Revenue', line=dict(color='#888888')))
    fig_ma.add_trace(go.Scatter(x=df['ds'], y=df['30d_ma'], mode='lines',
                                name='30-Day Moving Average', line=dict(color='#4CAF50', width=3)))
    st.plotly_chart(fig_ma, use_container_width=True)

st.subheader("📋 Raw Forecast Data")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))
st.download_button(
    label="Download Forecast Data (CSV)",
    data=forecast.to_csv(index=False),
    file_name="forecast.csv",
    mime="text/csv",
)
