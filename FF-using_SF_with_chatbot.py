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
import snowflake.connector

# --- CONFIGURATION ---
LOGO_PATH = "miracle-logo-dark.png"
CHAT_ICON_PATH = "miralogo.png"
#CSV_FILE_PATH = "financial_forecast_modified.csv"

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

encoded_logo = get_image_base64(LOGO_PATH)
encoded_chat_icon = get_image_base64(CHAT_ICON_PATH)

# --- Custom CSS for Styling (merged & improved) ---
st.markdown(
    f"""
    <style>
        /* Smooth scrolling */
        html {{ scroll-behavior: smooth; }}

        /* Google font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, .stApp {{ font-family: 'Inter', sans-serif; }}

        /* App background */
        .stApp {{
            background-color: #f6f8fb;
            color: #222;
        }}

        /* Section headers with icons */
        h2, h3 {{ display:flex; align-items:center; gap:0.5rem; }}

        /* KPI card styles (brand-aligned gradients) */
        .kpi-container {{
            display:flex;
            flex-direction:column;
            justify-content:space-between;
            background: linear-gradient(135deg,#eef6ff,#f2f8ff);
            padding:1.25rem;
            border-radius:12px;
            box-shadow:0 6px 18px rgba(25,39,64,0.06);
            transition: transform 0.25s ease-in-out;
            height:100%;
            margin-bottom:1rem;
        }}
        .kpi-container-historical {{
            background: linear-gradient(135deg,#eaf4ff,#eef8ff);
        }}
        .kpi-container-forecasted {{
            background: linear-gradient(135deg,#e9fbf0,#f7fff7);
        }}
        .kpi-container:hover {{ transform: translateY(-6px); }}

        .kpi-title {{ font-size:0.95rem; color:#354251; font-weight:600; margin:0; }}
        .kpi-value {{ font-size:1.8rem; font-weight:700; color:#0b6ef6; margin:0.2rem 0; }}
        .kpi-subtitle {{ font-size:0.85rem; color:#556370; margin:0; }}
        .kpi-delta {{ font-size:1rem; font-weight:600; margin-top:0.5rem; }}
        .positive-delta {{ color:#1f9d55; }}
        .negative-delta {{ color:#d45a5a; }}

        /* Sidebar sticky for better UX */
        section[data-testid="stSidebar"] > div:first-child {{
            position: sticky;
            top: 0;
            height: calc(100vh - 1rem);
            overflow-y: auto;
            padding-bottom: 1rem;
        }}

        /* Watermark centered at bottom */
        .watermark {{
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 13px;
            color: rgba(0,0,0,0.06);
            pointer-events: none;
            z-index: 9999;
        }}

        /* Download button styling */
        .stDownloadButton button {{
            background: #0b6ef6;
            color: #fff;
            border-radius: 8px;
            font-weight: 600;
        }}
        .stDownloadButton button:hover {{ background:#084d9c; }}

        /* Date inputs */
        div[data-testid="stDateInput"] input {{
            border-radius: 8px;
            border: 1px solid #ced4da;
            padding: 8px 12px;
            font-size: 1rem;
        }}
        div[data-testid="stDateInput"] input:focus {{
            border-color: #0b6ef6;
            box-shadow: 0 0 0 0.2rem rgba(11,110,246,0.12);
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- App header + guide ---
st.title("📈 Financial Forecasting - Revenue ")

# --- Sidebar controls (sticky) ---
with st.sidebar:
    # logo
    if encoded_logo:
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.error(f"Logo file not found at {LOGO_PATH}")

    st.header("⚙️ Settings")
    st.subheader("Forecast Period")
    forecast_months = st.slider("Select number of months to forecast:", min_value=1, max_value=60, value=36)
    forecast_period_days = forecast_months * 30  # Prophet uses days

    st.subheader("Model Configuration")
    confidence_interval = st.slider("Confidence Interval (%)", min_value=80, max_value=99, value=90, step=1) / 100

    st.markdown("**Seasonality Controls**")
    weekly_seasonality = st.checkbox("Weekly Seasonality", value=True)
    yearly_seasonality = st.checkbox("Yearly Seasonality", value=True)

    st.subheader("What-if Scenario Analysis")
    what_if_enabled = st.checkbox("Apply What-if Scenario to Forecast", value=True)
    what_if_change = st.number_input("Future Revenue Change (%)", min_value=-100.0, max_value=100.0, value=0.0, step=0.5, help="Enter a percentage change to simulate a what-if scenario. Ex: 10 for a 10% increase.")

    st.markdown("---")
    st.header("🔖 Bookmarks")
    st.markdown(
        """
        [Core KPIs](#core-kpis)
        [Growth Metrics](#growth-metrics)
        [Historical Trends](#historical-trends)
        [Daily Revenue](#daily-revenue)
        [Cumulative Revenue](#cumulative-revenue)
        [Forecast Table](#forecast-table)
        [Model Performance](#model-performance)
        [Insights](#insights--recommendations)
        """,
        unsafe_allow_html=True
    )

# --- Data & validation ---

try:
    conn = snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        database=st.secrets["snowflake"]["database"],
        schema=st.secrets["snowflake"]["schema"]
    )
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS financial_forecast_output (
            DS DATE,
            YHAT FLOAT,
            YHAT_LOWER FLOAT,
            YHAT_UPPER FLOAT,
            YHAT_WHAT_IF FLOAT,
            RUN_TIMESTAMP TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)
    conn.commit()


    query = "SELECT DS, Y FROM financial_forecast ORDER BY DS"
    df = pd.read_sql(query, conn)
    #conn.close()
    
    # Force rename columns
    df.columns = df.columns.str.lower()   # turns DS → ds, Y → y
    
    # Ensure correct dtypes
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'])
    forecast_df = pd.read_sql(
        "SELECT DS, YHAT, YHAT_LOWER, YHAT_UPPER, YHAT_WHAT_IF FROM financial_forecast_output ORDER BY DS",
        conn)
    forecast_df['ds'] = pd.to_datetime(forecast_df['DS'])
    conn.close()

except Exception as e:
    st.error(f"❌ Error fetching data from Snowflake: {e}")
    st.stop()

# --- Fit Prophet model with spinner (loading indicator) ---
with st.spinner(" ⏳ Training Prophet model and generating forecast..."):
    holidays_df = pd.DataFrame([
        {'holiday': 'Product Launch Spike', 'ds': pd.to_datetime('2022-07-15'), 'lower_window': -5, 'upper_window': 5},
        {'holiday': 'Supply Chain Dip', 'ds': pd.to_datetime('2023-11-20'), 'lower_window': -5, 'upper_window': 5},
    ])

    # floor to prevent negative forecasts
    df['floor'] = 0

    model = Prophet(weekly_seasonality=weekly_seasonality,
                    yearly_seasonality=yearly_seasonality,
                    holidays=holidays_df,
                    growth='linear',
                    interval_width=confidence_interval)
    model.fit(df)

    future = model.make_future_dataframe(periods=forecast_period_days)
    future['floor'] = 0
    ## this is where we compute the forecast
    forecast = model.predict(future)
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    forecast['yhat_what_if'] = forecast['yhat'] * (1 + what_if_change / 100.0)
    #Convert the DS column to a Python date object
    forecast['ds'] = forecast['ds'].dt.date
    forecast.columns = forecast.columns.str.upper()
# This is where we put the forecast values back into snowflake finance_forecast_output table
try:
    # --- RE-ESTABLISH CONNECTION FOR WRITING ---
    conn_write = snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        database=st.secrets["snowflake"]["database"],
        schema=st.secrets["snowflake"]["schema"]
    )
    cur_write = conn_write.cursor()
    
    # Clear previous run using the new cursor
    cur_write.execute("TRUNCATE TABLE financial_forecast_output")
    conn_write.commit()
    
    from snowflake.connector.pandas_tools import write_pandas
    #write_pandas(conn_write, forecast[['ds','yhat','yhat_lower','yhat_upper','yhat_what_if']], "FINANCIAL_FORECAST_OUTPUT")
    write_pandas(conn_write, forecast[['DS','YHAT','YHAT_LOWER','YHAT_UPPER','YHAT_WHAT_IF']], "FINANCIAL_FORECAST_OUTPUT")
    st.sidebar.success("✅ Forecast saved into Snowflake (financial_forecast_output)")
    forecast.columns = forecast.columns.str.lower()
    conn_write.close()
except Exception as e:
    st.sidebar.error(f"❌ Error saving forecast: {e}")


# Choose forecast column
if what_if_enabled:
    forecast_col = 'yhat_what_if'
else:
    forecast_col = 'yhat'

# Build combined dataframe (for charts that mix historical & forecast)
combined_df = pd.concat([
    df[['ds', 'y']].assign(type='Historical').set_index('ds'),
    forecast.rename(columns={forecast_col: 'y'})[['ds', 'y']].assign(type='Forecast').set_index('ds')
]).reset_index()

# --- Tabs (keep your original two + add Insights) ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Forecast", "📈 Model Performance", "📚 Insights & Recommendations", "💡 Deep Dive Analysis"])

# ---------------------- TAB 1: Forecast ----------------------
with tab1:
    st.markdown('', unsafe_allow_html=True)
    st.subheader("Historical Revenue & 30-Day Moving Average")

    # Calculate 30-day moving average
    df['30_day_avg'] = df['y'].rolling(window=30, min_periods=1).mean()

    # Build historical revenue figure with Plotly
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines',
        name='Historical Daily Revenue',
        line=dict(color='rgba(11,110,246,0.35)', width=1),
        hovertemplate='%{x|%Y-%m-%d}<br>Revenue: %{y:.2f}<extra></extra>'
    ))
    fig_hist.add_trace(go.Scatter(
        x=df['ds'],
        y=df['30_day_avg'],
        mode='lines',
        name='30-Day Moving Avg',
        line=dict(color='rgb(31,157,85)', width=3),
        hovertemplate='%{x|%Y-%m-%d}<br>30-Day Avg: %{y:.2f}<extra></extra>'
    ))

    # Prepare forecast filtering and metrics
    forecast_ds = pd.to_datetime(forecast['ds'])
    max_historical_date = df['ds'].max()
    forecast_filtered = forecast[(forecast_ds > max_historical_date)]

    if not forecast_filtered.empty:
        # Calculate forecast 30-day moving average
        forecast_filtered['30_day_avg'] = forecast_filtered[forecast_col.lower()].rolling(window=30, min_periods=1).mean()

        fig_hist.add_trace(go.Scatter(
            x=forecast_filtered['ds'],
            y=forecast_filtered[forecast_col.lower()],
            mode='lines',
            name='Forecasted Daily Revenue',
            line=dict(color='rgba(22,163,74,0.6)', width=1, dash='dash'),
            hovertemplate='%{x|%Y-%m-%d}<br>Forecast: %{y:.2f}<extra></extra>'
        ))

        fig_hist.add_trace(go.Scatter(
            x=forecast_filtered['ds'],
            y=forecast_filtered['30_day_avg'],
            mode='lines',
            name='Forecasted 30-Day Moving Avg',
            line=dict(color='purple', width=3, dash='dash'),
            hovertemplate='%{x|%Y-%m-%d}<br>Forecast 30-Day Avg: %{y:.2f}<extra></extra>'
        ))

    # Layout options
    fig_hist.update_layout(
        title='Historical Revenue and Moving Average',
        xaxis_title='Date',
        yaxis_title='Revenue',
        template='plotly_white',
        hovermode='x unified',
        xaxis_rangeslider_visible=True,
        transition_duration=500
    )

    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # Core Business Metrics KPIs
    total_historical_revenue = df['y'].sum()
    avg_historical_revenue = df['y'].mean()
    total_forecasted_revenue = forecast_filtered[forecast_col.lower()].sum() if not forecast_filtered.empty else 0.0
    avg_forecasted_revenue = forecast_filtered[forecast_col.lower()].mean() if not forecast_filtered.empty else 0.0

    # Calculate historical CAGR
    first_date_hist = df['ds'].min()
    last_date_hist = df['ds'].max()
    num_years_hist = (last_date_hist - first_date_hist).days / 365.25
    first_revenue_hist = df.loc[df['ds'] == first_date_hist, 'y'].values[0]
    last_revenue_hist = df.loc[df['ds'] == last_date_hist, 'y'].values[0]
    cagr_hist = safe_cagr(first_revenue_hist, last_revenue_hist, num_years_hist)

    # Calculate forecasted CAGR if forecast exists
    if not forecast_filtered.empty:
        first_date_forecast = forecast_filtered['ds'].min()
        last_date_forecast = forecast_filtered['ds'].max()
        num_years_forecast = (last_date_forecast - first_date_forecast).days / 365.25
        first_revenue_forecast = forecast_filtered.loc[forecast_filtered['ds'] == first_date_forecast, forecast_col.lower()].values[0]
        last_revenue_forecast = forecast_filtered.loc[forecast_filtered['ds'] == last_date_forecast, forecast_col.lower()].values[0]
        cagr_forecast = safe_cagr(first_revenue_forecast, last_revenue_forecast, num_years_forecast)
    else:
        cagr_forecast = 0.0

    # Revenue delta
    revenue_delta = total_forecasted_revenue - total_historical_revenue
    revenue_delta_pct = (revenue_delta / total_historical_revenue) * 100 if total_historical_revenue != 0 else 0.0

    # Display KPIs in three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"### Total Historical Revenue\n${total_historical_revenue:,.2f}")
        st.markdown(f"### Average Historical Daily Revenue\n${avg_historical_revenue:,.2f}")
        st.markdown(f"### Historical CAGR\n{cagr_hist:.2%}")

    with col2:
        st.markdown(f"### Total Forecasted Revenue\n${total_forecasted_revenue:,.2f}")
        st.markdown(f"### Average Forecasted Daily Revenue\n${avg_forecasted_revenue:,.2f}")
        st.markdown(f"### Forecasted CAGR\n{cagr_forecast:.2%}")

    with col3:
        st.markdown(f"### Revenue Change vs. Historical\n{revenue_delta_pct:.2f}%")

    st.markdown("---")

    # Growth Metrics (MoM, YoY)
    # Assuming monthly grouping df and forecast df exists, computing MoM and YoY in KPIs
    monthly_revenue_hist = df.set_index('ds').resample('M')['y'].sum().reset_index()
    monthly_revenue_forecast = pd.DataFrame()
    if not forecast_filtered.empty:
        monthly_revenue_forecast = forecast_filtered.set_index('ds').resample('M')[forecast_col.lower()].sum().reset_index()

    latest_mom_hist = 0.0
    if len(monthly_revenue_hist) >= 2:
        latest_mom_hist = ((monthly_revenue_hist['y'].iloc[-1] - monthly_revenue_hist['y'].iloc[-2]) / monthly_revenue_hist['y'].iloc[-2]) * 100

    latest_mom_forecast = 0.0
    if not monthly_revenue_forecast.empty and len(monthly_revenue_forecast) >= 2:
        latest_mom_forecast = ((monthly_revenue_forecast[forecast_col.lower()].iloc[1] - monthly_revenue_hist['y'].iloc[-1]) / monthly_revenue_hist['y'].iloc[-1]) * 100

    latest_yoy_hist = 0.0
    if len(monthly_revenue_hist) >= 13:
        latest_yoy_hist = ((monthly_revenue_hist['y'].iloc[-1] - monthly_revenue_hist['y'].iloc[-13]) / monthly_revenue_hist['y'].iloc[-13]) * 100

    latest_yoy_forecast = 0.0
    if not monthly_revenue_forecast.empty and len(monthly_revenue_forecast) >= 12:
        latest_yoy_forecast = ((monthly_revenue_forecast[forecast_col.lower()].iloc[-1] - monthly_revenue_forecast[forecast_col.lower()].iloc[-13]) / monthly_revenue_forecast[forecast_col.lower()].iloc[-13]) * 100

    # Display growth metrics KPIs
    colg1, colg2, colg3, colg4 = st.columns(4)
    with colg1: st.metric("Latest Historical MoM Growth %", f"{latest_mom_hist:.2f}")
    with colg2: st.metric("Latest Forecasted MoM Growth %", f"{latest_mom_forecast:.2f}")
    with colg3: st.metric("Latest Historical YoY Growth %", f"{latest_yoy_hist:.2f}")
    with colg4: st.metric("Latest Forecasted YoY Growth %", f"{latest_yoy_forecast:.2f}")

# ---------------------- TAB 2: Model Performance ----------------------
with tab2:
    st.markdown('', unsafe_allow_html=True)
    st.subheader("Model Performance Metrics and Time Series Decomposition")

    # Calculate residuals (difference between actual and forecast)
    historical_forecast = forecast.loc[forecast['DS'] <= df['ds'].max()]
    if historical_forecast.empty or df.empty:
        st.warning("Insufficient data for model performance metrics")
    else:
        merged_hist = pd.merge(df, historical_forecast, left_on='ds', right_on='DS')
        residuals = merged_hist['y'] - merged_hist[forecast_col.lower()]

        # Compute MAE, RMSE, WAPE, Forecast Bias
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        sum_actual = merged_hist['y'].sum()
        sum_abs_error = np.sum(np.abs(residuals))
        wape = (sum_abs_error / sum_actual) if sum_actual != 0 else np.nan
        forecast_bias = np.sum(residuals) / sum_actual if sum_actual != 0 else np.nan

        st.markdown("### Forecast Error Metrics")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        col_m2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
        col_m3.metric("Weighted Absolute Percentage Error (WAPE)", f"{wape:.2%}")
        col_m4.metric("Forecast Bias", f"{forecast_bias:.2%}")

        st.markdown("---")

        # Time series decomposition using Prophet's components plot
        try:
            components_fig = plot_components_plotly(Prophet().fit(df), forecast)
            st.plotly_chart(components_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate model component plots: {e}")

        st.markdown("---")

        # Residual distribution histogram with anomaly highlights
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Histogram(
            x=residuals,
            nbinsx=50,
            name="Residuals",
            marker_color='lightblue'
        ))

        anomalies = enhanced_anomaly_detection(residuals)
        if not anomalies.empty:
            fig_resid.add_trace(go.Scatter(
                x=anomalies,
                y=[0]*len(anomalies),
                mode='markers',
                marker=dict(color='red', size=10),
                name='Anomalies'
            ))

        fig_resid.update_layout(
            title="Residual Distribution with Detected Anomalies",
            xaxis_title="Residual",
            yaxis_title="Frequency",
            template="plotly_white"
        )
        st.plotly_chart(fig_resid, use_container_width=True)

        st.markdown("---")

        # Residual time series plot
        fig_resid_time = go.Figure()
        fig_resid_time.add_trace(go.Scatter(
            x=merged_hist['ds'],
            y=residuals,
            mode='lines+markers',
            name="Residuals",
            marker=dict(color='orange')
        ))
        st.plotly_chart(fig_resid_time, use_container_width=True)

        st.markdown("---")

        # Highlight forecasting bias
        st.markdown("### Forecast Bias Analysis")
        bias_txt = "Positive bias means systematic over-forecasting, negative bias means under-forecasting."
        st.info(bias_txt)
        st.metric("Overall Forecast Bias", f"{forecast_bias:.2%}")
   
# ---------------------- TAB 3: Insights & Recommendations ----------------------
with tab3:
    st.markdown('', unsafe_allow_html=True)
    st.subheader("Insights & Recommendations")

    # Summary Revenue Growth Metrics (MoM, YoY) - in percentage
    monthly_revenue_hist = df.set_index('ds').resample('M')['y'].sum()
    monthly_revenue_forecast = forecast.set_index('DS').resample('M')[forecast_col].sum()

    mom_growth_hist = monthly_revenue_hist.pct_change() * 100
    mom_growth_forecast = monthly_revenue_forecast.pct_change() * 100

    yoy_growth_hist = monthly_revenue_hist.pct_change(periods=12) * 100
    yoy_growth_forecast = monthly_revenue_forecast.pct_change(periods=12) * 100

    insights = {
        "Latest Historical MoM Growth": mom_growth_hist.iloc[-1] if len(mom_growth_hist) > 1 else np.nan,
        "Latest Forecasted MoM Growth": mom_growth_forecast.iloc[0] if len(mom_growth_forecast) > 0 else np.nan,
        "Latest Historical YoY Growth": yoy_growth_hist.iloc[-1] if len(yoy_growth_hist) > 12 else np.nan,
        "Latest Forecasted YoY Growth": yoy_growth_forecast.iloc[-1] if len(yoy_growth_forecast) > 12 else np.nan,
    }
    insights_df = pd.DataFrame(list(insights.items()), columns=["Metric", "Value"])
    st.table(insights_df.style.format({"Value": "{:.2f}%"}))

    # Anomaly detection summary from residuals
    merged_hist = pd.merge(df, forecast, left_on='ds', right_on='DS')
    residuals = merged_hist['y'] - merged_hist[forecast_col.lower()]
    anomalies = enhanced_anomaly_detection(residuals)

    if anomalies.empty:
        st.success("No significant revenue anomalies detected.")
    else:
        st.warning(f"Detected {len(anomalies)} significant anomalies in historical revenue.")

    # Textual Recommendations based on insights and anomalies
    st.markdown("### Recommendations:")
    if cagr_hist < 0:
        st.markdown("- Historical revenue growth is negative. Consider strategies for business turnaround.")
    else:
        st.markdown(f"- Historical CAGR is positive at {cagr_hist:.2%}. Focus on sustaining and accelerating growth.")

    if revenue_delta_pct > 0:
        st.markdown(f"- Forecast indicates revenue increase by {revenue_delta_pct:.2f}%. Consider operational scaling.")
    else:
        st.markdown(f"- Forecast indicates revenue decrease by {abs(revenue_delta_pct):.2f}%. Prioritize cost control and margin management.")

    if not anomalies.empty:
        st.markdown("- Multiple revenue anomalies were detected. Conduct detailed root cause analyses for these deviations.")

    # Volatility and Risk Metrics
    residual_std = np.std(residuals)
    st.metric("Forecast Residual Standard Deviation (Volatility)", f"{residual_std:.4f}")

    rolling_vol = residuals.rolling(window=30).std()
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=merged_hist['ds'],
        y=rolling_vol,
        mode='lines',
        name='30-day Residual Volatility'
    ))
    fig_vol.update_layout(
        title="Rolling Volatility of Forecast Residuals",
        xaxis_title="Date",
        yaxis_title="Residual Standard Deviation",
        template='plotly_white'
    )
    st.plotly_chart(fig_vol, use_container_width=True)   
# ---------------------- TAB 4: Deep Dive Analysis ----------------------
with tab4:
    st.markdown('', unsafe_allow_html=True)
    st.subheader("Deep Dive Analysis")

    # Example: Supply Chain Dips Visualized
    st.markdown("### Supply Chain Impact Analysis")
    
    supply_chain_events = [
        {"date": "2023-02-15", "event": "Supply Chain Delay"},
        {"date": "2023-06-10", "event": "Logistics Issue"},
        {"date": "2023-11-20", "event": "Supplier Strike"}
    ]
    events_df = pd.DataFrame(supply_chain_events)
    events_df['date'] = pd.to_datetime(events_df['date'])
    
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines',
        name='Historical Revenue',
        line=dict(color='blue')
    ))
    for _, row in events_df.iterrows():
        fig_sc.add_vline(x=row['date'], line_width=2, line_dash="dash", line_color="red")
        fig_sc.add_annotation(
            x=row['date'], y=max(df['y']),
            text=row['event'], showarrow=True, arrowhead=1,
            ay=-40, font=dict(color="red")
        )
    fig_sc.update_layout(
        title="Revenue with Supply Chain Events",
        xaxis_title="Date",
        yaxis_title="Revenue",
        template='plotly_white'
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")

    # Seasonality Strength Metrics (using Prophet seasonality components)
    try:
        model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
        model.fit(df)
        seasonal_components = model.predict_seasonal_components(df)
        # Calculate strength as max-min amplitude ratio
        weekly_strength = (seasonal_components['weekly'].max() - seasonal_components['weekly'].min()) / seasonal_components['weekly'].abs().max()
        yearly_strength = (seasonal_components['yearly'].max() - seasonal_components['yearly'].min()) / seasonal_components['yearly'].abs().max()
        st.metric("Weekly Seasonality Strength", f"{weekly_strength:.2%}")
        st.metric("Yearly Seasonality Strength", f"{yearly_strength:.2%}")
    except Exception as e:
        st.warning(f"Could not compute seasonality strength: {e}")

    st.markdown("---")

    # Revenue Drawdown Analysis
    st.markdown("### Revenue Drawdown Analysis")
    running_max = df['y'].cummax()
    drawdown = (df['y'] - running_max) / running_max
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=df['ds'],
        y=drawdown,
        mode='lines',
        name='Drawdown',
        line=dict(color='red')
    ))
    fig_dd.update_layout(
        yaxis=dict(tickformat=".0%"),
        title="Revenue Drawdowns",
        xaxis_title="Date",
        yaxis_title="Drawdown %",
        template='plotly_white'
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("---")

    # Growth Tracking & Forecast Accuracy
    st.markdown("### Growth Tracking vs Forecast Accuracy")

    growth_diff = (forecast_df[forecast_col] - df.set_index('ds')['y']).dropna()
    avg_growth_diff = growth_diff.mean()

    st.metric("Average Growth Difference (Forecast vs Actual)", f"{avg_growth_diff:.2f}")

    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(
        x=forecast_df['DS'],
        y=forecast_df[forecast_col],
        mode='lines',
        name='Forecast'
    ))
    fig_growth.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines',
        name='Actual'
    ))
    fig_growth.update_layout(
        title="Forecast vs Actual Revenue",
        xaxis_title="Date",
        yaxis_title="Revenue",
        template='plotly_white'
    )
    st.plotly_chart(fig_growth, use_container_width=True)

    st.markdown("---")

    # Automated Actionable Recommendations
    st.markdown("### Automated Actionable Recommendations")

    if revenue_delta_pct > 5:
        st.markdown("- Forecast indicates strong revenue growth. Consider planned scaling of production capacity.")
    elif revenue_delta_pct < -5:
        st.markdown("- Revenue decline ahead, assess and reduce operational expenses immediately.")
    else:
        st.markdown("- Revenue expected to stay steady. Focus on maintaining customer satisfaction and operational efficiency.")
# --- Centered Watermark (updated text as requested) ---
st.markdown('<p class="watermark">Created by Miracle Software Systems for AI for Business</p>', unsafe_allow_html=True)

# --- Floating Chat (DROP-IN) ---
# Keep this near the bottom of your file, AFTER your existing app logic.
# Uses your API and shows only the summarized text + tables.

# --- Floating Chat (GLOBAL OVERLAY via Shadow DOM) ---
# Works above the whole Streamlit page; includes robust API error messages.

import base64, os
import streamlit as st

CHAT_ICON_PATH = "miralogo.png"

try:
    with open(CHAT_ICON_PATH, "rb") as _img:
        _CHAT_ICON_B64 = base64.b64encode(_img.read()).decode()
except Exception:
    _CHAT_ICON_B64 = ""  # falls back to emoji

overlay_html = """
<script>
(function () {
  // Create a Shadow DOM overlay in the PARENT page (so it’s never clipped/hidden)
  var canParent = false;
  try { canParent = !!window.parent && !!window.parent.document && !!window.parent.document.body; } catch (e) { canParent = false; }
  var hostDoc = canParent ? window.parent.document : document;

  var ROOT_ID = "mira-overlay-root";
  var hostDiv = hostDoc.getElementById(ROOT_ID);
  if (!hostDiv) {
    hostDiv = hostDoc.createElement("div");
    hostDiv.id = ROOT_ID;
    hostDoc.body.appendChild(hostDiv);
  }

  var shadow = hostDiv.shadowRoot || hostDiv.attachShadow({ mode: "open" });

  var ICON_B64 = "__ICON_B64__";
  var hasIcon = !!ICON_B64;

  var css = ''
  + '.mira-chat-toggle{position:fixed;bottom:18px;right:18px;width:56px;height:56px;border-radius:50%;'
  + 'background:#fff;border:2px solid #007bff;box-shadow:0 8px 24px rgba(0,0,0,.18);display:grid;place-items:center;cursor:pointer;'
  + 'z-index:9999999999;background-repeat:no-repeat;background-position:center;background-size:30px 30px;animation:mira-pulse 2s infinite;}'
  + (hasIcon ? ('.mira-chat-toggle{background-image:url(data:image/png;base64,' + ICON_B64 + ');} .mira-chat-toggle span{display:none;}')
             : ('.mira-chat-toggle span{font-size:22px;display:block;}'))
  + '@keyframes mira-pulse{0%{box-shadow:0 0 0 0 rgba(0,123,255,.45);}70%{box-shadow:0 0 0 16px rgba(0,123,255,0);}100%{box-shadow:0 0 0 0 rgba(0,123,255,0);}}'
  + '.mira-chat-modal{position:fixed;bottom:84px;right:18px;width:360px;max-width:96vw;height:480px;max-height:70vh;background:#fff;color:#111827;'
  + 'border:1px solid #e5e7eb;border-radius:16px;box-shadow:0 24px 48px rgba(0,0,0,.18);display:none;flex-direction:column;overflow:hidden;z-index:9999999999;}'
  + '@media (max-width:480px){.mira-chat-modal{left:12px;right:12px;width:auto;height:70vh;}}'
  + '.mira-chat-header{background:#007bff;color:#fff;padding:10px 14px;font-weight:600;display:flex;align-items:center;justify-content:space-between;}'
  + '.mira-chat-header button{background:transparent;border:none;color:#fff;font-size:18px;cursor:pointer;}'
  + '.mira-chat-body{flex:1;background:#fafbfc;padding:12px;overflow-y:auto;display:flex;flex-direction:column;gap:8px;}'
  + '.mira-msg{max-width:85%;padding:10px 12px;border-radius:12px;border:1px solid #e5e7eb;line-height:1.35;font-size:14px;white-space:pre-wrap;word-break:break-word;}'
  + '.mira-user{align-self:flex-end;background:#e9f3ff;border-color:#cfe5ff;} .mira-bot{align-self:flex-start;background:#f5f7fb;}'
  + '.mira-typing{align-self:flex-start;display:inline-flex;gap:6px;padding:8px 10px;border-radius:12px;border:1px solid #e5e7eb;background:#f5f7fb;color:#6b7280;font-size:14px;}'
  + '.mira-typing .dot{width:6px;height:6px;border-radius:50%;background:#6b7280;opacity:.6;animation:mira-blink 1.2s infinite;}'
  + '.mira-typing .dot:nth-child(2){animation-delay:.2s;} .mira-typing .dot:nth-child(3){animation-delay:.4s;}'
  + '@keyframes mira-blink{0%,80%,100%{transform:scale(.6);opacity:.4}40%{transform:scale(1);opacity:1}}'
  + '.mira-chat-input{display:flex;gap:8px;padding:10px;border-top:1px solid #e5e7eb;background:#fff;}'
  + '.mira-chat-input input{flex:1;font-size:14px;border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;outline:none;}'
  + '.mira-chat-input button{background:#007bff;color:#fff;border:none;border-radius:10px;padding:10px 14px;font-weight:600;cursor:pointer;}'
  + '.mira-chat-input button:disabled{opacity:.65;cursor:not-allowed;}'
  + '.mira-table-wrap{overflow-x:auto;} .mira-table{border-collapse:collapse;width:100%;font-size:13px;margin-top:6px;}'
  + '.mira-table th,.mira-table td{border:1px solid #e5e7eb;padding:6px 8px;text-align:left;} .mira-table th{background:#f3f4f6;}';

  shadow.innerHTML = ''
    + '<style>' + css + '</style>'
    + '<button class="mira-chat-toggle" id="miraToggle" aria-label="Open chat"><span>💬</span></button>'
    + '<div class="mira-chat-modal" id="miraModal" role="dialog" aria-modal="true" aria-label="Mira Chat">'
    + '  <div class="mira-chat-header"><span>💬 Chat Assistant</span><button id="miraClose" aria-label="Close chat">✖</button></div>'
    + '  <div class="mira-chat-body" id="miraBody"></div>'
    + '  <div class="mira-chat-input"><input id="miraInput" type="text" placeholder="Ask your question..." /><button id="miraSend">Send</button></div>'
    + '</div>';

  // 👇 CHANGE #1: point to your Cloud Run proxy URL (no access key here)
  var API_URL = "https://mira-proxy-582396939090.us-central1.run.app";

  var modal  = shadow.getElementById("miraModal");
  var toggle = shadow.getElementById("miraToggle");
  var closeB = shadow.getElementById("miraClose");
  var body   = shadow.getElementById("miraBody");
  var input  = shadow.getElementById("miraInput");
  var send   = shadow.getElementById("miraSend");

  function openChat(){ modal.style.display="flex"; setTimeout(function(){ input.focus(); }, 50); }
  function closeChat(){ modal.style.display="none"; }
  function scrollToBottom(){ body.scrollTop = body.scrollHeight; }

  function bubble(role, html){
    var div = document.createElement("div");
    div.className = "mira-msg " + (role === "user" ? "mira-user" : "mira-bot");
    if (typeof html === "string") div.innerText = html; else div.appendChild(html);
    body.appendChild(div); scrollToBottom();
  }

  function tableFromRows(rows){
    var table = document.createElement("table"); table.className="mira-table";
    var head = document.createElement("thead"); var htr = document.createElement("tr");
    var keys = Object.keys(rows[0] || {});
    keys.forEach(function(k){ var th=document.createElement("th"); th.textContent=k; htr.appendChild(th); });
    head.appendChild(htr); table.appendChild(head);
    var tb = document.createElement("tbody");
    rows.slice(0,10).forEach(function(r){
      var tr=document.createElement("tr");
      keys.forEach(function(k){
        var td=document.createElement("td"); var v=r[k];
        td.textContent = (typeof v === "number") ? v.toLocaleString() : (v == null ? "" : String(v));
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
    table.appendChild(tb);
    var wrap=document.createElement("div"); wrap.className="mira-table-wrap"; wrap.appendChild(table); return wrap;
  }

  var busy=false, typingEl=null;
  function setBusy(state){
    busy=state; send.disabled=state; input.disabled=state;
    if(state){
      typingEl=document.createElement("div");
      typingEl.className="mira-typing";
      typingEl.innerHTML='<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
      body.appendChild(typingEl);
    } else if (typingEl){ typingEl.remove(); typingEl=null; }
    scrollToBottom();
  }

  async function sendMessage(){
    if (busy) return;
    var q = (input.value || "").trim(); if (!q) return;
    bubble("user", q); input.value=""; setBusy(true);

    try{
      // 👇 CHANGE #2: only Content-Type header; the proxy adds access-key
      var res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: q }),
        mode: "cors"
      });

      if (!res.ok){
        var txt = "";
        try{ txt = await res.text(); }catch(_){}
        bubble("bot", "API error " + res.status + (txt ? (": " + txt.slice(0,160)) : ""));
        return;
      }

      var data = {};
      try{ data = await res.json(); }catch(_){ data = {}; }

      var resp    = (data && data.response) ? data.response : {};
      var summary = resp.summerized || resp.summarized || "";
      var rows    = Array.isArray(resp.data) ? resp.data : [];

      if (summary) bubble("bot", summary);
      if (rows && rows.length) bubble("bot", tableFromRows(rows));
      if (!summary && (!rows || !rows.length)) bubble("bot", "No summary or data returned.");
    } catch (e){
      bubble("bot", "Request failed: " + (e && e.message ? e.message : ""));
      console.error(e);
    } finally {
      setBusy(false);
    }
  }

  toggle.addEventListener("click", function(){ (modal.style.display === "flex") ? closeChat() : openChat(); });
  closeB.addEventListener("click", closeChat);
  send.addEventListener("click", sendMessage);
  input.addEventListener("keydown", function(e){ if (e.key === "Enter" && !e.shiftKey){ e.preventDefault(); sendMessage(); } });
})();
</script>
"""


# inject the icon data without needing f-string brace escaping
overlay_html = overlay_html.replace("__ICON_B64__", _CHAT_ICON_B64)

# Render with zero height; overlay is attached to parent DOM and floats globally
st.components.v1.html(overlay_html, height=0, scrolling=False)
