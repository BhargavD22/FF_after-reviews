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
st.title("📈 Financial Forecasting Dashboard")
st.markdown("A **dynamic** application to analyze historical revenue data and forecast future trends using the **Prophet** model.")

with st.expander("ℹ️ Quick Guide", expanded=False):
    st.markdown(
        """
        **How to use**
        1. Configure the forecast horizon and what-if scenario in the sidebar.  
        2. Explore Forecast and Model Performance tabs.  
        3. Visit Insights for an auto-summarized recommendation.  
        """
    )

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
st.header("Data & Analysis")

if not os.path.exists(CSV_FILE_PATH):
    st.error(f"The required data file '{CSV_FILE_PATH}' was not found in the repository. Please ensure it is present.")
    st.stop()

try:
    df = pd.read_csv(CSV_FILE_PATH)
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'])
except Exception as e:
    st.error(f"Error reading the file from the repository. Please ensure it's a valid CSV with 'ds' and 'y' columns. Error: {e}")
    st.stop()

# --- Fit Prophet model with spinner (loading indicator) ---
with st.spinner("Training Prophet model and generating forecast..."):
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

    forecast = model.predict(future)
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    
    # Calculate what-if scenario forecast
    forecast['yhat_what_if'] = forecast['yhat'] * (1 + what_if_change / 100.0)

# Choose forecast column based on user selection
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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Forecast", "📈 Model Performance", "📚 Insights & Recommendations", "🔍 Deep Dive Analysis"])

# ---------------------- TAB 1: Forecast ----------------------
with tab1:
    # ---- KPIs (Core Business Metrics) ----
    st.markdown('<div id="core-kpis"></div>', unsafe_allow_html=True)
    st.markdown("### 🔑 Core Business Metrics")
    total_historical_revenue = df['y'].sum()
    avg_historical_revenue = df['y'].mean()
    forecast_df = forecast[forecast['ds'] > df['ds'].max()]
    total_forecasted_revenue = forecast_df[forecast_col].sum() if not forecast_df.empty else 0.0
    avg_forecasted_revenue = forecast_df[forecast_col].mean() if not forecast_df.empty else 0.0

    # deltas
    total_revenue_delta = ((total_forecasted_revenue - total_historical_revenue) / total_historical_revenue * 100) if total_historical_revenue != 0 else 0.0
    avg_revenue_delta = ((avg_forecasted_revenue - avg_historical_revenue) / avg_historical_revenue * 100) if avg_historical_revenue != 0 else 0.0

    # CAGR calculations (safely)
    first_date_hist = df['ds'].min()
    last_date_hist = df['ds'].max()
    first_revenue_hist = df.loc[df['ds'] == first_date_hist, 'y'].iloc[0]
    last_revenue_hist = df.loc[df['ds'] == last_date_hist, 'y'].iloc[0]
    num_years_hist = (last_date_hist - first_date_hist).days / 365.25
    cagr_hist = safe_cagr(first_revenue_hist, last_revenue_hist, num_years_hist)

    first_date_forecast = df['ds'].max()
    if not forecast_df.empty:
        last_date_forecast = forecast_df['ds'].max()
        first_revenue_forecast = df.loc[df['ds'] == first_date_forecast, 'y'].iloc[0]
        last_revenue_forecast = forecast_df.loc[forecast_df['ds'] == last_date_forecast, forecast_col].iloc[0]
        num_years_forecast = (last_date_forecast - first_date_forecast).days / 365.25
        cagr_forecast = safe_cagr(first_revenue_forecast, last_revenue_forecast, num_years_forecast)
    else:
        cagr_forecast = 0.0

    col_kpi1, col_kpi2 = st.columns(2)
    with col_kpi1:
        st.markdown(
            f"""
            <div class="kpi-container kpi-container-historical">
                <p class="kpi-title">Total Historical Revenue</p>
                <p class="kpi-value">${total_historical_revenue/1000:,.2f}M</p>
                <p class="kpi-subtitle">Sum of all past revenue</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="kpi-container kpi-container-historical">
                <p class="kpi-title">Avg. Daily Historical Revenue</p>
                <p class="kpi-value">${avg_historical_revenue:,.2f}</p>
                <p class="kpi-subtitle">Average daily revenue in the past</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="kpi-container kpi-container-historical">
                <p class="kpi-title">Historical CAGR</p>
                <p class="kpi-value">{cagr_hist:,.2%}</p>
                <p class="kpi-subtitle">Avg. annual growth rate</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_kpi2:
        st.markdown("#### Forecasted Metrics")
        delta_icon_total = "⬆️" if total_revenue_delta > 0 else "⬇️" if total_revenue_delta < 0 else "➡️"
        delta_class_total = "positive-delta" if total_revenue_delta > 0 else "negative-delta"
        st.markdown(
            f"""
            <div class="kpi-container kpi-container-forecasted">
                <p class="kpi-title">Total Forecasted Revenue</p>
                <p class="kpi-value">${total_forecasted_revenue/1000:,.2f}M</p>
                <p class="kpi-subtitle">Forecasted over {forecast_months} months</p>
                <div class="kpi-delta {delta_class_total}">
                    <span class="delta-icon">{delta_icon_total}</span>
                    <span>{total_revenue_delta:,.2f}% vs. Historical</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="kpi-container kpi-container-forecasted">
                <p class="kpi-title">Avg. Daily Forecasted Revenue</p>
                <p class="kpi-value">${avg_forecasted_revenue:,.2f}</p>
                <p class="kpi-subtitle">Forecasted Avg. over {forecast_months} months</p>
                <div class="kpi-delta {'positive-delta' if avg_revenue_delta>0 else 'negative-delta'}">
                    <span class="delta-icon">{'⬆️' if avg_revenue_delta>0 else '⬇️' if avg_revenue_delta<0 else '➡️'}</span>
                    <span>{avg_revenue_delta:,.2f}% vs. Historical</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="kpi-container kpi-container-forecasted">
                <p class="kpi-title">Forecasted CAGR</p>
                <p class="kpi-value">{cagr_forecast:,.2%}</p>
                <p class="kpi-subtitle">Avg. annual growth rate</p>
                <div class="kpi-delta {'positive-delta' if cagr_forecast>cagr_hist else 'negative-delta'}">
                    <span class="delta-icon">{'⬆️' if cagr_forecast>cagr_hist else '⬇️' if cagr_forecast<cagr_hist else '➡️'}</span>
                    <span>vs. Historical CAGR</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ---- Growth Metrics (MoM & YoY) with monthly chart and arrow markers ----
    with st.expander("📈 Growth Metrics", expanded=True):
        st.markdown('<div id="growth-metrics"></div>', unsafe_allow_html=True)
        st.subheader("Growth Metrics: MoM & YoY")

        # Date selectors
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            start_date_growth = st.date_input(
                "Start Date (Growth Metrics):",
                value=df['ds'].min().date(),
                min_value=df['ds'].min().date(),
                max_value=df['ds'].max().date(),
                key='growth_start'
            )
        with col_g2:
            end_date_growth = st.date_input(
                "End Date (Growth Metrics):",
                value=df['ds'].max().date(),
                min_value=df['ds'].min().date(),
                max_value=df['ds'].max().date(),
                key='growth_end'
            )

        historical_growth_df = df[(df['ds'].dt.date >= start_date_growth) & (df['ds'].dt.date <= end_date_growth)].copy()
        historical_growth_df['month'] = historical_growth_df['ds'].dt.to_period('M').dt.to_timestamp()
        monthly_revenue_hist = historical_growth_df.groupby('month')['y'].sum().reset_index()
        monthly_revenue_hist['MoM_Growth'] = monthly_revenue_hist['y'].pct_change() * 100
        monthly_revenue_hist['direction'] = monthly_revenue_hist['MoM_Growth'].apply(lambda x: 'up' if x>0 else ('down' if x<0 else 'flat'))

        # Forecast monthly aggregates
        forecast_df['month'] = forecast_df['ds'].dt.to_period('M').dt.to_timestamp()
        if what_if_enabled:
            monthly_revenue_forecast = forecast_df.groupby('month')['yhat_what_if'].sum().reset_index().rename(columns={'yhat_what_if':'y'})
        else:
            monthly_revenue_forecast = forecast_df.groupby('month')['yhat'].sum().reset_index().rename(columns={'yhat':'y'})
        monthly_revenue_forecast['MoM_Growth'] = monthly_revenue_forecast['y'].pct_change()*100
        monthly_revenue_forecast['direction'] = monthly_revenue_forecast['MoM_Growth'].apply(lambda x: 'up' if x>0 else ('down' if x<0 else 'flat'))

        # Show latest MoM & YoY as KPI cards
        latest_mom_hist = monthly_revenue_hist['MoM_Growth'].iloc[-1] if not monthly_revenue_hist.empty and len(monthly_revenue_hist) > 1 else 0
        latest_mom_forecast = monthly_revenue_forecast['MoM_Growth'].iloc[0] if not monthly_revenue_forecast.empty else 0

        col_growth_kpi1, col_growth_kpi2 = st.columns(2)
        with col_growth_kpi1:
            st.markdown(
                f"""
                <div class="kpi-container kpi-container-historical">
                    <p class="kpi-title">Latest Historical MoM Growth</p>
                    <p class="kpi-value">{latest_mom_hist:,.2f}%</p>
                    <p class="kpi-subtitle">Based on latest full month</p>
                </div>
                """, unsafe_allow_html=True)
        with col_growth_kpi2:
            st.markdown(
                f"""
                <div class="kpi-container kpi-container-forecasted">
                    <p class="kpi-title">First Month Forecasted MoM Growth</p>
                    <p class="kpi-value">{latest_mom_forecast:,.2f}%</p>
                    <p class="kpi-subtitle">Based on first forecasted month</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ---- Main chart: Historical vs. Forecast ----
    st.markdown('<div id="daily-revenue"></div>', unsafe_allow_html=True)
    st.markdown("### Daily Revenue: Historical & Forecast")
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines',
        name='Historical Revenue',
        line=dict(color='#0b6ef6')
    ))
    fig_daily.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast[forecast_col],
        mode='lines',
        name='Forecasted Revenue',
        line=dict(color='#32CD32', dash='dot')
    ))
    fig_daily.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        fill='none',
        mode='lines',
        line=dict(color='rgba(11,110,246,0.1)', width=0),
        showlegend=False
    ))
    fig_daily.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(11,110,246,0.1)', width=0),
        name=f'{confidence_interval*100}% Confidence Interval'
    ))
    fig_daily.update_layout(
        title='Historical vs. Forecasted Daily Revenue',
        xaxis_title='Date',
        yaxis_title='Revenue ($)',
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    st.markdown("---")

    # ---- Cumulative Revenue Chart ----
    st.markdown('<div id="cumulative-revenue"></div>', unsafe_allow_html=True)
    st.markdown("### Cumulative Revenue: Historical & Forecast")
    combined_df['cumulative_y'] = combined_df['y'].cumsum()
    fig_cumul = go.Figure()
    fig_cumul.add_trace(go.Scatter(
        x=combined_df['ds'],
        y=combined_df['cumulative_y'],
        mode='lines',
        name='Cumulative Revenue',
        line=dict(color='#0b6ef6')
    ))
    fig_cumul.update_layout(
        title='Cumulative Revenue Over Time',
        xaxis_title='Date',
        yaxis_title='Cumulative Revenue ($)',
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
    )
    st.plotly_chart(fig_cumul, use_container_width=True)

    st.markdown("---")
    
    # ---- Forecast Data Table ----
    st.markdown('<div id="forecast-table"></div>', unsafe_allow_html=True)
    st.markdown("### Forecast Data Table")
    forecast_df_display = forecast_df.rename(
        columns={'ds': 'Date', 'yhat': 'Baseline Forecast', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound', 'yhat_what_if': 'What-if Forecast'}
    )
    st.dataframe(forecast_df_display[['Date', 'Baseline Forecast', 'What-if Forecast', 'Lower Bound', 'Upper Bound']], use_container_width=True)
    
    csv_export = forecast_df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Forecast Data as CSV",
        data=csv_export,
        file_name='financial_forecast.csv',
        mime='text/csv',
    )


# ---------------------- TAB 2: Model Performance ----------------------
with tab2:
    st.markdown('<div id="model-performance"></div>', unsafe_allow_html=True)
    st.markdown("### 🎯 Model Performance")
    
    st.warning("Note: Prophet doesn't provide built-in performance metrics like MAE or RMSE. This section focuses on a visual check of the model's fit on historical data.")
    
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='markers',
        name='Actual Revenue',
        marker=dict(color='#0b6ef6', size=4)
    ))
    fig_perf.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Model Trend',
        line=dict(color='red', width=2)
    ))
    fig_perf.update_layout(
        title='Model Fit on Historical Data',
        xaxis_title='Date',
        yaxis_title='Revenue ($)',
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
    )
    st.plotly_chart(fig_perf, use_container_width=True)
    
# ---------------------- TAB 3: Insights & Recommendations ----------------------
with tab3:
    st.markdown('<div id="insights--recommendations"></div>', unsafe_allow_html=True)
    st.markdown("### 💡 AI-Powered Insights")
    
    # Use LLM to generate a summary
    placeholder = st.empty()
    if st.button("Generate Insights"):
        with st.spinner("Generating insights..."):
            prompt = f"""
            Act as a world-class financial analyst. Based on the following forecast data, provide a concise, single-paragraph summary of the key findings, including:
            - A summary of the total and average revenue for historical vs. forecasted periods.
            - A comparison of the historical vs. forecasted CAGR.
            - An analysis of the impact of the 'What-if Scenario' on the forecast.
            - A clear recommendation based on the data.
            Historical Total Revenue: ${total_historical_revenue:,.2f}
            Historical Average Daily Revenue: ${avg_historical_revenue:,.2f}
            Historical CAGR: {cagr_hist:,.2%}
            
            Forecasted Total Revenue: ${total_forecasted_revenue:,.2f}
            Forecasted Average Daily Revenue: ${avg_forecasted_revenue:,.2f}
            Forecasted CAGR: {cagr_forecast:,.2%}
            What-if Scenario Change: {what_if_change:,.2f}%
            """
            
            # Use Gemini API to generate content
            try:
                import requests
                
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "tools": [{ "google_search": {} }],
                    "systemInstruction": {
                        "parts": [{ "text": "Act as a world-class financial analyst. Provide a concise, single-paragraph summary of the key findings." }]
                    },
                }

                # Assuming __API_KEY__ is a predefined variable
                apiKey = ""
                apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"
                
                response = requests.post(apiUrl, json=payload)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                
                result = response.json()
                insight_text = result['candidates'][0]['content']['parts'][0]['text']
                placeholder.markdown(f"**Analysis:** {insight_text}")
                
            except Exception as e:
                placeholder.error(f"Failed to generate insights: {e}")

# ---------------------- TAB 4: Deep Dive Analysis ----------------------
with tab4:
    st.markdown('<div id="deep-dive-analysis"></div>', unsafe_allow_html=True)
    st.markdown("### 🔍 Deep Dive: Understanding the Forecast")
    
    st.markdown("#### Time Series Components")
    st.markdown("This chart breaks down the forecast into its underlying components: trend, weekly seasonality, and yearly seasonality. This helps you understand the drivers of the forecast.")
    
    # Prophet's built-in components plot
    fig_components = plot_components_plotly(model, forecast)
    st.plotly_chart(fig_components, use_container_width=True)

    st.markdown("---")

    st.markdown("#### What-if Scenario vs. Baseline Forecast")
    st.markdown("This section allows you to directly compare the default baseline forecast against the revenue projections you've created with your 'What-if Scenario'.")
    
    forecast_comparison_df = forecast[['ds', 'yhat', 'yhat_what_if']].copy()
    forecast_comparison_df['yhat_what_if_delta'] = forecast_comparison_df['yhat_what_if'] - forecast_comparison_df['yhat']
    
    # Chart for comparison
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=forecast_comparison_df['ds'],
        y=forecast_comparison_df['yhat'],
        mode='lines',
        name='Baseline Forecast',
        line=dict(color='grey', dash='dot')
    ))
    fig_comp.add_trace(go.Scatter(
        x=forecast_comparison_df['ds'],
        y=forecast_comparison_df['yhat_what_if'],
        mode='lines',
        name=f'What-if Scenario ({what_if_change:+.2f}%)',
        line=dict(color='#0b6ef6')
    ))
    fig_comp.update_layout(
        title='Baseline vs. What-if Scenario Forecast',
        xaxis_title='Date',
        yaxis_title='Revenue ($)',
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("#### Detailed Forecast Comparison Table")
    st.dataframe(forecast_comparison_df.rename(columns={
        'ds': 'Date',
        'yhat': 'Baseline Forecast',
        'yhat_what_if': 'What-if Forecast',
        'yhat_what_if_delta': 'Difference'
    }), use_container_width=True)
