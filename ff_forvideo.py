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
    forecast['yhat_what_if'] = forecast['yhat'] * (1 + what_if_change / 100.0)

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
        latest_mom_hist = monthly_revenue_hist['MoM_Growth'].iloc[-1] if not monthly_revenue_hist['MoM_Growth'].empty else 0
        latest_yoy_hist = (historical_growth_df.groupby(historical_growth_df['ds'].dt.to_period('Y'))['y'].sum().pct_change().dropna().iloc[-1]*100) if len(historical_growth_df)>365 else 0
        latest_mom_forecast = monthly_revenue_forecast['MoM_Growth'].iloc[-1] if not monthly_revenue_forecast['MoM_Growth'].empty else 0
        latest_yoy_forecast = (monthly_revenue_forecast.groupby(monthly_revenue_forecast['month'].dt.to_period('Y'))['y'].sum().pct_change().dropna().iloc[-1]*100) if len(monthly_revenue_forecast)>0 else 0

        col7, col8 = st.columns(2)
        with col7:
            st.markdown(
                f"""
                <div class="kpi-container kpi-container-historical">
                    <p class="kpi-title">Latest Historical MoM Growth</p>
                    <p class="kpi-value">{latest_mom_hist:,.2f}%</p>
                </div>
                """, unsafe_allow_html=True
            )
        with col8:
            st.markdown(
                f"""
                <div class="kpi-container kpi-container-forecasted">
                    <p class="kpi-title">Latest Forecasted MoM Growth</p>
                    <p class="kpi-value">{latest_mom_forecast:,.2f}%</p>
                </div>
                """, unsafe_allow_html=True
            )

        col9, col10 = st.columns(2)
        with col9:
            st.markdown(
                f"""
                <div class="kpi-container kpi-container-historical">
                    <p class="kpi-title">Latest Historical YoY Growth</p>
                    <p class="kpi-value">{latest_yoy_hist:,.2f}%</p>
                </div>
                """, unsafe_allow_html=True
            )
        with col10:
            st.markdown(
                f"""
                <div class="kpi-container kpi-container-forecasted">
                    <p class="kpi-title">Latest Forecasted YoY Growth</p>
                    <p class="kpi-value">{latest_yoy_forecast:,.2f}%</p>
                </div>
                """, unsafe_allow_html=True
            )

        # Growth chart (monthly) with arrows showing up/down points (use small annotation markers)
        growth_fig = go.Figure()
        if not monthly_revenue_hist.empty:
            growth_fig.add_trace(go.Scatter(
                x=monthly_revenue_hist['month'],
                y=monthly_revenue_hist['y'],
                mode='lines+markers',
                name='Historical Monthly Revenue',
                line=dict(color='#0b6ef6', width=2),
                hovertemplate='Month: %{x|%b %Y}<br>Revenue: %{y:$,.2f}<extra></extra>'
            ))
            # add arrow markers for directions
            for i, row in monthly_revenue_hist.iterrows():
                if i==0: continue
                if row['direction']=='up':
                    growth_fig.add_annotation(x=row['month'], y=row['y'], ax=0, ay=-30, showarrow=True, arrowhead=3, arrowsize=1, arrowcolor="#1f9d55", text="")
                elif row['direction']=='down':
                    growth_fig.add_annotation(x=row['month'], y=row['y'], ax=0, ay=30, showarrow=True, arrowhead=3, arrowsize=1, arrowcolor="#d45a5a", text="")

        if not monthly_revenue_forecast.empty:
            growth_fig.add_trace(go.Scatter(
                x=monthly_revenue_forecast['month'],
                y=monthly_revenue_forecast['y'],
                mode='lines+markers',
                name='Forecast Monthly Revenue',
                line=dict(color='#16a34a', width=2, dash='dot'),
                hovertemplate='Month: %{x|%b %Y}<br>Forecast: %{y:$,.2f}<extra></extra>'
            ))
            # forecast arrows
            for i, row in monthly_revenue_forecast.iterrows():
                if i==0: continue
                if row['direction']=='up':
                    growth_fig.add_annotation(x=row['month'], y=row['y'], ax=0, ay=-30, showarrow=True, arrowhead=3, arrowsize=1, arrowcolor="#1f9d55", text="")
                elif row['direction']=='down':
                    growth_fig.add_annotation(x=row['month'], y=row['y'], ax=0, ay=30, showarrow=True, arrowhead=3, arrowsize=1, arrowcolor="#d45a5a", text="")

        growth_fig.update_layout(title="Monthly Revenue (Historical vs Forecast) with Directional Arrows",
                                 xaxis_title="Month", yaxis_title="Revenue ($)", template="plotly_white", hovermode="x unified",
                                 transition_duration=600)
        st.plotly_chart(growth_fig, use_container_width=True)

    st.markdown("---")

    # ---- Historical Revenue & 30-Day Moving Average (unchanged but restyled) ----
    st.markdown('<div id="historical-trends"></div>', unsafe_allow_html=True)
    st.subheader("Historical Revenue & 30-Day Moving Average")
    df['30_day_avg'] = df['y'].rolling(window=30, min_periods=1).mean()

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=df['ds'], y=df['y'],
        mode='lines',
        name='Historical Daily Revenue',
        line=dict(color='rgba(11,110,246,0.35)', width=1),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Revenue:</b> %{y:$,.2f}<extra></extra>'
    ))
    fig_hist.add_trace(go.Scatter(
        x=df['ds'], y=df['30_day_avg'],
        mode='lines',
        name='30-Day Moving Avg',
        line=dict(color='#1f9d55', width=3),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%m}<br><b>30-Day Avg:</b> %{y:$,.2f}<extra></extra>'
    ))
    fig_hist.update_layout(title="Historical Revenue and Moving Average",
                           xaxis_title="Date", yaxis_title="Revenue (in thousands of $)",
                           yaxis=dict(tickprefix="$"), template="plotly_white", hovermode="x unified",
                           xaxis_rangeslider_visible=True, transition_duration=500)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # ---- Daily Revenue Chart (Combined historical + forecast) ----
    st.markdown('<div id="daily-revenue"></div>', unsafe_allow_html=True)
    st.subheader("Daily Revenue Forecast and Historical Trend")

    col_dr1, col_dr2 = st.columns(2)
    with col_dr1:
        start_date_daily = st.date_input(
            "Start Date (Daily Chart):",
            value=df['ds'].min().date(),
            min_value=df['ds'].min().date(),
            max_value=forecast['ds'].max().date(),
            key='daily_start'
        )
    with col_dr2:
        end_date_daily = st.date_input(
            "End Date (Daily Chart):",
            value=forecast['ds'].max().date(),
            min_value=df['ds'].min().date(),
            max_value=forecast['ds'].max().date(),
            key='daily_end'
        )

    combined_df_daily = combined_df[
        (combined_df['ds'].dt.date >= start_date_daily) &
        (combined_df['ds'].dt.date <= end_date_daily)
    ].copy()

    # 30-day rolling on combined (helps show moving avg across forecast)
    combined_df['30_day_avg'] = combined_df['y'].rolling(window=30, min_periods=1).mean()

    fig_daily = go.Figure()

    hist_filtered = combined_df_daily[combined_df_daily['type'] == 'Historical']
    forecast_filtered = combined_df_daily[combined_df_daily['type'] == 'Forecast']

    if not hist_filtered.empty:
        fig_daily.add_trace(go.Scatter(
            x=hist_filtered['ds'], y=hist_filtered['y'],
            mode='lines',
            name='Historical Daily Revenue',
            line=dict(color='rgba(11,110,246,0.35)', width=1),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Revenue:</b> %{y:$,.2f}<extra></extra>'
        ))
        fig_daily.add_trace(go.Scatter(
            x=hist_filtered['ds'], y=combined_df.loc[hist_filtered.index, '30_day_avg'],
            mode='lines',
            name='Historical 30-Day Moving Avg',
            line=dict(color='#1f9d55', width=3),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>30-Day Avg:</b> %{y:$,.2f}<extra></extra>'
        ))

    if not forecast_filtered.empty:
        fig_daily.add_trace(go.Scatter(
            x=forecast_filtered['ds'], y=forecast_filtered['y'],
            mode='lines',
            name='Forecasted Daily Revenue',
            line=dict(color='rgba(22,163,74,0.6)', width=1, dash='dot'),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Forecasted:</b> %{y:$,.2f}<extra></extra>'
        ))
        fig_daily.add_trace(go.Scatter(
            x=forecast_filtered['ds'], y=combined_df.loc[forecast_filtered.index, '30_day_avg'],
            mode='lines',
            name='Forecasted 30-Day Moving Avg',
            line=dict(color='purple', width=3, dash='dash'),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Forecast 30-Day Avg:</b> %{y:$,.2f}<extra></extra>'
        ))

    # Confidence interval shading for forecast within selected range
    forecast_filtered_bounds = forecast[
        (forecast['ds'].dt.date >= start_date_daily) &
        (forecast['ds'].dt.date <= end_date_daily) &
        (forecast['ds'] > df['ds'].max())
    ]
    if not forecast_filtered_bounds.empty:
        fig_daily.add_trace(go.Scatter(
            x=list(forecast_filtered_bounds['ds']) + list(forecast_filtered_bounds['ds'])[::-1],
            y=list(forecast_filtered_bounds['yhat_upper']) + list(forecast_filtered_bounds['yhat_lower'])[::-1],
            fill='toself',
            fillcolor='rgba(22,163,74,0.08)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name=f"{int(confidence_interval*100)}% Confidence Interval"
        ))

    # Mark forecast start
    start_of_forecast = df['ds'].max()
    if start_of_forecast >= pd.to_datetime(start_date_daily) and start_of_forecast <= pd.to_datetime(end_date_daily):
        fig_daily.add_vline(x=start_of_forecast, line_width=1, line_dash="dash", line_color="#ef4444")
        fig_daily.add_annotation(
            x=start_of_forecast,
            y=1.02,
            xref="x",
            yref="paper",
            text="Forecast begins here",
            showarrow=False,
            font=dict(color="#ef4444")
        )

    fig_daily.update_layout(
        title="Daily Revenue: Historical vs. Forecasted",
        xaxis_title="Date",
        yaxis_title="Revenue (in thousands of $)",
        yaxis=dict(tickprefix="$"),
        template="plotly_white",
        hovermode="x unified",
        xaxis_rangeslider_visible=True,
        transition_duration=500,
        xaxis=dict(range=[start_date_daily, end_date_daily])
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    st.markdown("---")

    # ---- Cumulative Revenue chart (unchanged logic, restyled) ----
    st.markdown('<div id="cumulative-revenue"></div>', unsafe_allow_html=True)
    st.subheader("📈 Cumulative Revenue Trend")

    col_cr1, col_cr2 = st.columns(2)
    with col_cr1:
        start_date_cumulative = st.date_input(
            "Start Date (Cumulative Chart):",
            value=df['ds'].min().date(),
            min_value=df['ds'].min().date(),
            max_value=forecast['ds'].max().date(),
            key='cumulative_start'
        )
    with col_cr2:
        end_date_cumulative = st.date_input(
            "End Date (Cumulative Chart):",
            value=forecast['ds'].max().date(),
            min_value=df['ds'].min().date(),
            max_value=forecast['ds'].max().date(),
            key='cumulative_end'
        )

    combined_df['cumulative_revenue'] = combined_df['y'].cumsum()

    cumulative_filtered = combined_df[
        (combined_df['ds'].dt.date >= start_date_cumulative) &
        (combined_df['ds'].dt.date <= end_date_cumulative)
    ].copy()

    fig_cumulative = go.Figure()
    historical_cumulative_filtered = cumulative_filtered[cumulative_filtered['type'] == 'Historical']
    if not historical_cumulative_filtered.empty:
        fig_cumulative.add_trace(go.Scatter(
            x=historical_cumulative_filtered['ds'],
            y=historical_cumulative_filtered['cumulative_revenue'],
            mode='lines',
            name='Historical Revenue',
            line=dict(color='#0b6ef6', width=3)
        ))

    forecasted_cumulative_filtered = cumulative_filtered[cumulative_filtered['type'] == 'Forecast']
    if not forecasted_cumulative_filtered.empty:
        last_historical_cum_sum = 0
        if not combined_df[combined_df['type'] == 'Historical'].empty:
            last_historical_cum_sum = combined_df[combined_df['type'] == 'Historical']['cumulative_revenue'].iloc[-1]
        forecasted_cumulative_filtered['cumulative_revenue_adjusted'] = forecasted_cumulative_filtered['y'].cumsum() + last_historical_cum_sum
        fig_cumulative.add_trace(go.Scatter(
            x=forecasted_cumulative_filtered['ds'],
            y=forecasted_cumulative_filtered['cumulative_revenue_adjusted'],
            mode='lines',
            name='Forecasted Revenue',
            line=dict(color='orange', width=3, dash='dash')
        ))

    # vertical line for forecast start
    start_of_forecast = df['ds'].max()
    if start_of_forecast >= pd.to_datetime(start_date_cumulative) and start_of_forecast <= pd.to_datetime(end_date_cumulative):
        fig_cumulative.add_vline(x=start_of_forecast, line_width=1, line_dash="dash", line_color="#ef4444")
        fig_cumulative.add_annotation(
            x=start_of_forecast,
            y=1.02,
            xref="x",
            yref="paper",
            text="Forecast begins here",
            showarrow=False,
            font=dict(color="#ef4444")
        )

    fig_cumulative.update_layout(
        title="Cumulative Revenue Over Time: Historical vs. Forecasted",
        xaxis_title="Date",
        yaxis_title="Cumulative Revenue (in thousands of $)",
        yaxis=dict(tickprefix="$"),
        template="plotly_white",
        hovermode="x unified",
        transition_duration=500,
        xaxis=dict(range=[start_date_cumulative, end_date_cumulative])
    )
    st.plotly_chart(fig_cumulative, use_container_width=True)

    st.markdown("---")

    # ---- Forecast Table + Download (styled) ----
    st.markdown('<div id="forecast-table"></div>', unsafe_allow_html=True)
    st.subheader(f"🧾 {forecast_months}-Month Forecast Table")
    display_table = forecast[['ds', forecast_col, 'yhat_lower', 'yhat_upper']].tail(forecast_period_days).rename(
        columns={"ds": "Date", forecast_col: "Predicted Revenue", "yhat_lower": "Lower Bound", "yhat_upper": "Upper Bound"}
    )
    st.dataframe(display_table, use_container_width=True, height=300)
    csv = display_table.to_csv(index=False)
    st.download_button(f"⬇️ Download {forecast_months}-Month Forecast CSV", csv, f"forecast_{forecast_months}_months.csv", "text/csv")

# ---------------------- TAB 2: Model Performance ----------------------
with tab2:
    st.markdown('<div id="model-performance"></div>', unsafe_allow_html=True)
    st.subheader("📊 Model Performance")

    historical_comparison = pd.merge(df, forecast, on='ds', how='inner')
    if not historical_comparison.empty:
        mae = np.mean(np.abs(historical_comparison['y'] - historical_comparison['yhat']))
        rmse = np.sqrt(np.mean((historical_comparison['y'] - historical_comparison['yhat'])**2))
        wape = np.sum(np.abs(historical_comparison['y'] - historical_comparison['yhat'])) / np.sum(np.abs(historical_comparison['y'])) * 100 if np.sum(np.abs(historical_comparison['y'])) != 0 else 0.0
        forecast_bias = np.mean(historical_comparison['yhat'] - historical_comparison['y'])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="kpi-container">
                    <p class="kpi-title">Mean Absolute Error (MAE)</p>
                    <p class="kpi-value">${mae:,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div class="kpi-container">
                    <p class="kpi-title">Root Mean Squared Error (RMSE)</p>
                    <p class="kpi-value">${rmse:,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f"""
                <div class="kpi-container">
                    <p class="kpi-title">WAPE</p>
                    <p class="kpi-value">{wape:,.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                f"""
                <div class="kpi-container">
                    <p class="kpi-title">Forecast Bias</p>
                    <p class="kpi-value">${forecast_bias:,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("""
        **What are these metrics?**

        * **Mean Absolute Error (MAE)**: The average dollar amount the forecast was off by.
        * **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily, useful for spotting major misses.
        * **WAPE (Weighted Absolute Percentage Error)**: Provides a single percentage for overall accuracy, making it easy to interpret.
        * **Forecast Bias**: A positive value means the model is consistently over-forecasting, while a negative value indicates under-forecasting.
        """)
    else:
        st.info("Not enough overlapping historical/prediction data to compute performance metrics.")

    st.markdown('<div id="time-series-components"></div>', unsafe_allow_html=True)
    st.subheader("📉 Time Series Components")
    try:
        components_fig = plot_components_plotly(model, forecast)
        # set y-axis prefixes where possible
        try:
            components_fig.update_yaxes(title_text='Revenue (in thousands of $)', tickprefix='$')
        except Exception:
            pass
        st.plotly_chart(components_fig, use_container_width=True)
    except Exception as e:
        st.info("Could not render components plot: " + str(e))

# ---------------------- TAB 3: Insights & Recommendations ----------------------
with tab3:
    st.markdown('<div id="insights--recommendations"></div>', unsafe_allow_html=True)
    st.subheader("📚 Insights & Recommendations")

    # compute summary KPIs (re-use values computed earlier)
    hist_total = total_historical_revenue
    fore_total = total_forecasted_revenue
    change_pct = total_revenue_delta
    hist_cagr = cagr_hist
    fore_cagr = cagr_forecast

    trend_icon = "📈" if change_pct > 0 else ("📉" if change_pct < 0 else "➡️")
    st.markdown(f"### {trend_icon} Summary")
    st.markdown(
        f"- **Total Historical Revenue:** ${hist_total:,.0f}\n"
        f"- **Total Forecasted Revenue ({forecast_months} mo):** ${fore_total:,.0f}\n"
        f"- **Change vs Historical:** {change_pct:.2f}%\n"
        f"- **Historical CAGR:** {hist_cagr:.2%}\n"
        f"- **Forecasted CAGR:** {fore_cagr:.2%}"
    )

    st.markdown("### Recommendations")
    if change_pct > 5:
        st.success("Forecast indicates meaningful positive growth — consider increasing investment in growth initiatives, hiring for capacity, and preparing inventory/operations for higher demand.")
    elif change_pct < -5:
        st.error("Forecast suggests a contraction. Investigate drivers (seasonality, recent dips) and consider cost optimization, marketing to stabilize revenue, or hedging cashflow exposure.")
    else:
        st.info("Forecast is relatively stable. Maintain current strategy but monitor leading indicators and re-run forecasts frequently if new data arrives.")

    st.markdown("---")
    st.markdown("### Actionable next steps")
    st.markdown(
        """
        - Align budgets and headcount planning with forecast peaks.
        - Investigate anomalies flagged in historical data (e.g., product launches, supply dips).
        - Re-run forecasts monthly and after major events (promotions, product launches).
        """
    )
    
# ---------------------- TAB 4: Deep Dive Analysis ----------------------
with tab4:
    st.subheader("📈 Growth & Trend Insights")

    # --- Revenue Momentum & Acceleration ---
    st.markdown("#### Revenue Momentum & Acceleration")
    
    # Calculate 7-day growth (based on last value vs previous value)
    if len(df) > 7:
        growth_7d = df['y'].tail(7).pct_change().iloc[-1] * 100
    else:
        growth_7d = 0.0
    
    # Calculate 30-day growth (based on last 30 days vs previous 30 days)
    if len(df) > 31:
        prev_30 = df['y'].iloc[-61:-31].sum()  # previous 30-day window
        curr_30 = df['y'].tail(30).sum()
        growth_30d = ((curr_30 / prev_30) - 1) * 100 if prev_30 > 0 else 0.0
    else:
        growth_30d = 0.0
    
    # Calculate 90-day growth (based on last 90 days vs previous 90 days)
    if len(df) > 181:
        prev_90 = df['y'].iloc[-181:-91].sum()  # previous 90-day window
        curr_90 = df['y'].tail(90).sum()
        growth_90d = ((curr_90 / prev_90) - 1) * 100 if prev_90 > 0 else 0.0
    else:
        growth_90d = 0.0
    
    # Display KPIs
    col_mom1, col_mom2, col_mom3 = st.columns(3)
    
    with col_mom1:
        st.markdown(
            f"""
            <div class="kpi-container">
                <p class="kpi-title">Latest 7-Day Growth</p>
                <p class="kpi-value" style="color:{'green' if growth_7d > 0 else 'red'};">
                    {growth_7d:,.2f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col_mom2:
        st.markdown(
            f"""
            <div class="kpi-container">
                <p class="kpi-title">Latest 30-Day Growth</p>
                <p class="kpi-value" style="color:{'green' if growth_30d > 0 else 'red'};">
                    {growth_30d:,.2f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col_mom3:
        st.markdown(
            f"""
            <div class="kpi-container">
                <p class="kpi-title">Latest 90-Day Growth</p>
                <p class="kpi-value" style="color:{'green' if growth_90d > 0 else 'red'};">
                    {growth_90d:,.2f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    

    # --- Revenue Recovery Analysis ---
    st.markdown("#### Revenue Recovery Analysis")
    supply_chain_dip_date = pd.to_datetime('2023-11-20')
    if supply_chain_dip_date in df['ds'].values:
        dip_start_date = df[df['ds'] == supply_chain_dip_date].iloc[0].ds
        dip_revenue = df[df['ds'] == dip_start_date].iloc[0]['y']
        
        pre_dip_avg = df[(df['ds'] < dip_start_date) & (df['ds'] > dip_start_date - pd.Timedelta(days=30))]['y'].mean()
        
        recovery_df = df[df['ds'] > dip_start_date].copy()
        
        recovery_period = recovery_df[recovery_df['y'] >= pre_dip_avg]
        if not recovery_period.empty:
            recovery_date = recovery_period.iloc[0].ds
            days_to_recover = (recovery_date - dip_start_date).days
            st.info(f"The **'Supply Chain Dip'** on {dip_start_date.strftime('%Y-%m-%d')} caused a drop in revenue. It took approximately **{days_to_recover} days** to recover to the pre-dip average.")
        else:
            st.info("The model could not identify a full recovery from the 'Supply Chain Dip' yet.")
    
    st.markdown("---")
    
    # --- Seasonality & Pattern Insights ---
    st.subheader("📊 Seasonality & Pattern Insights")
    df['day_of_week'] = df['ds'].dt.day_name()
    df['month_of_year'] = df['ds'].dt.month_name()
    
    avg_by_day = df.groupby('day_of_week')['y'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    avg_by_month = df.groupby('month_of_year')['y'].mean().reindex(pd.to_datetime(range(1, 13), format='%m').month_name())
    
    col_season1, col_season2 = st.columns(2)
    with col_season1:
        fig_day = go.Figure(data=[go.Bar(x=avg_by_day.index, y=avg_by_day.values, marker_color='#0b6ef6')])
        fig_day.update_layout(title="Average Revenue by Day of Week", yaxis_title="Average Revenue ($)", xaxis_title="")
        st.plotly_chart(fig_day, use_container_width=True)
    
    with col_season2:
        fig_month = go.Figure(data=[go.Bar(x=avg_by_month.index, y=avg_by_month.values, marker_color='#1f9d55')])
        fig_month.update_layout(title="Average Revenue by Month of Year", yaxis_title="Average Revenue ($)", xaxis_title="")
        st.plotly_chart(fig_month, use_container_width=True)
        
    st.markdown("---")
    
    # --- Seasonal Strength Index ---
    st.markdown("#### Seasonal Strength Index")
    model_components = forecast[['trend', 'yearly', 'weekly']]
    model_residuals = df['y'] - forecast['yhat'].iloc[:len(df)]
    
    trend_variance = model_components['trend'].var()
    seasonal_variance = model_components[['yearly', 'weekly']].var().sum()
    noise_variance = model_residuals.var()
    
    total_variance = trend_variance + seasonal_variance + noise_variance
    
    trend_pct = (trend_variance / total_variance) * 100
    seasonal_pct = (seasonal_variance / total_variance) * 100
    noise_pct = (noise_variance / total_variance) * 100
    
    seasonal_strength_index = (seasonal_variance / (seasonal_variance + trend_variance)) * 100 if (seasonal_variance + trend_variance) > 0 else 0
    
    col_ss1, col_ss2 = st.columns(2)
    with col_ss1:
        st.metric("Seasonal Strength Index", f"{seasonal_strength_index:.2f}%")
    with col_ss2:
        st.markdown(f"""
        <div class="kpi-container">
            <p class="kpi-title">Holiday Impact</p>
            <p class="kpi-value">${df[df['ds'].isin(holidays_df['ds'])]['y'].mean():,.2f}</p>
            <p class="kpi-subtitle">Average holiday revenue</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f"**Insight:** The model suggests that **{seasonal_pct:.2f}%** of the observed revenue variability is explained by predictable seasonal patterns, while **{trend_pct:.2f}%** is explained by the long-term trend, and the remaining **{noise_pct:.2f}%** is random noise.")
    st.markdown("---")
    
    # --- Risk & Volatility Insights ---
    st.subheader("📉 Risk & Volatility Insights")
    df['rolling_std'] = df['y'].rolling(window=30, min_periods=1).std()
    df['rolling_mean'] = df['y'].rolling(window=30, min_periods=1).mean()
    df['volatility_index'] = df['rolling_std'] / df['rolling_mean']
    
    fig_volatility = go.Figure(data=[go.Scatter(x=df['ds'], y=df['volatility_index'], mode='lines', name='Revenue Volatility Index', line=dict(color='#d45a5a'))])
    fig_volatility.update_layout(title="Revenue Volatility Index (30-day Rolling)", yaxis_title="Index", xaxis_title="Date", template="plotly_white")
    st.plotly_chart(fig_volatility, use_container_width=True)
    
    # Revenue Drawdowns
    df['cumulative_max'] = df['y'].cummax()
    df['drawdown'] = (df['y'] - df['cumulative_max']) / df['cumulative_max'] * 100
    max_drawdown = df['drawdown'].min()
    
    fig_drawdown = go.Figure(data=[go.Scatter(x=df['ds'], y=df['drawdown'], mode='lines', name='Revenue Drawdown', line=dict(color='orange'))])
    fig_drawdown.update_layout(title="Revenue Drawdown from Peak (%)", yaxis_title="Drawdown (%)", xaxis_title="Date", template="plotly_white")
    st.plotly_chart(fig_drawdown, use_container_width=True)
    st.info(f"The largest revenue drawdown was **{max_drawdown:.2f}%**, indicating the maximum peak-to-trough decline experienced in the historical data.")
    
    st.markdown("---")
    
    # Anomaly Detection
    historical_comparison['residuals'] = historical_comparison['y'] - historical_comparison['yhat']
    residuals_std = historical_comparison['residuals'].std()
    historical_comparison['z_score'] = historical_comparison['residuals'] / residuals_std
    anomalies = historical_comparison[abs(historical_comparison['z_score']) > 3] # Flagging anything > 3 std deviations
    
    if not anomalies.empty:
        st.markdown("#### Anomaly Detection")
        fig_anomalies = go.Figure()
        fig_anomalies.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Historical Revenue', line=dict(color='#0b6ef6')))
        fig_anomalies.add_trace(go.Scatter(x=anomalies['ds'], y=anomalies['y'], mode='markers', name='Anomalies', marker=dict(color='red', size=8)))
        fig_anomalies.update_layout(title="Historical Revenue with Detected Anomalies", yaxis_title="Revenue ($)")
        st.plotly_chart(fig_anomalies, use_container_width=True)
    
    st.markdown("---")
    
    # --- Financial KPI Insights ---
    st.subheader("🧾 Financial KPI Insights")
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        # Revenue Run-Rate
        last_90_days_avg_revenue = df['y'].tail(90).mean()
        annual_run_rate = last_90_days_avg_revenue * 365
        st.markdown(
            f"""
            <div class="kpi-container">
                <p class="kpi-title">Annual Run-Rate (ARR)</p>
                <p class="kpi-value">${annual_run_rate/1000000:,.2f}M</p>
                <p class="kpi-subtitle">Based on last 90 days</p>
            </div>
            """, unsafe_allow_html=True
        )
    
    with col_kpi2:
        # Growth Target Tracking
        st.markdown("<p class='kpi-title'>Growth Target Tracking</p>", unsafe_allow_html=True)
        growth_target_pct = st.number_input("Annual Growth Target (%)", value=15.0, step=1.0, format="%.1f")
        latest_yoy_growth = df['y'].iloc[-365:].sum() / df['y'].iloc[-730:-365].sum() - 1 if len(df) > 730 else 0
        
        target_status = "On Track" if latest_yoy_growth * 100 >= growth_target_pct else "Off Target"
        status_color = "#1f9d55" if target_status == "On Track" else "#d45a5a"
        
        st.markdown(f"""
        <div class="kpi-container">
            <p class="kpi-title">Actual YoY Growth</p>
            <p class="kpi-value" style="color:{status_color};">{latest_yoy_growth*100:,.2f}%</p>
            <p class="kpi-subtitle">Target: {growth_target_pct:.1f}% ({target_status})</p>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi3:
        st.markdown(f"""
        <div class="kpi-container">
            <p class="kpi-title">Revenue Milestones</p>
            <ul>
                <li>$1M: {df[df['y'].cumsum() >= 1000000]['ds'].min().strftime('%Y-%m-%d') if not df[df['y'].cumsum() >= 1000000].empty else "Not yet reached"}</li>
                <li>$5M: {df[df['y'].cumsum() >= 5000000]['ds'].min().strftime('%Y-%m-%d') if not df[df['y'].cumsum() >= 5000000].empty else "Not yet reached"}</li>
                <li>$10M: {df[df['y'].cumsum() >= 10000000]['ds'].min().strftime('%Y-%m-%d') if not df[df['y'].cumsum() >= 10000000].empty else "Not yet reached"}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Revenue Concentration by Period
    st.markdown("#### Revenue Concentration by Period")
    df_quarterly = df.copy()
    df_quarterly['year'] = df_quarterly['ds'].dt.year
    df_quarterly['quarter'] = df_quarterly['ds'].dt.quarter
    
    quarterly_rev = df_quarterly.groupby(['year', 'quarter'])['y'].sum().reset_index()
    quarterly_rev['quarter_str'] = 'Q' + quarterly_rev['quarter'].astype(str)
    
    fig_concentration = go.Figure()
    for year in quarterly_rev['year'].unique():
        year_data = quarterly_rev[quarterly_rev['year'] == year]
        total_year_rev = year_data['y'].sum()
        year_data['pct'] = year_data['y'] / total_year_rev
        fig_concentration.add_trace(go.Bar(
            x=year_data['quarter_str'],
            y=year_data['pct'],
            name=str(year),
            hovertemplate='Year: %{name}<br>Quarter: %{x}<br>Percentage: %{y:.1%}<extra></extra>'
        ))
    
    fig_concentration.update_layout(
        title="Revenue Concentration by Quarter",
        barmode='stack',
        yaxis=dict(tickformat=".0%"),
        yaxis_title="Percentage of Annual Revenue",
        xaxis_title="Quarter"
    )
    st.plotly_chart(fig_concentration, use_container_width=True)

# --- Centered Watermark (updated text as requested) ---
st.markdown('<p class="watermark">Created by Miracle Software Systems for AI for Business</p>', unsafe_allow_html=True)

# --- Floating Chat (GLOBAL OVERLAY via Shadow DOM) ---
# Keep overlay JS as you had it; inject base64 icon
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
  var hostDiv = hostDoc.createElement("div");
  hostDiv.id = ROOT_ID;
  hostDoc.body.appendChild(hostDiv);

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
  + '.mira-typing{align-self:flex-start;display:inline-flex;gap:6px;padding:8px 10px;border-radius:12px;border:1px solid #e5f7eb;background:#f5f7fb;color:#6b7280;font-size:14px;}'
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
  var API_URL = "https://mira-proxy-331663702828.us-central1.run.app/";

  var modal    = shadow.getElementById("miraModal");
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
      if (!summary && (!rows || !drows.length)) bubble("bot", "No summary or data returned.");
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

# inject the icon data
overlay_html = overlay_html.replace("__ICON_B64__", _CHAT_ICON_B64)

# Render overlay
components.html(overlay_html, height=0, scrolling=False)
