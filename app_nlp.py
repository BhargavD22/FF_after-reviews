# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import numpy as np
from prophet.plot import plot_components_plotly
import base64
import os
import re

# --- CONFIGURATION ---
LOGO_PATH = "miracle-logo-dark.png"

# Set Streamlit page config for wide layout
st.set_page_config(
    layout="wide",
    page_title="Financial Forecasting",
    initial_sidebar_state="expanded",
)

# === Streamlit App UI === #

# --- Custom CSS for Styling ---
def get_image_base64(path):
    """Encodes an image to a base64 string."""
    if os.path.exists(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return ""

encoded_string = get_image_base64(LOGO_PATH)

st.markdown(
    f"""
    <style>
        /* Enable smooth scrolling for bookmarks */
        html {{
            scroll-behavior: smooth;
        }}
        
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, .stApp {{
            font-family: 'Inter', sans-serif;
        }}
        
        /* Apply custom theme and background for light mode */
        .stApp {{
            background-color: #f0f2f6;
            color: #333333;
        }}

        /* Style for the main container */
        .st-emotion-cache-1r4qj8m {{
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }}

        /* Style for headers */
        h1, h2, h3, h4, h5, h6 {{
            color: #007bff; /* A contrasting color for light mode */
        }}

        /* Style for the slider */
        .stSlider .st-emotion-cache-6q9m8y e16fv1ov3 {{
            background-color: #007bff;
        }}
        
        /* Style the tabs */
        .stTabs [role="tablist"] button {{
            background-color: #ffffff;
            color: #333333;
            border-bottom: 3px solid transparent;
        }}
        .stTabs [role="tablist"] button[aria-selected="true"] {{
            color: #007bff;
            border-bottom: 3px solid #007bff;
        }}
        
        /* Style for the dataframe */
        .dataframe {{
            border-radius: 8px;
        }}
        
        /* Style for the download button */
        .stDownloadButton button {{
            background-color: #007bff;
            color: #ffffff;
            font-weight: bold;
            border-radius: 8px;
        }}
        .stDownloadButton button:hover {{
            background-color: #0056b3;
            color: #ffffff;
        }}

        /* New KPI card styles (Power BI inspired) */
        .kpi-container {{
            display: flex;
            flex-direction: column;
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease-in-out;
            margin-bottom: 1rem;
            height: 100%; /* Ensure all cards in a row have the same height */
        }}
        .kpi-container:hover {{
            transform: translateY(-5px);
        }}
        .kpi-title {{
            font-size: 1rem;
            color: #666;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        .kpi-value {{
            font-size: 2.2rem;
            font-weight: 700;
            color: #007bff;
            margin: 0.2rem 0;
        }}
        .kpi-subtitle {{
            font-size: 0.875rem;
            color: #888;
            margin-top: 0;
            font-weight: 400;
        }}
        .kpi-delta {{
            font-size: 1rem;
            font-weight: 600;
            margin-top: 0.75rem;
            display: flex;
            align-items: center;
        }}
        .positive-delta {{
            color: #28a745; /* Green */
        }}
        .negative-delta {{
            color: #dc3545; /* Red */
        }}
        .delta-icon {{
            margin-right: 0.5rem;
        }}
        .st-emotion-cache-1r4qj8m {{
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }}
        
        /* CORRECTED STYLES for date selectors */
        /* This targets Streamlit's internal container classes */
        div[data-testid="stHorizontalBlock"] > div.date-selector-container,
        .date-selector-container {{
            background-color: #f7f9fc;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e6e6e6;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
        }}
        .date-selector-container .stDateInput {{
            border-color: #ccc;
            border-radius: 8px;
        }}
        .date-selector-container label {{
            font-weight: 600;
            color: #555;
        }}

        /* Watermark style */
        .watermark {{
            position: fixed;
            bottom: 10px;
            right: 10px;
            font-size: 12px;
            color: rgba(0, 0, 0, 0.1);
            pointer-events: none;
            z-index: 9999;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Main App Title and Description ---
st.title("📈 Financial Forecasting Dashboard")
st.markdown("A **dynamic** application to analyze historical revenue data and forecast future trends using the **Prophet** model.")

with st.sidebar:
    # Add logo to the sidebar
    if encoded_string:
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.error(f"Logo file not found at {LOGO_PATH}")
    
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="The file must contain 'ds' (date) and 'y' (revenue) columns.")
    st.markdown("---")
    
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
    
    # --- Bookmarks Section ---
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
        [Time Series Components](#time-series-components)
        [Chatbot](#chatbot)
        """,
        unsafe_allow_html=True
    )
    
# --- Main Content Area ---
st.header("Data & Analysis")

# Main application logic runs only after a file is uploaded
if uploaded_file is not None:
    # Read the data from the uploaded file
    try:
        df = pd.read_csv(uploaded_file)
        # Ensure the 'ds' column is in datetime format and 'y' is numeric
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'])
    except Exception as e:
        st.error(f"Error reading the file. Please ensure it's a valid CSV with 'ds' and 'y' columns. Error: {e}")
        st.stop()

    # Create main content tabs
    tab1, tab2 = st.tabs(["📊 Forecast", "📈 Model Performance"])

    with tab1:
        # Define the holidays for Prophet to learn from the anomalies
        holidays_df = pd.DataFrame([
            {'holiday': 'Product Launch Spike', 'ds': pd.to_datetime('2022-07-15'), 'lower_window': -5, 'upper_window': 5},
            {'holiday': 'Supply Chain Dip', 'ds': pd.to_datetime('2023-11-20'), 'lower_window': -5, 'upper_window': 5},
        ])

        # --- FIX: Add a lower bound to prevent negative forecasts ---
        df['floor'] = 0

        # Fit Prophet model with user-defined seasonality and holidays
        model = Prophet(weekly_seasonality=weekly_seasonality, yearly_seasonality=yearly_seasonality, holidays=holidays_df, growth='linear')
        model.fit(df)

        # Make forecast with user-defined confidence interval
        future = model.make_future_dataframe(periods=forecast_period_days)
        
        # --- FIX: Add the floor to the future dataframe ---
        future['floor'] = 0

        forecast = model.predict(future)
        
        # --- Convert 'ds' column to datetime to avoid TypeError ---
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        
        # --- Apply what-if scenario to the forecast ---
        forecast['yhat_what_if'] = forecast['yhat'] * (1 + what_if_change / 100)
        
        # --- Combine historical and forecast data for unified plotting ---
        # Select the correct forecast column based on the checkbox
        if what_if_enabled:
            forecast_col = 'yhat_what_if'
        else:
            forecast_col = 'yhat'

        combined_df = pd.concat([
            df[['ds', 'y']].assign(type='Historical').set_index('ds'),
            forecast.rename(columns={forecast_col: 'y'})[['ds', 'y']].assign(type='Forecast').set_index('ds')
        ]).reset_index()
        
        # --- Calculate KPIs for comparison ---
        
        # Historical KPIs
        total_historical_revenue = df['y'].sum()
        avg_historical_revenue = df['y'].mean()
        
        # Forecasted KPIs (using the correct column based on the toggle)
        forecast_df = forecast[forecast['ds'] > df['ds'].max()]
        total_forecasted_revenue = forecast_df[forecast_col].sum()
        avg_forecasted_revenue = forecast_df[forecast_col].mean()
        
        # Calculate deltas (percentage change)
        total_revenue_delta = ((total_forecasted_revenue - total_historical_revenue) / total_historical_revenue) * 100
        avg_revenue_delta = ((avg_forecasted_revenue - avg_historical_revenue) / avg_historical_revenue) * 100

        # --- Calculate CAGR ---
        # Historical CAGR
        first_date_hist = df['ds'].min()
        last_date_hist = df['ds'].max()
        first_revenue_hist = df.loc[df['ds'] == first_date_hist, 'y'].iloc[0]
        last_revenue_hist = df.loc[df['ds'] == last_date_hist, 'y'].iloc[0]
        num_years_hist = (last_date_hist - first_date_hist).days / 365.25
        cagr_hist = (last_revenue_hist / first_revenue_hist)**(1 / num_years_hist) - 1 if num_years_hist > 0 else 0
        
        # Forecasted CAGR
        first_date_forecast = df['ds'].max()
        last_date_forecast = forecast_df['ds'].max()
        first_revenue_forecast = df.loc[df['ds'] == first_date_forecast, 'y'].iloc[0]
        last_revenue_forecast = forecast_df.loc[forecast_df['ds'] == last_date_forecast, forecast_col].iloc[0]
        num_years_forecast = (last_date_forecast - first_date_forecast).days / 365.25
        cagr_forecast = (last_revenue_forecast / first_revenue_forecast)**(1 / num_years_forecast) - 1 if num_years_forecast > 0 else 0
        
        
        st.markdown(
        """
        The revenue data for this dashboard is in **thousands**. All values shown are in thousands of dollars ($).
        """)
        st.markdown("---")
        
        with st.expander("🔑 Core Business Metrics", expanded=True):
            st.markdown('<div id="core-kpis"></div>', unsafe_allow_html=True)
            st.subheader("Core Revenue KPIs")
            
            col_kpi1, col_kpi2 = st.columns(2)
            with col_kpi1:
                st.markdown("#### Historical Metrics")
                st.markdown(
                    f"""
                    <div class="kpi-container">
                        <p class="kpi-title">Total Historical Revenue</p>
                        <p class="kpi-value">${total_historical_revenue/1000:,.2f}M</p>
                        <p class="kpi-subtitle">Sum of all past revenue</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""
                    <div class="kpi-container">
                        <p class="kpi-title">Avg. Daily Historical Revenue</p>
                        <p class="kpi-value">${avg_historical_revenue:,.2f}</p>
                        <p class="kpi-subtitle">Average daily revenue in the past</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""
                    <div class="kpi-container">
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
                    <div class="kpi-container">
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
                
                delta_icon_avg = "⬆️" if avg_revenue_delta > 0 else "⬇️" if avg_revenue_delta < 0 else "➡️"
                delta_class_avg = "positive-delta" if avg_revenue_delta > 0 else "negative-delta"
                st.markdown(
                    f"""
                    <div class="kpi-container">
                        <p class="kpi-title">Avg. Daily Forecasted Revenue</p>
                        <p class="kpi-value">${avg_forecasted_revenue:,.2f}</p>
                        <p class="kpi-subtitle">Forecasted Avg. over {forecast_months} months</p>
                        <div class="kpi-delta {delta_class_avg}">
                            <span class="delta-icon">{delta_icon_avg}</span>
                            <span>{avg_revenue_delta:,.2f}% vs. Historical</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                delta_icon_cagr = "⬆️" if cagr_forecast > cagr_hist else "⬇️" if cagr_forecast < cagr_hist else "➡️"
                delta_class_cagr = "positive-delta" if cagr_forecast > cagr_hist else "negative-delta"
                st.markdown(
                    f"""
                    <div class="kpi-container">
                        <p class="kpi-title">Forecasted CAGR</p>
                        <p class="kpi-value">{cagr_forecast:,.2%}</p>
                        <p class="kpi-subtitle">Avg. annual growth rate</p>
                        <div class="kpi-delta {delta_class_cagr}">
                            <span class="delta-icon">{delta_icon_cagr}</span>
                            <span>vs. Historical CAGR</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        st.markdown("---")
        
        with st.expander("📈 Growth Metrics", expanded=True):
            st.markdown('<div id="growth-metrics"></div>', unsafe_allow_html=True)
            st.subheader("Growth Metrics: MoM & YoY")
            
            # Separate Date Range Selector for Growth Metrics
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                # Use a container to group the date input and apply CSS class
                with st.container(border=True):
                    start_date_growth = st.date_input(
                        "Start Date (Growth Metrics):",
                        value=df['ds'].min().date(),
                        min_value=df['ds'].min().date(),
                        max_value=df['ds'].max().date(),
                        key='growth_start'
                    )
            with col_g2:
                # Use a container to group the date input and apply CSS class
                with st.container(border=True):
                    end_date_growth = st.date_input(
                        "End Date (Growth Metrics):",
                        value=df['ds'].max().date(),
                        min_value=df['ds'].min().date(),
                        max_value=df['ds'].max().date(),
                        key='growth_end'
                    )

            # Filter the historical data based on the user's date selection for Growth Metrics
            historical_growth_df = df[(df['ds'].dt.date >= start_date_growth) & (df['ds'].dt.date <= end_date_growth)].copy()
            
            # Recalculate monthly and yearly growth for the filtered historical data
            historical_growth_df['month'] = historical_growth_df['ds'].dt.to_period('M')
            monthly_revenue_hist = historical_growth_df.groupby('month')['y'].sum().reset_index()
            monthly_revenue_hist['MoM_Growth'] = monthly_revenue_hist['y'].pct_change() * 100
            
            historical_growth_df['year'] = historical_growth_df['ds'].dt.to_period('Y')
            yearly_revenue_hist = historical_growth_df.groupby('year')['y'].sum().reset_index()
            yearly_revenue_hist['YoY_Growth'] = yearly_revenue_hist['y'].pct_change() * 100

            # Calculate monthly and yearly growth for forecasted data
            forecast_df['month'] = forecast_df['ds'].dt.to_period('M')
            
            # Use the correct forecast column for growth calculations
            if what_if_enabled:
                monthly_revenue_forecast = forecast_df.groupby('month')['yhat_what_if'].sum().reset_index()
                monthly_revenue_forecast['MoM_Growth'] = monthly_revenue_forecast['yhat_what_if'].pct_change() * 100
            else:
                monthly_revenue_forecast = forecast_df.groupby('month')['yhat'].sum().reset_index()
                monthly_revenue_forecast['MoM_Growth'] = monthly_revenue_forecast['yhat'].pct_change() * 100

            forecast_df['year'] = forecast_df['ds'].dt.to_period('Y')
            
            if what_if_enabled:
                yearly_revenue_forecast = forecast_df.groupby('year')['yhat_what_if'].sum().reset_index()
                yearly_revenue_forecast['YoY_Growth'] = yearly_revenue_forecast['yhat_what_if'].pct_change() * 100
            else:
                yearly_revenue_forecast = forecast_df.groupby('year')['yhat'].sum().reset_index()
                yearly_revenue_forecast['YoY_Growth'] = yearly_revenue_forecast['yhat'].pct_change() * 100

            # Get the latest available growth rates for display
            latest_mom_hist = monthly_revenue_hist['MoM_Growth'].iloc[-1] if not monthly_revenue_hist['MoM_Growth'].empty else 0
            latest_yoy_hist = yearly_revenue_hist['YoY_Growth'].iloc[-1] if not yearly_revenue_hist['YoY_Growth'].empty else 0
            latest_mom_forecast = monthly_revenue_forecast['MoM_Growth'].iloc[-1] if not monthly_revenue_forecast['MoM_Growth'].empty else 0
            latest_yoy_forecast = yearly_revenue_forecast['YoY_Growth'].iloc[-1] if not yearly_revenue_forecast.empty else 0

            # Row 1: Month-over-Month Growth
            col7, col8 = st.columns(2)
            with col7:
                st.markdown(
                    f"""
                    <div class="kpi-container">
                        <p class="kpi-title">Latest Historical MoM Growth</p>
                        <p class="kpi-value">{latest_mom_hist:,.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col8:
                st.markdown(
                    f"""
                    <div class="kpi-container">
                        <p class="kpi-title">Latest Forecasted MoM Growth</p>
                        <p class="kpi-value">{latest_mom_forecast:,.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Row 2: Year-over-Year Growth
            col9, col10 = st.columns(2)
            with col9:
                st.markdown(
                    f"""
                    <div class="kpi-container">
                        <p class="kpi-title">Latest Historical YoY Growth</p>
                        <p class="kpi-value">{latest_yoy_hist:,.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col10:
                st.markdown(
                    f"""
                    <div class="kpi-container">
                        <p class="kpi-title">Latest Forecasted YoY Growth</p>
                        <p class="kpi-value">{latest_yoy_forecast:,.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        st.markdown("---")
        
        # --- NEW GRAPH: Historical Data Only ---
        st.markdown('<div id="historical-trends"></div>', unsafe_allow_html=True)
        st.subheader("Historical Revenue & 30-Day Moving Average")
        
        # Calculate 30-day moving average on the original historical data
        df['30_day_avg'] = df['y'].rolling(window=30, min_periods=1).mean()
        
        fig_hist = go.Figure()
        
        # Plot historical daily revenue
        fig_hist.add_trace(go.Scatter(
            x=df['ds'], y=df['y'],
            mode='lines',
            name='Historical Daily Revenue',
            line=dict(color='rgba(0,0,255,0.3)', width=1),
            hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> %{y:$,.2f}<extra></extra>'
        ))

        # Plot the 30-day moving average
        fig_hist.add_trace(go.Scatter(
            x=df['ds'], y=df['30_day_avg'],
            mode='lines',
            name='30-Day Moving Avg',
            line=dict(color='green', width=3),
            hovertemplate='<b>Date:</b> %{x}<br><b>30-Day Avg:</b> %{y:$,.2f}<extra></extra>'
        ))
        
        fig_hist.update_layout(
            title="Historical Revenue and Moving Average",
            xaxis_title="Date",
            yaxis_title="Revenue (in thousands of $)",
            yaxis=dict(tickprefix="$",),
            template="plotly_white",
            hovermode="x unified",
            xaxis_rangeslider_visible=True
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")

        # --- Daily Revenue Chart (Combined) ---
        st.markdown('<div id="daily-revenue"></div>', unsafe_allow_html=True)
        st.subheader("Daily Revenue Forecast and Historical Trend")
        
        # Separate Date Range Selector for Daily Revenue
        col_dr1, col_dr2 = st.columns(2)
        with col_dr1:
            with st.container(border=True):
                start_date_daily = st.date_input(
                    "Start Date (Daily Chart):",
                    value=df['ds'].min().date(),
                    min_value=df['ds'].min().date(),
                    max_value=forecast['ds'].max().date(),
                    key='daily_start'
                )
        with col_dr2:
            with st.container(border=True):
                end_date_daily = st.date_input(
                    "End Date (Daily Chart):",
                    value=forecast['ds'].max().date(),
                    min_value=df['ds'].min().date(),
                    max_value=forecast['ds'].max().date(),
                    key='daily_end'
                )

        # Filter the combined data based on the user's date selection for Daily Revenue
        combined_df_daily = combined_df[
            (combined_df['ds'].dt.date >= start_date_daily) & 
            (combined_df['ds'].dt.date <= end_date_daily)
        ]
        
        # Calculate 30-day moving average for the entire combined dataset first
        combined_df['30_day_avg'] = combined_df['y'].rolling(window=30, min_periods=1).mean()

        fig = go.Figure()

        # Get historical and forecast data from the combined filtered dataframe
        hist_filtered = combined_df_daily[combined_df_daily['type'] == 'Historical']
        forecast_filtered = combined_df_daily[combined_df_daily['type'] == 'Forecast']

        # Plot historical daily revenue (as a faint line)
        if not hist_filtered.empty:
            fig.add_trace(go.Scatter(
                x=hist_filtered['ds'], y=hist_filtered['y'],
                mode='lines',
                name='Historical Daily Revenue',
                line=dict(color='rgba(0,0,255,0.3)', width=1),
                hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> %{y:$,.2f}<extra></extra>'
            ))
        
        # Plot historical 30-day moving average
        if not hist_filtered.empty:
            fig.add_trace(go.Scatter(
                x=hist_filtered['ds'], y=combined_df.loc[hist_filtered.index, '30_day_avg'],
                mode='lines',
                name='Historical 30-Day Moving Avg',
                line=dict(color='green', width=3),
                hovertemplate='<b>Date:</b> %{x}<br><b>30-Day Avg:</b> %{y:$,.2f}<extra></extra>'
            ))

        # Plot forecasted daily revenue (dashed, distinct color)
        if not forecast_filtered.empty:
            fig.add_trace(go.Scatter(
                x=forecast_filtered['ds'], y=forecast_filtered['y'],
                mode='lines',
                name='Forecasted Daily Revenue',
                line=dict(color='rgba(255,0,0,0.4)', width=1, dash='dot'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Forecasted Revenue:</b> %{y:$,.2f}<extra></extra>'
            ))
        
        # Plot forecasted 30-day moving average (dashed, distinct color, thicker)
        if not forecast_filtered.empty:
            fig.add_trace(go.Scatter(
                x=forecast_filtered['ds'], y=combined_df.loc[forecast_filtered.index, '30_day_avg'],
                mode='lines',
                name='Forecasted 30-Day Moving Avg',
                line=dict(color='purple', width=3, dash='dash'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Forecasted 30-Day Avg:</b> %{y:$,.2f}<extra></extra>'
            ))

        # Confidence interval shading
        # Filter the original forecast data to get the correct bounds
        forecast_filtered_bounds = forecast[
            (forecast['ds'].dt.date >= start_date_daily) & 
            (forecast['ds'].dt.date <= end_date_daily) & 
            (forecast['ds'] > df['ds'].max())
        ]
        
        if not forecast_filtered_bounds.empty:
            fig.add_trace(go.Scatter(
                x=list(forecast_filtered_bounds['ds']) + list(forecast_filtered_bounds['ds'])[::-1],
                y=list(forecast_filtered_bounds['yhat_upper']) + list(forecast_filtered_bounds['yhat_lower'])[::-1],
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name=f"{confidence_interval*100:.0f}% Confidence Interval"
            ))
        
        # Add a vertical dashed line to mark the transition
        start_of_forecast = df['ds'].max()
        if start_of_forecast >= pd.to_datetime(start_date_daily) and start_of_forecast <= pd.to_datetime(end_date_daily):
            fig.add_vline(x=start_of_forecast, line_width=1, line_dash="dash", line_color="red")
            fig.add_annotation(
                x=start_of_forecast,
                y=1.05,
                xref="x",
                yref="paper",
                text='Forecast begins here',
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40
            )

        fig.update_layout(
            title="Daily Revenue: Historical vs. Forecasted",
            xaxis_title="Date",
            yaxis_title="Revenue (in thousands of $)",
            yaxis=dict(tickprefix="$",),
            template="plotly_white",
            hovermode="x unified",
            xaxis_rangeslider_visible=True,
            xaxis=dict(range=[start_date_daily, end_date_daily])
        )
        st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")

        # --- Cumulative Revenue Chart ---
        st.markdown('<div id="cumulative-revenue"></div>', unsafe_allow_html=True)
        st.subheader("📈 Cumulative Revenue Trend")

        # Separate Date Range Selector for Cumulative Revenue
        col_cr1, col_cr2 = st.columns(2)
        with col_cr1:
            with st.container(border=True):
                start_date_cumulative = st.date_input(
                    "Start Date (Cumulative Chart):",
                    value=df['ds'].min().date(),
                    min_value=df['ds'].min().date(),
                    max_value=forecast['ds'].max().date(),
                    key='cumulative_start'
                )
        with col_cr2:
            with st.container(border=True):
                end_date_cumulative = st.date_input(
                    "End Date (Cumulative Chart):",
                    value=forecast['ds'].max().date(),
                    min_value=df['ds'].min().date(),
                    max_value=forecast['ds'].max().date(),
                    key='cumulative_end'
                )

        # Calculate cumulative revenue for the full combined dataframe
        combined_df['cumulative_revenue'] = combined_df['y'].cumsum()
        
        # Filter the cumulative data based on the user's date selection
        cumulative_filtered = combined_df[
            (combined_df['ds'].dt.date >= start_date_cumulative) & 
            (combined_df['ds'].dt.date <= end_date_cumulative)
        ]

        # Create the plot
        fig_cumulative = go.Figure()

        # Plot historical cumulative revenue in one color
        historical_cumulative_filtered = cumulative_filtered[cumulative_filtered['type'] == 'Historical']
        if not historical_cumulative_filtered.empty:
            fig_cumulative.add_trace(go.Scatter(
                x=historical_cumulative_filtered['ds'],
                y=historical_cumulative_filtered['cumulative_revenue'],
                mode='lines',
                name='Historical Revenue',
                line=dict(color='blue', width=3)
            ))

        # Plot forecasted cumulative revenue in another color
        forecasted_cumulative_filtered = cumulative_filtered[cumulative_filtered['type'] == 'Forecast']
        if not forecasted_cumulative_filtered.empty:
            # We need to find the last value of the historical cumulative sum
            # to make the forecast cumulative sum continuous
            last_historical_cum_sum = 0
            if not combined_df[combined_df['type'] == 'Historical'].empty:
                last_historical_cum_sum = combined_df[combined_df['type'] == 'Historical']['cumulative_revenue'].iloc[-1]
            
            # The forecast cumulative starts from this last historical point
            forecasted_cumulative_filtered['cumulative_revenue_adjusted'] = forecasted_cumulative_filtered['y'].cumsum() + last_historical_cum_sum
            
            fig_cumulative.add_trace(go.Scatter(
                x=forecasted_cumulative_filtered['ds'],
                y=forecasted_cumulative_filtered['cumulative_revenue_adjusted'],
                mode='lines',
                name='Forecasted Revenue',
                line=dict(color='orange', width=3, dash='dash')
            ))

        # Add a vertical dashed line to mark the transition
        start_of_forecast = df['ds'].max()
        if start_of_forecast >= pd.to_datetime(start_date_cumulative) and start_of_forecast <= pd.to_datetime(end_date_cumulative):
            fig_cumulative.add_vline(x=start_of_forecast, line_width=1, line_dash="dash", line_color="red")
            fig_cumulative.add_annotation(
                x=start_of_forecast,
                y=1.05,
                xref="x",
                yref="paper",
                text='Forecast begins here',
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40
            )

        fig_cumulative.update_layout(
            title="Cumulative Revenue Over Time: Historical vs. Forecasted",
            xaxis_title="Date",
            yaxis_title="Cumulative Revenue (in thousands of $)",
            yaxis=dict(tickprefix="$",),
            template="plotly_white",
            hovermode="x unified",
            xaxis=dict(range=[start_date_cumulative, end_date_cumulative])
        )
        st.plotly_chart(fig_cumulative, use_container_width=True)

        st.markdown("---")

        # --- Forecast Table and Download ---
        st.markdown('<div id="forecast-table"></div>', unsafe_allow_html=True)
        st.subheader(f"🧾 {forecast_months}-Month Forecast Table")
        st.dataframe(
            forecast[['ds', forecast_col, 'yhat_lower', 'yhat_upper']].tail(forecast_period_days).rename(
                columns={
                    "ds": "Date",
                    forecast_col: "Predicted Revenue",
                    "yhat_lower": "Lower Bound",
                    "yhat_upper": "Upper Bound"
                }
            )
        )

        csv = forecast[['ds', forecast_col, 'yhat_lower', 'yhat_upper']].tail(forecast_period_days).to_csv(index=False)
        st.download_button(f"⬇️ Download {forecast_months}-Month Forecast CSV", csv, f"forecast_{forecast_months}_months.csv", "text/csv")


    with tab2:
        st.markdown('<div id="model-performance"></div>', unsafe_allow_html=True)
        st.subheader("📊 Model Performance")
        
        # Prepare data for comparison
        historical_comparison = pd.merge(df, forecast, on='ds', how='inner')
        
        # Calculate new KPIs
        wape = np.sum(np.abs(historical_comparison['y'] - historical_comparison['yhat'])) / np.sum(np.abs(historical_comparison['y'])) * 100
        forecast_bias = np.mean(historical_comparison['yhat'] - historical_comparison['y'])

        # Create columns for side-by-side metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div class="kpi-container">
                    <p class="kpi-title">Mean Absolute Error (MAE)</p>
                    <p class="kpi-value">${np.mean(np.abs(historical_comparison['y'] - historical_comparison['yhat'])):,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div class="kpi-container">
                    <p class="kpi-title">Root Mean Squared Error (RMSE)</p>
                    <p class="kpi-value">${np.sqrt(np.mean((historical_comparison['y'] - historical_comparison['yhat'])**2)):,.2f}</p>
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
        
        st.markdown('<div id="time-series-components"></div>', unsafe_allow_html=True)
        st.subheader("📉 Time Series Components")
        st.markdown("Prophet breaks down your data into trend, weekly seasonality, and yearly seasonality.")
        components_fig = plot_components_plotly(model, forecast)
        
        # Manually update the y-axis labels to include the currency symbol
        components_fig.update_yaxes(title_text='Revenue (in thousands of $)', tickprefix='$')
        
        st.plotly_chart(components_fig, use_container_width=True)
else:
    st.info("Please upload a CSV file from the sidebar to begin forecasting. The file must contain columns named 'ds' (for dates) and 'y' (for revenue).")

# --- WATERMARK ---
st.markdown('<p class="watermark">Created by Gemini for Data Analytics</p>', unsafe_allow_html=True)

# --- NLP CHATBOT LAYER ---
st.markdown("---")
st.markdown('<div id="chatbot"></div>', unsafe_allow_html=True)
st.header("🤖 Financial Chatbot")
st.markdown("Ask me questions about your data or the forecast. I'll do my best to help you!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I help you with your financial forecast?"})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a response based on a simple rule-based NLP system
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # A simple rule-based system to respond to common queries
            response = "I'm sorry, I don't understand that request. Try asking about 'forecast', 'historical', or 'KPIs'."
            
            # Convert prompt to lowercase for case-insensitive matching
            lower_prompt = prompt.lower()

            if "hello" in lower_prompt or "hi" in lower_prompt:
                response = "Hello there! What would you like to know about the financial data?"
            
            elif any(word in lower_prompt for word in ["forecast", "future"]):
                response = "I can tell you about the forecast. You can scroll down to the 'Daily Revenue Forecast' chart or the 'Forecast Table' to see future predictions. You can also adjust the forecast period using the slider in the sidebar!"
            
            elif "historical" in lower_prompt or "past" in lower_prompt:
                response = "The historical data can be viewed in the 'Historical Trends' chart and is the basis for the forecast. It shows the actual revenue up to the last available date."
            
            elif any(word in lower_prompt for word in ["kpis", "metrics", "performance"]):
                response = "The main KPIs are the total historical and forecasted revenue, as well as the average daily revenue. You can also see a detailed breakdown of model performance in the 'Model Performance' tab."

            elif "help" in lower_prompt or "options" in lower_prompt:
                response = "Here are a few things you can ask me:\n- Tell me about the **forecast**.\n- What are the **historical** trends?\n- Explain the **KPIs**.\n- What is **WAPE**?"

            elif "wape" in lower_prompt:
                response = "WAPE stands for Weighted Average Percentage Error. It provides an overall measure of how accurate the forecast is, expressed as a percentage of the total revenue. A lower percentage indicates a more accurate forecast."

            elif any(word in lower_prompt for word in ["revenue", "total", "what is"]):
                # This is a very basic attempt to handle numerical queries
                if "total historical revenue" in lower_prompt:
                    response = f"The total historical revenue is approximately ${total_historical_revenue:,.2f} thousand."
                elif "total forecasted revenue" in lower_prompt:
                    response = f"The total forecasted revenue is approximately ${total_forecasted_revenue:,.2f} thousand."
                else:
                    response = "I can provide total revenue figures for the historical and forecasted periods. Please be more specific with your query."

            elif any(word in lower_prompt for word in ["bye", "thanks", "thank you"]):
                response = "You're welcome! Feel free to ask if you have more questions."

        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
