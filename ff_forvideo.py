# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import numpy as np
from prophet.plot import plot_components_plotly
import base64
import os

# --- CONFIGURATION ---
LOGO_PATH = "miracle-logo-dark.png"
DATA_PATH = "financial_forecast_modified.csv"  # <-- The data file is now hardcoded to this path

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
            color: #262730; /* Dark charcoal text */
        }}
        
        /* Apply custom theme and background for light mode */
        .stApp {{
            background-color: #f0f2f6; /* Light gray background */
        }}
        
        /* Style the main content container */
        .main .block-container {{
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }}
        
        /* Style for cards/containers */
        .stContainer, .st-emotion-cache-1px211l, .st-emotion-cache-1c7v0u4 {{
            background-color: #ffffff;
            border-radius: 0.5rem;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.02);
            border: 1px solid #e0e0e0;
            margin-bottom: 1rem;
        }}

        /* Style for expanders */
        .st-emotion-cache-163v2p5 {{
            background-color: #ffffff;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.02);
            border: 1px solid #e0e0e0;
        }}
        
        /* Style for metrics with a fresh look */
        [data-testid="stMetric"] {{
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 0.5rem;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.02);
        }}
        [data-testid="stMetricLabel"] label {{
            font-size: 0.8rem;
            color: #6c757d;
        }}
        [data-testid="stMetricValue"] {{
            font-size: 1.5rem;
            font-weight: 600;
            color: #495057;
        }}
        
        /* Hide the default Streamlit footer and menu button for a cleaner look */
        #MainMenu, footer {{
            visibility: hidden;
            height: 0;
        }}
    </style>
    """
    , unsafe_allow_html=True)


# --- HEADER & INTRODUCTION ---
st.header("Retail & E-commerce Financial Forecasting 📈")
st.write("This application leverages the Prophet model to forecast key financial metrics for retail and e-commerce businesses.")

st.markdown("---")

# --- DATA LOADING AND FORECASTING PARAMETERS ---
st.sidebar.header("Configure Forecast 🛠️")

# Check if the data file exists and load it
if os.path.exists(DATA_PATH):
    data = pd.read_csv(DATA_PATH)
    st.sidebar.success(f"Data loaded from **{DATA_PATH}** automatically.")

    # Display a sample of the uploaded data
    st.subheader("Loaded Data Preview")
    st.write(data.head())

    st.markdown("---")

    # --- MAIN DASHBOARD CONTENT ---
    st.subheader("Financial Forecasting Dashboard")

    # Layout using columns for KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Sales (Actual)", value=f"${data['y'].sum():,.2f}")
    with col2:
        st.metric(label="Average Daily Sales (Actual)", value=f"${data['y'].mean():,.2f}")
    with col3:
        # Placeholder values for forecasted metrics
        st.metric(label="Total Sales (Forecasted)", value=f"${data['y'].sum() * 1.05:,.2f}")
    with col4:
        # Placeholder values for forecasted metrics
        st.metric(label="Average Daily Sales (Forecasted)", value=f"${data['y'].mean() * 1.05:,.2f}")

    st.markdown("---")
    
    # Forecast configuration
    periods = st.sidebar.slider("Periods to forecast (days)", min_value=30, max_value=365, value=90)
    
    # Fit the Prophet model
    st.spinner("Generating forecast...")
    m = Prophet(seasonality_mode='multiplicative',
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True)
    m.fit(data)
    
    # Create future dataframe and make predictions
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    
    st.sidebar.success("Forecast generated!")

    # --- VISUALIZATIONS ---
    st.subheader("Sales Forecast")
    
    # Plot the forecast
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted Sales', line=dict(color='#0077b6')))
    fig_forecast.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='markers', name='Actual Sales', marker=dict(color='#8d99ae')))
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', fill='tonexty', line_color='rgba(0,119,182,0.2)', name='Confidence Interval', showlegend=False))
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', fill='tonexty', line_color='rgba(0,119,182,0.2)', name='Confidence Interval', showlegend=False))

    fig_forecast.update_layout(
        title='Retail Sales Forecast',
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Display forecast components using expander
    with st.expander("Show Forecast Components"):
        st.write("Seasonal and trend components of the forecast.")
        fig_components = plot_components_plotly(m, forecast)
        fig_components.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_components, use_container_width=True)

    # --- KEY INSIGHTS & SUMMARY ---
    st.markdown("---")
    st.subheader("Key Insights")
    
    # Use a container for a clean insights summary
    with st.container():
        st.markdown(
            """
            * **Daily Trends**: The forecast indicates clear daily sales patterns, with peaks and troughs influenced by weekly and yearly seasonality.
            * **Confidence Interval**: The shaded area around the forecast line represents the model's confidence interval. A wider interval indicates higher uncertainty in the prediction.
            * **Data-driven Decisions**: This dashboard can help you with strategic planning, inventory management, and marketing campaign timing.
            """
        )
        
else:
    st.error(f"Error: The data file **{DATA_PATH}** was not found in the repository.")
    st.info("Please ensure your CSV file is in the same directory as the app.py script.")
    st.image("https://images.unsplash.com/photo-1542831371-29b0f74f9713?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb", use_column_width=True)
    st.write("Example data format: A CSV with `ds` (date) and `y` (sales) columns.")

# Optional: Add a subtle footer to the bottom of the page
st.markdown(
    """
    <div style="text-align: center; margin-top: 2rem;">
        <p style="color: #6c757d; font-size: 0.8rem;">
            Powered by Streamlit and Prophet
        </p>
    </div>
    """
    , unsafe_allow_html=True)
