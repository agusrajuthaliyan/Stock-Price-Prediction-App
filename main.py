import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
from src.data_loader import load_data
from src.model import train_and_save_model, load_model, predict_future
from src.utils import validate_ticker

# Page Config
st.set_page_config(
    page_title="Stock Market Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Premium" feel
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #1f2937;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #374151;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("assets/Home.png" if pd.io.common.file_exists("assets/Home.png") else "https://cdn-icons-png.flaticon.com/512/2910/2910312.png", width=50)
    st.title("Settings")
    
    # Stock Ticker Input
    user_ticker = st.text_input("Enter Stock Ticker", value="NVDA")
    stock_ticker = validate_ticker(user_ticker)
    if not stock_ticker:
        st.stop()
    
    # Date Range
    st.subheader("Training Data Range")
    end_date = st.date_input("End Date", value=pd.Timestamp.now())
    start_date = st.date_input("Start Date", value=pd.Timestamp.now() - timedelta(days=730)) # 2 years
    
    # Forecast Horizon
    forecast_days = st.slider("Forecast Days", min_value=7, max_value=90, value=30)
    
    if st.button("Train New Model"):
        # Force retrain
        st.session_state['force_retrain'] = True
    else:
        st.session_state['force_retrain'] = False

# Main Content
st.title(f"📈 {stock_ticker} Stock Price Prediction")
st.markdown("Advanced forecasting using **XGBoost** with Seasonality and Lag features.")

# Load Data
with st.spinner(f"Fetching data for {stock_ticker}..."):
    df = load_data(stock_ticker, start_date, end_date)

if df.empty:
    st.error("No data found. Please check the ticker symbol.")
    st.stop()

# Display Raw Data (Collapsible)
with st.expander("View Historical Data"):
    st.dataframe(df.tail(10).sort_index(ascending=False))

# --- Model Handling ---
# Check if model exists or needs retraining
model_data = load_model(stock_ticker)
should_retrain = st.session_state.get('force_retrain', False) or (model_data is None)

if should_retrain:
    with st.spinner("Training model... This usually takes a few seconds."):
        # Train
        try:
             # Just call the function, it saves to disk
             # We need to pass the *raw* dataframe to create features
             model, X_test, y_test, predictions = train_and_save_model(df, stock_ticker)
             model_data = load_model(stock_ticker) # Reload to get clean dictionary
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()
else:
    st.success("Loaded existing model.")

# --- Evaluation Metrics ---
st.subheader("Model Performance (Backtest)")
c1, c2, c3 = st.columns(3)
if model_data and 'mse' in model_data:
    c1.metric("RMSE (Root Mean Sq Error)", f"${model_data['rmse']:.2f}")
    c2.metric("MAE (Mean Abs Error)", f"${model_data['mae']:.2f}")
else:
    st.warning("Metrics not available.")

# --- Forecasting ---
st.subheader(f"Future Forecast ({forecast_days} Days)")

if model_data:
    model = model_data['model']
    
    with st.spinner("Generating Forecast..."):
        # We need to construct the input for predict_future
        feature_names = model_data.get('features', [])
        future_df = predict_future(model, df, days=forecast_days)
        
    # Plotting
    fig = go.Figure()
    
    # Historical Data (Last 90 days for clarity)
    history_df = df.iloc[-90:]
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['Close'], 
                             mode='lines', name='Historical Close',
                             line=dict(color='#ebecf0', width=2)))
                             
    # specific line for last connected part
    fig.add_trace(go.Scatter(x=[history_df.index[-1], future_df.index[0]], 
                             y=[history_df['Close'].iloc[-1], future_df['Predicted_Close'].iloc[0]],
                             mode='lines', line=dict(color='#2563eb', dash='dash', width=2), showlegend=False))

    # Forecast Data
    fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Predicted_Close'], 
                             mode='lines+markers', name='Forecast',
                             line=dict(color='#2563eb', width=3)))

    fig.update_layout(
        title=f"Price Forecast for {stock_ticker}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=50, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show forecast data
    with st.expander("View Forecast Data"):
        st.dataframe(future_df)

# Footer
st.markdown("---")
st.markdown("Disclaimer: This is a predictive model for educational purposes. Stock market data is volatile.")
