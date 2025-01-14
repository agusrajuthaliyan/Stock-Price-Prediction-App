import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def plot_stock_price(data, stock_ticker):
    fig = px.line(
        data,
        x=data.index,  # Use the index for the x-axis
        y=data['Close'].values.flatten(),  # Ensure 'Close' is 1D
        title=f"{stock_ticker} Closing Price Over Time",
        range_x=[data.index.min(), data.index.max()],  # Adjust range to use index
    )
    return fig

def plot_model_performance(y_true, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predictions'))
    fig.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()], 
                             mode='lines', name='Ideal Prediction', line=dict(color='red', dash='dash')))
    fig.update_layout(title='Actual vs Predicted Closing Prices',
                      xaxis_title='Actual Price',
                      yaxis_title='Predicted Price')
    st.plotly_chart(fig)

def validate_ticker(ticker):
    if not ticker or not ticker.isalpha():
        st.error("Invalid ticker. Please enter a valid stock symbol.")
        return ""
    return ticker.upper()

def validate_numeric_input(value, name):
    if value <= 0:
        st.error(f"Invalid {name}. Please enter a positive number.")
        return None
    return value
