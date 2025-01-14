import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=86400)  # Cache data for 24 hours
def load_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        # stock_data.reset_index(inplace=True)
        stock_data.index.name = 'Date'
        return pd.DataFrame(stock_data)
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

# def load_data(ticker, start_date, end_date):
#     # Fetch data from an API or database
#     data = yf.download(ticker, start_date, end_date)
    
#     # Ensure the data is in the correct format
#     if isinstance(data, pd.DataFrame):
#         for col in data.columns:
#             if len(data[col].shape) > 1:
#                 data[col] = data[col].values.flatten()  # Flatten 2D arrays
#     return data

def validate_date(date_str):
    try:
        return pd.to_datetime(date_str)
    except ValueError:
        st.error("Invalid date format. Please use YYYY-MM-DD.")
        return None