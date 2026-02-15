import pandas as pd
import numpy as np

def create_features(df):
    """
    Creates time series features from datetime index.
    Assumes df has a DateTimeIndex and column 'Close'.
    """
    df = df.copy()
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    
    # Seasonality
    # Use 365.25 for average year length
    df['sin_day'] = np.sin(2 * np.pi * df['dayofyear']/365.25)
    df['cos_day'] = np.cos(2 * np.pi * df['dayofyear']/365.25)
    df['sin_month'] = np.sin(2 * np.pi * df['month']/12)
    df['cos_month'] = np.cos(2 * np.pi * df['month']/12)

    # Lags (using 'Close')
    # Shift 1 means predicting Close(t) using Close(t-1)
    # So to predict Day T, inputs are from T-1, T-2...
    lags = [1, 2, 3, 7, 14, 30]
    for lag in lags:
        df[f'lag_{lag}'] = df['Close'].shift(lag)
        
    # Rolling features
    windows = [7, 14, 30]
    for window in windows:
        # Shift 1 to avoid data leakage (rolling mean of past values only)
        df[f'roll_mean_{window}'] = df['Close'].shift(1).rolling(window=window).mean()
        df[f'roll_std_{window}'] = df['Close'].shift(1).rolling(window=window).std()
        
    return df
