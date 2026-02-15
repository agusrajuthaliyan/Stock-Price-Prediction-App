import pandas as pd
import numpy as np

def create_features(df, label=None):
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df['date'] = df.index
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    
    # Seasonality
    df['sin_day'] = np.sin(2 * np.pi * df['dayofyear']/365.25)
    df['cos_day'] = np.cos(2 * np.pi * df['dayofyear']/365.25)
    df['sin_month'] = np.sin(2 * np.pi * df['month']/12)
    df['cos_month'] = np.cos(2 * np.pi * df['month']/12)

    # Lag features (past values)
    # We want to predict Close, so we use past Close values as features
    # Note: When training, we must shift features so that at time t we don't see target t
    # But usually we prepare X and y separately. 
    # Here we create columns that represent PAST values relative to the row's index.
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
        
    # Rolling window statistics
    for window in [7, 14, 30]:
        df[f'close_roll_mean_{window}'] = df['Close'].shift(1).rolling(window=window).mean()
        df[f'close_roll_std_{window}'] = df['Close'].shift(1).rolling(window=window).std()
        
    # Return features and label
    if label:
        return df, df[label]
    return df
