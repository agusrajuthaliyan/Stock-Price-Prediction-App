import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import streamlit as st
from src.feature_engineering import create_features

def save_model(stock_ticker, model_data):
    if not os.path.exists('models'):
        os.makedirs('models')
    filename = f"models/{stock_ticker}_model_v2.pkl"
    joblib.dump(model_data, filename)

def load_model(stock_ticker):
    filename = f"models/{stock_ticker}_model_v2.pkl"
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        return None

def train_and_save_model(data, stock_ticker):
    """
    Trains an XGBoost model on the provided data.
    """
    df = data.copy()
    # Create features
    df_features = create_features(df)
    
    # Drop rows with NaN (due to lags)
    df_features = df_features.dropna()
    
    if df_features.empty:
        st.error("Not enough data to train model after creating features.")
        return None, None, None

    features = [c for c in df_features.columns if c not in ['Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']]
    target = 'Close'

    X = df_features[features]
    y = df_features[target]

    # Time Series Split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    with st.spinner("Training XGBoost model (with Seasonality & Lags)..."):
        # XGBoost Regressor
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            early_stopping_rounds=50,
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42
        )
        
        # Fit model
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
    
    # Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    # Save model and necessary metadata (latest data for future predictions)
    model_data = {
        'model': model,
        'features': features,
        'last_data': df.iloc[-60:].copy(), # Keep enough history to generate lags for next day
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }
    save_model(stock_ticker, model_data)
    
    return model, X_test, y_test, predictions

def predict_future(model, last_known_data, days=30):
    """
    Predicts future 'days' using recursive forecasting.
    """
    future_predictions = []
    current_data = last_known_data.copy()
    
    # Get last date
    last_date = current_data.index[-1]
    
    # Generate future dates (business days)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='B')
    
    # Features used by model
    # Try to get features from model if available (XGBoost sklearn API stores this)
    if hasattr(model, 'feature_names_in_'):
        feature_cols = list(model.feature_names_in_)
    else:
        # Fallback to logic (risky if columns changed)
        feature_cols = [c for c in current_data.columns if c not in ['Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']]
    
    for date in future_dates:
        # Create a new row with NaN Close
        # We need to append strictly to generating features
        # The create_features function needs the whole history to calculate lags
        
        # Temp dataframe with new row
        new_row = pd.DataFrame(index=[date], columns=current_data.columns)
        new_row['Close'] = np.nan 
        
        temp_df = pd.concat([current_data, new_row])
        
        # Generate features
        temp_features = create_features(temp_df)
        
        # Get the feature row for the target date (last row)
        last_row_features = temp_features.iloc[[-1]]
        
        # Select and Order columns to match model
        try:
             X_future = last_row_features[feature_cols]
        except KeyError as e:
             st.error(f"Feature mismatch: {e}. The model might be incompatible with current data features.")
             return pd.DataFrame()
        
        # Predict
        pred_price = model.predict(X_future)[0]
        
        # Update current_data with the predicted value for next iteration's lag
        new_row['Close'] = pred_price
        current_data = pd.concat([current_data, new_row])
        
        future_predictions.append({'Date': date, 'Predicted_Close': pred_price})
        
    return pd.DataFrame(future_predictions).set_index('Date')
