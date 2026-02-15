import xgboost as xgb
import pandas as pd
import numpy as np

print("XGBoost version:", xgb.__version__)
df = pd.DataFrame({'x': np.arange(10), 'y': np.arange(10)})
model = xgb.XGBRegressor()
model.fit(df[['x']], df['y'])
print("Prediction:", model.predict(pd.DataFrame({'x': [10]})))
print("Success!")
