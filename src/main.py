import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import zscore
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load Dataset
df = pd.read_csv("demanddata_2024.csv")

# Data Cleaning & Processing
df["SETTLEMENT_DATE"] = pd.to_datetime(df["SETTLEMENT_DATE"], format="%d-%b-%Y")
df["DATETIME"] = df["SETTLEMENT_DATE"] + pd.to_timedelta((df["SETTLEMENT_PERIOD"] - 1) * 30, unit="m")
columns_to_keep = ["DATETIME", "ND", "TSD", "ENGLAND_WALES_DEMAND"]
df_cleaned = df[columns_to_keep].sort_values(by="DATETIME")

# Anomaly Detection
df_cleaned["Z_SCORE"] = zscore(df_cleaned["ENGLAND_WALES_DEMAND"])
anomalies = df_cleaned[abs(df_cleaned["Z_SCORE"]) > 3]

# Feature Engineering
df_cleaned["MONTH"] = df_cleaned["DATETIME"].dt.month
df_cleaned["WEEKDAY"] = df_cleaned["DATETIME"].dt.weekday
df_cleaned["HOUR"] = df_cleaned["DATETIME"].dt.hour
df_cleaned["LAG_1H"] = df_cleaned["ENGLAND_WALES_DEMAND"].shift(2)
df_cleaned["LAG_1D"] = df_cleaned["ENGLAND_WALES_DEMAND"].shift(48)
df_cleaned.dropna(inplace=True)

# Prepare Data for Machine Learning
features = ["HOUR", "WEEKDAY", "MONTH", "LAG_1H", "LAG_1D"]
target = "ENGLAND_WALES_DEMAND"
X_train, X_test, y_train, y_test = train_test_split(df_cleaned[features], df_cleaned[target], test_size=0.15, random_state=42, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Machine Learning Models
ridge_model = Ridge(alpha=1.0).fit(X_train_scaled, y_train)
lasso_model = Lasso(alpha=0.1).fit(X_train_scaled, y_train)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

ridge_preds = ridge_model.predict(X_test_scaled)
lasso_preds = lasso_model.predict(X_test_scaled)
rf_preds = rf_model.predict(X_test)

# Model Evaluation
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")

evaluate_model(y_test, ridge_preds, "Ridge Regression")
evaluate_model(y_test, lasso_preds, "Lasso Regression")
evaluate_model(y_test, rf_preds, "Random Forest")

# LSTM Model for Time-Series Forecasting
X_lstm = df_cleaned[["LAG_1H", "LAG_1D"]].values
y_lstm = df_cleaned[target].values
X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.15, random_state=42, shuffle=False)

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test_lstm))

lstm_preds = lstm_model.predict(X_test_lstm).flatten()
evaluate_model(y_test_lstm, lstm_preds, "LSTM Model")
