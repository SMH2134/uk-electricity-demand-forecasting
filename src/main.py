import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: write PNG files, never open a window
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import random
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Reproducibility — seed every RNG the pipeline touches
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Output directories (repo_root/models, repo_root/plots)
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load Dataset
df = pd.read_csv("demanddata_2024.csv")

# Data Cleaning & Processing
df["SETTLEMENT_DATE"] = pd.to_datetime(df["SETTLEMENT_DATE"], format="%d-%b-%Y")
df["DATETIME"] = df["SETTLEMENT_DATE"] + pd.to_timedelta((df["SETTLEMENT_PERIOD"] - 1) * 30, unit="m")
columns_to_keep = ["DATETIME", "ND", "TSD", "ENGLAND_WALES_DEMAND"]
df_cleaned = df[columns_to_keep].sort_values(by="DATETIME")

# Anomaly Detection — unsupervised Isolation Forest on demand signals
anomaly_features = ["ND", "TSD", "ENGLAND_WALES_DEMAND"]
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df_cleaned["ANOMALY_FLAG"] = iso_forest.fit_predict(df_cleaned[anomaly_features])
df_cleaned["ANOMALY_SCORE"] = iso_forest.decision_function(df_cleaned[anomaly_features])
anomalies = df_cleaned[df_cleaned["ANOMALY_FLAG"] == -1]
print(f"Isolation Forest flagged {len(anomalies)} anomalies out of {len(df_cleaned)} records")

# Persist the fitted detector immediately
joblib.dump(iso_forest, os.path.join(MODELS_DIR, "isolation_forest.pkl"))

# Anomaly detection chart — demand series with the flagged points highlighted
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_cleaned["DATETIME"], df_cleaned["ENGLAND_WALES_DEMAND"],
        lw=0.5, color="#3b6fb0", label="England & Wales demand")
ax.scatter(anomalies["DATETIME"], anomalies["ENGLAND_WALES_DEMAND"],
           color="#d1495b", s=18, zorder=5,
           label=f"Isolation Forest anomalies (n={len(anomalies)})")
ax.set_title(f"Isolation Forest anomaly detection — contamination=1%, "
             f"{len(anomalies)} of {len(df_cleaned)} points flagged")
ax.set_xlabel("Date")
ax.set_ylabel("Demand (MW)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "anomaly_detection.png"), dpi=120)
plt.close(fig)

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

# Persist the fitted scaler (used by the baseline linear models)
joblib.dump(scaler, os.path.join(MODELS_DIR, "preprocessing.pkl"))

# Train Machine Learning Models
ridge_model = Ridge(alpha=1.0).fit(X_train_scaled, y_train)
lasso_model = Lasso(alpha=0.1).fit(X_train_scaled, y_train)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

joblib.dump(ridge_model, os.path.join(MODELS_DIR, "ridge.pkl"))
joblib.dump(lasso_model, os.path.join(MODELS_DIR, "lasso.pkl"))
joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest.pkl"))

ridge_preds = ridge_model.predict(X_test_scaled)
lasso_preds = lasso_model.predict(X_test_scaled)
rf_preds = rf_model.predict(X_test)

# Model Evaluation
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")

# Persistence baseline — naive forecast: each prediction is the previous half-hourly reading.
# Evaluated on the same chronological test partition as the models below.
persistence_preds = df_cleaned[target].shift(1).loc[y_test.index]
evaluate_model(y_test, persistence_preds, "Persistence (naive t-1)")

evaluate_model(y_test, ridge_preds, "Ridge Regression")
evaluate_model(y_test, lasso_preds, "Lasso Regression")
evaluate_model(y_test, rf_preds, "Random Forest")

# LSTM Model for Time-Series Forecasting
# Feature parity with the baselines: the LSTM trains on the same 5 features
# (HOUR, WEEKDAY, MONTH, LAG_1H, LAG_1D) so the model comparison is like-for-like.
# The features and target span ~0-70,000 MW; feeding those magnitudes straight
# into the network keeps the MSE gradient enormous and the model collapses to a
# flat mean prediction (audit D3/D8). Scale both the inputs and the target into
# [0, 1] first, then invert the predictions back to MW.
lstm_feature_cols = features
X_lstm_raw = df_cleaned[lstm_feature_cols].values
y_lstm_raw = df_cleaned[target].values.reshape(-1, 1)

# Chronological hold-out (no shuffle); scalers fit on the training slice only
split_idx = int(len(X_lstm_raw) * 0.85)
X_train_raw, X_test_raw = X_lstm_raw[:split_idx], X_lstm_raw[split_idx:]
y_train_raw, y_test_raw = y_lstm_raw[:split_idx], y_lstm_raw[split_idx:]

lstm_feature_scaler = MinMaxScaler()
lstm_target_scaler = MinMaxScaler()
X_train_scaled_lstm = lstm_feature_scaler.fit_transform(X_train_raw)
X_test_scaled_lstm = lstm_feature_scaler.transform(X_test_raw)
y_train_scaled_lstm = lstm_target_scaler.fit_transform(y_train_raw)
y_test_scaled_lstm = lstm_target_scaler.transform(y_test_raw)

# Reshape to (samples, timesteps, features)
X_train_lstm = X_train_scaled_lstm.reshape((X_train_scaled_lstm.shape[0], len(lstm_feature_cols), 1))
X_test_lstm = X_test_scaled_lstm.reshape((X_test_scaled_lstm.shape[0], len(lstm_feature_cols), 1))

lstm_model = Sequential([
    Input(shape=(len(lstm_feature_cols), 1)),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
history = lstm_model.fit(
    X_train_lstm, y_train_scaled_lstm,
    epochs=50, batch_size=32,
    validation_data=(X_test_lstm, y_test_scaled_lstm),
    callbacks=[early_stopping, reduce_lr],
)

epochs_run = len(history.history["loss"])
if early_stopping.stopped_epoch:
    best_epoch = getattr(early_stopping, "best_epoch", None)
    best_txt = f"{best_epoch + 1}" if best_epoch is not None else "n/a"
    print(f"LSTM early-stopped after epoch {early_stopping.stopped_epoch + 1} "
          f"(best weights restored from epoch {best_txt})")
else:
    print(f"LSTM ran the full {epochs_run} epochs without early stopping")

# Predict in scaled space, then invert both sides back to megawatts to score
lstm_preds_scaled = lstm_model.predict(X_test_lstm)
lstm_preds = lstm_target_scaler.inverse_transform(lstm_preds_scaled).flatten()
y_test_lstm_mw = lstm_target_scaler.inverse_transform(y_test_scaled_lstm).flatten()
evaluate_model(y_test_lstm_mw, lstm_preds, "LSTM Model")

# Persist the trained LSTM architecture + weights, the LSTM scalers, and metadata
lstm_model.save(os.path.join(MODELS_DIR, "best_lstm_model.h5"))
joblib.dump(lstm_feature_scaler, os.path.join(MODELS_DIR, "lstm_feature_scaler.pkl"))
joblib.dump(lstm_target_scaler, os.path.join(MODELS_DIR, "lstm_target_scaler.pkl"))

model_metadata = {
    "baseline_features": features,
    "lstm_features": lstm_feature_cols,
    "target": target,
    "anomaly_features": anomaly_features,
    "lag_1h_periods": 2,
    "lag_1d_periods": 48,
    "lstm_input_scaler": "MinMaxScaler",
    "lstm_target_scaler": "MinMaxScaler",
    "lstm_feature_scaler_file": "lstm_feature_scaler.pkl",
    "lstm_target_scaler_file": "lstm_target_scaler.pkl",
    "trained_on": "demanddata_2024.csv",
}
joblib.dump(model_metadata, os.path.join(MODELS_DIR, "preprocessing_metadata.pkl"))
print(f"Saved model artifacts to {os.path.abspath(MODELS_DIR)}")

# LSTM training-history chart
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, epochs_run + 1), history.history["loss"], label="train loss")
ax.plot(range(1, epochs_run + 1), history.history["val_loss"], label="validation loss")
if early_stopping.stopped_epoch:
    ax.axvline(early_stopping.best_epoch + 1, color="grey", ls="--", lw=1,
               label=f"best epoch ({early_stopping.best_epoch + 1})")
ax.set_title("LSTM training history")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (MSE on MinMax-scaled target)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "training_history.png"), dpi=120)
plt.close(fig)

# Actual-vs-predicted chart over the held-out test partition
test_dt = df_cleaned.loc[y_test.index, "DATETIME"].values
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_dt, y_test.values, color="black", lw=1.0, label="Actual")
ax.plot(test_dt, rf_preds, color="#2a9d8f", lw=0.9, alpha=0.9,
        label=f"Random Forest (MAE {mean_absolute_error(y_test, rf_preds):.0f} MW)")
ax.plot(test_dt, lstm_preds, color="#e76f51", lw=0.9, alpha=0.9,
        label=f"LSTM (MAE {mean_absolute_error(y_test_lstm_mw, lstm_preds):.0f} MW)")
ax.set_title("Test partition: actual vs predicted demand")
ax.set_xlabel("Date")
ax.set_ylabel("Demand (MW)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "predictions_full.png"), dpi=120)
plt.close(fig)

print(f"Saved plots to {os.path.abspath(PLOTS_DIR)}")
