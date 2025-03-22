import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from datetime import datetime
import time

def fetch_data(interval='1h', lookback=500):
    """Fetch gold price data from Yahoo Finance."""
    data = yf.download('GC=F', period='60d', interval=interval) 
    if data.empty:
        raise ValueError(f"No data available for interval '{interval}'. Try a larger timeframe.")
    return data['Close'].tail(lookback).values.reshape(-1, 1)

def prepare_data(data, time_steps=60):
    """Prepare dataset for LSTM."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i-time_steps:i, 0])
        y.append(data_scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    if len(X.shape) >= 2: 
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    else:
        raise ValueError("Insufficient data for the specified time_steps. Try fetching more data or reducing time_steps.")
        
    return X, y, scaler

def build_lstm_model():  
    """Create and compile an LSTM model."""
    model = Sequential([
        LSTM(units=128, return_sequences=True, input_shape=(60, 1)),  # Increased neurons
        Dropout(0.2),  # Increased dropout to prevent overfitting
        
        LSTM(units=128, return_sequences=True),
        Dropout(0.2),

        LSTM(units=64, return_sequences=True),  # Extra LSTM layer
        Dropout(0.2),

        LSTM(units=64),
        Dropout(0.2),

        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model using different metrics."""
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))  

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    accuracy = 100 - (rmse / np.mean(y_test)) * 100

    print(f"Model Performance:\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR² Score: {r2:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    return mse, rmse, mae, r2

def train_and_predict(interval='1h'):
    """Train LSTM model, evaluate performance, and predict next value."""
    data = fetch_data(interval)
    X, y, scaler = prepare_data(data)
    
    # Splitting data into train and test sets
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = build_lstm_model()
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test, scaler)
    
    # Predict next value
    last_data = data[-60:].reshape(-1, 1)
    last_data_scaled = scaler.transform(last_data)
    last_data_scaled = np.reshape(last_data_scaled, (1, 60, 1))
    prediction = model.predict(last_data_scaled)
    return scaler.inverse_transform(prediction)[0, 0]

while True:
    try:
        predicted_price = train_and_predict(interval='1h')
        print("Predicted gold price for next hour:", predicted_price, "at", time.strftime("%Y-%m-%d %H:%M:%S"))
    except ValueError as e:
        print("Error:", e)
    time.sleep(3600)