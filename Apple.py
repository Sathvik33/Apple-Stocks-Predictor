import streamlit as st
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(__file__)

def load_model():
    model_path = os.path.join(BASE_DIR,"model", "apple_stock_model.pkl")
    return joblib.load(model_path)

def load_scaler():
    scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")
    return joblib.load(scaler_path)

def get_historical_data(stock_symbol, period="1y"):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period=period)
    return data

def calculate_indicators(data):
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()
    data["RSI"] = 100 - (100 / (1 + data["Close"].diff(1).apply(lambda x: (x if x > 0 else 0)).rolling(14).mean() / 
                                      data["Close"].diff(1).apply(lambda x: (-x if x < 0 else 0)).rolling(14).mean()))
    data["MACD"] = data["Close"].ewm(span=12, adjust=False).mean() - data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data.dropna(inplace=True)
    return data

def get_latest_features(data):
    return data.iloc[-1][["SMA_50", "SMA_200", "EMA_50", "RSI", "MACD", "MACD_Signal"]].values

st.title("Stock Price Predictor")

stock_symbol = st.text_input("Enter Stock Symbol:", "AAPL")
n_days = st.number_input("Enter number of days to predict:", min_value=1, max_value=365, value=7)

col1, col2 = st.columns(2)

if col1.button("Predict"):
    model = load_model()
    scaler = load_scaler()
  
    historical_data = get_historical_data(stock_symbol, period="2y")
    historical_data = calculate_indicators(historical_data)
    
    
    if len(historical_data) < 200:
        st.error("Insufficient historical data to calculate indicators. Please use a longer period.")
    else:
        simulated_prices = list(historical_data["Close"].values)
        predictions = []
        
        for _ in range(n_days):
            temp_df = pd.DataFrame(simulated_prices, columns=["Close"])
            temp_df = calculate_indicators(temp_df)
            
            latest_features = get_latest_features(temp_df).reshape(1, -1)
            latest_features_scaled = scaler.transform(latest_features)
            
            pred_price = model.predict(latest_features_scaled)[0]
            predictions.append(pred_price)

            simulated_prices.append(pred_price)
        
        df_predictions = pd.DataFrame({"Day": list(range(1, n_days + 1)), "Predicted Price": predictions})
        st.write(df_predictions)
        
        fig, ax = plt.subplots()
        ax.plot(historical_data.index, historical_data["Close"], label="Historical Prices")
        ax.plot(pd.date_range(historical_data.index[-1], periods=n_days + 1, freq="B")[1:], predictions, marker='o', linestyle='-', color='r', label='Predicted Prices')
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price (USD)")
        ax.set_title(f"{stock_symbol} Stock Price Prediction for {n_days} Days")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

if col2.button("Clear"):
    st.experimental_rerun()