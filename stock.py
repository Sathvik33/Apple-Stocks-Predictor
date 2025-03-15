import yfinance as yf
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,StackingClassifier,StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error,mean_squared_error,mean_absolute_error,precision_score,r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import os

BASE_DIR = os.path.dirname(__file__)



stock_symbol = "AAPL"
stock = yf.Ticker(stock_symbol)
data = stock.history(period="15y")

data["SMA_50"] = data["Close"].rolling(window=50).mean()
data["SMA_200"] = data["Close"].rolling(window=200).mean()
data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()
data["RSI"] = 100 - (100 / (1 + data["Close"].diff(1).apply(lambda x: (x if x > 0 else 0)).rolling(14).mean() / data["Close"].diff(1).apply(lambda x: (-x if x < 0 else 0)).rolling(14).mean()))
data["MACD"] = data["Close"].ewm(span=12, adjust=False).mean() - data["Close"].ewm(span=26, adjust=False).mean()
data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

data.dropna(inplace=True)
features=["SMA_50","SMA_200","EMA_50","RSI", "MACD", "MACD_Signal"]
x=data[features]
y=data["Close"].shift(-1).dropna()
x=x.iloc[:-1]


scaler=MinMaxScaler()
x=scaler.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=45)

params={
    'n_estimators': [100,200,300,500,1000],
    'learning_rate': [0.005,0.1,1],
    'max_depth': [1,3,5],
    "min_child_weight": [1, 3, 5, 7],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2, 0.3, 0.4],
    "reg_alpha": [0, 0.01, 0.1, 1, 10],
    "reg_lambda": [0, 0.01, 0.1, 1, 10]
}

model=RandomizedSearchCV(estimator=XGBRegressor(),param_distributions=params,n_jobs=-1,n_iter=300,cv=5,verbose=1,scoring="neg_mean_absolute_error")
model.fit(x_train, y_train,eval_set=[(x_test,y_test)],verbose=False)
y_pred=model.predict(x_test)

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual Price")
plt.plot(y_test.index, y_pred, label="Predicted Price", linestyle="--")
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.title(f"{stock_symbol} Actual vs Predicted Stock Price")
plt.legend()
plt.show()

print(pd.DataFrame({"Actual: ":y_test,"predictred":y_pred}))
print(f"Mean squared error: {mean_squared_error(y_test,y_pred)}")
print(f"Root Mean squared error: {root_mean_squared_error(y_test,y_pred)}")
print(f"r2 score: {r2_score(y_test,y_pred)}")
mae=mean_absolute_error(y_test,y_pred)
print(f"mean absolute error is: {mae}")

# data.to_csv(f"{stock_symbol}_stock_data_with_features.csv")

plt.figure(figsize=(12,6))
plt.plot(data.index, data["Close"], label="Closing Price")
plt.plot(data.index, data["SMA_50"], label="50-Day SMA", linestyle="--")
plt.plot(data.index, data["SMA_200"], label="200-Day SMA", linestyle="--")
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.title(f"{stock_symbol} Stock Price with Moving Averages")
plt.legend()
plt.show()
# plt.savefig("Apple_stock")

model_path = os.path.join(BASE_DIR,"model", "apple_stock_model.pkl")
joblib.dump(model.best_estimator_, model_path)
scaler_path = os.path.join(BASE_DIR,"model", "scaler.pkl")
joblib.dump(scaler, scaler_path)
# DATA_PATH = os.path.join(BASE_DIR, "data", "boston.csv")
# MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.joblib")
# SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")