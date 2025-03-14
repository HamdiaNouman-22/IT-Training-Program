import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from patsy.test_state import test_Center
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import streamlit as st
stock_sym="AAPL"
df=yf.download(stock_sym,start="2023-01-01",end="2024-01-01")
df.reset_index(inplace=True)
print(df.head())
print(df.info())
df["5_day_avg"]=df["Close"].rolling(window=5).mean()
df["10_day_avg"]=df["Close"].rolling(window=10).mean()
df["prev_day_closing"]=df["Close"].shift(1)
df["Return"]=df["Close"].pct_change()

df.dropna(inplace=True)
X=df[["5_day_avg","10_day_avg","prev_day_closing","Return"]]
y=df["Close"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

r2=r2_score(y_test,y_pred)
n=X_test.shape[0]
p=X_test.shape[1]
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adj_r2:.4f}")
X_test=X_test.reset_index()
print(X_test.index)
print(df.index)

test_dates = df.iloc[X_test.index]["Date"]
y_test_actual = df.iloc[test_dates.index]["Close"]

st.title(f"{stock_sym} Stock Price Prediction")
fig,ax=plt.subplots(figsize=(10,5))
ax.plot(test_dates,y_test_actual,label="Actual Price",color='blue')
ax.plot(test_dates,y_pred,label="Predicted Price",color='red')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price')
ax.legend()
st.pyplot(fig)
st.write(f"R-squared: {r2:.4f}")
st.write(f"Adjusted R-squared: {adj_r2:.4f}")