#%%
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
from sklearn.linear_model import SGDRegressor

data=pd.read_csv('house_data.csv')
data.info()
print("\n",data.isnull().sum())
#no missing value
non_numeric_columns=data.select_dtypes(include=['object']).columns.tolist()
if(non_numeric_columns):
    dataset=pd.get_dummies(data,columns=non_numeric_columns,drop_first=True)
X=dataset.drop(columns=['price'])
y=dataset['price']
if not np.issubdtype(X.dtypes.values[0], np.number):
    raise ValueError("Some features are still non-numeric. Check the dataset preprocessing.")
print("\n",X.dtypes)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
#we are using x_test for samples and predictors because R² and Adjusted R² is use to reflect how well the model performs on unseen data
#sample size
n=X_test.shape[0]
predictors=X_test.shape[1]

r2_adj = 1 - ((1 - r2) * (n - 1) / (n - predictors - 1))
print(f'\nModel Evaluation:\nMSE: {mse:.2f}, R-squared: {r2:.2f}, Adjusted R-squared: {r2_adj:.2f}')