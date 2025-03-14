import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

df=pd.read_csv("salary_data.csv")
print(df.head())
print(df.info())
df.dropna(inplace=True)

X=df[["YearsExperience"]]
y=df[["Salary"]]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"mse:{mse} r2:{r2}")

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
print(X_test.dtypes)
print(y_test.dtypes)

plt.figure(figsize=(10, 5))
sns.scatterplot(x=X_test["YearsExperience"], y=y_test,color="blue",label="Actual Data")
sns.scatterplot(x=X_train["YearsExperience"], y=y_pred,color="red",label="Predicted Data")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression on Salary Prediction")
plt.legend()
plt.show()

st.title("Salary Prediction System")
year_input=st.number_input("Enter years of experience: ",min_value=int(df['YearsExperience'].min()),max_value=int(df['YearsExperience'].max()),value=2025)
if st.button("Predict"):
    prediction = model.predict(np.array([[year_input]]))[0]
    st.success(f"Predicted Salary: {prediction:.2f} $")

