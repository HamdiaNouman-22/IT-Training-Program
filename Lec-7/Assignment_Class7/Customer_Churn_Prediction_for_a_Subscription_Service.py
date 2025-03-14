import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
from sklearn.linear_model import SGDRegressor

df = pd.read_csv('E-commerce.csv')
print(df.head())
print(df.info())

# Churn: Customers with low site time & no purchase history
df['Churn'] = ((df['Time on Site'] < df['Time on Site'].quantile(0.25)) &
               (df['Purchase History'] == 'None')).astype(int)

features = ['Age', 'Annual Income', 'Time on Site']
df = df[features + ['Churn']]

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

X = df[features]
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}, R-squared: {r2}")

sgd_reg = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
sgd_reg.fit(X_train, y_train)
y_pred = sgd_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:\nMSE: {mse:.4f}\nR² Score: {r2:.4f}")

st.title("Customer Churn Alert System")
churn_threshold = 0.6  #threshold for potential churners
high_risk_customers = X_test[y_pred >= churn_threshold]
high_risk_customers["Churn Probability"] = y_pred[y_pred >= churn_threshold]

st.subheader("High-Risk Customers")
if not high_risk_customers.empty:
    st.dataframe(high_risk_customers)
    st.error("⚠️ These customers are likely to churn!")
else:
    st.success("No high-risk customers detected!")

