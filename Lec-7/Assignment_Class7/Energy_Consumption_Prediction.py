import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
from sklearn.linear_model import SGDRegressor

df=pd.read_csv("owid-energy-data.csv")
print(df.info())
df.dropna(inplace=True)

print(df.isnull().sum())

energy_columns = [col for col in df.columns if "consumption" in col.lower()]
df.dropna(inplace=True)
if not energy_columns:
    raise KeyError("No column related to energy consumption found in the dataset.")

    # Use the first identified energy consumption column
energy_column = energy_columns[0]
    # One-hot encode categorical variables
plt.figure(figsize=(10, 5))
sns.lineplot(x=df.index, y=df[energy_column])
plt.title("Energy Consumption Over Time")
plt.xlabel("Time")
plt.ylabel("Energy Consumption")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(y=df[energy_column], color="red")
plt.title("Boxplot of Energy Consumption")
plt.ylabel("Energy Consumption")
plt.show()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define features and target variable
X = df.drop(columns=[energy_column])
y = df[energy_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse} R2: {r2}')

sgd_model = SGDRegressor(max_iter=1000, learning_rate='optimal', eta0=0.01)
sgd_model.fit(X_train, y_train)
y_pred_sgd = sgd_model.predict(X_test)

st.title('Energy Consumption Prediction')
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r',
lw=2)
ax.set_xlabel('Actual Energy Consumption')
ax.set_ylabel('Predicted Energy Consumption')
st.pyplot(fig)
st.sidebar.header('Predict Energy Consumption')
features = {col: st.sidebar.number_input(f'Enter {col}:',
                                         float(df[col].min()), float(df[col].max())) for col in X.columns}

if st.sidebar.button('Predict'):
    input_data = np.array([features[col] for col in X.columns]).reshape(1,
                                                                        1)
    prediction = model.predict(input_data)[0]
    st.sidebar.write(f'Predicted Energy Consumption: {prediction:.2f} kWh')