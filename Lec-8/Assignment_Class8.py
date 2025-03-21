import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report


#Sales Forecasting for a Retail Store

data_sales=pd.read_csv("retail_sales_dataset.csv")
print(data_sales.dtypes)
#Performed this step because i got "ValueError: could not convert string to float: 'Female'"
le=LabelEncoder()
data_sales['Gender']=le.fit_transform(data_sales["Gender"])
#Performed this step because i got "ValueError: could not convert string to float: 'Beauty'
data_sales=pd.get_dummies(data_sales,columns=['Product Category'],drop_first=True)
print(data_sales.columns)

X_sales = data_sales[['Gender', 'Age', 'Product Category_Electronics','Product Category_Clothing','Quantity','Price per Unit']]
y_sales = data_sales['Total Amount']
X_train_sales, X_test_sales, y_train_sales, y_test_sales =train_test_split(X_sales, y_sales, test_size=0.2, random_state=42)

print(X_train_sales.dtypes)
model_sales = LinearRegression()
model_sales.fit(X_train_sales, y_train_sales)

y_pred_sales = model_sales.predict(X_test_sales)

mse_sales = mean_squared_error(y_test_sales, y_pred_sales)
r2_sales = r2_score(y_test_sales, y_pred_sales)

print(f"Sales Forecast - MSE: {mse_sales}")
print(f"Sales Forecast - R-squared: {r2_sales}")
plt.scatter(y_test_sales, y_pred_sales)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()



#Email Spam Detection using SVM

df=pd.read_csv("spam (1).csv",encoding='latin-1')
print(df.columns)
df=df.iloc[:, :2]
df.columns=["label","message"]
df['label']=df['label'].map({'ham':0,'spam':1})
nltk.download('stopwords')
ps=PorterStemmer()
corpus=[]
for msg in df['message']:
    msg=re.sub('[^a-aA-Z]',' ',msg).lower().split()
    msg=[ps.stem(word) for word in msg if word not in stopwords.words('english')]
    corpus.append(" ".join(msg))
vectorizer=TfidfVectorizer(max_features=5000)
X=vectorizer.fit_transform(corpus).toarray()
y=df['label'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
svm_model=SVC(kernel='linear')
svm_model.fit(X_train,y_train)
y_pred=svm_model.predict(X_test)
print("Accuracy: ",accuracy_score(y_test,y_pred))



#Customer Churn Prediction using SVM
df=pd.read_csv("churn.csv")
print(df.columns)
le=LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col]=le.fit_transform(df[col])
X=df.drop(['Churn?'],axis=1)
y=df['Churn?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=SVC(kernel='rbf',C=1,gamma='scale')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("Acuuracy: ",accuracy_score(y_test,y_pred))
print("Classification Report: ",classification_report(y_test,y_pred))


#Fraud Detection in Credit Card Transactions
df=pd.read_csv('card_transdata.csv')
df=df.sample(frac=0.1,random_state=42)

X=df.drop(columns=['fraud'])
y=df['fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scalar=StandardScaler()
X_train=scalar.fit_transform(X_train)
X_test=scalar.transform(X_test)

model=SVC(kernel='rbf',C=1,gamma='scale')
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))