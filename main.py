import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
import pickle

df=pd.read_csv('loan_dataset.csv')

df.columns=df.columns.str.strip()
#print(df.head())
#print(df.info())

df.isnull().sum()
df=df.ffill()

df['loan_status'] = df['loan_status'].astype(str).str.strip().str.lower()
df['loan_status'] = df['loan_status'].map({'approved': 1, 'rejected': 0})

le = LabelEncoder()

for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = le.fit_transform(df[col])

df = df.drop('loan_id', axis=1)

X = df.drop("loan_status", axis=1)
y = df["loan_status"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = HistGradientBoostingClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)
print(classification_report(y_test, y_pred))

pickle.dump(model, open("model.pkl", "wb"))
print("Model saved successfully")