import pandas as pd 
import numpy as np 

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

columns = [
    "pregnancies",
    "Glucose",
    "Bloodpressure",
    "skinthickness",
    "insulin",
    "bmi",
    "diabetespedigreefuncion",
    "age",
    "outcome"
]

df = pd.read_csv(
    "https link"
    names = columns
)

for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    df[col] = df[col].replace(0,np.nan)
    df[col] = df[col].fillna(df[col].median())

x = df.drop("Outcome",axis=1)
y = df["Outcome"]

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    random_state = 42
)

model = LogisticRegression(max_iter = 1000)
model.fit(x_train,y_train)
pred = model.predict(x_test)

print("without scaling")
print("Accuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall   :", recall_score(y_test, pred))
print("F1 Score :", f1_score(y_test, pred))

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model_scaled = LogisticRegression(max_iter = 1000)
model_scaled.fit(x_train_scaled,y_train)
y_pred_scaled = model_scaled.predict(x_test_scaled)

print("\nwith scaling")
print("Accuracy :", accuracy_score(y_test, y_pred_scaled))
print("Precision:", precision_score(y_test, y_pred_scaled))
print("Recall   :", recall_score(y_test, y_pred_scaled))
print("F1 Score :", f1_score(y_test, y_pred_scaled))

result = pd.DataFrame({
    "Metric":["Accuracy","precision","recall","f1_score"],
    "without scaling":[
        accuracy_score(y_test,y_pred),
        precision_score(y_test,y_pred),
        recall_score(y_test,y_pred),
        f1_score(y_test,y_pred)
    ],
    "with scaling":[
        accuracy_score(y_test, y_pred_scaled),
        precision_score(y_test, y_pred_scaled),
        recall_score(y_test, y_pred_scaled),
        f1_score(y_test, y_pred_scaled)
    ]
})

print("\n comparison ")
print(result)
























