import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 

df = pd.read_excel("OnlineRetail.xlsx")

df = df.dropna()

df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

customer_data = df.groupby("CustomerID").agg({
    "Quantity":"sum",
    "TotalAmount":"sum",
    "InvoiceNo":"nunique"
}).reset_index()

customer_data.columns = [
    "customerID",
    "totalQuantity",
    "totalSpending",
    "purchaseFrequency"
]

customer_data["Segment"] = pd.qcut(
    customer_data["TotalSpending"],
    q=3,
    label = ["Low","Medium","High"]
)

x = customer_data[["TotalQuantity","TotalSpending","PurchaseFrequency"]]

y = customer_data["Segment"]

encoder = LabelEncoder()

y = encoder.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    random_state = 42
)

model = DecisionTreeClassifier(
    criterion = 'entropy',
    random_state = 42
)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

importance = pd.DataFrame({
    "Feature":x.columns,
    "importance": model.feature_importances_
})

print("\nFeature importance")
print(importance)

plt.figure(figsize=(16,8))

plot_tree(
    model,
    feature_names = x.columns,
    class_names = encoder.classes_,
    filled = True 
)

plot.show()
