from sklearn.neighbours import KNeighboursClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from tensorflow.keras.datasets import fashion_mnist

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

#normalizing the pixel values
x_train = x_train/255.0
x_test = x_test/255.0

#reshape the image array , not changing the data, -1 means 28x28
x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)

k_values = [1,3,5,7,9]
results = []

for k in k_values:
    model = KNeighboursClassifier(n_neighbors = k)
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)

    results.append([k,accuracy,precision,recall,f1])

    print("\nk =", k)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

result_df = pd.DataFrame({
    results,
    columns = ["k","Accuracy","precision","recall","f1"]
})

print("results")
print(result_df)

plt.plot(result_df["k"],result["Accuracy"])
plt.xlabel("k value")
plt.ylabel("accuracy")
plt.title("knn k vs accuracy graph")

plt.show()


