import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from tensorflow.keras.datasets import fashion_mnist
from sklearn.neighbours import KNeighboursClassifier
from sklearn.metric import accuracy_score,precision_score,recall_score, f1_score 

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

x_train = x_train/255.0
y_train = y_train/255.0

x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)

k_values = [1,3,5,7,9]

result = []

for k in k_values:
    knn = KNeighboursClassifier(n_neighbours=k)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred,average='weighted')
    recall = recall_score(y_test,y_pred,average='weighted')
    f1 = f1_score(y_test,y_pred,average='weighted')

    results.append([
        k,accuracy,precision,recall,f1 
    ])

    #printing result of current k 
    print("\n k = {k}")
    print("Accuracy:",accuracy)
    print("Precision:",precision)
    print("recall:",recall)
    print("F1:",f1)

results_df = pd.DataFrame({
    results,
    columns = [
        k,
        accuracy,
        precision,
        recall,
        f1
    ]
})

print("\nResults summary")
print(results_df)

plt.figure(figsize=(8,5))

plt.plot(
    results_df['k'],
    results.df['Accuracy'],
    marker='o'
)

plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("knn accuracy vs k")
plt.grid(True)
plt.show()

