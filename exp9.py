import numpy as np 
import pandas as pd 
from tensorflow.keras.datasets import fashion_mnist 
from sklearn.svm import svc 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0
x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)
x_train = x_train[:10000]
y_train = y_train[:10000]

x_test = x_test[:10000]
y_test = y_test[:10000]

linear_svm = svc(kernel = 'linear')

linear_svm.fit(x_train,y_train)
linear_pred = linear_svm.predict(x_test)

poly_svm = svc(kernel = 'poly',degree = 3)
poly_svm.fit(x_train,y_train)
poly_pred = poly_svm.predict(x_test)

rbf_svm = svc(kernel = 'rbf')
rbf_svm.fit(x_train,y_train)
rbf_pred = rbf_svm.predict(x_test)

def evaluate_model(y_true,y_pred,model_name):
    accuracy = accuracy_score(y_true,y_pred)
    precision = precision_score(y_true,y_pred,average = 'weighted')
    recall = recall_score(y_true,y_pred,average = 'weighted')
    f1 = f1_score(y_true,y_pred,average = 'weighted')

    print("\n",model_name)
    print("Accuracy:",accuracy)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F1 score:",f1)

evaluate_model(y_test,linear_pred,"Linear svm")
evaluate_model(y_test,poly_pred,"polynomial svm")
evaluate_model(y_test,rbf_svm,"rbf svm")
