import numpy as np
import pandas as pd 
from tensorflow.keras.datasets import fashion.mnist
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classificaiton_report

(X_train,Y_train),(X_test,Y_test) = fashion_mnist.load_data()
X_train = X_train/255.0
X_test = X_test/255.0
X_train = X_train.reshape(len(X_train),-1)
X_test = X_test.reshape(len(len(X_test),-1))
X_train = X_train[:10000]
Y_train = Y_train[:10000]
X_test = X_test[:2000]
Y_test = Y_test[:2000]

linear_svm = SVC(
        kernel = 'linear'
        )
linear_svm.fit(X_train,Y_train)
linear_pred = linear_svm.predict(X_test)

poly_svm = SVC(
        kernel = 'poly',
        degree = 3
        )
poly_svm.fit(X_train,Y_train)
poly_pred = poly_svm.predict(X_test)

rbf_svm = SVC(
        kernel = 'rbf'
        )
rbf_svm.fit(X_train,Y_train)
rbf_pred = rbf_svm.predict(X_test)

def evaluate_model(Y_true,Y_pred,model_name):
    accuracy = accuracy_score(Y_true,Y_pred)
    precision = precision_score(
            Y_true,
            Y_pred,
            average = 'weighted'
            )
    recall = recall_score(
            Y_true,
            Y_pred,
            average = 'weighted'
            )
    f1 = f1_score(
            Y_true,
            Y_pred,
            average = 'weighted'
            )
    print("\n",model_name)
    print("-"*40)
    print("Accuracy:",accuracy)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F1 score:".f1)

evaluate_model(Y_test,linear_pred,"Linear SVM")
evaluate_model(Y_test,poly_pred,"Polynomial SVM")
evaluate_model(Y_test,rbf_pred,"RBF SVM")   
