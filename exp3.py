import pandas as pd 

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision,recall_score,f1_score 

data = load_breast_cancer()

x = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size = 0.2,random_state = 42
)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

mle = LogisticRegression(c = 1e6, max_iter = 5000)
l2 = LogisticRegression(c=1,penalty="l2",max_iter=5000)
l1 = LogisticRegression(c=1,penalty="l1",max_iter=5000)

mle.fit(x_train,y_train)
l2.fit(x_train,y_train)
l1.fit(x_train,y_train)

pred_mle = mle.predict(x_test)
pred_l2 = l2.predict(x_test)
pred_l1 = l1.predict(x_test)

def evaluate(model,prediction):
    print("\n",model)
    print("Accuracy:",accuracy_score(y_test,prediction))
    print("Precision:",precision(y_test,prediction))
    print("Recall:",recall_score(y_test,prediction))
    print("f1_score:",f1_score(y_test,prediction))

evaluate("Mle",pred_mle)
evaluate("map l2",pred_l2)
evaluate("map l1",pred_l1)

print("\nCoefficient Comparison")

nCoefficient = pd.DataFrame({
    "MLe":mle.coef_[0],
    "map l2": l2.coef_[0],
    "map l1": l1.coef_[0],
})

print(nCoefficient.head(10))

