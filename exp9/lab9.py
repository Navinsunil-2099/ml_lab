import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.inspection import DecisionBoundaryDisplay

iris = datasets.load_data()
x = iris.data[:,2:4]
y = (iris.target == 0).astype(int)

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.3,
    random_state = 42
)
model = svc(kernel = 'linear', c = 1)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print("Done by Navin Sunil\n")
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("\nClassification Report")
print(classification_report(Y_test, Y_pred))

DecisionBoundaryDisplay.from_estimator(
    model,x,response_method = "decision_function",
    plot_method = "contour", levels = [-1,0,1],
    colors = 'black',linestyles=["--","-","--"]
)

plt.scatter(x[:,0],x[:,1],c=y,cmap="coolwarm")
plt.scatter(
    model.support_vectors_[:,0],
    model.support_vectors_[:,1],
    s = 100,facecolors = "none",edgecolors = "black"
)

plt.xlabel("petal length")
plt.ylabel("petal width")
plt.title("linear svm decision boundary and margin")
plt.show()
