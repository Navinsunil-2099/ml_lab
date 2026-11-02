import pandas as pd 
import numpy as np 

from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

url = "https link"

df = pd.readcsv(url)

df = df[["mpg","displ"]].dropna()

x = df[["displ"]]
y = df["mpg"]

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    random_state = 42
)

linear = LinearRegression()
linear.fit(x_train,y_train)

y_pred_linear = linear.predict(y_test)

print("linear regression")
print("mse:", mean_squared_error(y_test,y_pred_linear))
print("r2:", r2_score(y_test,y_pred_linear))

for degree in [2,3,4]:
    poly = PolynomialFeatures(degree = degree)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    model = LinearRegression()
    model.fit(x_train_poly,y_train)
    y_pred_poly = model.predict(x_test_poly)

    print("\nPolynomial Degree:",degree)
    print("mse:", mean_squared_error(y_test,y_pred_poly))
    print("r2:", r2_score(y_test,y_pred_poly))

x_plot = np.linspace(x_min(), x.max(), 100).reshape(-1,1)
plt.scatter(x,y,label = "Actual Data")

plt.plot(
    x_plot,
    linear.predict(x_plot),
    label = "Linear"
)

for degree in [2,3,4]:
    poly = PolynomialFeatures(degree = degree)
    model = LinearRegression()
    x_poly = poly.fit_transform(x)
    model.fit(x_poly,y)
    plt.plot(x_plot, model.predict(poly.transform(x_plot)), label = f"Polynomial Degree {degree}")

plt.xlabel("Engline Displacement")
plt.ylabel("mpg")
plt.title("linear vs polynomial")
plt.legend()
plt.show()

