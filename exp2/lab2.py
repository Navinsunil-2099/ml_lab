import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

url = "https://raw.githubusercontent.com/selva86/datasets/master/Auto.csv"

df = pd.read_csv(url)

df = df[["displacement","mpg"]]
df = df.dropna()

x = df["displacement"].values.reshape(-1,1)
y = df["mpg"].values 

linear = LinearRegression()
linear.fit(x,y)

y_pred_linear = linear.predict(x)
print("\nLinearRegression")
print("mse: ",mean_squared_error(y,y_pred_linear))
print("r2: ",r2_score(y,y_pred_linear))

plt.figure(figsize = (10,6))
plt.scatter(x,y,label = "data")

x_plot = np.linspace(x.min(),x.max(),300).reshape(-1,1)

plt.plot(
    x_plot,
    linear.predict(x_plot),
    color='red',
    linewidth = 2,
    label = 'Linear'
)

degrees = [2,3,4]

for d in degrees:
    poly = PolynomialFeatures(degree = d)
    x_poly = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly,y)

    y_pred = model.predict(x_poly)

    print("\nPolynomial Degree", d)
    print("MSE :", mean_squared_error(y, y_pred))
    print("R2  :", r2_score(y, y_pred))

    x_poly = poly.transform(x_poly)

    plt.plot(
        x_plot,
        model.predict(x_poly),
        linewidth=2,
        label = f"Degree {d}"
    )

plt.xlabel("displacement")
plt.ylabel("mpg")
plt.title("polynomial regression vs linear regression graph")
plt.legend()
plt.grid(True)
plt.show()
