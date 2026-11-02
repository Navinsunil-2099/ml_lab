import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

california = fetch_california_housing(as_frame = True)
df = california.frame 
x = df[['AveRooms']].values
y = df['MedHouseVal'].values.reshape(-1,1)

print(f"Feature Matrix Shape (x): {x.shape}")
print(f"Target Vector Shape (y): {y.shape}\n")

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)


m = len(y)

x_b = np.c_[np.ones((m,1)), x_scaled]

theta_normal_scaled = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_scaled)

y_pred_normal_scaled = x_b.dot(theta_normal_scaled)

y_pred_normal = scaler_y.inverse_transform(y_pred_normal_scaled)

def compute_cost(x,y,theta):
    m = len(y)
    predictions = x.dot(theta)
    cost = (1/(2*m))*np.sum((predictions -y)**2)
    
def gradient_descent(x,y,theta,learning_rate,iterations):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        gradient = (1/m) * x.T.dot(x.dot(theta) - y)
        theta = theta - learning_rate * gradient
        cost_history.append(compute_cost(x,y,theta))

    return cost_history

learning_rate = 0.01
iterations = 1500
initial_theta = np.zeroes((2,1))

theta_gd_scaled,cost_history = gradient_descent(
    x_b,y_scaled,initial_theta,learning_rate,iterations
)

y_pred_gd_scaled = x_b.dot(theta_gd_scaled)

y_pred_gd = scaler_y.inverse_transform(y_pred_gd_scaled)

mse = mean_squared_error(y,y_pred_normal)
r2 = r2_score(y,y_pred_normal)


