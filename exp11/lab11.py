import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

digits = load_digits()
x = digits.data 

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
k_values = range(2,11)

inertia = []
silhouette = []

print("k\tInertia\tSilhouette Score")
for k in k_values:
    model = KMeans(n_clusters = k, random_state = 42,n_init = 10)
    labels = model.fit_predict(x_scaled)
    inertia.append(model.inertia_)
    score = silhouette_score(x_scaled,labels)
    silhouette.append(score)
    print(f"{k}\t{model.inertia_:.2f}\t{score:.4f}")

plt.figure(figsize = (10.6))
plt.plot(k_values,inertia,marker = 'o')
plt.title("Elbow method")
plt.xlabel
