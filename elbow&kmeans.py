#elbow method

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("YOUR_FILE_PATH/YOUR_FILE.csv")  
# 🔴 CHANGE FILE PATH

X = df.select_dtypes(include='number')

inertia = []

for k in range(1, 11):  
    # 🔴 CHANGE RANGE IF NEEDED
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()



#Apply KMeans

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)  
# 🔴 CHANGE NUMBER OF CLUSTERS

df["Cluster"] = kmeans.fit_predict(X)

print(df.head())