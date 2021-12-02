import pandas as pd
from sklearn.cluster import KMeans
import csv
import numpy as np
import  matplotlib.pyplot as plt

df = pd.read_csv("Processed_data/star_with_gravity.csv")
print(df.head)
X=df.iloc[:,[3,4]].values
wcss=[]
for i in range(1,11):
    Kmeans=KMeans(n_clusters=i,init="k-means++",random_state=42)
    Kmeans.fit(X)
    wcss.append((Kmeans.inertia_))

plt.plot(range(1,11),wcss)
plt.title("elbowmethod")
plt.xlabel("number of clusters")
plt.show()
print(X)