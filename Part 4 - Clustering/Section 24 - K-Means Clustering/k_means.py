 #importing the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#using elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Training the k-menas model on the the dataset
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

print(y_kmeans)

#visualising the clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1], s = 100, c = 'red', label='Cluster 1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1], s = 100, c = 'blue', label='Cluster 2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1], s = 100, c = 'green', label='Cluster 3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1], s = 100, c = 'black', label='Cluster 4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1], s = 100, c = 'purple', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], s=300, color = 'yellow', label ='Centroids')
plt.title('Cluster of customers')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.show()