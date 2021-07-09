 #importing the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#using the dendogram to find the optimal number of cluster
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('dendogram')
plt.xlabel('customers')
plt.ylabel('euclidean distances')
plt.show()

#training the hierarchial clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage ='ward' )
y_hc = hc.fit_predict(X)

#visualising the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1], s = 100, c = 'red', label='Cluster 1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1], s = 100, c = 'blue', label='Cluster 2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1], s = 100, c = 'green', label='Cluster 3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1], s = 100, c = 'black', label='Cluster 4')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1], s = 100, c = 'purple', label='Cluster 5')

plt.title('Cluster of customers')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.show()