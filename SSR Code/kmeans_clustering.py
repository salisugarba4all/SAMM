# K-Means Clustering

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the mall dataset with pandas

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# Using the elbow method to find the optimal number of clusters

from sklearn.cluster import KMeans
wcss =[]
for i in range (1,8):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter =300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the graph to visualize the Elbow Method to find the optimal number of cluster  
plt.plot(range(1,8),wcss)
#plt.title('The Elbow Method')
plt.xlabel('Number of categories')
plt.ylabel('Within-category sums of squares (WCSS)')
plt.show()

# Applying KMeans to the dataset with the optimal number of cluster

kmeans=KMeans(n_clusters= 8, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
  

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1],s = 50, c='red', label = 'Messaging')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1],s = 50, c='blue', label = 'Video')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2,1],s = 50, c='green', label = 'Mapping')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3,1],s = 50, c='pink', label = 'Email')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4,1],s = 50, c='purple', label = 'Social')

plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5,1],s = 50, c='yellow', label = 'Photos')

plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6,1],s = 50, c='orange', label = 'Financial')

plt.scatter(X[y_kmeans == 7, 0], X[y_kmeans == 7,1],s = 50, c='grey', label = 'Games')

#plt.scatter(X[y_kmeans == 8, 0], X[y_kmeans == 8,1],s = 50, c='lime', label = 'Cluster 9')

#plt.scatter(X[y_kmeans == 9, 0], X[y_kmeans == 9,1],s = 50, c='skyblue', label = 'Cluster 10')

#plt.scatter(X[y_kmeans == 10, 0], X[y_kmeans == 10,1],s = 50, c='Cyan', label = 'Cluster 11')

#plt.scatter(X[y_kmeans == 11, 0], X[y_kmeans == 11,1],s = 50, c='violet', label = 'Cluster 12')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100, c = 'black', label = 'Centroids')

    
#plt.title('Clusters of clients')
plt.xlabel('Space vector (X)')
plt.ylabel('Space vector (Y)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

