#Import
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


#Functions
def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    index = np.random.randint(0, m, k)
    
    for i in range(k):
        centroids[i,:] = X[index[i],:]
    
    return centroids

def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    index = np.zeros(m)
    
    
    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i,:] - centroids[j,:]) ** 2)
            if dist < min_dist:
                min_dist = dist
                index[i] = j
    
    return index

def compute_centroids(X, index, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    
    for i in range(k):
        indices = np.where(index == i)
        centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()
    
    return centroids

def k_means(X, k, max_iter):
    m, n = X.shape
    index = np.zeros(m)
    centroids =  init_centroids(X, k)

    for i in range(max_iter):
        index = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, index, k)
    
    return index, centroids


#Data
data = loadmat('ex7data2.mat')
X = data['X']


#Run k_means
index, centroids = k_means(X, 3, 10)


#Graph
cluster1 = X[np.where(index == 0)[0],:]
cluster2 = X[np.where(index == 1)[0],:]
cluster3 = X[np.where(index == 2)[0],:]

fig, ax = plt.subplots(figsize=(9,6))
ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')
ax.scatter(centroids[0,0],centroids[0,1],s=300, color='r')

ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
ax.scatter(centroids[1,0],centroids[1,1],s=300, color='g')

ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
ax.scatter(centroids[2,0],centroids[2,1],s=300, color='b')
ax.legend()