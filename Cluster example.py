# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:15:47 2018

@author: thirumav
"""
#KMEANS
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from matplotlib import pyplot as plt

X, y = make_blobs(random_state= 10)

kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)

plt.scatter(X, kmeans.labels_)

#AGGLOMERATIVE CLUSTERING
cls = AgglomerativeClustering(n_clusters = 3)
cls_assignment = cls.fit_predict(X)

X, y = make_blobs(random_state = 0)
plt.scatter(X, cls_assignment)
