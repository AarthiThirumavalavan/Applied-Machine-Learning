# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:18:11 2018

@author: thirumav
"""

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X,y = make_blobs(random_state =0, n_samples = 25)

dbscan = DBSCAN(eps=2, min_samples = 2)

cls = dbscan.fit_predict(X)#fits and gets the cluster assignments in one step
print("Cluster membership values:\n{}" . format(cls))#-1 represents noise

plt.scatter(X, cls+1)

