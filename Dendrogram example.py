# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:57:04 2018

@author: thirumav
"""

from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

X, y = make_blobs(random_state= 10, n_samples= 10)
plt.figure()
dendrogram(ward(X))
plt.show()

