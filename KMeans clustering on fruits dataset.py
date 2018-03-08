# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:22:32 2018

@author: thirumav
"""

import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
#from adspy_shared_utilities import plot_labelled_scatter
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

fruits = pd.read_table('C:/Users/thirumav/Desktop/Python_exercises/fruit_data_with_colors.txt')
X_fruits = fruits[['mass', 'width', 'height', 'color_score']].as_matrix()
y_fruits = fruits[['fruit_label']] - 1
X_fruits_normalized = MinMaxScaler().fit(X_fruits).transform(X_fruits)

kmeans = KMeans(n_clusters = 4, random_state = 10)
kmeans.fit(X_fruits_normalized)

plt.scatter(X_fruits_normalized, kmeans.labels_)