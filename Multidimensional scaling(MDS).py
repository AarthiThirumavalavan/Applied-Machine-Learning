# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:54:59 2018

@author: thirumav
"""
#from adspy_shared_utilities import plot_labelled_scatter
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pandas as pd

fruits = pd.read_table('C:/Users/thirumav/Desktop/Python_exercises/fruit_data_with_colors.txt')
feature_names_fruits = ['height','width','mass','color_score']
X_fruits=fruits[feature_names_fruits]
y_fruits =fruits['fruit_label']

X_fruits_normalized = StandardScaler().fit(X_fruits).transform(X_fruits)

mds= MDS(n_components = 2)

X_fruits_mds = mds.fit_transform(X_fruits_normalized)

plt.scatter(X_fruits_mds, y_fruits, ['apple', 'mandarin', 'orange', 'lemon'])
plt.xlabel('First MDS feature')
plt.ylabel('Second MDS feature')
plt.title("Fruit sample dataset MDS")
