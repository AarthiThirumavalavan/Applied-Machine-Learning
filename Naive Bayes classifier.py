# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:06:06 2018

@author: thirumav
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

cmap_bold = ListedColormap(['#FFFF00','#00FF00','#0000FF','#000000'])
plt.figure()
plt.title('Sample binary classification problem with two informative features')
X_C2, y_C2 = make_classification(n_samples = 100, n_features=2,
                                n_redundant=0, n_informative=2,
                                n_clusters_per_class=1, flip_y = 0.1,
                                class_sep = 0.5, random_state=0)
plt.scatter(X_C2[:, 0], X_C2[:, 1], marker= 'o',
           c=y_C2, s=50, cmap=cmap_bold)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state=0)

nbclf = GaussianNB().fit(X_train, y_train)
plot_class_regions_for_classifier(nbclf, X_train, y_train, X_test, y_test,
                                 'Gaussian Naive Bayes classifier: Dataset 1')
print('Accuracy of Gaussian NB classifier on training set: {:.2f}'.format(nbclf.score(X_train, y_train)))
print('Accuracy of Gaussian NB classifier on test set: {:.2f}'.format(nbclf.score(X_test, y_test)))