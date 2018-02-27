# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:08:27 2018

@author: thirumav
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
#BLOBS DATASET
X_D2, y_D2 = make_blobs(n_samples = 100, n_features =2, centers=8, cluster_std = 1.3, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state =0)

fig, subaxes = plt.subplots(1, 1, figsize=(6,6))
clf = GradientBoostingClassifier().fit(X_train, y_train)
title = 'GBDT, complex binary dataset, default settings'
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, X_test, y_test, title, subaxes)
plt.show()

#BREAST CANCER DATASET
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

clf1 = GradientBoostingClassifier(random_state=0)
clf1.fit(X_train, y_train)

print('Breast cancer dataset(learning_rate = 0.1, max_depth = 3)')
print('Accuracy of GBDT classifier on training set: {:.2f}'.format(clf1.score(X_train, y_train)))
print('Accuracy of GBDT classifier on test set: {:.2f}'.format(clf1.score(X_test, y_test)))

clf2 = GradientBoostingClassifier(learning_rate = 0.01, max_depth=2, random_state=0)
clf2.fit(X_train, y_train)

print('Breast cancer dataset (learning_rate = 0.01, max_depth=2)')
print('Accuracy of GBDT classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of GBDT classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))