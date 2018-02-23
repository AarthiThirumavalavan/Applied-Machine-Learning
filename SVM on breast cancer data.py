# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:48:55 2018

@author: thirumav
"""

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#Without Normalization
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

clf = SVC(C=10).fit(X_train, y_train)
print('Breast cancer dataset(unnormalized features)')
print('Accuracy of RBF kerbel SVC on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of RBF kerbel SVC on test set: {:.2f}'.format(clf.score(X_test, y_test)))

#WitH Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = SVC(C=10).fit(X_train_scaled, y_train)
print('Breast cancer dataset(unnormalized features)')
print('Accuracy of RBF kerbel SVC on training set(With MinMax scaling): {:.2f}'.format(clf.score(X_train_scaled, y_train)))
print('Accuracy of RBF kerbel SVC on test set(With MinMax scaling): {:.2f}'.format(clf.score(X_test_scaled, y_test)))