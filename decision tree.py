# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:36:00 2018

@author: thirumav
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from os import system

iris=load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=3)
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of decision tree classifier on training set: {:.2f}' .format(clf.score(X_train, y_train)))
print('Accuracy of decision tree classifier on test set: {:.2f}' .format(clf.score(X_test, y_test)))

#Setting max decision tree depth to avoid overfitting
clf2 = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)
print('Accuracy of decision tree classifier on training set: {:.2f}' .format(clf2.score(X_train, y_train)))
print('Accuracy of decision tree classifier on test set: {:.2f}' .format(clf2.score(X_test, y_test)))

tree.export_graphviz(clf, out_file = 'decisiontree.dot')
