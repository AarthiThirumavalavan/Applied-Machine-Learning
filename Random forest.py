# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:11:22 2018

@author: thirumav
"""
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
#FRUIT DATASET
fruits = pd.read_table('C:/Users/thirumav/Desktop/Python_exercises/fruit_data_with_colors.txt')
feature_names_fruits = ['height','width','mass','color_score']
X_fruits=fruits[feature_names_fruits]
y_fruits =fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X_fruits.as_matrix(), y_fruits.as_matrix(), random_state= 0)
fig, subaxes = plt.subplots(6,1, figsize=(6,32))

title = 'Random forest, fruits dataset, default settings'
pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]] #feature pair list

#For each pair of feature in pair_list, we call the fit method on that subset of training data X using the labels y
for pair, axis in zip(pair_list, subaxes):
    X = X_train[:, pair]
    y = y_train
    clf=RandomForestClassifier().fit(X,y)
    plot_decision_regions(X, y, clf,
                      res=0.02, legend=2)
    plt.xlabel(feature_names_fruits[pair[0]])
    plt.ylabel(feature_names_fruits[pair[1]])
    plt.title('Random forest on Iris')
    plt.show()

#BREAST CANCER DATASET
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

clf = RandomForestClassifier(max_features = 8, random_state =0)
clf.fit(X_train, y_train)

print('Breast cancer dataset')
print('Accuracy of RF classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))


