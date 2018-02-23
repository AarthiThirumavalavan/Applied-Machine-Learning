# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:54:55 2018

@author: thirumav
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

dataset=load_digits()
X,y = dataset.data, dataset.target

for class_name, class_count in zip(dataset.target_names, np.bincount(dataset.target)):
    print(class_name, class_count)

y_binary_imbalanced = y.copy()
y_binary_imbalanced[y_binary_imbalanced != 1] = 0

print('Original labels:\t', y[1:30])
print('New binary labels:\t', y_binary_imbalanced[1:30])

X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)

from sklearn.svm import SVC
svm=SVC(kernel='rbf', C=1).fit(X_train, y_train)
svm.score(X_test, y_test)
#Confusion Matrix for SVM
svm1=SVC(kernel='linear', C=1).fit(X_train, y_train)
svm1.score(X_test, y_test)
svm_predicted = svm1.predict(X_test)
confusion1 = confusion_matrix(y_test, svm_predicted)
print('Support vector machine classifier (linear kernel, C=1)\n', confusion1)

#Confusion Matrix for Logistic Regression
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression().fit(X_train,y_train)
lr.score(X_test,y_test)
lr_predicted = lr.predict(X_test)
confusion2 = confusion_matrix(y_test, lr_predicted)
print('Logistic Regression\n', confusion2)

#Confusion Matrix for Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier().fit(X_train, y_train)
dt.score(X_test,y_test)
dt_predicted = dt.predict(X_test)
confusion3 = confusion_matrix(y_test, dt_predicted) 
print('Decision tree\n', confusion3)

#Dummy Classifiers

from sklearn.dummy import DummyClassifier
dummy_majority= DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
y_dummy_predictions = dummy_majority.predict(X_test)

dummy_majority.score(X_test,y_test)

#Confusion Matrix(two-class)
#Most frequent strategy
from sklearn.metrics import confusion_matrix
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
y_majority_predicted = dummy_majority.predict(X_test) 
confusion = confusion_matrix(y_test, y_majority_predicted)
print("Most frequent class(dummy classifier)\n", confusion)

#Stratified strategy
from sklearn.metrics import confusion_matrix
dummy_stratified = DummyClassifier(strategy = 'stratified').fit(X_train, y_train)
y_stratified_predicted = dummy_stratified.predict(X_test)
confusion = confusion_matrix(y_test, y_stratified_predicted)
print("Stratified class(dummy classifier)\n", confusion)

#Evaluation metrics for binary classification

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print('Accuracy:{:.2f}' .format(accuracy_score(y_test, dt_predicted)))
print('Precision:{:.2f}' .format(precision_score(y_test, dt_predicted)))
print('Recall:{:.2f}' .format(recall_score(y_test, dt_predicted)))
print('F1 score:{:.2f}' .format(f1_score(y_test, dt_predicted)))

#Classification reports for all classifiers
#Decision tree
from sklearn.metrics import classification_report
print(classification_report(y_test,dt_predicted, target_names=['Negative','Positive']))
#SVM
print(classification_report(y_test,svm_predicted, target_names=['Negative','Positive']))
#Logistic Regression
print(classification_report(y_test, lr_predicted, target_names=['Negative','Positive']))

#Decision functions
X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
y_scores_lr = lr.fit(X_train, y_train).decision_function(X_test)
y_score_list = list(zip(y_test[0:20], y_scores_lr))
y_score_list

#Predict proba
X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
y_proba_lr = lr.fit(X_train, y_train).predict_proba(X_test)
y_proba_list = list(zip(y_test[0:20], y_proba_lr[0:20,1]))
y_proba_list
