# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:56:27 2018

@author: thirumav
"""

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

dataset = datasets.load_digits()
X,y = dataset.data, dataset.target == 1 #checks if target is equal to 1 or not
#Cross validation for SVM
clf =SVC(kernel = 'linear', C=1)

print('Cross validation(accuracy)', cross_val_score(clf, X, y, cv=5 ))
print('Cross validation (AUC)', cross_val_score(clf, X, y, cv=5, scoring = 'roc_auc'))#Area under the curve of ROC 
print('Cross validation(recall)', cross_val_score(clf, X, y, cv=5, scoring ='recall'))

#GridSearchCV
X_train, X_test, y_train, y_test =train_test_split(X, y, random_state =0)
clf =SVC(kernel='rbf')
grid_values = {'gamma' :[0.001, 0.01, 0.05, 0.1, 1, 10, 100]}
#GridSearch using Accuracy: default metric to optimze over grid parameters
grid_clf_acc =GridSearchCV(clf, param_grid = grid_values)
grid_clf_acc.fit(X_train, y_train)
y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test)#decision_function() gives per-class score for each sample

print('Grid best parameter (max.accuracy):', grid_clf_acc.best_params_)
print('Grid best score(accuracy):', grid_clf_acc.best_score_)
#GridSearch using AUC: alternative metric to optimize over grid parameters
grid_clf_auc = GridSearchCV(clf, param_grid= grid_values, scoring = 'roc_auc')
grid_clf_auc.fit(X_train, y_train)
y_decision_fn_scores_auc = grid_clf_auc.decision_function(X_test)

print('Test set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))
print('Grid best parameter(max. AUC):' , grid_clf_auc.best_params_)
print('Grid best score(AUC:)', grid_clf_auc.best_score_)

#Evaluation metrics supported for model selection
from sklearn.metrics.scorer import SCORERS
print(sorted(list(SCORERS.keys())))
