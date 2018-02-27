# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:59:27 2018

@author: thirumav
"""
#We apply grid search here to explore the different values of the optional class weight parameter
#that controls how much weight is given to each of the two classes during training
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

dataset = load_digits()
X, y =dataset.data, dataset.target == 1
X_train, X_test, y_train, y_test = train_test_split (X, y,random_state = 0)

#Create a two-feature input vector
#We jitter the points(add a sall amount of random noise) in case there are areas in the feature space where many instances have the same features

jitter_delta = 0.25
X_twovar_train = X_train [:, [20,59]] + np.random.rand(X_train.shape[0], 2) - jitter_delta #random.rand : populates random samples from a uniform distribution over [0,1]
X_twovar_test = X_test [:, [20,59]] + np.random.rand(X_test.shape[0], 2) - jitter_delta #random.rand : populates random samples from a uniform distribution over [0,1]
clf = SVC(kernel='linear').fit(X_twovar_train, y_train)
grid_values ={'class_weight':['balanced', {1:2}, {1:3},{1:4},{1:5},{1:10},{1:20},{1:50}]}
plt.figure(figsize = (9,6))

for i, eval_metric in enumerate(('precision','recall','f1','roc_auc')):
    grid_clf_custom = GridSearchCV(clf, param_grid = grid_values, scoring = eval_metric)
    grid_clf_custom.fit(X_twovar_train, y_train)
    print('Grid best parameter (max.{0}:{1}' .format(eval_metric, grid_clf_custom.best_params_))
    print('Grid best score ({0}):{1}' .format(eval_metric, grid_clf_custom.best_score_))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.title(eval_metric +'-oriented SVC')
    #plot_class_regions_for_classifier_subplot(grid_clf_custom, X_twovar_test, y_test, None,
                                             #None, None,  plt.subplot(2, 2, i+1))

plt.tight_layout()
plt.show()
    
