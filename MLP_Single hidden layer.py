# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:58:56 2018

@author: thirumav
"""

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from adspy_shared_utilities import plot_class_regions_for_classifier

X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2,
                       centers = 8, cluster_std = 1.3,
                       random_state = 4)
X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)
fig, subaxes = plt.subplots(3, 1, figsize=(6,18))
#SINGLE LAYER PERCEPTRON
for units, axis in zip([1,10,100], subaxes):#units specifies the number of hidden layer in the MLP
    #solver: specifies the algorithm to use to learn the weights of the network
    nnclf = MLPClassifier(hidden_layer_sizes=[units], solver='lbfgs', random_state=0).fit(X_train, y_train)#random state used here so that weights initialised are kept constant for all
    print( 'Dataset 1: Neural Network Classifier, 1 layer, {} units'.format(units))
    plot_class_regions_for_classifier_subplot(nnclf, X_train, y_train,
                                             X_test, y_test, axis)
    plt.tight_layout()

#MULTILAYER PERCEPTRON
nnclf = MLPClassifier(hidden_layer_sizes = [10, 10], solver='lbfgs',
                     random_state = 0).fit(X_train, y_train)
plot_class_regions_for_classifier(nnclf, X_train, y_train, X_test, y_test,
                                 'Dataset 1: Neural net classifier, 2 layers, 10/10 units')

#REGULARIZATION PARAMETER: ALPHA
fig, subaxes = plt.subplots(4, 1, figsize=(6,23))
for this_alpha, axis in zip([0.01, 0.1, 1.0, 5.0], subaxes):
    nnclf = MLPClassifier(solver='lbfgs', activation='tanh', alpha= this_alpha, 
                          hidden_layer_sizes= [100, 100], random_state= 0).fit(X_train, y_train)
    title: 'Dataset 2: NN Classifier, alpha = {:.3f}'.format(this_alpha)
    plot_class_regions_for_classifier_subplot(nnclf, X_train, y_train,
                                             X_test, y_test, title, axis)
    plt.tight_layout()

#