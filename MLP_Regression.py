# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:06:38 2018

@author: thirumav
"""

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)
fig, subaxes = plt.subplots(2, 3, figsize = (11,8))
X_predict_input = np.linspace(-3, 3, 50).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X_R1[0::5], y_R1[0::5], random_state=0)

for thisaxisrow, thisactivation in zip(subaxes, ['tanh','relu']):
    for thisalpha, thisaxis in zip([0.0001, 1.0, 100], thisaxisrow):
        mlpReg = MLPRegressor(activation = thisactivation, random_state =0, 
                              alpha = thisalpha, solver = 'lbfgs').fit(X_train, y_train)
        y_predict_output = mlpReg.predict(X_predict_input)
        thisaxis.set_xlim([-2.5, 0.75])
        thisaxis.plot(X_predict_input, y_predict_output, '^', markersize=10)
        thisaxis.plot(X_train, y_train, 'o')
        thisaxis.set_xlabel('Input features')
        thisaxis.set_ylabel('Target value')
        thisaxis.set_title('MLP regression\nalpha = {}, activation = {}'.format(thisalpha, thisactivation))
        plt.tight_layout()