# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:12:19 2018

@author: thirumav
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

diabetes = datasets.load_diabetes()

X,y = diabetes.data[:, None, 6], diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lm = LinearRegression().fit(X_train, y_train)
lm_dummy_mean = DummyRegressor(strategy = 'mean').fit(X_train, y_train)

y_predict = lm.predict(X_test)
y_predict_dummy_mean = lm_dummy_mean.predict(X_test)

print('Linear model, coefficients: ', lm.coef_)
print('Mean squared error (dummy): {:.2f}' .format(mean_squared_error(y_test, y_predict_dummy_mean)))
print('Mean squared error (Linear model): {:.2f}'.format(mean_squared_error(y_test, y_predict_dummy_mean)))

print('R2 score (dummy): {:.2f}' .format(r2_score(y_test, y_predict_dummy_mean)))
print('R2 score (Linear model): {:.2f}'.format(r2_score(y_test, y_predict_dummy_mean)))

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_predict, color='green', linewidth=2)
plt.plot(X_test, y_predict_dummy_mean, color='red', linestyle = 'dashed', linewidth=2)
plt.show()
