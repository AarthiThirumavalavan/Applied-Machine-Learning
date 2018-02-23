# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:11:03 2018

@author: thirumav
"""
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

fruits=pd.read_table("C:/Users/thirumav/Desktop/fruit_data_with_colors.txt")
feature_names_fruits = ['height', 'width','mass','color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple','mandarin','orange','lemon']
X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state = 0)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
linlasso = Lasso(alpha=2.0, max_iter = 10000).fit(X_train_scaled, y_train)

print('lasso regression linear model intercept: {}'
      .format(linlasso.intercept_))
print('lasso regression linear model coeff:\n{}'
      .format(linlasso.coef_))
print('Non-zero features: {}'
      .format(np.sum(linlasso.coef_ !=0)))
print('R squared score (training):{:.3f}'
      .format(linlasso.score(X_train_scaled, y_train)))
print('R squared score(test):{:.3f}\n'
      .format(linlasso.score(X_test_scaled, y_test)))
print('Features with non-zero weight(sorted by absolute magnitude):')
for e in sorted(list(zip(list(X_fruits), linlasso.coef_)), key = lambda e: -abs(e[1])):
    if e[1] !=0:
        print('\t{}, {:.3f}' .format(e[0],e[1]))