# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:45:35 2018

@author: thirumav
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
fruits = pd.read_table('C:/Users/thirumav/Desktop/Python_exercises/fruit_data_with_colors.txt')
feature_names_fruits = ['height','width','mass','color_score']
X_fruits=fruits[feature_names_fruits]
y_fruits =fruits['fruit_label']
X_fruits_2d = fruits[['height','width']]
y_fruits_2d = fruits['fruit_label']
fig, subaxes = plt.subplots(1,1,figsize=(7,5))
y_fruits_apple = y_fruits_2d ==1
X_train, X_test, y_train, y_test = (train_test_split(X_fruits_2d.as_matrix(), y_fruits_apple.as_matrix(), random_state=0))

clf = LogisticRegression(C=100).fit(X_train, y_train)
plt.clf()
plt.scatter(X_train, y_train, color='blue')
plt.show()
h=6
w=8
print("A fruit with height{} and width{} is predicted to be:{}" .format(h,w,['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))
h=10
w=2
print("A fruit with height{} and width{} is predicted to be:{}" .format(h,w,['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))

plt.show()