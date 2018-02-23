# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:14:36 2018

@author: thirumav
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
crime = pd.read_table('C:/Users/thirumav/Desktop/Python_exercises/CommViolPredUnnormalizedData.txt', sep=',', na_values='?')
columns_to_keep = [5, 6] + list(range(11,26)) + list(range(32, 103)) + [145]
crime = crime.ix[:,columns_to_keep].dropna()
X_crime = crime.ix[:,range(0,88)]
y_crime=crime['ViolentCrimesPerPop']