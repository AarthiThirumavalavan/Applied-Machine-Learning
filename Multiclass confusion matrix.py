# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:24:43 2018

@author: thirumav
"""
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import classification_report

dataset = load_digits()
X, y = dataset.data, dataset.target
X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(X, y, random_state=0)

#Linear kernel
svm = SVC(kernel = 'linear').fit(X_train_mc, y_train_mc)
svm_predicted_mc1 = svm.predict(X_test_mc)
confusion_mc = confusion_matrix(y_test_mc, svm_predicted_mc1)
df_cm = pd.DataFrame(confusion_mc)

plt.figure(figsize=(5.5,4))
sns.heatmap(df_cm, annot = True)
plt.title('SVM linear kernel \Accuracy: {0:.3f}'.format(accuracy_score(y_test_mc, svm_predicted_mc1)))
plt.ylabel('True label')
plt.xlabel('Predicted label')

#RBF kernel
svm = SVC(kernel = 'rbf').fit(X_train_mc, y_train_mc)
svm_predicted_mc2 = svm.predict(X_test_mc)
confusion_mc = confusion_matrix(y_test_mc, svm_predicted_mc2)
df_cm = pd.DataFrame(confusion_mc)

plt.figure(figsize=(5.5,4))
sns.heatmap(df_cm, annot = True)
plt.title('SVM RBD kernel \Accuracy: {0:.3f}'.format(accuracy_score(y_test_mc, svm_predicted_mc2)))
plt.ylabel('True label')
plt.xlabel('Predicted label')

#Classification report
print(classification_report(y_test_mc, svm_predicted_mc1))
print(classification_report(y_test_mc, svm_predicted_mc2))

#Micro and Macro averaging: Precision score
print('Micro averaged precision = {:.2f} (treat instances equally)'.format(precision_score(y_test_mc, svm_predicted_mc1, average="micro")))
print('Macro averaged precision = {:.2f} (treat classes equally)'.format(precision_score(y_test_mc, svm_predicted_mc1, average="macro")))

#Micro and Macro averaging: F1 score
print('Micro averaged f1 = {:.2f} (treat instances equally)'.format(f1_score(y_test_mc, svm_predicted_mc1, average="micro")))
print('Macro averaged f1 = {:.2f} (treat classes equally)'.format(f1_score(y_test_mc, svm_predicted_mc1, average="macro")))