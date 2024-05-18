# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:46:53 2024

@author: btasd
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

# read data

data =  pd.read_csv('mushrooms.csv')

#print(data.head())

classes = data['class'].value_counts()
print(classes)


#bar for edible class
plt.bar('Edible', classes['e'])

#bar for poisonous class
plt.bar('Poissonous', classes['p'])

plt.show()

#creating the X variable for features
X = data.loc[:, ['cap-shape', 'cap-color', 'ring-number', 'ring-type' ]]

#creating the y variable for output layers
y = data.loc[:, 'class']

encoder = LabelEncoder()

for i in X.columns:
    X[i] = encoder.fit_transform(X[i])
    
y = encoder.fit_transform(y)

print(X)
print(y)

#split the dataset into train and test with 70-30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
 

#creating an object

logistic_classifier_model = LogisticRegression()

ridge_classifier_model = RidgeClassifier()

decision_tree_model = DecisionTreeClassifier()

naive_bayes_model = GaussianNB()

neural_network_model = MLPClassifier()


#train the model

logistic_classifier_model.fit(X_train, y_train)

ridge_classifier_model.fit(X_train, y_train)

decision_tree_model.fit(X_train, y_train)

naive_bayes_model.fit(X_train, y_train)

neural_network_model.fit(X_train, y_train)

# make prediction using test dataset

logistic_pred = logistic_classifier_model.predict(X_test)

ridge_pred = ridge_classifier_model.predict(X_test)

tree_pred = decision_tree_model.predict(X_test)

naive_bayes_pred = naive_bayes_model.predict(X_test)

neural_network_pred = neural_network_model.predict(X_test)

# create a classification report for models

logistic_report = classification_report(y_test, logistic_pred)

ridge_report = classification_report(y_test, ridge_pred)

tree_report = classification_report(y_test, tree_pred)

naive_bayes_report = classification_report(y_test, naive_bayes_pred)

neural_network_report = classification_report(y_test, neural_network_pred)



print('***** Logistic Regression *****')
print(logistic_report)

print('**** Ridge Regression')
print(ridge_report)

print('***** Decision Tree *****')
print(tree_report)

print('***** Naive Bayes *****')
print(naive_bayes_report)

print('***** Neural Network *****')
print(neural_network_report)


random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)
random_forest_pred = random_forest_model.predict(X_test)

random_forest_report = classification_report(y_test, random_forest_pred)


print('***** Random Forest *****')
print(random_forest_report)

