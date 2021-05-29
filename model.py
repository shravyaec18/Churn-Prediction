# -*- coding: utf-8 -*-
"""
Created on Sat May 29 10:43:34 2021

@author: shrut
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,accuracy_score,roc_curve,auc
random_state = 123
import pickle


df = pd.read_csv('customer_churn_modified.csv')

sc = StandardScaler()
df['MonthlyCharges'] = sc.fit_transform(df[['MonthlyCharges']].values)
df['TotalCharges'] = sc.fit_transform(df[['TotalCharges']].values)
df['tenure'] = sc.fit_transform(df[['tenure']].values)
df.head()

X = df.drop(['Churn'],axis=1)
y = df['Churn']
y.value_counts()

estimator = LogisticRegression(random_state=random_state)
rfecv = RFECV(estimator=estimator, cv=StratifiedKFold(10, random_state=random_state, shuffle=True), scoring="accuracy")
rfecv.fit(X, y)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
plt.grid()
plt.xticks(range(1, X.shape[1]+1))
plt.xlabel("Number of Selected Features")
plt.ylabel("CV Score")
plt.title("Recursive Feature Elimination (RFE)")
plt.show()

print("The optimal number of features: {}".format(rfecv.n_features_))

X_rfe = X.iloc[:, rfecv.support_]

print("\"X\" dimension: {}".format(X.shape))
print("\"X\" column list:", X.columns.tolist())
print("\"X_rfe\" dimension: {}".format(X_rfe.shape))
print("\"X_rfe\" column list:", X_rfe.columns.tolist())

X_rfe_train, X_rfe_test, y_train, y_test = train_test_split(X_rfe,y,train_size=0.75,stratify=y,random_state=random_state)
print("Train size: {}".format(len(y_train)))
print("Test size: {}".format(len(y_test)))

lr = LogisticRegression()
model = lr.fit(X_rfe_train,y_train)
y_pred = model.predict(X_rfe_test)

print(pd.DataFrame(confusion_matrix(y_test,y_pred)))

print(classification_report(y_test,y_pred))


pickle.dump(model, open('model.pkl','wb'))